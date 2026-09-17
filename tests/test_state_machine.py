"""Every scenario the demo shows, proven on synthetic sequences.

This is the point of keeping the state machine free of models, cameras and networks: the
loitering and intrusion scenarios need a second person in front of a camera to film, but they
need nothing at all to test. If these pass, the filming is theatre over verified logic.

Sequences are written as a list of (post_count, approach_count) at 1 frame per second.
"""

from __future__ import annotations

import dataclasses

import pytest

from relay.config import Settings
from relay.detect.state import PostStateMachine
from relay.schema import Detection, Observation


@pytest.fixture()
def cfg() -> Settings:
    return Settings(
        manned_confirm_frames=2, intrusion_confirm_frames=2,
        unattended_dwell_s=8.0, loiter_dwell_s=6.0, exit_grace_s=3.0,
    )


def det(zone: str, conf: float = 0.9, straddle: bool = False) -> Detection:
    # Geometry is irrelevant here -- the zone is already decided by ZoneModel upstream.
    return Detection(
        x0=.3, y0=.4, x1=.6, y1=.9, conf=conf, zone=zone,
        straddle=straddle, area_frac=.15, zone_term=.55 if straddle else .9,
    )


def obs(i: int, post: int, approach: int, luma: float = 120.0, conf: float = 0.9) -> Observation:
    dets = [det("post", conf) for _ in range(post)] + [det("approach", conf) for _ in range(approach)]
    return Observation(
        session_id="s", frame_index=i * 30, video_ts_ms=i * 1000,
        motion_score=5.0, mean_luma=luma, detections=dets,
    )


def run(cfg: Settings, seq: list[tuple[int, int]], *, luma: float = 120.0, conf: float = 0.9):
    """Feed a sequence at 1 fps; return (all drafts, machine)."""
    m = PostStateMachine(cfg)
    drafts = []
    for i, (p, a) in enumerate(seq):
        drafts.extend(m.update(obs(i, p, a, luma, conf)))
    return drafts, m


def opened(drafts, observed: str):
    return [d for d in drafts if d.kind == "open" and d.observed == observed]


# ---------------------------------------------------------------- scenario 1: manned

def test_guard_sits_down_and_the_post_becomes_manned(cfg):
    drafts, m = run(cfg, [(1, 0)] * 4)
    assert m.post_state == "MANNED"
    assert len(opened(drafts, "post_manned")) == 1


def test_manned_needs_two_consecutive_frames(cfg):
    """One frame is not evidence: YOLO produces single-frame false positives."""
    _, m = run(cfg, [(1, 0)])
    assert m.post_state == "UNKNOWN"


def test_manned_fires_exactly_once_however_long_they_sit(cfg):
    """An hour at the desk is one event, not 3,600 rows."""
    drafts, _ = run(cfg, [(1, 0)] * 120)
    assert len(opened(drafts, "post_manned")) == 1


# ---------------------------------------------------------------- scenario 2: unattended

def test_empty_post_fires_unattended_after_the_dwell(cfg):
    drafts, m = run(cfg, [(1, 0)] * 3 + [(0, 0)] * 10)
    assert m.post_state == "UNATTENDED"
    ev = opened(drafts, "post_unattended")
    assert len(ev) == 1
    # Sat for frames 0-2, empty from frame 3, dwell 8 s -> fires at t=11 s.
    assert ev[0].first_seen_ms == 11_000


def test_a_brief_absence_does_not_fire_unattended(cfg):
    """A guard leaning out of frame for a few seconds is not an unmanned post."""
    drafts, m = run(cfg, [(1, 0)] * 3 + [(0, 0)] * 5 + [(1, 0)] * 3)
    assert opened(drafts, "post_unattended") == []
    assert m.post_state == "MANNED"


def test_single_frame_yolo_miss_does_not_fire_unattended(cfg):
    """The most common real failure: one dropped detection mid-session."""
    drafts, m = run(cfg, [(1, 0)] * 5 + [(0, 0)] + [(1, 0)] * 5)
    assert opened(drafts, "post_unattended") == []
    assert m.post_state == "MANNED"


def test_unattended_from_the_start_of_an_empty_room(cfg):
    drafts, m = run(cfg, [(0, 0)] * 12)
    assert m.post_state == "UNATTENDED"
    assert len(opened(drafts, "post_unattended")) == 1


def test_guard_returns_and_the_post_is_manned_again(cfg):
    drafts, m = run(cfg, [(1, 0)] * 3 + [(0, 0)] * 10 + [(1, 0)] * 4)
    assert m.post_state == "MANNED"
    assert len(opened(drafts, "post_unattended")) == 1
    assert len(opened(drafts, "post_manned")) == 2      # left and came back: two occurrences
    assert any(d.kind == "close" and d.observed == "post_unattended" for d in drafts)


# ---------------------------------------------------------------- scenario 3: loitering

def test_someone_waiting_by_the_entrance_is_loitering(cfg):
    drafts, m = run(cfg, [(1, 1)] * 10)
    assert m.approach_state == "LOITERING"
    ev = opened(drafts, "person_loitering_near_entry")
    assert len(ev) == 1
    assert ev[0].first_seen_ms == 0      # dated from arrival, not from the timer expiring


def test_walking_through_is_not_loitering(cfg):
    """The whole point of the dwell timer: passing through must not page anyone."""
    drafts, m = run(cfg, [(1, 0)] * 2 + [(1, 1)] * 3 + [(1, 0)] * 6)
    assert opened(drafts, "person_loitering_near_entry") == []
    assert m.approach_state == "CLEAR"


def test_loitering_survives_a_single_missed_detection(cfg):
    """At 1 fps a dropped frame is routine; restarting the dwell on it means a loiterer is
    never confirmed."""
    drafts, _ = run(cfg, [(1, 1)] * 4 + [(1, 0)] + [(1, 1)] * 4)
    assert len(opened(drafts, "person_loitering_near_entry")) == 1


def test_loitering_is_dated_from_arrival_not_from_the_grace_period(cfg):
    drafts, _ = run(cfg, [(1, 1)] * 4 + [(1, 0)] + [(1, 1)] * 4)
    assert opened(drafts, "person_loitering_near_entry")[0].first_seen_ms == 0


def test_leaving_for_longer_than_the_grace_period_clears_the_zone(cfg):
    drafts, m = run(cfg, [(1, 1)] * 8 + [(1, 0)] * 6)
    assert m.approach_state == "CLEAR"
    assert any(d.kind == "close" and d.observed == "person_loitering_near_entry" for d in drafts)


def test_post_and_approach_are_tracked_independently(cfg):
    """Someone loitering at the entrance while the desk is manned is two facts, not one."""
    drafts, m = run(cfg, [(1, 1)] * 10)
    assert m.post_state == "MANNED" and m.approach_state == "LOITERING"
    assert len(opened(drafts, "post_manned")) == 1
    assert len(opened(drafts, "person_loitering_near_entry")) == 1


# ---------------------------------------------------------------- scenario 4: intrusion

def test_two_people_at_the_post_is_an_intrusion(cfg):
    drafts, m = run(cfg, [(1, 0)] * 3 + [(2, 0)] * 3)
    assert m.post_state == "INTRUSION"
    assert len(opened(drafts, "unidentified_person_at_post")) == 1


def test_intrusion_needs_two_consecutive_frames(cfg):
    drafts, m = run(cfg, [(1, 0)] * 3 + [(2, 0)] + [(1, 0)] * 3)
    assert opened(drafts, "unidentified_person_at_post") == []
    assert m.post_state == "MANNED"


def test_intrusion_from_an_unattended_post(cfg):
    drafts, m = run(cfg, [(0, 0)] * 10 + [(2, 0)] * 3)
    assert m.post_state == "INTRUSION"
    assert len(opened(drafts, "post_unattended")) == 1
    assert len(opened(drafts, "unidentified_person_at_post")) == 1


def test_post_returns_to_manned_after_the_second_person_leaves(cfg):
    drafts, m = run(cfg, [(1, 0)] * 3 + [(2, 0)] * 3 + [(1, 0)] * 3)
    assert m.post_state == "MANNED"
    assert len(opened(drafts, "unidentified_person_at_post")) == 1
    assert len(opened(drafts, "post_manned")) == 2


# ---------------------------------------------------------------- evidence terms

def test_terms_are_carried_onto_the_draft(cfg):
    drafts, _ = run(cfg, [(1, 0)] * 3, conf=0.42)
    d = opened(drafts, "post_manned")[0]
    assert d.yolo_term == pytest.approx(0.42)
    assert 0.0 <= d.luma_term <= 1.0


def test_darkness_lowers_the_luma_term_on_the_draft(cfg):
    bright = opened(run(cfg, [(1, 0)] * 3, luma=130.0)[0], "post_manned")[0]
    dark = opened(run(cfg, [(1, 0)] * 3, luma=10.0)[0], "post_manned")[0]
    assert bright.luma_term == pytest.approx(1.0)
    assert dark.luma_term < 0.15


def test_a_straddling_detection_marks_the_draft(cfg):
    m = PostStateMachine(cfg)
    drafts = []
    for i in range(3):
        o = Observation(
            session_id="s", frame_index=i * 30, video_ts_ms=i * 1000, motion_score=5.0,
            mean_luma=120.0, detections=[det("post", straddle=True)],
        )
        drafts.extend(m.update(o))
    assert opened(drafts, "post_manned")[0].straddle is True


def test_unattended_is_scored_on_absence_not_on_a_nonexistent_box(cfg):
    """There is no detection to be confident about, so the yolo term cannot come from one."""
    d = opened(run(cfg, [(0, 0)] * 12)[0], "post_unattended")[0]
    assert d.yolo_term == pytest.approx(0.90)
    assert d.zone_term == pytest.approx(1.0)


# ---------------------------------------------------------------- close-out & determinism

def test_finish_closes_whatever_is_still_open(cfg):
    _, m = run(cfg, [(1, 1)] * 10)
    closes = m.finish()
    assert {d.observed for d in closes} == {"post_manned", "person_loitering_near_entry"}
    assert all(d.kind == "close" for d in closes)


def test_the_same_sequence_always_produces_the_same_transitions(cfg):
    """No wall clock, no randomness: this is why a replay reproduces the same event ids."""
    seq = [(1, 0)] * 3 + [(0, 0)] * 10 + [(2, 1)] * 8
    a = [(d.observed, d.kind, d.first_seen_ms) for d in run(cfg, seq)[0]]
    b = [(d.observed, d.kind, d.first_seen_ms) for d in run(cfg, seq)[0]]
    assert a == b


def test_dwell_settings_are_honoured(cfg):
    """The timings are config, not constants -- production would use 60-120 s for unattended."""
    slow = dataclasses.replace(cfg, unattended_dwell_s=20.0)
    assert opened(run(slow, [(1, 0)] * 3 + [(0, 0)] * 10)[0], "post_unattended") == []
    assert len(opened(run(slow, [(1, 0)] * 3 + [(0, 0)] * 25)[0], "post_unattended")) == 1


def test_the_full_demo_sequence_produces_exactly_the_four_event_types(cfg):
    """The unbroken take from the demo script, as a test: sit, leave, someone waits, they
    approach the desk."""
    seq = [(1, 0)] * 4 + [(0, 0)] * 12 + [(0, 1)] * 8 + [(2, 0)] * 4
    drafts, _ = run(cfg, seq)
    fired = [d.observed for d in drafts if d.kind == "open"]
    assert set(fired) == {
        "post_manned", "post_unattended",
        "person_loitering_near_entry", "unidentified_person_at_post",
    }
