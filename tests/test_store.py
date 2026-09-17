"""Persistence guarantees: the ones the re-run promise in the README depends on."""

from __future__ import annotations

import pytest

from relay.ids import new_run_id
from relay.schema import Category, Detection, Event, Observation, Priority
from relay.store import Store


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def make_event(event_id: str = "abc123", **kw) -> Event:
    base = dict(
        event_id=event_id, source_file="data/sessions/sess-x.mp4", category=Category.routine,
        priority=Priority.low, observed="post_manned", video_ts="00:00:01.00",
        site_id="site-118", summary="Guard seated at reception desk.", confidence=0.91,
        needs_review=False,
    )
    base.update(kw)
    return Event(**base)


def prov(**kw):
    base = dict(
        session_id="sess-x", run_id="run-1", track_key="post_manned", bucket=0,
        first_seen_ms=1000, last_seen_ms=2000, frame_count=2, yolo_term=0.93,
        zone_term=0.95, luma_term=0.9, summary_source="template", review_reasons=[],
    )
    base.update(kw)
    return base


# ---------------------------------------------------------------- idempotency

def test_first_upsert_inserts_second_does_not(store):
    ev = make_event()
    assert store.upsert_event(ev, prov()) is True
    assert store.upsert_event(ev, prov()) is False
    assert len(store.list_events()) == 1


def test_reupsert_extends_the_occurrence_but_never_rewrites_it(store):
    """A second sighting makes the event *longer*, not different. Identity, classification
    and summary belong to the first observation."""
    store.upsert_event(make_event(), prov(last_seen_ms=2000, frame_count=2))
    store.upsert_event(
        make_event(summary="DIFFERENT SUMMARY", confidence=0.10),
        prov(last_seen_ms=9000, frame_count=7, run_id="run-2"),
    )
    row = store.get_event("abc123")
    assert row["last_seen_ms"] == 9000
    assert row["frame_count"] == 7
    assert row["summary"] == "Guard seated at reception desk."
    assert row["confidence"] == pytest.approx(0.91)
    assert row["run_id"] == "run-1"


def test_late_frame_cannot_shorten_an_event(store):
    """Frames can arrive out of order off the analysis worker; MAX() keeps the extent honest."""
    store.upsert_event(make_event(), prov(last_seen_ms=9000, frame_count=7))
    store.upsert_event(make_event(), prov(last_seen_ms=2000, frame_count=2))
    row = store.get_event("abc123")
    assert row["last_seen_ms"] == 9000
    assert row["frame_count"] == 7


# ---------------------------------------------------------------- drift guard

def test_resolve_event_id_is_a_pure_hash_when_nothing_exists(store):
    eid, reused = store.resolve_event_id("sess-x", "post_manned", 1000, 5, 5.0)
    assert reused is False
    same, _ = store.resolve_event_id("sess-x", "post_manned", 1000, 5, 5.0)
    assert same == eid


def test_drift_across_a_bucket_edge_reuses_the_existing_id(store):
    """The live-to-replay case: compression nudges a confidence, the transition lands 200 ms
    later and falls in the NEXT bucket. Without the window that is a duplicate event."""
    eid, _ = store.resolve_event_id("sess-x", "post_manned", 4_900, 5, 5.0)
    store.upsert_event(make_event(eid), prov(first_seen_ms=4_900))
    drifted, reused = store.resolve_event_id("sess-x", "post_manned", 5_100, 5, 5.0)
    assert reused is True
    assert drifted == eid
    assert store.upsert_event(make_event(drifted), prov(first_seen_ms=5_100)) is False
    assert len(store.list_events()) == 1


def test_window_does_not_merge_genuinely_separate_occurrences(store):
    """A guard leaving and returning 40 s later is two events, not one."""
    first, _ = store.resolve_event_id("sess-x", "post_manned", 1_000, 5, 5.0)
    store.upsert_event(make_event(first), prov(first_seen_ms=1_000))
    later, reused = store.resolve_event_id("sess-x", "post_manned", 41_000, 5, 5.0)
    assert reused is False
    assert later != first


def test_window_does_not_merge_different_states(store):
    first, _ = store.resolve_event_id("sess-x", "post_manned", 1_000, 5, 5.0)
    store.upsert_event(make_event(first), prov(first_seen_ms=1_000))
    other, reused = store.resolve_event_id("sess-x", "post_unattended", 1_100, 5, 5.0)
    assert reused is False and other != first


def test_window_does_not_reach_across_sessions(store):
    first, _ = store.resolve_event_id("sess-x", "post_manned", 1_000, 5, 5.0)
    store.upsert_event(make_event(first), prov(first_seen_ms=1_000))
    other, reused = store.resolve_event_id("sess-y", "post_manned", 1_100, 5, 5.0)
    assert reused is False and other != first


def test_window_can_be_disabled(store):
    store.upsert_event(make_event("e1"), prov(first_seen_ms=4_900))
    assert store.find_nearby_event("sess-x", "post_manned", 5_100, 0.0) is None


# ---------------------------------------------------------------- observations

def make_obs(frame_index: int = 30) -> Observation:
    det = Detection(x0=.25, y0=.45, x1=.61, y1=.99, conf=.93, zone="post", area_frac=.19, zone_term=.95)
    return Observation(
        session_id="sess-x", frame_index=frame_index, video_ts_ms=frame_index * 1000 // 30,
        motion_score=5.1, mean_luma=124.6, detections=[det],
    )


def test_observations_are_idempotent_per_frame(store):
    assert store.add_observation(make_obs(), "MANNED", "CLEAR") is True
    assert store.add_observation(make_obs(), "MANNED", "CLEAR") is False
    n = store.conn.execute("SELECT COUNT(*) c FROM observations").fetchone()["c"]
    assert n == 1


def test_observation_denormalises_counts_for_querying(store):
    store.add_observation(make_obs(), "MANNED", "CLEAR")
    row = store.conn.execute("SELECT * FROM observations").fetchone()
    assert row["post_count"] == 1 and row["approach_count"] == 0
    assert row["person_count"] == 1
    assert row["yolo_max_conf"] == pytest.approx(0.93)


# ---------------------------------------------------------------- runs ledger

def test_second_run_over_same_session_and_config_is_marked_a_replay(store):
    r1 = new_run_id()
    assert store.start_run(r1, "sess-x", "live", "cfg-aaa") is None
    r2 = new_run_id()
    assert store.start_run(r2, "sess-x", "replay", "cfg-aaa") == r1


def test_a_different_config_is_a_new_analysis_not_a_replay(store):
    r1 = new_run_id()
    store.start_run(r1, "sess-x", "live", "cfg-aaa")
    r2 = new_run_id()
    assert store.start_run(r2, "sess-x", "replay", "cfg-DIFFERENT") is None


def test_run_counters_and_derived_ratio(store):
    rid = new_run_id()
    store.start_run(rid, "sess-x", "live", "cfg")
    store.bump(rid, "frames_decoded", 300)
    store.bump(rid, "frames_analyzed", 10)
    store.bump(rid, "events_skipped", 6)
    store.end_run(rid)
    row = store.get_run(rid)
    assert row["frames_decoded"] == 300 and row["frames_analyzed"] == 10
    assert row["analyze_ratio"] == pytest.approx(0.033, abs=0.001)
    assert row["events_skipped"] == 6


def test_bump_rejects_an_unknown_counter(store):
    """The counter name is interpolated into SQL, so the allow-list is load-bearing."""
    rid = new_run_id()
    store.start_run(rid, "sess-x", "live", "cfg")
    with pytest.raises(ValueError):
        store.bump(rid, "frames_decoded = 0; DROP TABLE events; --")


# ---------------------------------------------------------------- review queue

def test_review_round_trip(store):
    store.upsert_event(make_event(needs_review=True), prov())
    store.queue_for_review("abc123", ["low_confidence", "missing_site_id"])
    assert len(store.pending_review()) == 1
    assert store.resolve_review("abc123", "false_alarm", by="gabriel", notes="lights") is True
    assert store.pending_review() == []


def test_a_resolution_cannot_be_overwritten(store):
    store.upsert_event(make_event(), prov())
    store.queue_for_review("abc123", ["low_confidence"])
    assert store.resolve_review("abc123", "confirmed") is True
    assert store.resolve_review("abc123", "dismissed") is False


def test_resolution_must_be_one_of_the_three(store):
    store.upsert_event(make_event(), prov())
    store.queue_for_review("abc123", ["low_confidence"])
    with pytest.raises(ValueError):
        store.resolve_review("abc123", "probably_fine")


def test_sweep_sees_a_resolution_once_and_only_once(store):
    """`resolution_notified_at` is the sweep's idempotency key: it is what stops a slow email
    from producing two 'review closed' messages for one resolution."""
    store.upsert_event(make_event(), prov())
    store.queue_for_review("abc123", ["low_confidence"])
    store.resolve_review("abc123", "confirmed")
    assert len(store.resolved_unnotified()) == 1
    assert store.ack_resolution_notified("abc123") is True
    assert store.resolved_unnotified() == []
    assert store.ack_resolution_notified("abc123") is False


# ---------------------------------------------------------------- dead letter & summary

def test_dead_letter_records_the_evidence(store):
    dl = store.dead_letter("vision", "VisionTimeout after 3 attempts", {"event_id": "abc123"}, 3, "abc123")
    assert dl > 0
    row = store.list_dead_letters()[0]
    assert row["stage"] == "vision" and row["attempts"] == 3


def test_summary_shape_is_stable_on_an_empty_database(store):
    s = store.summary()
    assert s["totals"]["events"] == 0
    assert s["review"] == {"open": 0, "closed": 0}
    assert s["dead_letters"] == 0
    assert s["by_category"] == []


def test_summary_counts_what_happened(store):
    store.upsert_event(make_event("e1"), prov())
    store.upsert_event(
        make_event("e2", category=Category.intrusion, priority=Priority.high, needs_review=True),
        prov(track_key="unidentified_person_at_post"),
    )
    store.queue_for_review("e2", ["low_confidence"])
    s = store.summary()
    assert s["totals"]["events"] == 2
    assert s["totals"]["needs_review"] == 1
    assert s["totals"]["high"] == 1
    assert s["review"]["open"] == 1


def test_insert_detection_survives_same_millisecond_writes(store):
    """Regression. `upsert_event` once decided 'did I insert?' by comparing created_at to
    the current timestamp. Back-to-back upserts land in the same millisecond, so the second
    one looked like an insert -- which is precisely how a replay would re-notify a human.
    Ten immediate re-upserts must produce exactly one insert.
    """
    ev = make_event("same-ms")
    assert store.upsert_event(ev, prov()) is True
    assert [store.upsert_event(ev, prov()) for _ in range(10)] == [False] * 10
    assert len(store.list_events()) == 1


def test_a_full_replay_inserts_nothing(store):
    """The headline claim: same session, same transitions, second pass -> 0 inserted."""
    transitions = [
        ("e1", "post_manned", 1_000),
        ("e2", "post_unattended", 20_000),
        ("e3", "person_loitering_near_entry", 45_000),
        ("e4", "unidentified_person_at_post", 70_000),
    ]
    first = [
        store.upsert_event(make_event(eid), prov(track_key=tk, first_seen_ms=ms))
        for eid, tk, ms in transitions
    ]
    second = [
        store.upsert_event(make_event(eid), prov(track_key=tk, first_seen_ms=ms, run_id="run-2"))
        for eid, tk, ms in transitions
    ]
    assert first == [True] * 4
    assert second == [False] * 4
    assert len(store.list_events()) == 4
