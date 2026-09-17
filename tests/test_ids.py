"""The identity guarantees, stated as tests.

If any of these break, replay stops being idempotent and the system can page a human twice
for one real-world event.
"""

from __future__ import annotations

import pytest

from relay.ids import (
    bucket_for,
    event_id,
    event_id_for,
    frame_to_ms,
    new_run_id,
    new_session_id,
    session_id_from_path,
    video_ts,
)


def test_event_id_is_deterministic():
    a = event_id("sess-20260917-104619-3e91", "post_manned", 3)
    b = event_id("sess-20260917-104619-3e91", "post_manned", 3)
    assert a == b
    assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)


def test_event_id_varies_with_every_input():
    base = event_id("sess-A", "post_manned", 3)
    assert base != event_id("sess-B", "post_manned", 3)
    assert base != event_id("sess-A", "post_unattended", 3)
    assert base != event_id("sess-A", "post_manned", 4)


def test_ids_do_not_depend_on_wall_clock_or_run():
    """The reason a replay is idempotent: nothing outside (session, track, bucket) is in it."""
    first = event_id_for("sess-A", "post_manned", 12_345, 5)
    second = event_id_for("sess-A", "post_manned", 12_345, 5)
    assert first == second


@pytest.mark.parametrize(
    ("ms", "expected_bucket"),
    [(0, 0), (4_999, 0), (5_000, 1), (9_999, 1), (10_000, 2), (93_450, 18)],
)
def test_bucket_math(ms, expected_bucket):
    assert bucket_for(ms, 5) == expected_bucket


def test_small_timing_differences_collapse_into_one_id():
    """Two analyses that place a transition 3.9 s apart inside a bucket agree on the id."""
    assert event_id_for("s", "post_manned", 1_000, 5) == event_id_for("s", "post_manned", 4_900, 5)


def test_bucket_edge_is_the_known_weakness():
    """Straddling a bucket boundary DOES produce different ids -- which is exactly why
    Store.find_nearby_event exists. Documenting the limit so it cannot regress silently."""
    assert event_id_for("s", "post_manned", 4_999, 5) != event_id_for("s", "post_manned", 5_001, 5)


def test_bucket_rejects_nonsense():
    with pytest.raises(ValueError):
        bucket_for(1000, 0)


def test_session_id_round_trips_through_a_filename():
    sid = new_session_id()
    assert session_id_from_path(f"data/sessions/{sid}.mp4") == sid
    assert session_id_from_path(r"C:\data\sessions\%s.mp4" % sid) == sid


def test_session_id_absent_from_a_foreign_filename():
    assert session_id_from_path("samples/some_other_clip.mp4") is None


def test_run_ids_are_unique():
    assert len({new_run_id() for _ in range(100)}) == 100


@pytest.mark.parametrize(
    ("ms", "expected"),
    [(0, "00:00:00.00"), (1_000, "00:00:01.00"), (93_450, "00:01:33.45"), (3_661_990, "01:01:01.99")],
)
def test_video_ts_format(ms, expected):
    assert video_ts(ms) == expected


def test_video_ts_clamps_negative():
    assert video_ts(-5) == "00:00:00.00"


@pytest.mark.parametrize(("frame", "fps", "ms"), [(0, 30.0, 0), (30, 30.0, 1000), (15, 30.0, 500), (45, 15.0, 3000)])
def test_frame_to_ms_is_index_based(frame, fps, ms):
    """Frame index, never wall clock -- so a live run and its replay agree on timestamps."""
    assert frame_to_ms(frame, fps) == ms


def test_frame_to_ms_rejects_bad_fps():
    with pytest.raises(ValueError):
        frame_to_ms(30, 0)
