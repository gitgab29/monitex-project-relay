"""The motion gate's two jobs: skip what is genuinely unchanged, and never go blind."""

from __future__ import annotations

import numpy as np
import pytest

from relay.capture.motion import MotionGate, luma_term


def flat(value: int = 120, shape=(360, 640, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def noisy(value: int = 120, sigma: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = flat(value).astype(float)
    return np.clip(base + rng.normal(0, sigma, base.shape), 0, 255).astype(np.uint8)


def test_first_frame_is_always_analyzed():
    d = MotionGate().evaluate(flat(), 0)
    assert d.analyze is True and d.reason == "first"


def test_sensor_noise_on_a_static_scene_is_skipped():
    """The gate only pays for itself if webcam noise does not trip it."""
    g = MotionGate(threshold=4.0)
    g.evaluate(noisy(seed=1), 0)
    d = g.evaluate(noisy(seed=2), 1000)
    assert d.analyze is False and d.reason == "quiet"
    assert d.score < 4.0


def test_a_person_entering_trips_the_gate():
    g = MotionGate(threshold=4.0)
    g.evaluate(flat(), 0)
    frame = flat()
    frame[150:350, 200:400] = 30
    d = g.evaluate(frame, 1000)
    assert d.analyze is True and d.reason == "motion"
    assert d.score > 4.0


def test_heartbeat_forces_analysis_of_a_motionless_scene():
    """The gate cannot see an absence. Without the heartbeat, an empty post is never
    confirmed and post_unattended can never fire."""
    g = MotionGate(threshold=4.0, heartbeat_s=5.0)
    g.evaluate(flat(), 0)
    assert g.evaluate(flat(), 4_900).analyze is False
    d = g.evaluate(flat(), 5_000)
    assert d.analyze is True and d.reason == "heartbeat"


def test_heartbeat_measures_from_the_last_analysis_not_the_last_frame():
    g = MotionGate(threshold=4.0, heartbeat_s=5.0)
    g.evaluate(flat(), 0)
    moving = flat()
    moving[150:350, 200:400] = 30
    g.evaluate(moving, 3_000)          # motion resets the heartbeat clock
    assert g.evaluate(moving, 7_000).analyze is False
    assert g.evaluate(moving, 8_000).reason == "heartbeat"


def test_heartbeat_can_be_disabled():
    g = MotionGate(threshold=4.0, heartbeat_s=0.0)
    g.evaluate(flat(), 0)
    assert g.evaluate(flat(), 60_000).analyze is False


def test_comparison_is_against_the_last_analyzed_frame_so_slow_drift_accumulates():
    """Each step is 1 grey level -- against the *previous* frame that is forever below a
    threshold of 4, so a slowly darkening room would never be looked at. Measured against the
    last ANALYSED frame the difference accumulates and trips exactly when the total reaches 4.
    """
    g = MotionGate(threshold=4.0, heartbeat_s=0.0)
    g.evaluate(flat(100), 0)
    decisions = [(v, g.evaluate(flat(v), i * 100)) for i, v in enumerate(range(101, 106), start=1)]
    tripped = [v for v, d in decisions if d.analyze]
    assert tripped == [104], f"expected the gate to trip once, at 104; tripped at {tripped}"
    assert all(d.reason == "quiet" for v, d in decisions if v != 104)
    # and having re-based on 104, the next single step is quiet again
    assert decisions[-1][1].analyze is False


def test_counters_track_the_gate_ratio():
    g = MotionGate(threshold=4.0, heartbeat_s=0.0)
    g.evaluate(flat(), 0)
    for i in range(1, 11):
        g.evaluate(flat(), i * 100)
    assert g.n_analyzed == 1 and g.n_skipped == 10


@pytest.mark.parametrize(
    ("luma", "lo", "hi"),
    [(0, 0.0, 0.01), (5, 0.0, 0.10), (20, 0.10, 0.20), (40, 0.25, 0.35), (90, 0.99, 1.0), (200, 0.99, 1.0)],
)
def test_luma_term_collapses_in_the_dark_and_saturates_in_good_light(luma, lo, hi):
    assert lo <= luma_term(luma) <= hi


def test_luma_term_is_monotonic():
    values = [luma_term(v) for v in range(0, 256, 5)]
    assert values == sorted(values)


def test_darkness_alone_can_push_an_event_under_the_review_bar():
    """The lights-off demo is not a staged threshold: confidence is min(...) of the three
    terms, so a genuinely dark frame drags a strong YOLO detection under 0.75 by itself."""
    yolo_term, zone_term = 0.93, 0.95
    assert min(yolo_term, zone_term, luma_term(120)) >= 0.75
    assert min(yolo_term, zone_term, luma_term(12)) < 0.75
