"""A photograph is not a person, and a person holding still is.

Both halves matter, but they are not equally dangerous. Suppressing a poster for a few seconds
too long is a cosmetic problem. Failing to re-detect a real person who had been sitting still
is a security system that stops working exactly when someone stops moving. So the tests below
lean hard on the second one.

No camera and no model here: the filter is a pure function of images and boxes, which is the
whole reason it can be checked this way.
"""

from __future__ import annotations

import numpy as np
import pytest

from relay.config import Settings
from relay.detect.static_filter import StaticObjectFilter, iou
from relay.schema import Detection


@pytest.fixture()
def cfg() -> Settings:
    return Settings(static_filter=True, static_min_frames=3,
                    static_pixel_eps=2.5, static_match_iou=0.85)


def det(x0=0.2, y0=0.2, x1=0.4, y1=0.6, conf=0.9) -> Detection:
    return Detection(x0=x0, y0=y0, x1=x1, y1=y1, conf=conf, zone="post",
                     straddle=False, area_frac=0.1, zone_term=0.9)


def scene(seed: int = 0, noise: float = 0.0) -> np.ndarray:
    """A deterministic 'frame'. `noise` simulates real-world change inside the box."""
    rng = np.random.default_rng(seed)
    img = np.full((240, 320, 3), 120, dtype=np.uint8)
    img[40:160, 60:140] = rng.integers(60, 200, (120, 80, 3), dtype=np.uint8)
    if noise:
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


# ------------------------------------------------------------------ the picture on the wall

def test_an_unchanging_box_is_eventually_suppressed(cfg):
    f = StaticObjectFilter(cfg)
    img, d = scene(1), det()
    kept = [f.apply(img, [d]) for _ in range(6)]
    assert kept[0] == [d], "must not suppress on the first sight of it"
    assert kept[-1] == [], "an identical box forever is furniture"


def test_it_takes_at_least_min_frames(cfg):
    f = StaticObjectFilter(cfg)
    img, d = scene(1), det()
    for i in range(cfg.static_min_frames):
        assert f.apply(img, [d]) == [d], f"suppressed too early, at frame {i}"
    assert f.apply(img, [d]) == []


def test_suppression_is_counted_so_it_is_visible(cfg):
    f = StaticObjectFilter(cfg)
    img, d = scene(1), det()
    for _ in range(6):
        f.apply(img, [d])
    assert f.suppressed_total > 0


# ------------------------------------------------------------------ the person who holds still

def test_a_moving_box_is_never_suppressed(cfg):
    f = StaticObjectFilter(cfg)
    for i in range(10):
        d = det(x0=0.2 + i * 0.01, x1=0.4 + i * 0.01)
        assert f.apply(scene(i), [d]) == [d]


def test_one_frame_of_change_immediately_restores_a_suppressed_box(cfg):
    """The dangerous direction. A guard who sat motionless long enough to be called furniture
    must count as a person again the moment they move -- not after another min_frames."""
    f = StaticObjectFilter(cfg)
    still, d = scene(1), det()
    for _ in range(6):
        f.apply(still, [d])
    assert f.apply(still, [d]) == [], "precondition: it is currently suppressed"

    moved = scene(99)  # same box, different pixels
    assert f.apply(moved, [d]) == [d], "must come back on the FIRST frame of movement"


def test_content_change_alone_is_enough_even_if_the_box_does_not_move(cfg):
    """Someone turning their head fills the same box with different pixels."""
    f = StaticObjectFilter(cfg)
    d = det()
    for i in range(8):
        assert f.apply(scene(i), [d]) == [d]


# ------------------------------------------------------------------ behaviour and config

def test_the_filter_can_be_turned_off_entirely():
    f = StaticObjectFilter(Settings(static_filter=False, static_min_frames=1))
    img, d = scene(1), det()
    for _ in range(10):
        assert f.apply(img, [d]) == [d]


def test_an_empty_frame_is_passed_through(cfg):
    assert StaticObjectFilter(cfg).apply(scene(), []) == []


def test_two_objects_are_tracked_independently(cfg):
    """A picture on the wall and a person at the desk, at once: only the picture goes."""
    f = StaticObjectFilter(cfg)
    picture = det(x0=0.05, y0=0.05, x1=0.15, y1=0.25)
    for i in range(6):
        person = det(x0=0.5 + i * 0.02, y0=0.3, x1=0.7 + i * 0.02, y1=0.9)
        kept = f.apply(scene(i), [picture, person])
    assert person in kept
    assert picture not in kept


def test_iou_is_sane():
    a = det(0.0, 0.0, 1.0, 1.0)
    assert iou(a, (0.0, 0.0, 1.0, 1.0)) == pytest.approx(1.0)
    assert iou(a, (2.0, 2.0, 3.0, 3.0)) == 0.0
    assert 0.0 < iou(a, (0.5, 0.0, 1.5, 1.0)) < 1.0
