"""Suppress "people" that are photographs, posters, portraits and screens.

YOLO is doing its job correctly when it boxes a face in a picture frame on the wall -- that
region really does look like a person. It is the *system* that is wrong to then report an
unidentified person at the post. On a fixed camera the two are easy to tell apart, and not by
appearance:

    a real person drifts; a printed one is pixel-identical forever.

So this holds a small memory of where boxes have been, and for each one asks two questions:
is it in the same place as last time, and did the pixels inside it change at all? A box that
answers "same place, no change" for long enough is furniture, and furniture is not an alert.

Two properties matter more than the accuracy:

* **It un-suppresses instantly.** One frame of real movement inside the box clears the counter.
  A person who sat motionless long enough to be called furniture becomes a person again the
  moment they twitch. Getting this backwards -- slow to re-detect -- would be dangerous.
* **It is a pure function of images and boxes.** No camera, no model, no wall clock, so the
  behaviour above is unit-tested rather than demonstrated by standing very still.

Defaults are deliberately reluctant: ~20 consecutive analysed frames of near-zero change. It is
better to alert on a poster for twenty seconds than to stop reporting a guard who is reading.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import Settings
from ..schema import Detection

log = logging.getLogger(__name__)

#: Every remembered box is reduced to this before comparison. Small enough that sensor noise
#: averages out, large enough that a shift of a few real pixels still registers.
THUMB = 24


def iou(a: Detection, b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a.x0, a.y0, a.x1, a.y1
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def thumbnail(image: np.ndarray, det: Detection) -> np.ndarray | None:
    """The box's contents as a small greyscale patch, or None if the crop is degenerate."""
    import cv2

    h, w = image.shape[:2]
    x0, x1 = int(det.x0 * w), int(det.x1 * w)
    y0, y1 = int(det.y0 * h), int(det.y1 * h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = image[y0:y1, x0:x1]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(crop, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32)


class _Slot:
    __slots__ = ("box", "thumb", "still_for")

    def __init__(self, box, thumb):
        self.box = box
        self.thumb = thumb
        self.still_for = 0


class StaticObjectFilter:
    """Remembers boxes between analysed frames and drops the ones that never change."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.enabled = cfg.static_filter
        self.min_frames = cfg.static_min_frames
        self.eps = cfg.static_pixel_eps
        self.match_iou = cfg.static_match_iou
        self._slots: list[_Slot] = []
        self.suppressed_total = 0

    def apply(self, image: np.ndarray, detections: list[Detection]) -> list[Detection]:
        if not self.enabled or not detections:
            self._slots = [] if not detections else self._slots
            return detections

        kept: list[Detection] = []
        seen: list[_Slot] = []

        for det in detections:
            thumb = thumbnail(image, det)
            slot = self._match(det)

            if slot is None:
                slot = _Slot((det.x0, det.y0, det.x1, det.y1), thumb)
            else:
                moved = True
                if thumb is not None and slot.thumb is not None and thumb.shape == slot.thumb.shape:
                    delta = float(np.mean(np.abs(thumb - slot.thumb)))
                    moved = delta >= self.eps
                if moved:
                    # One frame of real change is enough to call it alive again. Being slow
                    # here would mean failing to report a person who had been sitting still.
                    slot.still_for = 0
                else:
                    slot.still_for += 1
                slot.box = (det.x0, det.y0, det.x1, det.y1)
                slot.thumb = thumb if thumb is not None else slot.thumb

            seen.append(slot)

            if slot.still_for >= self.min_frames:
                self.suppressed_total += 1
                log.debug(
                    "suppressed a static 'person' at (%.2f,%.2f)-(%.2f,%.2f): unchanged for "
                    "%d analysed frames -- treating it as a picture, screen or reflection",
                    det.x0, det.y0, det.x1, det.y1, slot.still_for,
                )
            else:
                kept.append(det)

        self._slots = seen
        return kept

    def _match(self, det: Detection) -> _Slot | None:
        best, best_iou = None, 0.0
        for slot in self._slots:
            v = iou(det, slot.box)
            if v > best_iou:
                best, best_iou = slot, v
        return best if best_iou >= self.match_iou else None
