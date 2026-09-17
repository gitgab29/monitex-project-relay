"""The motion gate: decide which frames are worth running a detector on.

At 30 fps a reception camera produces 108,000 frames an hour, and almost all of them are
identical to the one before. Running YOLO on each is pointless work. The gate compares each
candidate frame against **the last frame actually analysed** -- not the immediately previous
frame -- so slow drift accumulates instead of being repeatedly dismissed as "no change".

The heartbeat is the part that is easy to get wrong. A pure motion gate can never confirm an
*absence*: an empty post generates no motion, so the gate would skip forever and
`post_unattended` would never fire. Forcing an analysis every HEARTBEAT_S seconds of video
time makes absence observable, and costs one inference every few seconds.

All timing is in video time (from frame indices), never wall-clock, so the gate makes exactly
the same decisions on a live run and on a replay of its recording.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

#: Small enough that the diff is cheap and sensor noise is averaged away, large enough that a
#: person-sized change is still obvious.
WORK_W, WORK_H = 160, 90


@dataclass
class GateDecision:
    analyze: bool
    score: float
    reason: str  # 'motion' | 'heartbeat' | 'first' | 'quiet'
    mean_luma: float


class MotionGate:
    def __init__(self, threshold: float = 4.0, heartbeat_s: float = 5.0):
        self.threshold = float(threshold)
        self.heartbeat_s = float(heartbeat_s)
        self._last_analyzed: np.ndarray | None = None
        self._last_analyzed_ms: int | None = None
        self.n_analyzed = 0
        self.n_skipped = 0

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        """Grayscale, downscale, blur. The blur is what keeps webcam sensor noise from
        reading as motion on a completely static scene."""
        small = cv2.resize(image, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def evaluate(self, image: np.ndarray, video_ts_ms: int) -> GateDecision:
        prepared = self._prepare(image)
        mean_luma = float(prepared.mean())

        if self._last_analyzed is None:
            return self._accept(prepared, video_ts_ms, 255.0, "first", mean_luma)

        score = float(cv2.absdiff(prepared, self._last_analyzed).mean())

        if score >= self.threshold:
            return self._accept(prepared, video_ts_ms, score, "motion", mean_luma)

        since = video_ts_ms - (self._last_analyzed_ms or 0)
        if self.heartbeat_s > 0 and since >= self.heartbeat_s * 1000:
            # Absence has no motion. Without this the system could never confirm an empty post.
            return self._accept(prepared, video_ts_ms, score, "heartbeat", mean_luma)

        self.n_skipped += 1
        return GateDecision(analyze=False, score=score, reason="quiet", mean_luma=mean_luma)

    def _accept(
        self, prepared: np.ndarray, video_ts_ms: int, score: float, reason: str, mean_luma: float
    ) -> GateDecision:
        self._last_analyzed = prepared
        self._last_analyzed_ms = video_ts_ms
        self.n_analyzed += 1
        return GateDecision(analyze=True, score=score, reason=reason, mean_luma=mean_luma)


def luma_term(mean_luma: float, *, dark: float = 40.0, good: float = 90.0) -> float:
    """How much the *lighting* lets us trust a detection, as 0..1.

    This is the term that makes the lights-off demo honest rather than a staged threshold:
    when the room goes dark the frame really does carry less information, the term collapses,
    and because confidence is the MINIMUM of the three terms the whole event drops below the
    review bar on its own. Nothing special-cases darkness.
    """
    if mean_luma <= dark:
        return max(0.0, mean_luma / dark * 0.3)
    if mean_luma >= good:
        return 1.0
    return 0.3 + 0.7 * (mean_luma - dark) / (good - dark)
