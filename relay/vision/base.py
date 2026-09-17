"""The vision backend seam.

Stage 2 of the cascade, and deliberately the smallest interface in the system. YOLO already
knows *that* there are people and *where* they are; the model is asked only for the things a
detector genuinely cannot give you -- a sentence a human can read, and an independent count
that can be compared against the detector's.

Everything the backend returns is advisory. Category, priority and the final confidence are
rules (see rules.py), so a model that hallucinates can produce a poor sentence and a
disagreement flag, and nothing else. That is the entire point of the split.

One call per event, not per frame: at ~1 analysed frame per second a per-frame call would be
both ruinous on a free tier and pointless, because nothing changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..reliability import CallTimeout
from ..schema import VisionVerdict


class VisionError(Exception):
    """Base for every failure the pipeline is expected to survive."""


class VisionUnavailable(VisionError):
    """No backend configured, or no credentials. Not a fault -- a mode."""


class VisionTimeout(VisionError, CallTimeout):
    """Also a CallTimeout, so `run_with_timeout` and this agree on what a timeout is."""


class VisionRateLimited(VisionError):
    pass


class VisionBadSchema(VisionError):
    """The model replied, but not with the shape we asked for."""

    def __init__(self, message: str, raw_text: str = "", validation_error: str = ""):
        super().__init__(message)
        self.raw_text = raw_text
        self.validation_error = validation_error


@dataclass
class VisionRequest:
    jpeg: bytes
    observed: str
    yolo_person_count: int
    yolo_post_count: int
    site_id: str | None
    video_ts: str


class VisionBackend(Protocol):
    name: str

    def verdict(self, req: VisionRequest, *, repair_error: str | None = None) -> VisionVerdict:
        """Return a verdict or raise one of the VisionErrors above.

        `repair_error` is set on the single retry after a schema failure: the model is shown
        what was wrong with its previous answer and asked again.
        """
        ...


def counts_agree(verdict: VisionVerdict, req: VisionRequest, tolerance: int = 1) -> bool:
    """Does the model's count match the detector's?

    This is the adjudication, and it is a comparison rather than a second prompt -- asking a
    model to grade itself mostly measures its agreeableness. A tolerance of one person absorbs
    the genuinely ambiguous edge cases (someone half out of frame) without hiding a real
    disagreement like "YOLO says two at the desk, the model says nobody is there".
    """
    return abs(verdict.person_count - req.yolo_person_count) <= tolerance
