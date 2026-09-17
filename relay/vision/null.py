"""The no-model backend.

`--vision none` is not a degraded mode bolted on for testing -- it is the path a reviewer
takes. `relay replay samples/reception_demo.mp4 --vision none --no-n8n` runs the entire
pipeline from a clean checkout with no API key, no Docker and no network, and still produces
events, because every event has a usable template summary before the model is ever consulted.

Raising rather than returning a blank verdict keeps one rule true everywhere: a summary that
did not come from a model is always a template, and the event always says which.
"""

from __future__ import annotations

from ..schema import VisionVerdict
from .base import VisionRequest, VisionUnavailable


class NullBackend:
    name = "none"

    def verdict(self, req: VisionRequest, *, repair_error: str | None = None) -> VisionVerdict:
        raise VisionUnavailable("no vision backend configured (VISION_BACKEND=none)")
