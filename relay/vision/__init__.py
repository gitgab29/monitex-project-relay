"""Vision backends, selected by one environment variable.

`VISION_BACKEND=gemini|fake|none` is the whole swap. Nothing above this package imports a
concrete backend, which is what makes "we could move to another provider" a claim with
evidence behind it rather than an aspiration.
"""

from __future__ import annotations

import logging

from ..config import Settings
from ..reliability import ChaosConfig
from .base import (
    VisionBackend,
    VisionBadSchema,
    VisionError,
    VisionRateLimited,
    VisionRequest,
    VisionTimeout,
    VisionUnavailable,
    counts_agree,
)

log = logging.getLogger(__name__)

__all__ = [
    "VisionBackend", "VisionRequest", "VisionError", "VisionTimeout", "VisionRateLimited",
    "VisionBadSchema", "VisionUnavailable", "counts_agree", "get_backend",
]


def get_backend(cfg: Settings, chaos: ChaosConfig | None = None) -> VisionBackend:
    """Pick a backend, and fall back rather than crash.

    A missing API key is a normal condition (a reviewer running from a clean checkout), not an
    error: it degrades to the null backend, which means template summaries and needs_review.
    Chaos forces the fake backend so a failure demo never burns free-tier quota and is exactly
    reproducible.
    """
    if chaos and chaos.mode:
        from .fake import FakeBackend

        log.info("chaos mode %r active -> using the fake backend (never the real API)", chaos.mode)
        return FakeBackend(chaos)

    name = (cfg.vision_backend or "none").lower()
    if name == "fake":
        from .fake import FakeBackend

        return FakeBackend(chaos)
    if name == "gemini":
        from .gemini import GeminiBackend

        try:
            return GeminiBackend(cfg)
        except VisionUnavailable as e:
            log.warning("gemini unavailable (%s); falling back to template summaries", e)
    from .null import NullBackend

    return NullBackend()
