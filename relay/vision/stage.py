"""The policy around the model call: encode, pace, retry, repair, give up gracefully.

The backend is a dumb pipe that either answers or raises. Everything about *how hard we try*
and *what we do when it fails* lives here, in one place, so the answer to "what happens when
the model is down" is a function you can read rather than a story you have to believe.

The contract this upholds, in order of importance:

1. A failure never loses an event -- the template summary already exists before the call.
2. A failure is always visible: `needs_review`, a reason code, and a dead-letter row.
3. A failure never blocks capture -- this runs on the analysis worker, not the frame loop.
"""

from __future__ import annotations

import logging
import time

import cv2

from ..config import Settings
from ..ids import video_ts
from ..reliability import CallTimeout, ChaosConfig, RateLimiter, retry, run_with_timeout
from ..schema import Observation, VisionVerdict
from .base import (
    VisionBadSchema,
    VisionError,
    VisionRateLimited,
    VisionRequest,
    VisionTimeout,
    VisionUnavailable,
    counts_agree,
)

log = logging.getLogger(__name__)


class VisionStage:
    """Wraps a backend with the retry/repair policy. `describe` is what enrich.py calls."""

    def __init__(self, backend, cfg: Settings, chaos: ChaosConfig | None = None):
        self.backend = backend
        self.cfg = cfg
        self.chaos = chaos or ChaosConfig()
        self.limiter = RateLimiter(cfg.vision_min_interval_s)

    @property
    def name(self) -> str:
        return self.backend.name

    def _encode(self, image) -> bytes:
        """Downscale then JPEG. Uploading 1280x720 costs latency and quota for detail the
        model does not use to count people in a room."""
        h, w = image.shape[:2]
        if w > self.cfg.vision_max_width:
            scale = self.cfg.vision_max_width / w
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.vision_jpeg_quality]
        )
        if not ok:
            raise VisionError("failed to JPEG-encode the key frame")
        return buf.tobytes()

    def describe(self, draft, obs: Observation, pipe):
        """-> (verdict | None, reasons, vision_conf | None, agrees | None)

        Never raises. A vision failure is a degraded event, not a failed run.
        """
        if draft.key_frame is None:
            return None, ["vision_unavailable"], None, None

        req = VisionRequest(
            jpeg=self._encode(draft.key_frame), observed=draft.observed,
            yolo_person_count=obs.person_count, yolo_post_count=obs.post_count,
            site_id=self.cfg.site_id, video_ts=video_ts(draft.first_seen_ms),
        )

        try:
            verdict = self._call_with_retries(req, pipe)
        except VisionUnavailable:
            # Not a fault: no key, or --vision none. The template summary stands, and the
            # event is flagged so nobody mistakes it for a model-written one.
            return None, ["vision_unavailable"], None, None
        except VisionBadSchema as e:
            self._dead_letter(pipe, e, req, attempts=self.cfg.retry_attempts + 1)
            return None, ["vision_bad_schema"], None, None
        except (VisionTimeout, VisionRateLimited, VisionError) as e:
            self._dead_letter(pipe, e, req, attempts=self.cfg.retry_attempts)
            reason = (
                "vision_timeout" if isinstance(e, VisionTimeout)
                else "vision_rate_limited" if isinstance(e, VisionRateLimited)
                else "vision_failed"
            )
            return None, [reason], None, None

        agrees = counts_agree(verdict, req)
        reasons: list[str] = []
        if not agrees and verdict.confidence >= 0.7:
            # A confident disagreement is the case a human should see. An unconfident one is
            # just the model hedging, and is not worth anyone's attention.
            reasons.append("stage_disagreement")
            log.info(
                "stage disagreement: detector saw %d, model saw %d (model conf %.2f)",
                req.yolo_person_count, verdict.person_count, verdict.confidence,
            )
        if verdict.lighting == "dark":
            reasons.append("model_reports_dark")
        return verdict, reasons, verdict.confidence, agrees

    def _call_with_retries(self, req: VisionRequest, pipe) -> VisionVerdict:
        cfg = self.cfg

        def attempt_once(repair_error: str | None = None) -> VisionVerdict:
            self.limiter.wait()
            started = time.perf_counter()
            purpose = "repair" if repair_error else "verdict"
            try:
                try:
                    v = run_with_timeout(
                        lambda: self.backend.verdict(req, repair_error=repair_error),
                        cfg.vision_timeout_s,
                    )
                except CallTimeout as e:
                    raise VisionTimeout(str(e)) from e
            except Exception as e:
                self._log_call(pipe, started, ok=False, purpose=purpose, error=str(e))
                raise
            self._log_call(pipe, started, ok=True, purpose=purpose)
            return v

        try:
            return retry(
                attempt_once, attempts=cfg.retry_attempts, base=cfg.retry_base_s,
                cap=cfg.retry_cap_s, retry_on=(VisionTimeout, VisionRateLimited),
                on_attempt=lambda n, e, d: log.warning(
                    "vision attempt %d/%d failed: %s; sleeping %.2fs", n, cfg.retry_attempts, e, d
                ),
            )
        except VisionBadSchema as e:
            # Exactly one repair attempt. A model that cannot produce the schema when shown
            # its own error is not going to manage it on the third try, and each attempt costs
            # quota and seconds a dispatcher is waiting through.
            log.warning("vision returned an unparseable reply (%s); attempting one repair",
                        e.validation_error or e)
            return attempt_once(repair_error=e.validation_error or str(e))

    def _log_call(self, pipe, started: float, *, ok: bool, purpose: str,
                  error: str | None = None) -> None:
        latency_ms = int((time.perf_counter() - started) * 1000)
        pipe.stats.vision_calls += 1
        if not ok:
            pipe.stats.vision_failures += 1
        try:
            pipe.store.log_vision_call(
                event_id=None, run_id=pipe.run_id,
                model=getattr(self.backend, "model", self.name),
                attempt=pipe.stats.vision_calls, purpose=purpose, latency_ms=latency_ms,
                ok=ok, error=error, raw_text=getattr(self.backend, "last_raw", None),
            )
        except Exception:
            log.debug("could not record vision call metrics", exc_info=True)

    @staticmethod
    def _dead_letter(pipe, exc: Exception, req: VisionRequest, attempts: int) -> None:
        log.error("vision failed after %d attempt(s): %s -- degrading to a template summary",
                  attempts, exc)
        try:
            pipe.store.dead_letter(
                stage="vision", error=f"{type(exc).__name__}: {exc}", attempts=attempts,
                payload={
                    "observed": req.observed, "video_ts": req.video_ts,
                    "yolo_person_count": req.yolo_person_count,
                    "raw_text": str(getattr(exc, "raw_text", ""))[:500],
                },
            )
        except Exception:
            log.exception("could not write the dead-letter row")
