"""The real backend: Gemini via google-genai, with a structured-output contract.

Two things are worth pointing at:

**The schema is enforced at the API, not just parsed afterwards.** `response_json_schema` is
VisionVerdict's own JSON schema, so the model is constrained rather than merely asked nicely.
Pydantic still validates the reply, because "constrained" is not "guaranteed" -- and when that
validation fails, the repair path exists precisely because this is the failure that actually
happens in practice.

**Errors are classified before they are raised.** A 429 and a hung socket both look like "it
didn't work", but one should be retried patiently and the other quickly; mapping them to
distinct exceptions is what lets the retry policy and the dead-letter row say something true.
"""

from __future__ import annotations

import logging

from ..config import Settings
from ..schema import VisionVerdict
from .base import (
    VisionBadSchema,
    VisionError,
    VisionRateLimited,
    VisionRequest,
    VisionTimeout,
    VisionUnavailable,
)
from .prompts import REPAIR_SYSTEM, VERDICT_SYSTEM, repair_user, verdict_user

log = logging.getLogger(__name__)


def _classify(exc: Exception) -> Exception:
    """Map an SDK exception onto one of ours, on the text since the SDK's types vary by
    version and a demo should not fall over because an exception class moved."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(s in text for s in ("429", "resource_exhausted", "quota", "rate limit")):
        return VisionRateLimited(str(exc))
    if any(s in text for s in ("timeout", "timed out", "deadline")):
        return VisionTimeout(str(exc))
    if any(s in text for s in ("api key", "permission", "unauthenticated", "401", "403")):
        return VisionUnavailable(str(exc))
    return VisionError(f"{type(exc).__name__}: {exc}")


class GeminiBackend:
    name = "gemini"

    def __init__(self, cfg: Settings):
        if not cfg.gemini_api_key:
            raise VisionUnavailable("GEMINI_API_KEY is not set")
        from google import genai

        self.cfg = cfg
        self.model = cfg.gemini_model
        self.client = genai.Client(api_key=cfg.gemini_api_key)
        self.last_raw: str = ""

    def verdict(self, req: VisionRequest, *, repair_error: str | None = None) -> VisionVerdict:
        from google.genai import types

        if repair_error:
            system = REPAIR_SYSTEM
            user = repair_user(repair_error, self.last_raw)
        else:
            system = VERDICT_SYSTEM
            user = verdict_user(
                req.observed, req.yolo_person_count, req.yolo_post_count, req.site_id, req.video_ts
            )

        contents = [
            types.Part.from_bytes(data=req.jpeg, mime_type="image/jpeg"),
            types.Part.from_text(text=user),
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=VisionVerdict.model_json_schema(),
            temperature=0.2,          # description, not creative writing
            max_output_tokens=400,
            http_options=types.HttpOptions(timeout=int(self.cfg.vision_timeout_s * 1000)),
        )

        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except Exception as e:  # SDK exception types vary by version; classify on the message
            raise _classify(e) from e

        raw = (getattr(resp, "text", None) or "").strip()
        self.last_raw = raw
        if not raw:
            raise VisionBadSchema("model returned an empty response", raw_text="")

        try:
            return VisionVerdict.model_validate_json(raw)
        except Exception as e:
            raise VisionBadSchema(
                "response did not match VisionVerdict", raw_text=raw, validation_error=str(e)
            ) from e
