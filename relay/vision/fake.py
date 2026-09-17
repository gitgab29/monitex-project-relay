"""A deterministic stand-in for the real model, and the home of the chaos modes.

Two jobs:

* **Tests and keyless demos.** It returns a plausible verdict derived from the detector's own
  counts, with no network and no quota, so the whole pipeline can be exercised by anyone who
  clones the repo.
* **Failure injection.** `--chaos timeout|badschema|ratelimit` makes the recovery paths run on
  demand. Injecting the failure in the *backend* rather than in the retry helper means the
  real code path is exercised -- the pipeline cannot tell the difference between this and a
  genuinely unhappy Gemini.
"""

from __future__ import annotations

import logging
import time

from ..reliability import ChaosConfig
from ..schema import VisionVerdict
from .base import VisionBadSchema, VisionRateLimited, VisionRequest, VisionTimeout

log = logging.getLogger(__name__)

#: What a malformed reply actually looks like: a model wrapping JSON in prose and fences, and
#: dropping a required field. Invented garbage would not exercise the same parsing path.
GARBAGE_REPLY = """Sure! Here is the analysis of the frame you provided:

```json
{"summary": "A person appears to be seated at the reception desk.",
 "people_at_desk": 1, "lighting": "good"}
```

Let me know if you would like me to look at anything else."""


class FakeBackend:
    name = "fake"

    def __init__(self, chaos: ChaosConfig | None = None, latency_s: float = 0.0):
        self.chaos = chaos or ChaosConfig()
        self.latency_s = latency_s
        self.calls = 0

    def verdict(self, req: VisionRequest, *, repair_error: str | None = None) -> VisionVerdict:
        self.calls += 1
        attempt = self.calls
        mode = self.chaos.mode

        if mode and self.chaos.active_for(attempt):
            if mode == "timeout":
                # Sleep past any plausible ceiling so run_with_timeout is what fires, exactly
                # as it would against a real hung request.
                log.debug("chaos: sleeping to force a timeout (call %d)", attempt)
                time.sleep(3600)
            if mode == "ratelimit":
                raise VisionRateLimited("429 RESOURCE_EXHAUSTED (injected)")
            if mode == "badschema":
                raise VisionBadSchema(
                    "response did not match VisionVerdict (injected)",
                    raw_text=GARBAGE_REPLY,
                    validation_error="person_count: Field required",
                )

        if self.latency_s:
            time.sleep(self.latency_s)

        # A plausible verdict, derived from what the detector saw so agreement holds by
        # default and a test can make it disagree on purpose.
        n = req.yolo_person_count
        at_desk = req.yolo_post_count
        text = {
            "post_manned": "A person is seated at the reception desk, facing the camera.",
            "post_unattended": "The reception desk is empty; no one is visible at the post.",
            "person_loitering_near_entry": "A person is standing near the entrance, not moving on.",
            "unidentified_person_at_post": "Two people are standing at the reception desk.",
        }.get(req.observed, f"Scene shows {n} person(s).")
        return VisionVerdict(
            summary=text, person_count=n, people_at_desk=at_desk,
            lighting="good", confidence=0.86,
        )


class DisagreeingFakeBackend(FakeBackend):
    """Always sees a different number of people than the detector did.

    Used to prove that a confident disagreement between the two stages routes to a human
    rather than one stage silently overruling the other.
    """

    name = "fake-disagree"

    def verdict(self, req: VisionRequest, *, repair_error: str | None = None) -> VisionVerdict:
        v = super().verdict(req, repair_error=repair_error)
        return v.model_copy(update={"person_count": v.person_count + 3, "confidence": 0.82})
