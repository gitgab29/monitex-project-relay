"""What the system does when the model misbehaves.

Every test here asserts the same three things in different failure modes, because they are
the promises the README makes:

1. an event is still produced,
2. it is flagged for a human with a reason that names the failure,
3. there is a dead-letter row as evidence.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from relay.config import Settings
from relay.detect.state import EventDraft
from relay.reliability import CallTimeout, ChaosConfig, backoff_delay, retry, run_with_timeout
from relay.schema import Detection, Observation, VisionVerdict
from relay.store import Store
from relay.vision.base import VisionRateLimited, VisionTimeout
from relay.vision.fake import DisagreeingFakeBackend, FakeBackend
from relay.vision.null import NullBackend
from relay.vision.stage import VisionStage


@pytest.fixture()
def cfg() -> Settings:
    # Short waits: these tests assert behaviour, not patience.
    return Settings(vision_timeout_s=0.5, retry_attempts=3, retry_base_s=0.01,
                    retry_cap_s=0.05, vision_min_interval_s=0.0, site_id="site-118")


@dataclasses.dataclass
class _Stats:
    vision_calls: int = 0
    vision_failures: int = 0


class FakePipe:
    """The few attributes VisionStage touches on a pipeline."""

    def __init__(self, store: Store):
        self.store = store
        self.run_id = "run-test"
        self.stats = _Stats()


@pytest.fixture()
def pipe(tmp_path):
    return FakePipe(Store(tmp_path / "t.db"))


def draft(observed: str = "post_manned") -> EventDraft:
    return EventDraft(
        observed=observed, first_seen_ms=1000, last_seen_ms=2000, frame_count=2,
        yolo_term=0.93, zone_term=0.95, luma_term=1.0,
        key_frame=np.full((360, 640, 3), 120, dtype=np.uint8),
    )


def obs(post: int = 1, approach: int = 0) -> Observation:
    dets = [
        Detection(x0=.3, y0=.4, x1=.6, y1=.9, conf=.9, zone=z, area_frac=.15, zone_term=.9)
        for z in (["post"] * post + ["approach"] * approach)
    ]
    return Observation(session_id="s", frame_index=30, video_ts_ms=1000,
                       motion_score=5.0, mean_luma=120.0, detections=dets)


# ---------------------------------------------------------------- happy path

def test_a_working_backend_returns_a_verdict(cfg, pipe):
    stage = VisionStage(FakeBackend(), cfg)
    verdict, reasons, vconf, agrees = stage.describe(draft(), obs(), pipe)
    assert isinstance(verdict, VisionVerdict)
    assert reasons == [] and agrees is True
    assert vconf == pytest.approx(0.86)
    assert pipe.store.list_dead_letters() == []


# ---------------------------------------------------------------- timeout

def test_timeout_retries_three_times_then_degrades(cfg, pipe):
    stage = VisionStage(FakeBackend(ChaosConfig("timeout")), cfg)
    verdict, reasons, _, _ = stage.describe(draft(), obs(), pipe)
    assert verdict is None
    assert reasons == ["vision_timeout"]
    assert pipe.stats.vision_calls == 3
    dl = pipe.store.list_dead_letters()
    assert len(dl) == 1 and dl[0]["stage"] == "vision" and dl[0]["attempts"] == 3


# ---------------------------------------------------------------- bad schema

def test_bad_schema_gets_exactly_one_repair_attempt(cfg, pipe):
    """One repair, not three. A model that cannot produce the schema when shown its own
    error will not manage it on the third try, and each attempt costs quota and seconds."""
    backend = FakeBackend(ChaosConfig("badschema", first_n=99))
    verdict, reasons, _, _ = VisionStage(backend, cfg).describe(draft(), obs(), pipe)
    assert verdict is None
    assert reasons == ["vision_bad_schema"]
    assert backend.calls == 2          # the original and one repair
    assert len(pipe.store.list_dead_letters()) == 1


def test_a_repair_that_succeeds_produces_a_normal_event(cfg, pipe):
    """The path that matters most: the model fixes itself and nobody is bothered."""
    backend = FakeBackend(ChaosConfig("badschema", first_n=1))
    verdict, reasons, _, _ = VisionStage(backend, cfg).describe(draft(), obs(), pipe)
    assert verdict is not None
    assert reasons == []
    assert backend.calls == 2
    assert pipe.store.list_dead_letters() == []


def test_the_bad_reply_is_kept_as_evidence(cfg, pipe):
    backend = FakeBackend(ChaosConfig("badschema", first_n=99))
    VisionStage(backend, cfg).describe(draft(), obs(), pipe)
    assert "raw_text" in pipe.store.list_dead_letters()[0]["payload_json"]


# ---------------------------------------------------------------- rate limit

def test_rate_limit_is_retried_like_a_timeout(cfg, pipe):
    stage = VisionStage(FakeBackend(ChaosConfig("ratelimit")), cfg)
    verdict, reasons, _, _ = stage.describe(draft(), obs(), pipe)
    assert verdict is None and reasons == ["vision_rate_limited"]
    assert pipe.stats.vision_calls == 3


# ---------------------------------------------------------------- no backend

def test_the_null_backend_is_a_mode_not_a_failure(cfg, pipe):
    """`--vision none` is the path a reviewer takes from a clean checkout: no key, no
    network, still a usable event. It is flagged, but it is not a dead letter."""
    verdict, reasons, _, _ = VisionStage(NullBackend(), cfg).describe(draft(), obs(), pipe)
    assert verdict is None
    assert reasons == ["vision_unavailable"]
    assert pipe.store.list_dead_letters() == []


# ---------------------------------------------------------------- adjudication

def test_a_confident_disagreement_routes_to_a_human(cfg, pipe):
    """Neither stage overrules the other: the detector and the model looking at one frame and
    seeing different things is exactly what a person should arbitrate."""
    stage = VisionStage(DisagreeingFakeBackend(), cfg)
    verdict, reasons, _, agrees = stage.describe(draft(), obs(post=1), pipe)
    assert agrees is False
    assert "stage_disagreement" in reasons
    assert verdict is not None          # keep its summary, just do not trust the count


def test_a_one_person_difference_is_tolerated(cfg, pipe):
    """Someone half out of frame is a genuine edge case, not a disagreement worth waking
    anyone for."""
    stage = VisionStage(FakeBackend(), cfg)
    _, reasons, _, agrees = stage.describe(draft(), obs(post=1), pipe)
    assert agrees is True and "stage_disagreement" not in reasons


# ---------------------------------------------------------------- the machinery itself

def test_backoff_grows_and_is_capped():
    delays = [backoff_delay(a, base=1.0, cap=8.0, jitter=0.0) for a in range(1, 7)]
    assert delays == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_backoff_jitter_desynchronises_clients():
    """Without jitter every client that failed together retries together."""
    assert len({backoff_delay(3, jitter=0.5) for _ in range(20)}) > 1


def test_retry_reraises_the_last_error_not_a_wrapper():
    """The caller has to tell a timeout from a rate limit to record the right dead letter."""
    def always_rate_limited():
        raise VisionRateLimited("429")

    with pytest.raises(VisionRateLimited):
        retry(always_rate_limited, attempts=2, base=0.001,
              retry_on=(VisionTimeout, VisionRateLimited), sleep=lambda s: None)


def test_retry_stops_at_the_first_success():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise VisionTimeout("once")
        return "ok"

    got = retry(flaky, attempts=3, base=0.001, retry_on=(VisionTimeout,), sleep=lambda s: None)
    assert got == "ok" and len(calls) == 2


def test_abandoned_timeouts_do_not_starve_later_calls():
    """Regression. run_with_timeout used a shared 4-worker pool; a call that times out is
    abandoned and keeps running, so four forced timeouts occupied every worker and the fifth
    submit blocked forever -- the timeout helper became the thing that hung."""
    import time

    for _ in range(6):
        with pytest.raises(CallTimeout):
            run_with_timeout(lambda: time.sleep(30), 0.05)
    assert run_with_timeout(lambda: "still working", 1.0) == "still working"


def test_vision_timeout_is_a_call_timeout():
    """So run_with_timeout and the vision layer agree on what a timeout is."""
    assert issubclass(VisionTimeout, CallTimeout)


def test_chaos_never_touches_the_real_api(cfg):
    """`--chaos` must not burn free-tier quota, and a failure demo has to be reproducible."""
    from relay.vision import get_backend

    gemini_cfg = dataclasses.replace(cfg, vision_backend="gemini", gemini_api_key="not-a-real-key")
    assert get_backend(gemini_cfg, ChaosConfig("timeout")).name == "fake"


def test_a_missing_key_degrades_to_templates_rather_than_crashing(cfg):
    """A reviewer cloning the repo has no key; that must be a mode, not a stack trace."""
    from relay.vision import get_backend

    no_key = dataclasses.replace(cfg, vision_backend="gemini", gemini_api_key="")
    assert get_backend(no_key).name == "none"
