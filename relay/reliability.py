"""Retries, timeouts and deliberate failure injection.

Everything here exists because the interesting question about an integration is not "does it
work" but "what does it do when it doesn't". The chaos flags make those paths demonstrable on
command instead of something you have to take on trust.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
import threading
from dataclasses import dataclass
from typing import Literal, TypeVar

log = logging.getLogger(__name__)


class CallTimeout(Exception):
    """A call exceeded its wall-clock ceiling.

    Defined here rather than imported from `vision` so this module stays dependency-free:
    retries and timeouts are general machinery, and the sinks use them too. `VisionTimeout`
    subclasses it, so vision code can keep catching its own exception type.
    """

T = TypeVar("T")

ChaosMode = Literal["timeout", "badschema", "sink-fail", "ratelimit"]


@dataclass
class ChaosConfig:
    """Forced failures, applied by FakeBackend and by the n8n sink.

    Gemini is never wrapped in chaos: `--chaos` forces `--vision fake` so a demonstration of
    the failure paths does not burn free-tier quota, and so the failure is exactly reproducible.
    """

    mode: ChaosMode | None = None
    first_n: int = 3

    def active_for(self, attempt: int) -> bool:
        return self.mode is not None and attempt <= self.first_n


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 8.0, jitter: float = 0.5) -> float:
    """Exponential backoff with jitter: min(cap, base * 2**(attempt-1)) + U(0, jitter).

    The jitter is not decoration. Without it, every client that failed at the same moment
    retries at the same moment, and a service that is merely struggling gets a synchronised
    stampede each round.
    """
    return min(cap, base * (2 ** max(0, attempt - 1))) + random.uniform(0, jitter)


def retry(
    fn: Callable[[], T], *, attempts: int = 3, base: float = 1.0, cap: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    on_attempt: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn` until it succeeds or the attempts run out.

    Re-raises the LAST exception rather than a wrapper, so the caller can still tell a timeout
    from a rate limit and record the right thing in the dead-letter row.
    """
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as e:
            last = e
            if attempt == attempts:
                break
            delay = backoff_delay(attempt, base, cap)
            if on_attempt:
                on_attempt(attempt, e, delay)
            else:
                log.warning("attempt %d/%d failed: %s; sleeping %.2fs", attempt, attempts, e, delay)
            sleep(delay)
    assert last is not None
    raise last


def run_with_timeout(fn: Callable[[], T], seconds: float) -> T:
    """Enforce a wall-clock ceiling on a blocking call.

    Belt and braces: the SDK's own http timeout is set too, but that only covers what the SDK
    considers a request. This covers the whole call, whatever it does internally -- including
    an SDK that decides to retry for us.

    A fresh daemon thread per call, deliberately, rather than a shared pool. Python cannot
    kill a thread, so a call that times out is abandoned and keeps running. With a bounded
    pool those abandoned threads accumulate until every worker is occupied and the next
    submit blocks forever -- the timeout helper becomes the thing that hangs. (Found exactly
    that way: three forced timeouts in a row deadlocked the pipeline.) A daemon thread costs
    microseconds at one call per second, never starves a later call, and does not hold up
    interpreter exit.
    """
    box: dict[str, object] = {}
    done = threading.Event()

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 -- re-raised on the calling thread below
            box["error"] = e
        finally:
            done.set()

    threading.Thread(target=target, daemon=True, name="relay-timeout").start()
    if not done.wait(seconds):
        raise CallTimeout(f"call exceeded {seconds:.1f}s")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]  # type: ignore[return-value]


class RateLimiter:
    """Client-side minimum interval between calls.

    A free tier's published RPM is a ceiling, not a target. Pacing ourselves under it is far
    cheaper than discovering the limit through 429s during a demo.
    """

    def __init__(self, min_interval_s: float):
        self.min_interval_s = float(min_interval_s)
        self._last: float | None = None

    def wait(self, sleep: Callable[[float], None] = time.sleep) -> float:
        if self.min_interval_s <= 0:
            return 0.0
        now = time.monotonic()
        if self._last is not None:
            remaining = self.min_interval_s - (now - self._last)
            if remaining > 0:
                log.debug("rate limiter: waiting %.2fs", remaining)
                sleep(remaining)
                self._last = time.monotonic()
                return remaining
        self._last = time.monotonic()
        return 0.0
