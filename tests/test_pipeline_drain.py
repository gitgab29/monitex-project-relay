"""The shutdown drain must outlast the retry chain it is draining.

This is a regression test for a bug that cost the degrade path its own evidence. The join at
the end of a run was a flat 30 s, while a vision call that keeps timing out costs
`vision_timeout_s` per attempt plus backoff -- 39 s at the defaults. So on a short clip the
worker was abandoned mid-retry: the event shipped without its `vision_timeout` review reason
and without its dead-letter row, and `--chaos timeout` appeared to show backoff and then
nothing. The failure was invisible precisely because the abandoned work was the bookkeeping.
"""

from __future__ import annotations

import pytest

from relay.config import Settings
from relay.pipeline import Pipeline


def _pipeline(cfg: Settings) -> Pipeline:
    return Pipeline(cfg, store=None, mode="replay")


def _cfg(**over) -> Settings:
    return Settings(**over)


def test_drain_outlasts_the_worst_case_retry_chain() -> None:
    cfg = _cfg(vision_timeout_s=12.0, retry_attempts=3,
               retry_base_s=1.0, retry_cap_s=8.0)
    worst_case = 3 * 12.0 + (1.0 + 2.0)  # three attempts, two backoffs between them
    assert _pipeline(cfg)._drain_timeout_s() > worst_case


def test_drain_beats_the_old_hardcoded_thirty_seconds() -> None:
    """At the shipped defaults the old constant was too small. That is the actual bug."""
    cfg = _cfg(vision_timeout_s=12.0, retry_attempts=3,
               retry_base_s=1.0, retry_cap_s=8.0)
    assert _pipeline(cfg)._drain_timeout_s() > 30.0


@pytest.mark.parametrize("attempts", [1, 2, 3, 5, 8])
def test_drain_scales_with_attempts(attempts: int) -> None:
    cfg = _cfg(vision_timeout_s=4.0, retry_attempts=attempts,
               retry_base_s=1.0, retry_cap_s=8.0)
    drain = _pipeline(cfg)._drain_timeout_s()
    backoff = sum(min(8.0, 1.0 * (2 ** n)) for n in range(attempts - 1))
    assert drain >= attempts * 4.0 + backoff


def test_backoff_respects_the_cap() -> None:
    """A large retry_attempts must not imply an unbounded drain: backoff is capped."""
    cfg = _cfg(vision_timeout_s=1.0, retry_attempts=10,
               retry_base_s=1.0, retry_cap_s=8.0)
    # 9 backoffs, capped at 8 s each, so strictly less than the uncapped 2**9 sum
    assert _pipeline(cfg)._drain_timeout_s() < 10 * 1.0 + 9 * 8.0 + 5.0 + 1


def test_a_single_attempt_has_no_backoff() -> None:
    cfg = _cfg(vision_timeout_s=7.0, retry_attempts=1,
               retry_base_s=1.0, retry_cap_s=8.0)
    assert _pipeline(cfg)._drain_timeout_s() == pytest.approx(7.0 + 5.0)
