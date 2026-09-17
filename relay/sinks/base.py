"""The notification seam.

A sink takes an event and tries to tell somebody. That is the entire contract, and it is
small on purpose: adding Slack, SMS or a ticketing system means implementing `deliver` and
nothing else.

`deliver` returns a result rather than raising, because "the email bounced" is an outcome the
router has to act on, not an exception that should unwind a frame-processing loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..schema import Event


@dataclass
class DeliveryResult:
    ok: bool
    channel: str
    #: Goes straight into events.notify_status, so it is phrased to be readable in a query.
    status: str
    error: str | None = None

    @classmethod
    def success(cls, channel: str, status: str | None = None) -> DeliveryResult:
        return cls(ok=True, channel=channel, status=status or f"emailed:{channel}")

    @classmethod
    def failure(cls, channel: str, error: str) -> DeliveryResult:
        return cls(ok=False, channel=channel, status="failed", error=error)

    @classmethod
    def skipped(cls, reason: str = "store") -> DeliveryResult:
        """Not every event should page someone. A medium-priority row that is persisted and
        visible in the summary has been handled correctly."""
        return cls(ok=True, channel=reason, status="none")


class Sink(Protocol):
    name: str

    def deliver(self, event: Event, *, review_link: str | None = None) -> DeliveryResult: ...


def format_subject(event: Event, *, needs_review: bool) -> str:
    site = event.site_id or "unknown-site"
    if needs_review:
        return f"[{site}] REVIEW NEEDED: {event.observed} ({event.confidence:.2f})"
    return f"[{site}] {event.priority.upper()}: {event.observed} at {event.video_ts}"


def format_body(event: Event, *, reasons: list[str] | None = None, review_link: str | None = None) -> str:
    """The email body, written to be read on a phone at 3 a.m.

    Short on purpose. This used to end with a pretty-printed JSON dump of the whole event,
    which roughly tripled the length and pushed the one thing the reader has to act on --
    the link -- below the fold on a phone. The dump was there "in case someone wants the
    detail", but the link leads to the detail, and nobody scrolls past a wall of JSON at 3
    a.m. to find it.

    What is left is: what happened, where and how sure we are, why a human is being asked,
    and the link. Five lines and an action.
    """
    site = event.site_id or "(site not configured)"
    lines = [
        event.summary,
        "",
        f"{event.observed}  |  {event.category}/{event.priority}  |  confidence {event.confidence:.2f}",
        f"{site}  |  at {event.video_ts}",
    ]
    if reasons:
        lines += ["", "Needs a human because: " + ", ".join(reasons)]
    if review_link:
        lines += ["", f"Review it: {review_link}"]
    lines += ["", f"[{event.event_id}]"]
    return "\n".join(lines)
