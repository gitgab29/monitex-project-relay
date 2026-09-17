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
    """The email body.

    Ordered for someone reading it on a phone at 3 a.m.: what happened, then why we are
    unsure, then the link to act, then the raw record for anyone who wants it.
    """
    import json

    lines = [
        event.summary,
        "",
        f"site:       {event.site_id or '(not configured)'}",
        f"observed:   {event.observed}",
        f"category:   {event.category} / {event.priority}",
        f"confidence: {event.confidence:.2f}",
        f"at:         {event.video_ts} in {event.source_file}",
    ]
    if reasons:
        lines += ["", "flagged for review because:"]
        lines += [f"  - {r}" for r in reasons]
    if review_link:
        lines += ["", f"review: {review_link}"]
    lines += ["", "--- event record ---", json.dumps(event.model_dump(), indent=2)]
    return "\n".join(lines)
