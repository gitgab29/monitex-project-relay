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


#: The state machine's labels are identifiers -- fine in a database, wrong in a sentence
#: somebody reads on a phone. Each one gets a headline and the action it implies.
HEADLINE: dict[str, tuple[str, str]] = {
    "unidentified_person_at_post": (
        "Unidentified person at your post",
        "Someone who is not the assigned officer is at the desk. Worth a look now.",
    ),
    "post_unattended": (
        "Your post is unattended",
        "Nobody is at the desk. Check whether the officer has stepped away.",
    ),
    "person_loitering_near_entry": (
        "Someone is loitering near the entry",
        "A person has stayed by the entrance without coming in.",
    ),
    "post_manned": (
        "Your post is manned",
        "The desk is covered. No action needed.",
    ),
}


def headline_for(event: Event) -> tuple[str, str]:
    """Plain-English headline and next step. Unknown states degrade to something readable
    rather than leaking an identifier into the subject line."""
    return HEADLINE.get(
        event.observed,
        (event.observed.replace("_", " ").capitalize(), "Worth a look."),
    )


def _when(event: Event) -> str:
    """`video_ts` is an offset into the recording, which reads like nonsense on its own -- an
    alert saying "at 00:00:01.00" tells a human nothing. Label it for what it is."""
    return f"{event.video_ts} into the recording"


def format_subject(event: Event, *, needs_review: bool) -> str:
    site = event.site_id or "unknown site"
    head, _ = headline_for(event)
    if needs_review:
        return f"[{site}] Check please: {head.lower()}"
    if event.priority == "high":
        return f"[{site}] {head}"
    return f"[{site}] {head.lower()}"


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
    site = event.site_id or "an unconfigured site"
    head, action = headline_for(event)

    lines = [
        f"{head}.",
        f"{site} - {_when(event)}",
        "",
        action,
    ]
    # The model's own sentence, only when it adds something the headline did not already say.
    if event.summary and event.summary.rstrip(".").lower() not in head.lower():
        lines += ["", f"What the camera saw: {event.summary}"]

    if reasons:
        lines += ["", "We are not certain: " + _explain(reasons) + "."]

    lines += [
        "",
        "Details",
        f"  Confidence   {event.confidence:.0%}",
        f"  Severity     {event.priority}",
        f"  Type         {event.category}",
        f"  Recording    {event.source_file}",
    ]
    if review_link:
        lines += ["", f"Review it here: {review_link}"]
    lines += ["", f"Reference {event.event_id}"]
    return "\n".join(lines)


#: Reason codes are for the database. This is what they mean to the person being paged.
_REASON_PROSE: dict[str, str] = {
    "low_confidence": "the reading was not a confident one",
    "high_priority_low_confidence": "this is urgent but the reading was not confident",
    "other_low_confidence": "it did not fit a known pattern cleanly",
    "missing_site_id": "no site is configured, so this could not be routed properly",
    "missing_video_ts": "the timestamp into the recording is missing",
    "zone_straddle": "the person was right on the edge of the post area",
    "stage_disagreement": "the detector and the model counted different numbers of people",
    "vision_unavailable": "the description was written automatically, not by the model",
    "vision_bad_schema": "the model's reply could not be read",
    "vision_timeout": "the model did not answer in time",
    "vision_rate_limited": "the model was rate-limited",
    "vision_failed": "the model call failed",
    "model_reports_dark": "the model said the frame was too dark to read",
}


def _explain(reasons: list[str]) -> str:
    parts = [_REASON_PROSE.get(r, r.replace("_", " ")) for r in reasons]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]
