"""The decisions that are rules, not model output.

The split this module exists to enforce: **the model writes prose, the rules decide what
happens.** Category, priority, confidence and whether a human is needed are all computed here,
from numbers, deterministically, and are therefore testable and auditable. If the model is
having a bad day the worst it can do is write a poor sentence.

`review_reasons` is added in Block 5 along with the sinks that act on it.
"""

from __future__ import annotations

from .schema import Category, Priority

#: observed state -> (category, priority). The full table for this feed.
#:
#: `post_unattended` is 'other' rather than a dedicated category because the brief's six
#: categories are fixed and none of them means "the thing that should be here isn't". It is
#: high priority regardless: an unmanned reception post is the situation a dispatcher most
#: needs to know about, and 'other' carries a stricter review bar precisely because it is the
#: catch-all.
CLASSIFICATION: dict[str, tuple[Category, Priority]] = {
    "post_manned": (Category.routine, Priority.low),
    "post_unattended": (Category.other, Priority.high),
    "person_loitering_near_entry": (Category.loitering, Priority.medium),
    "unidentified_person_at_post": (Category.intrusion, Priority.high),
}


def classify(observed: str) -> tuple[Category, Priority]:
    """Unknown states degrade to other/medium rather than raising: a new state added to the
    machine should produce a reviewable event, not crash a running system."""
    return CLASSIFICATION.get(observed, (Category.other, Priority.medium))


def compose_confidence(yolo_term: float, zone_term: float, luma_term: float) -> float:
    """Confidence is the MINIMUM of the three terms -- as certain as the weakest signal.

    A product would be wrong: three independent 0.9s would multiply to 0.73 and put a
    perfectly healthy detection under the review bar. The minimum says something defensible
    in one sentence -- "we are only as sure as the least reliable thing we measured" -- and it
    is what makes the lights-off demo honest: a dark frame collapses `luma_term`, which drags
    the whole event down on its own without anything special-casing darkness.
    """
    return round(min(yolo_term, zone_term, luma_term), 4)


def template_summary(observed: str, site_id: str | None, post_count: int, approach_count: int) -> str:
    """The summary used when no model is available -- keyless replay, a timeout, a bad schema.

    Written to be genuinely useful rather than a placeholder, because this is what a
    dispatcher reads when the model is down. It says what was seen and that nobody wrote it.
    """
    where = site_id or "an unspecified site"
    people = (
        f"{post_count} person(s) in the post zone, {approach_count} in the approach zone"
        if (post_count or approach_count)
        else "no people detected"
    )
    phrasing = {
        "post_manned": f"Guard post at {where} is manned",
        "post_unattended": f"Guard post at {where} appears unattended",
        "person_loitering_near_entry": f"A person is lingering near the entrance at {where}",
        "unidentified_person_at_post": f"More than one person is at the guard post at {where}",
    }.get(observed, f"{observed} at {where}")
    return f"{phrasing} ({people}). Auto-generated: no model summary available."


#: Every reason code the system can attach to an event, with the one-line explanation the
#: README table is built from. Keeping them in one dict means a new trigger cannot be added
#: without documenting it.
REVIEW_REASONS: dict[str, str] = {
    "low_confidence": "composed confidence fell below the threshold for this category/priority",
    "high_priority_low_confidence": "a high-priority event did not clear the stricter 0.90 bar",
    "other_low_confidence": "an 'other' event did not clear the stricter 0.85 bar",
    "missing_site_id": "SITE_ID is not configured, so the event cannot be routed to a site",
    "missing_video_ts": "the event has no timestamp into the source recording",
    "zone_straddle": "a detection sat on the post-zone boundary and was resolved by size",
    "stage_disagreement": "the detector and the model disagreed on how many people are present",
    "vision_unavailable": "no model was reachable, so the summary is a template",
    "vision_bad_schema": "the model's reply could not be parsed, even after one repair",
    "vision_timeout": "the model did not answer within the timeout, after retries",
    "vision_rate_limited": "the model rate-limited us, after retries",
    "vision_failed": "the model call failed for an unclassified reason",
    "model_reports_dark": "the model itself said the frame was too dark to read",
}


def review_reasons(
    *, confidence: float, category: str, priority: str, site_id: str | None,
    video_ts: str | None, straddle: bool, cfg, extra: list[str] | None = None,
) -> list[str]:
    """Every reason this event should be looked at by a person.

    Two genuinely different kinds of trigger live here, and the distinction is the point:

    * **Low confidence** -- the system saw something but is not sure enough to act. The bar
      moves with the cost of being wrong (see `Settings.review_threshold_for`).
    * **Incomplete data** -- the system is perfectly confident about a record it cannot use.
      A missing `site_id` is not a low-confidence detection; it is an event nobody can route,
      and no amount of model certainty fixes it.

    Returned sorted and de-duplicated so the list is stable across runs and therefore
    comparable between a live run and its replay.
    """
    reasons = list(extra or [])

    if site_id is None:
        reasons.append("missing_site_id")
    if not video_ts:
        reasons.append("missing_video_ts")
    if straddle:
        reasons.append("zone_straddle")

    threshold = cfg.review_threshold_for(category, priority)
    if confidence < threshold:
        reasons.append("low_confidence")
        if priority == "high":
            reasons.append("high_priority_low_confidence")
        if category == "other":
            reasons.append("other_low_confidence")

    unknown = set(reasons) - set(REVIEW_REASONS)
    if unknown:
        # A reason with no documented meaning is a reason nobody can act on.
        raise ValueError(f"undocumented review reason(s): {sorted(unknown)}")
    return sorted(set(reasons))
