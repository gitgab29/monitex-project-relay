"""Turn an `EventDraft` from the state machine into a stored, routed `Event`.

This is the seam where the deterministic half of the system meets the probabilistic half.
The order matters and is deliberate:

1. Compose the confidence from the three measured terms -- before any model is consulted.
2. Build a template summary, so a usable event exists even if everything downstream fails.
3. Ask the vision backend for a better summary (Block 4). If it answers, its prose replaces
   the template and its counts are compared against YOLO's; if it does not, the template
   stands and the event is flagged for review.
4. Resolve the event id, upsert, and route **only if a row was actually created**.

Step 4 is what makes re-running safe: everything above it is pure computation, and the single
side-effecting step is guarded by the store's insert/skip answer.
"""

from __future__ import annotations

import logging

from .config import Settings
from .ids import video_ts
from .rules import classify, compose_confidence, template_summary
from .schema import Event, Observation

log = logging.getLogger(__name__)


def enrich_and_store(draft, obs: Observation, pipe) -> Event | None:
    """Called by the pipeline for each draft the state machine emits.

    `close` drafts only extend an existing occurrence's extent, so they are upserted but never
    re-routed -- a guard standing up should not send a second email about them sitting down.
    """
    cfg: Settings = pipe.cfg
    store = pipe.store

    category, priority = classify(draft.observed)
    confidence = compose_confidence(draft.yolo_term, draft.zone_term, draft.luma_term)
    summary = template_summary(draft.observed, cfg.site_id, obs.post_count, obs.approach_count)
    summary_source = "template"
    vision_conf = vision_agrees = None
    reasons: list[str] = []

    # --- stage 2: the model, when there is one (wired in Block 4) --------------------
    backend = getattr(pipe, "vision", None)
    if backend is not None and draft.kind == "open":
        verdict, reasons_from_vision, vconf, agrees = backend.describe(draft, obs, pipe)
        reasons.extend(reasons_from_vision)
        if verdict is not None:
            summary = verdict.summary
            summary_source = backend.name
            vision_conf, vision_agrees = vconf, agrees
            if agrees is False:
                # A confident disagreement between the detector and the model is exactly the
                # case a human should look at, so it lowers confidence rather than picking a
                # winner.
                confidence = round(min(confidence, 0.5), 4)
            elif vconf is not None:
                confidence = round((confidence + vconf) / 2, 4)

    # --- review triggers (extended in Block 5) ---------------------------------------
    if cfg.site_id is None:
        reasons.append("missing_site_id")
    if draft.straddle:
        reasons.append("zone_straddle")
    threshold = cfg.review_threshold_for(category.value, priority.value)
    if confidence < threshold:
        reasons.append("low_confidence")
        if priority.value == "high":
            reasons.append("high_priority_low_confidence")
        if category.value == "other":
            reasons.append("other_low_confidence")
    reasons = sorted(set(reasons))

    event_id, reused = store.resolve_event_id(
        pipe.session_id, draft.observed, draft.first_seen_ms,
        cfg.event_bucket_s, cfg.dedupe_window_s,
    )
    event = Event(
        event_id=event_id, source_file=pipe.source_file, category=category, priority=priority,
        observed=draft.observed, video_ts=video_ts(draft.first_seen_ms), site_id=cfg.site_id,
        summary=summary, confidence=confidence, needs_review=bool(reasons),
    )
    prov = {
        "session_id": pipe.session_id, "run_id": pipe.run_id, "track_key": draft.observed,
        "bucket": draft.first_seen_ms // (cfg.event_bucket_s * 1000),
        "first_seen_ms": draft.first_seen_ms, "last_seen_ms": draft.last_seen_ms,
        "frame_count": draft.frame_count, "yolo_term": draft.yolo_term,
        "zone_term": draft.zone_term, "luma_term": draft.luma_term,
        "vision_conf": vision_conf, "vision_agrees": vision_agrees,
        "summary_source": summary_source, "review_reasons": reasons,
    }
    inserted = store.upsert_event(event, prov)

    if not inserted:
        # A `close` draft is this run extending an occurrence it opened moments ago -- an
        # update, not a duplicate. Counting it as a skip would double the "N skipped" figure
        # that the idempotency claim is stated in, and make a single live run look like it
        # had already skipped something.
        if draft.kind == "close":
            log.debug("event %s %s -> extent updated (%d frames)",
                      event_id, draft.observed, draft.frame_count)
            return event
        pipe.stats.events_skipped += 1
        log.info(
            "event %s %s -> skipped (already recorded%s)",
            event_id, draft.observed, " via the drift window" if reused else "",
        )
        return event

    pipe.stats.events_inserted += 1
    log.info(
        "event %s %s %s/%s conf=%.2f%s summary=%r (%s)",
        event_id, draft.observed, category.value, priority.value, confidence,
        f" NEEDS REVIEW {reasons}" if reasons else "", summary[:70], summary_source,
    )
    if reasons:
        store.queue_for_review(event_id, reasons)

    pipe.set_banner(f"EVENT {draft.observed}  {category.value}/{priority.value}  conf={confidence:.2f}")

    if draft.kind == "open" and pipe.router is not None:
        result = pipe.router.route(event, reasons=reasons)
        store.set_notify_status(event_id, result.status)
    return event
