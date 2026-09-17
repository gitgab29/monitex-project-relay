"""Who gets told, and what happens when telling them fails.

Two decisions live here, and they are the ones that make the difference between a useful
system and an alarm nobody reads:

**Not everything is worth a human's attention.** A guard sitting down is recorded and shows up
in the summary; it does not page anyone. High priority and anything flagged for review do.
Medium is stored for the digest. An alerting system that alerts on everything gets muted, and
then it may as well not exist.

**A failed notification is never silent.** The primary sink is retried, then the failure is
written to the dead-letter table with its payload, then the fallback is tried. An event that
nobody could be told about is still a row somebody can find.
"""

from __future__ import annotations

import logging

from .config import Settings
from .schema import Event
from .sinks.base import DeliveryResult

log = logging.getLogger(__name__)


class Router:
    def __init__(self, cfg: Settings, store, primary=None, fallback=None):
        self.cfg = cfg
        self.store = store
        self.primary = primary
        self.fallback = fallback

    def _should_notify(self, event: Event, reasons: list[str]) -> bool:
        return bool(reasons) or event.priority == "high"

    def review_link(self, event: Event) -> str:
        return f"{self.cfg.review_link_base}/review/{event.event_id}"

    def route(self, event: Event, *, reasons: list[str] | None = None) -> DeliveryResult:
        reasons = reasons or []

        if not self._should_notify(event, reasons):
            log.info("event %s (%s/%s) stored, not notified", event.event_id,
                     event.category, event.priority)
            return DeliveryResult.skipped()

        link = self.review_link(event)
        if self.primary is not None:
            result = self.primary.deliver(event, review_link=link, reasons=reasons)
            if result.ok:
                return result
            self._dead_letter(event, result, stage=self.primary.name, reasons=reasons)
            log.warning("primary sink %r failed for %s; trying the fallback",
                        self.primary.name, event.event_id)

        if self.fallback is not None:
            result = self.fallback.deliver(event, review_link=link, reasons=reasons)
            if result.ok:
                log.info("fallback sink %r delivered %s", self.fallback.name, event.event_id)
                return result
            self._dead_letter(event, result, stage=self.fallback.name, reasons=reasons)

        log.error("no sink could deliver %s -- the event is stored and dead-lettered",
                  event.event_id)
        return DeliveryResult.failure("none", "no sink could deliver this event")

    def _dead_letter(self, event: Event, result: DeliveryResult, stage: str,
                     reasons: list[str]) -> None:
        # The payload is stored so the delivery can be replayed by hand later -- a dead letter
        # you cannot act on is just a log line with extra steps.
        try:
            self.store.dead_letter(
                stage=stage, error=result.error or "unknown", attempts=self.cfg.retry_attempts,
                payload={**event.model_dump(), "review_reasons": reasons},
                event_id=event.event_id,
            )
        except Exception:
            log.exception("could not write a dead-letter row for %s", event.event_id)


def build_router(cfg: Settings, store, *, use_n8n: bool = True, chaos=None) -> Router:
    """Assemble the sink chain.

    n8n first when it is configured, SMTP behind it, the plain webhook behind that. Each step
    down is less capable and more certain to work, which is the right shape for a fallback
    chain: degrade toward the thing with the fewest moving parts.
    """
    from .sinks.email import SmtpEmailSink
    from .sinks.n8n import N8nWebhookSink
    from .sinks.webhook import GenericWebhookSink

    smtp = SmtpEmailSink(cfg)
    webhook = GenericWebhookSink(cfg)
    last_resort = smtp if smtp.configured else (webhook if webhook.configured else None)

    if use_n8n and cfg.n8n_webhook_url:
        return Router(cfg, store, primary=N8nWebhookSink(cfg, chaos), fallback=last_resort)
    return Router(cfg, store, primary=last_resort, fallback=webhook if smtp.configured else None)
