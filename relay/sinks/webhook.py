"""A plain webhook, for when email is the thing that is broken.

Gmail app passwords depend on 2-Step Verification being on and on Google not deciding the
sign-in looks suspicious. The brief accepts a webhook as a notification channel, so this is
the escape hatch: point WEBHOOK_FALLBACK_URL at a Discord webhook or webhook.site and alerts
still land somewhere a human looks.
"""

from __future__ import annotations

import logging

import httpx

from ..config import Settings
from ..schema import Event
from .base import DeliveryResult, format_subject

log = logging.getLogger(__name__)


class GenericWebhookSink:
    name = "webhook"

    def __init__(self, cfg: Settings):
        self.cfg = cfg

    @property
    def configured(self) -> bool:
        return bool(self.cfg.webhook_fallback_url)

    def deliver(
        self, event: Event, *, review_link: str | None = None, reasons: list[str] | None = None
    ) -> DeliveryResult:
        if not self.configured:
            return DeliveryResult(ok=False, channel="webhook", status="none",
                                  error="webhook_not_configured")
        subject = format_subject(event, needs_review=event.needs_review)
        # "content" is what Discord reads; the rest is there for anything else.
        body = {
            "content": f"**{subject}**\n{event.summary}"
                       + (f"\nreview: {review_link}" if review_link else ""),
            "event": event.model_dump(),
            "review_reasons": reasons or [],
        }
        try:
            r = httpx.post(self.cfg.webhook_fallback_url, json=body, timeout=10.0)
            r.raise_for_status()
        except Exception as e:
            return DeliveryResult.failure("webhook", f"{type(e).__name__}: {e}")
        log.info("posted %s to the fallback webhook", event.event_id)
        return DeliveryResult.success("webhook", status="emailed:webhook")
