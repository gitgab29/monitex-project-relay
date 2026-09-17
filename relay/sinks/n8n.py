"""POST the event to n8n, which owns the routing.

The division of labour, and the reason for it: Python decides *what happened* and n8n decides
*who hears about it*. Escalation policy is the thing that changes weekly in a real operation --
add a shift supervisor, route fire alarms differently, quiet hours at the weekend -- and that
belongs somewhere a non-programmer can edit and where a failed run is visible in an execution
list. Detection logic changes far less often and needs tests, so it stays in Python.

What crosses the wire is exactly the brief's event schema plus the review reasons. n8n never
sees a frame, a detection box or a database row.
"""

from __future__ import annotations

import logging

import httpx

from ..config import Settings
from ..reliability import ChaosConfig, retry
from ..schema import Event
from .base import DeliveryResult

log = logging.getLogger(__name__)


class SinkFailure(Exception):
    """Injected by --chaos sink-fail, or raised on a non-2xx response."""


class N8nWebhookSink:
    name = "n8n"

    def __init__(self, cfg: Settings, chaos: ChaosConfig | None = None):
        self.cfg = cfg
        self.chaos = chaos or ChaosConfig()
        self.calls = 0

    def deliver(
        self, event: Event, *, review_link: str | None = None, reasons: list[str] | None = None
    ) -> DeliveryResult:
        payload = {
            **event.model_dump(),
            "review_reasons": reasons or [],
            "review_link": review_link,
        }

        def post() -> httpx.Response:
            self.calls += 1
            if self.chaos.mode == "sink-fail" and self.chaos.active_for(self.calls):
                raise SinkFailure("injected sink failure")
            r = httpx.post(self.cfg.n8n_webhook_url, json=payload, timeout=self.cfg.n8n_timeout_s)
            if r.status_code >= 300:
                raise SinkFailure(f"n8n returned {r.status_code}: {r.text[:200]}")
            return r

        try:
            retry(
                post, attempts=self.cfg.retry_attempts, base=self.cfg.retry_base_s,
                cap=self.cfg.retry_cap_s,
                retry_on=(httpx.TransportError, httpx.TimeoutException, SinkFailure),
                on_attempt=lambda n, e, d: log.warning(
                    "n8n attempt %d/%d failed: %s; sleeping %.2fs",
                    n, self.cfg.retry_attempts, e, d
                ),
            )
        except Exception as e:
            log.error("n8n delivery failed for %s: %s", event.event_id, e)
            return DeliveryResult.failure("n8n", f"{type(e).__name__}: {e}")

        log.info("n8n accepted %s", event.event_id)
        # 'accepted' not 'emailed': n8n has taken responsibility, but the email has not been
        # sent yet. The workflow calls POST /events/{id}/notified when it actually has been,
        # so the two states stay distinguishable in the database.
        return DeliveryResult(ok=True, channel="n8n", status="accepted_by_n8n")
