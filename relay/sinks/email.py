"""Direct SMTP, used when n8n is not there.

This is the fallback that makes the n8n dependency optional rather than load-bearing. If the
container is down, the workflow was never imported, or a reviewer simply has not set n8n up,
high-priority events still reach a human -- the routing is just less pretty.
"""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

from ..config import Settings
from ..reliability import retry
from ..schema import Event
from .base import DeliveryResult, format_body, format_subject

log = logging.getLogger(__name__)


class SmtpEmailSink:
    name = "python"

    def __init__(self, cfg: Settings):
        self.cfg = cfg

    @property
    def configured(self) -> bool:
        return bool(self.cfg.smtp_user and self.cfg.smtp_app_password and self.cfg.alert_to)

    def deliver(
        self, event: Event, *, review_link: str | None = None, reasons: list[str] | None = None
    ) -> DeliveryResult:
        if not self.configured:
            # Missing credentials is a configuration state, not a delivery failure: saying so
            # plainly is more useful than a stack trace, and it keeps a keyless clean checkout
            # from looking broken.
            log.warning("SMTP is not configured (SMTP_USER / SMTP_APP_PASSWORD / ALERT_TO); "
                        "event %s stored but not emailed", event.event_id)
            return DeliveryResult(ok=False, channel="python", status="none",
                                  error="smtp_not_configured")

        msg = EmailMessage()
        msg["Subject"] = format_subject(event, needs_review=event.needs_review)
        msg["From"] = self.cfg.smtp_user
        msg["To"] = self.cfg.alert_to
        msg.set_content(format_body(event, reasons=reasons, review_link=review_link))

        def send() -> None:
            with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port, timeout=20) as s:
                s.starttls()
                s.login(self.cfg.smtp_user, self.cfg.smtp_app_password)
                s.send_message(msg)

        try:
            retry(
                send, attempts=self.cfg.retry_attempts, base=self.cfg.retry_base_s,
                cap=self.cfg.retry_cap_s, retry_on=(smtplib.SMTPException, OSError),
                on_attempt=lambda n, e, d: log.warning(
                    "smtp attempt %d/%d failed: %s; sleeping %.2fs",
                    n, self.cfg.retry_attempts, e, d
                ),
            )
        except Exception as e:
            log.error("smtp delivery failed for %s: %s", event.event_id, e)
            return DeliveryResult.failure("python", f"{type(e).__name__}: {e}")

        log.info("emailed %s via smtplib -> %s", event.event_id, self.cfg.alert_to)
        return DeliveryResult.success("python")
