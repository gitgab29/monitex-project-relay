"""What actually goes over the SMTP wire.

tests/test_sinks.py proves the routing and the text; it never exercises the transport. So the
one part that can only fail against a real server -- the order of starttls/login/send, and the
headers on the message object -- was untested. Calling login() before starttls() leaks the
password in clear text and still passes every text-level assertion, which is exactly the kind
of bug that should not be discovered live on stage.

A fake SMTP stands in for Gmail. That keeps this runnable with no credentials, no network and
no budget, and it means the only untested step left is Gmail's own handshake.
"""

from __future__ import annotations

import smtplib

import pytest

from relay.config import Settings
from relay.schema import Category, Event, Priority
from relay.sinks.email import SmtpEmailSink


@pytest.fixture()
def cfg() -> Settings:
    return Settings(site_id="site-118", retry_attempts=2, retry_base_s=0.001, retry_cap_s=0.01,
                    review_link_base="http://localhost:8080",
                    smtp_host="smtp.example.com", smtp_port=587,
                    smtp_user="guard@example.com", smtp_app_password="app-password",
                    alert_to="duty@example.com")


def event(**kw) -> Event:
    base = dict(event_id="e1", source_file="data/sessions/s.mp4",
                category=Category.intrusion, priority=Priority.high,
                observed="unidentified_person_at_post", video_ts="00:04:12.40",
                site_id="site-118", summary="Someone unrecognised is at the post.",
                confidence=0.91, needs_review=False)
    base.update(kw)
    return Event(**base)


class FakeSMTP:
    """Records the call sequence. Instances land in FakeSMTP.instances."""

    instances: list["FakeSMTP"] = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.calls: list[str] = []
        self.credentials = None
        self.messages = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        self.calls.append("open")
        return self

    def __exit__(self, *exc):
        self.calls.append("close")
        return False

    def starttls(self):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append("login")
        self.credentials = (user, password)

    def send_message(self, msg):
        self.calls.append("send_message")
        self.messages.append(msg)


@pytest.fixture()
def fake_smtp(monkeypatch):
    FakeSMTP.instances = []
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    return FakeSMTP


def deliver(cfg, **kw):
    return SmtpEmailSink(cfg).deliver(event(**kw.pop("event_kw", {})), **kw)


# ------------------------------------------------------------------ the handshake

def test_the_connection_uses_the_configured_host_and_port(cfg, fake_smtp):
    deliver(cfg)
    s = fake_smtp.instances[0]
    assert (s.host, s.port) == ("smtp.example.com", 587)


def test_starttls_happens_before_login(cfg, fake_smtp):
    """The password must never cross an unencrypted connection."""
    calls = (deliver(cfg), fake_smtp.instances[0].calls)[1]
    assert calls.index("starttls") < calls.index("login")


def test_the_full_call_sequence(cfg, fake_smtp):
    deliver(cfg)
    assert fake_smtp.instances[0].calls == [
        "open", "starttls", "login", "send_message", "close"]


def test_the_connection_is_closed_even_though_send_succeeded(cfg, fake_smtp):
    deliver(cfg)
    assert fake_smtp.instances[0].calls[-1] == "close"


def test_login_uses_the_app_password_not_the_account_password(cfg, fake_smtp):
    deliver(cfg)
    assert fake_smtp.instances[0].credentials == ("guard@example.com", "app-password")


def test_a_timeout_is_set_so_a_hung_server_cannot_stall_the_pipeline(cfg, fake_smtp):
    deliver(cfg)
    assert fake_smtp.instances[0].timeout is not None


# ------------------------------------------------------------------ the message

def test_the_envelope_headers_are_set(cfg, fake_smtp):
    deliver(cfg)
    msg = fake_smtp.instances[0].messages[0]
    assert msg["From"] == "guard@example.com"
    assert msg["To"] == "duty@example.com"
    assert msg["Subject"]


def test_the_subject_carries_the_site_and_reads_as_english(cfg, fake_smtp):
    deliver(cfg)
    subject = fake_smtp.instances[0].messages[0]["Subject"]
    assert "site-118" in subject
    assert "_" not in subject, f"a raw state identifier leaked into the subject: {subject!r}"


def test_a_review_event_asks_for_a_person_in_the_subject(cfg, fake_smtp):
    deliver(cfg, event_kw={"needs_review": True})
    subject = fake_smtp.instances[0].messages[0]["Subject"]
    assert "check" in subject.lower()


def test_the_review_link_reaches_the_body(cfg, fake_smtp):
    link = "http://localhost:8080/review/e1"
    deliver(cfg, review_link=link)
    assert link in fake_smtp.instances[0].messages[0].get_content()


def test_the_reasons_reach_the_body_as_prose(cfg, fake_smtp):
    deliver(cfg, event_kw={"needs_review": True}, reasons=["low_confidence", "missing_site_id"])
    body = fake_smtp.instances[0].messages[0].get_content()
    assert "not a confident one" in body
    assert "no site is configured" in body
    assert "low_confidence" not in body


def test_the_body_is_not_empty(cfg, fake_smtp):
    deliver(cfg)
    assert fake_smtp.instances[0].messages[0].get_content().strip()


# ------------------------------------------------------------------ when it fails

def test_a_refused_send_is_retried_then_reported_not_raised(cfg, fake_smtp, monkeypatch):
    """A sink that raises would take the pipeline down with it. It must return a failure."""
    def explode(self, msg):
        self.calls.append("send_message")
        raise smtplib.SMTPRecipientsRefused({"duty@example.com": (550, b"no such user")})

    # setattr via monkeypatch, not a bare assignment: FakeSMTP is a module-level class, so a
    # bare assignment would outlive this test and silently break whatever ran after it.
    monkeypatch.setattr(fake_smtp, "send_message", explode)
    result = deliver(cfg)
    assert result.ok is False
    assert len(fake_smtp.instances) == cfg.retry_attempts  # a fresh connection per attempt


def test_an_auth_failure_reports_rather_than_raises(cfg, fake_smtp, monkeypatch):
    def bad_login(self, user, password):
        self.calls.append("login")
        raise smtplib.SMTPAuthenticationError(535, b"app password rejected")

    monkeypatch.setattr(fake_smtp, "login", bad_login)
    result = deliver(cfg)
    assert result.ok is False
    assert result.error


def test_nothing_is_sent_when_smtp_is_unconfigured(fake_smtp):
    result = SmtpEmailSink(Settings(site_id="site-118")).deliver(event())
    assert result.ok is False
    assert result.error == "smtp_not_configured"
    assert fake_smtp.instances == []  # no connection was even attempted
