"""Routing and delivery, including every way delivery can go wrong.

The promise being tested: an event that nobody could be told about is still a row somebody
can find, with the payload needed to send it by hand.
"""

from __future__ import annotations

import pytest

from relay.config import Settings
from relay.router import Router, build_router
from relay.schema import Category, Event, Priority
from relay.sinks.base import DeliveryResult, format_body, format_subject
from relay.store import Store


@pytest.fixture()
def cfg() -> Settings:
    return Settings(site_id="site-118", retry_attempts=2, retry_base_s=0.001, retry_cap_s=0.01,
                    review_link_base="http://localhost:8080")


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


def event(**kw) -> Event:
    base = dict(
        event_id="e1", source_file="data/sessions/s.mp4", category=Category.routine,
        priority=Priority.low, observed="post_manned", video_ts="00:00:01.00",
        site_id="site-118", summary="Guard seated at the reception desk.",
        confidence=0.91, needs_review=False,
    )
    base.update(kw)
    return Event(**base)


class RecordingSink:
    def __init__(self, name="test", ok=True):
        self.name = name
        self.ok = ok
        self.delivered: list[Event] = []
        self.links: list[str | None] = []

    def deliver(self, ev, *, review_link=None, reasons=None):
        self.delivered.append(ev)
        self.links.append(review_link)
        if self.ok:
            return DeliveryResult.success(self.name)
        return DeliveryResult.failure(self.name, "simulated failure")


# ---------------------------------------------------------------- what gets notified

def test_a_routine_event_is_stored_not_notified(cfg, store):
    """An alerting system that alerts on everything gets muted, and then it may as well not
    exist."""
    sink = RecordingSink()
    result = Router(cfg, store, primary=sink).route(event(), reasons=[])
    assert result.ok and result.status == "none"
    assert sink.delivered == []


def test_a_high_priority_event_is_notified(cfg, store):
    sink = RecordingSink()
    ev = event(category=Category.intrusion, priority=Priority.high)
    assert Router(cfg, store, primary=sink).route(ev, reasons=[]).ok
    assert len(sink.delivered) == 1


def test_anything_needing_review_is_notified_whatever_its_priority(cfg, store):
    """A low-priority event the system is unsure about still needs a person."""
    sink = RecordingSink()
    Router(cfg, store, primary=sink).route(event(needs_review=True), reasons=["low_confidence"])
    assert len(sink.delivered) == 1


def test_a_medium_event_is_stored_for_the_digest(cfg, store):
    sink = RecordingSink()
    ev = event(category=Category.loitering, priority=Priority.medium)
    assert Router(cfg, store, primary=sink).route(ev, reasons=[]).status == "none"
    assert sink.delivered == []


def test_the_review_link_points_at_the_event(cfg, store):
    sink = RecordingSink()
    Router(cfg, store, primary=sink).route(event(needs_review=True), reasons=["low_confidence"])
    assert sink.links[0] == "http://localhost:8080/review/e1"


# ---------------------------------------------------------------- failure handling

def test_a_failed_primary_falls_back(cfg, store):
    primary, fallback = RecordingSink("n8n", ok=False), RecordingSink("python", ok=True)
    result = Router(cfg, store, primary=primary, fallback=fallback).route(
        event(priority=Priority.high), reasons=[]
    )
    assert result.ok and result.channel == "python"
    assert len(primary.delivered) == 1 and len(fallback.delivered) == 1


def test_a_failed_primary_leaves_a_dead_letter(cfg, store):
    primary, fallback = RecordingSink("n8n", ok=False), RecordingSink("python", ok=True)
    Router(cfg, store, primary=primary, fallback=fallback).route(
        event(priority=Priority.high), reasons=[]
    )
    rows = store.list_dead_letters()
    assert len(rows) == 1
    assert rows[0]["stage"] == "n8n" and rows[0]["event_id"] == "e1"


def test_the_dead_letter_carries_enough_to_resend_by_hand(cfg, store):
    """A dead letter you cannot act on is a log line with extra steps."""
    Router(cfg, store, primary=RecordingSink("n8n", ok=False)).route(
        event(priority=Priority.high), reasons=["low_confidence"]
    )
    payload = store.list_dead_letters()[0]["payload_json"]
    assert "post_manned" in payload and "site-118" in payload and "low_confidence" in payload


def test_both_sinks_failing_still_records_everything(cfg, store):
    primary, fallback = RecordingSink("n8n", ok=False), RecordingSink("python", ok=False)
    result = Router(cfg, store, primary=primary, fallback=fallback).route(
        event(priority=Priority.high), reasons=[]
    )
    assert result.ok is False
    assert len(store.list_dead_letters()) == 2      # one per sink that failed


# ---------------------------------------------------------------- message formatting

def test_the_subject_leads_with_the_site_and_the_severity():
    s = format_subject(event(priority=Priority.high), needs_review=False)
    assert s.startswith("[site-118] HIGH:") and "post_manned" in s


def test_a_review_subject_says_so_and_carries_the_confidence():
    s = format_subject(event(confidence=0.42), needs_review=True)
    assert "REVIEW NEEDED" in s and "0.42" in s


def test_a_missing_site_id_is_visible_in_the_subject():
    assert "[unknown-site]" in format_subject(event(site_id=None), needs_review=False)


def test_the_body_explains_why_review_was_triggered():
    body = format_body(event(), reasons=["low_confidence", "missing_site_id"],
                       review_link="http://localhost:8080/review/e1")
    assert "low_confidence" in body and "missing_site_id" in body
    assert "http://localhost:8080/review/e1" in body
    assert "Guard seated at the reception desk." in body


def test_the_body_identifies_the_event_without_dumping_it():
    """The body used to end with a pretty-printed JSON dump of the whole event. It was
    removed: it tripled the length and pushed the review link below the fold on a phone,
    which is the one thing the reader has to act on. The id still has to be there, because
    it is how a reply or a support call refers to this specific alert."""
    body = format_body(event())
    assert "e1" in body
    assert '"event_id":' not in body


def test_the_body_stays_short_enough_to_read_on_a_phone():
    body = format_body(event(), reasons=["low_confidence", "missing_site_id"],
                       review_link="http://localhost:8080/review/e1")
    assert len(body.splitlines()) <= 12
    assert len(body) < 600


# ---------------------------------------------------------------- assembly

def test_n8n_is_primary_when_configured(cfg, store):
    r = build_router(cfg, store, use_n8n=True)
    assert r.primary.name == "n8n"


def test_no_n8n_promotes_the_direct_sink(cfg, store):
    """`--no-n8n` must leave a working notifier, not no notifier."""
    import dataclasses

    with_smtp = dataclasses.replace(cfg, smtp_user="a@b.c", smtp_app_password="x", alert_to="d@e.f")
    r = build_router(with_smtp, store, use_n8n=False)
    assert r.primary is not None and r.primary.name == "python"


def test_unconfigured_smtp_reports_a_config_state_not_a_delivery_failure(cfg, store):
    """A keyless clean checkout should not look broken."""
    from relay.sinks.email import SmtpEmailSink

    result = SmtpEmailSink(cfg).deliver(event())
    assert result.ok is False and result.error == "smtp_not_configured"
    assert result.status == "none"


def test_no_sink_configured_is_a_config_state_not_a_failure(cfg, store):
    """The keyless clean-checkout path is a reviewer's FIRST run. It must not print an error
    or fill the dead-letter table: a dead letter means "we tried and could not", not "you
    have not set this up yet"."""
    result = Router(cfg, store, primary=None, fallback=None).route(
        event(priority=Priority.high), reasons=[]
    )
    assert result.ok is True and result.status == "none"
    assert store.list_dead_letters() == []


def test_a_configured_sink_that_fails_still_dead_letters(cfg, store):
    """The other side of the same distinction."""
    Router(cfg, store, primary=RecordingSink("n8n", ok=False)).route(
        event(priority=Priority.high), reasons=[]
    )
    assert len(store.list_dead_letters()) == 1
