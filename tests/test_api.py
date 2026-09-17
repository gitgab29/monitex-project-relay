"""The HTTP surface n8n talks to.

The API exists because Python must remain the only writer to SQLite -- n8n has no SQLite node
and sharing the file between processes is a corruption path. So these tests are really about
one thing: can an external workflow read what it needs and acknowledge what it did, without
ever being able to corrupt the database or double-notify a human.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from relay.api import create_app
from relay.config import Settings
from relay.schema import Category, Event, Priority
from relay.store import Store


@pytest.fixture()
def cfg() -> Settings:
    return Settings(site_id="site-118")


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "api.db")
    yield s
    s.close()


@pytest.fixture()
def client(cfg, store):
    return TestClient(create_app(cfg, store))


def seed(store, event_id="e1", needs_review=True, priority=Priority.high):
    ev = Event(
        event_id=event_id, source_file="data/sessions/s.mp4", category=Category.intrusion,
        priority=priority, observed="unidentified_person_at_post", video_ts="00:00:05.00",
        site_id="site-118", summary="Two people at the desk.", confidence=0.62,
        needs_review=needs_review,
    )
    store.upsert_event(ev, {
        "session_id": "s", "run_id": "r", "track_key": "unidentified_person_at_post",
        "bucket": 1, "first_seen_ms": 5000, "last_seen_ms": 6000, "frame_count": 2,
        "summary_source": "template", "review_reasons": ["low_confidence"],
    })
    if needs_review:
        store.queue_for_review(event_id, ["low_confidence"])
    return event_id


# ---------------------------------------------------------------- health

def test_health_reports_the_config_hash(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["site_id"] == "site-118"
    assert len(body["config_hash"]) == 16


def test_health_never_leaks_a_secret(client):
    """This endpoint is reachable from any container on the host."""
    text = client.get("/health").text.lower()
    assert "api_key" not in text and "password" not in text


# ---------------------------------------------------------------- events

def test_events_are_listable_and_filterable(client, store):
    seed(store, "e1", needs_review=True)
    seed(store, "e2", needs_review=False)
    assert len(client.get("/events").json()["events"]) == 2
    flagged = client.get("/events?needs_review=true").json()["events"]
    assert [e["event_id"] for e in flagged] == ["e1"]


def test_a_missing_event_is_a_404(client):
    assert client.get("/events/nope").status_code == 404


def test_notified_turns_accepted_into_sent(client, store):
    """The difference between n8n having taken the job and n8n having done it."""
    seed(store, "e1")
    r = client.post("/events/e1/notified", json={"channel": "n8n", "ok": True})
    assert r.json()["notify_status"] == "emailed:n8n"
    assert store.get_event("e1")["notify_status"] == "emailed:n8n"


def test_a_failed_notification_is_recorded_as_failed(client, store):
    seed(store, "e1")
    client.post("/events/e1/notified", json={"channel": "n8n", "ok": False, "error": "SMTP 535"})
    assert store.get_event("e1")["notify_status"] == "failed"


def test_notifying_an_unknown_event_is_a_404(client):
    assert client.post("/events/nope/notified", json={"channel": "n8n", "ok": True}).status_code == 404


# ---------------------------------------------------------------- review

def test_pending_review_is_visible(client, store):
    seed(store, "e1")
    pending = client.get("/review/pending").json()["pending"]
    assert len(pending) == 1 and pending[0]["event_id"] == "e1"


def test_resolving_closes_the_item(client, store):
    seed(store, "e1")
    assert client.post("/review/e1/resolve", json={"resolution": "false_alarm"}).status_code == 200
    assert client.get("/review/pending").json()["pending"] == []


def test_a_second_resolution_is_refused(client, store):
    """A human decision is not something a retried workflow should be able to overwrite."""
    seed(store, "e1")
    client.post("/review/e1/resolve", json={"resolution": "confirmed"})
    assert client.post("/review/e1/resolve", json={"resolution": "dismissed"}).status_code == 409


def test_an_invalid_resolution_is_rejected(client, store):
    seed(store, "e1")
    assert client.post("/review/e1/resolve", json={"resolution": "probably_fine"}).status_code == 400


def test_the_sweep_sees_a_resolution_exactly_once(client, store):
    """The whole point of `unnotified`: a slow email must not produce two 'review closed'
    messages for one human decision."""
    seed(store, "e1")
    client.post("/review/e1/resolve", json={"resolution": "confirmed"})
    assert len(client.get("/review/resolved?unnotified=true").json()["resolved"]) == 1
    assert client.post("/review/e1/ack-notified").json()["ok"] is True
    assert client.get("/review/resolved?unnotified=true").json()["resolved"] == []


def test_a_repeated_ack_is_not_an_error(client, store):
    """A retried n8n execution should be harmless, not a failure."""
    seed(store, "e1")
    client.post("/review/e1/resolve", json={"resolution": "confirmed"})
    client.post("/review/e1/ack-notified")
    r = client.post("/review/e1/ack-notified")
    assert r.status_code == 200 and r.json()["ok"] is False


# ---------------------------------------------------------------- dead letters

def test_n8n_can_report_its_own_failures(client, store):
    """A failure inside the workflow lands in the same table as a failure inside Python.
    One place to look."""
    r = client.post("/dead-letter", json={
        "event_id": "e1", "stage": "email", "error": "SMTP 535", "attempts": 3,
        "payload": {"to": "someone@example.com"},
    })
    assert r.json()["ok"] is True
    rows = store.list_dead_letters()
    assert len(rows) == 1 and rows[0]["stage"] == "email" and rows[0]["attempts"] == 3


# ---------------------------------------------------------------- summary

def test_summary_json_is_stable_when_empty(client):
    body = client.get("/summary").json()
    assert body["totals"]["events"] == 0
    assert body["review"] == {"open": 0, "closed": 0}


def test_summary_counts_events(client, store):
    seed(store, "e1")
    body = client.get("/summary").json()
    assert body["totals"]["events"] == 1 and body["totals"]["high"] == 1


def test_summary_renders_html(client, store):
    seed(store, "e1")
    r = client.get("/summary?format=html")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "site-118" in r.text and "need review" in r.text
