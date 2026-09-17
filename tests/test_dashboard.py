"""The dashboard's two dangerous bits: wiping everything, and spawning a process.

Both are one click away from a demo, so they are tested rather than eyeballed.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from relay.api import create_app
from relay.config import Settings
from relay.dashboard import CameraProcess
from relay.schema import Category, Event, Priority
from relay.store import Store


@pytest.fixture()
def cfg(tmp_path) -> Settings:
    return Settings(site_id="site-118", data_dir=tmp_path, db_path=tmp_path / "t.db",
                    review_link_base="http://localhost:8080")


@pytest.fixture()
def store(cfg):
    s = Store(cfg.db_path)
    yield s
    s.close()


@pytest.fixture()
def client(cfg, store):
    cfg.ensure_dirs()
    return TestClient(create_app(cfg, store))


def seed(store, n=3):
    store.add_session("sess-1", "webcam", "x.mp4", 30.0, 1280, 720, "site-118")
    for i in range(n):
        ev = Event(event_id=f"evt{i:013d}", source_file="x.mp4", category=Category.intrusion,
                   priority=Priority.high, observed="unidentified_person_at_post",
                   video_ts="00:00:01.00", site_id="site-118", summary="seeded",
                   confidence=0.5, needs_review=True)
        store.upsert_event(ev, {"session_id": "sess-1", "run_id": "r",
                                "track_key": ev.observed, "bucket": i, "first_seen_ms": i * 1000,
                                "last_seen_ms": i * 1000, "frame_count": 1, "yolo_term": .9,
                                "zone_term": .9, "luma_term": .9, "vision_conf": None,
                                "vision_agrees": None, "summary_source": "template",
                                "review_reasons": ["low_confidence"]})
        store.queue_for_review(ev.event_id, ["low_confidence"])
    store.dead_letter(event_id="evt0000000000000", stage="email", attempts=3,
                      error="SMTP 535", payload={})


# ------------------------------------------------------------------ wipe

def test_wipe_empties_every_operational_table(store):
    seed(store)
    assert store.summary()["totals"]["events"] == 3
    removed = store.wipe_all()
    assert removed["events"] == 3
    assert removed["review_queue"] == 3
    assert removed["dead_letter"] == 1
    assert store.summary()["totals"]["events"] == 0


def test_wipe_keeps_the_schema_usable(store):
    """Deleting rows, not the file: the store must still work straight afterwards, because
    the API is serving throughout and the next run must be able to write."""
    seed(store)
    store.wipe_all()
    seed(store, n=1)
    assert store.summary()["totals"]["events"] == 1


def test_wipe_is_safe_on_an_already_empty_database(store):
    assert store.wipe_all()["events"] == 0
    assert store.wipe_all()["events"] == 0


def test_wipe_does_not_drop_meta(store):
    """meta carries the schema version. An empty database must not look like an old one."""
    seed(store)
    store.wipe_all()
    assert store.conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0] > 0


def test_wipe_endpoint_also_removes_evidence_frames(client, cfg, store):
    seed(store)
    (cfg.evidence_dir / "evt0000000000000.jpg").write_bytes(b"not-really-a-jpeg")
    r = client.post("/control/wipe")
    assert r.status_code == 200
    assert r.json()["removed"]["evidence_frames"] == 1
    assert list(cfg.evidence_dir.glob("*.jpg")) == []


def test_wipe_refuses_while_the_camera_is_running(client, monkeypatch):
    """Wiping out from under a live pipeline leaves it writing into tables that vanished.
    A button that says no is far easier to explain than that half-state."""
    monkeypatch.setattr(CameraProcess, "running", property(lambda self: True))
    r = client.post("/control/wipe")
    assert r.status_code == 409
    assert "camera" in r.text.lower()


# ------------------------------------------------------------------ camera control

def test_camera_refuses_to_start_twice(cfg, monkeypatch):
    cam = CameraProcess(cfg)
    monkeypatch.setattr(CameraProcess, "running", property(lambda self: True))
    with pytest.raises(HTTPException) as e:
        cam.start()
    assert e.value.status_code == 409


def test_stopping_a_camera_that_is_not_running_is_an_error_not_a_crash(cfg):
    with pytest.raises(HTTPException) as e:
        CameraProcess(cfg).stop()
    assert e.value.status_code == 409


def test_status_of_an_idle_camera(cfg):
    assert CameraProcess(cfg).status() == {"running": False, "pid": None, "uptime_s": None}


# ------------------------------------------------------------------ the feed

def test_feed_returns_everything_the_page_needs_in_one_call(client, store):
    seed(store)
    d = client.get("/api/feed").json()
    for key in ("summary", "events", "review", "dead_letters", "camera", "site_id", "outbox"):
        assert key in d, f"the page reads {key} and it was missing"


def test_feed_marks_which_events_have_an_evidence_frame(client, cfg, store):
    seed(store)
    (cfg.evidence_dir / "evt0000000000000.jpg").write_bytes(b"x")
    events = {e["event_id"]: e for e in client.get("/api/feed").json()["events"]}
    assert events["evt0000000000000"]["has_evidence"] is True
    assert events["evt0000000000001"]["has_evidence"] is False


def test_evidence_route_rejects_a_traversal_attempt(client):
    assert client.get("/evidence/..%2F..%2Fsecret.jpg").status_code in (400, 404)


def test_evidence_route_404s_when_there_is_no_frame(client, store):
    seed(store)
    assert client.get("/evidence/evt0000000000000.jpg").status_code == 404


def test_email_preview_uses_the_real_formatters(client, store):
    seed(store)
    m = client.get("/api/events/evt0000000000000/email").json()
    assert "REVIEW NEEDED" in m["subject"]
    assert "low_confidence" in m["body"]
    assert "/review/evt0000000000000" in m["body"]


def test_email_preview_404s_for_an_unknown_event(client):
    assert client.get("/api/events/nope/email").status_code == 404
