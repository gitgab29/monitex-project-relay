"""SQLite persistence. Python is the single writer; everything else reads through the API.

Three things this module is responsible for and the rest of the system trusts:

* **Idempotent writes.** `upsert_event` reports whether it created a row. That boolean is the
  whole re-run guarantee -- the pipeline routes a notification only when a row was created,
  so replaying a session cannot email anyone twice.
* **The runs ledger.** Every invocation, live or replay, gets a row with its config hash and
  its counters, so "what did this run actually do" is a query, not a guess.
* **The audit trail.** Every analysed frame becomes an `observations` row, so an event can be
  traced back to the frames that produced it.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ids import bucket_for, event_id as make_event_id
from .schema import Event

SCHEMA_VERSION = 1


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY, value TEXT NOT NULL
);

-- One row per capture or replay invocation. The runs ledger.
CREATE TABLE IF NOT EXISTS runs (
  run_id            TEXT PRIMARY KEY,
  session_id        TEXT NOT NULL,
  mode              TEXT NOT NULL,
  config_hash       TEXT NOT NULL,
  replay_of_run_id  TEXT,
  chaos             TEXT,
  started_at        TEXT NOT NULL,
  ended_at          TEXT,
  status            TEXT NOT NULL DEFAULT 'running',
  frames_decoded    INTEGER DEFAULT 0,
  frames_analyzed   INTEGER DEFAULT 0,
  frames_skipped_quiet INTEGER DEFAULT 0,
  vision_calls      INTEGER DEFAULT 0,
  vision_failures   INTEGER DEFAULT 0,
  events_inserted   INTEGER DEFAULT 0,
  events_skipped    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS runs_session_cfg ON runs(session_id, config_hash);

-- One row per recorded capture.
CREATE TABLE IF NOT EXISTS sessions (
  session_id  TEXT PRIMARY KEY,
  source      TEXT NOT NULL,
  file_path   TEXT NOT NULL,
  fps         REAL NOT NULL,
  width       INTEGER,
  height      INTEGER,
  site_id     TEXT,
  started_at  TEXT NOT NULL,
  ended_at    TEXT
);

-- Raw per-analysed-frame record. Audit trail; never shown to n8n.
CREATE TABLE IF NOT EXISTS observations (
  session_id      TEXT NOT NULL,
  frame_index     INTEGER NOT NULL,
  video_ts_ms     INTEGER NOT NULL,
  motion_score    REAL,
  mean_luma       REAL,
  person_count    INTEGER,
  post_count      INTEGER,
  approach_count  INTEGER,
  yolo_max_conf   REAL,
  detections_json TEXT,
  post_state      TEXT,
  approach_state  TEXT,
  created_at      TEXT NOT NULL,
  PRIMARY KEY (session_id, frame_index)
);

-- The brief's schema, plus provenance. One row per state transition.
CREATE TABLE IF NOT EXISTS events (
  event_id      TEXT PRIMARY KEY,
  source_type   TEXT NOT NULL DEFAULT 'video_event',
  source_file   TEXT NOT NULL,
  category      TEXT NOT NULL CHECK (category IN ('intrusion','vehicle','fire_smoke','loitering','routine','other')),
  priority      TEXT NOT NULL CHECK (priority IN ('low','medium','high')),
  observed      TEXT NOT NULL,
  video_ts      TEXT NOT NULL,
  site_id       TEXT,
  summary       TEXT NOT NULL,
  confidence    REAL NOT NULL,
  needs_review  INTEGER NOT NULL DEFAULT 0,
  session_id    TEXT NOT NULL,
  run_id        TEXT NOT NULL,
  track_key     TEXT NOT NULL,
  bucket        INTEGER NOT NULL,
  first_seen_ms INTEGER NOT NULL,
  last_seen_ms  INTEGER,
  frame_count   INTEGER,
  yolo_term     REAL,
  zone_term     REAL,
  luma_term     REAL,
  vision_conf   REAL,
  vision_agrees INTEGER,
  summary_source TEXT NOT NULL,
  review_reasons_json TEXT NOT NULL DEFAULT '[]',
  notify_status TEXT NOT NULL DEFAULT 'none',
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_session_track ON events(session_id, track_key, first_seen_ms);

CREATE TABLE IF NOT EXISTS review_queue (
  event_id     TEXT PRIMARY KEY REFERENCES events(event_id),
  reasons_json TEXT NOT NULL,
  queued_at    TEXT NOT NULL,
  resolution   TEXT CHECK (resolution IN ('confirmed','false_alarm','dismissed')),
  resolved_by  TEXT,
  notes        TEXT,
  resolved_at  TEXT,
  resolution_notified_at TEXT
);

CREATE TABLE IF NOT EXISTS dead_letter (
  dl_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id   TEXT,
  stage      TEXT NOT NULL,
  attempts   INTEGER NOT NULL,
  error      TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  first_failed_at TEXT NOT NULL,
  last_failed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vision_calls (
  call_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id   TEXT,
  run_id     TEXT,
  model      TEXT,
  attempt    INTEGER,
  purpose    TEXT,
  latency_ms INTEGER,
  ok         INTEGER,
  error      TEXT,
  raw_text   TEXT,
  created_at TEXT
);

CREATE VIEW IF NOT EXISTS v_events_by_category AS
  SELECT category, priority, COUNT(*) n, ROUND(AVG(confidence),2) avg_conf
  FROM events GROUP BY 1,2;

CREATE VIEW IF NOT EXISTS v_review_depth AS
  SELECT COUNT(*) FILTER (WHERE resolution IS NULL) open,
         COUNT(*) FILTER (WHERE resolution IS NOT NULL) closed
  FROM review_queue;

CREATE VIEW IF NOT EXISTS v_run_metrics AS
  SELECT run_id, mode, frames_decoded, frames_analyzed, frames_skipped_quiet,
         ROUND(1.0*frames_analyzed/MAX(frames_decoded,1),3) analyze_ratio,
         vision_calls, vision_failures, events_inserted, events_skipped
  FROM runs;
"""


class Store:
    """Thin, explicit SQLite wrapper. No ORM: the schema is the interface."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the pipeline decodes on the main thread and analyses on a
        # worker. Writes are still serialised -- only the analysis thread writes.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(DDL)
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ sessions

    def add_session(
        self, session_id: str, source: str, file_path: str, fps: float,
        width: int, height: int, site_id: str | None, started_at: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO sessions(session_id, source, file_path, fps, width, height, site_id, started_at) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(session_id) DO NOTHING",
            (session_id, source, str(file_path), fps, width, height, site_id, started_at or utcnow()),
        )
        self.conn.commit()

    def end_session(self, session_id: str) -> None:
        self.conn.execute("UPDATE sessions SET ended_at=? WHERE session_id=?", (utcnow(), session_id))
        self.conn.commit()

    def get_session(self, session_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()

    def list_sessions(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM events e WHERE e.session_id=s.session_id) n_events "
            "FROM sessions s ORDER BY started_at DESC"
        ).fetchall()

    # ------------------------------------------------------------------ runs ledger

    def start_run(
        self, run_id: str, session_id: str, mode: str, config_hash: str, chaos: str | None = None
    ) -> str | None:
        """Open a run. Returns the id of the first earlier run over the same session AND the
        same config, which is what makes this run a *replay of* that one rather than a fresh
        analysis. A different config hash means a different analysis, so it is not a replay.
        """
        prior = self.conn.execute(
            "SELECT run_id FROM runs WHERE session_id=? AND config_hash=? ORDER BY started_at LIMIT 1",
            (session_id, config_hash),
        ).fetchone()
        replay_of = prior["run_id"] if prior else None
        self.conn.execute(
            "INSERT INTO runs(run_id, session_id, mode, config_hash, replay_of_run_id, chaos, started_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (run_id, session_id, mode, config_hash, replay_of, chaos, utcnow()),
        )
        self.conn.commit()
        return replay_of

    def bump(self, run_id: str, field: str, n: int = 1) -> None:
        allowed = {
            "frames_decoded", "frames_analyzed", "frames_skipped_quiet",
            "vision_calls", "vision_failures", "events_inserted", "events_skipped",
        }
        if field not in allowed:
            raise ValueError(f"not a run counter: {field}")
        self.conn.execute(f"UPDATE runs SET {field} = {field} + ? WHERE run_id=?", (n, run_id))

    def end_run(self, run_id: str, status: str = "complete") -> None:
        self.conn.execute("UPDATE runs SET ended_at=?, status=? WHERE run_id=?", (utcnow(), status, run_id))
        self.conn.commit()

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM v_run_metrics WHERE run_id=?", (run_id,)).fetchone()

    # ------------------------------------------------------------------ observations

    def add_observation(
        self, obs: Any, post_state: str | None = None, approach_state: str | None = None
    ) -> bool:
        """Idempotent by (session_id, frame_index): a replay overwrites nothing and inserts
        nothing new, so the audit trail stays a faithful record of the first analysis."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO observations(session_id, frame_index, video_ts_ms, motion_score, "
            "mean_luma, person_count, post_count, approach_count, yolo_max_conf, detections_json, "
            "post_state, approach_state, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                obs.session_id, obs.frame_index, obs.video_ts_ms, obs.motion_score, obs.mean_luma,
                obs.person_count, obs.post_count, obs.approach_count, obs.yolo_max_conf,
                json.dumps([d.model_dump() for d in obs.detections]),
                post_state, approach_state, utcnow(),
            ),
        )
        return cur.rowcount > 0

    # ------------------------------------------------------------------ events

    def find_nearby_event(
        self, session_id: str, track_key: str, first_seen_ms: int, window_s: float
    ) -> str | None:
        """The drift guard.

        A live run and the replay of its recording can place the same transition an analysis
        frame apart -- MP4 compression nudges a YOLO confidence, a confirm-frame counter trips
        one frame later, and `first_seen_ms` moves just far enough to fall in the next bucket.
        A pure hash would then mint a second id for one real-world occurrence.

        So before hashing, look for an event with the same (session, track_key) that started
        within `window_s`. If one exists, that IS this occurrence: reuse its id.
        """
        if window_s <= 0:
            return None
        w = int(window_s * 1000)
        row = self.conn.execute(
            "SELECT event_id FROM events WHERE session_id=? AND track_key=? "
            "AND ABS(first_seen_ms - ?) <= ? ORDER BY ABS(first_seen_ms - ?) LIMIT 1",
            (session_id, track_key, first_seen_ms, w, first_seen_ms),
        ).fetchone()
        return row["event_id"] if row else None

    def resolve_event_id(
        self, session_id: str, track_key: str, first_seen_ms: int, bucket_s: int, window_s: float
    ) -> tuple[str, bool]:
        """The id this occurrence should carry, and whether it came from the drift guard.

        Exact-hash first so the common path stays purely deterministic and needs no database
        lookup to be correct; the window is only a safety net for live-to-replay drift.
        """
        bucket = bucket_for(first_seen_ms, bucket_s)
        eid = make_event_id(session_id, track_key, bucket)
        exists = self.conn.execute("SELECT 1 FROM events WHERE event_id=?", (eid,)).fetchone()
        if exists:
            return eid, False
        near = self.find_nearby_event(session_id, track_key, first_seen_ms, window_s)
        if near:
            return near, True
        return eid, False

    def upsert_event(self, event: Event, prov: dict[str, Any]) -> bool:
        """Insert the event, or update the existing row in place.

        Returns True only when a row was **created**. The pipeline routes notifications on
        that boolean alone, which is what stops a replay from paging anyone a second time.
        An update refreshes only how long the occurrence lasted -- never its identity, its
        classification or its summary, because those belong to the first observation of it.
        """
        now = utcnow()
        d = event.model_dump()
        # Ask before writing. sqlite3 reports rowcount 1 for an INSERT and for a DO UPDATE
        # alike, and comparing created_at to `now` afterwards is wrong whenever both writes
        # land in the same millisecond -- which a replay does constantly, and which would
        # make a replay claim it inserted and re-notify. Python is the only writer, so a
        # read immediately before the write is authoritative.
        existed = self.conn.execute(
            "SELECT 1 FROM events WHERE event_id=?", (d["event_id"],)
        ).fetchone() is not None
        self.conn.execute(
            """
            INSERT INTO events(
              event_id, source_type, source_file, category, priority, observed, video_ts,
              site_id, summary, confidence, needs_review,
              session_id, run_id, track_key, bucket, first_seen_ms, last_seen_ms, frame_count,
              yolo_term, zone_term, luma_term, vision_conf, vision_agrees, summary_source,
              review_reasons_json, notify_status, created_at, updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(event_id) DO UPDATE SET
              last_seen_ms = MAX(COALESCE(excluded.last_seen_ms, 0), COALESCE(events.last_seen_ms, 0)),
              frame_count  = MAX(COALESCE(excluded.frame_count, 0), COALESCE(events.frame_count, 0)),
              updated_at   = excluded.updated_at
            """,
            (
                d["event_id"], d["source_type"], d["source_file"], d["category"], d["priority"],
                d["observed"], d["video_ts"], d["site_id"], d["summary"], d["confidence"],
                int(d["needs_review"]),
                prov["session_id"], prov["run_id"], prov["track_key"], prov["bucket"],
                prov["first_seen_ms"], prov.get("last_seen_ms"), prov.get("frame_count"),
                prov.get("yolo_term"), prov.get("zone_term"), prov.get("luma_term"),
                prov.get("vision_conf"), prov.get("vision_agrees"),
                prov.get("summary_source", "template"),
                json.dumps(prov.get("review_reasons", [])),
                prov.get("notify_status", "none"), now, now,
            ),
        )
        self.conn.commit()
        return not existed

    def get_event(self, event_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()

    def list_events(self, limit: int = 50, needs_review: bool | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM events"
        params: list[Any] = []
        if needs_review is not None:
            q += " WHERE needs_review=?"
            params.append(int(needs_review))
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return self.conn.execute(q, params).fetchall()

    def set_notify_status(self, event_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE events SET notify_status=?, updated_at=? WHERE event_id=?",
            (status, utcnow(), event_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ review queue

    def queue_for_review(self, event_id: str, reasons: Iterable[str]) -> None:
        self.conn.execute(
            "INSERT INTO review_queue(event_id, reasons_json, queued_at) VALUES(?,?,?) "
            "ON CONFLICT(event_id) DO UPDATE SET reasons_json=excluded.reasons_json",
            (event_id, json.dumps(list(reasons)), utcnow()),
        )
        self.conn.commit()

    def pending_review(self, limit: int = 100) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT r.*, e.observed, e.category, e.priority, e.confidence, e.summary, e.video_ts, e.site_id "
            "FROM review_queue r JOIN events e USING(event_id) "
            "WHERE r.resolution IS NULL ORDER BY r.queued_at LIMIT ?",
            (limit,),
        ).fetchall()

    def resolve_review(
        self, event_id: str, resolution: str, by: str | None = None, notes: str | None = None
    ) -> bool:
        if resolution not in {"confirmed", "false_alarm", "dismissed"}:
            raise ValueError(f"resolution must be confirmed|false_alarm|dismissed, got {resolution!r}")
        cur = self.conn.execute(
            "UPDATE review_queue SET resolution=?, resolved_by=?, notes=?, resolved_at=? "
            "WHERE event_id=? AND resolution IS NULL",
            (resolution, by or "cli", notes, utcnow(), event_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def resolved_unnotified(self, limit: int = 50) -> list[sqlite3.Row]:
        """What the n8n sweep asks for once a minute: resolutions a human has set that nobody
        has been told about yet. `resolution_notified_at` is the idempotency key for the
        sweep -- it is why a slow email cannot produce two 'review closed' messages."""
        return self.conn.execute(
            "SELECT r.event_id, r.resolution, r.notes, r.resolved_by, r.resolved_at, "
            "e.observed, e.category, e.priority, e.video_ts, e.site_id, e.summary, e.confidence "
            "FROM review_queue r JOIN events e USING(event_id) "
            "WHERE r.resolution IS NOT NULL AND r.resolution_notified_at IS NULL "
            "ORDER BY r.resolved_at LIMIT ?",
            (limit,),
        ).fetchall()

    def ack_resolution_notified(self, event_id: str) -> bool:
        cur = self.conn.execute(
            "UPDATE review_queue SET resolution_notified_at=? "
            "WHERE event_id=? AND resolution_notified_at IS NULL",
            (utcnow(), event_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------ dead letter

    def dead_letter(
        self, stage: str, error: str, payload: Any, attempts: int, event_id: str | None = None
    ) -> int:
        now = utcnow()
        cur = self.conn.execute(
            "INSERT INTO dead_letter(event_id, stage, attempts, error, payload_json, "
            "first_failed_at, last_failed_at) VALUES(?,?,?,?,?,?,?)",
            (event_id, stage, attempts, str(error)[:2000],
             json.dumps(payload, default=str)[:8000], now, now),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def list_dead_letters(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM dead_letter ORDER BY last_failed_at DESC LIMIT ?", (limit,)
        ).fetchall()

    # ------------------------------------------------------------------ vision call metrics

    def log_vision_call(
        self, *, event_id: str | None, run_id: str | None, model: str, attempt: int,
        purpose: str, latency_ms: int, ok: bool, error: str | None = None,
        raw_text: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO vision_calls(event_id, run_id, model, attempt, purpose, latency_ms, ok, "
            "error, raw_text, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (event_id, run_id, model, attempt, purpose, latency_ms, int(ok),
             (str(error)[:1000] if error else None),
             (str(raw_text)[:4000] if raw_text else None), utcnow()),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ summary

    def summary(self, since: str | None = None) -> dict[str, Any]:
        """Everything `relay summary`, GET /summary and the digest all render from. One
        source of truth, three thin renderers."""
        where, params = ("WHERE created_at >= ?", [since]) if since else ("", [])
        by_cat = [
            dict(r) for r in self.conn.execute(
                f"SELECT category, priority, COUNT(*) n, ROUND(AVG(confidence),2) avg_conf "
                f"FROM events {where} GROUP BY 1,2 ORDER BY n DESC", params
            ).fetchall()
        ]
        totals = dict(self.conn.execute(
            f"SELECT COUNT(*) events, "
            f"COALESCE(SUM(needs_review),0) needs_review, "
            f"COALESCE(SUM(priority='high'),0) high "
            f"FROM events {where}", params
        ).fetchone())
        review = dict(self.conn.execute("SELECT * FROM v_review_depth").fetchone())
        # Ordered off the base table, not the view: a view has no rowid, and
        # `CREATE VIEW IF NOT EXISTS` would never update a view already in an existing db.
        runs = [
            dict(r) for r in self.conn.execute(
                "SELECT run_id, mode, started_at, frames_decoded, frames_analyzed, frames_skipped_quiet, "
                "ROUND(1.0*frames_analyzed/MAX(frames_decoded,1),3) analyze_ratio, "
                "vision_calls, vision_failures, events_inserted, events_skipped "
                "FROM runs ORDER BY started_at DESC LIMIT 10"
            ).fetchall()
        ]
        frames = dict(self.conn.execute(
            "SELECT COALESCE(SUM(frames_decoded),0) decoded, COALESCE(SUM(frames_analyzed),0) analyzed, "
            "COALESCE(SUM(frames_skipped_quiet),0) skipped_quiet, COALESCE(SUM(vision_calls),0) vision_calls, "
            "COALESCE(SUM(vision_failures),0) vision_failures, COALESCE(SUM(events_inserted),0) inserted, "
            "COALESCE(SUM(events_skipped),0) skipped FROM runs"
        ).fetchone())
        dl = self.conn.execute("SELECT COUNT(*) n FROM dead_letter").fetchone()["n"]
        return {
            "totals": totals, "by_category": by_cat, "review": review,
            "frames": frames, "dead_letters": dl, "recent_runs": runs,
        }
