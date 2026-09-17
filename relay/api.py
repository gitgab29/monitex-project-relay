"""The local HTTP API.

This exists for one structural reason: **Python is the only writer to SQLite.**

n8n has no SQLite node, its Code node cannot require a driver, and bind-mounting a .db file
into a container so two processes can write it is a well-known corruption path. So n8n never
touches the database; it asks this service over HTTP, and Python serialises every write.

It pays for itself three times over: it is the n8n callback surface, it serves the review link
that goes in the alert email, and it renders the summary.

Bound to 0.0.0.0 so a container can reach it on host.docker.internal. When n8n runs natively
(`npx n8n`) it is plain localhost -- one env var, API_BASE.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings
from .store import Store

log = logging.getLogger(__name__)


class NotifiedIn(BaseModel):
    channel: str = "email"
    ok: bool = True
    error: str | None = None


class DeadLetterIn(BaseModel):
    event_id: str | None = None
    stage: str = "email"
    error: str = "unknown"
    payload: dict[str, Any] = Field(default_factory=dict)
    attempts: int = 1


class ResolveIn(BaseModel):
    resolution: str
    notes: str | None = None
    by: str | None = "api"


def _rows(rows) -> list[dict]:
    return [dict(r) for r in rows]


def create_app(cfg: Settings, store: Store) -> FastAPI:
    app = FastAPI(
        title="Project Relay", version="0.1.0",
        description="Local control surface. n8n reads and acknowledges through this; "
                    "Python remains the only writer to SQLite.",
    )

    # ------------------------------------------------------------------ health

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "site_id": cfg.site_id,
            "config_hash": cfg.config_hash(),
            "vision_backend": cfg.vision_backend,
            "db": str(cfg.db_path),
        }

    # ------------------------------------------------------------------ events

    @app.get("/events")
    def list_events(limit: int = 50, needs_review: bool | None = None) -> dict:
        return {"events": _rows(store.list_events(limit=limit, needs_review=needs_review))}

    @app.get("/events/{event_id}")
    def get_event(event_id: str) -> dict:
        row = store.get_event(event_id)
        if row is None:
            raise HTTPException(404, f"no such event: {event_id}")
        return dict(row)

    @app.post("/events/{event_id}/notified")
    def mark_notified(event_id: str, body: NotifiedIn) -> dict:
        """Called by the n8n router once the email has actually been sent.

        This is what turns 'accepted_by_n8n' into 'emailed:n8n' -- the difference between n8n
        having taken the job and n8n having done it.
        """
        if store.get_event(event_id) is None:
            raise HTTPException(404, f"no such event: {event_id}")
        status = f"emailed:{body.channel}" if body.ok else "failed"
        store.set_notify_status(event_id, status)
        log.info("n8n reports %s -> %s", event_id, status)
        return {"ok": True, "event_id": event_id, "notify_status": status}

    @app.post("/dead-letter")
    def post_dead_letter(body: DeadLetterIn) -> dict:
        """The n8n error branch posts here, so a failure inside the workflow lands in the same
        table as a failure inside Python. One place to look."""
        dl_id = store.dead_letter(
            stage=body.stage, error=body.error, payload=body.payload,
            attempts=body.attempts, event_id=body.event_id,
        )
        log.warning("dead letter %d from n8n: stage=%s error=%s", dl_id, body.stage, body.error)
        return {"ok": True, "dl_id": dl_id}

    @app.get("/dead-letters")
    def list_dead_letters(limit: int = 50) -> dict:
        return {"dead_letters": _rows(store.list_dead_letters(limit))}

    # ------------------------------------------------------------------ review

    @app.get("/review/pending")
    def review_pending(limit: int = 100) -> dict:
        return {"pending": _rows(store.pending_review(limit))}

    @app.get("/review/resolved")
    def review_resolved(unnotified: bool = True, limit: int = 50) -> dict:
        """What the n8n sweep polls once a minute.

        `unnotified=true` is the whole trick: the sweep only ever sees resolutions nobody has
        been told about, so a slow email cannot produce two 'review closed' messages.
        """
        rows = store.resolved_unnotified(limit) if unnotified else []
        return {"resolved": _rows(rows)}

    @app.post("/review/{event_id}/resolve")
    def resolve(event_id: str, body: ResolveIn) -> dict:
        try:
            ok = store.resolve_review(event_id, body.resolution, by=body.by, notes=body.notes)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if not ok:
            raise HTTPException(409, "not in the review queue, or already resolved")
        return {"ok": True, "event_id": event_id, "resolution": body.resolution}

    @app.post("/review/{event_id}/ack-notified")
    def ack_notified(event_id: str) -> dict:
        """The sweep's acknowledgement. Idempotent: a second ack reports ok=False rather than
        erroring, because a retried n8n execution should not be a failure."""
        return {"ok": store.ack_resolution_notified(event_id), "event_id": event_id}

    # ------------------------------------------------------------------ summary

    @app.get("/summary")
    def summary(since: str | None = None, format: str = "json"):
        data = store.summary(since)
        if format == "html":
            return HTMLResponse(_summary_html(data, cfg))
        return data

    # The dashboard: evidence frames, camera control, and the single page over all of it.
    from .dashboard import build_router

    app.include_router(build_router(cfg, store))

    return app


def _summary_html(data: dict, cfg: Settings) -> str:
    """The stretch-goal dashboard, kept deliberately thin.

    The SQL views are the single source of truth; this is one of three renderers over the same
    `store.summary()` payload (the others being the CLI table and the JSON the digest reads).
    Adding a renderer should never mean re-deriving a number.
    """
    t, f, r = data["totals"], data["frames"], data["review"]
    rows = "".join(
        f"<tr><td>{c['category']}</td><td>{c['priority']}</td>"
        f"<td class=n>{c['n']}</td><td class=n>{c['avg_conf']}</td></tr>"
        for c in data["by_category"]
    ) or "<tr><td colspan=4 class=empty>no events yet</td></tr>"
    pct = 100 * f["analyzed"] / max(f["decoded"], 1)
    warn_review = "warn" if t["needs_review"] else ""
    warn_dl = "warn" if data["dead_letters"] else ""
    return f"""<!doctype html><meta charset=utf-8><title>Relay summary</title>
<style>
 body{{font:14px/1.55 ui-monospace,Menlo,Consolas,monospace;margin:2rem auto;max-width:46rem;
      padding:0 1rem;color:#e8e8e8;background:#16171a}}
 h1{{font-size:1.05rem;letter-spacing:.02em}}
 h2{{font-size:.78rem;color:#9aa0a6;margin-top:2rem;text-transform:uppercase;letter-spacing:.09em}}
 table{{border-collapse:collapse;width:100%}}
 td,th{{padding:.35rem .6rem;border-bottom:1px solid #2a2c31;text-align:left}}
 th{{color:#9aa0a6;font-weight:500}} .n{{text-align:right}}
 .big{{font-size:1.7rem;line-height:1.1}} .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-top:1.2rem}}
 .grid div div{{color:#9aa0a6;font-size:.78rem}} .empty{{color:#6b7076}} .warn{{color:#f0a45d}}
</style>
<h1>Project Relay &mdash; {cfg.site_id or "site not configured"}</h1>
<div class=grid>
 <div><span class=big>{t["events"]}</span><div>events</div></div>
 <div><span class="big {warn_review}">{t["needs_review"]}</span><div>need review</div></div>
 <div><span class=big>{t["high"]}</span><div>high priority</div></div>
 <div><span class="big {warn_dl}">{data["dead_letters"]}</span><div>dead letters</div></div>
</div>
<h2>By category</h2>
<table><tr><th>category</th><th>priority</th><th class=n>n</th><th class=n>avg conf</th></tr>{rows}</table>
<h2>Review queue</h2>
<table><tr><td>open</td><td class=n>{r["open"]}</td></tr>
<tr><td>closed</td><td class=n>{r["closed"]}</td></tr></table>
<h2>Frames</h2>
<table>
 <tr><td>decoded</td><td class=n>{f["decoded"]}</td></tr>
 <tr><td>analysed</td><td class=n>{f["analyzed"]} ({pct:.1f}%)</td></tr>
 <tr><td>skipped as quiet</td><td class=n>{f["skipped_quiet"]}</td></tr>
 <tr><td>model calls</td><td class=n>{f["vision_calls"]} ({f["vision_failures"]} failed)</td></tr>
 <tr><td>events inserted / skipped</td><td class=n>{f["inserted"]} / {f["skipped"]}</td></tr>
</table>"""
