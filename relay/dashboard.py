"""Monitex Demo -- the single page a human actually looks at.

Everything here is a *renderer over the existing API*. It adds exactly two capabilities the
HTTP surface did not already have:

1. **Evidence.** One JPEG per event, served at `/evidence/<event_id>.jpg`. A summary saying
   "a person is at the post" is a claim; the frame is what makes it checkable.
2. **Process control.** Start and stop the live camera from the browser, so a demo does not
   need a second terminal.

The control half is deliberately narrow. It runs exactly one child process, it refuses to
start a second, and it only ever launches this package's own CLI with a fixed argument list --
nothing from the request reaches a shell. A web endpoint that spawns processes is worth being
boring about.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from .config import Settings
from .schema import Event
from .sinks.base import format_body, format_subject
from .store import Store

log = logging.getLogger(__name__)


class CameraProcess:
    """One child process, or none. Guarded by a lock because two browser tabs both pressing
    Start is the normal case, not the exotic one."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self._proc: subprocess.Popen | None = None
        self._started_at: float | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        return {
            "running": self.running,
            "pid": self._proc.pid if self.running else None,
            "uptime_s": round(time.time() - self._started_at, 1)
            if (self.running and self._started_at) else None,
        }

    def start(self, *, show: bool = True) -> dict:
        with self._lock:
            if self.running:
                raise HTTPException(409, "the camera is already running")
            # Fixed argv. Nothing from the HTTP request is interpolated, and there is no shell.
            argv = [sys.executable, "-m", "relay.cli", "run", "--source", "webcam"]
            if show:
                argv.append("--show")
            log.info("dashboard starting the camera: %s", " ".join(argv))
            creation = 0
            if sys.platform == "win32":
                # Its own console, so the overlay window and its log are visible on screen and
                # a Ctrl-C in the server does not take the camera down with it.
                creation = subprocess.CREATE_NEW_CONSOLE
            self._proc = subprocess.Popen(argv, cwd=str(Path.cwd()), creationflags=creation)
            self._started_at = time.time()
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            if not self.running:
                raise HTTPException(409, "the camera is not running")
            log.info("dashboard stopping the camera (pid %s)", self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # It gets ten seconds to close the recording cleanly. After that the file
                # matters less than not leaving an orphan behind.
                log.warning("camera did not exit in 10s; killing it")
                self._proc.kill()
            out = self.status()
            self._proc, self._started_at = None, None
            return out


def _reasons(raw) -> list[str]:
    """review_reasons_json is a JSON list on the event row. Never let a malformed one take
    the whole page down -- an unreadable reason list is a cosmetic problem, not an outage."""
    if not raw:
        return []
    try:
        v = json.loads(raw)
        return [str(x) for x in v] if isinstance(v, list) else []
    except Exception:
        return []


def _event_row(r, cfg: Settings) -> dict:
    d = dict(r)
    d["has_evidence"] = (cfg.evidence_dir / f"{d['event_id']}.jpg").exists()
    d["reasons"] = _reasons(d.get("review_reasons_json"))
    return d


def build_router(cfg: Settings, store: Store) -> APIRouter:
    router = APIRouter()
    camera = CameraProcess(cfg)

    # ------------------------------------------------------------------ evidence

    @router.get("/evidence/{event_id}.jpg", include_in_schema=False)
    def evidence(event_id: str):
        # Reject anything that is not a plain id, so no request can walk out of the directory.
        if not event_id.isalnum():
            raise HTTPException(400, "bad event id")
        path = cfg.evidence_dir / f"{event_id}.jpg"
        if not path.exists():
            raise HTTPException(404, "no evidence frame for this event")
        return FileResponse(path, media_type="image/jpeg")

    # ------------------------------------------------------------------ control

    @router.get("/control/status")
    def control_status() -> dict:
        return camera.status()

    @router.post("/control/start")
    def control_start(show: bool = True) -> dict:
        return camera.start(show=show)

    @router.post("/control/stop")
    def control_stop() -> dict:
        return camera.stop()

    # ------------------------------------------------------------------ data for the page

    @router.get("/api/feed")
    def feed(limit: int = 30) -> dict:
        """One call, everything the page needs. The page polls this; a page that polls five
        endpoints can render five different moments in time at once."""
        rows = [_event_row(r, cfg) for r in store.list_events(limit)]
        review = []
        for r in store.pending_review(50):
            d = dict(r)
            d["reasons"] = _reasons(d.get("reasons_json"))
            review.append(d)
        return {
            "summary": store.summary(),
            "events": rows,
            "review": review,
            "dead_letters": [dict(r) for r in store.list_dead_letters(20)],
            "camera": camera.status(),
            "site_id": cfg.site_id,
        }

    @router.get("/api/events/{event_id}/email")
    def event_email(event_id: str) -> dict:
        """What was (or would be) sent for this event, rendered by the SAME functions the
        sinks use -- so this is the email, not a mock-up of it."""
        row = store.get_event(event_id)
        if row is None:
            raise HTTPException(404, "no such event")
        d = dict(row)
        ev = Event(**{k: d[k] for k in Event.model_fields if k in d})
        reasons = _reasons(d.get("review_reasons_json"))
        link = f"{cfg.review_link_base}/review/{event_id}"
        return {
            "to": cfg.alert_to or "(ALERT_TO not set)",
            "subject": format_subject(ev, needs_review=ev.needs_review),
            "body": format_body(ev, reasons=reasons, review_link=link),
            "notify_status": d.get("notify_status"),
        }

    @router.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse(PAGE)

    return router


PAGE = r"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Monitex Demo</title>
<style>
:root{
  --bg:#0f1012; --panel:#17191d; --line:#262a31; --ink:#e9eaec; --dim:#8b929c;
  --hi:#f0a45d; --ok:#5cc98c; --bad:#e2686b; --acc:#6ea8fe;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;
       padding:1rem 1.25rem;border-bottom:1px solid var(--line);background:var(--panel);
       position:sticky;top:0;z-index:10}
h1{font-size:1rem;margin:0;letter-spacing:.02em}
h1 small{color:var(--dim);font-weight:400;margin-left:.5rem}
.spacer{flex:1}
button{font:inherit;padding:.5rem .9rem;border-radius:7px;border:1px solid var(--line);
       background:#20242a;color:var(--ink);cursor:pointer}
button:hover:not(:disabled){border-color:var(--acc)}
button:disabled{opacity:.4;cursor:not-allowed}
button.go{background:var(--ok);color:#06240f;border-color:transparent;font-weight:600}
button.stop{background:var(--bad);color:#2a0709;border-color:transparent;font-weight:600}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:.45rem}
.live{background:var(--ok);box-shadow:0 0 0 4px rgba(92,201,140,.18)}
.off{background:#4b525c}
main{padding:1.25rem;max-width:1180px;margin:0 auto}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.8rem}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem}
.tile b{display:block;font-size:1.9rem;line-height:1.15;font-weight:600}
.tile span{color:var(--dim);font-size:.75rem;text-transform:uppercase;letter-spacing:.07em}
.warn b{color:var(--hi)} .bad b{color:var(--bad)}
h2{font-size:.76rem;color:var(--dim);text-transform:uppercase;letter-spacing:.1em;
   margin:2rem 0 .75rem}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:.9rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;
      display:flex;flex-direction:column}
.card img{width:100%;aspect-ratio:16/9;object-fit:cover;background:#0a0b0d;display:block;
          cursor:zoom-in}
.noimg{width:100%;aspect-ratio:16/9;display:grid;place-items:center;color:#4b525c;
       background:#0a0b0d;font-size:.78rem}
.body{padding:.8rem .9rem}
.obs{font-weight:600;margin-bottom:.15rem}
.meta{color:var(--dim);font-size:.76rem;margin-bottom:.5rem}
.sum{font-size:.85rem;margin-bottom:.6rem}
.tags{display:flex;flex-wrap:wrap;gap:.3rem}
.tag{font-size:.68rem;padding:.16rem .45rem;border-radius:4px;background:#22262c;color:var(--dim)}
.tag.p-high{background:rgba(226,104,107,.16);color:#ff9a9c}
.tag.p-medium{background:rgba(240,164,93,.16);color:var(--hi)}
.tag.p-low{background:rgba(110,168,254,.14);color:var(--acc)}
.tag.rev{background:rgba(240,164,93,.14);color:var(--hi)}
.act{margin-top:.6rem;display:flex;gap:.4rem;flex-wrap:wrap}
.act button{padding:.3rem .6rem;font-size:.76rem}
table{width:100%;border-collapse:collapse;background:var(--panel);
      border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:.55rem .8rem;text-align:left;border-bottom:1px solid var(--line);font-size:.83rem}
th{color:var(--dim);font-weight:500;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em}
tr:last-child td{border-bottom:none}
.empty{color:#565d67;padding:1.4rem;text-align:center;background:var(--panel);
       border:1px solid var(--line);border-radius:10px;font-size:.85rem}
dialog{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:12px;
       max-width:760px;width:92vw;padding:0}
dialog::backdrop{background:rgba(0,0,0,.72)}
dialog header{position:static;border-radius:12px 12px 0 0}
dialog .inner{padding:1rem 1.15rem}
pre{white-space:pre-wrap;word-break:break-word;background:#0c0d10;border:1px solid var(--line);
    border-radius:8px;padding:.8rem;font-size:.78rem;max-height:46vh;overflow:auto}
dialog img{width:100%;border-radius:8px}
.kv{color:var(--dim);font-size:.78rem;margin-bottom:.5rem}
footer{color:#4b525c;font-size:.75rem;text-align:center;padding:2.5rem 1rem 1.5rem}
</style></head><body>

<header>
  <h1>Monitex Demo <small id=site></small></h1>
  <div class=spacer></div>
  <span id=camstate><span class="dot off"></span>camera off</span>
  <button id=start class=go>▶ Start camera</button>
  <button id=stop class=stop disabled>■ Stop</button>
  <button id=refresh>↻</button>
</header>

<main>
  <div class=tiles id=tiles></div>

  <h2>Events <span id=evcount style="text-transform:none;letter-spacing:0"></span></h2>
  <div class=cards id=events></div>

  <h2>Review queue</h2>
  <div id=review></div>

  <h2>Delivery failures</h2>
  <div id=dead></div>
</main>

<dialog id=modal><div class=inner id=modalbody></div></dialog>
<footer>Monitex Demo — polls every 3s. Python decides what happened; n8n decides who hears about it.</footer>

<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function jget(u){ const r = await fetch(u); if(!r.ok) throw new Error(await r.text()); return r.json(); }
async function jpost(u){ const r = await fetch(u,{method:'POST'});
  if(!r.ok){ const t = await r.text(); alert(t); throw new Error(t);} return r.json(); }

function tile(n,label,cls){ return `<div class="tile ${cls||''}"><b>${n}</b><span>${label}</span></div>`; }

function card(e){
  const img = e.has_evidence
    ? `<img src="/evidence/${e.event_id}.jpg" loading=lazy onclick="zoom('${e.event_id}')">`
    : `<div class=noimg>no evidence frame</div>`;
  const reasons = e.reasons || [];
  return `<div class=card>${img}<div class=body>
    <div class=obs>${esc(e.observed)}</div>
    <div class=meta>${esc(e.video_ts)} &middot; conf ${e.confidence} &middot; ${esc(e.event_id)}</div>
    <div class=sum>${esc(e.summary)}</div>
    <div class=tags>
      <span class="tag p-${esc(e.priority)}">${esc(e.priority)}</span>
      <span class=tag>${esc(e.category)}</span>
      ${e.needs_review ? '<span class="tag rev">needs review</span>' : ''}
      ${e.notify_status ? `<span class=tag>${esc(e.notify_status)}</span>` : ''}
      ${reasons.map(r=>`<span class=tag>${esc(r)}</span>`).join('')}
    </div>
    <div class=act><button onclick="showEmail('${e.event_id}')">✉ the email</button></div>
  </div></div>`;
}

window.zoom = id => {
  $('#modalbody').innerHTML = `<img src="/evidence/${id}.jpg">
    <div class=kv style="margin-top:.7rem">${id}</div>
    <button onclick="modal.close()">Close</button>`;
  modal.showModal();
};

window.showEmail = async id => {
  const m = await jget(`/api/events/${id}/email`);
  $('#modalbody').innerHTML = `<div class=kv><b>To:</b> ${esc(m.to)}<br>
    <b>Subject:</b> ${esc(m.subject)}<br>
    <b>Delivery:</b> ${esc(m.notify_status||'not sent')}</div>
    <pre>${esc(m.body)}</pre><button onclick="modal.close()">Close</button>`;
  modal.showModal();
};

function render(d){
  $('#site').textContent = d.site_id || 'site not configured';
  const t = d.summary.totals, f = d.summary.frames;
  $('#tiles').innerHTML =
      tile(t.events,'events')
    + tile(t.needs_review,'need review', t.needs_review? 'warn':'')
    + tile(t.high,'high priority')
    + tile(d.summary.dead_letters,'dead letters', d.summary.dead_letters? 'bad':'')
    + tile(f.analyzed+'/'+f.decoded,'frames analysed')
    + tile(f.vision_calls,'model calls');

  $('#evcount').textContent = d.events.length ? `(${d.events.length})` : '';
  $('#events').innerHTML = d.events.length
    ? d.events.map(card).join('')
    : `<div class=empty>No events yet. Press <b>Start camera</b>, then sit at the desk and walk away.</div>`;

  $('#review').innerHTML = d.review.length ? `<table>
    <tr><th>event</th><th>observed</th><th>conf</th><th>why</th></tr>` +
    d.review.map(r=>`<tr><td>${esc(r.event_id)}</td><td>${esc(r.observed)}</td>
      <td>${r.confidence}</td><td>${esc((r.reasons||[]).join(', '))}</td></tr>`).join('') + `</table>`
    : `<div class=empty>Nothing waiting on a human.</div>`;

  $('#dead').innerHTML = d.dead_letters.length ? `<table>
    <tr><th>stage</th><th>event</th><th>attempts</th><th>error</th></tr>` +
    d.dead_letters.map(r=>`<tr><td>${esc(r.stage)}</td><td>${esc(r.event_id||'—')}</td>
      <td>${r.attempts}</td><td>${esc((r.error||'').slice(0,90))}</td></tr>`).join('') + `</table>`
    : `<div class=empty>Nothing failed to deliver.</div>`;

  const on = d.camera.running;
  $('#camstate').innerHTML = on
    ? `<span class="dot live"></span>camera live &middot; ${d.camera.uptime_s}s`
    : `<span class="dot off"></span>camera off`;
  $('#start').disabled = on;
  $('#stop').disabled  = !on;
}

async function tick(){ try{ render(await jget('/api/feed')); }catch(e){ console.error(e); } }

$('#start').onclick = async () => { $('#start').disabled = true; await jpost('/control/start'); setTimeout(tick,600); };
$('#stop').onclick  = async () => { $('#stop').disabled  = true; await jpost('/control/stop');  setTimeout(tick,600); };
$('#refresh').onclick = tick;

tick(); setInterval(tick, 3000);
</script></body></html>
"""
