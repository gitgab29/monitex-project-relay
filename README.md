# Project Relay

Camera feed → structured security events → routed actions.

A live webcam stands in for a reception-post CCTV camera. Python decides *what happened*;
n8n decides *who hears about it*. Everything is recorded, so any run can be replayed and will
produce the same events.

---

## 1 · Demo

**▶ [Demo video](DRIVE_LINK_HERE)** *(3–5 min, anyone with the link)*

What you are looking at, in one paragraph: a webcam pointed at a desk. The overlay draws the
**post zone** (the desk itself) and the **approach zone** (everything else). A motion gate
decides which frames are worth looking at; YOLO finds people in those; a small state machine
turns "a person is there" into "the post became unattended at 00:04:12"; Gemini writes one
sentence describing the frame; the event lands in SQLite and is routed to email. Pull the
n8n container down and the same event still reaches a human, because the fallback is tested.

---

## 2 · What it does

The dispatcher task being removed: **someone watching a reception-post camera, noticing when
it is unmanned or when a stranger is at it, and telling the right person.** That job is mostly
waiting, and the parts that are not waiting are judgement calls a machine can make badly and a
human can correct.

```
                        ┌──────────── recorded to data/sessions/*.mp4 ────────────┐
                        │                (so any run can be replayed)             │
                        ▼                                                          │
  webcam ──▶ capture ──▶ motion gate ──▶ YOLOv8n ──▶ zones ──▶ state machine ──────┤
  (live)     30 fps      mean |Δ| vs     person     post /     MANNED/UNATTENDED/  │
                         last ANALYSED   boxes      approach   INTRUSION/LOITERING │
                         frame                                                     │
                                                                                   ▼
                                                                          event draft
                                                                                   │
                        ┌──────────────────────────────────────────────────────────┤
                        ▼                                                          ▼
                 rules: classify              ┌─────────────────────────┐   Gemini (stage 2)
                 + compose confidence         │  confidence < threshold │   one sentence,
                 + review reasons  ◀──────────┤  or fields missing      │   strict JSON
                        │                     │  or stage disagreement  │   (optional)
                        │                     └─────────────────────────┘
                        ▼
                 SQLite (WAL, single writer = Python)
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   review_queue    local HTTP API    Router
   (humans)        :8080 (n8n        n8n ──▶ SMTP ──▶ webhook
                   talks to this)    (each falls through to the next)
```

**Why a webcam as a guard post.** The brief allows any short clip. A live camera was chosen
deliberately: it is the only option where the *timing* logic is real. "The post has been empty
for eight seconds" cannot be faked by a pre-cut file without the file already containing the
answer. Everything is still recorded, so the reproducibility a file gives you is not lost.

---

## 3 · Run it

### The keyless path — no API keys, no Docker, no camera

```bash
pip install -e .
relay replay samples/reception_demo.mp4 --vision none --no-n8n
```

This is deliberately the first thing in this README. It runs the whole pipeline, writes real
events to SQLite, and needs nothing but Python. With `--vision none` the summaries come from a
template and every event is flagged `needs_review` — which is the *designed* degrade path, not
a broken mode.

```bash
relay summary          # what is in the database
relay review list      # what a human is being asked to look at
pytest -q              # 195 tests
```

### The full path

```bash
cp .env.example .env   # then fill in the two credentials below
```

| What | Where to get it | Env vars |
|---|---|---|
| Gemini API key | https://aistudio.google.com/apikey | `GEMINI_API_KEY` |
| Gmail app password | https://myaccount.google.com/apppasswords (needs 2-Step Verification on) | `SMTP_USER`, `SMTP_APP_PASSWORD`, `ALERT_TO` |

Verify both actually work before relying on them — a wrong key degrades *silently* by design,
so it looks identical to a working one until it matters:

```bash
python scripts/verify_credentials.py        # or: .\scripts\Verify-Credentials.ps1
```

Then:

```bash
relay api                                   # the HTTP surface n8n calls back on, :8080
relay run --source webcam --show            # live, with the overlay window
```

For n8n, see [n8n/README.md](n8n/README.md). **It is optional** — if it is not running, the
router falls through to direct SMTP and says so in the log.

### If `pip install` fails on Windows

If the install dies with an `OSError: [Errno 2] No such file or directory` naming a file deep
inside `torch/_functorch/...`, that is **Windows' 260-character path limit**, not a broken
package. Torch ships some very long filenames, and a deep clone location pushes them over.

Clone somewhere shallow — `C:\dev\relay` rather than
`C:\Users\you\Downloads\assessments\...` — or enable long paths:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

The error names torch, which sends you looking in the wrong place; the cause is the path.

### The feed used

A **live USB webcam** (`CAMERA_INDEX=1`, 1280×720 requested, 30 fps nominal) pointed at a desk
standing in for a reception post. No footage of any real site is used. `samples/reception_demo.mp4`
is a recorded session from that same camera, included so the keyless path above works with no
hardware.

---

## 4 · Video → events

### Sampling and the motion gate

Decoding every frame and running YOLO on every frame are different costs. The pipeline decodes
everything (it has to, to record) and **analyses roughly 1 per second**, with two escapes:

- **Motion gate** — mean absolute grayscale difference against the *last analysed* frame, not
  the last decoded one. Comparing against the last decoded frame makes a slow walk invisible,
  because each individual step is small.
- **Heartbeat** — if nothing has been analysed for `HEARTBEAT_S=5.0`, analyse anyway. Without
  this, a scene that goes perfectly still is indistinguishable from a crashed pipeline.

Real numbers from the recorded demo session:

```
181 decoded, 7 analysed (3.9%), 0 skipped quiet
```

3.9% is the point. On a static reception camera most frames say nothing.

### Zones

```
POST_ZONE=0.15,0.35,0.85,1.00     # normalised x0,y0,x1,y1 — the desk, foreground
ZONE_MARGIN=0.04                  # centre within this of an edge → straddle=true
```

Normalised so the config survives a resolution change. The **approach zone is everything
outside the post zone** — defining it as the complement means a person cannot be in neither.

A detection whose centre sits within `ZONE_MARGIN` of a post edge is marked `straddle` and
resolved by box size, and **that fact is recorded as a review reason** rather than hidden.

### The four scenarios

| `observed` | Category | Priority | Fires when |
|---|---|---|---|
| `post_manned` | routine | low | a person is in the post zone for 2 consecutive analysed frames |
| `post_unattended` | other | **high** | post continuously empty for `UNATTENDED_DWELL_S=8.0` |
| `unidentified_person_at_post` | intrusion | **high** | someone at the post while it should be unmanned |
| `person_loitering_near_entry` | loitering | medium | in the approach zone for `LOITER_DWELL_S=6.0` |

Dwell values are **demo-tuned** so the scenarios are reachable in a 4-minute video. Production
values would be 60–120 s for `UNATTENDED_DWELL_S`.

**Events fire on state transitions, not on frames.** A guard sitting still for an hour is one
`post_manned` event, not 3,600. This is the difference between an event log and a frame log.

The state machine has hysteresis on purpose: a single missed detection is common and means
nothing, so MANNED needs two consecutive frames to open, and absence must be *continuous* for
the dwell before UNATTENDED is believed.

### Confidence

Composed from three measured terms **before any model is consulted**:

| Term | What it measures |
|---|---|
| `yolo_term` | detector confidence on the box |
| `zone_term` | how cleanly the box sits inside its zone |
| `luma_term` | frame brightness — can the camera actually see? |

```python
confidence = min(yolo_term, zone_term, luma_term)
```

**The minimum, not the product.** Three independent 0.9s multiply to 0.73 and would push a
perfectly healthy detection under the review bar. The minimum says something defensible in one
sentence — *we are only as sure as the least reliable thing we measured* — and it makes the
lights-off case honest for free: a dark frame collapses `luma_term`, which drags the event
down on its own without anything special-casing darkness.

---

## 5 · AI integration

### Two stages, and why stage 2 is narrow

**Stage 1 (YOLOv8n)** answers *is there a person, and where*. It is cheap, local, and runs on
every analysed frame.

**Stage 2 (Gemini)** answers *what does this look like to a human*. It runs only when an event
is already being created — never per frame. On the demo session that is **1 call per event**,
not 181.

The model is asked to **describe, not to decide**. No prompt asks for a category, a priority or
an action. Those are rules, and a rule can be unit-tested; a model's opinion cannot. This is
the single most important design decision in the AI layer.

### The schema it must answer in

```python
class VisionVerdict(BaseModel):
    summary: str = Field(max_length=160)      # lands in an email subject
    person_count: int = Field(ge=0, le=10)
    people_at_desk: int = Field(ge=0, le=10)
    lighting: Literal["good", "dim", "dark"]
    confidence: float = Field(ge=0.0, le=1.0)
```

### The prompt, verbatim

```
You are a security camera analyst. You will be shown a single frame from a fixed camera
watching a reception desk (the "guard post") and the area in front of it (the "approach zone").

Describe only what is visible in this frame. Do not speculate about intent, identity or what
happened before or after. If the image is too dark or blurred to tell, say so plainly and lower
your confidence -- an honest low-confidence answer is far more useful to us than a confident
guess.

Return JSON matching the given schema. Field notes:
- summary: one sentence, at most 160 characters, describing what is happening. Plain factual
  language, no preamble.
- person_count: how many people you can see anywhere in the frame.
- people_at_desk: how many of those are at or behind the reception desk itself.
- lighting: "good" if the scene is clearly visible, "dim" if it is murky but readable, "dark"
  if you genuinely cannot make out the scene.
- confidence: 0 to 1, how much you trust your own reading of this frame.
```

The user message **tells the model what the detector already found**. Withholding it to "avoid
bias" sounds principled but throws away the only thing that makes disagreement meaningful — we
want to know when the model looks at the same frame and sees something else.

### Adjudication is a count comparison

Not a second opinion, not a tie-break. If YOLO says 3 people and the model says 1 *with
confidence ≥ 0.7*, that is a **confident disagreement**, and it caps event confidence at 0.50
and adds `stage_disagreement` — routing it to a human rather than picking a winner.

This fired for real on the demo footage:

```
stage disagreement: detector saw 3, model saw 1 (model conf 0.95)
event ... conf=0.50 NEEDS REVIEW ['high_priority_low_confidence', 'low_confidence',
                                  'stage_disagreement']
```

An unconfident disagreement just averages the two, because a model that is unsure of itself
disagreeing is not evidence of much.

### Swapping the backend

One env var: `VISION_BACKEND=gemini|fake|none`. `fake` is deterministic and is what the chaos
modes use, so failure paths are reproducible and cost nothing.

### Privacy trade-off, stated plainly

Free-tier Gemini **may use submitted content to improve Google's models**. Frames are
downscaled to `VISION_MAX_WIDTH=960` and only key frames are sent, but for a real deployment
this is a paid-tier or on-prem decision, not a technical one. Saying so is part of the answer.

---

## 6 · Reliability & error handling

Everything below is demonstrable with a flag. Nothing here is theoretical.

| Failure | How it is detected | Response | Evidence left behind |
|---|---|---|---|
| Model timeout | per-call ceiling `VISION_TIMEOUT_S=12` | 3 attempts, exponential backoff | `dead_letter(stage='vision')`, `vision_timeout` reason, template summary |
| Bad JSON from model | Pydantic validation | **one repair attempt**, then degrade | `dead_letter` with the unparseable text kept, `vision_bad_schema` |
| Rate limited | 429 / `RESOURCE_EXHAUSTED` | backoff, then degrade | `vision_rate_limited` |
| n8n down | connection refused / non-2xx | 3 attempts, then **fall through to SMTP** | `dead_letter(stage='n8n')`, `notify_status='emailed:python'` |
| SMTP down | `SMTPException` | 3 attempts, then fall through to webhook | `dead_letter(stage='email')` with the full payload |
| Nothing configured | no sink has credentials | log what *would* have been sent | **not** a dead letter — a config state is not a failure |
| Re-run | `event_id` collision | upsert, skip | `0 inserted, N skipped` |

Backoff: `min(RETRY_CAP_S, RETRY_BASE_S * 2**n) + jitter(0, 0.5)`, defaults `1.0` / `8.0`.
The jitter is not decoration — without it every client that failed together retries together.

**Dead letters carry the payload**, so a failed notification can be re-sent by hand. A dead
letter row that only says "it failed" is an apology, not a recovery path.

### Run the failure paths yourself

```bash
relay replay <session.mp4> --chaos timeout     # 3 attempts, backoff, degrade
relay replay <session.mp4> --chaos badschema   # 1 repair attempt, then degrade
relay replay <session.mp4> --chaos sink-fail   # primary sink fails, fallback delivers
relay replay <session.mp4> --chaos ratelimit
```

Real output:

```
vision attempt 1/3 failed: call exceeded 12.0s; sleeping 1.29s
vision attempt 2/3 failed: call exceeded 12.0s; sleeping 2.28s
vision failed after 3 attempt(s) -- degrading to a template summary
```

> **One bug worth naming.** The shutdown drain was a flat 30 s while the worst-case retry chain
> is 12 s × 3 + backoff = 39 s. So on a short clip the worker was abandoned mid-retry and the
> event shipped **without** its `vision_timeout` reason and **without** its dead-letter row —
> the degrade path ran correctly and then lost its own evidence. Found by running `--chaos
> timeout` rather than trusting it. The drain is now derived from the retry budget, with tests
> pinning the two together.

---

## 7 · Human review

Two genuinely different reasons a human is needed, and they are not merged:

1. **We are not sure** — confidence below the bar.
2. **We cannot act** — a required field is missing, e.g. no `SITE_ID`, so the event cannot be
   routed to a site at all. No amount of confidence fixes this.

Thresholds are tiered, because the cost of being wrong is not uniform:

| Config | Default | Why |
|---|---|---|
| `REVIEW_CONF_THRESHOLD` | 0.75 | baseline |
| `REVIEW_CONF_OTHER` | 0.85 | stricter for `other` — the catch-all we trust least |
| `REVIEW_CONF_HIGH` | 0.90 | strictest for high priority — a false page wakes someone up |

### Every reason code

| Code | Meaning |
|---|---|
| `low_confidence` | composed confidence below the threshold for this category/priority |
| `high_priority_low_confidence` | a high-priority event did not clear the stricter 0.90 bar |
| `other_low_confidence` | an `other` event did not clear the stricter 0.85 bar |
| `missing_site_id` | `SITE_ID` not configured, so the event cannot be routed |
| `missing_video_ts` | no timestamp into the source recording |
| `zone_straddle` | a detection sat on the post-zone boundary and was resolved by size |
| `stage_disagreement` | detector and model disagreed on how many people are present |
| `vision_unavailable` | no model reachable, summary is a template |
| `vision_bad_schema` | reply could not be parsed, even after one repair |
| `vision_timeout` | model did not answer within the timeout, after retries |
| `vision_rate_limited` | model rate-limited us, after retries |
| `vision_failed` | model call failed for an unclassified reason |
| `model_reports_dark` | the model itself said the frame was too dark to read |

These live in one dict, and **attaching an undocumented reason raises**. A reason nobody can
act on should not reach a dispatcher, and a test asserts every documented reason is reachable —
so this table and the code cannot drift apart.

### Resolving

```bash
relay review list
relay review resolve <event_id> --as confirmed|false_alarm|dismissed --notes "..."
```

Resolution sets `resolution`, `resolved_by`, `notes`, and leaves `resolution_notified_at` NULL.
The n8n sweep polls for exactly that combination — resolved **and** not yet announced — and
acks only *after* the mail is away. A slow email therefore cannot produce two "review closed"
messages, and a failed one is retried on the next sweep rather than silently lost.

---

## 8 · Idempotency

```python
event_id = sha256(f"{session_id}|{track_key}|{bucket}")[:16]
```

Derived from what the event *is*, not when it was written, so the same moment in the same
recording always produces the same id. Writes are upserts.

- **`runs` ledger** — every run is recorded, so a repeat run is recognised:
  `this run repeats session chaos-to-03 under the same config (first run cbac3178)`.
- **`DEDUPE_WINDOW_S=5.0`** — a drift guard. Live capture and a later replay of the *same*
  footage can land a track a fraction of a second apart after compression; a proximity match
  on `(session_id, track_key)` absorbs that.

Proven, and proven through the **degrade** path rather than only the happy one:

```
pass 1:  1 inserted, 0 skipped
pass 2:  0 inserted, 1 skipped     ← plus the "repeats session" notice
pass 3:  0 inserted, 1 skipped
```

---

## 9 · Data model

SQLite, WAL mode, **single writer = Python**. n8n never touches the database; it asks the HTTP
API, and Python serialises every write. n8n has no SQLite node, and bind-mounting a `.db` into
a container so two processes can write it is a well-known corruption path.

| Table | One line |
|---|---|
| `sessions` | one recorded video per row: path, fps, dimensions, site |
| `runs` | every execution: mode, config fingerprint, counters, status |
| `observations` | per-analysed-frame record: motion, luma, detections JSON |
| `events` | the brief's event schema, plus `notify_status` |
| `review_queue` | what a human must look at, and how it was resolved |
| `dead_letter` | anything that could not be delivered, **with its payload** |
| `vision_calls` | one row per model call: attempt, purpose, latency, ok, error |

Views: `v_events_by_category`, `v_review_depth`, `v_run_metrics`.

---

## 10 · Trade-offs & cut corners

Honest list. These are choices and limits, not oversights.

- **n8n is designed and exported, but was never run live.** The build machine has 7.7 GB of RAM;
  Docker Desktop never started its daemon, and `npx n8n` died twice with a *native* zone
  allocation OOM. This was a pre-decided fallback (risk R2), and the consequence is that
  **Python's SMTP sink is the primary notifier** — which is tested, and which is what the demo
  shows. Both workflow JSONs are contract-verified against the live API (every field binding,
  every callback body, ack idempotency) but not against a running n8n. See
  [n8n/README.md](n8n/README.md).
- **Four of six categories are reachable on this feed.** `vehicle` and `fire_smoke` cannot be
  demonstrated with a webcam on a desk. The enum is still complete because a partial enum
  leaks "we only handled the easy ones" into every downstream consumer.
- **Rectangle zones, not polygons.** A polygon editor is a day of work and buys nothing for a
  rectangular desk.
- **CPU-only, YOLOv8n.** ~80–150 ms/frame. Fine at 1 fps analysis; would not hold at 30 fps.
- **Single camera.** The schema carries `site_id` and sessions are per-camera, so multi-camera
  is config plus a process per camera — but it is not built or tested.
- **Dwell values are demo-tuned** (8 s / 6 s). Production would be 60–120 s.
- **No tracker.** Identity across frames is a zone-and-proximity heuristic, not ByteTrack, so
  two people swapping positions can look like one person. `stage_disagreement` catches some of
  this, which is partly why it exists.
- **Free-tier Gemini may train on submitted frames.** A paid-tier or on-prem decision for real use.
- **Stretch items not built:** HTML review form, scheduled digest email, Google Sheets export.
  The 15-minute digest exists as a documented n8n branch, not a running one.

---

## 11 · What I'd do next

1. **A real tracker (ByteTrack)** for per-person identity — the single biggest accuracy win, and
   it would make "the *same* person has been loitering" provable rather than inferred.
2. **Multi-camera site config** — a YAML per site, one process per camera, events already carry
   `site_id`.
3. **Polygon zones** with an editor, for doorways that are not rectangles.
4. **Migration tooling** (Alembic or equivalent). The schema is currently `CREATE TABLE IF NOT
   EXISTS`, which is fine for a take-home and not fine for a second deployment.
5. **A metrics endpoint** — `vision_calls` already has the data; it wants Prometheus, not a view.
6. **Document intake**, the brief's stretch: the same event schema with `source_type='document'`.

---

## 12 · Code map

```
relay/
  capture/     source.py motion.py recorder.py overlay.py   frames in, MP4 out, motion gate
  detect/      yolo.py zones.py state.py                    boxes → zones → state machine
  vision/      base.py gemini.py fake.py null.py            the model seam
               prompts.py stage.py                          prompts as text; retry/repair/degrade
  sinks/       base.py n8n.py email.py webhook.py           delivery, one interface
  schema.py    the brief's Event, plus VisionVerdict
  rules.py     classify, compose_confidence, review_reasons  ← the testable judgement
  store.py     SQLite, the only writer
  router.py    who gets told, and what happens when telling fails
  api.py       the HTTP surface n8n calls back on
  pipeline.py  the loop that holds it together
tests/         12 files, 195 tests
```

**To add a scenario:** add a state to `detect/state.py`, then one row in `CLASSIFICATION` in
`rules.py`. Unknown states already degrade to `other/medium` rather than crashing, so a
half-finished scenario produces a reviewable event instead of an outage.

**To add a sink:** implement `deliver(event, *, review_link, reasons) -> DeliveryResult` and add
it to the chain in `router.py`. The router handles retry, dead-lettering and fallback for you.

---

MIT licensed. Built as a take-home assessment.
