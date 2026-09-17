# Project Relay

Camera feed -> structured security events -> routed actions.

> **Status: build in progress.** Sections are filled with real numbers from real runs as they
> are produced; nothing below is aspirational once it is written.

## 1 · Demo

_(Drive link, anyone with the link — added at ship time.)_

## 2 · What it does

_(The dispatcher task removed; architecture diagram; why webcam-as-guard-post.)_

## 3 · Run it

_(Keyless path first: `pip install -e .` then `relay replay samples/reception_demo.mp4 --vision none --no-n8n`.
Then the full path: `.env`, Gemini key, Gmail app password, `docker compose up -d`, import `n8n/*.json`,
`relay api`, `relay run --source webcam --show`. The feed used, stated plainly.)_

## 4 · Video → events

_(Sampling: 1 fps + heartbeat. Motion gate with real decoded/analysed/skipped counts. Zone model +
overlay screenshot. The four-row scenario table. Why events fire on transitions. Confidence composition.)_

## 5 · AI integration

_(The two-stage cascade and why stage 2 is deliberately narrow. The `VisionVerdict` schema. The prompt,
verbatim. The repair prompt. Adjudication as a count comparison. Backend swap via one env var. Call
count per run. Free-tier data-policy trade-off.)_

## 6 · Reliability & error handling

_(Table: failure → detection → response → evidence in the DB, for model timeout, bad schema, rate limit,
n8n down, SMTP down, re-run. Backoff parameters. Dead-letter table. The chaos flags.)_

## 7 · Human review

_(Trigger table: config keys, defaults, reason codes. The two distinct paths — low confidence vs.
incomplete data. Resolution flow.)_

## 8 · Idempotency

_(`event_id` derivation, upsert, runs ledger, drift window, the `0 inserted / N skipped` log line.)_

## 9 · Data model

_(Table list, one line each. The views.)_

## 10 · Trade-offs & cut corners

_(Honest bullet list.)_

## 11 · What I'd do next

_(Multi-camera site config, polygon zones, ByteTrack for per-person identity, document intake,
migration tooling, metrics endpoint.)_

## 12 · Code map

_(Layout, one line each. How to add a scenario. How to add a sink.)_

---

MIT licensed. Built as a take-home assessment.
