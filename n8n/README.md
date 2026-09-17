# The n8n layer

Two workflows. `relay_router.json` decides who hears about an event; `relay_review_sweep.json`
tells people when a review item is closed.

## Status: designed and exported, not demonstrated

Be clear on this before reading further, because it is the honest state of this directory.

These workflows are **exported artifacts that have not been run on a live n8n instance.** The
build machine has 7.7 GB of RAM. Docker Desktop wedged without ever starting its daemon, and the
fallback — `npx n8n` on the host — died twice with
`FATAL ERROR: Zone Allocation failed - process out of memory`. That is a *native* zone
allocation failure, not a V8 heap limit, so `--max-old-space-size` does not help it; the OS was
simply out of physical memory.

`BUILD_PLAN.md` Section 2 pre-decided this exact outcome as cut-order rungs 1 and 2, and
`BUILD_PLAN.md` Section 10 risk R2 named the third rung: **Python's `SmtpEmailSink` is the
primary notifier.** That is what the system actually does today, and it is tested. The n8n
hybrid is designed, exported and importable — it is not proven.

What that means in practice:

- **Nothing depends on n8n being up.** `relay.router` degrades `n8n -> SMTP -> generic webhook`,
  and the degrade path has tests. An n8n that is down is a failed sink, not a failed pipeline.
- The JSON below is written against the real API surface in [`relay/api.py`](../relay/api.py)
  and the real payload in [`relay/sinks/n8n.py`](../relay/sinks/n8n.py), and it is
  structurally validated — node counts, connection endpoints, and every `$('Node')` expression
  resolve. It is not validated against a running n8n's node schemas.
- Expect to fix small things on first import: node type versions move between n8n releases.
  The routing logic, the field bindings and the callback contract are the parts worth reviewing.

## What each workflow does

### `relay_router` — webhook, 10 nodes

Python POSTs one event; n8n decides the audience.

```
Webhook (POST /webhook/relay-event)
  -> Route fields (Set)            flattens $json.body, builds subjects + body, holds api_base
  -> Switch                        needs_review -> high -> medium -> (fallback) routine
       out 0 review   -> Send review email        --success--> Mark notified   -> Respond 200
       out 1 high     -> Send high-priority email --error----> Post dead letter -> Respond 200
       out 2 medium   -> No-Op (stored already, digest picks it up)             -> Respond 200
       out 3 routine  -> No-Op (store only)                                     -> Respond 200
```

Two things in there are deliberate and are the reason this layer exists at all:

**`needs_review` is checked before `priority`.** A low-confidence event is a question for a
human, and that outranks how urgent the machine guessed it was. A high-priority event the model
is not sure about should read `REVIEW NEEDED`, not `HIGH`.

**Both email nodes have an error output wired to `POST /dead-letter`.** Retry on fail, 3 tries,
2 s apart; if it still fails the payload lands in the dead-letter table so it can be resent by
hand. A notification that quietly failed is worse than no notification, because the operation
believes someone was told.

The success path calls `POST /events/{event_id}/notified`, which is why the Python sink records
`accepted_by_n8n` rather than `emailed` — n8n has taken responsibility, but the mail has not
gone yet. The two states stay distinguishable in the database.

### `relay_review_sweep` — schedule, 7 nodes

```
Every minute -> Sweep config (Set) -> GET /review/resolved?unnotified=true
             -> Split Out `resolved` -> Send review-closed email
                  --success--> POST /review/{id}/ack-notified
                  --error----> POST /dead-letter   (NOT acked: next sweep retries)
```

`unnotified=true` is the whole trick. The sweep only ever sees resolutions nobody has been told
about, and the ack happens *after* the mail is away, so a slow email cannot produce two "review
closed" messages and a failed one is not silently marked as sent.

## Import

1. Start n8n and open `http://localhost:5678`. Create the owner account — it is local-only, a
   throwaway email is fine.
2. **Credentials -> Add credential -> SMTP:**
   | Field | Value |
   |---|---|
   | Host | `smtp.gmail.com` |
   | Port | `587` |
   | SSL/TLS | off (STARTTLS) |
   | User | your Gmail address |
   | Password | a Gmail **app password**, not the account password (2-Step Verification must be on) |
3. **Workflows -> Import from File** for each of `relay_router.json` and
   `relay_review_sweep.json`.
4. Attach the SMTP credential to every email node — 2 in the router, 1 in the sweep. Imported
   JSON carries no credential ids on purpose, so nothing points at a credential that does not
   exist on your instance.
5. Set the API base — see below.
6. **Activate both.** The router's production URL `/webhook/relay-event` only exists while the
   workflow is active; an inactive workflow answers on `/webhook-test/` and only for one call.

## The one field to change

Both files hold the API base in a single `Set` node — `Route fields` in the router, `Sweep
config` in the sweep. Everything downstream references it, so there is nothing to find-and-replace.

```
={{ $env.RELAY_API_BASE || 'http://localhost:8080' }}
```

| Where n8n runs | api_base |
|---|---|
| `npx n8n` on the host | `http://localhost:8080` (the default — nothing to change) |
| n8n in Docker | `http://host.docker.internal:8080` |

Either edit that one field, or set `RELAY_API_BASE` in n8n's environment and leave the JSON
alone. The email `to`/`from` read `$env.ALERT_TO` / `$env.ALERT_FROM` the same way, falling back
to `alerts@example.com` — change those in the email nodes if you would rather not use env vars.

The reverse direction, Python -> n8n, is `N8N_WEBHOOK_URL` in `.env`:
`http://localhost:5678/webhook/relay-event`.

On Windows, uvicorn will trigger a firewall prompt the first time n8n calls back. Allow it on
private networks.

## Test it

With `relay api` running on 8080 and the router active:

```bash
# high-priority branch -> "HIGH" subject
curl -X POST http://localhost:5678/webhook/relay-event \
     -H "Content-Type: application/json" \
     -d @samples/sample_event.json

# review branch -> "REVIEW NEEDED" subject, null site_id, confidence 0.41
curl -X POST http://localhost:5678/webhook/relay-event \
     -H "Content-Type: application/json" \
     -d @samples/sample_event_review.json
```

Both should answer `{"ok":true,"event_id":"..."}`. Then confirm the callback actually landed —
this is the part worth checking, because it proves the loop closed rather than just that the
webhook accepted:

```bash
curl http://localhost:8080/events/8f2a1c4e9b7d3a60     # notified_at should be set
curl http://localhost:8080/dead-letters                # empty unless a send failed
```

For the sweep, resolve something and wait a minute:

```bash
relay review resolve 3d5b8e07a1c26f94 --as false_alarm --notes "cleaner, badged"
# within 60s: a "Review closed" email, and the row stops appearing in
curl "http://localhost:8080/review/resolved?unnotified=true"
```

If the SMTP credential is missing or wrong, the email nodes fail, the error output fires, and the
events land in `/dead-letters` with the SMTP error attached. That is the designed behaviour, and
it is a reasonable way to test the failure path on purpose.
