"""`relay` -- the command line.

    relay run     --source webcam [--camera N] [--show] [--vision ...] [--no-n8n] [--chaos M]
    relay replay  PATH.mp4 [--show] [--vision ...] [--session-id ID]
    relay review  list | resolve EVENT_ID --as confirmed|false_alarm|dismissed
    relay summary [--json]
    relay sessions list
    relay api     [--port 8080]

Every flag that changes behaviour also has an environment variable, because the demo needs to
change one thing and restart without editing code.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import torch_threads
from .config import load_settings
from .logging_setup import get_logger, setup_logging
from .store import Store

log = get_logger("relay.cli")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--vision", choices=["gemini", "fake", "none"], help="override VISION_BACKEND")
    p.add_argument("--no-n8n", action="store_true", help="skip the n8n sink; notify directly or not at all")
    p.add_argument("--chaos", choices=["timeout", "badschema", "sink-fail", "ratelimit"],
                   help="force a failure mode so the recovery path can be watched")
    p.add_argument("--max-seconds", type=float, help="stop after this much VIDEO time")
    p.add_argument("--show", action="store_true", help="draw the overlay window")
    p.add_argument("--env", default=".env", help="path to the env file (default: .env)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="relay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="capture live from a camera")
    run.add_argument("--source", default="webcam", choices=["webcam"])
    run.add_argument("--camera", type=int, help="override CAMERA_INDEX")
    _common(run)

    rep = sub.add_parser("replay", help="re-run the pipeline over a recorded session")
    rep.add_argument("path", help="path to an .mp4")
    rep.add_argument("--session-id", help="override the session id recovered from the filename")
    _common(rep)

    rev = sub.add_parser("review", help="the human review queue")
    revsub = rev.add_subparsers(dest="review_cmd", required=True)
    revsub.add_parser("list", help="show everything awaiting a decision")
    res = revsub.add_parser("resolve", help="record a human decision")
    res.add_argument("event_id")
    res.add_argument("--as", dest="resolution", required=True,
                     choices=["confirmed", "false_alarm", "dismissed"])
    res.add_argument("--notes", default=None)
    res.add_argument("--by", default="cli")

    s = sub.add_parser("summary", help="what this database contains")
    s.add_argument("--json", action="store_true")

    sess = sub.add_parser("sessions", help="recorded sessions")
    sess.add_argument("sessions_cmd", nargs="?", default="list", choices=["list"])

    api = sub.add_parser("api", help="the local HTTP API n8n talks to")
    api.add_argument("--port", type=int, help="override API_PORT")

    for p in (rev, s, sess, api):
        p.add_argument("--env", default=".env", help="path to the env file (default: .env)")
    return ap


# ---------------------------------------------------------------------- commands

def _build_pipeline(args, cfg, store, mode: str):
    """Assemble the pipeline. Everything expensive is constructed here, once, so the run loop
    stays a loop."""
    from .detect.state import PostStateMachine
    from .detect.yolo import YoloPersonDetector
    from .detect.zones import ZoneModel
    from .enrich import enrich_and_store
    from .pipeline import Pipeline
    from .reliability import ChaosConfig
    from .router import build_router
    from .vision import get_backend
    from .vision.stage import VisionStage

    zones = ZoneModel(cfg)
    log.info("zones: %s", zones.describe())

    chaos = ChaosConfig(args.chaos) if args.chaos else ChaosConfig()
    backend = get_backend(cfg, chaos)
    vision = VisionStage(backend, cfg, chaos)
    log.info("vision backend: %s (timeout %.0fs, %d attempts, min interval %.1fs)",
             vision.name, cfg.vision_timeout_s, cfg.retry_attempts, cfg.vision_min_interval_s)

    router = build_router(cfg, store, use_n8n=not args.no_n8n, chaos=chaos)
    log.info(
        "notifications: primary=%s fallback=%s",
        getattr(router.primary, "name", "none"), getattr(router.fallback, "name", "none"),
    )

    pipe = Pipeline(
        cfg, store, mode=mode, show=args.show, max_seconds=args.max_seconds, chaos=args.chaos,
        detector=YoloPersonDetector(cfg, zones=zones),
        state_machine=PostStateMachine(cfg),
        enricher=enrich_and_store,
        router=router,
    )
    pipe.vision = vision
    return pipe


def cmd_run(args, cfg, store) -> int:
    import dataclasses

    if args.camera is not None:
        cfg = dataclasses.replace(cfg, camera_index=args.camera)
    _build_pipeline(args, cfg, store, "live").run_live()
    return 0


def cmd_replay(args, cfg, store) -> int:
    _build_pipeline(args, cfg, store, "replay").run_replay(args.path, session_id=args.session_id)
    return 0


def cmd_review(args, cfg, store) -> int:
    if args.review_cmd == "list":
        rows = store.pending_review()
        if not rows:
            print("review queue is empty")
            return 0
        print(f"{'event_id':18} {'observed':30} {'conf':>5}  reasons")
        for r in rows:
            reasons = ", ".join(json.loads(r["reasons_json"]))
            print(f"{r['event_id']:18} {r['observed']:30} {r['confidence']:5.2f}  {reasons}")
        return 0

    ok = store.resolve_review(args.event_id, args.resolution, by=args.by, notes=args.notes)
    if ok:
        print(f"{args.event_id} -> {args.resolution}")
        return 0
    print(f"{args.event_id}: not in the review queue, or already resolved", file=sys.stderr)
    return 1


def cmd_summary(args, cfg, store) -> int:
    data = store.summary()
    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return 0

    t, f, r = data["totals"], data["frames"], data["review"]
    print(f"\nevents: {t['events']}   needs review: {t['needs_review']}   high priority: {t['high']}")
    if data["by_category"]:
        print("\n  category        priority     n   avg conf")
        for row in data["by_category"]:
            print(f"  {row['category']:15} {row['priority']:8} {row['n']:4}   {row['avg_conf']}")
    print(f"\nreview queue:  {r['open']} open, {r['closed']} closed")
    print(f"dead letters:  {data['dead_letters']}")
    analysed = f"{f['analyzed']}/{f['decoded']}"
    pct = 100 * f["analyzed"] / max(f["decoded"], 1)
    print(f"\nframes:        {analysed} analysed ({pct:.1f}%), {f['skipped_quiet']} skipped as quiet")
    print(f"vision calls:  {f['vision_calls']} ({f['vision_failures']} failed)")
    print(f"events:        {f['inserted']} inserted, {f['skipped']} skipped as duplicates\n")
    return 0


def cmd_sessions(args, cfg, store) -> int:
    rows = store.list_sessions()
    if not rows:
        print("no sessions recorded yet")
        return 0
    print(f"{'session_id':28} {'started':26} {'fps':>5} {'events':>6}  file")
    for r in rows:
        print(f"{r['session_id']:28} {r['started_at']:26} {r['fps']:5.1f} {r['n_events']:6}  {r['file_path']}")
    return 0


def cmd_api(args, cfg, store) -> int:
    import uvicorn

    from .api import create_app

    port = args.port or cfg.api_port
    log.info("serving the relay API on %s:%d", cfg.api_host, port)
    uvicorn.run(create_app(cfg, store), host=cfg.api_host, port=port, log_level="warning")
    return 0


# ---------------------------------------------------------------------- entry

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_settings(args.env)
    if getattr(args, "vision", None):
        cfg = __import__("dataclasses").replace(cfg, vision_backend=args.vision)
    cfg.ensure_dirs()
    setup_logging(cfg.log_level, cfg.logs_dir)
    torch_threads()

    store = Store(cfg.db_path)
    try:
        handler = {
            "run": cmd_run, "replay": cmd_replay, "review": cmd_review,
            "summary": cmd_summary, "sessions": cmd_sessions, "api": cmd_api,
        }[args.cmd]
        return handler(args, cfg, store)
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
