"""Logging: readable on the console, structured on disk.

Two audiences. A human watching a demo needs one short line per thing that happened. A
reviewer asking "what actually ran" afterwards needs machine-readable records with the event
ids in them. Same call site, two handlers, so neither is an afterthought.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Fields we never render in the message body because they are structured-only.
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
}


class JsonLineFormatter(logging.Formatter):
    """One JSON object per line. Anything passed via `extra=` rides along as a field."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                try:
                    json.dumps(v)
                    payload[k] = v
                except (TypeError, ValueError):
                    payload[k] = str(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """`10:41:07 INFO  event post_manned routine low conf=0.91`  -- one line, no clutter."""

    def format(self, record: logging.LogRecord) -> str:
        t = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        msg = record.getMessage()
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"
        return f"{t} {record.levelname:<5} {msg}"


def setup_logging(level: str = "INFO", log_dir: str | Path = "data/logs") -> Path:
    """Attach both handlers to the root logger. Idempotent: calling it twice does not
    double every line, which matters because the CLI and the API can both initialise."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    for h in list(root.handlers):
        if getattr(h, "_relay", False):
            root.removeHandler(h)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ConsoleFormatter())
    console._relay = True  # type: ignore[attr-defined]
    root.addHandler(console)

    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "relay.jsonl"
    fileh = logging.FileHandler(path, encoding="utf-8")
    fileh.setFormatter(JsonLineFormatter())
    fileh.setLevel(logging.DEBUG)
    fileh._relay = True  # type: ignore[attr-defined]
    root.addHandler(fileh)

    # These two are chatty and say nothing we need.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    return path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
