"""Identity: session ids, run ids, and the deterministic event id.

The whole re-run guarantee rests on this module. An event id is a pure function of
(session, what happened, roughly when), so the same session file analysed twice produces the
same ids and the second pass upserts instead of inserting.

Deliberately *not* in the hash: the run id, the wall clock, the config, the frame index.
Including any of them would make a replay mint new ids, which is exactly the bug.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime

#: sess-YYYYMMDD-HHMMSS-xxxx . The session id is minted once at capture and then carried in
#: the recording's filename, so a replay can recover it without a database.
SESSION_RE = re.compile(r"(sess-\d{8}-\d{6}-[0-9a-f]{4})")


def new_session_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"sess-{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def new_run_id() -> str:
    return uuid.uuid4().hex


def session_id_from_path(path: str) -> str | None:
    """Recover the session id from a recording's filename. `None` if it isn't one of ours,
    in which case the caller mints a fresh id (or the user passes --session-id)."""
    m = SESSION_RE.search(str(path))
    return m.group(1) if m else None


def bucket_for(first_seen_ms: int, bucket_s: int) -> int:
    """Quantise the start of an event into a coarse time bucket.

    Without this, two runs that place a transition 40 ms apart would hash differently. The
    bucket is what makes the id tolerant of small timing differences -- and `DEDUPE_WINDOW_S`
    in the store catches the case where the transition falls either side of a bucket edge.
    """
    if bucket_s <= 0:
        raise ValueError("bucket_s must be positive")
    return int(first_seen_ms) // (int(bucket_s) * 1000)


def event_id(session_id: str, track_key: str, bucket: int) -> str:
    """sha256(session|track_key|bucket), truncated to 16 hex chars.

    16 hex chars is 64 bits. Collisions need ~2**32 events in one session before they are
    even worth thinking about; a busy session produces tens.
    """
    raw = f"{session_id}|{track_key}|{bucket}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def event_id_for(session_id: str, track_key: str, first_seen_ms: int, bucket_s: int) -> str:
    """The composition the pipeline actually calls."""
    return event_id(session_id, track_key, bucket_for(first_seen_ms, bucket_s))


def video_ts(ms: int) -> str:
    """Milliseconds into the file -> 'HH:MM:SS.ff', the brief's format.

    Hundredths, not milliseconds: it is a human-readable offset for someone scrubbing to the
    moment in the recording, and `first_seen_ms` is kept separately for anything exact.
    """
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, rem = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{rem // 10:02d}"


def frame_to_ms(frame_index: int, fps: float) -> int:
    """Frame index -> milliseconds. Frame-INDEX based, never wall clock.

    This is the other half of the idempotency guarantee (BUILD_PLAN 0.2): a live run and the
    replay of its recording walk the same frame indices, so they agree on every timestamp.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    return int(round((frame_index / fps) * 1000.0))
