"""Settings, loaded from the environment, with a hash over the non-secret ones.

Everything tunable lives here and nowhere else, so `.env.example` is a complete and honest
description of the system's behaviour. Two rules the rest of the code depends on:

* `config_hash()` covers the settings that change *what the system decides* and excludes the
  ones that only change where things go or who to tell. That is what makes it meaningful to
  say "this replay ran under the same config as the original run".
* No secret is ever in the hash, in a log line, or in `__repr__`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from dotenv import load_dotenv


def _env(key: str, default: str) -> str:
    v = os.getenv(key)
    return default if v is None or v == "" else v


def _f(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError as e:
        raise ValueError(f"{key} must be a number, got {os.getenv(key)!r}") from e


def _i(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError as e:
        raise ValueError(f"{key} must be an integer, got {os.getenv(key)!r}") from e


def _b(key: str, default: bool) -> bool:
    return _env(key, "true" if default else "false").strip().lower() in {"1", "true", "yes", "on"}


def _rect(key: str, default: str) -> tuple[float, float, float, float]:
    raw = _env(key, default)
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError(f"{key} must be four comma-separated numbers, got {raw!r}")
    x0, y0, x1, y1 = (float(p) for p in parts)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"{key}={raw!r}: need 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return x0, y0, x1, y1


#: Settings that change what the system *decides*. A replay under a different value for any
#: of these is a genuinely different analysis, so it gets a different config_hash and is not
#: treated as a re-run of the original. Secrets and destinations are deliberately absent.
HASHED_FIELDS = (
    "analyze_fps", "heartbeat_s", "motion_threshold",
    "post_zone", "zone_margin", "post_min_area",
    "yolo_model", "yolo_conf", "yolo_imgsz",
    "manned_confirm_frames", "intrusion_confirm_frames",
    "unattended_dwell_s", "loiter_dwell_s", "exit_grace_s",
    "event_bucket_s",
    "review_conf_threshold", "review_conf_other", "review_conf_high",
)

SECRET_FIELDS = frozenset({"gemini_api_key", "smtp_app_password"})


@dataclass(frozen=True)
class Settings:
    # site & source
    site_id: str | None = None
    camera_index: int = 1
    frame_width: int = 1280
    frame_height: int = 720
    nominal_fps: float = 30.0

    # sampling
    analyze_fps: float = 1.0
    heartbeat_s: float = 5.0
    motion_threshold: float = 4.0

    # zones
    post_zone: tuple[float, float, float, float] = (0.15, 0.35, 0.85, 1.00)
    zone_margin: float = 0.04
    post_min_area: float = 0.10

    # detector
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.30
    yolo_imgsz: int = 640

    # state machine
    manned_confirm_frames: int = 2
    intrusion_confirm_frames: int = 2
    unattended_dwell_s: float = 8.0
    loiter_dwell_s: float = 6.0
    exit_grace_s: float = 3.0

    # event identity
    event_bucket_s: int = 5
    dedupe_window_s: float = 5.0

    # review thresholds
    review_conf_threshold: float = 0.75
    review_conf_other: float = 0.85
    review_conf_high: float = 0.90

    # vision
    # --- static-object suppression (pictures, posters, screens) -------------------
    static_filter: bool = True
    static_min_frames: int = 20
    static_pixel_eps: float = 2.5
    static_match_iou: float = 0.85

    #: Free-text description of whoever is SUPPOSED to be at this post, e.g.
    #: "wears glasses and over-ear headphones". Given to the model so it can tell the
    #: assigned officer from a stranger. Empty = every person at the post is equally
    #: unidentified, which is the safe default for a site that has not configured it.
    expected_occupant: str = ""

    vision_backend: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    vision_timeout_s: float = 12.0
    vision_min_interval_s: float = 4.0
    vision_jpeg_quality: int = 80
    vision_max_width: int = 960

    # reliability
    retry_attempts: int = 3
    retry_base_s: float = 1.0
    retry_cap_s: float = 8.0

    # sinks
    n8n_webhook_url: str = "http://localhost:5678/webhook/relay-event"
    n8n_timeout_s: float = 10.0
    #: Separate, short CONNECT budget. If n8n is up, connecting is instant; if it is
    #: not, Windows retries the SYN and a refusal can take seconds -- which is pure
    #: delay in front of the fallback that was always going to deliver the alert.
    n8n_connect_timeout_s: float = 1.0
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_app_password: str = ""
    alert_to: str = ""
    webhook_fallback_url: str = ""
    sheets_enabled: bool = False

    # api
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_base: str = "http://host.docker.internal:8080"
    review_link_base: str = "http://localhost:8080"

    # storage
    data_dir: Path = field(default=Path("data"))
    db_path: Path = field(default=Path("data/relay.db"))
    log_level: str = "INFO"

    # ---------------------------------------------------------------- derived

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def evidence_dir(self) -> Path:
        """One JPEG per event -- the frame the event was actually created from.

        A summary saying "a person is at the post" is a claim; the frame is the evidence for
        it. Kept out of the database on purpose: SQLite is for rows people query, not blobs.
        """
        return self.data_dir / "evidence"

    def analyze_every(self, fps: float) -> int:
        """How many decoded frames per analysed frame, at this source's fps.

        Frame-index based on purpose: a live run and its replay then analyse *identical*
        frame indices, so their transitions land at the same video_ts and hash the same.
        """
        if fps <= 0:
            fps = self.nominal_fps
        return max(1, int(round(fps / max(self.analyze_fps, 1e-6))))

    def review_threshold_for(self, category: str, priority: str) -> float:
        """The confidence bar, which differs by what a wrong answer would cost.

        A false `routine` row costs nothing; a false high-priority page wakes a supervisor at
        3 a.m. The strictest applicable bar wins.
        """
        t = self.review_conf_threshold
        if category == "other":
            t = max(t, self.review_conf_other)
        if priority == "high":
            t = max(t, self.review_conf_high)
        return t

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.sessions_dir, self.logs_dir, self.evidence_dir):
            d.mkdir(parents=True, exist_ok=True)

    def config_hash(self) -> str:
        """Stable digest of the decision-making settings. Never includes a secret."""
        payload = {k: getattr(self, k) for k in HASHED_FIELDS}
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def redacted(self) -> dict[str, object]:
        """Everything, safe to log: secrets become a marker, never a value."""
        out: dict[str, object] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = ("<set>" if v else "<unset>") if f.name in SECRET_FIELDS else v
        return out

    def __repr__(self) -> str:
        return (
            f"Settings(site_id={self.site_id!r}, config_hash={self.config_hash()}, "
            f"backend={self.vision_backend!r})"
        )


def load_settings(env_file: str | os.PathLike[str] | None = ".env", *, override: bool = False) -> Settings:
    """Read `.env` (if present) then the process environment, and validate.

    Validation is eager and loud: a typo in a dwell time should stop the run at second zero,
    not produce a subtly wrong event forty minutes into a demo.
    """
    if env_file is not None and Path(env_file).exists():
        load_dotenv(env_file, override=override)

    site = _env("SITE_ID", "")
    data_dir = Path(_env("DATA_DIR", "data"))

    s = Settings(
        site_id=site or None,
        camera_index=_i("CAMERA_INDEX", 1),
        frame_width=_i("FRAME_WIDTH", 1280),
        frame_height=_i("FRAME_HEIGHT", 720),
        nominal_fps=_f("NOMINAL_FPS", 30.0),
        analyze_fps=_f("ANALYZE_FPS", 1.0),
        heartbeat_s=_f("HEARTBEAT_S", 5.0),
        motion_threshold=_f("MOTION_THRESHOLD", 4.0),
        post_zone=_rect("POST_ZONE", "0.15,0.35,0.85,1.00"),
        zone_margin=_f("ZONE_MARGIN", 0.04),
        post_min_area=_f("POST_MIN_AREA", 0.10),
        yolo_model=_env("YOLO_MODEL", "yolov8n.pt"),
        yolo_conf=_f("YOLO_CONF", 0.30),
        yolo_imgsz=_i("YOLO_IMGSZ", 640),
        manned_confirm_frames=_i("MANNED_CONFIRM_FRAMES", 2),
        intrusion_confirm_frames=_i("INTRUSION_CONFIRM_FRAMES", 2),
        unattended_dwell_s=_f("UNATTENDED_DWELL_S", 8.0),
        loiter_dwell_s=_f("LOITER_DWELL_S", 6.0),
        exit_grace_s=_f("EXIT_GRACE_S", 3.0),
        event_bucket_s=_i("EVENT_BUCKET_S", 5),
        dedupe_window_s=_f("DEDUPE_WINDOW_S", 5.0),
        review_conf_threshold=_f("REVIEW_CONF_THRESHOLD", 0.75),
        review_conf_other=_f("REVIEW_CONF_OTHER", 0.85),
        review_conf_high=_f("REVIEW_CONF_HIGH", 0.90),
        static_filter=_b("STATIC_FILTER", True),
        static_min_frames=_i("STATIC_MIN_FRAMES", 20),
        static_pixel_eps=_f("STATIC_PIXEL_EPS", 2.5),
        static_match_iou=_f("STATIC_MATCH_IOU", 0.85),
        expected_occupant=_env("EXPECTED_OCCUPANT", "").strip(),
        vision_backend=_env("VISION_BACKEND", "gemini").strip().lower(),
        gemini_api_key=_env("GEMINI_API_KEY", ""),
        gemini_model=_env("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        vision_timeout_s=_f("VISION_TIMEOUT_S", 12.0),
        vision_min_interval_s=_f("VISION_MIN_INTERVAL_S", 4.0),
        vision_jpeg_quality=_i("VISION_JPEG_QUALITY", 80),
        vision_max_width=_i("VISION_MAX_WIDTH", 960),
        retry_attempts=_i("RETRY_ATTEMPTS", 3),
        retry_base_s=_f("RETRY_BASE_S", 1.0),
        retry_cap_s=_f("RETRY_CAP_S", 8.0),
        n8n_webhook_url=_env("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/relay-event"),
        n8n_timeout_s=_f("N8N_TIMEOUT_S", 10.0),
        n8n_connect_timeout_s=_f("N8N_CONNECT_TIMEOUT_S", 1.0),
        smtp_host=_env("SMTP_HOST", "smtp.gmail.com"),
        smtp_port=_i("SMTP_PORT", 587),
        smtp_user=_env("SMTP_USER", ""),
        smtp_app_password=_env("SMTP_APP_PASSWORD", ""),
        alert_to=_env("ALERT_TO", ""),
        webhook_fallback_url=_env("WEBHOOK_FALLBACK_URL", ""),
        sheets_enabled=_b("SHEETS_ENABLED", False),
        api_host=_env("API_HOST", "0.0.0.0"),
        api_port=_i("API_PORT", 8080),
        api_base=_env("API_BASE", "http://host.docker.internal:8080").rstrip("/"),
        review_link_base=_env("REVIEW_LINK_BASE", "http://localhost:8080").rstrip("/"),
        data_dir=data_dir,
        db_path=Path(_env("DB_PATH", str(data_dir / "relay.db"))),
        log_level=_env("LOG_LEVEL", "INFO").upper(),
    )
    _validate(s)
    return s


#: Settings whose value is a credential or an address someone pastes in by hand.
CREDENTIALISH = ("gemini_api_key", "smtp_user", "smtp_app_password", "alert_to", "webhook_fallback_url")


def _validate(s: Settings) -> None:
    # `KEY=   # explanation` in a .env parses the COMMENT as the value when the key is
    # otherwise empty, so a blank credential silently becomes the string '# explanation'
    # and you find out when SMTP auth fails mid-demo. Catch it at load time instead.
    for name in CREDENTIALISH:
        v = getattr(s, name)
        if isinstance(v, str) and v.startswith("#"):
            raise ValueError(
                f"{name.upper()} looks like a comment, not a value ({v[:40]!r}...). "
                "Put the comment on its own line above the key in .env."
            )
    if s.vision_backend not in {"gemini", "fake", "none"}:
        raise ValueError(f"VISION_BACKEND must be gemini, fake or none, got {s.vision_backend!r}")
    if s.analyze_fps <= 0:
        raise ValueError("ANALYZE_FPS must be > 0")
    if s.motion_threshold < 0:
        raise ValueError("MOTION_THRESHOLD must be >= 0")
    if not 0 <= s.yolo_conf <= 1:
        raise ValueError("YOLO_CONF must be between 0 and 1")
    for name in ("review_conf_threshold", "review_conf_other", "review_conf_high"):
        v = getattr(s, name)
        if not 0 <= v <= 1:
            raise ValueError(f"{name.upper()} must be between 0 and 1, got {v}")
    if s.event_bucket_s <= 0:
        raise ValueError("EVENT_BUCKET_S must be > 0")
    if s.retry_attempts < 1:
        raise ValueError("RETRY_ATTEMPTS must be >= 1")
    for name in ("unattended_dwell_s", "loiter_dwell_s", "exit_grace_s", "heartbeat_s"):
        if getattr(s, name) < 0:
            raise ValueError(f"{name.upper()} must be >= 0")
