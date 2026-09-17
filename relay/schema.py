"""The data contracts.

`Event` is exactly the schema the brief asks for, plus nothing. Provenance (which run
produced it, which frames, how the confidence was composed) lives in the database and in
`EventDraft`, never on the event itself -- an event that travels to n8n and into an email
should carry only what a dispatcher needs to act on.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Category(StrEnum):
    intrusion = "intrusion"
    vehicle = "vehicle"
    fire_smoke = "fire_smoke"
    loitering = "loitering"
    routine = "routine"
    other = "other"


class Priority(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


#: The four state transitions this feed can produce. `track_key` is one of these, which is
#: what makes an event id stable across a live run and the replay of its recording.
Observed = Literal[
    "post_manned",
    "post_unattended",
    "person_loitering_near_entry",
    "unidentified_person_at_post",
]

Zone = Literal["post", "approach"]


class Event(BaseModel):
    """The brief's schema. `extra='forbid'` so a typo becomes a test failure, not a silent
    field that never reaches the dispatcher."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    event_id: str
    source_type: Literal["video_event", "document"] = "video_event"
    source_file: str
    category: Category
    priority: Priority
    observed: str
    video_ts: str = Field(description="HH:MM:SS.ff offset into the source file")
    site_id: str | None = Field(
        default=None,
        description="Nullable on purpose: a missing site_id is a review trigger, not an error.",
    )
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = False


class Detection(BaseModel):
    """One YOLO person box, normalised to the frame so it is resolution-independent."""

    model_config = ConfigDict(extra="forbid")

    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = Field(ge=0.0, le=1.0)
    zone: Zone
    straddle: bool = False
    area_frac: float = Field(ge=0.0, le=1.0)
    zone_term: float = Field(ge=0.0, le=1.0)

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0


class Observation(BaseModel):
    """One analysed frame. Written to the `observations` table as the audit trail; the state
    machine consumes a stream of these and nothing else."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    frame_index: int
    video_ts_ms: int
    motion_score: float
    mean_luma: float
    detections: list[Detection] = Field(default_factory=list)

    @property
    def post(self) -> list[Detection]:
        return [d for d in self.detections if d.zone == "post"]

    @property
    def approach(self) -> list[Detection]:
        return [d for d in self.detections if d.zone == "approach"]

    @property
    def post_count(self) -> int:
        return len(self.post)

    @property
    def approach_count(self) -> int:
        return len(self.approach)

    @property
    def person_count(self) -> int:
        return len(self.detections)

    @property
    def yolo_max_conf(self) -> float:
        return max((d.conf for d in self.detections), default=0.0)


class VisionVerdict(BaseModel):
    """What the vision model is allowed to return.

    Deliberately small and closed. The model writes prose (the summary) and reports what it
    counts; it does not get to set the category, the priority or the final confidence. Those
    are rules, and rules are testable. The counts exist so we can *disagree* with the model:
    if it sees a different number of people than YOLO did, that is a review trigger.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(max_length=160)
    person_count: int = Field(ge=0, le=10)
    people_at_desk: int = Field(ge=0, le=10)
    lighting: Literal["good", "dim", "dark"]
    confidence: float = Field(ge=0.0, le=1.0)
