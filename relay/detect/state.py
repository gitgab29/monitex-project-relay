"""The state machine: turn a stream of per-frame observations into a few meaningful events.

This is the piece that decides *what happened*, and it is deliberately the piece with no
model, no network and no camera in it -- so every scenario in the demo can be proven by a unit
test on a synthetic sequence, including the ones that need a second person to film.

Two design rules do most of the work:

**Events fire on transitions, not on frames.** A guard sitting at a desk for an hour is one
`post_manned` event, not 3,600 rows. What a dispatcher needs to know is when something
*changed*.

**Entering a state must be confirmed; leaving it is forgiving.** A single-frame YOLO miss is
common and means nothing, so MANNED needs two consecutive frames to start and the post must be
continuously empty for UNATTENDED_DWELL_S before absence is believed. Without that hysteresis
the system flaps between states and pages a human each time.

All timing comes from `video_ts_ms`, which is derived from the frame index, so a replay
reproduces the same transitions at the same timestamps and therefore the same event ids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..capture.motion import luma_term
from ..config import Settings
from ..schema import Observation

log = logging.getLogger(__name__)

PostState = Literal["UNKNOWN", "MANNED", "UNATTENDED", "INTRUSION"]
ApproachState = Literal["CLEAR", "PRESENT", "LOITERING"]

#: state -> the `observed` value it reports. States absent from this map are not reportable
#: (UNKNOWN is a startup condition, PRESENT is merely "someone is walking past").
OBSERVED_FOR = {
    "MANNED": "post_manned",
    "UNATTENDED": "post_unattended",
    "INTRUSION": "unidentified_person_at_post",
    "LOITERING": "person_loitering_near_entry",
}


@dataclass
class EventDraft:
    """What the state machine emits. `rules.py` turns it into an `Event`."""

    observed: str
    first_seen_ms: int
    last_seen_ms: int
    frame_count: int
    yolo_term: float
    zone_term: float
    luma_term: float
    straddle: bool = False
    kind: Literal["open", "close"] = "open"
    key_frame: np.ndarray | None = field(default=None, repr=False)


class _Track:
    """Bookkeeping for one occupied state: when it started, how long it has run."""

    def __init__(self, started_ms: int):
        self.first_seen_ms = started_ms
        self.last_seen_ms = started_ms
        self.frame_count = 1
        self.yolo_terms: list[float] = []
        self.zone_terms: list[float] = []
        self.luma_terms: list[float] = []
        self.straddle = False

    def touch(self, ts: int) -> None:
        self.last_seen_ms = ts
        self.frame_count += 1

    def observe(self, yolo: float, zone: float, luma: float, straddle: bool) -> None:
        self.yolo_terms.append(yolo)
        self.zone_terms.append(zone)
        self.luma_terms.append(luma)
        self.straddle = self.straddle or straddle

    def terms(self) -> tuple[float, float, float]:
        """Best evidence seen during the occurrence, not the latest frame's.

        An event is a claim about the whole occurrence, so it should be judged on the clearest
        look the system got -- otherwise a guard glancing away on the final frame would drag
        down the confidence of an event that was obvious throughout.
        """
        best = lambda xs: max(xs) if xs else 0.0  # noqa: E731
        return best(self.yolo_terms), best(self.zone_terms), best(self.luma_terms)


class PostStateMachine:
    """Consumes `Observation`s in video-time order; emits drafts on transitions.

    Two independent machines share one update call because they share one input:

    * **post zone** UNKNOWN -> MANNED -> UNATTENDED, with INTRUSION cutting across both.
    * **approach zone** CLEAR -> PRESENT -> LOITERING.

    They are independent on purpose: someone loitering by the entrance while the desk is
    manned is two facts about one frame, and collapsing them into one state would lose one.
    """

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.post_state: PostState = "UNKNOWN"
        self.approach_state: ApproachState = "CLEAR"
        self._post_track: _Track | None = None
        self._approach_track: _Track | None = None
        self._streak_one = 0       # consecutive frames with exactly one person at the post
        self._streak_two = 0       # consecutive frames with two or more
        self._empty_since: int | None = None
        self._approach_empty_since: int | None = None

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _evidence(dets, mean_luma: float) -> tuple[float, float, float, bool]:
        yolo = max((d.conf for d in dets), default=0.0)
        zone = max((d.zone_term for d in dets), default=1.0)
        return yolo, zone, luma_term(mean_luma), any(d.straddle for d in dets)

    def _open(self, state: str, ts: int, obs: Observation, dets) -> EventDraft | None:
        observed = OBSERVED_FOR.get(state)
        if observed is None:
            return None
        yolo, zone, lu, straddle = self._evidence(dets, obs.mean_luma)
        # An empty post is an absence: there is no detection to be confident about, so the
        # evidence is the detector reporting nothing while the frame is well lit. Scoring it
        # off a box that does not exist would be meaningless.
        if state == "UNATTENDED":
            yolo, zone, straddle = 0.90, 1.0, False
        return EventDraft(
            observed=observed, first_seen_ms=ts, last_seen_ms=ts, frame_count=1,
            yolo_term=yolo, zone_term=zone, luma_term=lu, straddle=straddle, kind="open",
        )

    @staticmethod
    def _close(track: _Track | None, state: str) -> EventDraft | None:
        observed = OBSERVED_FOR.get(state)
        if observed is None or track is None:
            return None
        y, z, lu = track.terms()
        return EventDraft(
            observed=observed, first_seen_ms=track.first_seen_ms, last_seen_ms=track.last_seen_ms,
            frame_count=track.frame_count, yolo_term=y, zone_term=z, luma_term=lu,
            straddle=track.straddle, kind="close",
        )

    def _start_track(self, ts: int, dets, obs: Observation, unattended: bool = False) -> _Track:
        t = _Track(ts)
        if unattended:
            t.observe(0.90, 1.0, luma_term(obs.mean_luma), False)
        else:
            t.observe(*self._evidence(dets, obs.mean_luma))
        return t

    # ------------------------------------------------------------------ update

    def update(self, obs: Observation) -> list[EventDraft]:
        ts = obs.video_ts_ms
        return self._update_post(obs, ts) + self._update_approach(obs, ts)

    def _update_post(self, obs: Observation, ts: int) -> list[EventDraft]:
        cfg = self.cfg
        n, dets = obs.post_count, obs.post
        out: list[EventDraft] = []

        self._streak_one = self._streak_one + 1 if n == 1 else 0
        self._streak_two = self._streak_two + 1 if n >= 2 else 0
        if n == 0:
            self._empty_since = ts if self._empty_since is None else self._empty_since
        else:
            self._empty_since = None

        # Keep the running occurrence's evidence current before deciding anything.
        if self._post_track is not None and self.post_state in OBSERVED_FOR:
            self._post_track.touch(ts)
            if dets:
                self._post_track.observe(*self._evidence(dets, obs.mean_luma))
            elif self.post_state == "UNATTENDED":
                self._post_track.observe(0.90, 1.0, luma_term(obs.mean_luma), False)

        target: PostState | None = None
        if self._streak_two >= cfg.intrusion_confirm_frames and self.post_state != "INTRUSION":
            target = "INTRUSION"
        elif n == 1 and self._streak_one >= cfg.manned_confirm_frames and self.post_state != "MANNED":
            target = "MANNED"
        elif (
            n == 0
            and self._empty_since is not None
            and ts - self._empty_since >= cfg.unattended_dwell_s * 1000
            and self.post_state != "UNATTENDED"
        ):
            # Absence is believed only after a continuous dwell, which is what stops a guard
            # leaning out of frame for one analysed frame from paging anyone.
            target = "UNATTENDED"

        if target is not None:
            closed = self._close(self._post_track, self.post_state)
            if closed:
                out.append(closed)
            log.info("post %s -> %s at %d ms (%d in zone)", self.post_state, target, ts, n)
            self.post_state = target
            opened = self._open(target, ts, obs, dets)
            self._post_track = self._start_track(ts, dets, obs, unattended=(target == "UNATTENDED"))
            if opened:
                out.append(opened)
        return out

    def _update_approach(self, obs: Observation, ts: int) -> list[EventDraft]:
        cfg = self.cfg
        n, dets = obs.approach_count, obs.approach
        out: list[EventDraft] = []

        if n > 0:
            self._approach_empty_since = None
            if self.approach_state == "CLEAR":
                self.approach_state = "PRESENT"
                self._approach_track = self._start_track(ts, dets, obs)
                return out

            assert self._approach_track is not None
            self._approach_track.touch(ts)
            self._approach_track.observe(*self._evidence(dets, obs.mean_luma))
            if (
                self.approach_state == "PRESENT"
                and ts - self._approach_track.first_seen_ms >= cfg.loiter_dwell_s * 1000
            ):
                log.info("approach PRESENT -> LOITERING at %d ms", ts)
                self.approach_state = "LOITERING"
                opened = self._open("LOITERING", ts, obs, dets)
                if opened:
                    # The occurrence began when they arrived, not when the timer expired --
                    # otherwise the event id would depend on the dwell setting.
                    opened.first_seen_ms = self._approach_track.first_seen_ms
                    opened.last_seen_ms = ts
                    opened.frame_count = self._approach_track.frame_count
                    out.append(opened)
            return out

        if self._approach_empty_since is None:
            self._approach_empty_since = ts
        # A grace period rather than an immediate reset: at ~1 fps a single missed detection
        # is routine, and treating it as "they left" would restart the dwell timer, so a
        # loiterer who flickers once would never be confirmed.
        elif ts - self._approach_empty_since >= cfg.exit_grace_s * 1000:
            if self.approach_state == "LOITERING":
                closed = self._close(self._approach_track, "LOITERING")
                if closed:
                    out.append(closed)
            if self.approach_state != "CLEAR":
                log.info("approach %s -> CLEAR at %d ms", self.approach_state, ts)
            self.approach_state = "CLEAR"
            self._approach_track = None
        return out

    def finish(self) -> list[EventDraft]:
        """Close whatever is still open when the stream ends, so no event is left without an
        extent just because the run stopped."""
        out = []
        for track, state in (
            (self._post_track, self.post_state),
            (self._approach_track, self.approach_state),
        ):
            closed = self._close(track, state)
            if closed:
                out.append(closed)
        return out
