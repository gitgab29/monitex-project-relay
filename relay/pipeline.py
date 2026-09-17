"""Wiring: source -> record -> gate -> analyse -> state -> enrich -> persist -> route.

The one structural rule, from which everything else follows:

    **Recording is on the main loop. Everything expensive is not.**

Decode and write happen for every frame, unconditionally. Detection, the model call and the
network sinks happen on a worker thread fed by a bounded queue. A slow Gemini call therefore
delays an *event*; it can never drop a *frame*. Since the recording is the source of truth, a
delayed event can always be recovered by replaying the file.

Block 2 builds the capture half. Detection, the state machine and the sinks arrive in the
blocks that follow and plug into `_analyze`, which is deliberately the only place that knows
what "analysing a frame" means.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .capture.motion import MotionGate
from .capture.overlay import OverlayWindow, draw
from .capture.recorder import SessionRecorder
from .capture.source import FileSource, Frame, WebcamSource, open_source
from .config import Settings
from .detect.static_filter import StaticObjectFilter
from .ids import new_run_id, session_id_from_path, video_ts
from .schema import Observation
from .store import Store

log = logging.getLogger(__name__)

#: Bounded so a stalled worker applies backpressure we can see and count, rather than
#: growing until the process dies.
ANALYSIS_QUEUE_MAX = 8


@dataclass
class RunStats:
    frames_decoded: int = 0
    frames_analyzed: int = 0
    frames_skipped_quiet: int = 0
    frames_dropped_busy: int = 0
    events_inserted: int = 0
    events_skipped: int = 0
    vision_calls: int = 0
    vision_failures: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at


class Pipeline:
    def __init__(
        self, cfg: Settings, store: Store, *, mode: str, show: bool = False,
        max_seconds: float | None = None, chaos: str | None = None,
        detector=None, state_machine=None, enricher=None, router=None,
    ):
        self.cfg = cfg
        self.store = store
        self.mode = mode
        self.show = show
        self.max_seconds = max_seconds
        self.chaos = chaos
        # Injected in later blocks; None here means "capture only", which is exactly what
        # `--vision none --no-n8n` should do.
        self.detector = detector
        self.static_filter = StaticObjectFilter(cfg) if cfg.static_filter else None
        self.state_machine = state_machine
        self.enricher = enricher
        self.router = router

        self.stats = RunStats()
        self.gate = MotionGate(cfg.motion_threshold, cfg.heartbeat_s)
        self.run_id = new_run_id()
        self.session_id = ""
        self.source_file = ""
        self._q: queue.Queue = queue.Queue(maxsize=ANALYSIS_QUEUE_MAX)
        self._stop = threading.Event()
        self._latest_render: dict = {}
        self._render_lock = threading.Lock()
        self._last_obs: Observation | None = None
        #: Set by the CLI after construction; None means template summaries only.
        self.vision = None

    # ------------------------------------------------------------------ entry points

    def run_live(self) -> RunStats:
        cfg = self.cfg
        src = open_source(
            webcam_index=cfg.camera_index, width=cfg.frame_width,
            height=cfg.frame_height, nominal_fps=cfg.nominal_fps,
        )
        recorder = SessionRecorder(cfg.sessions_dir, src.fps, src.width, src.height)
        self.session_id = recorder.session_id
        self.source_file = str(recorder.path)
        self.store.add_session(
            recorder.session_id, src.source_name, str(recorder.path),
            src.fps, src.width, src.height, cfg.site_id,
        )
        return self._drive(src, recorder)

    def run_replay(self, path: str | Path, session_id: str | None = None) -> RunStats:
        cfg = self.cfg
        src = open_source(path=path, nominal_fps=cfg.nominal_fps)
        # The session id lives in the recording's filename, so a replay recovers it without
        # touching the database. A foreign file gets a stable id derived from its name, so
        # replaying the SAME file twice is still idempotent.
        self.session_id = session_id or session_id_from_path(str(path)) or f"file-{Path(path).stem}"
        self.source_file = str(path)
        self.store.add_session(
            self.session_id, src.source_name, str(path), src.fps, src.width, src.height, cfg.site_id,
        )
        return self._drive(src, recorder=None)

    # ------------------------------------------------------------------ shared loop

    def _drive(self, src: WebcamSource | FileSource, recorder: SessionRecorder | None) -> RunStats:
        cfg = self.cfg
        replay_of = self.store.start_run(
            self.run_id, self.session_id, self.mode, cfg.config_hash(), self.chaos
        )
        if replay_of:
            log.info("this run repeats session %s under the same config (first run %s)",
                     self.session_id, replay_of[:8])

        every = cfg.analyze_every(src.fps)
        log.info(
            "run %s | %s | %s | analysing every %d frames (~%.1f fps) | motion>=%.1f | heartbeat %.0fs",
            self.run_id[:8], self.mode, self.session_id, every, src.fps / every,
            cfg.motion_threshold, cfg.heartbeat_s,
        )

        worker = threading.Thread(target=self._worker, name="analysis", daemon=True)
        worker.start()
        window = OverlayWindow(enabled=self.show)
        status = "complete"

        try:
            for frame in src.frames():
                self.stats.frames_decoded += 1
                if recorder is not None:
                    recorder.write(frame.image)

                # Frame-INDEX sampling: live and replay pick identical frames.
                if frame.index % every == 0:
                    decision = self.gate.evaluate(frame.image, frame.video_ts_ms)
                    self._remember_render(decision)
                    if decision.analyze:
                        self.stats.frames_analyzed += 1
                        try:
                            self._q.put_nowait((frame, decision))
                        except queue.Full:
                            self.stats.frames_dropped_busy += 1
                            log.warning(
                                "analysis queue full at frame %d; skipping this analysis "
                                "(the frame is still recorded)", frame.index,
                            )
                    else:
                        self.stats.frames_skipped_quiet += 1
                        log.debug("QUIET f%d score=%.2f", frame.index, decision.score)

                # Rendered once and used twice: the desktop window, and the JPEG the dashboard
                # streams. Recording a demo should not mean juggling two windows.
                if self.show or self.cfg.live_preview:
                    rendered = self._render(frame)
                    if self.cfg.live_preview and frame.index % self.cfg.live_preview_every == 0:
                        self._publish_live(rendered)
                    if self.show:
                        window.show(rendered)
                        if window.should_quit():
                            log.info("quit requested from the overlay window")
                            break

                if self.max_seconds and frame.video_ts_ms >= self.max_seconds * 1000:
                    log.info("reached --max-seconds %.0f", self.max_seconds)
                    break
        except KeyboardInterrupt:
            log.info("interrupted; closing the recording cleanly")
            status = "interrupted"
        finally:
            self._stop.set()
            worker.join(timeout=self._drain_timeout_s())
            window.close()
            src.release()
            if recorder is not None:
                recorder.close()
                self.store.end_session(self.session_id)
            self._finish_states()
            self._flush_counters()
            self.store.end_run(self.run_id, status)

        self._log_summary()
        return self.stats

    # ------------------------------------------------------------------ worker

    def _worker(self) -> None:
        while not (self._stop.is_set() and self._q.empty()):
            try:
                frame, decision = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._analyze(frame, decision)
            except Exception:
                log.exception("analysis failed for frame %d", frame.index)
            finally:
                self._q.task_done()

    def _analyze(self, frame: Frame, decision) -> None:
        """One analysed frame. The only place that knows what analysis means.

        Block 2: build the Observation and persist it. Block 3 adds the detector and the
        state machine; Block 4 the model call; Block 5 the routing.
        """
        detections = self.detector.detect(frame.image) if self.detector else []
        # A photograph on the wall is a person-shaped region, and YOLO is right to box it.
        # Dropping it here rather than in the state machine keeps "what is in the frame"
        # honest for everything downstream, including the recorded observation.
        if self.static_filter is not None:
            detections = self.static_filter.apply(frame.image, detections)
        obs = Observation(
            session_id=self.session_id, frame_index=frame.index, video_ts_ms=frame.video_ts_ms,
            motion_score=decision.score, mean_luma=decision.mean_luma, detections=detections,
        )
        post_state = approach_state = None
        if self.state_machine is not None:
            drafts = self.state_machine.update(obs)
            post_state = self.state_machine.post_state
            approach_state = self.state_machine.approach_state
            for draft in drafts:
                draft.key_frame = frame.image
                if self.enricher is not None:
                    self.enricher(draft, obs, self)
        self._last_obs = obs
        self.store.add_observation(obs, post_state, approach_state)
        with self._render_lock:
            self._latest_render["detections"] = detections
            if post_state:
                self._latest_render["post_state"] = post_state
            if approach_state:
                self._latest_render["approach_state"] = approach_state

    # ------------------------------------------------------------------ rendering

    def set_banner(self, text: str) -> None:
        """Put the most recent event on the overlay. Called from the analysis thread, so it
        takes the same lock the renderer does."""
        with self._render_lock:
            self._latest_render["extra"] = text

    def _remember_render(self, decision) -> None:
        with self._render_lock:
            self._latest_render.update(
                motion_score=decision.score, gate_reason=decision.reason,
                mean_luma=decision.mean_luma, analyzed=decision.analyze,
            )

    def _publish_live(self, rendered) -> None:
        """Drop the annotated frame where the dashboard can pick it up.

        A file rather than a socket, because the camera runs in its own process: the API cannot
        reach into it for frames, and a file is the simplest thing that crosses that boundary.

        Written to a temp name and then replaced, so a reader never catches a half-written
        JPEG -- os.replace is atomic, and the alternative is an occasional torn frame on
        screen during the one recording that matters.

        Never raises. A preview that fails is a cosmetic problem; it must not take down a run.
        """
        try:
            import cv2

            path = self.cfg.live_frame_path
            tmp = path.with_suffix(".tmp.jpg")
            ok, buf = cv2.imencode(".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                return
            tmp.write_bytes(buf.tobytes())
            os.replace(tmp, path)
        except Exception:
            log.debug("could not publish the live frame", exc_info=True)

    def _render(self, frame: Frame):
        with self._render_lock:
            r = dict(self._latest_render)
        return draw(
            frame.image, post_zone=self.cfg.post_zone,
            detections=r.get("detections", ()),
            post_state=r.get("post_state", "UNKNOWN"),
            approach_state=r.get("approach_state", "CLEAR"),
            motion_score=r.get("motion_score", 0.0), gate_reason=r.get("gate_reason", ""),
            mean_luma=r.get("mean_luma", 0.0), analyzed=r.get("analyzed", False),
            video_ts=video_ts(frame.video_ts_ms), frame_index=frame.index,
            session_id=self.session_id, extra=r.get("extra", ""),
        )

    # ------------------------------------------------------------------ bookkeeping

    def _drain_timeout_s(self) -> float:
        """How long to let an in-flight vision call finish after the source ends.

        This used to be a flat 30 s, which was quietly shorter than the worst case it had to
        cover. A vision call that keeps timing out costs `vision_timeout_s` per attempt plus
        backoff between them -- at the defaults, 12 s x 3 + 1 s + 2 s = 39 s. So the join
        expired mid-retry, the worker was abandoned, and the event shipped WITHOUT its
        `vision_timeout` reason and WITHOUT its dead-letter row: the degrade path ran and then
        lost its own evidence. `--chaos timeout` on a short clip showed the backoff and then
        silently swallowed the ending.

        Deriving it from the retry budget means the two cannot drift apart again. The +5 s is
        for the encode and the store write after the last attempt.
        """
        cfg = self.cfg
        attempts = max(1, cfg.retry_attempts)
        backoff = sum(
            min(cfg.retry_cap_s, cfg.retry_base_s * (2 ** n)) for n in range(attempts - 1)
        )
        return attempts * cfg.vision_timeout_s + backoff + 5.0

    def _finish_states(self) -> None:
        """An occurrence still running when the video ends is still an occurrence; close it
        so its extent is recorded rather than left dangling."""
        if self.state_machine is None or self.enricher is None or self._last_obs is None:
            return
        try:
            for draft in self.state_machine.finish():
                self.enricher(draft, self._last_obs, self)
        except Exception:
            log.exception("failed to close out open states")

    def _flush_counters(self) -> None:
        s = self.stats
        for name, value in (
            ("frames_decoded", s.frames_decoded), ("frames_analyzed", s.frames_analyzed),
            ("frames_skipped_quiet", s.frames_skipped_quiet), ("vision_calls", s.vision_calls),
            ("vision_failures", s.vision_failures), ("events_inserted", s.events_inserted),
            ("events_skipped", s.events_skipped),
        ):
            if value:
                self.store.bump(self.run_id, name, value)
        self.store.conn.commit()

    def _log_summary(self) -> None:
        s = self.stats
        ratio = s.frames_analyzed / max(s.frames_decoded, 1)
        log.info(
            "run complete: %d decoded, %d analysed (%.1f%%), %d skipped quiet | "
            "%d events - %d inserted, %d skipped | %.1fs",
            s.frames_decoded, s.frames_analyzed, ratio * 100, s.frames_skipped_quiet,
            s.events_inserted + s.events_skipped, s.events_inserted, s.events_skipped, s.elapsed(),
        )
        if s.frames_dropped_busy:
            log.warning("%d analyses skipped because the worker was busy", s.frames_dropped_busy)
