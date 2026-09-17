"""Session recording. Every decoded frame goes to disk, unconditionally.

The recording is the source of truth, which is what makes the rest of the design safe:

* A slow Gemini call delays an *event*, never a frame. Analysis runs on a worker; recording
  does not wait for it.
* Anything the system got wrong can be re-run against the exact same input, because the input
  still exists. "Re-run the pipeline over this session" is a real operation, not a promise.
* The session id lives in the filename, so a replay recovers it with no database at all.

Written at the source's nominal fps so the file's index-to-timestamp mapping matches the
live run's exactly -- see capture/source.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..ids import new_session_id

log = logging.getLogger(__name__)


class SessionRecorder:
    """Owns the session id and the MP4. Opened lazily on the first frame so a run that fails
    to get any frames does not leave a 0-byte file behind."""

    def __init__(
        self, out_dir: str | Path, fps: float, width: int, height: int,
        session_id: str | None = None, fourcc: str = "mp4v",
    ):
        self.session_id = session_id or new_session_id()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"{self.session_id}.mp4"
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.fourcc = fourcc
        self.frames_written = 0
        self._writer: cv2.VideoWriter | None = None

    def _open(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (self.width, self.height))
        if not self._writer.isOpened():
            raise RuntimeError(
                f"could not open a video writer for {self.path} "
                f"({self.fourcc} @ {self.fps} fps, {self.width}x{self.height})"
            )
        log.info(
            "recording session %s -> %s (%dx%d @ %.1f fps)",
            self.session_id, self.path, self.width, self.height, self.fps,
        )

    def write(self, image: np.ndarray) -> None:
        if self._writer is None:
            self._open()
        assert self._writer is not None
        h, w = image.shape[:2]
        # A camera can hand back a size other than the one it was asked for. Resizing keeps
        # the file playable rather than silently producing a corrupt stream.
        if (w, h) != (self.width, self.height):
            image = cv2.resize(image, (self.width, self.height))
        self._writer.write(image)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            log.info("recorded %d frames -> %s", self.frames_written, self.path)

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
