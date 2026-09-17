"""Frame sources: a live webcam and a recorded file, behind one interface.

The single most important property here is that **timestamps come from the frame index, not
the wall clock**:

    video_ts_ms = round(frame_index / fps * 1000)

A live run and the replay of its own recording therefore walk the same indices and agree on
every timestamp, which is what lets `event_id` match across the two. Timing analysis off
`time.time()` instead would put the same transition at a different `video_ts` on replay, mint
a different id, and duplicate the event -- the bug BUILD_PLAN 0.2 exists to prevent.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

#: A camera reports garbage fps often enough that we need a sanity band.
FPS_MIN, FPS_MAX = 1.0, 240.0


@dataclass(frozen=True)
class Frame:
    index: int
    video_ts_ms: int
    image: np.ndarray


class FrameSource(Protocol):
    fps: float
    width: int
    height: int
    source_name: str

    def frames(self) -> Iterator[Frame]: ...
    def release(self) -> None: ...


def _sane_fps(reported: float, fallback: float) -> float:
    if reported is None or not (FPS_MIN <= reported <= FPS_MAX):
        log.debug("source reported fps=%r, using %.1f", reported, fallback)
        return fallback
    return float(reported)


class WebcamSource:
    """Live camera. `fps` is what the driver claims, sanity-checked.

    The claimed rate is what the recorder writes into the MP4 header, so replay of that file
    reproduces the same index-to-timestamp mapping even if the true capture rate drifted.
    That consistency matters more here than absolute accuracy.
    """

    def __init__(self, index: int, width: int = 1280, height: int = 720, nominal_fps: float = 30.0):
        self.index = index
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"could not open camera index {index}. Another app may hold it "
                f"(Teams/Zoom), or try a different CAMERA_INDEX."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        self.fps = _sane_fps(self.cap.get(cv2.CAP_PROP_FPS), nominal_fps)
        self.source_name = f"webcam:{index}"
        self._stop = False
        log.info("camera %d opened: %dx%d @ %.1f fps", index, self.width, self.height, self.fps)

    def stop(self) -> None:
        self._stop = True

    def frames(self) -> Iterator[Frame]:
        i = 0
        # The first few frames after opening are often black while the sensor settles;
        # analysing them would report a dark scene that was never real.
        for _ in range(5):
            self.cap.read()
        while not self._stop:
            ok, img = self.cap.read()
            if not ok or img is None:
                log.warning("camera returned no frame at index %d; stopping", i)
                break
            yield Frame(index=i, video_ts_ms=int(round(i / self.fps * 1000)), image=img)
            i += 1

    def release(self) -> None:
        self.cap.release()


class FileSource:
    """A recorded session. Deterministic: same file, same frames, same indices, every time."""

    def __init__(self, path: str | Path, nominal_fps: float = 30.0):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no such video: {self.path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video: {self.path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = _sane_fps(self.cap.get(cv2.CAP_PROP_FPS), nominal_fps)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.source_name = f"file:{self.path.as_posix()}"
        self._stop = False
        log.info(
            "opened %s: %dx%d @ %.1f fps, %d frames",
            self.path.name, self.width, self.height, self.fps, self.frame_count,
        )

    def stop(self) -> None:
        self._stop = True

    def frames(self) -> Iterator[Frame]:
        i = 0
        while not self._stop:
            ok, img = self.cap.read()
            if not ok or img is None:
                break
            yield Frame(index=i, video_ts_ms=int(round(i / self.fps * 1000)), image=img)
            i += 1

    def release(self) -> None:
        self.cap.release()


def open_source(
    *, webcam_index: int | None = None, path: str | Path | None = None,
    width: int = 1280, height: int = 720, nominal_fps: float = 30.0,
) -> WebcamSource | FileSource:
    if path is not None:
        return FileSource(path, nominal_fps=nominal_fps)
    if webcam_index is None:
        raise ValueError("open_source needs either a path or a webcam_index")
    return WebcamSource(webcam_index, width=width, height=height, nominal_fps=nominal_fps)
