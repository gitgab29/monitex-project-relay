"""Person detection: YOLOv8n, CPU, one class.

Stage 1 of the cascade. Its job is narrow on purpose -- find people and say where they are.
It does not classify the situation; that is the state machine's job, and the state machine is
testable without a camera. Keeping the boundary here is what makes the whole thing reviewable.

`classes=[0]` is COCO's 'person'. Filtering inside the model is cheaper than detecting 80
classes and discarding 79.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

import numpy as np

from ..config import Settings
from ..schema import Detection

log = logging.getLogger(__name__)


class Detector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[Detection]: ...


class YoloPersonDetector:
    """Ultralytics YOLO, wrapped so the rest of the system sees only `Detection`.

    Zone assignment is injected rather than done here: the detector reports geometry, the
    ZoneModel decides what that geometry means for this site. Swapping the zone shape then
    touches one file, and the detector stays a pure function of the image.
    """

    name = "yolov8n"

    def __init__(self, cfg: Settings, zones=None):
        from ultralytics import YOLO

        self.cfg = cfg
        self.zones = zones
        t0 = time.perf_counter()
        self.model = YOLO(cfg.yolo_model)
        log.info("loaded %s in %.1fs", cfg.yolo_model, time.perf_counter() - t0)
        self._warmup()

    def _warmup(self) -> None:
        """Run one throwaway inference at startup.

        Measured on this machine: the first call costs ~24 s (lazy init, graph setup) against
        ~43 ms warm. Without this the first real frame of a live run stalls for half a minute
        and the operator concludes the thing is broken. Better to pay it before the camera
        opens, where it is honest startup cost.
        """
        t0 = time.perf_counter()
        blank = np.zeros((self.cfg.yolo_imgsz, self.cfg.yolo_imgsz, 3), dtype=np.uint8)
        self.model.predict(blank, classes=[0], conf=self.cfg.yolo_conf,
                           imgsz=self.cfg.yolo_imgsz, verbose=False)
        log.info("detector warm-up took %.1fs (first inference is the expensive one)",
                 time.perf_counter() - t0)

    def detect(self, image: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        result = self.model.predict(
            image, classes=[0], conf=self.cfg.yolo_conf,
            imgsz=self.cfg.yolo_imgsz, verbose=False,
        )[0]
        ms = (time.perf_counter() - t0) * 1000

        out: list[Detection] = []
        boxes = result.boxes
        if boxes is not None:
            for b in boxes:
                x0, y0, x1, y1 = (float(v) for v in b.xyxyn[0])
                conf = float(b.conf[0])
                area_frac = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
                centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
                if self.zones is not None:
                    zone, straddle, zone_term = self.zones.classify(centre, area_frac)
                else:
                    zone, straddle, zone_term = "post", False, 1.0
                out.append(Detection(
                    x0=x0, y0=y0, x1=x1, y1=y1, conf=conf, zone=zone,
                    straddle=straddle, area_frac=min(area_frac, 1.0), zone_term=zone_term,
                ))
        log.debug("yolo %.0f ms, %d person(s)", ms, len(out))
        return out


class ScriptedDetector:
    """A detector that replays a fixed list of frames' detections.

    Lets the state machine and the pipeline be tested end to end with no camera, no model and
    no randomness -- which is how the four scenarios get covered without a second person.
    """

    name = "scripted"

    def __init__(self, frames: list[list[Detection]]):
        self.frames = frames
        self.i = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        out = self.frames[self.i] if self.i < len(self.frames) else []
        self.i += 1
        return list(out)
