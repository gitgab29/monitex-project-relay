"""Block 0 smoke test: prove a camera opens and that YOLO sees a person in it.

    python scripts/smoke_webcam.py            # try indices 0,1,2
    python scripts/smoke_webcam.py --index 1  # just one

Run it SEATED at the desk, not standing: the seated, half-occluded case (R5) is the one
that actually has to work, and it is the one YOLO is most likely to miss.
"""

from __future__ import annotations

import argparse
import time

import cv2


def probe(index: int, model, imgsz: int, conf: float) -> bool:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"  camera {index}: could not open")
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # First grabs after open are often black while the sensor warms up.
    for _ in range(5):
        cap.read()
        time.sleep(0.05)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print(f"  camera {index}: opened but returned no frame")
        return False

    h, w = frame.shape[:2]
    mean_luma = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    print(f"  camera {index}: {w}x{h}, mean luma {mean_luma:.1f}")

    t0 = time.perf_counter()
    result = model.predict(frame, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000

    boxes = result.boxes
    n = 0 if boxes is None else len(boxes)
    print(f"  camera {index}: YOLO {ms:.0f} ms, {n} person detection(s)")
    for b in (boxes or []):
        x0, y0, x1, y1 = (float(v) for v in b.xyxyn[0])
        area = (x1 - x0) * (y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        print(
            f"    conf {float(b.conf[0]):.2f}  centre ({cx:.2f},{cy:.2f})  "
            f"area {area:.3f}  box ({x0:.2f},{y0:.2f})-({x1:.2f},{y1:.2f})"
        )

    out = f"data/smoke_cam{index}.jpg"
    cv2.imwrite(out, result.plot())
    print(f"  camera {index}: annotated frame -> {out}")
    return n > 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, default=None, help="probe only this camera index")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.30)
    args = ap.parse_args()

    import os

    os.makedirs("data", exist_ok=True)

    print(f"loading {args.model} (downloads on first run)...")
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = YOLO(args.model)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s\n")

    indices = [args.index] if args.index is not None else [0, 1, 2]
    hits = [i for i in indices if probe(i, model, args.imgsz, args.conf)]

    print()
    if hits:
        print(f"OK - person detected on camera index/indices {hits}. Set CAMERA_INDEX={hits[0]}.")
    else:
        print("No person detected on any index. Check you are in frame and lit, then re-run.")


if __name__ == "__main__":
    main()
