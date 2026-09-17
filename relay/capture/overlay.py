"""The `--show` window: zones, boxes, states and the gate's decision, drawn on the frame.

This exists to make the system's reasoning visible while it runs. On the demo video you can
see *why* a detection counted as being at the post rather than approaching it, and watch the
motion score sit near zero on a still scene. It is also how POST_ZONE gets tuned: draw it,
look at it, edit .env, restart.

Nothing here feeds a decision -- it renders onto a copy and returns it.
"""

from __future__ import annotations

import cv2
import numpy as np

# BGR. Post zone green, approach amber, straddle/ambiguous magenta so it is unmissable.
C_POST = (120, 220, 120)
C_APPROACH = (60, 180, 245)
C_STRADDLE = (220, 90, 220)
C_TEXT = (245, 245, 245)
C_SHADE = (24, 24, 24)
C_ALERT = (70, 70, 240)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img: np.ndarray, text: str, x: int, y: int, colour=C_TEXT, scale: float = 0.5) -> None:
    """Text with a filled backing box -- unreadable white-on-white otherwise.

    Clamped to the frame: a box near the right edge would otherwise push its own label off
    screen, and the straddle labels are exactly the ones you need to read.
    """
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    h, w = img.shape[:2]
    x = max(2, min(int(x), w - tw - 6))
    y = max(th + 4, min(int(y), h - base - 2))
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 4, y + base), C_SHADE, -1)
    cv2.putText(img, text, (x, y), FONT, scale, colour, 1, cv2.LINE_AA)


def draw(
    image: np.ndarray, *, post_zone: tuple[float, float, float, float],
    detections=(), post_state: str = "UNKNOWN", approach_state: str = "CLEAR",
    motion_score: float = 0.0, gate_reason: str = "", mean_luma: float = 0.0,
    video_ts: str = "", frame_index: int = 0, analyzed: bool = False,
    session_id: str = "", extra: str = "",
) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]

    # --- the post zone rectangle, tinted so the boundary is obvious ---------------
    x0, y0, x1, y1 = post_zone
    px0, py0, px1, py1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    tint = out.copy()
    cv2.rectangle(tint, (px0, py0), (px1, py1), C_POST, -1)
    cv2.addWeighted(tint, 0.12, out, 0.88, 0, out)
    cv2.rectangle(out, (px0, py0), (px1, py1), C_POST, 2)
    _label(out, "POST ZONE", px0 + 6, py0 + 18, C_POST)
    _label(out, "APPROACH ZONE", 8, 18, C_APPROACH)

    # --- detections --------------------------------------------------------------
    for d in detections:
        colour = C_STRADDLE if getattr(d, "straddle", False) else (C_POST if d.zone == "post" else C_APPROACH)
        bx0, by0 = int(d.x0 * w), int(d.y0 * h)
        bx1, by1 = int(d.x1 * w), int(d.y1 * h)
        cv2.rectangle(out, (bx0, by0), (bx1, by1), colour, 2)
        cx, cy = int((d.x0 + d.x1) / 2 * w), int((d.y0 + d.y1) / 2 * h)
        cv2.circle(out, (cx, cy), 4, colour, -1)  # membership is by centre; show the centre
        tag = f"{d.zone} {d.conf:.2f}"
        if getattr(d, "straddle", False):
            tag += f" STRADDLE area={d.area_frac:.2f}"
        _label(out, tag, bx0, max(by0 - 6, 14), colour)

    # --- status band -------------------------------------------------------------
    band_h = 58
    cv2.rectangle(out, (0, h - band_h), (w, h), C_SHADE, -1)
    post_colour = C_ALERT if post_state in {"UNATTENDED", "INTRUSION"} else C_POST
    _label(out, f"POST: {post_state}", 10, h - band_h + 20, post_colour, 0.6)
    appr_colour = C_ALERT if approach_state == "LOITERING" else C_APPROACH
    _label(out, f"APPROACH: {approach_state}", 210, h - band_h + 20, appr_colour, 0.6)

    mark = "ANALYZED" if analyzed else "skipped"
    gate = f"motion {motion_score:6.2f}  luma {mean_luma:5.1f}  [{mark}{'/' + gate_reason if gate_reason else ''}]"
    _label(out, gate, 10, h - band_h + 44, C_TEXT, 0.48)
    _label(out, f"{video_ts}  f{frame_index}", w - 250, h - band_h + 44, C_TEXT, 0.48)
    if session_id:
        _label(out, session_id, w - 250, h - band_h + 20, C_TEXT, 0.45)
    if extra:
        _label(out, extra, 10, h - band_h - 10, C_ALERT, 0.55)
    return out


class OverlayWindow:
    """cv2.imshow wrapper. `should_quit()` is how Q or Esc stops a live run cleanly, so the
    recording is closed properly instead of being truncated by a Ctrl-C."""

    def __init__(self, title: str = "Project Relay", enabled: bool = True):
        self.title = title
        self.enabled = enabled

    def show(self, image: np.ndarray) -> None:
        if not self.enabled:
            return
        cv2.imshow(self.title, image)

    def should_quit(self) -> bool:
        if not self.enabled:
            return False
        return cv2.waitKey(1) & 0xFF in (ord("q"), 27)

    def close(self) -> None:
        if self.enabled:
            cv2.destroyAllWindows()
