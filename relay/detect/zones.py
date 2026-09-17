"""Which zone is a detection in, and how sure are we?

Membership is by **bbox centre**, not by overlap. A person standing at the edge of the desk
overlaps the post zone with a sliver of shoulder; that is not "at the post". The centre is
where the body actually is.

The interesting case is the boundary. A detection whose centre sits within `ZONE_MARGIN` of a
post edge is genuinely ambiguous, and the honest response is not to pick a side quietly:

* it is marked `straddle`, which is a review trigger in its own right,
* membership falls back to apparent size (`area_frac >= POST_MIN_AREA`), because on a single
  fixed camera a bigger box means closer to the camera, which here means at the desk,
* and `zone_term` drops, so the event's confidence drops with it.

A rectangle is used rather than a polygon because it is the thing most likely to need
changing live on a follow-up call, and `POST_ZONE=x0,y0,x1,y1` in a .env is a five-second
edit. Polygon support is a one-line swap to `cv2.pointPolygonTest`.
"""

from __future__ import annotations

from ..config import Settings
from ..schema import Zone

#: Confidence contributed by the zone decision, by how that decision was reached.
TERM_CLEAR = 1.00      # comfortably inside or outside
TERM_STRADDLE_AREA = 0.55   # on the line, resolved by apparent size
TERM_STRADDLE_WEAK = 0.40   # on the line, and the size test was not convincing either


class ZoneModel:
    def __init__(self, cfg: Settings):
        self.x0, self.y0, self.x1, self.y1 = cfg.post_zone
        self.margin = cfg.zone_margin
        self.min_area = cfg.post_min_area

    def _inside(self, cx: float, cy: float) -> bool:
        return self.x0 <= cx <= self.x1 and self.y0 <= cy <= self.y1

    def _distance_to_edge(self, cx: float, cy: float) -> float:
        """Smallest distance from the centre to any edge of the rectangle.

        Signed distances would be tidier, but what matters is only *how close to a boundary*
        the point is, inside or outside, so the absolute value is the whole story.
        """
        return min(abs(cx - self.x0), abs(cx - self.x1), abs(cy - self.y0), abs(cy - self.y1))

    def classify(self, centre: tuple[float, float], area_frac: float) -> tuple[Zone, bool, float]:
        """-> (zone, straddle, zone_term)"""
        cx, cy = centre
        inside = self._inside(cx, cy)
        near_edge = self._distance_to_edge(cx, cy) <= self.margin

        if not near_edge:
            return ("post" if inside else "approach"), False, TERM_CLEAR

        # Ambiguous: decide on apparent size, flag it, and lower the confidence either way.
        big_enough = area_frac >= self.min_area
        zone: Zone = "post" if big_enough else "approach"
        # If the size test agrees with the raw geometry we are a little happier than if it
        # had to overrule it.
        term = TERM_STRADDLE_AREA if big_enough == inside else TERM_STRADDLE_WEAK
        return zone, True, term

    def describe(self) -> str:
        return (f"post zone ({self.x0:.2f},{self.y0:.2f})-({self.x1:.2f},{self.y1:.2f}), "
                f"margin {self.margin:.2f}, min area {self.min_area:.2f}")
