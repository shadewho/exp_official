from __future__ import annotations
from ..dev_registry import register_section
from ..dev_draw_prims import draw_text, draw_rect, draw_line_strip, samples_to_poly
from ..dev_state import STATE

class GraphsSection:
    key = "graphs"
    column = "LEFT"
    order = 50
    prop_toggle = "dev_hud_graphs"

    def __init__(self):
        self._gh_factor = 2.0  # graph height = 2*lh

    def measure(self, scene, STATE, BUS, scale, lh, width):
        if not getattr(scene, "dev_hud_graphs", False): return 0
        per = int(0.6*lh) + int(self._gh_factor*lh) + int(0.5*lh)
        return 3 * per

    def _graph(self, x, y, w, lh, label, series_key, ymin=None, ymax=None, color=(0.31,0.66,1.0,1.0), zero=None, scale=1):
        draw_text(x, y, label, 12*scale); y -= int(0.6*lh)
        gh = int(self._gh_factor * lh)
        sy = y - gh
        draw_rect(x, sy, w, gh, (0.10,0.12,0.15,0.85))
        s = STATE.series.get(series_key)
        if s:
            lo, hi = s.minmax()
            if ymin is not None or ymax is not None:
                lo = lo if ymin is None else float(ymin)
                hi = hi if ymax is None else float(ymax)
                if hi <= lo: hi = lo + 1.0
            if (zero is not None) and (lo < zero < hi):
                zq = (zero - lo)/(hi-lo); zy = sy + (1.0 - zq)*gh
                draw_line_strip([(x, zy), (x+w, zy)], (0.45,0.50,0.58,0.8), width=1.0)
            pts = samples_to_poly(s, x, sy, w, gh, lo, hi)
            draw_line_strip(pts, color, width=1.4*scale)
        y = sy - int(0.5*lh)
        return y

    def draw(self, x, y, scene, STATE, BUS, scale, lh, width):
        y = self._graph(x, y, width, lh, "Frame ms", "frame_ms", ymin=0.0, ymax=max(40.0, STATE.ms_ema*1.5), color=(0.50,1.0,0.60,1.0), scale=scale)
        last = STATE.series['rtt_ms'].last or 0.0
        y = self._graph(x, y, width, lh, "XR RTT (ms)", "rtt_ms", ymin=0.0, ymax=max(12.0, last*2.0), color=(0.95,0.80,0.40,1.0), scale=scale)
        y = self._graph(x, y, width, lh, "View Allowed (m)", "view_allowed", ymin=0.0, ymax=None, color=(0.65,0.75,1.0,1.0), scale=scale)
        return y

register_section(GraphsSection())
