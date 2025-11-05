from __future__ import annotations
from ..dev_registry import register_section
from ..dev_draw_prims import draw_text

class CoreSummary:
    key = "core.summary"
    column = "LEFT"
    order = 0
    prop_toggle = None  # always on when HUD enabled

    def _p01_low_fps(self, STATE):
        vals = list(STATE.series["frame_ms"].values)
        if len(vals) < 60: return 0.0
        vals_sorted = sorted(vals)
        worst_idx = max(0, int(len(vals_sorted) * 0.99) - 1)
        p01_ms = vals_sorted[worst_idx]
        return (1000.0 / p01_ms) if p01_ms > 0.0 else 0.0

    def _grade(self, STATE):
        ema_ms  = float(STATE.ms_ema or 0.0)
        ema_fps = float(STATE.fps_ema or 0.0)
        budget_ms = 33.3333
        ratio = (ema_ms / budget_ms) if budget_ms > 0 else 0.0
        if   ratio <= 0.60: lab, col = ("EXCELLENT", (0.65, 1.00, 0.65, 1.0))
        elif ratio <= 0.85: lab, col = ("GREAT",     (0.75, 1.00, 0.75, 1.0))
        elif ratio <= 1.00: lab, col = ("GOOD",      (1.00, 1.00, 0.70, 1.0))
        elif ratio <= 1.50: lab, col = ("OK",        (1.00, 0.90, 0.60, 1.0))
        elif ratio <= 2.00: lab, col = ("POOR",      (1.00, 0.70, 0.60, 1.0))
        else:               lab, col = ("CRITICAL",  (1.00, 0.55, 0.55, 1.0))
        low_fps = self._p01_low_fps(STATE)
        if ema_fps > 0.0 and low_fps > 0.0 and low_fps < 0.7 * ema_fps:
            ladder = ["EXCELLENT","GREAT","GOOD","OK","POOR","CRITICAL"]
            try:
                i = ladder.index(lab)
                lab = ladder[min(i+1, len(ladder)-1)]
                col = (min(1.0, col[0] + 0.05), col[1] * 0.98, col[2] * 0.98, 1.0)
            except ValueError:
                pass
        return lab, col, budget_ms, low_fps

    def measure(self, scene, STATE, BUS, scale, lh, width):
        return int(1.3*lh) + lh

    def draw(self, x, y, scene, STATE, BUS, scale, lh, width):
        draw_text(x, y, "DEVELOPER HUD", 14*scale); y -= lh
        lab, col, budget, low_fps = self._grade(STATE)
        ema_ms  = float(STATE.ms_ema or 0.0)
        ema_fps = float(STATE.fps_ema or 0.0)
        draw_text(x, y, f"Frame {ema_ms:5.2f} ms   ~{ema_fps:0.1f} FPS  (1% {low_fps:0.0f})   Budget {budget:0.1f} ms   Rating:", 12*scale)
        draw_text(x + int(360*scale), y, lab, 12*scale, col); y -= lh
        return y

register_section(CoreSummary())
