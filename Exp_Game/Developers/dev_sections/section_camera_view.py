from __future__ import annotations
import time
from ..dev_registry import register_section
from ..dev_draw_prims import draw_text
from ..dev_state import STATE

class CameraViewSection:
    key = "view"
    column = "LEFT"
    order = 40
    prop_toggle = "dev_hud_show_view"

    def measure(self, scene, STATE, BUS, scale, lh, width):
        return 0 if not getattr(scene, "dev_hud_show_view", False) else 3*lh

    def draw(self, x, y, scene, STATE, BUS, scale, lh, width):
        nowt = time.perf_counter()
        qhz = STATE.meter_view_queue.rate(nowt, 1.0)
        ahz = STATE.meter_view_apply.rate(nowt, 1.0)
        va  = STATE.series["view_allowed"].last
        vc  = STATE.series["view_candidate"].last
        dd  = (vc - va) if (vc is not None and va is not None) else 0.0
        lag = STATE.view_lag_ema_ms
        jit = STATE.view_jitter_ema
        draw_text(x, y, f"View Hz   queue {qhz:0.1f}   apply {ahz:0.1f}", 12*scale); y -= lh
        draw_text(x, y, f"View dist  allowed {(va or 0):0.3f} m   cand {(vc or 0):0.3f} m   Δ {dd:+0.3f} m", 12*scale); y -= lh
        hit = BUS.temp.get("VIEW.hit", BUS.scalars.get("VIEW.hit", "—"))
        src = BUS.temp.get("VIEW.src", BUS.scalars.get("VIEW.src", "—"))
        draw_text(x, y, f"View lag {lag:0.1f} ms   jitter |d|/s ~{jit:0.2f}   src {src}   hit {hit}", 12*scale); y -= lh
        return y

register_section(CameraViewSection())
