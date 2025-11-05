from __future__ import annotations
import time
from ..dev_registry import register_section
from ..dev_draw_prims import draw_text
from ..dev_utils import pyget, safe_modal
from ..dev_state import STATE

class XRSection:
    key = "xr"
    column = "LEFT"
    order = 10
    prop_toggle = "dev_hud_show_xr"

    def measure(self, scene, STATE, BUS, scale, lh, width):
        return 0 if not getattr(scene, "dev_hud_show_xr", False) else 2*lh

    def draw(self, x, y, scene, STATE, BUS, scale, lh, width):
        modal = safe_modal()
        xs = pyget(modal, "_xr_stats", {}) if modal else {}
        xr_ok = bool(pyget(modal, "_xr", None)) if modal else False
        rtt = float(xs.get("last_lat_ms", 0.0)) if xs else 0.0
        ph  = float(xs.get("phase_ms", 0.0))    if xs else 0.0
        rqhz = STATE.meter_xr_req.rate(time.perf_counter(), 1.0)
        okhz = STATE.meter_xr_ok.rate(time.perf_counter(), 1.0)
        flhz = STATE.meter_xr_fail.rate(time.perf_counter(), 1.0)
        status = "✓ XR ALIVE" if xr_ok else "… XR Idle"
        draw_text(x, y, f"{status}   rtt {rtt:.2f} ms   phase {ph:+.2f} ms", 12*scale); y -= lh
        draw_text(x, y, f"XR req/s {rqhz:0.1f}  ok/s {okhz:0.1f}  fail/s {flhz:0.1f}", 12*scale); y -= lh
        return y

register_section(XRSection())
