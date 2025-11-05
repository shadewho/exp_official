from __future__ import annotations
import time
from ..dev_registry import register_section
from ..dev_draw_prims import draw_text
from ..dev_utils import pyget, safe_modal
from ..dev_state import STATE

class PhysicsSection:
    key = "physics"
    column = "LEFT"
    order = 30
    prop_toggle = "dev_hud_show_physics"

    def measure(self, scene, STATE, BUS, scale, lh, width):
        return 0 if not getattr(scene, "dev_hud_show_physics", False) else 2 * lh

    def draw(self, x, y, scene, STATE, BUS, scale, lh, width):
        from time import perf_counter
        modal = safe_modal()
        steps = int(pyget(modal, "_perf_last_physics_steps", 0) or 0) if modal else 0
        hz = int(pyget(modal, "physics_hz", 30) or 30)
        timer_hz = STATE.meter_timer.rate(perf_counter(), 1.0)
        draw_text(x, y, f"Physics  {steps} step{'s' if steps!=1 else ''} @ {hz} Hz   TIMER ~{timer_hz:0.1f} Hz", 12*scale); y -= lh

        wish_xy = (0.0,0.0); wish_age = 0.0
        acc_xy  = (0.0,0.0); acc_age  = 0.0
        cl_xy   = (0.0,0.0); cl_age   = 0.0
        try:
            pc = getattr(modal, "physics_controller", None)
            now = perf_counter()
            if pc:
                wx = getattr(pc, "_xr_wish_xy", None); wa = float(getattr(pc, "_xr_wish_age", 0.0) or 0.0)
                ax = getattr(pc, "_xr_accel_xy", None); aa = float(getattr(pc, "_xr_accel_age", 0.0) or 0.0)
                cx = getattr(pc, "_xr_clamp_xy", None); ca = float(getattr(pc, "_xr_clamp_age", 0.0) or 0.0)
                if isinstance(wx, (tuple, list)) and len(wx)==2: wish_xy = (float(wx[0]), float(wx[1])); wish_age = (now-wa)*1000.0 if wa>0.0 else 0.0
                if isinstance(ax, (tuple, list)) and len(ax)==2: acc_xy  = (float(ax[0]), float(ax[1])); acc_age  = (now-aa)*1000.0 if aa>0.0 else 0.0
                if isinstance(cx, (tuple, list)) and len(cx)==2: cl_xy   = (float(cx[0]), float(cx[1])); cl_age   = (now-ca)*1000.0 if ca>0.0 else 0.0
        except Exception:
            pass

        draw_text(x, y, f"KCC  wish ({wish_xy[0]: .2f},{wish_xy[1]: .2f}) {wish_age:4.0f} ms   "
                        f"accel ({acc_xy[0]: .2f},{acc_xy[1]: .2f}) {acc_age:4.0f} ms   "
                        f"clamp ({cl_xy[0]: .2f},{cl_xy[1]: .2f}) {cl_age:4.0f} ms", 12*scale); y -= lh
        return y


register_section(PhysicsSection())
