# Exp_Game/Developers/dev_sections/section_xr_geom.py
from __future__ import annotations
import time
from ..dev_registry import register_section
from ..dev_draw_prims import draw_text
from ..dev_state import STATE

class XRGeomSection:
    key = "xr_geom"
    column = "LEFT"
    order = 15
    prop_toggle = "dev_hud_show_geom"

    def measure(self, scene, STATE, BUS, scale, lh, width):
        return 0 if not getattr(scene, "dev_hud_show_geom", False) else 5 * lh

    @staticmethod
    def _f(BUS, key: str, default: float = 0.0) -> float:
        v = BUS.scalars.get(key, default)
        try: return float(v)
        except Exception: return float(default)

    @staticmethod
    def _i(BUS, key: str, default: int = 0) -> int:
        v = BUS.scalars.get(key, default)
        try: return int(v)
        except Exception: return int(default)

    def draw(self, x, y, scene, STATE, BUS, scale, lh, width):
        if not getattr(scene, "dev_hud_show_geom", False):
            return y

        nowt = time.perf_counter()
        cast_m   = getattr(STATE, "meter_geom_cast", None)
        near_m   = getattr(STATE, "meter_geom_nearest", None)
        xform_m  = getattr(STATE, "meter_geom_xforms", None)
        cast_s   = (cast_m.rate(nowt, 1.0)   if cast_m else 0.0)
        near_s   = (near_m.rate(nowt, 1.0)   if near_m else 0.0)
        xforms_s = (xform_m.rate(nowt, 1.0)  if xform_m else 0.0)

        mode = BUS.scalars.get("XR.geom.mode", "STATIC_STORE")
        auth = BUS.scalars.get("XR.geom.authority", "BLENDER")
        draw_text(x, y, f"XR.geom  MODE {mode}   AUTH {auth}", 12*scale); y -= lh

        static_tris = self._i(BUS, "XR.geom.static_tris", 0)
        build_ms    = self._f(BUS, "XR.geom.build_ms", 0.0)
        mem_mb      = self._f(BUS, "XR.geom.mem_MB", 0.0)
        draw_text(x, y, f"static_tris {static_tris:,}   build_ms {build_ms:0.1f}   mem_MB {mem_mb:0.1f}", 12*scale); y -= lh

        dyn_objs = self._i(BUS, "XR.geom.dyn_objs", 0)
        dyn_tris = self._i(BUS, "XR.geom.dyn_tris", 0)
        dyn_active = self._i(BUS, "XR.geom.active", 0)
        gate_sw = self._i(BUS, "XR.geom.gate_switches", 0)
        draw_text(x, y, f"dyn_active {dyn_active}/{dyn_objs}   dyn_tris {dyn_tris:,}   gate_toggles {gate_sw}", 12*scale); y -= lh

        xf_last = self._f(BUS, "XR.geom.xf_last_ms", 0.0)
        xf_ema  = self._f(BUS, "XR.geom.xf_ema_ms", 0.0)
        draw_text(x, y, f"xforms/s {xforms_s:0.1f}   xf_last {xf_last:0.2f} ms   xf_ema {xf_ema:0.2f} ms", 12*scale); y -= lh

        draw_text(x, y, f"cast/s {cast_s:0.1f}   nearest/s {near_s:0.1f}", 12*scale); y -= lh
        return y

register_section(XRGeomSection())
