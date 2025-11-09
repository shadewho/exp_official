from __future__ import annotations
import time
from ..dev_registry import register_section
from ..dev_draw_prims import draw_text
from ..dev_state import STATE, xr_catalog_update_from_bus

def _fmt(v):
    try:
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)
    except Exception:
        return str(v)

class XRGeomSection:
    key = "xr_geom"
    column = "LEFT"
    order = 15
    prop_toggle = "dev_hud_show_geom"

    def _f(self, BUS, key, d=0.0):
        try: return float(BUS.scalars.get(key, d))
        except Exception: return float(d)
    def _i(self, BUS, key, d=0):
        try: return int(BUS.scalars.get(key, d))
        except Exception: return int(d)
    def _s(self, BUS, key, d="—"):
        try: return str(BUS.scalars.get(key, d))
        except Exception: return str(d)

    def measure(self, scene, STATE, BUS, scale, lh, width):
        if not getattr(scene, "dev_hud_show_geom", False) or not getattr(scene, "dev_hud_show_xr", False):
            return 0

        # One line per enabled summary sub-section
        flags = (
            "dev_xr_geom_mode_auth","dev_xr_geom_static","dev_xr_geom_dynamic",
            "dev_xr_geom_xforms","dev_xr_geom_down_dyn","dev_xr_geom_authority",
            "dev_xr_geom_parity","dev_xr_geom_rates"
        )
        lines = sum(1 for f in flags if getattr(scene, f, False))

        # Stable XR RAW block (header + fixed rows from the catalog)
        if getattr(scene, "dev_xr_geom_dump", False):
            xr_catalog_update_from_bus(BUS)  # grows catalog; never shrinks
            rows = len(getattr(STATE, "xr_keys_order", []) or [])
            lines += (1 + rows)

        return lines * lh

    def draw(self, x, y, scene, STATE, BUS, scale, lh, width):
        if not getattr(scene, "dev_hud_show_geom", False) or not getattr(scene, "dev_hud_show_xr", False):
            return y

        idx = 1
        if getattr(scene, "dev_xr_geom_mode_auth", True):
            draw_text(x, y, f"({idx}) XR.geom  MODE {self._s(BUS,'XR.geom.mode','STATIC_STORE')}   AUTH {self._s(BUS,'XR.geom.authority','BLENDER')}", 12*scale); y -= lh; idx+=1
        if getattr(scene, "dev_xr_geom_static", True):
            draw_text(x, y, f"({idx}) static_tris {self._i(BUS,'XR.geom.static_tris',0):,}   build_ms {self._f(BUS,'XR.geom.build_ms',0.0):0.1f}   mem_MB {self._f(BUS,'XR.geom.mem_MB',0.0):0.2f}", 12*scale); y -= lh; idx+=1
        if getattr(scene, "dev_xr_geom_dynamic", True):
            da = self._i(BUS,"XR.geom.active",0); dobj=self._i(BUS,"XR.geom.dyn_objs",0); dtri=self._i(BUS,"XR.geom.dyn_tris",0); gate=self._i(BUS,"XR.geom.gate_switches",0)
            draw_text(x, y, f"({idx}) dyn_active {da}/{dobj}   dyn_tris {dtri:,}   gate_toggles {gate}", 12*scale); y -= lh; idx+=1
        if getattr(scene, "dev_xr_geom_xforms", True):
            xf_last=self._f(BUS,"XR.geom.xf_last_ms",0.0); xf_ema=self._f(BUS,"XR.geom.xf_ema_ms",0.0)
            rate = getattr(STATE,'meter_geom_xforms',None).rate(time.perf_counter(),1.0) if hasattr(STATE,'meter_geom_xforms') else 0.0
            draw_text(x, y, f"({idx}) xforms/s {rate:0.1f}   xf_last {xf_last:0.2f} ms   xf_ema {xf_ema:0.2f} ms", 12*scale); y -= lh; idx+=1
        if getattr(scene, "dev_xr_geom_down_dyn", False):
            draw_text(x, y, f"({idx}) downDyn req {self._i(BUS,'XR.downDyn.req',0)}  ok {self._i(BUS,'XR.downDyn.ok',0)}  hit {self._i(BUS,'XR.downDyn.hit',0)}  lat_ms {self._f(BUS,'XR.downDyn.lat_ms',0.0):0.2f}", 12*scale); y -= lh; idx+=1
        if getattr(scene, "dev_xr_geom_authority", True):
            draw_text(x, y, f"({idx}) auth↓ src {self._s(BUS,'XR.auth.down.src','—')}  pick {self._s(BUS,'XR.auth.down.pick','—')}  Δ {self._s(BUS,'XR.auth.down.delta_mm','—')} mm", 12*scale); y -= lh; idx+=1
        if getattr(scene, "dev_xr_geom_parity", True):
            gate = self._s(BUS,'XR.parity.gate', self._s(BUS,'XR.parity.status','—'))
            draw_text(x, y, f"({idx}) parity↓ [{gate}]  p95Δ {self._f(BUS,'XR.parity.p95_mm',0.0):0.2f} mm  n·dot {self._f(BUS,'XR.parity.p95_dot',1.0):0.5f}  miss {self._i(BUS,'XR.parity.miss',0)}", 12*scale); y -= lh; idx+=1
        if getattr(scene, "dev_xr_geom_rates", False):
            cast_s = getattr(STATE,'meter_geom_cast',None).rate(time.perf_counter(),1.0) if hasattr(STATE,'meter_geom_cast') else 0.0
            near_s = getattr(STATE,'meter_geom_nearest',None).rate(time.perf_counter(),1.0) if hasattr(STATE,'meter_geom_nearest') else 0.0
            draw_text(x, y, f"({idx}) cast/s {cast_s:0.1f}   nearest/s {near_s:0.1f}", 12*scale); y -= lh; idx+=1

        # --- Stable RAW XR dump (no flicker) ---
        if getattr(scene, "dev_xr_geom_dump", False):
            xr_catalog_update_from_bus(BUS)  # update values + grow catalog
            keys = list(getattr(STATE, "xr_keys_order", []) or [])
            draw_text(x, y, f"({idx}) XR RAW — {len(keys)} keys", 12*scale); y -= lh

            NORMAL = (1.0, 1.0, 1.0, 1.0)
            STALE  = (0.78, 0.82, 0.90, 0.90)

            for k in keys:
                last_seen = getattr(STATE, "xr_last_seen", {}).get(k, -999999)
                age = int(STATE.frame_idx) - int(last_seen)
                col = NORMAL if age == 0 else STALE
                val = getattr(STATE, "xr_last_val", {}).get(k, "—")
                short = k[3:] if k.startswith("XR.") else (k[2:] if k.startswith("XR") else k)
                draw_text(x, y, f"{short}: {_fmt(val)}", 12*scale, col); y -= lh

        return y

register_section(XRGeomSection())
