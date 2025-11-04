# Exploratory/Exp_Game/xr_systems/xr_ports/kcc.py
# Blender-side helper to enqueue tiny KCC jobs (no stalls).

import time
from ..xr_queue import xr_enqueue
import bpy
def _ok_to_print(tag_key: str, hz_prop: str, enable_prop: str) -> bool:
    try:
        sc = bpy.context.scene
        if not getattr(sc, enable_prop, False):
            return False
        hz = float(getattr(sc, hz_prop, 4.0) or 4.0)
        now = time.perf_counter()
        last = float(getattr(sc, tag_key, 0.0) or 0.0)
        if (now - last) >= (1.0 / max(0.1, hz)):
            setattr(sc, tag_key, now)
            return True
    except Exception:
        pass
    return False

def queue_move_xy(kcc_sink, dx: float, dy: float, yaw: float,
                  on_ground: bool, normal_xyz, max_slope_dot: float):
    """
    Enqueue one XR job that computes world-space wish XY from intent+yaw
    with steep-slope uphill removal. Result is applied back to kcc_sink._xr_wish_xy.
    """
    payload = {
        "dx": float(dx), "dy": float(dy), "yaw": float(yaw),
        "on_ground": bool(on_ground),
        "normal": (
            float(normal_xyz[0]), float(normal_xyz[1]), float(normal_xyz[2])
        ) if (isinstance(normal_xyz, (list, tuple)) and len(normal_xyz) == 3) else (0.0, 0.0, 1.0),
        "max_slope_dot": float(max_slope_dot),
    }

    def _apply(res: dict):
        try:
            xy = res.get("xy")
            if isinstance(xy, (list, tuple)) and len(xy) == 2:
                kcc_sink._xr_wish_xy  = (float(xy[0]), float(xy[1]))
                kcc_sink._xr_wish_age = time.perf_counter()
                try:
                    from ...Developers.exp_dev_interface import devhud_set, devhud_series_push
                    devhud_set("KCC.src", "XR", volatile=True)
                    devhud_set("KCC.xy", f"({xy[0]:.3f}, {xy[1]:.3f})", volatile=True)
                    devhud_series_push("kcc_xy_x", float(xy[0]))
                    devhud_series_push("kcc_xy_y", float(xy[1]))
                except Exception:
                    pass
                if _ok_to_print("_last_log_kccxr", "dev_log_kcc_hz", "dev_log_kcc_console"):
                    print(f"[KCC] XR move_xy  xy=({xy[0]:.3f},{xy[1]:.3f})  seq={res.get('_frame_seq','?')}")
            else:
                try:
                    from ...Developers.exp_dev_interface import devhud_set
                    devhud_set("KCC.src", "MISS", volatile=True)
                except Exception:
                    pass
        except Exception:
            pass

    xr_enqueue("kcc.move_xy.v1", payload, _apply)

