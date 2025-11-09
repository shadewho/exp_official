# Blender-side helpers to enqueue tiny KCC jobs (no stalls).
from __future__ import annotations
import time
import bpy
from ..xr_queue import xr_enqueue

def _ok_to_print_kcc(kcc_sink, tag_attr: str, hz_prop: str, enable_prop: str) -> bool:
    try:
        sc = bpy.context.scene
        if not getattr(sc, "dev_hud_log_console", True):  # MASTER GATE
            return False
        if not getattr(sc, enable_prop, False):
            return False
        hz = float(getattr(sc, hz_prop, 4.0) or 4.0)
        now = time.perf_counter()
        last = float(getattr(kcc_sink, tag_attr, 0.0) or 0.0)
        if (now - last) >= (1.0 / max(0.1, hz)):
            setattr(kcc_sink, tag_attr, now)
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

                # One-time banner
                sc = getattr(bpy.context, "scene", None)
                if sc and getattr(sc, "dev_log_kcc_console", False) and not getattr(kcc_sink, "_xr_move_banner", False):
                    print("[KCC] XR move_xy ACTIVE — wish vector computed in XR (no local fallback).")
                    setattr(kcc_sink, "_xr_move_banner", True)

                # DevHUD
                try:
                    from ...Developers.exp_dev_interface import devhud_set, devhud_series_push
                    devhud_set("KCC.src", "XR", volatile=True)
                    devhud_set("KCC.xy", f"({xy[0]:.3f}, {xy[1]:.3f})", volatile=True)
                    devhud_series_push("kcc_xy_x", float(xy[0]))
                    devhud_series_push("kcc_xy_y", float(xy[1]))
                except Exception:
                    pass

                # Per-step log (rate-limited, stored on KCC)
                if _ok_to_print_kcc(kcc_sink, "_last_log_kccxr", "dev_log_kcc_hz", "dev_log_kcc_console"):
                    print(f"[KCC] XR move_xy -> ({xy[0]:.3f},{xy[1]:.3f})  seq={res.get('_frame_seq','?')}")
            else:
                try:
                    from ...Developers.exp_dev_interface import devhud_set
                    devhud_set("KCC.src", "MISS", volatile=True)
                except Exception:
                    pass
        except Exception:
            pass

    xr_enqueue("kcc.move_xy.v1", payload, _apply)


def queue_accel_xy(kcc_sink, vx: float, vy: float, wx: float, wy: float,
                   target_speed: float, accel: float, dt: float):
    """
    Enqueue XR job to compute new horizontal velocity via one-step accel blend.
    Result is applied back to kcc_sink._xr_accel_xy (no local compute).
    """
    payload = {
        "vx": float(vx), "vy": float(vy),
        "wx": float(wx), "wy": float(wy),
        "target_speed": float(target_speed),
        "accel": float(accel),
        "dt": float(dt),
    }

    def _apply(res: dict):
        try:
            xy = res.get("xy")
            if isinstance(xy, (list, tuple)) and len(xy) == 2:
                kcc_sink._xr_accel_xy  = (float(xy[0]), float(xy[1]))
                kcc_sink._xr_accel_age = time.perf_counter()

                # One-time banner
                sc = getattr(bpy.context, "scene", None)
                if sc and getattr(sc, "dev_log_kcc_console", False) and not getattr(kcc_sink, "_xr_accel_banner", False):
                    print("[KCC] XR accel_xy ACTIVE — horizontal acceleration computed in XR ONLY (no main-thread fallback).")
                    setattr(kcc_sink, "_xr_accel_banner", True)

                # DevHUD
                try:
                    from ...Developers.exp_dev_interface import devhud_set, devhud_series_push
                    devhud_set("KCC.accel", f"({xy[0]:.3f}, {xy[1]:.3f})", volatile=True)
                    devhud_series_push("kcc_accel_x", float(xy[0]))
                    devhud_series_push("kcc_accel_y", float(xy[1]))
                except Exception:
                    pass

                # Per-step log (rate-limited, stored on KCC)
                if _ok_to_print_kcc(kcc_sink, "_last_log_kccacc", "dev_log_kcc_hz", "dev_log_kcc_console"):
                    print(f"[KCC] XR accel_xy -> ({xy[0]:.3f},{xy[1]:.3f})  seq={res.get('_frame_seq','?')}")
            else:
                try:
                    from ...Developers.exp_dev_interface import devhud_set
                    devhud_set("KCC.accel", "MISS", volatile=True)
                except Exception:
                    pass
        except Exception:
            pass

    xr_enqueue("kcc.accel_xy.v1", payload, _apply)


def queue_clamp_xy(kcc_sink, hx: float, hy: float, normal_xyz, floor_cos: float):
    """
    Enqueue XR job to clamp XY against a steep slope. No local compute.
    Result stored on kcc_sink._xr_clamp_xy.
    """
    payload = {
        "hx": float(hx), "hy": float(hy),
        "normal": (
            float(normal_xyz[0]), float(normal_xyz[1]), float(normal_xyz[2])
        ) if (isinstance(normal_xyz, (list, tuple)) and len(normal_xyz) == 3) else (0.0, 0.0, 1.0),
        "floor_cos": float(floor_cos),
    }

    def _apply(res: dict):
        try:
            xy = res.get("xy")
            if isinstance(xy, (list, tuple)) and len(xy) == 2:
                kcc_sink._xr_clamp_xy  = (float(xy[0]), float(xy[1]))
                kcc_sink._xr_clamp_age = time.perf_counter()

                sc = getattr(bpy.context, "scene", None)
                if sc and getattr(sc, "dev_log_kcc_console", False) and not getattr(kcc_sink, "_xr_clamp_banner", False):
                    print("[KCC] XR clamp_xy ACTIVE — steep-slope clamp computed in XR ONLY.")
                    setattr(kcc_sink, "_xr_clamp_banner", True)

                try:
                    from ...Developers.exp_dev_interface import devhud_set, devhud_series_push
                    devhud_set("KCC.clamp", f"({xy[0]:.3f}, {xy[1]:.3f})", volatile=True)
                    devhud_series_push("kcc_clamp_x", float(xy[0]))
                    devhud_series_push("kcc_clamp_y", float(xy[1]))
                except Exception:
                    pass

                if _ok_to_print_kcc(kcc_sink, "_last_log_kccclamp", "dev_log_kcc_hz", "dev_log_kcc_console"):
                    print(f"[KCC] XR clamp_xy -> ({xy[0]:.3f},{xy[1]:.3f})  seq={res.get('_frame_seq','?')}")
        except Exception:
            pass

    xr_enqueue("kcc.clamp_xy.v1", payload, _apply)