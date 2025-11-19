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


def queue_forward_sweep_min3_probe(kcc_sink,
                                   base_xyz,           # (x,y,z) BEFORE local sweep
                                   final_local_xyz,    # (x,y,z) AFTER local sweep (baseline for pos Δ)
                                   dir_xyz,            # (x,y,z) world forward (same you used locally)
                                   step_len: float,
                                   r: float,
                                   h: float,
                                   floor_cos: float,
                                   vel_xy_before,      # (vx,vy) BEFORE local sweep
                                   vel_xy_after_local  # (vx,vy) AFTER local sweep (to compare with XR)
                                   ):
    """
    Non-blocking XR mirror of _forward_sweep_min3 for parity only.
    Sampling is gated by dev_geom_parity_enable + dev_log_geom_hz (0.25..2 Hz),
    and further gated to only run after we've seen at least one dynamic xform batch.
    Writes stable HUD keys: XR.auth.fwd.sweep.*
    """
    import time as _t, math, bpy
    from .geom import _hud_set as _hud, _ok_to_print as _okprint, get_last_xf_seq
    from ..xr_queue import xr_enqueue

    sc = getattr(getattr(bpy, "context", None), "scene", None)
    if not (sc and getattr(sc, "dev_geom_parity_enable", False)):
        return

    # Low, controllable sampling from existing knob (no new prefs)
    hz = float(getattr(sc, "dev_log_geom_hz", 1.0) or 1.0)
    hz = max(0.25, min(2.0, hz))
    now = _t.perf_counter()
    last = float(getattr(queue_forward_sweep_min3_probe, "_last_enq", 0.0) or 0.0)
    if (now - last) < (1.0 / hz):
        return

    # Startup/xforms guard: only sample when we've applied a recent xforms batch
    seq = int(get_last_xf_seq())
    prev_seq = int(getattr(queue_forward_sweep_min3_probe, "_last_seq", -1))
    if seq <= 0 or seq == prev_seq:
        return
    queue_forward_sweep_min3_probe._last_seq = seq
    queue_forward_sweep_min3_probe._last_enq = now

    try:
        bx, by, bz = float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2])
        fx, fy, fz = float(final_local_xyz[0]), float(final_local_xyz[1]), float(final_local_xyz[2])
        dx, dy, dz = float(dir_xyz[0]), float(dir_xyz[1]), float(dir_xyz[2])
        vbx, vby   = float(vel_xy_before[0]),     float(vel_xy_before[1])
        vlx, vly   = float(vel_xy_after_local[0]), float(vel_xy_after_local[1])
    except Exception:
        return

    payload = {
        "pos": (bx, by, bz),
        "dir": (dx, dy, dz),
        "r": float(r), "h": float(h),
        "step_len": float(step_len),
        "floor_cos": float(floor_cos),
        "vel_xy": (vbx, vby),
    }
    t0 = _t.perf_counter()

    def _apply(res: dict):
        lat_ms = (_t.perf_counter() - t0) * 1000.0

        # Position Δ (mm)
        rx, ry, rz = res.get("pos", (bx, by, bz))
        ddx = fx - float(rx); ddy = fy - float(ry); ddz = fz - float(rz)
        pos_mm = (ddx*ddx + ddy*ddy + ddz*ddz) ** 0.5 * 1000.0

        # Velocity Δ (m/s) — XR post-sweep vs local post-sweep
        vdelta = None
        vxy = res.get("vel_xy_after", None)
        if isinstance(vxy, (list, tuple)) and len(vxy) == 2:
            dvx = vlx - float(vxy[0])
            dvy = vly - float(vxy[1])
            vdelta = math.sqrt(dvx*dvx + dvy*dvy)

        src  = str(res.get("src", "—")) if res.get("hit", False) else "—"
        band = str(res.get("band", "—")) if res.get("hit", False) else "—"
        slid = bool(res.get("slid", False))
        allow  = float(res.get("allow", 0.0))
        allow2 = res.get("allow2", None)

        # Stable HUD keys (no flicker)
        _hud("XR.auth.fwd.sweep.src",    src,                           volatile=True)
        _hud("XR.auth.fwd.sweep.band",   band,                          volatile=True)
        _hud("XR.auth.fwd.sweep.pos_mm", float(round(pos_mm, 2)),       volatile=True)
        _hud("XR.auth.fwd.sweep.vΔ",     (None if vdelta is None else float(round(vdelta, 4))), volatile=True)
        _hud("XR.auth.fwd.sweep.slid",   ("Y" if slid else "N"),        volatile=True)
        _hud("XR.auth.fwd.sweep.allow",  float(round(allow, 4)),        volatile=True)
        _hud("XR.auth.fwd.sweep.allow2", (None if allow2 is None else float(round(allow2, 4))), volatile=True)
        _hud("XR.auth.fwd.sweep.lat_ms", float(round(lat_ms, 2)),       volatile=True)

        # Console one-liner (quiet, useful)
        if _okprint("_last_fwd_sweep", "dev_log_geom_hz", "dev_log_geom_console"):
            a2txt = "—" if allow2 is None else f"{allow2:0.3f}"
            vdtxt = "—" if vdelta is None else f"{vdelta:0.4f}"
            print(f"[FWD.SWEEP] src={src} band={band} posΔ={pos_mm:0.2f} mm  vΔ={vdtxt} m/s  slid={'Y' if slid else 'N'}  allow={allow:0.3f}  allow2={a2txt}  lat={lat_ms:0.2f} ms")
    xr_enqueue("kcc.forward_sweep_min3.v1", payload, _apply)
