# Exp_Game/xr_systems/xr_ports/view.py
from __future__ import annotations
import time
import bpy
from ..xr_queue import xr_enqueue

def _ok_to_print(op, tag_attr: str, hz_prop: str, enable_prop: str) -> bool:
    try:
        sc = bpy.context.scene
        if not getattr(sc, "dev_hud_log_console", True):  # MASTER GATE
            return False
        if not getattr(sc, enable_prop, False):
            return False
        hz = float(getattr(sc, hz_prop, 4.0) or 4.0)
        now = time.perf_counter()
        last = float(getattr(op, tag_attr, 0.0) or 0.0)
        if (now - last) >= (1.0 / max(0.1, hz)):
            setattr(op, tag_attr, now)
            return True
    except Exception:
        pass
    return False


def queue_view_third(op,
                     op_key,
                     anchor_xyz, dir_xyz,
                     min_cam: float,
                     desired_max: float,
                     r_cam: float,
                     candidate_allowed: float,
                     hit_token):
    """
    XR-authoritative view distance:
      • Blender computes candidate (same geometry as today).
      • XR filters (rate-limit + latch) and returns 'allowed'.
      • We apply ONLY XR's allowed (no local fallback), min_cam if missing.
    """
    payload = {
        "op_key": str(op_key),
        "anchor": (float(anchor_xyz[0]), float(anchor_xyz[1]), float(anchor_xyz[2])),
        "dir":    (float(dir_xyz[0]),    float(dir_xyz[1]),    float(dir_xyz[2])),
        "min_cam":       float(min_cam),
        "desired_max":   float(desired_max),
        "r_cam":         float(r_cam),
        "candidate_allowed": float(candidate_allowed),
        "hit_token": (str(hit_token) if hit_token is not None else None),
    }

    t_send = time.perf_counter()

    def _apply(res: dict):
        # XR is authoritative
        try:
            allowed = float(res.get("allowed"))
        except Exception:
            allowed = float(min_cam)

        # Write back for apply_view_from_xr(...)
        try:
            op._xr_view_allowed = allowed
        except Exception:
            pass

        # --- NEW: bump apply meter so aHz is correct ---
        try:
            from ...Developers.dev_state import STATE
            m = getattr(STATE, "meter_view_apply", None)
            if m and hasattr(m, "hit"):
                m.hit()
        except Exception:
            pass

        # DevHUD series + scalars
        try:
            from ...Developers.exp_dev_interface import devhud_set, devhud_series_push
            devhud_series_push("view_allowed", allowed)
            devhud_set("VIEW.src", "XR", volatile=True)
            devhud_set("VIEW.hit", res.get("hit_token") or "—", volatile=True)
            devhud_set("VIEW.allowed", f"{allowed:.3f} m", volatile=True)
            devhud_set("VIEW.delta", f"{(candidate_allowed - allowed):+0.3f} m", volatile=True)
            je = res.get("jitter_ema", None)
            if isinstance(je, (int, float)):
                devhud_set("VIEW.jitter", f"{je:.2f} m/s", volatile=True)
        except Exception:
            pass

        # Console (rate-limited)
        if _ok_to_print(op, "_last_log_view_apply", "dev_log_view_hz", "dev_log_view_console"):
            dt_net_ms = (time.perf_counter() - t_send) * 1000.0
            print(
                f"[VIEW.XR] apply allowed={allowed:.3f}  cand={candidate_allowed:.3f}  "
                f"Δ={(candidate_allowed-allowed):+0.3f}  hit={res.get('hit_token') or '—'}  "
                f"r_cam={r_cam:.3f}  min={min_cam:.3f}  net={dt_net_ms:.2f} ms  seq={res.get('_frame_seq','?')}"
            )
    xr_enqueue("view.third.filter.v1", payload, _apply)