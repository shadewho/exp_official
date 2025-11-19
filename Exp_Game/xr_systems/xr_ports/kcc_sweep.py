# Exp_Game/xr_systems/xr_ports/kcc_sweep.py
# -----------------------------------------------------------------------------
# XR PORTS: Forward-sweep parity probe enqueue + console formatting.
# Uses its OWN console toggle + Hz:
#   dev_log_forward_sweep_min3_console  (bool)
#   dev_log_forward_sweep_min3_hz       (0.1..60.0 Hz)
#
# Master gate: dev_hud_log_console (bool)
# No dependency on HUD visibility (dev_hud_enable) for printing.
# -----------------------------------------------------------------------------
from __future__ import annotations
import time as _t
import math
import bpy
from .geom import _hud_set as _hud, get_last_xf_seq
from ..xr_queue import xr_enqueue

# ------------- Local gates (no HUD dependency) ------------------------------

def _read_hz(sc) -> float:
    # fall back to XR.Geom Hz if forward-sweep Hz unset
    hz = getattr(sc, "dev_log_forward_sweep_min3_hz",
                 getattr(sc, "dev_log_geom_hz", 1.0))
    try:
        hz = float(hz or 1.0)
    except Exception:
        hz = 1.0
    # Honor your property range (0.1..60.0)
    return max(0.1, min(60.0, hz))

def _enq_gate(tag_attr: str) -> bool:
    """
    Gate parity ENQUEUE by master console + forward-sweep toggle + Hz.
    This is independent of HUD visibility and avoids over-enqueue.
    """
    sc = getattr(getattr(bpy, "context", None), "scene", None)
    if not sc:
        return False
    if not getattr(sc, "dev_hud_log_console", True):
        return False
    if not getattr(sc, "dev_log_forward_sweep_min3_console", False):
        return False

    hz = _read_hz(sc)
    now = _t.perf_counter()
    last = float(getattr(_enq_gate, tag_attr, 0.0) or 0.0)
    if (now - last) >= (1.0 / hz):
        setattr(_enq_gate, tag_attr, now)
        return True
    return False

def _ok_to_print_forward(tag_attr: str) -> bool:
    """
    Gate CONSOLE PRINTS by the same toggle + Hz.
    Not tied to dev_hud_enable or dev_hud_show_xr.
    """
    sc = getattr(getattr(bpy, "context", None), "scene", None)
    if not sc:
        return False
    if not getattr(sc, "dev_hud_log_console", True):
        return False
    if not getattr(sc, "dev_log_forward_sweep_min3_console", False):
        return False

    hz = _read_hz(sc)
    now = _t.perf_counter()
    last = float(getattr(_ok_to_print_forward, tag_attr, 0.0) or 0.0)
    if (now - last) >= (1.0 / hz):
        setattr(_ok_to_print_forward, tag_attr, now)
        return True
    return False

# ------------- Public API ----------------------------------------------------

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
    Sampling is controlled by:
      • dev_log_forward_sweep_min3_hz (falls back to dev_log_geom_hz)

    Console output is gated by:
      • dev_hud_log_console (master)
      • dev_log_forward_sweep_min3_console (category)
    """
    sc = getattr(getattr(bpy, "context", None), "scene", None)
    if sc is None:
        return

    # Gate enqueue by local frequency + toggle
    if not _enq_gate("_last_fwd_sweep_enq"):
        return

    # XR readiness: require at least one xforms batch applied once (seq > 0).
    # We DO NOT require seq to change every time; frequency above already limits spam.
    seq = int(get_last_xf_seq())
    if seq <= 0:
        return

    try:
        bx, by, bz = float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2])
        fx, fy, fz = float(final_local_xyz[0]), float(final_local_xyz[1]), float(final_local_xyz[2])
        dx, dy, dz = float(dir_xyz[0]),  float(dir_xyz[1]),  float(dir_xyz[2])
        vbx, vby   = float(vel_xy_before[0]),      float(vel_xy_before[1])
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
            dvx = vlx - float(vxy[0]); dvy = vly - float(vxy[1])
            vdelta = math.sqrt(dvx*dvx + dvy*dvy)

        src  = str(res.get("src", "—"))  if res.get("hit", False) else "—"
        band = str(res.get("band", "—")) if res.get("hit", False) else "—"
        slid = bool(res.get("slid", False))
        allow  = float(res.get("allow", 0.0))
        allow2 = res.get("allow2", None)

        # Stable HUD keys (don’t flicker)
        _hud("XR.auth.fwd.sweep.src",    src,                           volatile=True)
        _hud("XR.auth.fwd.sweep.band",   band,                          volatile=True)
        _hud("XR.auth.fwd.sweep.pos_mm", float(round(pos_mm, 2)),       volatile=True)
        _hud("XR.auth.fwd.sweep.vΔ",     (None if vdelta is None else float(round(vdelta, 4))), volatile=True)
        _hud("XR.auth.fwd.sweep.slid",   ("Y" if slid else "N"),        volatile=True)
        _hud("XR.auth.fwd.sweep.allow",  float(round(allow, 4)),        volatile=True)
        _hud("XR.auth.fwd.sweep.allow2", (None if allow2 is None else float(round(allow2, 4))), volatile=True)
        _hud("XR.auth.fwd.sweep.lat_ms", float(round(lat_ms, 2)),       volatile=True)

        # Console print gated ONLY by forward-sweep toggle + Hz (not HUD visibility)
        if _ok_to_print_forward("_last_fwd_sweep_print"):
            a2txt = "—" if allow2 is None else f"{allow2:0.3f}"
            vdtxt = "—" if vdelta is None else f"{vdelta:0.4f}"
            print(
                f"[FWD.SWEEP] src={src} band={band} posΔ={pos_mm:0.2f} mm  "
                f"vΔ={vdtxt} m/s  slid={'Y' if slid else 'N'}  "
                f"allow={allow:0.3f}  allow2={a2txt}  lat={lat_ms:0.2f} ms"
            )

    xr_enqueue("kcc.forward_sweep_min3.v1", payload, _apply)
