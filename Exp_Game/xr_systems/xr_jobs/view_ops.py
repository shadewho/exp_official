# Exploratory/Exp_Game/xr_systems/xr_jobs/view_ops.py
# XR-side stateful latch + smoothing for third-person boom distance.

import time

_OUTWARD_RATE_MPS = 10.0        # outward growth clamp (m/s) — same behavior as your _CamSmoother
_LATCH_HOLD_S     = 0.14        # hold time after a hit (s)   — same as your _CamLatch
_LATCH_PAD_MIN    = 0.06        # absolute release pad (m)
_LATCH_PAD_K      = 1.6         # pad = max(0.06, 1.6 * r_cam)

# Per-operator state (keyed by op_id)
_STATE = {}  # op_id -> dict(last_allowed, t_last, latched_obj, latched_until, latched_allowed)

def _st(op_id: int):
    s = _STATE.get(op_id)
    if s is None:
        s = {
            "last_allowed": None,
            "t_last": time.perf_counter(),
            "latched_obj": None,
            "latched_until": 0.0,
            "latched_allowed": None,
        }
        _STATE[op_id] = s
    return s

def _clamp(v, lo, hi):
    if v < lo: return lo
    if v > hi: return hi
    return v

def _apply_latch(s, hit_token: str | None, allowed_now: float, r_cam: float):
    now = time.perf_counter()

    if hit_token is not None:
        s["latched_obj"]     = hit_token
        s["latched_until"]   = now + _LATCH_HOLD_S
        s["latched_allowed"] = float(allowed_now)
        return float(allowed_now), True

    latched_obj   = s.get("latched_obj")
    latched_until = float(s.get("latched_until", 0.0))
    latched_val   = s.get("latched_allowed")

    if latched_obj is not None and latched_val is not None:
        pad = max(_LATCH_PAD_MIN, _LATCH_PAD_K * float(r_cam))
        if (now < latched_until) and (allowed_now < (float(latched_val) + pad)):
            return float(latched_val), True
        # release
        s["latched_obj"] = None
        s["latched_until"] = 0.0
        s["latched_allowed"] = None
    return allowed_now, False

def _apply_smoothing(s, target: float):
    now = time.perf_counter()
    last = s.get("last_allowed", None)
    dt   = max(1.0e-6, now - float(s.get("t_last", now)))

    if last is None:
        s["last_allowed"] = float(target)
    elif target >= last:
        s["last_allowed"] = min(float(target), float(last) + _OUTWARD_RATE_MPS * dt)
    else:
        # immediate pull-in
        s["last_allowed"] = float(target)

    s["t_last"] = now
    return float(s["last_allowed"])

def _solve(payload: dict) -> dict:
    op_id       = int(payload.get("op_id", 0))
    cand        = float(payload.get("candidate", 0.0))
    min_cam     = float(payload.get("min_cam", 0.0))
    desired_max = float(payload.get("desired_max", 0.0))
    r_cam       = float(payload.get("r_cam", 0.0))
    hit_token   = payload.get("hit_token", None)
    if not isinstance(hit_token, str):
        hit_token = None

    # Clamp candidate to safety bounds from Blender
    cand = _clamp(cand, min_cam, desired_max)

    s = _st(op_id)
    latched, _ = _apply_latch(s, hit_token, cand, r_cam)
    smooth     = _apply_smoothing(s, latched)
    final      = _clamp(smooth, min_cam, desired_max)

    return {"allowed": float(final)}

def register(register_job):
    register_job("view.solve_third.v1", _solve)
