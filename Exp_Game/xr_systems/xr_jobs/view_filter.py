# Exploratory/Exp_Game/xr_systems/xr_jobs/view_filter.py
# XR-side filtering & latching for third-person view distance.
#
# Moves the final boom distance decision-making into XR (no geometry).
# • Outward growth smoothing (m/s clamp)
# • Short latch/hold when we just hit something (prevents “pogo”)
# • Release pad based on camera thickness
# • Jitter metric (|Δallowed| / s) with EMA for DevHUD
#
# Inputs (payload):
#   op_key: str     unique per Blender operator instance
#   anchor: (x,y,z) unused here (kept for parity / future)
#   dir:    (x,y,z) unused here
#   min_cam: float
#   desired_max: float
#   r_cam: float
#   candidate_allowed: float   # from Blender’s local geometry pass
#   hit_token: str|None        # "__STATIC__", object name, or None
#
# Output:
#   { allowed: float, hit_token: str|None, jitter_ema: float }

import time

# Per-operator state (latch, smoother, jitter)
_STATE = {}  # op_key -> dict

# Tunables — match your Blender-side feel
_OUTWARD_RATE_MPS = 10.0   # outward growth clamp (meters per second)
_LATCH_HOLD_S     = 0.14   # how long to hold after a hit
_REL_PAD_MIN      = 0.06   # minimum release pad (meters)
_REL_PAD_K        = 1.6    # pad factor vs camera radius r_cam
_EMA_A            = 0.10   # jitter EMA alpha

def _now():
    return time.perf_counter()

def _get(op_key: str):
    st = _STATE.get(op_key)
    if st is None:
        st = {
            "last_allowed": None,
            "last_t": _now(),
            "latched_until": 0.0,
            "latched_allowed": None,
            "latched_obj": None,
            "jitter_ema": 0.0,
        }
        _STATE[op_key] = st
    return st

def _clamp(v, lo, hi):
    if v < lo: return lo
    if v > hi: return hi
    return v

def _view_third_filter(payload: dict) -> dict:
    op_key   = str(payload.get("op_key", "0"))
    min_cam  = float(payload.get("min_cam", 0.01))
    dmax     = float(payload.get("desired_max", 6.0))
    r_cam    = float(payload.get("r_cam", 0.01))
    cand     = float(payload.get("candidate_allowed", min_cam))
    hit_tok  = payload.get("hit_token", None)
    hit_token = str(hit_tok) if isinstance(hit_tok, str) else None

    # clamp candidate to legal range
    def _clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v
    cand = _clamp(cand, min_cam, max(min_cam, dmax))

    st  = _get(op_key)
    now = _now()

    # latch
    if hit_token is not None:
        st["latched_obj"]     = hit_token
        st["latched_allowed"] = cand
        st["latched_until"]   = now + _LATCH_HOLD_S
        allowed = cand
    else:
        if st["latched_obj"] is not None and st["latched_allowed"] is not None:
            pad = max(_REL_PAD_MIN, _REL_PAD_K * r_cam)
            if now < st["latched_until"] and cand < (st["latched_allowed"] + pad):
                allowed = st["latched_allowed"]
            else:
                st["latched_obj"] = None
                st["latched_allowed"] = None
                st["latched_until"] = 0.0
                allowed = cand
        else:
            allowed = cand

    # never exceed the candidate
    if allowed > cand:
        allowed = cand

    # outward smoothing
    last_allowed = st["last_allowed"]
    dt = max(1e-6, now - st["last_t"])
    if last_allowed is None:
        smoothed = allowed
    else:
        if allowed >= last_allowed:
            smoothed = min(allowed, last_allowed + _OUTWARD_RATE_MPS * dt)
        else:
            smoothed = allowed

    # second safety
    if smoothed > cand:
        smoothed = cand

    # jitter ema
    if last_allowed is None:
        je = st["jitter_ema"]
    else:
        jitter = abs(smoothed - last_allowed) / dt
        a = float(_EMA_A)
        je = (1.0 - a) * float(st["jitter_ema"]) + a * float(jitter)

    st["last_allowed"] = float(smoothed)
    st["last_t"]       = now
    st["jitter_ema"]   = float(je)

    return {"allowed": float(smoothed), "hit_token": hit_token, "jitter_ema": float(je)}



def register(register_job):
    register_job("view.third.filter.v1", _view_third_filter)
