# Exp_Game/xr_systems/xr_jobs/view_third.py
import time

# Per-op state
_STATE = {}  # key -> dict(last_allowed, last_t, jitter_ema, latch_until, latched_allowed, latched_obj)

# Tunables: same intent as your local filters
OUTWARD_RATE     = 10.0   # m/s outward growth clamp; inward is immediate
HOLD_TIME        = 0.14   # seconds to hold a latched hit distance
RELEASE_PAD_MIN  = 0.06   # meters
RELEASE_PAD_K    = 1.6    # * r_cam

def _state(key: str):
    s = _STATE.get(key)
    if s is None:
        s = {
            "last_allowed": None,
            "last_t": time.perf_counter(),
            "jitter_ema": 0.0,
            "latch_until": 0.0,
            "latched_allowed": None,
            "latched_obj": None,
        }
        _STATE[key] = s
    return s

def _filter_allowed(s, cand: float, r_cam: float, hit_token):
    now = time.perf_counter()
    dt = max(1e-6, now - s["last_t"])
    last = s["last_allowed"]

    # Outward-only growth clamp
    if last is None:
        smooth = cand
    elif cand >= last:
        smooth = min(cand, last + OUTWARD_RATE * dt)
    else:
        smooth = cand  # immediate inward pull

    # Latch behavior
    latched = False
    if hit_token is not None:
        s["latched_obj"] = hit_token
        s["latched_allowed"] = smooth
        s["latch_until"] = now + HOLD_TIME
        allowed = smooth
        latched = True
    else:
        if s["latched_obj"] is not None and s["latched_allowed"] is not None and now < s["latch_until"]:
            need_pad = max(RELEASE_PAD_MIN, RELEASE_PAD_K * r_cam)
            if smooth < (s["latched_allowed"] + need_pad):
                allowed = s["latched_allowed"]
                latched = True
            else:
                s["latched_obj"] = None
                s["latched_allowed"] = None
                s["latch_until"] = 0.0
                allowed = smooth
        else:
            s["latched_obj"] = None
            s["latched_allowed"] = None
            s["latch_until"] = 0.0
            allowed = smooth

    # Jitter EMA in m/s
    if last is None:
        jitter = 0.0
    else:
        jitter = abs(allowed - last) / dt
    s["jitter_ema"] = jitter if s["jitter_ema"] == 0.0 else (s["jitter_ema"] * 0.85 + jitter * 0.15)

    s["last_allowed"] = allowed
    s["last_t"] = now
    return allowed, latched, dt

def _view_third_filter(payload: dict):
    key           = str(payload.get("op_key", "0"))
    min_cam       = float(payload.get("min_cam", 0.0))
    desired_max   = float(payload.get("desired_max", 0.0))
    r_cam         = float(payload.get("r_cam", 0.01))
    cand          = float(payload.get("candidate_allowed", 0.0))
    hit_token     = payload.get("hit_token", None)

    # Safety clamp to input bounds
    cand = max(min_cam, min(desired_max, cand))

    s = _state(key)
    allowed, latched, dt = _filter_allowed(s, cand, r_cam, hit_token)

    return {
        "allowed": float(allowed),
        "latched": bool(latched),
        "jitter_ema": float(s["jitter_ema"]),
        "dt_ms": float(dt * 1000.0),
        "hit_token": (str(hit_token) if hit_token is not None else None),
    }

def register(register_job):
    # Pure decision/filter job — geometry stays on Blender
    register_job("view.third.filter.v1", _view_third_filter)
