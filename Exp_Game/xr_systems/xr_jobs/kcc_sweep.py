# Exp_Game/xr_systems/xr_jobs/kcc_sweep.py
# -----------------------------------------------------------------------------
# XR JOBS: Authoritative, self-contained math for forward sweep parity/mirror.
# All calculations for _forward_sweep_min3 live here.
# -----------------------------------------------------------------------------
import math

# Reuse the exact union casting used by XR geometry
from .geom_union import _static_best, _dynamic_best

_EPS = 1.0e-12
_UP  = (0.0, 0.0, 1.0)

def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def _len3(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def _norm3(v):
    l = _len3(v)
    if l <= _EPS:
        return (0.0, 0.0, 0.0), 0.0
    inv = 1.0 / l
    return (v[0]*inv, v[1]*inv, v[2]*inv), l

def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _mad(a, s, b):  # a + s*b
    return (a[0] + s*b[0], a[1] + s*b[1], a[2] + s*b[2])

def _triplet_union(base, d_norm, r, h, step_len):
    """
    Feet/Mid/Head union pick; returns (hit, best_d, best_n, src, band).
    src in {"STATIC","DYNAMIC"}, band in {"low","mid","high"}.
    """
    midz = _clamp(h * 0.5, r, h - r)
    bands = [("low", r), ("mid", midz), ("high", h - r)]
    max_d = step_len + r

    best = None
    for label, z in bands:
        o = (base[0], base[1], base[2] + z)
        sb = _static_best(o, d_norm, max_d)
        db = _dynamic_best(o, d_norm, max_d)

        pick = None
        if sb and db:
            pick = sb if sb[1] <= db[1] else db
        elif sb:
            pick = sb
        elif db:
            pick = db

        if pick:
            src = "STATIC" if pick[0] == "static" else "DYNAMIC"
            dist = float(pick[1])
            n = pick[2]
            if _dot(n, d_norm) > 0.0:  # face against ray
                n = (-n[0], -n[1], -n[2])
            if (best is None) or (dist < best[1]):
                best = (label, dist, n, src)

    if best is None:
        return (False, None, None, None, None)

    label, dist, n, src = best
    ln = math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]) or 1.0
    n = (n[0]/ln, n[1]/ln, n[2]/ln)
    return (True, dist, n, src, label)

def _remove_normal_from_xy(vx, vy, n):
    """hvel=(vx,vy,0) minus component along n; returns (vx,vy)."""
    vn = vx*n[0] + vy*n[1] + 0.0*n[2]
    if vn > 0.0:
        vx -= n[0] * vn
        vy -= n[1] * vn
    return vx, vy

def _forward_sweep_min3_job(payload: dict) -> dict:
    """
    XR mirror of Blender _forward_sweep_min3:
      • three forward rays (low/mid/high) for block
      • clamp to contact; remove XY normal component from velocity
      • single optional slide attempt (angle gate; floor clamp)

    Inputs:
      pos: (x,y,z) base
      dir: (x,y,z) (need not be normalized)
      r:   float radius
      h:   float height
      step_len: float
      floor_cos: float  (cos(slope_limit_deg))
      vel_xy: (vx,vy)   (for parity of side-effect)

    Output:
      {hit:bool, src:str|None, band:str|None,
       pos:(x,y,z), allow:float, slid:bool, allow2:float|None,
       vel_xy_after:(vx,vy), n:(nx,ny,nz)|None}
    """
    pos = payload.get("pos", None)
    d   = payload.get("dir", None)
    r   = float(payload.get("r", 0.0))
    h   = float(payload.get("h", 0.0))
    step_len  = float(payload.get("step_len", 0.0))
    floor_cos = float(payload.get("floor_cos", 0.7))
    vxy = payload.get("vel_xy", (0.0, 0.0))

    if (not isinstance(pos, (list, tuple)) or len(pos) != 3 or
        not isinstance(d,   (list, tuple)) or len(d)   != 3 or
        step_len <= 1.0e-9 or r <= 0.0 or h <= (2.0*r + 1.0e-9)):
        return {"hit": False, "pos": (float(pos[0]) if pos else 0.0,
                                      float(pos[1]) if pos else 0.0,
                                      float(pos[2]) if pos else 0.0),
                "allow": 0.0, "slid": False, "allow2": None,
                "vel_xy_after": tuple(vxy) if isinstance(vxy,(list,tuple)) else (0.0,0.0),
                "src": None, "band": None, "n": None}

    d, _ = _norm3((float(d[0]), float(d[1]), float(d[2])))
    if _len3(d) <= _EPS:
        return {"hit": False, "pos": (float(pos[0]), float(pos[1]), float(pos[2])),
                "allow": 0.0, "slid": False, "allow2": None,
                "vel_xy_after": tuple(vxy) if isinstance(vxy,(list,tuple)) else (0.0,0.0),
                "src": None, "band": None, "n": None}

    base = (float(pos[0]), float(pos[1]), float(pos[2]))
    vx, vy = (float(vxy[0]) if isinstance(vxy,(list,tuple)) and len(vxy)==2 else 0.0,
              float(vxy[1]) if isinstance(vxy,(list,tuple)) and len(vxy)==2 else 0.0)

    # ---- primary triplet ----
    hit, best_d, best_n, src, band = _triplet_union(base, d, r, h, step_len)
    if not hit:
        new_pos = _mad(base, step_len, d)
        return {"hit": False, "pos": new_pos, "allow": step_len,
                "slid": False, "allow2": None, "vel_xy_after": (vx, vy),
                "src": None, "band": None, "n": None}

    # Clamp to contact (minus radius)
    allow = max(0.0, float(best_d) - r)
    moved = base if allow <= 1.0e-9 else _mad(base, allow, d)

    # Remove normal component from XY velocity
    vx, vy = _remove_normal_from_xy(vx, vy, best_n)

    # ---- decide if slide ----
    remaining = max(0.0, step_len - allow)
    if remaining <= (0.15 * r):
        return {"hit": True, "src": src, "band": band, "n": best_n,
                "pos": moved, "allow": allow, "slid": False, "allow2": None,
                "vel_xy_after": (vx, vy)}

    # angle gate: 20°..85°
    dot = abs(_dot(d, best_n))
    dot = 0.0 if dot < 0.0 else (1.0 if dot > 1.0 else dot)
    theta_deg = math.degrees(math.acos(dot))
    if not (20.0 <= theta_deg <= 85.0):
        return {"hit": True, "src": src, "band": band, "n": best_n,
                "pos": moved, "allow": allow, "slid": False, "allow2": None,
                "vel_xy_after": (vx, vy)}

    # tangent slide dir
    proj = _dot(d, best_n)
    slide = (d[0] - best_n[0]*proj, d[1] - best_n[1]*proj, d[2] - best_n[2]*proj)
    # clamp to XY if too steep
    if best_n[2] < floor_cos:  # n·up < floor_cos
        slide = (slide[0], slide[1], 0.0)
    slide, _ = _norm3(slide)
    if _len3(slide) <= _EPS:
        return {"hit": True, "src": src, "band": band, "n": best_n,
                "pos": moved, "allow": allow, "slid": False, "allow2": None,
                "vel_xy_after": (vx, vy)}

    # slow during slide
    remaining *= 0.65
    vx *= 0.65; vy *= 0.65

    # one slide attempt with same 3-ray scheme
    hit2, d2, _n2, _src2, _band2 = _triplet_union(moved, slide, r, h, remaining)
    if not hit2:
        end_pos = _mad(moved, remaining, slide)
        return {"hit": True, "src": src, "band": band, "n": best_n,
                "pos": end_pos, "allow": allow, "slid": True, "allow2": remaining,
                "vel_xy_after": (vx, vy)}

    allow2 = max(0.0, float(d2) - r)
    end_pos = moved if allow2 <= 1.0e-9 else _mad(moved, allow2, slide)
    return {"hit": True, "src": src, "band": band, "n": best_n,
            "pos": end_pos, "allow": allow, "slid": True, "allow2": allow2,
            "vel_xy_after": (vx, vy)}

def register(register_job):
    register_job("kcc.forward_sweep_min3.v1", _forward_sweep_min3_job)
