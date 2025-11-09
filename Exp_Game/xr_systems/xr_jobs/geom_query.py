# Exp_Game/xr_systems/xr_jobs/geom_query.py
# DEV-only parity: dynamic-only ray cast using local-space triangles + cached xforms.

import math

try:
    # share dynamic store from geom_dynamic
    from .geom_dynamic import _DYN
except Exception:
    _DYN = {}

_EPS = 1.0e-9

def _len3(v):  # v=(x,y,z)
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def _norm3(v):
    l = _len3(v)
    if l <= _EPS: return (0.0, 0.0, 0.0), 0.0
    inv = 1.0 / l
    return (v[0]*inv, v[1]*inv, v[2]*inv), l

def _dot(a,b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def _m3_mul_v(m,v):
    # m is row-major [0..8]
    return (m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
            m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
            m[6]*v[0] + m[7]*v[1] + m[8]*v[2])

def _m4_mul_point_rowmajor(m,p):
    # m row-major 4x4, p=(x,y,z), assume w=1
    x = m[0]*p[0] + m[1]*p[1] + m[2]*p[2] + m[3]
    y = m[4]*p[0] + m[5]*p[1] + m[6]*p[2] + m[7]
    z = m[8]*p[0] + m[9]*p[1] + m[10]*p[2] + m[11]
    w = m[12]*p[0] + m[13]*p[1] + m[14]*p[2] + m[15]
    if abs(w) > _EPS:
        inv = 1.0 / w
        return (x*inv, y*inv, z*inv)
    return (x, y, z)

def _ray_tri_mt(o, d, v0, v1, v2, tmax):
    # Möller–Trumbore; returns t or None
    e1 = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
    e2 = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
    pvec = _cross(d, e2)
    det = _dot(e1, pvec)
    if abs(det) < 1.0e-9:
        return None
    inv_det = 1.0 / det
    tvec = (o[0]-v0[0], o[1]-v0[1], o[2]-v0[2])
    u = _dot(tvec, pvec) * inv_det
    if u < 0.0 or u > 1.0:
        return None
    qvec = _cross(tvec, e1)
    v = _dot(d, qvec) * inv_det
    if v < 0.0 or (u + v) > 1.0:
        return None
    t = _dot(e2, qvec) * inv_det
    if t < 0.0 or t > tmax:
        return None
    return t

def _build_tuple3(a,b,c):
    return (float(a), float(b), float(c))

def _ray_dynamic(payload: dict) -> dict:
    """
    Inputs:
      origin: (x,y,z) world
      dir:    (x,y,z) world (need not be normalized)
      max_d:  float  (world)
    Output:
      {hit: bool, dist: float, normal: (x,y,z), id: int} or {hit: False}
    """
    o = payload.get("origin", None)
    d = payload.get("dir", None)
    max_d = float(payload.get("max_d", 0.0))
    if (not isinstance(o, (list, tuple)) or len(o) != 3 or
        not isinstance(d, (list, tuple)) or len(d) != 3 or
        max_d <= 1.0e-9):
        return {"hit": False}

    # normalize world dir
    d, _ = _norm3((float(d[0]), float(d[1]), float(d[2])))
    if _len3(d) <= _EPS:
        return {"hit": False}
    o = (float(o[0]), float(o[1]), float(o[2]))

    best_world_dist = None
    best_world_norm = None
    best_id = None

    for mover_id, entry in _DYN.items():
        tris = entry.get("tris")
        Rinv = entry.get("Minv3")      # 3x3 inverse (row-major)
        RinvT= entry.get("MinvT3")     # transpose of inverse
        M3   = entry.get("M3")         # 3x3 top-left of M
        Minv = entry.get("Minv")       # 4x4 full inverse (row-major)
        if not tris or not Rinv or not RinvT or not Minv or not M3:
            continue

        # world -> local
        o_l = _m4_mul_point_rowmajor(Minv, o)
        d_l = _m3_mul_v(Rinv, d)
        d_l, _ = _norm3(d_l)
        if _len3(d_l) <= _EPS:
            continue

        # map world max_d to local step length
        step_l = _m3_mul_v(Rinv, (d[0]*max_d, d[1]*max_d, d[2]*max_d))
        max_l = _len3(step_l)
        if max_l <= _EPS:
            continue

        n = len(tris) // 9
        best_t_l = None
        best_n_l = None
        for i in range(n):
            base = i * 9
            v0 = _build_tuple3(tris[base+0], tris[base+1], tris[base+2])
            v1 = _build_tuple3(tris[base+3], tris[base+4], tris[base+5])
            v2 = _build_tuple3(tris[base+6], tris[base+7], tris[base+8])
            t = _ray_tri_mt(o_l, d_l, v0, v1, v2, max_l)
            if t is None:
                continue
            if (best_t_l is None) or (t < best_t_l):
                # local face normal (not area-weighted; fine for parity)
                e1 = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
                e2 = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
                n_l = _cross(e1, e2)
                best_t_l = t
                best_n_l = n_l

        if best_t_l is None:
            continue

        # local t -> world distance using M3
        vec_l = (d_l[0]*best_t_l, d_l[1]*best_t_l, d_l[2]*best_t_l)
        vec_w = _m3_mul_v(M3, vec_l)
        dist_w = _len3(vec_w)
        if dist_w <= _EPS:
            continue

        # transform normal with inverse-transpose
        n_w = _m3_mul_v(RinvT, best_n_l)
        n_w, _ = _norm3(n_w)
        # Flip to oppose ray direction to avoid sign mismatches
        if _dot(n_w, d) > 0.0:
            n_w = (-n_w[0], -n_w[1], -n_w[2])

        if (best_world_dist is None) or (dist_w < best_world_dist):
            best_world_dist = dist_w
            best_world_norm = n_w
            best_id = int(mover_id)

    if best_world_dist is None:
        return {"hit": False}
    return {"hit": True, "dist": float(best_world_dist), "normal": best_world_norm, "id": best_id}

def register(register_job):
    register_job("geom.ray_dynamic.v1", _ray_dynamic)
