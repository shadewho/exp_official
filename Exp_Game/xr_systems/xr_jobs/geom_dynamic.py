# Exp_Game/xr_systems/xr_jobs/geom_dynamic.py
# Stores LOCAL triangles per mover and caches transforms + inverses for parity/query jobs.

import time

# Per-mover local-space triangles and latest transform.
# id:int -> {"tris":[float], "n_tris":int,
#            "M":[16], "M3":[9], "Minv":[16], "Minv3":[9], "MinvT3":[9]}
_DYN = {}
_STATS = {"dyn_objs": 0, "dyn_tris": 0}

def _mat4_inv_gauss_rowmajor(m):
    # Gauss-Jordan for 4x4 row-major list[16]
    A = [[float(m[0]), float(m[1]), float(m[2]),  float(m[3]) ],
         [float(m[4]), float(m[5]), float(m[6]),  float(m[7]) ],
         [float(m[8]), float(m[9]), float(m[10]), float(m[11])],
         [float(m[12]),float(m[13]),float(m[14]), float(m[15])]]
    I = [[1.0,0.0,0.0,0.0],
         [0.0,1.0,0.0,0.0],
         [0.0,0.0,1.0,0.0],
         [0.0,0.0,0.0,1.0]]
    for col in range(4):
        # pivot
        piv = col
        maxabs = abs(A[col][col])
        for r in range(col+1,4):
            v = abs(A[r][col])
            if v > maxabs:
                maxabs = v; piv = r
        if maxabs <= 1.0e-12:
            return None
        if piv != col:
            A[col], A[piv] = A[piv], A[col]
            I[col], I[piv] = I[piv], I[col]
        # scale row
        pivv = A[col][col]
        invp = 1.0 / pivv
        for c in range(4):
            A[col][c] *= invp
            I[col][c] *= invp
        # eliminate
        for r in range(4):
            if r == col: continue
            f = A[r][col]
            if f == 0.0: continue
            for c in range(4):
                A[r][c] -= f * A[col][c]
                I[r][c] -= f * I[col][c]
    # flatten row-major
    return [I[0][0],I[0][1],I[0][2],I[0][3],
            I[1][0],I[1][1],I[1][2],I[1][3],
            I[2][0],I[2][1],I[2][2],I[2][3],
            I[3][0],I[3][1],I[3][2],I[3][3]]

def _m3_from_m4_rowmajor(m):
    return [m[0],m[1],m[2],
            m[4],m[5],m[6],
            m[8],m[9],m[10]]

def _mat3_inv(m):
    # row-major 3x3 inverse via adjugate
    a,b,c,d,e,f,g,h,i = m
    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    if abs(det) <= 1.0e-12: return None
    inv = [ (e*i - f*h), -(b*i - c*h),  (b*f - c*e),
            -(d*i - f*g), (a*i - c*g), -(a*f - c*d),
            (d*h - e*g), -(a*h - b*g), (a*e - b*d) ]
    inv = [v / det for v in inv]
    return inv

def _transpose3(m):  # row-major
    return [m[0],m[3],m[6],
            m[1],m[4],m[7],
            m[2],m[5],m[8]]

def _init_dynamic(payload: dict) -> dict:
    mover_id = int(payload.get("id", -1))
    tris     = payload.get("tris") or []
    if mover_id < 0 or not isinstance(tris, list):
        return {"ok": False, "error": "bad_payload"}
    n_tris = len(tris) // 9
    if mover_id not in _DYN:
        _DYN[mover_id] = {"tris": tris, "n_tris": n_tris,
                          "M": None, "M3": None, "Minv": None, "Minv3": None, "MinvT3": None}
    else:
        _DYN[mover_id]["tris"]   = tris
        _DYN[mover_id]["n_tris"] = n_tris

    _STATS["dyn_objs"] = len(_DYN)
    _STATS["dyn_tris"] = sum(e["n_tris"] for e in _DYN.values())

    return {"ok": True,
            "dyn_objs": int(_STATS["dyn_objs"]),
            "dyn_tris": int(_STATS["dyn_tris"]),
            "added_tris": int(n_tris)}

def _update_xforms(payload: dict) -> dict:
    """
    Payload: {batch:[{"id":int, "M":[16]}, ...]}
    We compute/caches Minv, Minv3, MinvT3 and M3 for each mover.
    """
    batch = payload.get("batch") or []
    updated = 0
    for rec in batch:
        try:
            mover_id = int(rec.get("id", -1))
            M = rec.get("M")
            if mover_id not in _DYN or not isinstance(M, list) or len(M) != 16:
                continue
            M4 = [float(x) for x in M]
            Minv = _mat4_inv_gauss_rowmajor(M4)
            if Minv is None:
                continue
            M3 = _m3_from_m4_rowmajor(M4)
            Minv3 = _m3_from_m4_rowmajor(Minv)
            MinvT3 = _transpose3(Minv3)
            ent = _DYN[mover_id]
            ent["M"] = M4
            ent["M3"] = M3
            ent["Minv"] = Minv
            ent["Minv3"] = Minv3
            ent["MinvT3"] = MinvT3
            updated += 1
        except Exception:
            continue
    return {"ok": True, "updated": int(updated)}

def register(register_job):
    register_job("geom.init_dynamic.v1",  _init_dynamic)
    register_job("geom.update_xforms.v1", _update_xforms)

# Expose store for other jobs
__all__ = ["_DYN"]
