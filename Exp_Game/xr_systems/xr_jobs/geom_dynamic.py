# Exp_Game/xr_systems/xr_jobs/geom_dynamic.py
import time

# Per-mover local-space triangles and latest transform.
_DYN = {}  # id:int -> {"tris":[float], "n_tris":int, "M":[16]|None}
_STATS = {"dyn_objs": 0, "dyn_tris": 0}

def _init_dynamic(payload: dict) -> dict:
    """
    Register one dynamic mover's LOCAL-SPACE triangles.
    Payload: {id:int, tris:[x,y,z,...]}
    """
    mover_id = int(payload.get("id", -1))
    tris     = payload.get("tris") or []
    if mover_id < 0 or not isinstance(tris, list):
        return {"ok": False, "error": "bad_payload"}

    n_tris = len(tris) // 9
    if mover_id not in _DYN:
        _DYN[mover_id] = {"tris": tris, "n_tris": n_tris, "M": None}
    else:
        # Idempotent: allow re-init to replace geometry (rare)
        _DYN[mover_id]["tris"]   = tris
        _DYN[mover_id]["n_tris"] = n_tris

    # Recompute stats
    _STATS["dyn_objs"] = len(_DYN)
    _STATS["dyn_tris"] = sum(entry["n_tris"] for entry in _DYN.values())

    return {
        "ok": True,
        "dyn_objs": int(_STATS["dyn_objs"]),
        "dyn_tris": int(_STATS["dyn_tris"]),
        "added_tris": int(n_tris),
    }

def _update_xforms(payload: dict) -> dict:
    """
    Batch update transforms. Payload: {batch:[{"id":int, "M":[16]}, ...]}
    We only store the 4x4 row-major matrix; inverses will be computed later when needed.
    """
    batch = payload.get("batch") or []
    updated = 0
    for rec in batch:
        try:
            mover_id = int(rec.get("id", -1))
            M = rec.get("M")
            if mover_id in _DYN and isinstance(M, list) and len(M) == 16:
                _DYN[mover_id]["M"] = [float(v) for v in M]
                updated += 1
        except Exception:
            continue
    return {"ok": True, "updated": int(updated)}

def register(register_job):
    register_job("geom.init_dynamic.v1",  _init_dynamic)
    register_job("geom.update_xforms.v1", _update_xforms)
