# Exp_Game/xr_systems/xr_jobs/geom_static.py
import time

# Module-level store for static triangles uploaded in chunks.
_STATIC = {
    "tris": None,   # flat [x,y,z, ...]
    "n_tris": 0,
    "mem_MB": 0.0,
    "t0": 0.0,      # upload start time
}

def _init_static(payload: dict) -> dict:
    """
    Back-compat single-shot init (unused after we switch to chunked upload).
    Kept so older callers won't crash.
    """
    t0 = time.perf_counter()
    tris = payload.get("tris") or []
    n_f = len(tris)
    n_tris = n_f // 9
    _STATIC["tris"] = tris
    _STATIC["n_tris"] = int(n_tris)
    est_bytes = n_f * 8
    mem_MB = est_bytes / (1024.0 * 1024.0)
    _STATIC["mem_MB"] = float(mem_MB)
    build_ms = (time.perf_counter() - t0) * 1000.0
    return {"static_tris": int(n_tris), "build_ms": float(build_ms), "mem_MB": float(mem_MB)}

def _begin(payload: dict) -> dict:
    _STATIC["tris"] = []
    _STATIC["n_tris"] = 0
    _STATIC["mem_MB"] = 0.0
    _STATIC["t0"] = time.perf_counter()
    return {"ok": True}

def _chunk(payload: dict) -> dict:
    tris = payload.get("tris") or []
    buf = _STATIC.get("tris")
    if buf is None:
        # Begin was not called; initialize defensively.
        _begin({})
        buf = _STATIC["tris"]
    buf.extend(tris)
    n_tris = len(buf) // 9
    _STATIC["n_tris"] = n_tris
    return {"added_tris": int(len(tris) // 9), "total_tris": int(n_tris)}

def _end(payload: dict) -> dict:
    if _STATIC.get("tris") is None:
        _begin({})
    n_f = len(_STATIC["tris"])
    n_tris = n_f // 9
    est_bytes = n_f * 8
    mem_MB = est_bytes / (1024.0 * 1024.0)
    _STATIC["n_tris"] = int(n_tris)
    _STATIC["mem_MB"] = float(mem_MB)
    build_ms = (time.perf_counter() - float(_STATIC.get("t0", time.perf_counter()))) * 1000.0
    return {"static_tris": int(n_tris), "build_ms": float(build_ms), "mem_MB": float(mem_MB)}

def register(register_job):
    # Back-compat single-shot
    register_job("geom.init_static.v1", _init_static)
    # Streaming API
    register_job("geom.init_static.begin.v1", _begin)
    register_job("geom.init_static.chunk.v1", _chunk)
    register_job("geom.init_static.end.v1",   _end)
