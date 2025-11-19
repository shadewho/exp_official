# Exp_Game/xr_systems/xr_ports/geom.py
from __future__ import annotations
import time
import bpy
from ..xr_queue import xr_enqueue

# ----------------------
# Console gating / HUD
# ----------------------
def _ok_to_print(tag_attr: str, hz_prop: str, enable_prop: str) -> bool:
    try:
        sc = bpy.context.scene
        if not sc or not getattr(sc, "dev_hud_enable", False):
            return False
        if not getattr(sc, "dev_hud_show_xr", False):
            return False
        if not getattr(sc, "dev_hud_log_console", True):  # MASTER GATE
            return False
        if not getattr(sc, enable_prop, False):
            return False
        hz = float(getattr(sc, hz_prop, 1.0) or 1.0)
        now = time.perf_counter()
        last = float(getattr(_ok_to_print, tag_attr, 0.0) or 0.0)
        if (now - last) >= (1.0 / max(0.1, hz)):
            setattr(_ok_to_print, tag_attr, now)
            return True
    except Exception:
        pass
    return False


_LAST_XF_SEQ = -1
def _mark_last_xf_seq(seq: int):
    global _LAST_XF_SEQ
    try: _LAST_XF_SEQ = int(seq)
    except Exception: _LAST_XF_SEQ = -1

def get_last_xf_seq() -> int:
    return int(_LAST_XF_SEQ)

def _hud_set(key: str, val, volatile=False):
    try:
        from ...Developers.exp_dev_interface import devhud_set
        devhud_set(key, val, volatile=bool(volatile))
    except Exception:
        pass
# ---------------------------------
# STATIC world triangles (streamed)
# ---------------------------------
def start_static_upload_from_meshes(op, mesh_objs, *, chunk_tris: int = 2048):
    """Collect world-space tris and begin chunked upload to XR."""
    from ...physics.exp_raycastutils import collect_world_triangles
    tris = collect_world_triangles(mesh_objs)
    total_f = len(tris)

    op._xr_geom_upload = {
        "tris": tris,
        "off": 0,
        "total_floats": total_f,
        "total_tris": total_f // 9,
        "chunk_floats": int(max(1, int(chunk_tris))) * 9,
        "ended": False,
        "started": True,
        "t0": time.perf_counter(),
    }

    _hud_set("XR.geom.static_tris", 0)
    _hud_set("XR.geom.build_ms", 0.0)
    _hud_set("XR.geom.mem_MB", 0.0)
    _hud_set("XR.geom.uploaded_tris", 0)
    _hud_set("XR.geom.upload_pct", "0.0%")

    xr_enqueue("geom.init_static.begin.v1", {}, None)

def step_static_upload(op, *, chunks_per_frame: int = 3):
    """Send a few chunks per frame; finalize and publish HUD stats on end."""
    st = getattr(op, "_xr_geom_upload", None)
    if not st or st.get("ended", False):
        return

    total, off, chunk = int(st["total_floats"]), int(st["off"]), int(st["chunk_floats"])

    def _progress(sent_tris, total_tris):
        try:
            from ...Developers.exp_dev_interface import devhud_set
            pct = (100.0 * float(sent_tris) / max(1, int(total_tris))) if total_tris > 0 else 0.0
            devhud_set("XR.geom.upload_pct", f"{pct:0.1f}%", volatile=True)
            devhud_set("XR.geom.uploaded_tris", int(sent_tris), volatile=True)
        except Exception:
            pass

    sent = 0
    while sent < int(max(1, chunks_per_frame)) and off < total:
        end = min(off + chunk, total)
        xr_enqueue("geom.init_static.chunk.v1", {"tris": st["tris"][off:end]}, None)
        off = end
        sent += 1
        _progress(off // 9, int(st["total_tris"]))

    st["off"] = off

    if off >= total and not st["ended"]:
        t_send = time.perf_counter()

        def _apply_end(res: dict):
            static_tris = int(res.get("static_tris", 0))
            build_ms    = round(float(res.get("build_ms", 0.0)), 1)
            mem_MB      = round(float(res.get("mem_MB", 0.0)), 2)
            _hud_set("XR.geom.static_tris", static_tris)
            _hud_set("XR.geom.build_ms",    build_ms)
            _hud_set("XR.geom.mem_MB",      mem_MB)
            _hud_set("XR.geom.upload_pct",  "100.0%")
            _hud_set("XR.geom.mode",        "STATIC_STORE")
            _hud_set("XR.geom.authority",   "BLENDER")

            if _ok_to_print("_last_geom_log", "dev_log_geom_hz", "dev_log_geom_console"):
                dt_ms = (time.perf_counter() - float(st.get("t0", t_send))) * 1000.0
                print(f"[GEOM.XR] static upload complete  static_tris={static_tris:,}  build={build_ms:0.1f} ms  mem={mem_MB:0.2f} MB  net={dt_ms:0.1f} ms")

            try: op._xr_geom_static_done = True
            except Exception: pass

        xr_enqueue("geom.init_static.end.v1", {}, _apply_end)
        st["ended"] = True

# ---------------------------------
# DYNAMIC movers (ids, init, xforms)
# ---------------------------------
def _rev_put(did: int, obj):
    """
    Put Blender object into module-level reverse map for this id.
    Try weakref; if unsupported, store a strong ref. Returns stored entry.
    """
    rev = globals().setdefault("_REV_MAP", {})
    kind = globals().get("_REV_KIND", None)
    try:
        import weakref
        entry = weakref.ref(obj)  # many bpy objects do NOT support this
        rev[int(did)] = entry
        globals()["_REV_KIND"] = "weak"
        kind = "weak"
    except Exception:
        # fallback: strong reference (safe for your finite set of movers)
        rev[int(did)] = obj
        globals()["_REV_KIND"] = "strong"
        kind = "strong"
    try:
        # HUD breadcrumbs
        _hud_set("XR.geom.revmap", int(len(rev)), volatile=True)
        _hud_set("XR.geom.revmap_kind", str(kind), volatile=True)
    except Exception:
        pass
    return rev[int(did)]

def _rev_get(did: int):
    """
    Get Blender object from reverse map. Handles weakref or strong ref.
    """
    ent = globals().get("_REV_MAP", {}).get(int(did))
    if ent is None:
        return None
    # weakref.ref is callable → returns obj or None
    try:
        return ent() if callable(ent) else ent
    except Exception:
        return None

def _revmap_len():
    try:
        return len(globals().get("_REV_MAP", {}) or {})
    except Exception:
        return 0
def _ensure_dyn_maps(op):
    if not hasattr(op, "_xr_dyn_id_map"):
        op._xr_dyn_id_map = {}   # obj -> int id
    if not hasattr(op, "_xr_dyn_id_seq"):
        op._xr_dyn_id_seq = 1
    if not hasattr(op, "_xr_dyn_inited"):
        op._xr_dyn_inited = set()
    # module-level reverse map bucket
    if "_REV_MAP" not in globals():
        globals()["_REV_MAP"] = {}
        globals()["_REV_KIND"] = "unset"


def ensure_dyn_id(op, obj) -> int:
    _ensure_dyn_maps(op)
    if obj in op._xr_dyn_id_map:
        did = int(op._xr_dyn_id_map[obj])
    else:
        did = int(op._xr_dyn_id_seq)
        op._xr_dyn_id_seq = did + 1
        op._xr_dyn_id_map[obj] = did
    # put/refresh reverse mapping (weak if possible, else strong)
    _rev_put(did, obj)
    return did

def _backfill_revmap_from_op(op) -> int:
    """
    Ensure reverse map has entries for all ids in op._xr_dyn_id_map.
    Safe to call every frame. Returns revmap size or -1 on error.
    """
    try:
        _ensure_dyn_maps(op)
        m = getattr(op, "_xr_dyn_id_map", {}) or {}
        filled = 0
        for obj, did in m.items():
            if did is None:
                continue
            cur = globals().get("_REV_MAP", {}).get(int(did))
            ok = (cur is not None) and (not callable(cur) or cur() is not None)
            if not ok:
                _rev_put(int(did), obj)
                filled += 1
        if filled:
            _hud_set("XR.geom.revmap_filled", int(filled), volatile=True)
        return _revmap_len()
    except Exception:
        return -1

def dyn_obj_from_id(did: int):
    return _rev_get(int(did))



def dyn_obj_from_id(did: int):
    try:
        wr = globals().get("_REV_MAP", {}).get(int(did))
        return (wr() if callable(wr) else wr) if wr is not None else None
    except Exception:
        return None
    
def _revmap_len():
    try:
        return len(globals().get("_REV_MAP", {}) or {})
    except Exception:
        return 0

def init_dynamic_single(op, obj, dyn_id: int | None = None):
    _ensure_dyn_maps(op)
    if obj in op._xr_dyn_inited:
        # Even if already inited, keep reverse map healthy
        try: _backfill_revmap_from_op(op)
        except Exception: pass
        return

    # Assign/ensure id first (populates reverse map immediately)
    did = int(dyn_id if dyn_id is not None else ensure_dyn_id(op, obj))

    # Send LOCAL tris once (if any)
    try:
        from ...physics.exp_raycastutils import collect_local_triangles
        tris = collect_local_triangles(obj)
    except Exception as e:
        if _ok_to_print("_last_geom_dyn_err", "dev_log_geom_hz", "dev_log_geom_console"):
            print(f"[GEOM.XR] dyn init failed for {getattr(obj,'name','?')}: {e}")
        tris = []

    if tris:
        payload = {"id": did, "tris": tris}
        def _apply(res: dict):
            _hud_set("XR.geom.dyn_objs", int(res.get("dyn_objs", 0)))
            _hud_set("XR.geom.dyn_tris", int(res.get("dyn_tris", 0)))
            _hud_set("XR.geom.revmap", int(_revmap_len()), volatile=True)
            if _ok_to_print("_last_geom_dyn_init", "dev_log_geom_hz", "dev_log_geom_console"):
                print(f"[GEOM.XR] dyn init id={did} name={getattr(obj,'name','?')} tris={int(res.get('added_tris',0))}")
        xr_enqueue("geom.init_dynamic.v1", payload, _apply)

    op._xr_dyn_inited.add(obj)
    # Make sure we see a non-zero revmap immediately
    try: _backfill_revmap_from_op(op)
    except Exception: pass

def update_xforms_batch(op, batch_records: list[tuple[int, list[float]]]):
    """Send per-frame transforms for movers; also backfill id->obj map (self-heal)."""
    if not batch_records:
        return
    t0 = time.perf_counter()
    n = len(batch_records)
    payload = {"batch": [{"id": int(did), "M": list(M)} for (did, M) in batch_records]}

    def _apply(res: dict):
        # reverse-map self-heal (handles pre-existing ids)
        try:
            revsz = _backfill_revmap_from_op(op)
            if revsz >= 0:
                _hud_set("XR.geom.revmap", int(revsz), volatile=True)
        except Exception:
            pass

        # diagnostics & HUD (unchanged)
        try: _mark_last_xf_seq(int(res.get("_frame_seq", -1)))
        except Exception: _mark_last_xf_seq(-1)

        last_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        try: prev = float(getattr(op, "_xr_geom_xf_ema_ms", 0.0) or 0.0)
        except Exception: prev = 0.0
        ema = round(last_ms if prev <= 0.0 else (prev * 0.80 + last_ms * 0.20), 2)
        try: op._xr_geom_xf_ema_ms = ema
        except Exception: pass

        _hud_set("XR.geom.xf_last_ms", float(last_ms), volatile=True)
        _hud_set("XR.geom.xf_ema_ms",  float(ema),     volatile=True)

        if _ok_to_print("_last_geom_dyn_xf", "dev_log_geom_hz", "dev_log_geom_console"):
            print(f"[GEOM.XR] xforms batch {int(res.get('updated',0))}  net={last_ms:0.2f} ms  seq={res.get('_frame_seq','—')}  n={n}")

    xr_enqueue("geom.update_xforms.v1", payload, _apply)


# -----------------
# Parity aggregators (DEV HUD)
# -----------------
_PAR_MM, _PAR_DOT = [], []
_PAR_MAX = 5000
_PAR_MIS = 0
_PAR_SKIP_OVERHEAD = 0
_PAR_MM_FAIL = 0
_PAR_DOT_FAIL = 0
_PAR_WALK_MIS = 0
_PAR_THRESH = {"mm_p95": 2.0, "min_samples": 200}
_PAR_GATE = {"state": "WIP", "printed": False}

def _par_push(mm: float, dot_abs: float):
    _PAR_MM.append(float(mm)); _PAR_DOT.append(float(dot_abs))
    if len(_PAR_MM) > _PAR_MAX:   del _PAR_MM[:-_PAR_MAX]
    if len(_PAR_DOT) > _PAR_MAX:  del _PAR_DOT[:-_PAR_MAX]
    _hud_set("XR.parity.samples", int(len(_PAR_MM)), volatile=True)

def _p95(seq):
    if not seq:
        return 0.0
    s = sorted(seq)
    k = int(0.95 * (len(s) - 1))
    return float(s[k])

def _publish_parity():
    mm95  = round(_p95(_PAR_MM), 2)
    dot95 = round(_p95(_PAR_DOT), 5)
    _hud_set("XR.parity.p95_mm",        mm95,              volatile=True)
    _hud_set("XR.parity.p95_dot",       dot95,             volatile=True)
    _hud_set("XR.parity.miss",          int(_PAR_MIS),     volatile=True)
    _hud_set("XR.parity.mm_fail",       int(_PAR_MM_FAIL), volatile=True)
    _hud_set("XR.parity.dot_fail",      int(_PAR_DOT_FAIL),volatile=True)
    _hud_set("XR.parity.walk_mis",      int(_PAR_WALK_MIS),volatile=True)
    _hud_set("XR.parity.skip_overhead", int(_PAR_SKIP_OVERHEAD), volatile=True)
    _hud_set("XR.parity.samples",       int(len(_PAR_MM)), volatile=True)

    state = "WIP"
    if len(_PAR_MM) >= _PAR_THRESH["min_samples"]:
        state = "PASS" if (mm95 <= _PAR_THRESH["mm_p95"] and _PAR_MIS == 0 and _PAR_WALK_MIS == 0) else "FAIL"
    _hud_set("XR.parity.gate", state, volatile=True)
    _PAR_GATE["state"] = state

    if state == "PASS" and not _PAR_GATE.get("printed", False):
        if _ok_to_print("_last_par_gate", "dev_log_geom_hz", "dev_log_geom_console"):
            print(f"[GEOM.XR][par] GATE PASS  samples={len(_PAR_MM)}  p95Δ={mm95:.2f} mm  miss={_PAR_MIS}  walk_mis={_PAR_WALK_MIS}")
        _PAR_GATE["printed"] = True

# ---------------------------------
# Parity jobs (dynamic-only and union)
# ---------------------------------
def parity_down_dynamic(*, origin_world, dir_world, max_dist,
                        local_dist: float, local_normal,
                        guard_offset: float = 1.0,
                        origin_z: float | None = None,
                        hit_z: float | None = None,
                        floor_cos: float | None = None):
    """DOWN ray DEV parity vs XR dynamic-only."""
    # overhead filter on local
    try:
        if (origin_z is not None) and (hit_z is not None) and (float(hit_z) > float(origin_z) + 1.0e-4):
            global _PAR_SKIP_OVERHEAD
            _PAR_SKIP_OVERHEAD += 1
            _hud_set("XR.parity.skip_overhead", int(_PAR_SKIP_OVERHEAD), volatile=True)
            if _ok_to_print("_last_down_dyn_skip", "dev_log_geom_hz", "dev_log_geom_console"):
                print(f"[GEOM.XR][par] DOWN SKIP(overhead-local)  local={local_dist:0.3f} m")
            return
    except Exception:
        pass

    payload = {
        "origin": (float(origin_world[0]), float(origin_world[1]), float(origin_world[2])),
        "dir":    (float(dir_world[0]),    float(dir_world[1]),    float(dir_world[2])),
        "max_d":  float(max_dist),
    }
    t_send = time.perf_counter()

    def _apply(res: dict):
        global _PAR_MIS, _PAR_MM_FAIL, _PAR_DOT_FAIL, _PAR_WALK_MIS
        if not isinstance(res, dict) or not res.get("hit", False):
            _PAR_MIS += 1
            _publish_parity()
            if _ok_to_print("_last_down_dyn_miss", "dev_log_geom_hz", "dev_log_geom_console"):
                print(f"[GEOM.XR][par] DOWN MISS  local={local_dist:0.3f} m")
            return

        xr_raw = float(res.get("dist", 0.0))  # from START
        if xr_raw < float(guard_offset) - 1.0e-4:
            # XR thinks hit above start
            global _PAR_SKIP_OVERHEAD
            _PAR_SKIP_OVERHEAD += 1
            _hud_set("XR.parity.skip_overhead", int(_PAR_SKIP_OVERHEAD), volatile=True)
            if _ok_to_print("_last_down_dyn_skip2", "dev_log_geom_hz", "dev_log_geom_console"):
                print(f"[GEOM.XR][par] DOWN SKIP(overhead-xr)  raw={xr_raw:0.3f}  guard={guard_offset:0.3f}")
            return

        xr_dist = xr_raw - float(guard_offset)

        # normals + metrics
        try:
            lx, ly, lz = (float(local_normal[0]), float(local_normal[1]), float(local_normal[2]))
        except Exception:
            lx, ly, lz = 0.0, 0.0, 1.0
        ln = (lx*lx + ly*ly + lz*lz) ** 0.5 or 1.0
        lx, ly, lz = lx/ln, ly/ln, lz/ln

        nx, ny, nz = res.get("normal", (0.0, 0.0, 1.0))
        nn = (nx*nx + ny*ny + nz*nz) ** 0.5 or 1.0
        nx, ny, nz = nx/nn, ny/nn, nz/nn

        ndot = abs(lx*nx + ly*ny + lz*nz)
        mm   = abs(float(local_dist) - xr_dist) * 1000.0
        _par_push(mm, ndot)

        mm_ok  = (mm <= 2.0)
        dot_ok = (ndot >= 0.999)
        if not mm_ok: _PAR_MM_FAIL += 1
        elif not dot_ok: _PAR_DOT_FAIL += 1

        if floor_cos is not None:
            walk_loc = (lz >= float(floor_cos))
            walk_xr  = (nz >= float(floor_cos))
            if walk_loc != walk_xr:
                _PAR_WALK_MIS += 1

        _publish_parity()

        if _ok_to_print("_last_down_dyn_log", "dev_log_geom_hz", "dev_log_geom_console"):
            dt_ms = (time.perf_counter() - t_send) * 1000.0
            tag = "OK" if (mm_ok and (floor_cos is None or _PAR_WALK_MIS == 0)) else ("BAD" if not mm_ok else "WARN(n)")
            print(f"[GEOM.XR][par] DOWN {tag}  Δ={mm:0.2f} mm  n·dot={ndot:0.5f}  local={local_dist:0.3f} m  xr_adj={xr_dist:0.3f} m  (raw {xr_raw:0.3f})  net={dt_ms:0.2f} ms")

    xr_enqueue("geom.ray_dynamic.v1", payload, _apply)

def parity_forward_dynamic(*, origin_world, dir_world, max_dist, local_dist: float, local_normal):
    """FWD ray DEV parity vs XR dynamic-only (no guard offset)."""
    payload = {
        "origin": (float(origin_world[0]), float(origin_world[1]), float(origin_world[2])),
        "dir":    (float(dir_world[0]),    float(dir_world[1]),    float(dir_world[2])),
        "max_d":  float(max_dist),
    }
    t_send = time.perf_counter()

    def _apply(res: dict):
        global _PAR_MIS, _PAR_MM_FAIL, _PAR_DOT_FAIL
        if not isinstance(res, dict) or not res.get("hit", False):
            _PAR_MIS += 1
            _publish_parity()
            if _ok_to_print("_last_fwd_dyn_miss", "dev_log_geom_hz", "dev_log_geom_console"):
                print(f"[GEOM.XR][par] FWD MISS  local={local_dist:0.3f} m")
            return

        xr_dist = float(res.get("dist", 0.0))

        try:
            lx, ly, lz = (float(local_normal[0]), float(local_normal[1]), float(local_normal[2]))
        except Exception:
            lx, ly, lz = 0.0, 0.0, 1.0
        ln = (lx*lx + ly*ly + lz*lz) ** 0.5 or 1.0
        lx, ly, lz = lx/ln, ly/ln, lz/ln

        nx, ny, nz = res.get("normal", (0.0, 0.0, 1.0))
        nn = (nx*nx + ny*ny + nz*nz) ** 0.5 or 1.0
        nx, ny, nz = nx/nn, ny/nn, nz/nn

        ndot = abs(lx*nx + ly*ny + lz*nz)
        mm   = abs(float(local_dist) - xr_dist) * 1000.0
        _par_push(mm, ndot); _publish_parity()

        ok = (mm <= 2.0 and ndot >= 0.999)
        if not ok: _PAR_MIS += 1; _publish_parity()

        if _ok_to_print("_last_fwd_dyn_log", "dev_log_geom_hz", "dev_log_geom_console"):
            dt_ms = (time.perf_counter() - t_send) * 1000.0
            status = "OK" if ok else "BAD"
            print(f"[GEOM.XR][par] FWD {status}  Δ={mm:0.2f} mm  n·dot={ndot:0.5f}  local={local_dist:0.3f} m  xr={xr_dist:0.3f} m  net={dt_ms:0.2f} ms")

    xr_enqueue("geom.ray_dynamic.v1", payload, _apply)

def parity_down_both(*, origin_world, dir_world, max_dist,
                     local_dist: float, local_normal,
                     guard_offset: float = 1.0,
                     origin_z: float | None = None,
                     hit_z: float | None = None,
                     floor_cos: float | None = None):
    """DOWN ray DEV parity vs XR union (static+dynamic)."""
    # skip if local says overhead
    try:
        if (origin_z is not None) and (hit_z is not None) and (float(hit_z) > float(origin_z) + 1.0e-4):
            global _PAR_SKIP_OVERHEAD
            _PAR_SKIP_OVERHEAD += 1
            _hud_set("XR.parity.skip_overhead", int(_PAR_SKIP_OVERHEAD), volatile=True)
            if _ok_to_print("_last_down_u_skip", "dev_log_geom_hz", "dev_log_geom_console"):
                print(f"[GEOM.XR][par] DOWN(u) SKIP(overhead-local)  local={local_dist:0.3f} m")
            return
    except Exception:
        pass

    payload = {
        "origin": (float(origin_world[0]), float(origin_world[1]), float(origin_world[2])),
        "dir":    (float(dir_world[0]),    float(dir_world[1]),    float(dir_world[2])),
        "max_d":  float(max_dist),
    }
    t_send = time.perf_counter()

    def _apply(res: dict):
        global _PAR_MIS, _PAR_MM_FAIL, _PAR_DOT_FAIL, _PAR_WALK_MIS, _PAR_SKIP_OVERHEAD
        if not isinstance(res, dict) or not res.get("hit", False):
            _PAR_MIS += 1
            _publish_parity()
            if _ok_to_print("_last_down_u_miss", "dev_log_geom_hz", "dev_log_geom_console"):
                print(f"[GEOM.XR][par] DOWN(u) MISS  local={local_dist:0.3f} m")
            return

        xr_raw = float(res.get("dist", 0.0))
        src    = str(res.get("src", "?")).upper()
        if xr_raw < float(guard_offset) - 1.0e-4:
            _PAR_SKIP_OVERHEAD += 1
            _hud_set("XR.parity.skip_overhead", int(_PAR_SKIP_OVERHEAD), volatile=True)
            if _ok_to_print("_last_down_u_skip2", "dev_log_geom_hz", "dev_log_geom_console"):
                print(f"[GEOM.XR][par] DOWN(u) SKIP(overhead-xr)  raw={xr_raw:0.3f}  guard={guard_offset:0.3f}")
            return

        xr_dist = xr_raw - float(guard_offset)

        try:
            lx, ly, lz = (float(local_normal[0]), float(local_normal[1]), float(local_normal[2]))
        except Exception:
            lx, ly, lz = 0.0, 0.0, 1.0
        ln = (lx*lx + ly*ly + lz*lz) ** 0.5 or 1.0
        lx, ly, lz = lx/ln, ly/ln, lz/ln

        nx, ny, nz = res.get("normal", (0.0, 0.0, 1.0))
        nn = (nx*nx + ny*ny + nz*nz) ** 0.5 or 1.0
        nx, ny, nz = nx/nn, ny/nn, nz/nn

        ndot = abs(lx*nx + ly*ny + lz*nz)
        mm   = abs(float(local_dist) - xr_dist) * 1000.0
        _par_push(mm, ndot)

        if mm > 2.0: _PAR_MM_FAIL += 1
        elif ndot < 0.999: _PAR_DOT_FAIL += 1

        if floor_cos is not None:
            walk_loc = (lz >= float(floor_cos))
            walk_xr  = (nz >= float(floor_cos))
            if walk_loc != walk_xr:
                _PAR_WALK_MIS += 1

        _publish_parity()

        if _ok_to_print("_last_down_u_log", "dev_log_geom_hz", "dev_log_geom_console"):
            dt_ms = (time.perf_counter() - t_send) * 1000.0
            tag = "OK" if (mm <= 2.0 and (floor_cos is None or _PAR_WALK_MIS == 0)) else ("BAD" if mm > 2.0 else "WARN(n)")
            print(f"[GEOM.XR][par] DOWN(u) {tag}  src={src}  Δ={mm:0.2f} mm  n·dot={ndot:0.5f}  local={local_dist:0.3f} m  xr_adj={xr_dist:0.3f} m  (raw {xr_raw:0.3f})  net={dt_ms:0.2f} ms")

    xr_enqueue("geom.ray_union.v1", payload, _apply)

def parity_forward_both(*, origin_world, dir_world, max_dist, local_dist: float, local_normal):
    """FWD ray DEV parity vs XR union (static+dynamic)."""
    payload = {
        "origin": (float(origin_world[0]), float(origin_world[1]), float(origin_world[2])),
        "dir":    (float(dir_world[0]),    float(dir_world[1]),    float(dir_world[2])),
        "max_d":  float(max_dist),
    }
    t_send = time.perf_counter()

    def _apply(res: dict):
        global _PAR_MIS, _PAR_MM_FAIL, _PAR_DOT_FAIL
        if not isinstance(res, dict) or not res.get("hit", False):
            _PAR_MIS += 1
            _publish_parity()
            if _ok_to_print("_last_fwd_u_miss", "dev_log_geom_hz", "dev_log_geom_console"):
                print(f"[GEOM.XR][par] FWD(u) MISS  local={local_dist:0.3f} m")
            return

        xr_dist = float(res.get("dist", 0.0))
        src     = str(res.get("src", "?")).upper()

        try:
            lx, ly, lz = (float(local_normal[0]), float(local_normal[1]), float(local_normal[2]))
        except Exception:
            lx, ly, lz = 0.0, 0.0, 1.0
        ln = (lx*lx + ly*ly + lz*lz) ** 0.5 or 1.0
        lx, ly, lz = lx/ln, ly/ln, lz/ln

        nx, ny, nz = res.get("normal", (0.0, 0.0, 1.0))
        nn = (nx*nx + ny*ny + nz*nz) ** 0.5 or 1.0
        nx, ny, nz = nx/nn, ny/nn, nz/nn

        ndot = abs(lx*nx + ly*ny + lz*nz)
        mm   = abs(float(local_dist) - xr_dist) * 1000.0
        _par_push(mm, ndot); _publish_parity()

        if _ok_to_print("_last_fwd_u_log", "dev_log_geom_hz", "dev_log_geom_console"):
            dt_ms = (time.perf_counter() - t_send) * 1000.0
            status = "OK" if (mm <= 2.0 and ndot >= 0.999) else "BAD"
            print(f"[GEOM.XR][par] FWD(u) {status}  src={src}  Δ={mm:0.2f} mm  n·dot={ndot:0.5f}  local={local_dist:0.3f} m  xr={xr_dist:0.3f} m  net={dt_ms:0.2f} ms")

    xr_enqueue("geom.ray_union.v1", payload, _apply)

# ---------------------------------
# XR DOWN jobs (dynamic, union, both)
# ---------------------------------
_DOWN_DYN   = {"req": 0, "ok": 0, "hit": 0}
_DOWN_UNION = {"req": 0, "ok": 0, "hit": 0, "src_static": 0, "src_dynamic": 0}
_DOWN_BOTH  = {"req": 0, "ok": 0, "hit": 0, "src_static": 0, "src_dynamic": 0}

def queue_down_dynamic(kcc_sink, start_world_xyz, max_dist_with_guard: float, guard: float = 1.0):
    """Enqueue XR dynamic-only down ray; stash last reply on controller._xr_down_dyn."""
    import time as _t
    sx, sy, sz = map(float, start_world_xyz)
    payload = {"origin": (sx, sy, sz), "dir": (0.0, 0.0, -1.0), "max_d": float(max_dist_with_guard)}

    _DOWN_DYN["req"] += 1
    _hud_set("XR.downDyn.req", int(_DOWN_DYN["req"]), volatile=True)
    t0 = _t.perf_counter()

    def _apply(res: dict):
        lat_ms = (_t.perf_counter() - t0) * 1000.0
        _DOWN_DYN["ok"] += 1
        _hud_set("XR.downDyn.ok", int(_DOWN_DYN["ok"]), volatile=True)
        _hud_set("XR.downDyn.lat_ms", float(round(lat_ms, 2)), volatile=True)

        hit = bool(isinstance(res, dict) and res.get("hit", False))
        if hit:
            _DOWN_DYN["hit"] += 1
            _hud_set("XR.downDyn.hit", int(_DOWN_DYN["hit"]), volatile=True)

        try:
            n = res.get("normal", (0.0, 0.0, 1.0)) if isinstance(res, dict) else (0.0, 0.0, 1.0)
            kcc_sink._xr_down_dyn = {
                "t": _t.perf_counter(),
                "hit": hit,
                "raw": float(res.get("dist", 0.0)) if hit else 0.0,
                "normal": (float(n[0]), float(n[1]), float(n[2])),
                "start": (sx, sy, sz),
                "guard": float(guard),
                "lat_ms": float(lat_ms),
                "id": (int(res.get("id")) if isinstance(res, dict) and res.get("id") is not None else None),
                "_frame_seq": (res.get("_frame_seq") if isinstance(res, dict) else None),
            }
        except Exception:
            pass

        if _ok_to_print("_last_down_dyn", "dev_log_geom_hz", "dev_log_geom_console"):
            hs = "Y" if hit else "N"
            raw_txt = f"{kcc_sink._xr_down_dyn.get('raw', 0.0):0.3f}" if hit else "—"
            seq_txt = (kcc_sink._xr_down_dyn.get("_frame_seq", "—")
                       if isinstance(getattr(kcc_sink, "_xr_down_dyn", None), dict) else "—")
            mid = kcc_sink._xr_down_dyn.get("id", "—") if isinstance(getattr(kcc_sink, "_xr_down_dyn", None), dict) else "—"
            print(f"[DOWN.DYN] ok hit={hs} raw={raw_txt} guard={guard:0.3f} lat={lat_ms:0.2f} ms seq={seq_txt} id={mid}")

    xr_enqueue("geom.ray_dynamic.v1", payload, _apply)

def queue_down_union(kcc_sink, start_world_xyz, max_dist_with_guard: float, guard: float = 1.0):
    """Enqueue union (static+dynamic) down ray; stash on controller._xr_down_union."""
    import time as _t
    sx, sy, sz = map(float, start_world_xyz)
    payload = {"origin": (sx, sy, sz), "dir": (0.0, 0.0, -1.0), "max_d": float(max_dist_with_guard)}

    _DOWN_UNION["req"] += 1
    _hud_set("XR.down.req", int(_DOWN_UNION["req"]), volatile=True)
    _hud_set("XR.geom.authority", "XR", volatile=True)

    t0 = _t.perf_counter()

    def _apply(res: dict):
        lat_ms = (_t.perf_counter() - t0) * 1000.0
        _DOWN_UNION["ok"] += 1
        _hud_set("XR.down.ok", int(_DOWN_UNION["ok"]), volatile=True)
        _hud_set("XR.down.lat_ms", float(round(lat_ms, 2)), volatile=True)

        hit = bool(isinstance(res, dict) and res.get("hit", False))
        src = str(res.get("src", "—")).upper() if isinstance(res, dict) else "—"
        if hit:
            _DOWN_UNION["hit"] += 1
            _hud_set("XR.down.hit", int(_DOWN_UNION["hit"]), volatile=True)
            if src == "STATIC":
                _DOWN_UNION["src_static"] += 1;  _hud_set("XR.down.src_static", int(_DOWN_UNION["src_static"]), volatile=True)
            elif src == "DYNAMIC":
                _DOWN_UNION["src_dynamic"] += 1; _hud_set("XR.down.src_dynamic", int(_DOWN_UNION["src_dynamic"]), volatile=True)

        try:
            n = res.get("normal", (0.0, 0.0, 1.0)) if isinstance(res, dict) else (0.0, 0.0, 1.0)
            kcc_sink._xr_down_union = {
                "t": _t.perf_counter(),
                "hit": hit,
                "raw": float(res.get("dist", 0.0)) if hit else 0.0,
                "normal": (float(n[0]), float(n[1]), float(n[2])),
                "src": src,
                "start": (sx, sy, sz),
                "guard": float(guard),
                "lat_ms": float(lat_ms),
                "_frame_seq": (res.get("_frame_seq") if isinstance(res, dict) else None),
            }
        except Exception:
            pass

        if _ok_to_print("_last_down_union", "dev_log_geom_hz", "dev_log_geom_console"):
            hs = "Y" if hit else "N"
            raw_txt = f"{kcc_sink._xr_down_union.get('raw', 0.0):0.3f}" if hit else "—"
            seq_txt = (kcc_sink._xr_down_union.get("_frame_seq", "—")
                       if isinstance(getattr(kcc_sink, "_xr_down_union", None), dict) else "—")
            print(f"[DOWN.UNION] ok hit={hs} src={src} raw={raw_txt} guard={guard:0.3f} lat={lat_ms:0.2f} ms seq={seq_txt}")

    xr_enqueue("geom.ray_union.v1", payload, _apply)

def queue_down_both(kcc_sink, start_world_xyz, max_dist_with_guard: float, guard: float = 1.0):
    """
    Enqueue XR job that returns STATIC, DYNAMIC, and UNION pick in one reply.
    Stores on controller._xr_down_both:
      {t, hit, raw, normal, src, id|None, start, guard, lat_ms, _frame_seq,
       stat:{hit,raw,normal}, dyn:{hit,raw,normal,id|None}}
    Raw distances are from 'start'; consumer subtracts guard.
    """
    import time as _t
    sx, sy, sz = map(float, start_world_xyz)
    payload = {"origin": (sx, sy, sz), "dir": (0.0, 0.0, -1.0), "max_d": float(max_dist_with_guard)}

    _DOWN_BOTH["req"] += 1
    _hud_set("XR.downBoth.req", int(_DOWN_BOTH["req"]), volatile=True)
    t0 = _t.perf_counter()

    def _apply(res: dict):
        lat_ms = (_t.perf_counter() - t0) * 1000.0
        _DOWN_BOTH["ok"] += 1
        _hud_set("XR.downBoth.ok", int(_DOWN_BOTH["ok"]), volatile=True)
        _hud_set("XR.downBoth.lat_ms", float(round(lat_ms, 2)), volatile=True)

        hit = bool(isinstance(res, dict) and res.get("hit", False))
        if hit:
            _DOWN_BOTH["hit"] += 1
            _hud_set("XR.downBoth.hit", int(_DOWN_BOTH["hit"]), volatile=True)

        src = str(res.get("src", "—")).upper() if hit else "—"
        if src == "STATIC":
            _DOWN_BOTH["src_static"] += 1;  _hud_set("XR.downBoth.src_static", int(_DOWN_BOTH["src_static"]), volatile=True)
        elif src == "DYNAMIC":
            _DOWN_BOTH["src_dynamic"] += 1; _hud_set("XR.downBoth.src_dynamic", int(_DOWN_BOTH["src_dynamic"]), volatile=True)

        try:
            n  = res.get("normal", (0.0, 0.0, 1.0)) if hit else (0.0, 0.0, 1.0)
            st = res.get("stat", {"hit": False})
            dy = res.get("dyn",  {"hit": False})

            kcc_sink._xr_down_both = {
                "t": _t.perf_counter(),
                "hit": hit,
                "raw": float(res.get("dist", 0.0)) if hit else 0.0,
                "normal": (float(n[0]), float(n[1]), float(n[2])),
                "src": src,
                "id": (int(res.get("id")) if res.get("id") is not None else None),
                "start": (sx, sy, sz),
                "guard": float(guard),
                "lat_ms": float(lat_ms),
                "_frame_seq": (res.get("_frame_seq") if isinstance(res, dict) else None),
                "stat": {
                    "hit": bool(st.get("hit", False)),
                    "raw": float(st.get("dist", 0.0)) if st.get("hit", False) else 0.0,
                    "normal": st.get("normal", (0.0, 0.0, 1.0)),
                },
                "dyn": {
                    "hit": bool(dy.get("hit", False)),
                    "raw": float(dy.get("dist", 0.0)) if dy.get("hit", False) else 0.0,
                    "normal": dy.get("normal", (0.0, 0.0, 1.0)),
                    "id": (int(dy.get("id")) if dy.get("id") is not None else None),
                },
            }
        except Exception:
            pass

        if _ok_to_print("_last_down_both_log", "dev_log_geom_hz", "dev_log_geom_console"):
            hs = "Y" if hit else "N"
            raw_txt = f"{kcc_sink._xr_down_both.get('raw', 0.0):0.3f}" if hit else "—"
            seq_txt = (kcc_sink._xr_down_both.get("_frame_seq", "—")
                       if isinstance(getattr(kcc_sink, "_xr_down_both", None), dict) else "—")
            print(f"[DOWN.BOTH] ok hit={hs} src={src} raw={raw_txt} guard={guard:0.3f} lat={lat_ms:0.2f} ms seq={seq_txt}")

    xr_enqueue("geom.ray_both.v1", payload, _apply)
