# Exp_Game/xr_systems/xr_ports/geom.py
from __future__ import annotations
import time
import bpy
from ..xr_queue import xr_enqueue

# ----------------------
# Shared logging / HUD
# ----------------------
def _ok_to_print(tag_attr: str, hz_prop: str, enable_prop: str) -> bool:
    try:
        sc = bpy.context.scene
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

def _hud_set(key: str, val, volatile=False):
    try:
        from ...Developers.exp_dev_interface import devhud_set
        devhud_set(key, val, volatile=bool(volatile))
    except Exception:
        pass

# Tiny meter we can inject into STATE if missing
class _LiteMeter:
    __slots__ = ("_t", "_n")
    def __init__(self): self._t, self._n = [], []
    def hit(self):
        import time as _t
        now = _t.perf_counter()
        self._t.append(now); self._n.append(1)
        if len(self._t) > 300:
            self._t = self._t[-300:]; self._n = self._n[-300:]
    def rate(self, now, window_s=1.0):
        cutoff = now - float(window_s)
        s = 0; i = len(self._t) - 1
        while i >= 0 and self._t[i] >= cutoff:
            s += self._n[i]; i -= 1
        return float(s) / max(1e-6, float(window_s))

def _ensure_geom_meters():
    try:
        from ...Developers.dev_state import STATE
        if not hasattr(STATE, "meter_geom_xforms"):
            STATE.meter_geom_xforms = _LiteMeter()
        if not hasattr(STATE, "meter_geom_cast"):
            STATE.meter_geom_cast = _LiteMeter()
        if not hasattr(STATE, "meter_geom_nearest"):
            STATE.meter_geom_nearest = _LiteMeter()
        return STATE
    except Exception:
        return None

# ---------------------------------
# STATIC world triangles (streamed)
# ---------------------------------
def start_static_upload_from_meshes(op, mesh_objs, *, chunk_tris: int = 2048):
    from ...physics.exp_raycastutils import collect_world_triangles
    tris = collect_world_triangles(mesh_objs)
    total_floats = len(tris)
    total_tris = total_floats // 9

    op._xr_geom_upload = {
        "tris": tris,
        "off": 0,
        "total_floats": total_floats,
        "total_tris": total_tris,
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
    st = getattr(op, "_xr_geom_upload", None)
    if not st or st.get("ended", False):
        return

    total = int(st["total_floats"])
    off   = int(st["off"])
    chunk = int(st["chunk_floats"])

    def _progress(sent_tris, total_tris):
        try:
            from ...Developers.exp_dev_interface import devhud_set
            pct = (100.0 * float(sent_tris) / max(1, int(total_tris))) if total_tris > 0 else 0.0
            devhud_set("XR.geom.upload_pct", f"{pct:0.1f}%", volatile=True)
            devhud_set("XR.geom.uploaded_tris", int(sent_tris), volatile=True)
        except Exception:
            pass

    sent_chunks = 0
    while sent_chunks < int(max(1, chunks_per_frame)) and off < total:
        end = min(off + chunk, total)
        payload_slice = st["tris"][off:end]
        off = end
        sent_chunks += 1
        xr_enqueue("geom.init_static.chunk.v1", {"tris": payload_slice}, None)
        _progress((off // 9), int(st["total_tris"]))

    st["off"] = off

    if off >= total and not st["ended"]:
        t_send = time.perf_counter()
        def _apply_end(res: dict):
            static_tris = int(res.get("static_tris", 0))
            build_ms    = float(res.get("build_ms", 0.0))
            mem_MB      = float(res.get("mem_MB", 0.0))

            # Round before storing (kept as floats so curated HUD can format)
            build_ms = round(build_ms, 1)
            mem_MB   = round(mem_MB, 2)

            _hud_set("XR.geom.static_tris", static_tris)
            _hud_set("XR.geom.build_ms",    build_ms)
            _hud_set("XR.geom.mem_MB",      mem_MB)
            _hud_set("XR.geom.upload_pct",  "100.0%")
            _hud_set("XR.geom.mode", "STATIC_STORE")
            _hud_set("XR.geom.authority", "BLENDER")

            if _ok_to_print("_last_geom_log", "dev_log_geom_hz", "dev_log_geom_console"):
                dt_net_ms = (time.perf_counter() - float(st.get("t0", time.perf_counter()))) * 1000.0
                print(f"[GEOM.XR] static upload complete  static_tris={static_tris:,}  "
                    f"build={build_ms:0.1f} ms  mem={mem_MB:0.2f} MB  net={dt_net_ms:0.1f} ms")

            try:
                op._xr_geom_static_done = True
            except Exception:
                pass

        xr_enqueue("geom.init_static.end.v1", {}, _apply_end)
        st["ended"] = True

# ---------------------------------
# DYNAMIC movers (M2)
# ---------------------------------
def _ensure_dyn_maps(op):
    if not hasattr(op, "_xr_dyn_id_map"):
        op._xr_dyn_id_map = {}   # obj -> int id
    if not hasattr(op, "_xr_dyn_id_seq"):
        op._xr_dyn_id_seq = 1
    if not hasattr(op, "_xr_dyn_inited"):
        op._xr_dyn_inited = set()

def ensure_dyn_id(op, obj) -> int:
    _ensure_dyn_maps(op)
    if obj in op._xr_dyn_id_map:
        return op._xr_dyn_id_map[obj]
    did = int(op._xr_dyn_id_seq)
    op._xr_dyn_id_seq = did + 1
    op._xr_dyn_id_map[obj] = did
    return did

def init_dynamic_single(op, obj, dyn_id: int | None = None):
    _ensure_dyn_maps(op)
    if obj in op._xr_dyn_inited:
        return
    try:
        from ...physics.exp_raycastutils import collect_local_triangles
        tris = collect_local_triangles(obj)
    except Exception as e:
        if _ok_to_print("_last_geom_dyn_err", "dev_log_geom_hz", "dev_log_geom_console"):
            print(f"[GEOM.XR] dyn init failed for {getattr(obj,'name','?')}: {e}")
        return

    if not tris:
        op._xr_dyn_inited.add(obj)
        return

    did = int(dyn_id if dyn_id is not None else ensure_dyn_id(op, obj))
    payload = {"id": did, "tris": tris}

    def _apply(res: dict):
        _hud_set("XR.geom.dyn_objs", int(res.get("dyn_objs", 0)))
        _hud_set("XR.geom.dyn_tris", int(res.get("dyn_tris", 0)))
        if _ok_to_print("_last_geom_dyn_init", "dev_log_geom_hz", "dev_log_geom_console"):
            print(f"[GEOM.XR] dyn init id={did} name={getattr(obj,'name','?')} tris={int(res.get('added_tris',0))}")

    xr_enqueue("geom.init_dynamic.v1", payload, _apply)
    op._xr_dyn_inited.add(obj)

def update_xforms_batch(op, batch_records: list[tuple[int, list[float]]]):
    """
    Enqueue a per-frame xform batch: [(dyn_id, M16_rowmajor), ...]
    Rounds latency numbers before pushing to HUD (floats remain floats).
    """
    if not batch_records:
        return
    _ensure_geom_meters()

    t0 = time.perf_counter()
    payload = {"batch": [{"id": int(did), "M": list(M)} for (did, M) in batch_records]}
    n = len(batch_records)

    def _apply(res: dict):
        STATE = _ensure_geom_meters()
        if STATE and hasattr(STATE, "meter_geom_xforms"):
            for _ in range(max(1, int(n))):
                STATE.meter_geom_xforms.hit()

        # xform latency: last + EMA, rounded
        last_ms = (time.perf_counter() - t0) * 1000.0
        last_ms = round(last_ms, 2)

        try:
            prev = float(getattr(op, "_xr_geom_xf_ema_ms", 0.0) or 0.0)
        except Exception:
            prev = 0.0
        a = 0.20
        ema = last_ms if prev <= 0.0 else (prev * (1.0 - a) + last_ms * a)
        ema = round(ema, 2)

        try:
            op._xr_geom_xf_ema_ms = ema
        except Exception:
            pass

        _hud_set("XR.geom.xf_last_ms", float(last_ms), volatile=True)
        _hud_set("XR.geom.xf_ema_ms",  float(ema),     volatile=True)

        if _ok_to_print("_last_geom_dyn_xf", "dev_log_geom_hz", "dev_log_geom_console"):
            print(f"[GEOM.XR] xforms batch {int(res.get('updated',0))}  net={last_ms:0.2f} ms")

    xr_enqueue("geom.update_xforms.v1", payload, _apply)

