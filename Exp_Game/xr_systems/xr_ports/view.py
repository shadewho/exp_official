# Exploratory/Exp_Game/xr_systems/xr_ports/view.py
# Blender-side helpers for VIEW: sync colliders to XR and queue full geometry solve.

import time
import bpy
from ..xr_queue import xr_enqueue
from ...Developers.exp_dev_interface import devhud_set, devhud_series_push

# -----------------------------
# Console rate-limiter
# -----------------------------
def _ok_to_print(tag_key: str, hz_prop: str, enable_prop: str) -> bool:
    try:
        sc = bpy.context.scene
        if not getattr(sc, enable_prop, False):
            return False
        hz = float(getattr(sc, hz_prop, 4.0) or 4.0)
        now = time.perf_counter()
        last = float(getattr(sc, tag_key, 0.0) or 0.0)
        if (now - last) >= (1.0 / max(0.1, hz)):
            setattr(sc, tag_key, now)
            return True
    except Exception:
        pass
    return False


# -----------------------------
# Static colliders → XR (once)
# -----------------------------
def sync_view_static_colliders(context, op):
    """
    Gather AABBs for all non-moving proxy meshes and send once to XR.
    """
    scene = context.scene
    aabbs = []
    for pm in getattr(scene, "proxy_meshes", []):
        obj = getattr(pm, "mesh_object", None)
        if not obj or getattr(pm, "is_moving", False):
            continue
        if obj.type != 'MESH':
            continue
        # World-space AABB from bound_box
        corners = [obj.matrix_world @ c for c in obj.bound_box]
        mnx = min(c.x for c in corners); mny = min(c.y for c in corners); mnz = min(c.z for c in corners)
        mxx = max(c.x for c in corners); mxy = max(c.y for c in corners); mxz = max(c.z for c in corners)
        aabbs.append({"id": obj.name, "min": [mnx, mny, mnz], "max": [mxx, mxy, mxz]})

    payload = {"aabbs": aabbs}
    def _after(res: dict):
        if _ok_to_print("_last_log_view_static", "dev_log_view_hz", "dev_log_view_console"):
            count = (res or {}).get("static_count", len(aabbs))
            print(f"[VIEW][XR] static AABBs synced: {count}")

    xr_enqueue("view.load_static_aabbs.v1", payload, _after)

# -----------------------------
# Dynamic colliders → XR (every tick)
# -----------------------------
def sync_view_dynamic_colliders(op):
    """
    Send dynamic spheres (center + conservative radius) for all active movers.
    Relies on modal_op._cached_dyn_radius computed by update_dynamic_meshes().
    """
    # Build spheres from cached radii
    spheres = []
    dyn_map = getattr(op, "dynamic_bvh_map", {}) or {}
    rad_map = getattr(op, "_cached_dyn_radius", {}) or {}

    for obj in dyn_map.keys():
        try:
            c = obj.matrix_world.translation
        except Exception:
            continue
        R = float(rad_map.get(obj, 0.0) or 0.0)
        spheres.append({"id": obj.name, "c": [float(c.x), float(c.y), float(c.z)], "r": R})

    if not spheres:
        return

    xr_enqueue("view.update_dynamic_spheres.v1", {"spheres": spheres}, None)

# -----------------------------
# Full XR geometry solve (THIRD)
# -----------------------------
# ---- EXACT static (triangles, world) → XR once ------------------------------
def sync_view_static_meshes_exact(context, op):
    """
    Gather world-space triangles for all non-moving proxy meshes and send once.
    """
    import bpy
    from ...Developers.exp_dev_interface import devhud_set
    scene = context.scene
    meshes = []
    tri_total = 0
    for pm in getattr(scene, "proxy_meshes", []):
        obj = getattr(pm, "mesh_object", None)
        if not obj or getattr(pm, "is_moving", False) or obj.type != 'MESH':
            continue
        mw = obj.matrix_world.copy()
        me = obj.data
        verts = [mw @ v.co for v in me.vertices]
        tris = []
        for poly in me.polygons:
            idx = list(poly.vertices)
            if len(idx) < 3: continue
            for i in range(1, len(idx)-1):
                v0 = verts[idx[0]]; v1 = verts[idx[i]]; v2 = verts[idx[i+1]]
                tris.append([v0.x,v0.y,v0.z, v1.x,v1.y,v1.z, v2.x,v2.y,v2.z])
        if tris:
            meshes.append({"id": obj.name, "tris": tris})
            tri_total += len(tris)

    def _after(res: dict):
        objs = int((res or {}).get("objs", len(meshes)))
        tris = int((res or {}).get("tris", tri_total))
        try:
            devhud_set("VIEW.static_meshes", objs, volatile=True)
            devhud_set("VIEW.static_tris", tris, volatile=True)
            devhud_set("VIEW.src", "XR_FULL_BVH", volatile=True)
        except Exception:
            pass
        if _ok_to_print("_last_log_view_static_exact", "dev_log_view_hz", "dev_log_view_console"):
            print(f"[VIEW][XR] static EXACT synced: {objs} objs, {tris} tris")

    xr_enqueue("view.load_static_meshes_exact.v2", {"meshes": meshes}, _after)

# ---- EXACT dynamic (triangles, local) → XR when first seen ------------------
def sync_view_dynamic_meshes_exact(op):
    """
    Send local-space triangles for dynamic movers that aren't registered yet.
    Uses op.dynamic_bvh_map keys (actual movers).
    """
    import bpy
    from ...Developers.exp_dev_interface import devhud_set
    dyn_map = getattr(op, "dynamic_bvh_map", {}) or {}
    if not dyn_map: return

    if not hasattr(op, "_xr_dyn_exact_synced"):
        op._xr_dyn_exact_synced = set()

    meshes = []
    added = 0
    for obj in dyn_map.keys():
        name = getattr(obj, "name", None)
        if not name or name in op._xr_dyn_exact_synced:
            continue
        if getattr(obj, "type", None) != 'MESH': 
            continue
        me = obj.data
        verts = [v.co.copy() for v in me.vertices]  # LOCAL coords
        tris = []
        for poly in me.polygons:
            idx = list(poly.vertices)
            if len(idx) < 3: continue
            for i in range(1, len(idx)-1):
                v0 = verts[idx[0]]; v1 = verts[idx[i]]; v2 = verts[idx[i+1]]
                tris.append([v0.x,v0.y,v0.z, v1.x,v1.y,v1.z, v2.x,v2.y,v2.z])
        if tris:
            meshes.append({"id": name, "tris": tris})
            op._xr_dyn_exact_synced.add(name)
            added += 1

    if not meshes:
        return

    def _after(res: dict):
        total = int((res or {}).get("total", len(op._xr_dyn_exact_synced)))
        if _ok_to_print("_last_log_view_dynamic_exact", "dev_log_view_hz", "dev_log_view_console"):
            print(f"[VIEW][XR] dynamic EXACT synced: +{added} (total {total})")

    xr_enqueue("view.load_dynamic_meshes_exact.v2", {"meshes": meshes}, _after)

# ---- Full EXACT solve (static + dynamic together) ---------------------------
def queue_view_third_full_exact(op, op_key: int, anchor_xyz, dir_xyz,
                                min_cam: float, desired_max: float, r_cam: float,
                                dyn_objects: list):
    """
    Enqueue one XR job that computes the final allowed distance using exact BVHs
    for BOTH static and dynamic, moving in tandem. Returns allowed (latched/smoothed) + candidate + hit.
    """

    # One enqueue per TIMER tick
    cnt = int(getattr(op, "_view_enq_guard", 0))
    if cnt >= 1:
        if _ok_to_print("_view_dup_warn", "dev_log_view_hz", "dev_log_view_console"):
            print("[VIEW][WARN] duplicate enqueue ignored this tick")
        return
    op._view_enq_guard = 1

    payload = {
        "op_id": int(op_key),
        "anchor": tuple(float(x) for x in anchor_xyz),
        "dir":    tuple(float(x) for x in dir_xyz),
        "min_cam": float(min_cam),
        "desired_max": float(desired_max),
        "r_cam": float(r_cam),
        "dyn_xforms": dyn_objects,  # [{"id": str, "M": [16]}]
    }

    def _apply(res: dict):
        allowed = res.get("allowed", None)
        cand    = res.get("candidate", None)
        hit_tok = res.get("hit", None)
        if not isinstance(allowed, (int,float)):
            try: devhud_set("VIEW.src", "MISS", volatile=True)
            except Exception: pass
            return

        op._xr_view_allowed = float(allowed)
        op._xr_view_age     = time.perf_counter()

        # HUD + console
        try:
            if isinstance(cand, (int,float)):
                devhud_set("VIEW.candidate", f"{float(cand):.3f} m", volatile=True)
                devhud_series_push("view_candidate", float(cand))
            devhud_set("VIEW.allowed", f"{float(allowed):.3f} m", volatile=True)
            devhud_set("VIEW.src", "XR_FULL_BVH", volatile=True)
            if hit_tok is not None:
                devhud_set("VIEW.hit", str(hit_tok), volatile=True)
            devhud_series_push("view_allowed", float(allowed))
        except Exception:
            pass

        if _ok_to_print("_last_log_viewxr_exact", "dev_log_view_hz", "dev_log_view_console"):
            ht = (hit_tok if hit_tok is not None else "NONE")
            print(f"[VIEW] qHz ?  aHz ?  dist {float(allowed):.3f} m  cand {float(cand or 0.0):.3f} m  hit {ht}")

    xr_enqueue("view.solve_third_full_exact.v1", payload, _apply)

