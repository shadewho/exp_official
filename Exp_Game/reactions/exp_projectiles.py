# Exploratory / Exp_Game / reactions / exp_projectiles.py
# KEEPING OLD BVH COLLISION BEHAVIOR EXACTLY:
# - Ray tests use the modal's proxy-mesh BVH (same as your old code)
# - Projectiles stop on proxy meshes (BVH hit) exactly as before
# - Lightweight pooled, linked duplicates for visuals (per-reaction Max Active)
# - No idle work when nothing is active
# - Full cleanup via clear()

import bpy
import math
from mathutils import Vector, Euler
from bpy_extras import view3d_utils
from ..props_and_utils.exp_time import get_game_time
from ..audio import exp_globals  # ACTIVE_MODAL_OP

# ─────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────
_POOLS = {}  # key(str) -> {"src": Object, "free": [Object], "active": set(Object)}
_ACTIVE_PROJECTILES = []  # each: {'obj','pos','vel','gravity','despawn_time','stop_on_contact','align','ignore_objs'}
_ACTIVE_VISUALS = []      # list[{'obj','despawn_time'}] for hitscan decals / markers
_MAX_GLOBAL_PROJECTILES = 64  # safety ceiling

# ─────────────────────────────────────────────────────────
# Modal / scene helpers
# ─────────────────────────────────────────────────────────
def _active_modal():
    return getattr(exp_globals, "ACTIVE_MODAL_OP", None)

def _scene():
    scn = getattr(bpy.context, "scene", None)
    if scn:
        return scn
    return bpy.data.scenes[0] if bpy.data.scenes else None

def _character_object() -> bpy.types.Object | None:
    scn = _scene()
    return getattr(scn, "target_armature", None)

# ─────────────────────────────────────────────────────────
# Pool helpers (linked duplicates; minimal overhead)
# ─────────────────────────────────────────────────────────
def _pool_key(obj: bpy.types.Object) -> str:
    """Stable, session-unique key as a STRING (avoid 32-bit int overflow)."""
    if not obj:
        return "0"
    try:
        return f"{int(obj.as_pointer()):x}"
    except Exception:
        try:
            return f"data:{int(obj.data.as_pointer()):x}"
        except Exception:
            return f"name:{getattr(obj, 'name', 'unknown')}"

def _link_to_scene(obj: bpy.types.Object):
    scn = _scene()
    if not scn or not obj:
        return
    try:
        scn.collection.objects.link(obj)
    except Exception:
        pass

def _unique_name_like(base: str) -> str:
    scn = _scene()
    existing = {o.name for o in getattr(scn, "objects", [])}
    if base not in existing:
        return base
    i = 1
    while True:
        nm = f"{base} (EXP {i})"
        if nm not in existing:
            return nm
        i += 1

def _make_instance(src: bpy.types.Object) -> bpy.types.Object:
    inst = src.copy()              # linked duplicate (shares mesh data)
    try:
        inst.data = src.data
    except Exception:
        pass
    inst.animation_data_clear()
    inst.hide_viewport = True
    inst.hide_render   = True
    inst.name = _unique_name_like(src.name)
    inst["_exp_pool_key"] = _pool_key(src)  # store STRING key
    _link_to_scene(inst)
    return inst

def _acquire(src: bpy.types.Object, cap: int) -> bpy.types.Object | None:
    if not src:
        return None
    k = _pool_key(src)
    pool = _POOLS.get(k)
    if pool is None:
        pool = {"src": src, "free": [], "active": set()}
        _POOLS[k] = pool

    if len(pool["active"]) >= max(1, int(cap)):
        return None

    inst = pool["free"].pop() if pool["free"] else _make_instance(src)
    pool["active"].add(inst)
    inst.hide_viewport = False
    inst.hide_render   = False
    return inst

def _release(inst: bpy.types.Object):
    if not inst:
        return
    k = None
    try:
        k = inst["_exp_pool_key"]
    except Exception:
        pass
    if not isinstance(k, str) or not k:
        try:
            k = _pool_key(inst)
        except Exception:
            k = None
    inst.hide_viewport = True
    inst.hide_render   = True
    if not k:
        return
    pool = _POOLS.get(k)
    if pool:
        pool["active"].discard(inst)
        pool["free"].append(inst)

def _destroy(inst: bpy.types.Object):
    if not inst:
        return
    try:
        for coll in list(inst.users_collection):
            try:
                coll.objects.unlink(inst)
            except Exception:
                pass
        bpy.data.objects.remove(inst, do_unlink=True)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────
# OLD aiming/origin helpers (unchanged behavior)
# ─────────────────────────────────────────────────────────
def _get_active_view3d_region():
    """Return (area, region, rv3d) for the first VIEW_3D/WINDOW found."""
    wm = bpy.context.window_manager
    if not wm:
        return None, None, None
    for win in wm.windows:
        scr = win.screen
        if not scr:
            continue
        for area in scr.areas:
            if area.type != 'VIEW_3D':
                continue
            rv3d = None
            try:
                rv3d = area.spaces.active.region_3d
            except Exception:
                rv3d = None
            for reg in area.regions:
                if reg.type == 'WINDOW':
                    return area, reg, rv3d
    return None, None, None

def _center_ray_world(max_range: float):
    """
    Crosshair ray in WORLD space from the active 3D View.
    Returns (origin_world, dir_world, far_point_world) or (None, None, None).
    """
    area, region, rv3d = _get_active_view3d_region()
    if not (area and region and rv3d):
        return None, None, None
    cx = region.width * 0.5
    cy = region.height * 0.5
    origin_w = view3d_utils.region_2d_to_origin_3d(region, rv3d, (cx, cy))
    dir_w    = view3d_utils.region_2d_to_vector_3d(region, rv3d, (cx, cy)).normalized()
    far_w    = origin_w + dir_w * float(max_range)
    return origin_w, dir_w, far_w

def _character_forward_from_scene():
    """
    World forward from character (+Y local). Falls back to op.yaw if needed.
    """
    scn = _scene()
    arm = getattr(scn, "target_armature", None)
    if arm:
        try:
            return (arm.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()
        except Exception:
            pass
    op = _active_modal()
    yaw = float(getattr(op, "yaw", 0.0)) if op else 0.0
    return Vector((math.cos(yaw), math.sin(yaw), 0.0)).normalized()

def _origin_from_reaction(r) -> Vector:
    """
    EXACTLY like old: origin = base.translation + (base.rotation * local_offset)
    (Ignores scale to match your previous behavior.)
    """
    scn = _scene()
    base_obj = scn.target_armature if getattr(r, "proj_use_character_origin", True) else getattr(r, "proj_origin_object", None)
    off_local = Vector(getattr(r, "proj_origin_offset", (0.0, 0.2, 1.4)))
    if base_obj:
        base_loc = base_obj.matrix_world.translation.copy()
        off_world = base_obj.matrix_world.to_3x3() @ off_local
        return base_loc + off_world
    return off_local.copy()

def _align_object_to_dir(obj, direction: Vector):
    try:
        q = direction.normalized().to_track_quat('Y', 'Z')  # +Y forward, Z up
        obj.rotation_euler = q.to_euler('XYZ')
    except Exception:
        pass

# ─────────────────────────────────────────────────────────
# OLD BVH raycast (unchanged semantics) + guarded fallback
# ─────────────────────────────────────────────────────────
def _raycast_world(op, origin: Vector, direction: Vector, max_dist: float, ignore: set | None = None):
    """
    EXACT old behavior when BVH is present:
      - Raycast against proxy-mesh BVH built by the modal (static only).
    Only if BVH is missing do we fallback to scene.ray_cast, where we ignore
    a small set (instance, source, character) to avoid self-hits while moving.
    Returns (hit:bool, loc:Vector|None, normal:Vector|None, dist:float|None)
    """
    if max_dist <= 1e-6:
        return (False, None, None, None)

    # OLD: BVH path
    bvh = getattr(op, "bvh_tree", None) if op else None
    if bvh:
        dir_n = direction.normalized()
        hit = bvh.ray_cast(origin, dir_n, max_dist)
        if not hit or hit[0] is None:
            return (False, None, None, None)
        loc, nor, _idx, _d = hit
        d = (loc - origin).length
        return (True, loc, nor, d)

    # Fallback only if BVH missing (keep it robust while moving)
    scn = _scene()
    if not scn:
        return (False, None, None, None)
    deps = bpy.context.evaluated_depsgraph_get()
    dir_n = direction.normalized()
    try:
        h, loc, nor, _face, hit_obj, _mtx = scn.ray_cast(deps, origin, dir_n, distance=max_dist)
    except TypeError:
        h, loc, nor, _face, hit_obj, _mtx = scn.ray_cast(deps, origin, dir_n * max_dist)
    if h:
        if ignore and hit_obj in ignore:
            return (False, None, None, None)
        d = (loc - origin).length
        return (True, loc, nor, d)
    return (False, None, None, None)

def _resolve_direction(r, origin) -> Vector:
    """
    EXACT old aiming logic:
      CROSSHAIR: aim from camera center; use BVH to pick target if available.
      otherwise: character forward.
    """
    aim_src = (getattr(r, "proj_aim_source", "CROSSHAIR") or "CROSSHAIR")
    if aim_src == "CAMERA":  # legacy alias
        aim_src = "CROSSHAIR"

    if aim_src == "CROSSHAIR":
        hs_range = float(getattr(r, "proj_max_range", 60.0))
        cam_o, cam_dir, cam_far = _center_ray_world(max_range=hs_range)
        if cam_o is not None:
            op = _active_modal()
            ok, loc, _n, _d = _raycast_world(op, cam_o, cam_dir, hs_range)
            target = loc if ok else cam_far
            d = (target - origin)
            return d.normalized() if d.length > 0.0 else _character_forward_from_scene()
        return _character_forward_from_scene()
    else:
        return _character_forward_from_scene()

# ─────────────────────────────────────────────────────────
# Executors
# ─────────────────────────────────────────────────────────
def execute_hitscan_reaction(r):
    """
    Instant ray from origin along resolved direction (BVH). On hit, optional pooled visual.
    """
    op = _active_modal()
    if not op:
        return

    origin = _origin_from_reaction(r)
    direction = _resolve_direction(r, origin)
    max_range = float(getattr(r, "proj_max_range", 60.0))

    hit, loc, _nor, _dist = _raycast_world(op, origin, direction, max_range)
    impact = loc if hit else (origin + direction * max_range)

    vis_src = getattr(r, "proj_object", None)
    if vis_src and getattr(r, "proj_place_hitscan_object", True):
        inst = _acquire(vis_src, int(getattr(r, "proj_pool_limit", 8)))
        if inst:
            inst.location = impact
            if getattr(r, "proj_align_object_to_velocity", True):
                _align_object_to_dir(inst, direction)
            ttl = float(getattr(r, "proj_lifetime", 0.0))
            if ttl > 0.0:
                _ACTIVE_VISUALS.append({"obj": inst, "despawn_time": get_game_time() + ttl})

def execute_projectile_reaction(r):
    """
    Spawn a pooled visual and simulate simple ballistic motion (BVH collision).
    """
    op = _active_modal()
    if not op:
        return

    if len(_ACTIVE_PROJECTILES) >= _MAX_GLOBAL_PROJECTILES:
        return

    src = getattr(r, "proj_object", None)
    if not src:
        return  # nothing to visualize; keep it cheap

    inst = _acquire(src, int(getattr(r, "proj_pool_limit", 8)))
    if not inst:
        return  # per-reaction cap reached

    origin = _origin_from_reaction(r)
    direction = _resolve_direction(r, origin)

    speed   = float(getattr(r, "proj_speed", 24.0))
    gravity = float(getattr(r, "proj_gravity", -21.0))
    life    = float(getattr(r, "proj_lifetime", 3.0))
    stop_on_contact = bool(getattr(r, "proj_on_contact_stop", True))
    align   = bool(getattr(r, "proj_align_object_to_velocity", True))

    inst.location = origin
    if align and speed > 0.0:
        _align_object_to_dir(inst, direction)

    # Only used for fallback (if BVH missing)
    ignore_objs = {inst, src}
    ch = _character_object()
    if ch:
        ignore_objs.add(ch)

    _ACTIVE_PROJECTILES.append({
        "obj": inst,
        "pos": origin.copy(),
        "vel": direction * speed,
        "gravity": gravity,
        "align": align,
        "stop_on_contact": stop_on_contact,
        "despawn_time": get_game_time() + max(0.0, life) if life > 0.0 else None,
        "ignore_objs": ignore_objs,
    })

# ─────────────────────────────────────────────────────────
# Fixed-step updater (30 Hz). Exact stop-on-proxy behavior via BVH.
# ─────────────────────────────────────────────────────────
def update_projectile_tasks(dt: float):
    """
    Advance active projectiles by dt (seconds). O(1) when idle.
    """
    if dt <= 0.0 and not _ACTIVE_VISUALS:
        return

    op = _active_modal()
    if not op:
        _ACTIVE_PROJECTILES.clear()
        _reclaim_visuals()
        return

    now = get_game_time()
    keep = []

    for p in _ACTIVE_PROJECTILES:
        obj = p["obj"]
        if not obj:
            continue

        # Lifetime
        endt = p.get("despawn_time")
        if endt is not None and now >= endt:
            _release(obj)
            continue

        old_pos = p["pos"]
        vel = p["vel"]

        # integrate with gravity (world -Z)
        vel.z += p["gravity"] * dt
        step = vel * dt

        # Ray between old_pos → new_pos (BVH path => exact old behavior)
        start = old_pos
        seg_len = step.length
        hit = False
        hit_loc = None

        if seg_len > 1e-7:
            ok, loc, _n, _d = _raycast_world(op, start, step / seg_len, seg_len, ignore=p.get("ignore_objs"))
            if ok:
                hit = True
                hit_loc = loc

        if hit and p.get("stop_on_contact", True):
            p["pos"] = hit_loc
            try:
                obj.location = hit_loc
            except ReferenceError:
                pass
            _release(obj)
            continue

        # keep flying
        new_pos = start + step
        p["pos"] = new_pos
        p["vel"] = vel
        try:
            obj.location = new_pos
            if p["align"] and vel.length > 1e-6:
                _align_object_to_dir(obj, vel)
        except ReferenceError:
            # instance killed externally; drop it
            continue

        keep.append(p)

    _ACTIVE_PROJECTILES[:] = keep
    _reclaim_visuals()

def _reclaim_visuals():
    """Releases TTL-based hitscan visuals that have expired."""
    if not _ACTIVE_VISUALS:
        return
    now = get_game_time()
    keep = []
    for v in _ACTIVE_VISUALS:
        obj = v.get("obj")
        if not obj:
            continue
        t = v.get("despawn_time")
        if t is not None and now >= t:
            _release(obj)
        else:
            keep.append(v)
    _ACTIVE_VISUALS[:] = keep

# ─────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────
def has_active() -> bool:
    return bool(_ACTIVE_PROJECTILES or _ACTIVE_VISUALS)

def clear():
    """
    Full cleanup:
      - Release and DELETE every pooled instance (free+active)
      - Clear active lists
    Called by reset_all_tasks() on reset/game end.
    """
    for k, pool in list(_POOLS.items()):
        for inst in list(pool["active"]):
            _destroy(inst)
        for inst in list(pool["free"]):
            _destroy(inst)
    _POOLS.clear()

    _ACTIVE_PROJECTILES.clear()
    _ACTIVE_VISUALS.clear()
