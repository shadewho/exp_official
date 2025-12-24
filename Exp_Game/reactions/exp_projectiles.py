# Exploratory/Exp_Game/reactions/exp_projectiles.py
"""
Projectile and Hitscan reaction system.

Supports two modes:
1. LOCAL: Raycasting done on main thread (legacy, simpler)
2. WORKER: Raycasting offloaded to engine worker (better for many projectiles)

Worker mode uses PROJECTILE_UPDATE_BATCH and HITSCAN_BATCH jobs.
Visual handling (spawn, position, parent) always runs on main thread (requires bpy).
"""
import bpy
import math
from mathutils import Vector
from bpy_extras import view3d_utils

from ..props_and_utils.exp_time import get_game_time
from ..audio import exp_globals  # for ACTIVE_MODAL_OP
from ..developer.dev_logger import log_game

# ─────────────────────────────────────────────────────────
# Lightweight, demand-driven projectile manager (O(1) idle)
# ─────────────────────────────────────────────────────────

# Entry schema (local tracking):
# {
#   'r_id': object,                    # the ReactionDefinition python proxy
#   'obj':  bpy.types.Object | None,   # visual clone (optional)
#   'pos':  Vector,                    # world
#   'vel':  Vector,                    # world (m/s)
#   'g':    float,                     # gravity (m/s^2)
#   'end_time': float,                 # absolute game time when it expires
#   'stop_on_contact': bool,
#   'align': bool,                     # align visual to velocity while flying
#   'worker_id': int | None,           # worker-assigned ID (if using worker mode)
# }
_active_projectiles: list[dict] = []

_active_hitscan_visuals: list[dict] = []
# All visuals we created (for safe cleanup on reset)
_spawned_visual_clones: set[bpy.types.Object] = set()

# NEW: per-reaction FIFO of visuals currently present in the scene (stuck or flying)
# Keyed by a stable owner key (see _owner_key()).
_per_reaction_visual_fifo: dict[int, list[bpy.types.Object]] = {}

# Reverse map visual -> owner key, for quick eviction bookkeeping
_visual_owner: dict[bpy.types.Object, int] = {}

_MAX_GLOBAL_PROJECTILES = 64  # hard ceiling safeguard

# --- Impact event queue ---
_impact_events: dict[int, list[dict]] = {}

# ─────────────────────────────────────────────────────────
# WORKER MODE: Pending batches for offloaded computation
# ─────────────────────────────────────────────────────────

# Pending hitscans to submit as batch
# Each: {"id": int, "origin": tuple, "direction": tuple, "max_range": float, "owner_key": int, "reaction": r}
_pending_hitscans: list[dict] = []
_inflight_hitscans: list[dict] = []  # Hitscans currently being processed by worker
_next_hitscan_id = 0

# Pending new projectiles to spawn in worker
# Each: {"pos": tuple, "vel": tuple, "gravity": float, "lifetime": float, "stop_on_contact": bool, "owner_key": int, "reaction": r, "visual_obj": obj, "client_id": int}
_pending_projectile_spawns: list[dict] = []

# Map worker projectile ID -> local tracking entry
_worker_projectile_map: dict[int, dict] = {}

# Client-generated projectile IDs (for matching spawns to worker results)
_next_client_projectile_id: int = 0

# Pending job IDs
_pending_hitscan_job_id: int | None = None
_pending_projectile_job_id: int | None = None

def _emit_impact_event(r, loc: Vector, when: float):
    key = _owner_key(r)
    _impact_events.setdefault(key, []).append({"time": float(when), "loc": loc.copy()})

def pop_impact_events_by_owner_key(owner_key: int) -> list[dict]:
    return _impact_events.pop(int(owner_key), [])

def _reaction_from_owner_key(owner_key: int):
    """Return (reaction_obj, reaction_index) from the scene by owner_key (now an index), or (None, -1)."""
    scn = bpy.context.scene
    reactions = getattr(scn, "reactions", None)
    if reactions is None:
        return None, -1
    idx = int(owner_key)
    if 0 <= idx < len(reactions):
        return reactions[idx], idx
    return None, -1

def _nodes_for_reaction_index(r_idx: int):
    """Yield Reaction nodes whose .reaction_index == r_idx in any Exploratory node tree."""
    for ng in bpy.data.node_groups:
        if getattr(ng, "bl_idname", "") != "ExploratoryNodesTreeType":
            continue
        for node in getattr(ng, "nodes", []):
            if getattr(node, "reaction_index", -1) == r_idx:
                yield node

def _dest_nodes_from_outputs(src_node):
    """Collect unique to_nodes from the Projectile/Hit-scan node's Impact outputs."""
    outs = []
    for sock in getattr(src_node, "outputs", []):
        if getattr(sock, "bl_idname", "") in {"ImpactEventOutputSocketType",
                                              "ImpactLocationOutputSocketType"}:
            outs.append(sock)
    dest = []
    seen = set()
    for s in outs:
        for lk in getattr(s, "links", []):
            n = getattr(lk, "to_node", None)
            if n and n not in seen:
                dest.append(n); seen.add(n)
    return dest

def _collect_chain_indices_from_reaction_node(start_node):
    """
    From a Reaction node, follow its 'Reaction Output' through the graph and
    return an ordered list of reaction indices to execute.
    """
    res = []
    visited = set()
    queue = [start_node]

    while queue:
        n = queue.pop(0)
        if not n or n in visited:
            continue
        visited.add(n)

        if hasattr(n, "reaction_index"):
            idx = getattr(n, "reaction_index", -1)
            if idx >= 0:
                res.append(idx)
            # only continue via the standard Reaction Output chain
            out2 = n.outputs.get("Reaction Output") or n.outputs.get("Output")
            if out2:
                for lk in out2.links:
                    if lk.to_node:
                        queue.append(lk.to_node)
        else:
            # passthrough nodes (frames/reroutes)
            for s in n.outputs:
                for lk in s.links:
                    if lk.to_node:
                        queue.append(lk.to_node)
    return res

def _flush_impact_events_to_graph():
    """Dispatch queued Impact (Bool) events via the Impact socket only."""
    if not _impact_events:
        return

    from ..interactions.exp_interactions import run_reactions, _fire_interaction  # lazy import

    scn = bpy.context.scene
    owners = list(_impact_events.keys())

    log_game("HITSCAN", f"IMPACT_FLUSH events={len(_impact_events)} owners={len(owners)}")

    for owner in owners:
        events = pop_impact_events_by_owner_key(owner)
        if not events:
            continue

        _r, r_idx = _reaction_from_owner_key(owner)
        if r_idx < 0:
            log_game("HITSCAN", f"IMPACT_FLUSH owner={owner} reaction_not_found")
            continue

        # Ensure one node per reaction index
        src_nodes = _ensure_unique_node_binding_for_reaction_index(r_idx)

        log_game("HITSCAN", f"IMPACT_FLUSH r_idx={r_idx} src_nodes={len(src_nodes)} events={len(events)}")

        for src in src_nodes:
            # Only nodes wired from the Impact (Bool) output
            dst_nodes = list(_dest_nodes_from_impact_event_outputs(src))
            log_game("HITSCAN", f"IMPACT_FLUSH src={getattr(src, 'bl_idname', '?')} dst_nodes={len(dst_nodes)}")

            for dst in dst_nodes:
                # A) Impact → Reaction Input (kick a chain)
                if hasattr(dst, "reaction_index"):
                    chain = _collect_chain_indices_from_reaction_node(dst)
                    if not chain:
                        log_game("HITSCAN", f"IMPACT_CHAIN dst={getattr(dst, 'bl_idname', '?')} chain=empty")
                        continue
                    to_run = [scn.reactions[i] for i in chain if 0 <= i < len(scn.reactions)]
                    log_game("HITSCAN", f"IMPACT_CHAIN dst={getattr(dst, 'bl_idname', '?')} chain={chain} running={len(to_run)}")
                    for _ev in events:
                        run_reactions(to_run)

                # B) Impact → Trigger node (EXTERNAL)
                elif getattr(dst, "bl_idname", "") == "ExternalTriggerNodeType" or getattr(dst, "KIND", "") == "EXTERNAL":
                    iidx = getattr(dst, "interaction_index", -1)
                    if 0 <= iidx < len(getattr(scn, "custom_interactions", [])):
                        inter = scn.custom_interactions[iidx]
                        log_game("HITSCAN", f"IMPACT_EXTERNAL trigger_idx={iidx}")
                        for ev in events:
                            _fire_interaction(inter, ev["time"])



def _ensure_unique_node_binding_for_reaction_index(r_idx: int):
    """
    Guarantee one node → one ReactionDefinition.
    If multiple nodes reference r_idx, duplicate the reaction for all but the first,
    rebind those nodes, and return the single kept node list.
    """
    # Find all nodes currently bound to this reaction index
    nodes = list(_nodes_for_reaction_index(r_idx))
    if len(nodes) <= 1:
        return nodes

    scn = bpy.context.scene
    keep = nodes[0]  # keep the first node on the original index

    # Duplicate reaction for the rest and rebind
    for n in nodes[1:]:
        try:
            res = bpy.ops.exploratory.duplicate_global_reaction('EXEC_DEFAULT', index=r_idx)
            if 'CANCELLED' in res:
                continue
            new_idx = len(scn.reactions) - 1
            n.reaction_index = new_idx
        except Exception:
            pass

    # Return only the node still bound to r_idx
    return [keep]                         


# --- Impact-Location wiring (vector only) ------------------------------------

def _reaction_index_of(r) -> int:
    """Return the index of ReactionDefinition 'r' in scene.reactions, or -1."""
    scn = bpy.context.scene
    for i, rx in enumerate(getattr(scn, "reactions", [])):
        if rx == r:
            return i
    return -1


def _dest_nodes_from_impact_event_outputs(src_node):
    """
    Collect unique to_nodes wired from the *Impact (Bool)* output only.
    (We deliberately ignore Impact Location here.)
    """
    dest = []
    seen = set()
    for sock in getattr(src_node, "outputs", []):
        if getattr(sock, "bl_idname", "") != "ImpactEventOutputSocketType":
            continue
        for lk in getattr(sock, "links", []):
            n = getattr(lk, "to_node", None)
            if n and n not in seen:
                dest.append(n); seen.add(n)
    return dest


def _push_impact_location_to_links(r, vec):
    """
    Send 'vec' (x,y,z) ONLY through the 'Impact Location' socket wiring.
    If the target node exposes write_from_graph(vec), we call it
    (generic hook for capture/data nodes).
    """
    r_idx = _reaction_index_of(r)
    if r_idx < 0:
        return

    for src in _nodes_for_reaction_index(r_idx):
        for sock in getattr(src, "outputs", []):
            if getattr(sock, "bl_idname", "") != "ImpactLocationOutputSocketType":
                continue
            for lk in getattr(sock, "links", []):
                to_node = getattr(lk, "to_node", None)
                if not to_node:
                    continue

                # Generic: if node advertises a writer, use it
                writer = getattr(to_node, "write_from_graph", None)
                if callable(writer):
                    try:
                        writer(vec, timestamp=get_game_time())
                    except Exception:
                        pass


# ---------- Public helpers ----------


def has_active() -> bool:
    return bool(_active_projectiles)


def clear():
    """
    Clear active projectiles and delete visuals we spawned.
    Safe to call on reset/cancel.
    """
    global _next_hitscan_id, _pending_hitscan_job_id, _pending_projectile_job_id, _next_client_projectile_id
    global _cached_dynamic_transforms_frame, _cached_dynamic_transforms_time

    _active_projectiles.clear()
    _active_hitscan_visuals.clear()

    to_delete = list(_spawned_visual_clones)
    _spawned_visual_clones.clear()
    for ob in to_delete:
        _hard_delete_object(ob)

    _per_reaction_visual_fifo.clear()
    _visual_owner.clear()
    _impact_events.clear()

    # Reset worker mode state
    _pending_hitscans.clear()
    _inflight_hitscans.clear()
    _pending_projectile_spawns.clear()
    _worker_projectile_map.clear()
    _next_hitscan_id = 0
    _next_client_projectile_id = 0
    _pending_hitscan_job_id = None
    _pending_projectile_job_id = None

    # Clear per-physics-step caches
    _cached_dynamic_transforms_frame = {}
    _cached_dynamic_transforms_time = -1.0


# ---------- Internals ----------
def _active_modal():
    return getattr(exp_globals, "ACTIVE_MODAL_OP", None)


def _owner_key(r) -> int:
    """
    Stable integer key for a ReactionDefinition proxy.
    Returns the reaction's index in scene.reactions.

    Note: In Blender, PropertyGroup proxies can be recreated, so we compare
    by name + reaction_type to find the right index reliably.
    """
    scn = bpy.context.scene
    reactions = getattr(scn, "reactions", None)
    if reactions is None:
        return -1

    # Get identifying attributes from the reaction
    r_name = getattr(r, "name", "")
    r_type = getattr(r, "reaction_type", "")

    for i, rx in enumerate(reactions):
        rx_name = getattr(rx, "name", "")
        rx_type = getattr(rx, "reaction_type", "")
        if rx_name == r_name and rx_type == r_type:
            return i

    # Fallback (shouldn't happen in normal use)
    return -1


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
            rv3d = getattr(area.spaces.active, "region_3d", None)
            for reg in area.regions:
                if reg.type == 'WINDOW':
                    return area, reg, rv3d
    return None, None, None


def _center_ray_world(max_range: float):
    """
    Crosshair ray in WORLD space from the active 3D View.
    Returns (origin_world, dir_world_normalized, far_point_world) or (None, None, None).
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
    scn = bpy.context.scene
    arm = getattr(scn, "target_armature", None)
    if arm:
        try:
            return (arm.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()
        except Exception:
            pass
    op = _active_modal()
    yaw = float(getattr(op, "yaw", 0.0)) if op else 0.0
    return Vector((math.cos(yaw), math.sin(yaw), 0.0)).normalized()


def _resolve_origin(r):
    """
    Spawn origin in WORLD space. Offset is applied in the origin object's LOCAL space.
    """
    scn = bpy.context.scene
    base_obj = scn.target_armature if getattr(r, "proj_use_character_origin", True) else getattr(r, "proj_origin_object", None)
    off_local = Vector(getattr(r, "proj_origin_offset", (0.0, 0.2, 1.4)))

    if base_obj:
        base_loc = base_obj.matrix_world.translation.copy()
        off_world = base_obj.matrix_world.to_3x3() @ off_local
        return base_loc + off_world

    # No base object → treat offset as world-space
    return off_local.copy()


def _raycast_any(op, origin: Vector, direction: Vector, max_dist: float):
    """
    Ray against STATIC BVH only.
    Dynamic mesh collision handled by worker (KCC physics).
    Returns (hit:bool, loc:Vector|None, normal:Vector|None, hit_obj:bpy.types.Object|None, dist:float|None)
    """
    if not op or max_dist <= 1e-9 or direction.length <= 1e-9:
        return (False, None, None, None, None)

    static_bvh = getattr(op, "bvh_tree", None)
    if not static_bvh:
        return (False, None, None, None, None)

    dnorm = direction.normalized()
    hit = static_bvh.ray_cast(origin, dnorm, max_dist)
    if hit and hit[0] is not None:
        return (True, hit[0], hit[1], None, hit[3])
    return (False, None, None, None, None)


def _resolve_direction(r, origin):
    """
    Resolve a world-space direction using proj_aim_source.
    If CROSSHAIR: aim from camera center across static/dynamic; else use character forward.

    NOTE: When aiming at dynamic meshes, we can't raycast them here (no bpy BVH for dynamic).
    If static raycast misses, we use camera direction directly instead of aiming at a far point.
    This ensures the shot goes where the crosshair is pointing, regardless of what's there.
    """
    aim_src = (getattr(r, "proj_aim_source", "CROSSHAIR") or "CROSSHAIR")
    if aim_src == "CAMERA":  # legacy alias
        aim_src = "CROSSHAIR"

    if aim_src == "CROSSHAIR":
        hs_range = float(getattr(r, "proj_max_range", 60.0))
        cam_o, cam_dir, cam_far = _center_ray_world(max_range=hs_range)
        if cam_o is not None:
            op = _active_modal()
            # Only tests static geometry - dynamic meshes handled by worker
            hit, loc, _n, _obj, _d = _raycast_any(op, cam_o, cam_dir, hs_range)
            if hit and loc:
                # Static hit: aim at that point
                return (loc - origin).normalized()
            else:
                # No static hit: aim at where crosshair points at max_range
                # This makes the shot converge with crosshair at distance
                # (using cam_dir directly would create parallel offset, missing the target)
                far_point = cam_o + cam_dir * hs_range
                return (far_point - origin).normalized()
        return _character_forward_from_scene()

    # CHAR_FORWARD
    return _character_forward_from_scene()


def _align_object_to_dir(obj, direction: Vector):
    if not obj:
        return
    try:
        q = direction.normalized().to_track_quat('Y', 'Z')  # +Y forward, Z up
        obj.rotation_euler = q.to_euler('XYZ')
    except Exception:
        pass


def _spawn_visual_instance(template_obj: bpy.types.Object | None, owner_key: int) -> bpy.types.Object | None:
    """
    Create a lightweight visual clone (linked mesh). Returns the new object or None.
    Also registers it in the per-reaction FIFO and reverse map.
    """
    if not template_obj:
        return None
    try:
        new_ob = bpy.data.objects.new(name=f"{template_obj.name}_Shot", object_data=template_obj.data)
        bpy.context.scene.collection.objects.link(new_ob)
        new_ob.hide_viewport = False
        new_ob.hide_render = False
        new_ob.display_type = template_obj.display_type
        new_ob.scale = template_obj.scale.copy()

        _spawned_visual_clones.add(new_ob)
        _visual_owner[new_ob] = owner_key
        _per_reaction_visual_fifo.setdefault(owner_key, []).append(new_ob)
        return new_ob
    except Exception:
        return None


def _hard_delete_object(ob: bpy.types.Object | None):
    """Unparent, unlink, and remove an object datablock safely."""
    if not ob:
        return
    try:
        ob.parent = None
        for coll in list(ob.users_collection):
            try:
                coll.objects.unlink(ob)
            except Exception:
                pass
        bpy.data.objects.remove(ob, do_unlink=True)
    except Exception:
        pass


def _remove_active_entries_by_visual(ob: bpy.types.Object | None):
    """Remove any active sim entries that reference this visual."""
    if not ob or not _active_projectiles:
        return
    keep = []
    for p in _active_projectiles:
        if p.get('obj') is ob:
            # drop entry
            continue
        keep.append(p)
    _active_projectiles[:] = keep


def _delete_visual_if_clone(ob: bpy.types.Object | None):
    """Remove 'ob' only if it is one of the clones we created, and update FIFO bookkeeping."""
    if not ob or ob not in _spawned_visual_clones:
        return
    _spawned_visual_clones.discard(ob)

    # Remove from FIFO & reverse map
    key = _visual_owner.pop(ob, None)
    if key is not None:
        lst = _per_reaction_visual_fifo.get(key)
        if lst:
            try:
                lst.remove(ob)
            except ValueError:
                pass
            if not lst:
                _per_reaction_visual_fifo.pop(key, None)

    # If still flying, drop its sim entry too
    _remove_active_entries_by_visual(ob)

    _hard_delete_object(ob)


def _prune_expired(now: float):
    """
    Remove expired projectiles immediately (so counts are accurate when firing rapidly).
    Deletes their visuals too.
    """
    if not _active_projectiles:
        return
    keep = []
    for p in _active_projectiles:
        if now >= p['end_time']:
            _delete_visual_if_clone(p.get('obj'))
        else:
            keep.append(p)
    _active_projectiles[:] = keep


def _evict_oldest_visuals_for_reaction(owner_key: int, n_to_evict: int):
    """
    FIFO eviction on the real, visible OBJECTS for this reaction.
    Also removes their active sim entries if still flying.
    """
    if n_to_evict <= 0:
        return
    lst = _per_reaction_visual_fifo.get(owner_key)
    if not lst:
        return
    count = min(n_to_evict, len(lst))
    for _ in range(count):
        ob = lst.pop(0)
        _delete_visual_if_clone(ob)
    if not lst:
        _per_reaction_visual_fifo.pop(owner_key, None)


def _evict_oldest_active_for_reaction(r, n_to_evict: int):
    """
    Fallback FIFO eviction for reactions with NO visuals (just sim entries).
    """
    if n_to_evict <= 0 or not _active_projectiles:
        return
    removed = 0
    keep = []
    for p in _active_projectiles:
        if removed < n_to_evict and p.get('r_id') is r:
            _delete_visual_if_clone(p.get('obj'))  # usually None in this mode
            removed += 1
            continue
        keep.append(p)
    _active_projectiles[:] = keep


# ─────────────────────────────────────────────────────────
# Executors (called by reaction dispatcher)
# ─────────────────────────────────────────────────────────

def execute_hitscan_reaction(r):
    """
    Execute a hitscan reaction.
    Queues the hitscan for batch submission to worker.
    """
    global _next_hitscan_id

    op = _active_modal()
    if not op:
        return

    origin = _resolve_origin(r)
    direction = _resolve_direction(r, origin)
    max_range = float(getattr(r, "proj_max_range", 60.0))
    owner = _owner_key(r)

    # Queue for worker batch submission
    _next_hitscan_id += 1
    _pending_hitscans.append({
        "id": _next_hitscan_id,
        "origin": (origin.x, origin.y, origin.z),
        "direction": (direction.x, direction.y, direction.z),
        "max_range": max_range,
        "owner_key": owner,
        "reaction": r,
    })

    log_game("HITSCAN", f"QUEUE id={_next_hitscan_id} origin=({origin.x:.2f},{origin.y:.2f},{origin.z:.2f}) dir=({direction.x:.2f},{direction.y:.2f},{direction.z:.2f}) range={max_range:.1f}")





def execute_projectile_reaction(r):
    """
    Spawn a simulated projectile - physics runs on worker.
    Enforces per-reaction FIFO cap on the actual visible objects:
      - If adding would exceed r.proj_pool_limit, delete the oldest stuck/flying visuals first.
      - If no visual template is set, cap the active sim entries instead.
    """
    op = _active_modal()
    if not op:
        return

    now = get_game_time()
    _prune_expired(now)  # keep counts honest when spamming shots

    # Global guardrail (simple)
    if len(_active_projectiles) >= _MAX_GLOBAL_PROJECTILES:
        return

    # Per-reaction cap (≥1)
    per_cap = int(getattr(r, "proj_pool_limit", 8) or 8)
    if per_cap < 1:
        per_cap = 1

    template = getattr(r, "proj_object", None)
    owner = _owner_key(r)

    if template is not None:
        # Enforce on **visuals present** (stuck or flying)
        present = len(_per_reaction_visual_fifo.get(owner, []))
        if present >= per_cap:
            _evict_oldest_visuals_for_reaction(owner, present - per_cap + 1)
    else:
        # No visuals → enforce on active sim entries only
        acc = sum(1 for p in _active_projectiles if p.get('r_id') is r)
        if acc >= per_cap:
            _evict_oldest_active_for_reaction(r, acc - per_cap + 1)

    # Resolve spawn state AFTER eviction
    origin = _resolve_origin(r)
    direction = _resolve_direction(r, origin)

    speed = float(getattr(r, "proj_speed", 24.0))
    g     = float(getattr(r, "proj_gravity", -21.0))
    life  = float(getattr(r, "proj_lifetime", 3.0))

    vel = direction * speed
    align = bool(getattr(r, "proj_align_object_to_velocity", True))
    stop_on_contact = bool(getattr(r, "proj_on_contact_stop", True))

    # Visual: clone per shot (tracked in FIFO)
    pov = _spawn_visual_instance(template, owner) if template is not None else None
    if pov:
        pov.matrix_world.translation = origin
        if align and vel.length > 1e-6:
            _align_object_to_dir(pov, vel)

    # Generate client-side ID for matching
    global _next_client_projectile_id
    _next_client_projectile_id += 1
    client_id = _next_client_projectile_id

    # Queue spawn for worker batch submission
    _pending_projectile_spawns.append({
        "client_id": client_id,
        "pos": (origin.x, origin.y, origin.z),
        "vel": (vel.x, vel.y, vel.z),
        "gravity": g,
        "lifetime": life,
        "stop_on_contact": stop_on_contact,
        "owner_key": owner,
        "reaction": r,
        "visual_obj": pov,
        "align": align,
    })

    log_game("PROJECTILE", f"SPAWN cid={client_id} origin=({origin.x:.2f},{origin.y:.2f},{origin.z:.2f}) vel=({vel.x:.2f},{vel.y:.2f},{vel.z:.2f}) speed={speed:.1f} gravity={g:.1f} life={life:.1f}s")


# ─────────────────────────────────────────────────────────
# Local visual interpolation (call EVERY FRAME for smooth motion)
# ─────────────────────────────────────────────────────────

def interpolate_projectile_visuals(dt: float):
    """
    Update projectile visual positions locally every frame.
    This provides smooth motion while waiting for worker results.
    Worker handles collision detection; this is just visual interpolation.

    Uses cumulative time tracking to extrapolate correctly across multiple frames.
    Worker results reset the extrapolation base.
    Also prunes expired projectiles locally (safety net if worker results delayed).
    """
    if not _active_projectiles:
        return

    # Local lifetime check (safety net - worker also checks, but this ensures cleanup)
    now = get_game_time()
    keep = []
    for p in _active_projectiles:
        if now >= p.get('end_time', float('inf')):
            _delete_visual_if_clone(p.get('obj'))
            # Clean up worker map if we had a worker_id
            worker_id = p.get('worker_id')
            if worker_id and worker_id in _worker_projectile_map:
                del _worker_projectile_map[worker_id]
            continue
        keep.append(p)
    _active_projectiles[:] = keep

    for p in _active_projectiles:
        obj_v = p.get('obj')
        if not obj_v:
            continue

        # Get stored velocity (base from last worker update)
        vel = p.get('vel')
        if vel is None:
            continue

        pos = p.get('pos')
        if pos is None:
            continue

        g = p.get('g', 0.0)
        align = p.get('align', True)

        # Track cumulative time since last worker update (for extrapolation)
        interp_time = p.get('_interp_time', 0.0) + dt
        p['_interp_time'] = interp_time

        # Extrapolate from base pos/vel using cumulative time
        # This gives smooth motion even when worker updates are delayed
        # v(t) = v0 + g*t
        # p(t) = p0 + v0*t + 0.5*g*t^2
        predicted_pos = Vector((
            pos.x + vel.x * interp_time,
            pos.y + vel.y * interp_time,
            pos.z + vel.z * interp_time + 0.5 * g * interp_time * interp_time
        ))

        # Update visual only (not stored state)
        obj_v.location = predicted_pos

        # Align to predicted velocity: v(t) = v0 + g*t
        if align:
            predicted_vel = Vector((vel.x, vel.y, vel.z + g * interp_time))
            if predicted_vel.length > 1e-6:
                _align_object_to_dir(obj_v, predicted_vel)


# ─────────────────────────────────────────────────────────
# 30 Hz updater (call from your fixed-step loop)
# ─────────────────────────────────────────────────────────

def update_projectile_tasks(dt: float):
    """
    Advance active projectiles by dt (seconds). O(1) when idle.
    On contact:
      • freeze visual at impact (and parent to dynamic hit object),
      • push Impact Location vector to linked nodes,
      • queue Impact (Bool) event,
      • remove the sim entry (visual remains for FIFO/eviction).
    """
    if not _active_projectiles:
        return

    op = _active_modal()
    if not op:
        clear()
        return

    now = get_game_time()
    keep: list[dict] = []

    for p in _active_projectiles:
        # lifetime check
        if now >= p['end_time']:
            _delete_visual_if_clone(p.get('obj'))
            continue

        old_pos = p['pos']
        vel = p['vel']

        # Integrate gravity and predict next position
        vel.z += p['g'] * dt
        new_pos = old_pos + vel * dt

        # Segment sweep: old_pos -> new_pos
        seg = new_pos - old_pos
        seg_len = seg.length

        hit = False
        hit_loc = None
        hit_obj = None

        if seg_len > 1e-7:
            ok, loc, _n, obj, _d = _raycast_any(op, old_pos, seg / seg_len, seg_len)
            if ok:
                hit = True
                hit_loc = loc
                hit_obj = obj

        if hit and p['stop_on_contact']:
            # Freeze visual at impact (stick to dynamic if present)
            p['pos'] = hit_loc
            obj_v = p.get('obj')
            if obj_v:
                obj_v.location = hit_loc
                if hit_obj:
                    try:
                        obj_v.parent = hit_obj
                        obj_v.matrix_parent_inverse = hit_obj.matrix_world.inverted()
                    except Exception:
                        pass

            # Vector-only ping through Impact Location socket + queue bool event
            _push_impact_location_to_links(p.get('r_id'), (hit_loc.x, hit_loc.y, hit_loc.z))
            _emit_impact_event(p.get('r_id'), hit_loc, now)

            # Despawn sim entry (visual remains for FIFO/eviction)
            continue

        # Keep flying
        p['pos'] = new_pos
        p['vel'] = vel

        obj_v = p.get('obj')
        if obj_v:
            obj_v.location = new_pos
            if p['align'] and vel.length > 1e-6:
                _align_object_to_dir(obj_v, vel)

        keep.append(p)

    _active_projectiles[:] = keep
    _flush_impact_events_to_graph()


def update_hitscan_tasks():
    """
    Despawn cloned hitscan visuals when their lifetime ends.
    O(1) idle overhead. Safe if called multiple times per frame.
    """
    if not _active_hitscan_visuals:
        return
    now = get_game_time()
    keep: list[dict] = []
    for rec in _active_hitscan_visuals:
        ob = rec.get("obj")
        # Dropped earlier via eviction or already deleted?
        if not ob or ob not in _spawned_visual_clones:
            continue
        if now >= float(rec.get("end_time", 0.0)):
            _delete_visual_if_clone(ob)
            continue
        keep.append(rec)
    _active_hitscan_visuals[:] = keep


# ─────────────────────────────────────────────────────────
# WORKER: Job submission and result processing
# ─────────────────────────────────────────────────────────

# Per-physics-step cache for dynamic transforms (avoid recomputing for hitscan + projectile)
_cached_dynamic_transforms_frame: dict = {}
_cached_dynamic_transforms_time: float = -1.0


def get_dynamic_transforms() -> dict:
    """
    Get current dynamic mesh transforms for worker jobs.
    Returns {obj_id: matrix_16_tuple, ...}

    Cached per-physics-step to avoid recomputing when both hitscan and projectile
    systems submit jobs in the same physics step.
    """
    global _cached_dynamic_transforms_frame, _cached_dynamic_transforms_time

    op = _active_modal()
    if not op:
        return {}

    # Use game time for cache invalidation (changes each physics step)
    current_time = get_game_time()
    if current_time == _cached_dynamic_transforms_time and _cached_dynamic_transforms_frame:
        return _cached_dynamic_transforms_frame

    transforms = {}
    # Use dynamic_objects_map which is populated by exp_dynamic.update_dynamic_meshes()
    # Maps: Blender object -> bounding radius
    dynamic_objects_map = getattr(op, 'dynamic_objects_map', {})
    for obj, _radius in dynamic_objects_map.items():
        if obj and hasattr(obj, 'matrix_world'):
            obj_id = id(obj)  # Must match the obj_id used in CACHE_DYNAMIC_MESH
            m = obj.matrix_world
            transforms[obj_id] = (
                m[0][0], m[0][1], m[0][2], m[0][3],
                m[1][0], m[1][1], m[1][2], m[1][3],
                m[2][0], m[2][1], m[2][2], m[2][3],
                m[3][0], m[3][1], m[3][2], m[3][3],
            )

    # Cache for this physics step
    _cached_dynamic_transforms_frame = transforms
    _cached_dynamic_transforms_time = current_time
    return transforms


def submit_hitscan_batch(engine) -> int | None:
    """
    Submit pending hitscans as a HITSCAN_BATCH job to the engine.
    Call this once per frame after all execute_hitscan_reaction() calls.

    Args:
        engine: EngineCore instance

    Returns:
        job_id if submitted, None if no pending hitscans
    """
    global _pending_hitscans, _pending_hitscan_job_id

    # Early exit: nothing to do
    if not _pending_hitscans or not engine or not engine.is_alive():
        return None

    # Don't queue another job if one is already pending (prevents backup)
    if _pending_hitscan_job_id is not None:
        return None

    # Build job data
    rays = []
    for hs in _pending_hitscans:
        rays.append({
            "id": hs["id"],
            "origin": hs["origin"],
            "direction": hs["direction"],
            "max_range": hs["max_range"],
            "owner_key": hs["owner_key"],
        })

    dynamic_transforms = get_dynamic_transforms()
    job_data = {
        "rays": rays,
        "dynamic_transforms": dynamic_transforms,
    }

    job_id = engine.submit_job("HITSCAN_BATCH", job_data)
    if job_id is not None and job_id >= 0:
        _pending_hitscan_job_id = job_id
        # Move submitted hitscans to inflight (for result matching)
        # This prevents losing hitscans added between submit and result
        global _inflight_hitscans
        _inflight_hitscans = list(_pending_hitscans)
        _pending_hitscans.clear()
        log_game("HITSCAN", f"BATCH_SUBMIT job={job_id} rays={len(rays)} dyn_transforms={len(dynamic_transforms)}")
        return job_id

    return None


def process_hitscan_results(result) -> int:
    """
    Process HITSCAN_BATCH result and apply visuals/impacts.

    Args:
        result: EngineResult from HITSCAN_BATCH job

    Returns:
        Number of hitscans processed
    """
    global _pending_hitscan_job_id, _inflight_hitscans

    if not result.success or result.job_id != _pending_hitscan_job_id:
        return 0

    _pending_hitscan_job_id = None
    results_list = result.result.get("results", [])

    processed = 0
    now = get_game_time()

    # Build lookup for inflight hitscans by ID (these are the ones we submitted)
    pending_by_id = {hs["id"]: hs for hs in _inflight_hitscans}

    # Build obj_id -> Blender object lookup for parenting to dynamic meshes
    op = _active_modal()
    dynamic_obj_lookup = {}
    if op:
        dynamic_objects_map = getattr(op, 'dynamic_objects_map', {})
        for obj, _radius in dynamic_objects_map.items():
            if obj:
                dynamic_obj_lookup[id(obj)] = obj

    for res in results_list:
        hs_id = res.get("id")
        hs_data = pending_by_id.get(hs_id)
        if not hs_data:
            continue

        r = hs_data.get("reaction")
        if not r:
            continue

        hit = res.get("hit", False)
        pos = res.get("pos")
        owner_key = hs_data["owner_key"]
        hit_source = res.get("source", "static")
        hit_obj_id = res.get("obj_id")
        hit_dist = res.get("distance", 0)

        # Convert pos tuple to Vector
        impact = Vector(pos) if pos else Vector(hs_data["origin"])

        # Debug: log hit details for all results to diagnose dynamic mesh issues
        origin = hs_data.get("origin", (0,0,0))
        direction = hs_data.get("direction", (0,0,0))
        if hit:
            log_game("HITSCAN", f"RESULT hit=True source={hit_source} dist={hit_dist:.3f} pos={pos} obj_id={hit_obj_id}")
        else:
            # Check if origin and pos are suspiciously close (indicates direction or range issue)
            if pos:
                dist_from_origin = ((pos[0]-origin[0])**2 + (pos[1]-origin[1])**2 + (pos[2]-origin[2])**2)**0.5
                log_game("HITSCAN", f"RESULT hit=False origin={origin} dir={direction} pos={pos} dist_from_origin={dist_from_origin:.3f}")

        # Visual handling (same as local mode)
        template = getattr(r, "proj_object", None)
        place = bool(getattr(r, "proj_place_hitscan_object", True))
        align = bool(getattr(r, "proj_align_object_to_velocity", True))
        direction = Vector(hs_data["direction"])

        if template and place:
            life = float(getattr(r, "proj_lifetime", 0.0) or 0.0)
            if life > 0.0:
                cap = int(getattr(r, "proj_pool_limit", 8) or 8)
                if cap < 1:
                    cap = 1

                present = len(_per_reaction_visual_fifo.get(owner_key, []))
                if present >= cap:
                    _evict_oldest_visuals_for_reaction(owner_key, present - cap + 1)

                new_ob = _spawn_visual_instance(template, owner_key)
                if new_ob:
                    new_ob.location = impact
                    if align:
                        _align_object_to_dir(new_ob, direction)

                    # Parent to dynamic mesh so hitscan visual moves/rotates with it
                    if hit_source == "dynamic" and hit_obj_id:
                        hit_obj = dynamic_obj_lookup.get(hit_obj_id)
                        if hit_obj:
                            try:
                                # Use impact directly - matrix_world isn't updated until depsgraph eval
                                world_pos = impact.copy()
                                world_rot = new_ob.rotation_euler.to_quaternion()

                                # Parent with identity inverse (local = relative to parent)
                                new_ob.parent = hit_obj
                                new_ob.matrix_parent_inverse.identity()

                                # Compute local position: where in parent's local space is world_pos?
                                local_pos = hit_obj.matrix_world.inverted() @ world_pos
                                new_ob.location = local_pos

                                # Compute local rotation: world_rot relative to parent's rotation
                                parent_rot = hit_obj.matrix_world.to_quaternion()
                                local_rot = parent_rot.inverted() @ world_rot
                                new_ob.rotation_euler = local_rot.to_euler('XYZ')

                                log_game("HITSCAN", f"PARENTED to dynamic mesh {hit_obj.name}")
                            except Exception as e:
                                log_game("HITSCAN", f"PARENT_FAILED: {e}")

                    _active_hitscan_visuals.append({"obj": new_ob, "end_time": now + life})
            else:
                try:
                    template.location = impact
                    if align:
                        _align_object_to_dir(template, direction)
                except Exception:
                    pass

        # Impact events
        if hit:
            _push_impact_location_to_links(r, (impact.x, impact.y, impact.z))
            _emit_impact_event(r, impact, now)
            dist = res.get("distance", 0.0)
            log_game("HITSCAN", f"HIT id={hs_id} source={hit_source} pos=({impact.x:.2f},{impact.y:.2f},{impact.z:.2f}) dist={dist:.2f}")
        else:
            log_game("HITSCAN", f"MISS id={hs_id}")

        processed += 1

    # Clear inflight (pending was already cleared on submit)
    _inflight_hitscans.clear()

    # Log batch results
    hits = sum(1 for r in results_list if r.get("hit", False))
    log_game("HITSCAN", f"BATCH_RESULT processed={processed} hits={hits} misses={processed - hits}")

    # Flush impact events
    if processed > 0:
        _flush_impact_events_to_graph()

    return processed


def submit_projectile_update(engine, dt: float) -> int | None:
    """
    Submit PROJECTILE_UPDATE_BATCH job to the engine.
    Includes pending spawns and update for all active projectiles.

    Args:
        engine: EngineCore instance
        dt: Time step in seconds

    Returns:
        job_id if submitted, None if nothing to update
    """
    global _pending_projectile_spawns, _pending_projectile_job_id

    if not engine or not engine.is_alive():
        return None

    # Nothing to do if no active projectiles and no pending spawns
    if not _active_projectiles and not _pending_projectile_spawns:
        return None

    # Don't queue another job if one is already pending (prevents backup)
    # Worker will process current job and we'll submit next frame
    if _pending_projectile_job_id is not None:
        return None

    # Build new projectiles data (include client_id for result matching)
    new_projectiles = []
    for spawn in _pending_projectile_spawns:
        new_projectiles.append({
            "client_id": spawn["client_id"],
            "pos": spawn["pos"],
            "vel": spawn["vel"],
            "gravity": spawn["gravity"],
            "lifetime": spawn["lifetime"],
            "stop_on_contact": spawn["stop_on_contact"],
            "owner_key": spawn["owner_key"],
        })

    job_data = {
        "dt": dt,
        "game_time": get_game_time(),
        "new_projectiles": new_projectiles,
        "dynamic_transforms": get_dynamic_transforms(),
    }

    job_id = engine.submit_job("PROJECTILE_UPDATE_BATCH", job_data)
    if job_id is not None and job_id >= 0:
        _pending_projectile_job_id = job_id
        log_game("PROJECTILE", f"BATCH_SUBMIT job={job_id} active={len(_active_projectiles)} new={len(new_projectiles)} dt={dt*1000:.1f}ms")

        # Track spawns for result processing (use client_id for matching)
        for spawn in _pending_projectile_spawns:
            entry = {
                'r_id': spawn["reaction"],
                'obj': spawn.get("visual_obj"),
                'pos': Vector(spawn["pos"]),
                'vel': Vector(spawn["vel"]),
                'g': spawn["gravity"],
                'end_time': get_game_time() + spawn["lifetime"],
                'stop_on_contact': spawn["stop_on_contact"],
                'align': spawn.get("align", True),
                'owner_key': spawn["owner_key"],
                'client_id': spawn["client_id"],  # For matching with worker results
                'worker_id': None,  # Assigned when result arrives
            }
            _active_projectiles.append(entry)

        _pending_projectile_spawns.clear()
        return job_id

    return None


def process_projectile_results(result) -> int:
    """
    Process PROJECTILE_UPDATE_BATCH result and update visuals/impacts.

    Args:
        result: EngineResult from PROJECTILE_UPDATE_BATCH job

    Returns:
        Number of projectiles updated
    """
    global _pending_projectile_job_id, _worker_projectile_map

    if not result.success or result.job_id != _pending_projectile_job_id:
        return 0

    _pending_projectile_job_id = None
    result_data = result.result

    updated_projectiles = result_data.get("updated_projectiles", [])
    impacts = result_data.get("impacts", [])
    expired = result_data.get("expired", [])

    now = get_game_time()
    processed = 0

    # Build client_id -> local entry map for matching
    client_id_map = {p.get('client_id'): p for p in _active_projectiles if p.get('client_id')}

    # Match worker results to local entries by client_id
    for up in updated_projectiles:
        worker_id = up.get("id")
        client_id = up.get("client_id", 0)

        # Find entry by client_id (reliable) instead of by order (fragile)
        entry = client_id_map.get(client_id)
        if entry and entry.get('worker_id') is None:
            entry['worker_id'] = worker_id
            _worker_projectile_map[worker_id] = entry

    # Update positions for active projectiles
    for up in updated_projectiles:
        worker_id = up.get("id")
        client_id = up.get("client_id", 0)
        active = up.get("active", True)
        pos = up.get("pos")
        vel = up.get("vel")

        # Try worker_id first (already assigned), then fall back to client_id
        entry = _worker_projectile_map.get(worker_id) or client_id_map.get(client_id)
        if not entry:
            continue

        if pos:
            entry['pos'] = Vector(pos)
        if vel:
            entry['vel'] = Vector(vel)

        # Reset interpolation time (worker results are new base)
        entry['_interp_time'] = 0.0

        # Update visual
        obj_v = entry.get('obj')
        if obj_v and pos:
            obj_v.location = Vector(pos)
            if entry.get('align') and vel:
                vel_v = Vector(vel)
                if vel_v.length > 1e-6:
                    _align_object_to_dir(obj_v, vel_v)

        processed += 1

    # Build obj_id -> Blender object lookup for parenting to dynamic meshes
    op = _active_modal()
    dynamic_obj_lookup = {}
    if op:
        dynamic_objects_map = getattr(op, 'dynamic_objects_map', {})
        for obj, _radius in dynamic_objects_map.items():
            if obj:
                dynamic_obj_lookup[id(obj)] = obj

    # Process impacts
    for impact in impacts:
        worker_id = impact.get("id")
        client_id = impact.get("client_id", 0)
        owner_key = impact.get("owner_key")
        pos = impact.get("pos")
        hit_source = impact.get("source", "static")
        hit_obj_id = impact.get("obj_id")

        # Try worker_id first, then fall back to client_id
        entry = _worker_projectile_map.get(worker_id) or client_id_map.get(client_id)
        if entry:
            r = entry.get('r_id')
            obj_v = entry.get('obj')

            # Freeze visual at impact
            if obj_v and pos:
                obj_v.location = Vector(pos)

                # Parent to dynamic mesh so projectile moves/rotates with it
                if hit_source == "dynamic" and hit_obj_id:
                    hit_obj = dynamic_obj_lookup.get(hit_obj_id)
                    if hit_obj:
                        try:
                            # Use pos directly - matrix_world isn't updated until depsgraph eval
                            world_pos = Vector(pos)
                            world_rot = obj_v.rotation_euler.to_quaternion()

                            # Parent with identity inverse (local = relative to parent)
                            obj_v.parent = hit_obj
                            obj_v.matrix_parent_inverse.identity()

                            # Compute local position: where in parent's local space is world_pos?
                            local_pos = hit_obj.matrix_world.inverted() @ world_pos
                            obj_v.location = local_pos

                            # Compute local rotation: world_rot relative to parent's rotation
                            parent_rot = hit_obj.matrix_world.to_quaternion()
                            local_rot = parent_rot.inverted() @ world_rot
                            obj_v.rotation_euler = local_rot.to_euler('XYZ')

                            log_game("PROJECTILE", f"PARENTED to dynamic mesh {hit_obj.name}")
                        except Exception as e:
                            log_game("PROJECTILE", f"PARENT_FAILED: {e}")

            # Impact events
            if r and pos:
                pos_v = Vector(pos)
                _push_impact_location_to_links(r, (pos_v.x, pos_v.y, pos_v.z))
                _emit_impact_event(r, pos_v, now)
                log_game("PROJECTILE", f"IMPACT id={worker_id} source={hit_source} pos=({pos_v.x:.2f},{pos_v.y:.2f},{pos_v.z:.2f})")

            # Remove from active list (keep visual for FIFO)
            if entry in _active_projectiles:
                _active_projectiles.remove(entry)

            # Clean up map
            if worker_id in _worker_projectile_map:
                del _worker_projectile_map[worker_id]

    # Handle expired projectiles (now includes client_id for reliable matching)
    for exp_data in expired:
        # Support both old format (int) and new format (dict with id/client_id)
        if isinstance(exp_data, dict):
            worker_id = exp_data.get("id")
            client_id = exp_data.get("client_id", 0)
        else:
            worker_id = exp_data
            client_id = 0

        # Try worker_id first, then fall back to client_id
        entry = _worker_projectile_map.get(worker_id) or client_id_map.get(client_id)
        if entry:
            _delete_visual_if_clone(entry.get('obj'))
            if entry in _active_projectiles:
                _active_projectiles.remove(entry)
            if worker_id and worker_id in _worker_projectile_map:
                del _worker_projectile_map[worker_id]

    # Log batch results
    log_game("PROJECTILE", f"BATCH_RESULT updated={processed} impacts={len(impacts)} expired={len(expired)}")

    # Flush impact events
    if impacts:
        _flush_impact_events_to_graph()

    return processed


def has_pending_jobs() -> bool:
    """Check if there are pending worker jobs."""
    return _pending_hitscan_job_id is not None or _pending_projectile_job_id is not None


def get_pending_job_ids() -> tuple:
    """Get pending job IDs for result matching."""
    return (_pending_hitscan_job_id, _pending_projectile_job_id)