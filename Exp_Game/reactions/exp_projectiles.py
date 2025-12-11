# Exploratory/Exp_Game/reactions/exp_projectiles.py
import bpy
import math
from mathutils import Vector
from bpy_extras import view3d_utils

from ..props_and_utils.exp_time import get_game_time
from ..audio import exp_globals  # for ACTIVE_MODAL_OP

# ─────────────────────────────────────────────────────────
# Lightweight, demand-driven projectile manager (O(1) idle)
# ─────────────────────────────────────────────────────────

# Entry schema:
# {
#   'r_id': object,                    # the ReactionDefinition python proxy
#   'obj':  bpy.types.Object | None,   # visual clone (optional)
#   'pos':  Vector,                    # world
#   'vel':  Vector,                    # world (m/s)
#   'g':    float,                     # gravity (m/s^2)
#   'end_time': float,                 # absolute game time when it expires
#   'stop_on_contact': bool,
#   'align': bool,                     # align visual to velocity while flying
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

def _emit_impact_event(r, loc: Vector, when: float):
    key = _owner_key(r)
    _impact_events.setdefault(key, []).append({"time": float(when), "loc": loc.copy()})

def pop_impact_events_by_owner_key(owner_key: int) -> list[dict]:
    return _impact_events.pop(int(owner_key), [])

def _reaction_from_owner_key(owner_key: int):
    """Return (reaction_obj, reaction_index) from the scene by owner_key, or (None, -1)."""
    scn = bpy.context.scene
    for i, rx in enumerate(getattr(scn, "reactions", [])):
        try:
            if _owner_key(rx) == int(owner_key):
                return rx, i
        except Exception:
            pass
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
    for owner in owners:
        events = pop_impact_events_by_owner_key(owner)
        if not events:
            continue

        _r, r_idx = _reaction_from_owner_key(owner)
        if r_idx < 0:
            continue

        # Ensure one node per reaction index
        src_nodes = _ensure_unique_node_binding_for_reaction_index(r_idx)

        for src in src_nodes:
            # Only nodes wired from the Impact (Bool) output
            for dst in _dest_nodes_from_impact_event_outputs(src):
                # A) Impact → Reaction Input (kick a chain)
                if hasattr(dst, "reaction_index"):
                    chain = _collect_chain_indices_from_reaction_node(dst)
                    if not chain:
                        continue
                    to_run = [scn.reactions[i] for i in chain if 0 <= i < len(scn.reactions)]
                    for _ev in events:
                        run_reactions(to_run)

                # B) Impact → Trigger node (EXTERNAL)
                elif getattr(dst, "bl_idname", "") == "ExternalTriggerNodeType" or getattr(dst, "KIND", "") == "EXTERNAL":
                    iidx = getattr(dst, "interaction_index", -1)
                    if 0 <= iidx < len(getattr(scn, "custom_interactions", [])):
                        inter = scn.custom_interactions[iidx]
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
    If a link goes to a UtilityCaptureFloatVector node, we write straight to its store.
    No reaction chains, no triggers. If the target node exposes write_from_graph(vec),
    we call it (generic hook for future capture nodes).
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

                # Preferred: our capture node
                if getattr(to_node, "bl_idname", "") == "UtilityCaptureFloatVectorNodeType":
                    try:
                        to_node.write_from_graph(vec, timestamp=get_game_time())
                    except Exception:
                        pass
                    continue

                # Generic fallback: if node advertises a writer, use it
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
    _active_projectiles.clear()
    _active_hitscan_visuals.clear()

    to_delete = list(_spawned_visual_clones)
    _spawned_visual_clones.clear()
    for ob in to_delete:
        _hard_delete_object(ob)

    _per_reaction_visual_fifo.clear()
    _visual_owner.clear()
    _impact_events.clear()


# ---------- Internals ----------
def _active_modal():
    return getattr(exp_globals, "ACTIVE_MODAL_OP", None)


def _owner_key(r) -> int:
    """
    Stable integer key for a ReactionDefinition proxy.
    """
    try:
        return int(r.as_pointer())
    except Exception:
        return id(r)


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
    """
    aim_src = (getattr(r, "proj_aim_source", "CROSSHAIR") or "CROSSHAIR")
    if aim_src == "CAMERA":  # legacy alias
        aim_src = "CROSSHAIR"

    if aim_src == "CROSSHAIR":
        hs_range = float(getattr(r, "proj_max_range", 60.0))
        cam_o, cam_dir, cam_far = _center_ray_world(max_range=hs_range)
        if cam_o is not None:
            op = _active_modal()
            hit, loc, _n, _obj, _d = _raycast_any(op, cam_o, cam_dir, hs_range)
            target = loc if hit else cam_far
            return (target - origin).normalized()
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
    op = _active_modal()
    if not op:
        return

    origin = _resolve_origin(r)
    direction = _resolve_direction(r, origin)
    max_range = float(getattr(r, "proj_max_range", 60.0))

    hit, loc, _nor, _hit_obj, _dist = _raycast_any(op, origin, direction, max_range)
    impact = loc if hit else (origin + direction * max_range)

    # --- Visual handling ---
    template = getattr(r, "proj_object", None)
    place = bool(getattr(r, "proj_place_hitscan_object", True))
    align = bool(getattr(r, "proj_align_object_to_velocity", True))

    if template and place:
        life = float(getattr(r, "proj_lifetime", 0.0) or 0.0)
        if life > 0.0:
            # Per-reaction FIFO cap (reuse projectile pool_limit)
            cap = int(getattr(r, "proj_pool_limit", 8) or 8)
            if cap < 1:
                cap = 1
            owner = _owner_key(r)

            # Enforce cap on **visuals present** (stuck/alive)
            present = len(_per_reaction_visual_fifo.get(owner, []))
            if present >= cap:
                _evict_oldest_visuals_for_reaction(owner, present - cap + 1)

            # Clone and place
            new_ob = _spawn_visual_instance(template, owner)
            if new_ob:
                new_ob.location = impact
                if align:
                    _align_object_to_dir(new_ob, direction)
                # Track for timed despawn
                now = get_game_time()
                _active_hitscan_visuals.append({"obj": new_ob, "end_time": now + life})
        else:
            # Lifetime == 0 → legacy behavior: move the template (no clone, no pooling)
            try:
                template.location = impact
                if align:
                    _align_object_to_dir(template, direction)
            except Exception:
                pass

    # --- Impact event + vector push ---
    if hit:
        _push_impact_location_to_links(r, (loc.x, loc.y, loc.z))
        _emit_impact_event(r, loc, get_game_time())
        _flush_impact_events_to_graph()  # immediate for hitscan





def execute_projectile_reaction(r):
    """
    Spawn a simulated projectile advanced by update_projectile_tasks().
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

    # Visual: clone per shot (tracked in FIFO)
    pov = _spawn_visual_instance(template, owner) if template is not None else None
    if pov:
        pov.matrix_world.translation = origin
        if getattr(r, "proj_align_object_to_velocity", True) and vel.length > 1e-6:
            _align_object_to_dir(pov, vel)

    entry = {
        'r_id': r,
        'obj': pov,
        'pos': origin.copy(),
        'vel': vel.copy(),
        'g': g,
        'end_time': now + life,
        'stop_on_contact': bool(getattr(r, "proj_on_contact_stop", True)),
        'align': bool(getattr(r, "proj_align_object_to_velocity", True)),
    }
    _active_projectiles.append(entry)


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