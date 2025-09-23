import bpy
import mathutils
from mathutils import Vector
from .exp_bvh_local import LocalBVH

def update_dynamic_meshes(modal_op):
    """
    Distance-gated dynamic proxies with minimal depsgraph traffic:
      • Distance gate runs BEFORE any evaluated_get().
      • Active movers only: contribute LocalBVH + velocities/ω.
      • Caches: LocalBVH per object and a bounding-sphere radius.
      • Adds small hysteresis around register_distance to avoid flapping.
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # --- Caches / state (create once) ---
    if not hasattr(modal_op, "cached_local_bvhs"):
        modal_op.cached_local_bvhs = {}
    if not hasattr(modal_op, "_dyn_active_state"):
        modal_op._dyn_active_state = {}
    if not hasattr(modal_op, "platform_prev_positions"):
        modal_op.platform_prev_positions = {}
    if not hasattr(modal_op, "platform_prev_matrices"):
        modal_op.platform_prev_matrices = {}
    # NEW: cache a simple world-space bounding-sphere radius per obj
    if not hasattr(modal_op, "_cached_dyn_radius"):
        modal_op._cached_dyn_radius = {}

    # --- Outputs (recreated each tick) ---
    modal_op.dynamic_bvh_map = {}
    modal_op.platform_motion_map = {}
    modal_op.platform_linear_velocity_map = {}
    modal_op.platform_ang_velocity_map = {}
    modal_op.platform_delta_quat_map = {}
    modal_op.platform_delta_map = {}

    # Safe dt for velocity calc — prefer fixed physics dt (30 Hz)
    frame_dt = getattr(modal_op, "physics_dt", None)
    if frame_dt is None or frame_dt <= 0.0:
        frame_dt = getattr(modal_op, "delta_time", 0.0)
    if frame_dt is None or frame_dt <= 1e-8:
        frame_dt = 1e-8

    # Player location (once)
    player_loc = None
    if getattr(modal_op, "target_object", None):
        player_loc = modal_op.target_object.matrix_world.translation

    active_count = 0

    for pm in scene.proxy_meshes:
        dyn_obj = pm.mesh_object
        if not dyn_obj or dyn_obj.type != 'MESH' or not pm.is_moving:
            continue

        # -------- 1) Distance gate BEFORE depsgraph/evaluated_get ----------
        # Quick world matrix is fine for coarse distance
        cur_M_quick = dyn_obj.matrix_world
        cur_pos_quick = cur_M_quick.translation

        dist = None
        active = True
        prev_active = modal_op._dyn_active_state.get(dyn_obj)

        if pm.register_distance > 0.0 and player_loc is not None:
            dist = (cur_pos_quick - player_loc).length
            # Small hysteresis (~10%) to avoid on/off flapping at the edge
            margin = pm.register_distance * 0.10
            if prev_active is True:
                threshold = pm.register_distance + margin
            elif prev_active is False:
                threshold = max(0.0, pm.register_distance - margin)
            else:
                threshold = pm.register_distance
            active = (dist <= threshold)

        # Log & remember state transitions
        if prev_active is None or prev_active != active:
            dist_msg = "n/a" if dist is None else f"{dist:.2f}"
            if active:
                print(f"[DynProxy] ACTIVATED '{dyn_obj.name}' (dist={dist_msg}, thresh={pm.register_distance:.2f})")
            else:
                print(f"[DynProxy] DEACTIVATED '{dyn_obj.name}' (dist={dist_msg}, thresh={pm.register_distance:.2f})")
            modal_op._dyn_active_state[dyn_obj] = active

        if not active:
            # Keep “previous” pose updated cheaply, then skip heavy work
            modal_op.platform_prev_positions[dyn_obj] = cur_pos_quick.copy()
            modal_op.platform_prev_matrices[dyn_obj] = cur_M_quick.copy()
            continue

        # -------- 2) ACTIVE path: only now touch evaluated depsgraph if needed --------
        needs_eval = (
            dyn_obj.modifiers or dyn_obj.constraints or
            (dyn_obj.animation_data and (dyn_obj.animation_data.action or dyn_obj.animation_data.drivers))
        )
        if needs_eval:
            eval_obj = dyn_obj.evaluated_get(depsgraph)
            cur_M = eval_obj.matrix_world.copy()
        else:
            cur_M = cur_M_quick.copy()

        active_count += 1

        # Build or fetch LocalBVH once (rigid motion => no rebuilds)
        lbvh = modal_op.cached_local_bvhs.get(dyn_obj)
        if lbvh is None:
            lbvh = LocalBVH(dyn_obj)
            modal_op.cached_local_bvhs[dyn_obj] = lbvh
            # Compute & cache a simple world-space bounding-sphere radius once
            bbox_world = [dyn_obj.matrix_world @ mathutils.Vector(c) for c in dyn_obj.bound_box]
            center = sum(bbox_world, mathutils.Vector()) / 8.0
            rad = max((p - center).length for p in bbox_world)
            modal_op._cached_dyn_radius[dyn_obj] = rad

        rad = modal_op._cached_dyn_radius.get(dyn_obj, 0.0)

        # Contribute to dynamic collisions / occlusion only when active
        modal_op.dynamic_bvh_map[dyn_obj] = (lbvh, rad)

        # Linear motion / velocity
        prev_pos = modal_op.platform_prev_positions.get(dyn_obj)
        cur_pos  = cur_M.translation.copy()
        if prev_pos is not None:
            disp = cur_pos - prev_pos
            modal_op.platform_motion_map[dyn_obj] = disp
            modal_op.platform_linear_velocity_map[dyn_obj] = disp / frame_dt
        modal_op.platform_prev_positions[dyn_obj] = cur_pos

        # Angular motion / ω (rad/s)
        prev_M = modal_op.platform_prev_matrices.get(dyn_obj, cur_M.copy())
        delta_M = cur_M @ prev_M.inverted()
        R = delta_M.to_3x3(); R.normalize()
        dq = R.to_quaternion()

        modal_op.platform_delta_quat_map[dyn_obj] = dq
        modal_op.platform_delta_map[dyn_obj] = delta_M
        modal_op.platform_prev_matrices[dyn_obj] = cur_M.copy()

        try:
            axis, angle = dq.to_axis_angle()
        except Exception:
            axis, angle = mathutils.Vector((0.0, 0.0, 1.0)), 0.0
        omega = axis * (angle / frame_dt) if angle > 1.0e-9 else mathutils.Vector((0.0, 0.0, 0.0))
        modal_op.platform_ang_velocity_map[dyn_obj] = omega

    # Optional debug count
    last = getattr(modal_op, "_dyn_active_count", None)
    if last is None or last != active_count:
        modal_op._dyn_active_count = active_count
