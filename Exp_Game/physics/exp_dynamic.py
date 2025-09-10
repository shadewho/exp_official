# Exp_Game/physics/exp_dynamic.py
import bpy
import mathutils
from mathutils import Vector
from .exp_bvh_local import LocalBVH

def update_dynamic_meshes(modal_op):
    """
    Distance-gated dynamic proxies:
      • Only 'activate' a dynamic proxy when player is within pm.register_distance.
      • Active proxies contribute to dynamic_bvh_map and platform velocities.
      • Activation/deactivation prints fire on state change (not every frame).
      • Summary print shows active-count when it changes.
    """
    scene = bpy.context.scene

    # Caches / state
    if not hasattr(modal_op, "cached_local_bvhs"):
        modal_op.cached_local_bvhs = {}
    if not hasattr(modal_op, "_dyn_active_state"):
        modal_op._dyn_active_state = {}   # {obj: bool}
    if not hasattr(modal_op, "platform_prev_positions"):
        modal_op.platform_prev_positions = {}
    if not hasattr(modal_op, "platform_prev_matrices"):
        modal_op.platform_prev_matrices = {}

    # Outputs (recreated each tick)
    modal_op.dynamic_bvh_map = {}
    modal_op.platform_motion_map = {}
    modal_op.platform_linear_velocity_map = {}
    modal_op.platform_ang_velocity_map = {}
    modal_op.platform_delta_quat_map = {}
    modal_op.platform_delta_map = {}

    # Safe dt for velocity calc — prefer the fixed physics dt (30 Hz)
    frame_dt = getattr(modal_op, "physics_dt", None)
    if frame_dt is None or frame_dt <= 0.0:
        frame_dt = getattr(modal_op, "delta_time", 0.0)
    if frame_dt is None or frame_dt <= 1e-8:
        frame_dt = 1e-8

    # Player location
    player_loc = None
    if getattr(modal_op, "target_object", None):
        player_loc = modal_op.target_object.matrix_world.translation

    active_count = 0

    for pm in scene.proxy_meshes:
        dyn_obj = pm.mesh_object
        if not dyn_obj or dyn_obj.type != 'MESH' or not pm.is_moving:
            continue

        # --- Distance gate ---
        dist = None
        active = True
        if pm.register_distance > 0.0 and player_loc is not None:
            dist = (dyn_obj.matrix_world.translation - player_loc).length
            active = (dist <= pm.register_distance)

        # Print on activation state change (not every frame)
        prev_active = modal_op._dyn_active_state.get(dyn_obj)
        if prev_active is None or prev_active != active:
            dist_msg = "n/a" if dist is None else f"{dist:.2f}"
            if active:
                print(f"[DynProxy] ACTIVATED '{dyn_obj.name}' (dist={dist_msg}, thresh={pm.register_distance:.2f})")
            else:
                print(f"[DynProxy] DEACTIVATED '{dyn_obj.name}' (dist={dist_msg}, thresh={pm.register_distance:.2f})")
            modal_op._dyn_active_state[dyn_obj] = active

        if not active:
            # Keep continuity for when it becomes active again
            modal_op.platform_prev_positions[dyn_obj] = dyn_obj.matrix_world.translation.copy()
            modal_op.platform_prev_matrices[dyn_obj] = dyn_obj.matrix_world.copy()
            continue

        # --- ACTIVE PATH ---
        active_count += 1

        # Build or fetch LocalBVH once (rigid motion doesn't require rebuilds)
        lbvh = modal_op.cached_local_bvhs.get(dyn_obj)
        if lbvh is None:
            lbvh = LocalBVH(dyn_obj)
            modal_op.cached_local_bvhs[dyn_obj] = lbvh

        # Rough radius for camera/aux checks
        bbox_world = [dyn_obj.matrix_world @ mathutils.Vector(corner) for corner in dyn_obj.bound_box]
        center = sum(bbox_world, mathutils.Vector()) / 8.0
        rad = max((p - center).length for p in bbox_world)

        # Contribute to collisions only when active
        modal_op.dynamic_bvh_map[dyn_obj] = (lbvh, rad)

        # Linear motion / velocity (only when active)
        cur_pos = dyn_obj.matrix_world.translation.copy()
        prev_pos = modal_op.platform_prev_positions.get(dyn_obj)
        if prev_pos is not None:
            disp = cur_pos - prev_pos
            modal_op.platform_motion_map[dyn_obj] = disp
            modal_op.platform_linear_velocity_map[dyn_obj] = disp / frame_dt
        modal_op.platform_prev_positions[dyn_obj] = cur_pos

        # Angular motion / ω (rad/s)
        prev_M = modal_op.platform_prev_matrices.get(dyn_obj, dyn_obj.matrix_world.copy())
        cur_M  = dyn_obj.matrix_world.copy()
        delta_M = cur_M @ prev_M.inverted()

        R = delta_M.to_3x3()
        R.normalize()
        dq = R.to_quaternion()
        modal_op.platform_delta_quat_map[dyn_obj] = dq
        modal_op.platform_delta_map[dyn_obj] = delta_M
        modal_op.platform_prev_matrices[dyn_obj] = cur_M

        try:
            axis, angle = dq.to_axis_angle()
        except Exception:
            axis, angle = mathutils.Vector((0.0,0.0,1.0)), 0.0
        omega = axis * (angle / frame_dt) if angle > 1e-9 else mathutils.Vector((0.0, 0.0, 0.0))
        modal_op.platform_ang_velocity_map[dyn_obj] = omega

    # One small summary print when the count changes
    last = getattr(modal_op, "_dyn_active_count", None)
    if last is None or last != active_count:
        print(f"[DynProxy] Active dynamic proxies this frame: {active_count}")
        modal_op._dyn_active_count = active_count
