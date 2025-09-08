# Exp_Game/physics/exp_dynamic.py
import bpy
import mathutils
from mathutils import Vector
from .exp_bvh_local import LocalBVH

def update_dynamic_meshes(modal_op):
    """
    Lightweight dynamic proxy updater:
      • Rigid movers -> LocalBVH (no rebuild on rigid transform)
      • (Optional) deformers -> rebuild when flagged externally
      • Computes per-frame platform linear velocity v and angular velocity ω
        so the KCC can apply physically correct carry (v + ω×r) and yaw rotation.

    Produces/updates on modal_op:
      dynamic_bvh_map:               {obj: (bvh_like, approx_radius)}
      cached_local_bvhs:             {obj: LocalBVH}
      platform_motion_map:           {obj: Vector}                # frame displacement (legacy/back-compat)
      platform_linear_velocity_map:  {obj: Vector}                # m/s
      platform_ang_velocity_map:     {obj: Vector}                # rad/s (world)
      platform_delta_quat_map:       {obj: Quaternion}            # world delta rotation this frame
      platform_prev_positions:       {obj: Vector}
      platform_prev_matrices:        {obj: Matrix}
    """
    scene = bpy.context.scene
    if not hasattr(modal_op, "cached_local_bvhs"):
        modal_op.cached_local_bvhs = {}

    # Output maps
    modal_op.dynamic_bvh_map = {}
    modal_op.platform_motion_map = {}
    modal_op.platform_linear_velocity_map = {}
    modal_op.platform_ang_velocity_map = {}
    modal_op.platform_delta_quat_map = {}
    modal_op.platform_delta_map = {}

    # Guard against zero frame dt
    frame_dt = getattr(modal_op, "delta_time", 0.0)
    if frame_dt is None or frame_dt <= 1e-8:
        frame_dt = 1e-8

    # Cache init
    if not hasattr(modal_op, "platform_prev_positions"):
        modal_op.platform_prev_positions = {}
    if not hasattr(modal_op, "platform_prev_matrices"):
        modal_op.platform_prev_matrices = {}

    # Figure out player location for optional register_distance optimizations
    player_loc = modal_op.target_object.matrix_world.translation if getattr(modal_op, "target_object", None) else None

    # --------- Build / cache LocalBVH for dynamic rigid proxies ----------
    for pm in scene.proxy_meshes:
        dyn_obj = pm.mesh_object
        if not dyn_obj or dyn_obj.type != 'MESH' or not pm.is_moving:
            continue

        actively_update = True
        if pm.register_distance > 0.0 and player_loc is not None:
            if (dyn_obj.matrix_world.translation - player_loc).length > pm.register_distance:
                actively_update = False

        # Cache or (re)build LocalBVH (for DEFORMING you could rebuild here when needed)
        lbvh = modal_op.cached_local_bvhs.get(dyn_obj)
        deforming = getattr(pm, "dynamic_mode", "KINEMATIC_RIGID") == "DEFORMING"

        if lbvh is None or deforming:
            lbvh = LocalBVH(dyn_obj)
            modal_op.cached_local_bvhs[dyn_obj] = lbvh

        # Rough world-space radius for broad camera obstruction checks
        bbox_world = [dyn_obj.matrix_world @ mathutils.Vector(corner) for corner in dyn_obj.bound_box]
        center = sum(bbox_world, mathutils.Vector()) / 8.0
        rad = max((p - center).length for p in bbox_world)

        # Keep it in the map (even if not actively updating, so collisions still exist)
        modal_op.dynamic_bvh_map[dyn_obj] = (lbvh, rad)

    # --------- Per-frame linear & angular velocity for platforms ----------
    for pm in scene.proxy_meshes:
        dyn = pm.mesh_object
        if not dyn or not pm.is_moving:
            continue

        actively_update = True
        if pm.register_distance > 0.0 and player_loc is not None:
            if (dyn.matrix_world.translation - player_loc).length > pm.register_distance:
                actively_update = False

        # Linear displacement & velocity
        cur_pos = dyn.matrix_world.translation.copy()
        prev_pos = modal_op.platform_prev_positions.get(dyn)
        if prev_pos is not None:
            disp = cur_pos - prev_pos
            modal_op.platform_motion_map[dyn] = disp  # legacy
            if actively_update:
                modal_op.platform_linear_velocity_map[dyn] = disp / frame_dt
        modal_op.platform_prev_positions[dyn] = cur_pos

        # Rotation delta & angular velocity (world)
        prev_M = modal_op.platform_prev_matrices.get(dyn, dyn.matrix_world.copy())
        cur_M  = dyn.matrix_world.copy()
        delta_M = cur_M @ prev_M.inverted()

        # Extract pure rotation from delta_M and convert to quaternion
        R = delta_M.to_3x3()
        R.normalize()  # strip scale/skew if any
        dq = R.to_quaternion()
        modal_op.platform_delta_quat_map[dyn] = dq
        modal_op.platform_delta_map[dyn] = delta_M
        modal_op.platform_prev_matrices[dyn] = cur_M

        if actively_update:
            # Axis-angle -> angular velocity in world (rad/s)
            axis = mathutils.Vector((0.0, 0.0, 1.0))
            angle = 0.0
            try:
                axis, angle = dq.to_axis_angle()
            except Exception:
                axis, angle = mathutils.Vector((0.0,0.0,1.0)), 0.0

            if angle > 1e-9:
                # axis is already world-space because delta_M is world-space
                omega = axis * (angle / frame_dt)  # rad/s
            else:
                omega = mathutils.Vector((0.0, 0.0, 0.0))
            modal_op.platform_ang_velocity_map[dyn] = omega
