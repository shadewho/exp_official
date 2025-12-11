import bpy
import mathutils
import time
from .exp_bvh_local import LocalBVH

# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED PHYSICS NOTE:
# - KCC character physics: Handled entirely in worker (no main thread BVH)
# - Camera/Projectiles/Tracking: Still use LocalBVH on main thread (fast for LoS)
# LocalBVH is kept for non-KCC systems that need quick raycasts
# ═══════════════════════════════════════════════════════════════════════════


def apply_dynamic_activation_result(modal_op, engine_result):
    """
    Apply worker result for dynamic mesh activation decisions.
    Called from game loop when DYNAMIC_MESH_ACTIVATION job completes.

    Args:
        modal_op: The modal operator
        engine_result: EngineResult object with result.result containing activation_decisions
    """
    scene = bpy.context.scene

    result_data = engine_result.result
    activation_decisions = result_data.get("activation_decisions", [])
    if not activation_decisions:
        return

    # Apply activation states
    transitions = []
    for obj_name, should_activate, prev_active in activation_decisions:
        # Find the object by name
        dyn_obj = scene.objects.get(obj_name)
        if dyn_obj is None:
            continue

        # Update activation state
        modal_op._dyn_active_state[dyn_obj] = should_activate

        # Track transitions for debug output
        if prev_active != should_activate:
            transitions.append((obj_name, prev_active, should_activate))


def update_dynamic_meshes(modal_op):
    """
    Distance-gated dynamic proxies with minimal main thread work:
      • Distance gate runs BEFORE any evaluated_get().
      • Active movers only: contribute transforms + velocities.
      • Caches: LocalBVH per object (for camera/projectiles), bounding-sphere radius.
      • Adds small hysteresis around register_distance to avoid flapping.
      • Reuses dicts (no per-tick reallocation).

    UNIFIED PHYSICS:
      • KCC character physics: Fully offloaded to worker (uses triangles, not BVH)
      • Camera/Projectiles/Tracking: Use LocalBVH on main thread (fast for LoS)
    """
    scene = bpy.context.scene

    # --- Caches / state (create once) ---
    if not hasattr(modal_op, "cached_local_bvhs"):
        modal_op.cached_local_bvhs = {}  # For camera/projectiles (NOT KCC)
    if not hasattr(modal_op, "_dyn_active_state"):
        modal_op._dyn_active_state = {}
    if not hasattr(modal_op, "platform_prev_positions"):
        modal_op.platform_prev_positions = {}
    if not hasattr(modal_op, "platform_prev_matrices"):
        modal_op.platform_prev_matrices = {}
    if not hasattr(modal_op, "_cached_dyn_radius"):
        modal_op._cached_dyn_radius = {}

    # --- Outputs (reuse dicts to avoid churn) ---
    # dynamic_bvh_map: Used by camera/projectiles/tracking (LocalBVH)
    if not hasattr(modal_op, "dynamic_bvh_map"):
        modal_op.dynamic_bvh_map = {}
    else:
        modal_op.dynamic_bvh_map.clear()

    # dynamic_objects_map: Simpler map for KCC platform carry lookup
    if not hasattr(modal_op, "dynamic_objects_map"):
        modal_op.dynamic_objects_map = {}
    else:
        modal_op.dynamic_objects_map.clear()

    if not hasattr(modal_op, "platform_motion_map"):
        modal_op.platform_motion_map = {}
    else:
        modal_op.platform_motion_map.clear()

    if not hasattr(modal_op, "platform_linear_velocity_map"):
        modal_op.platform_linear_velocity_map = {}
    else:
        modal_op.platform_linear_velocity_map.clear()

    if not hasattr(modal_op, "platform_ang_velocity_map"):
        modal_op.platform_ang_velocity_map = {}
    else:
        modal_op.platform_ang_velocity_map.clear()

    if not hasattr(modal_op, "platform_delta_quat_map"):
        modal_op.platform_delta_quat_map = {}
    else:
        modal_op.platform_delta_quat_map.clear()

    if not hasattr(modal_op, "platform_delta_map"):
        modal_op.platform_delta_map = {}
    else:
        modal_op.platform_delta_map.clear()

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

    # ========== WORKER OFFLOAD: Dynamic Mesh Proximity Checks ==========
    # Submit worker job for distance calculations (1-frame latency acceptable)
    # Worker results will update activation states for NEXT frame
    engine = getattr(modal_op, "engine", None)
    use_workers = engine is not None and engine.is_alive()

    if use_workers and player_loc is not None:
        # Snapshot mesh data for worker
        mesh_positions = []
        mesh_objects = []  # (obj_name, prev_active)
        base_distances = []

        for pm in scene.proxy_meshes:
            dyn_obj = pm.mesh_object
            if not dyn_obj or dyn_obj.type != 'MESH' or not pm.is_moving:
                continue

            if pm.register_distance > 0.0:
                cur_pos = dyn_obj.matrix_world.translation
                mesh_positions.append((cur_pos.x, cur_pos.y, cur_pos.z))
                prev_active = modal_op._dyn_active_state.get(dyn_obj)
                mesh_objects.append((dyn_obj.name, prev_active if prev_active is not None else True))
                base_distances.append(float(pm.register_distance))
            else:
                # No distance gating, always active
                mesh_positions.append((0, 0, 0))  # Dummy position
                mesh_objects.append((dyn_obj.name, True))
                base_distances.append(0.0)  # Zero distance = always active

        # Submit job to workers if we have meshes to check
        if mesh_positions:
            job_data = {
                "mesh_positions": mesh_positions,
                "mesh_objects": mesh_objects,
                "player_position": (player_loc.x, player_loc.y, player_loc.z),
                "base_distances": base_distances,
            }

            job_id = engine.submit_job("DYNAMIC_MESH_ACTIVATION", job_data)

    # ========== Main Thread: BVH & Velocity Calculations ==========
    # Use current activation states (updated by worker results from previous frame)
    # This has 1-frame latency, which is acceptable for performance gating
    # CRITICAL: Do NOT return early - BVH data is needed for collision!
    for pm in scene.proxy_meshes:
        dyn_obj = pm.mesh_object
        if not dyn_obj or dyn_obj.type != 'MESH' or not pm.is_moving:
            continue

        # -------- 1) Check activation state (updated by worker or fallback) ----------
        cur_M_quick = dyn_obj.matrix_world
        cur_pos_quick = cur_M_quick.translation

        # Get activation state (set by worker results or defaults to True)
        active = modal_op._dyn_active_state.get(dyn_obj, True)

        if not active:
            # Keep “previous” pose updated cheaply, then skip heavy work
            modal_op.platform_prev_positions[dyn_obj] = cur_pos_quick.copy()
            modal_op.platform_prev_matrices[dyn_obj] = cur_M_quick.copy()
            continue

        # -------- 2) ACTIVE path ----------
        cur_M = cur_M_quick.copy()

        # ═══════════════════════════════════════════════════════════════════
        # Build LocalBVH for camera/projectiles (NOT used by KCC - KCC uses worker)
        # ═══════════════════════════════════════════════════════════════════
        lbvh = modal_op.cached_local_bvhs.get(dyn_obj)
        if lbvh is None:
            lbvh = LocalBVH(dyn_obj)
            modal_op.cached_local_bvhs[dyn_obj] = lbvh

            # ═══════════════════════════════════════════════════════════════
            # UNIFIED PHYSICS: Cache triangles in worker (one-time per mesh)
            # Worker uses these for KCC collision (faster than main thread BVH)
            # ═══════════════════════════════════════════════════════════════
            if use_workers and engine:
                obj_id = id(dyn_obj)
                if not hasattr(modal_op, '_cached_dynamic_mesh_ids'):
                    modal_op._cached_dynamic_mesh_ids = set()

                if obj_id not in modal_op._cached_dynamic_mesh_ids:
                    # Extract triangles in LOCAL space (sent once to worker)
                    mesh = dyn_obj.data
                    mesh.calc_loop_triangles()

                    triangles = []
                    for tri in mesh.loop_triangles:
                        v0 = tuple(mesh.vertices[tri.vertices[0]].co)
                        v1 = tuple(mesh.vertices[tri.vertices[1]].co)
                        v2 = tuple(mesh.vertices[tri.vertices[2]].co)
                        triangles.append((v0, v1, v2))

                    # Compute radius for bounding sphere
                    if dyn_obj not in modal_op._cached_dyn_radius:
                        bbox_world = [dyn_obj.matrix_world @ mathutils.Vector(c) for c in dyn_obj.bound_box]
                        center_world = sum(bbox_world, mathutils.Vector()) / 8.0
                        rad_center = max((p - center_world).length for p in bbox_world)
                        origin_world = dyn_obj.matrix_world.translation
                        center_offset = (center_world - origin_world).length
                        modal_op._cached_dyn_radius[dyn_obj] = rad_center + center_offset

                    radius = modal_op._cached_dyn_radius.get(dyn_obj, 1.0)

                    # Send triangles to worker (one-time cache)
                    job_data = {
                        "obj_id": obj_id,
                        "triangles": triangles,
                        "radius": radius,
                    }
                    engine.submit_job("CACHE_DYNAMIC_MESH", job_data)
                    modal_op._cached_dynamic_mesh_ids.add(obj_id)

        # Update LocalBVH transform for camera/projectiles
        lbvh.update_xform(cur_M)

        # Compute & cache radius if not already done
        if dyn_obj not in modal_op._cached_dyn_radius:
            bbox_world = [dyn_obj.matrix_world @ mathutils.Vector(c) for c in dyn_obj.bound_box]
            center_world = sum(bbox_world, mathutils.Vector()) / 8.0
            rad_center = max((p - center_world).length for p in bbox_world)
            origin_world = dyn_obj.matrix_world.translation
            center_offset = (center_world - origin_world).length
            modal_op._cached_dyn_radius[dyn_obj] = rad_center + center_offset

        rad = modal_op._cached_dyn_radius.get(dyn_obj, 0.0)

        # Store in both maps:
        # - dynamic_bvh_map: For camera/projectiles/tracking (with LocalBVH)
        # - dynamic_objects_map: For KCC platform carry lookup (just radius)
        modal_op.dynamic_bvh_map[dyn_obj] = (lbvh, rad)
        modal_op.dynamic_objects_map[dyn_obj] = rad

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
    