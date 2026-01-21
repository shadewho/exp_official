import bpy
import mathutils

# Generation counter - increment to invalidate all cached transforms
# (call reset_dyn_transform_generation() when workers reset)
_dyn_transform_generation = 0


def reset_dyn_transform_generation():
    """Call this when workers are reset to force re-sending all transforms."""
    global _dyn_transform_generation
    _dyn_transform_generation += 1


def _matrix_close(m1, m2, tolerance=1e-5):
    """Fast matrix comparison - returns True if matrices are nearly identical."""
    for i in range(4):
        for j in range(4):
            if abs(m1[i][j] - m2[i][j]) > tolerance:
                return False
    return True

# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED DYNAMIC MESH SYSTEM (2025-12-13)
# ═══════════════════════════════════════════════════════════════════════════
#
# ARCHITECTURE:
#   Main Thread (THIN):
#     - Cache mesh triangles to worker (one-time per mesh)
#     - Send transforms for meshes that moved (dirty check)
#     - Compute velocities for platform carry
#     - NO activation logic, NO AABB checks
#
#   Worker (SMART):
#     - Maintains persistent transform cache for ALL dynamic meshes
#     - Uses cached transforms when no update received (stationary meshes = zero cost)
#     - Same unified_raycast for static + dynamic (one code path)
#     - Worker decides what to test based on spatial queries
#
# BENEFITS:
#   - Stationary dynamic meshes: ZERO main thread cost
#   - Moving meshes: One matrix comparison + one send
#   - No chicken-and-egg timing issues
#   - Dynamic meshes behave like static: always available for physics
#
# FUTURE (Phase 2+):
#   - Worker-side spatial hash for O(1) broad-phase
#   - Mesh-to-mesh collision detection
#   - Rigid body physics
# ═══════════════════════════════════════════════════════════════════════════


def _compute_bounding_radius(dyn_obj, matrix_world):
    """
    Compute bounding sphere radius for a dynamic mesh.
    Accounts for object origin not being at mesh center.

    Args:
        dyn_obj: Blender mesh object
        matrix_world: Object's world matrix

    Returns:
        float: Bounding sphere radius
    """
    bbox_world = [matrix_world @ mathutils.Vector(c) for c in dyn_obj.bound_box]
    center_world = sum(bbox_world, mathutils.Vector()) / 8.0
    rad_center = max((p - center_world).length for p in bbox_world)
    origin_world = matrix_world.translation
    center_offset = (center_world - origin_world).length
    return rad_center + center_offset


def update_dynamic_meshes(modal_op):
    """
    Thin main-thread function for dynamic mesh management.

    Responsibilities:
    1. Cache mesh triangles to worker (one-time)
    2. Send transform updates for meshes that moved
    3. Compute velocities for platform carry (moving platforms)

    NO activation logic - worker handles all spatial decisions.
    """
    scene = bpy.context.scene

    # --- Initialize caches (once) ---
    if not hasattr(modal_op, "platform_prev_positions"):
        modal_op.platform_prev_positions = {}
    # NOTE: platform_prev_matrices replaced with platform_prev_quaternions (4 floats vs 16)
    if not hasattr(modal_op, "_cached_dyn_radius"):
        modal_op._cached_dyn_radius = {}
    if not hasattr(modal_op, "_cached_dynamic_mesh_ids"):
        modal_op._cached_dynamic_mesh_ids = set()

    # --- Output maps (cleared each frame) ---
    if not hasattr(modal_op, "dynamic_objects_map"):
        modal_op.dynamic_objects_map = {}
    else:
        modal_op.dynamic_objects_map.clear()

    if not hasattr(modal_op, "platform_linear_velocity_map"):
        modal_op.platform_linear_velocity_map = {}
    else:
        modal_op.platform_linear_velocity_map.clear()

    if not hasattr(modal_op, "platform_ang_velocity_map"):
        modal_op.platform_ang_velocity_map = {}
    else:
        modal_op.platform_ang_velocity_map.clear()

    # NOTE: platform_delta_quat_map, platform_delta_map, and platform_motion_map were REMOVED
    # They were computed but never read (dead code)
    # Only platform_ang_velocity_map is actually used downstream

    if not hasattr(modal_op, "platform_prev_quaternions"):
        modal_op.platform_prev_quaternions = {}

    # Safe dt for velocity calculation
    frame_dt = getattr(modal_op, "physics_dt", None)
    if frame_dt is None or frame_dt <= 0.0:
        frame_dt = getattr(modal_op, "delta_time", 0.0)
    if frame_dt is None or frame_dt <= 1e-8:
        frame_dt = 1e-8

    # Engine reference
    engine = getattr(modal_op, "engine", None)
    use_workers = engine is not None and engine.is_alive()

    # Debug flags
    debug_dyn_cache = getattr(scene, 'dev_debug_dynamic_cache', False)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: Cache mesh triangles to worker (ONE-TIME per mesh)
    # ═══════════════════════════════════════════════════════════════════════
    if use_workers and engine:
        for pm in scene.proxy_meshes:
            dyn_obj = pm.mesh_object
            if not dyn_obj or dyn_obj.type != 'MESH' or not pm.is_moving:
                continue

            obj_id = id(dyn_obj)
            if obj_id not in modal_op._cached_dynamic_mesh_ids:
                # Extract triangles in LOCAL space (sent once)
                mesh = dyn_obj.data
                mesh.calc_loop_triangles()

                triangles = []
                for tri in mesh.loop_triangles:
                    v0 = tuple(mesh.vertices[tri.vertices[0]].co)
                    v1 = tuple(mesh.vertices[tri.vertices[1]].co)
                    v2 = tuple(mesh.vertices[tri.vertices[2]].co)
                    triangles.append((v0, v1, v2))

                # Compute bounding radius (uses helper to avoid duplicate code)
                radius = _compute_bounding_radius(dyn_obj, dyn_obj.matrix_world)
                modal_op._cached_dyn_radius[dyn_obj] = radius

                # Broadcast to all workers with per-worker targeting (guaranteed delivery)
                job_data = {
                    "obj_id": obj_id,
                    "triangles": triangles,
                    "radius": radius,
                }
                engine.broadcast_job("CACHE_DYNAMIC_MESH", job_data)
                modal_op._cached_dynamic_mesh_ids.add(obj_id)

                if debug_dyn_cache:
                    from ..developer.dev_logger import log_game
                    log_game("DYN-CACHE", f"CACHED {dyn_obj.name}: {len(triangles)} tris, radius={radius:.2f}m")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: Send transforms for CHANGED dynamic meshes only (dirty checking)
    # ═══════════════════════════════════════════════════════════════════════
    # Worker caches transforms, so stationary meshes have zero main thread cost.
    # We use generation counter to detect worker resets - if generation changed,
    # invalidate all cached matrices and force re-send.
    # ═══════════════════════════════════════════════════════════════════════

    # Initialize dirty checking state
    if not hasattr(modal_op, "_prev_dyn_matrices"):
        modal_op._prev_dyn_matrices = {}
        modal_op._prev_dyn_generation = _dyn_transform_generation

    # Check if generation changed (worker was reset)
    if modal_op._prev_dyn_generation != _dyn_transform_generation:
        modal_op._prev_dyn_matrices.clear()
        modal_op._prev_dyn_generation = _dyn_transform_generation

    mesh_count = 0
    moved_count = 0

    # Zero velocity vector for stationary meshes
    zero_velocity = mathutils.Vector((0.0, 0.0, 0.0))

    for pm in scene.proxy_meshes:
        dyn_obj = pm.mesh_object
        if not dyn_obj or dyn_obj.type != 'MESH' or not pm.is_moving:
            continue

        mesh_count += 1
        obj_id = id(dyn_obj)
        cur_M = dyn_obj.matrix_world

        # Get cached radius (or compute if missing - fallback only)
        if dyn_obj not in modal_op._cached_dyn_radius:
            modal_op._cached_dyn_radius[dyn_obj] = _compute_bounding_radius(dyn_obj, cur_M)

        rad = modal_op._cached_dyn_radius.get(dyn_obj, 1.0)

        # Add to output map (used by KCC for transform serialization)
        modal_op.dynamic_objects_map[dyn_obj] = rad

        # ─────────────────────────────────────────────────────────────────
        # Dirty checking: skip expensive velocity computation for stationary meshes
        # IMPORTANT: Always update caches to avoid velocity spikes when movement resumes
        # ─────────────────────────────────────────────────────────────────
        prev_M = modal_op._prev_dyn_matrices.get(obj_id)
        is_stationary = prev_M is not None and _matrix_close(cur_M, prev_M)

        # Always update matrix cache to avoid drift accumulation
        modal_op._prev_dyn_matrices[obj_id] = cur_M.copy()

        # Get current position (cheap - just a vector copy)
        cur_pos = cur_M.translation.copy()

        if is_stationary:
            # Stationary - set zero velocities, skip expensive quaternion math
            modal_op.platform_linear_velocity_map[dyn_obj] = zero_velocity
            modal_op.platform_ang_velocity_map[dyn_obj] = zero_velocity
            # Still update position cache to avoid velocity spikes when movement resumes
            modal_op.platform_prev_positions[dyn_obj] = cur_pos
            # Note: We skip updating prev_quaternions here since it requires to_quaternion()
            # This is safe because angular velocity will be recalculated when movement resumes
            continue

        moved_count += 1

        # ─────────────────────────────────────────────────────────────────
        # Compute velocities for platform carry (moving platforms)
        # ─────────────────────────────────────────────────────────────────
        prev_pos = modal_op.platform_prev_positions.get(dyn_obj)

        if prev_pos is not None:
            disp = cur_pos - prev_pos
            modal_op.platform_linear_velocity_map[dyn_obj] = disp / frame_dt

        modal_op.platform_prev_positions[dyn_obj] = cur_pos

        # Angular velocity - OPTIMIZED: use quaternions instead of matrix inversion
        # Quaternion inversion is O(1) vs matrix inversion O(n³)
        # Also stores 4 floats instead of 16
        cur_quat = cur_M.to_quaternion()
        prev_quat = modal_op.platform_prev_quaternions.get(dyn_obj)

        if prev_quat is None:
            # First frame for this mesh - no angular velocity yet
            modal_op.platform_ang_velocity_map[dyn_obj] = zero_velocity
        else:
            # Compute delta rotation: delta = cur @ prev.inverted()
            # Quaternion inversion is just conjugate for unit quaternions - O(1)!
            delta_quat = cur_quat @ prev_quat.inverted()

            try:
                axis, angle = delta_quat.to_axis_angle()
            except Exception as e:
                if debug_dyn_cache:
                    from ..developer.dev_logger import log_game
                    log_game("DYN-CACHE", f"WARN: axis_angle failed: {type(e).__name__}")
                axis, angle = mathutils.Vector((0.0, 0.0, 1.0)), 0.0

            omega = axis * (angle / frame_dt) if angle > 1.0e-9 else zero_velocity
            modal_op.platform_ang_velocity_map[dyn_obj] = omega

        # Store current quaternion for next frame (4 floats vs 16 for matrix)
        modal_op.platform_prev_quaternions[dyn_obj] = cur_quat

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: Summary logging
    # ═══════════════════════════════════════════════════════════════════════
    if debug_dyn_cache and mesh_count > 0:
        from ..developer.dev_logger import log_game
        log_game("DYN-CACHE", f"TRANSFORMS: total={mesh_count} moved={moved_count} stationary={mesh_count - moved_count}")


def cleanup_dynamic_mesh_state(modal_op):
    """
    Clean up dynamic mesh state on game end.
    Clears all per-operator caches to prevent stale data on restart.
    """
    # Clear velocity maps
    if hasattr(modal_op, "platform_prev_positions"):
        modal_op.platform_prev_positions.clear()
    if hasattr(modal_op, "platform_linear_velocity_map"):
        modal_op.platform_linear_velocity_map.clear()
    if hasattr(modal_op, "platform_ang_velocity_map"):
        modal_op.platform_ang_velocity_map.clear()
    if hasattr(modal_op, "platform_prev_quaternions"):
        modal_op.platform_prev_quaternions.clear()

    # Clear dirty checking state
    if hasattr(modal_op, "_prev_dyn_matrices"):
        modal_op._prev_dyn_matrices.clear()

    # Clear mesh tracking
    if hasattr(modal_op, "_cached_dyn_radius"):
        modal_op._cached_dyn_radius.clear()
    if hasattr(modal_op, "_cached_dynamic_mesh_ids"):
        modal_op._cached_dynamic_mesh_ids.clear()
    if hasattr(modal_op, "dynamic_objects_map"):
        modal_op.dynamic_objects_map.clear()