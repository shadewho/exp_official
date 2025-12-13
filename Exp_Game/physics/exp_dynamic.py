import bpy
import mathutils
import time

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
    if not hasattr(modal_op, "platform_prev_matrices"):
        modal_op.platform_prev_matrices = {}
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

    if not hasattr(modal_op, "platform_motion_map"):
        modal_op.platform_motion_map = {}
    else:
        modal_op.platform_motion_map.clear()

    if not hasattr(modal_op, "platform_delta_quat_map"):
        modal_op.platform_delta_quat_map = {}
    else:
        modal_op.platform_delta_quat_map.clear()

    if not hasattr(modal_op, "platform_delta_map"):
        modal_op.platform_delta_map = {}
    else:
        modal_op.platform_delta_map.clear()

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

                # Compute bounding radius
                bbox_world = [dyn_obj.matrix_world @ mathutils.Vector(c) for c in dyn_obj.bound_box]
                center_world = sum(bbox_world, mathutils.Vector()) / 8.0
                rad_center = max((p - center_world).length for p in bbox_world)
                origin_world = dyn_obj.matrix_world.translation
                center_offset = (center_world - origin_world).length
                radius = rad_center + center_offset
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
    # PHASE 2: Send transforms for ALL dynamic meshes
    # ═══════════════════════════════════════════════════════════════════════
    # Worker caches transforms, so we only need to send when mesh MOVES.
    # For now, send all transforms every frame (simple, correct).
    # Optimization: Track dirty state and only send changes.
    # ═══════════════════════════════════════════════════════════════════════

    mesh_count = 0

    for pm in scene.proxy_meshes:
        dyn_obj = pm.mesh_object
        if not dyn_obj or dyn_obj.type != 'MESH' or not pm.is_moving:
            continue

        mesh_count += 1
        cur_M = dyn_obj.matrix_world

        # Get cached radius (or compute if missing)
        if dyn_obj not in modal_op._cached_dyn_radius:
            bbox_world = [cur_M @ mathutils.Vector(c) for c in dyn_obj.bound_box]
            center_world = sum(bbox_world, mathutils.Vector()) / 8.0
            rad_center = max((p - center_world).length for p in bbox_world)
            origin_world = cur_M.translation
            center_offset = (center_world - origin_world).length
            modal_op._cached_dyn_radius[dyn_obj] = rad_center + center_offset

        rad = modal_op._cached_dyn_radius.get(dyn_obj, 1.0)

        # Add to output map (used by KCC for transform serialization)
        modal_op.dynamic_objects_map[dyn_obj] = rad

        # ─────────────────────────────────────────────────────────────────
        # Compute velocities for platform carry (moving platforms)
        # ─────────────────────────────────────────────────────────────────
        cur_pos = cur_M.translation.copy()
        prev_pos = modal_op.platform_prev_positions.get(dyn_obj)

        if prev_pos is not None:
            disp = cur_pos - prev_pos
            modal_op.platform_motion_map[dyn_obj] = disp
            modal_op.platform_linear_velocity_map[dyn_obj] = disp / frame_dt

        modal_op.platform_prev_positions[dyn_obj] = cur_pos

        # Angular velocity
        prev_M = modal_op.platform_prev_matrices.get(dyn_obj, cur_M.copy())
        delta_M = cur_M @ prev_M.inverted()
        R = delta_M.to_3x3()
        R.normalize()
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

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: Summary logging
    # ═══════════════════════════════════════════════════════════════════════
    if debug_dyn_cache and mesh_count > 0:
        from ..developer.dev_logger import log_game
        total_tris = sum(len(pm.mesh_object.data.loop_triangles)
                        for pm in scene.proxy_meshes
                        if pm.mesh_object and pm.mesh_object.type == 'MESH' and pm.is_moving)
        log_game("DYN-CACHE", f"SEND: {mesh_count} meshes ({total_tris} tris) → worker")
