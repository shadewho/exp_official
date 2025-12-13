import bpy
import mathutils
import time

# ═══════════════════════════════════════════════════════════════════════════
# FULLY OFFLOADED PHYSICS (2025-12-11):
# - ALL raycasting offloaded to worker (KCC, Camera, Projectiles, Tracking)
# - Worker caches static grid + dynamic mesh triangles
# - Main thread only sends transforms for active meshes
# - No LocalBVH on main thread - worker handles everything
#
# AABB-BASED ACTIVATION:
# - Main thread checks player vs world AABB each frame (zero latency)
# - 2m margin around AABB for ~6 frames buffer at max player speed
# ═══════════════════════════════════════════════════════════════════════════

AABB_ACTIVATION_MARGIN = 2.0  # meters around AABB for activation buffer


def compute_world_aabb(dyn_obj):
    """
    Compute world-space AABB from Blender's bound_box.
    Returns (min_pt, max_pt) as tuples.

    Fast: Uses Blender's pre-computed bound_box, just transforms 8 corners.
    """
    M = dyn_obj.matrix_world
    corners = [M @ mathutils.Vector(c) for c in dyn_obj.bound_box]

    min_x = min(c.x for c in corners)
    min_y = min(c.y for c in corners)
    min_z = min(c.z for c in corners)
    max_x = max(c.x for c in corners)
    max_y = max(c.y for c in corners)
    max_z = max(c.z for c in corners)

    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def player_near_aabb(player_pos, aabb_min, aabb_max, margin=AABB_ACTIVATION_MARGIN):
    """
    Check if player is within AABB + margin. Zero-latency activation check.

    Args:
        player_pos: mathutils.Vector or tuple (x, y, z)
        aabb_min: (min_x, min_y, min_z)
        aabb_max: (max_x, max_y, max_z)
        margin: Extra buffer around AABB (default 2m = ~6 frames at max speed)

    Returns:
        bool: True if player is within expanded AABB
    """
    px, py, pz = player_pos.x, player_pos.y, player_pos.z
    return (aabb_min[0] - margin <= px <= aabb_max[0] + margin and
            aabb_min[1] - margin <= py <= aabb_max[1] + margin and
            aabb_min[2] - margin <= pz <= aabb_max[2] + margin)


def update_dynamic_meshes(modal_op):
    """
    AABB-gated dynamic proxies with zero-latency activation:
      • Main thread AABB check: player vs mesh bounds (6 comparisons, ~10µs/100 meshes)
      • Zero latency: activation happens same frame (no 1-frame worker delay)
      • Active movers only: contribute transforms + velocities to worker
      • Worker caches triangles, main thread only sends transforms

    FULLY OFFLOADED:
      • ALL raycasting in worker (KCC, Camera, Projectiles, Tracking)
      • No LocalBVH on main thread - worker handles everything
    """
    scene = bpy.context.scene

    # --- Caches / state (create once) ---
    if not hasattr(modal_op, "_dyn_active_state"):
        modal_op._dyn_active_state = {}
    if not hasattr(modal_op, "platform_prev_positions"):
        modal_op.platform_prev_positions = {}
    if not hasattr(modal_op, "platform_prev_matrices"):
        modal_op.platform_prev_matrices = {}
    if not hasattr(modal_op, "_cached_dyn_radius"):
        modal_op._cached_dyn_radius = {}

    # --- Outputs (reuse dicts to avoid churn) ---
    # dynamic_objects_map: For camera submission and platform carry lookup
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

    # Engine reference for worker jobs
    engine = getattr(modal_op, "engine", None)
    use_workers = engine is not None and engine.is_alive()

    # Debug logging setup
    debug_dyn_cache = getattr(scene, 'dev_debug_dynamic_cache', False)
    debug_activation = getattr(scene, 'dev_debug_dynamic_activation', False)

    # Get current ground object from KCC (if player is standing on a dynamic mesh)
    # This mesh MUST stay active regardless of AABB to prevent bouncing
    kcc = getattr(modal_op, 'physics_controller', None)
    standing_on_mesh = getattr(kcc, 'ground_obj', None) if kcc else None

    # Track activation counts for summary logging
    total_dynamic_meshes = 0
    active_count = 0
    inactive_count = 0
    standing_override_count = 0

    # ========== PHASE 1: Cache ALL dynamic meshes to worker (one-time) ==========
    # This must happen REGARDLESS of activation state so meshes are ready when needed
    if use_workers and engine:
        if not hasattr(modal_op, '_cached_dynamic_mesh_ids'):
            modal_op._cached_dynamic_mesh_ids = set()

        for pm in scene.proxy_meshes:
            dyn_obj = pm.mesh_object
            if not dyn_obj or dyn_obj.type != 'MESH' or not pm.is_moving:
                continue

            obj_id = id(dyn_obj)
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
                bbox_world = [dyn_obj.matrix_world @ mathutils.Vector(c) for c in dyn_obj.bound_box]
                center_world = sum(bbox_world, mathutils.Vector()) / 8.0
                rad_center = max((p - center_world).length for p in bbox_world)
                origin_world = dyn_obj.matrix_world.translation
                center_offset = (center_world - origin_world).length
                radius = rad_center + center_offset
                modal_op._cached_dyn_radius[dyn_obj] = radius

                # Send triangles to ALL workers (broadcast ensures every worker has the cache)
                job_data = {
                    "obj_id": obj_id,
                    "triangles": triangles,
                    "radius": radius,
                }
                engine.broadcast_job("CACHE_DYNAMIC_MESH", job_data)
                modal_op._cached_dynamic_mesh_ids.add(obj_id)

    # ========== PHASE 2: AABB Activation + BVH + Velocity ==========
    # AABB check: Zero-latency activation (no 1-frame worker delay)
    for pm in scene.proxy_meshes:
        dyn_obj = pm.mesh_object
        if not dyn_obj or dyn_obj.type != 'MESH' or not pm.is_moving:
            continue

        total_dynamic_meshes += 1

        # -------- 1) AABB-based activation (zero latency) ----------
        cur_M_quick = dyn_obj.matrix_world
        cur_pos_quick = cur_M_quick.translation

        # Always compute AABB for logging purposes
        aabb_min, aabb_max = compute_world_aabb(dyn_obj)
        prev_active = modal_op._dyn_active_state.get(dyn_obj)
        activation_reason = ""

        # CRITICAL: If player is STANDING on this mesh, ALWAYS keep it active
        # This prevents bouncing when moving platforms shift the AABB away from player
        if standing_on_mesh is not None and standing_on_mesh == dyn_obj:
            active = True
            standing_override_count += 1
            activation_reason = "STANDING_ON_OVERRIDE"
        elif player_loc is not None:
            # Check player proximity to AABB with margin
            active = player_near_aabb(player_loc, aabb_min, aabb_max)
            activation_reason = "AABB_NEAR" if active else "AABB_FAR"
        else:
            # No player = always active (fallback)
            active = True
            activation_reason = "NO_PLAYER_FALLBACK"

        # Count active/inactive
        if active:
            active_count += 1
        else:
            inactive_count += 1

        # ═══════════════════════════════════════════════════════════════════
        # COMPREHENSIVE ACTIVATION LOGGING
        # ═══════════════════════════════════════════════════════════════════
        if debug_activation and player_loc is not None:
            from ..developer.dev_logger import log_game

            # Calculate distances to each AABB face (negative = inside, positive = outside)
            px, py, pz = player_loc.x, player_loc.y, player_loc.z
            margin = AABB_ACTIVATION_MARGIN

            # Distance to each face (with margin applied)
            dist_x_min = px - (aabb_min[0] - margin)  # + = inside margin, - = outside
            dist_x_max = (aabb_max[0] + margin) - px
            dist_y_min = py - (aabb_min[1] - margin)
            dist_y_max = (aabb_max[1] + margin) - py
            dist_z_min = pz - (aabb_min[2] - margin)
            dist_z_max = (aabb_max[2] + margin) - pz

            # Find the smallest margin (closest to exiting the activation zone)
            min_margin_dist = min(dist_x_min, dist_x_max, dist_y_min, dist_y_max, dist_z_min, dist_z_max)

            # Determine which axis is closest to boundary
            margin_breakdown = []
            if dist_x_min == min_margin_dist:
                margin_breakdown.append(f"X_MIN:{dist_x_min:.2f}")
            if dist_x_max == min_margin_dist:
                margin_breakdown.append(f"X_MAX:{dist_x_max:.2f}")
            if dist_y_min == min_margin_dist:
                margin_breakdown.append(f"Y_MIN:{dist_y_min:.2f}")
            if dist_y_max == min_margin_dist:
                margin_breakdown.append(f"Y_MAX:{dist_y_max:.2f}")
            if dist_z_min == min_margin_dist:
                margin_breakdown.append(f"Z_MIN:{dist_z_min:.2f}")
            if dist_z_max == min_margin_dist:
                margin_breakdown.append(f"Z_MAX:{dist_z_max:.2f}")

            closest_axis = margin_breakdown[0] if margin_breakdown else "NONE"

            # State transition detection
            state_changed = prev_active is not None and prev_active != active
            state_str = "→ACTIVE" if active else "→INACTIVE"
            if state_changed:
                state_str = f"CHANGED {state_str}"

            # Log every frame with full context
            log_game("DYN-ACTIVATE",
                f"{dyn_obj.name} {state_str} reason={activation_reason} | "
                f"player=({px:.2f},{py:.2f},{pz:.2f}) | "
                f"aabb=[({aabb_min[0]:.2f},{aabb_min[1]:.2f},{aabb_min[2]:.2f})->({aabb_max[0]:.2f},{aabb_max[1]:.2f},{aabb_max[2]:.2f})] | "
                f"margin={margin:.1f}m closest={closest_axis} min_dist={min_margin_dist:.2f}m")

            # Additional warning if very close to deactivation boundary
            if active and min_margin_dist < 0.5:
                log_game("DYN-ACTIVATE",
                    f"⚠ {dyn_obj.name} NEAR_BOUNDARY min_dist={min_margin_dist:.2f}m < 0.5m (may flicker)")

        # Log state transitions to DYN-CACHE
        if debug_dyn_cache and prev_active is not None and prev_active != active:
            from ..developer.dev_logger import log_game
            state_str = "ACTIVATED" if active else "DEACTIVATED"
            log_game("DYN-CACHE", f"AABB {state_str}: {dyn_obj.name} reason={activation_reason}")

        modal_op._dyn_active_state[dyn_obj] = active

        if not active:
            # Keep "previous" pose updated cheaply, then skip heavy work
            modal_op.platform_prev_positions[dyn_obj] = cur_pos_quick.copy()
            modal_op.platform_prev_matrices[dyn_obj] = cur_M_quick.copy()
            continue

        # -------- 2) ACTIVE path ----------
        cur_M = cur_M_quick.copy()

        # Compute & cache radius if not already done
        if dyn_obj not in modal_op._cached_dyn_radius:
            bbox_world = [dyn_obj.matrix_world @ mathutils.Vector(c) for c in dyn_obj.bound_box]
            center_world = sum(bbox_world, mathutils.Vector()) / 8.0
            rad_center = max((p - center_world).length for p in bbox_world)
            origin_world = dyn_obj.matrix_world.translation
            center_offset = (center_world - origin_world).length
            modal_op._cached_dyn_radius[dyn_obj] = rad_center + center_offset

        rad = modal_op._cached_dyn_radius.get(dyn_obj, 0.0)

        # Store in map for camera submission and platform carry lookup
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

    # ========== PHASE 3: Summary Logging ==========
    # Log overall activation state summary
    if debug_activation and total_dynamic_meshes > 0:
        from ..developer.dev_logger import log_game
        log_game("DYN-ACTIVATE",
            f"SUMMARY: total={total_dynamic_meshes} active={active_count} inactive={inactive_count} "
            f"standing_override={standing_override_count} | "
            f"transforms_to_send={len(modal_op.dynamic_objects_map)}")