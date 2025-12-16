# Exp_Game/engine/worker/physics.py
"""
KCC (Kinematic Character Controller) physics implementation for worker process.
Handles full physics frame computation: movement, collision, ground detection, etc.
"""

import math
import time

# Import raycast functions from sibling module
from .raycast import (
    unified_raycast,
    cast_ray,
    test_dynamic_meshes_ray,
)

# Import math utilities
from .math import (
    ray_triangle_intersect,
    compute_triangle_normal,
    compute_facing_normal,
    ray_sphere_intersect,
    compute_bounding_sphere,
    compute_aabb,
    transform_aabb_by_matrix,
    ray_aabb_intersect,
    invert_matrix_4x4,
    transform_ray_to_local,
    transform_point,
    transform_triangle,
    get_adaptive_grid_resolution,
    build_triangle_grid,
    ray_grid_traverse,
)


def handle_kcc_physics_step(job_data, cached_grid, cached_dynamic_meshes, cached_dynamic_transforms):
    """
    Process a full KCC physics step. This is the main physics computation function.
    
    Args:
        job_data: dict with input parameters (pos, vel, config, etc.)
        cached_grid: static collision grid data
        cached_dynamic_meshes: dict of dynamic mesh data
        cached_dynamic_transforms: dict of {obj_id: (matrix, aabb, inv_matrix)}
    
    Returns:
        dict with computed physics result (new pos, vel, ground state, etc.)
    """
    calc_start = time.perf_counter()

    # =================================================================
    # FULL KCC PHYSICS STEP - Worker computes entire physics frame
    # =================================================================
    # This is the new architecture: worker does ALL physics computation,
    # main thread only applies results to Blender objects.
    # NO prediction needed - worker computes actual result.
    # =================================================================
    import math

    calc_start = time.perf_counter()

    # ─────────────────────────────────────────────────────────────────
    # UNPACK INPUT
    # ─────────────────────────────────────────────────────────────────
    pos = job_data.get("pos", (0.0, 0.0, 0.0))
    vel = job_data.get("vel", (0.0, 0.0, 0.0))
    on_ground = job_data.get("on_ground", False)
    on_walkable = job_data.get("on_walkable", True)
    ground_normal = job_data.get("ground_normal", (0.0, 0.0, 1.0))

    wish_dir = job_data.get("wish_dir", (0.0, 0.0))
    is_running = job_data.get("is_running", False)
    jump_requested = job_data.get("jump_requested", False)

    coyote_remaining = job_data.get("coyote_remaining", 0.0)
    jump_buffer_remaining = job_data.get("jump_buffer_remaining", 0.0)

    dt = job_data.get("dt", 1.0/30.0)

    # Config
    cfg = job_data.get("config", {})
    radius = cfg.get("radius", 0.22)
    height = cfg.get("height", 1.8)
    gravity = cfg.get("gravity", -9.81)
    max_walk = cfg.get("max_walk", 2.5)
    max_run = cfg.get("max_run", 5.5)
    accel_ground = cfg.get("accel_ground", 20.0)
    accel_air = cfg.get("accel_air", 5.0)
    step_height = cfg.get("step_height", 0.4)
    snap_down = cfg.get("snap_down", 0.5)
    slope_limit_deg = cfg.get("slope_limit_deg", 50.0)
    jump_speed = cfg.get("jump_speed", 7.0)
    coyote_time = cfg.get("coyote_time", 0.08)

    floor_cos = math.cos(math.radians(slope_limit_deg))

    # Debug flags
    debug_flags = job_data.get("debug_flags", {})
    # UNIFIED PHYSICS: All flags control unified physics (static + dynamic identical)
    debug_physics = debug_flags.get("physics", False)      # Physics summary
    debug_ground = debug_flags.get("ground", False)        # Ground detection
    debug_horizontal = debug_flags.get("horizontal", False) # Horizontal collision
    debug_body = debug_flags.get("body", False)            # Body integrity
    debug_ceiling = debug_flags.get("ceiling", False)      # Ceiling check
    debug_step = debug_flags.get("step", False)            # Step-up
    debug_slide = debug_flags.get("slide", False)          # Wall slide
    debug_slopes = debug_flags.get("slopes", False)        # Slopes
    debug_dynamic_opt = debug_flags.get("dynamic_opt", False)  # Dynamic mesh optimization stats

    # Worker log buffer (collected during computation, returned to main thread)
    worker_logs = []

    # Convert to mutable lists for computation
    px, py, pz = pos
    vx, vy, vz = vel
    gn_x, gn_y, gn_z = ground_normal
    wish_x, wish_y = wish_dir

    # Debug counters
    total_rays = 0
    total_tris = 0
    total_cells = 0
    h_blocked = False
    did_step_up = False
    did_slide = False
    hit_ceiling = False
    jump_consumed = False

    # ─────────────────────────────────────────────────────────────────
    # PERSISTENT TRANSFORM CACHE - UNIFIED ARCHITECTURE
    # ─────────────────────────────────────────────────────────────────
    # Worker maintains last-known transform for ALL dynamic meshes.
    # Main thread only sends transforms when meshes MOVE (thin/efficient).
    # Worker uses cached transforms for stationary meshes (zero main-thread cost).
    #
    # This eliminates the "activation" concept entirely:
    # - No AABB checks on main thread
    # - No chicken-and-egg timing issues
    # - Dynamic meshes behave like static: always available for testing
    # ─────────────────────────────────────────────────────────────────

    # cached_dynamic_transforms passed as parameter
    dynamic_transforms_update = job_data.get("dynamic_transforms", {})
    dynamic_velocities = job_data.get("dynamic_velocities", {})  # For diagnostics
    dynamic_transform_time_us = 0.0
    transforms_updated = 0
    transforms_from_cache = 0

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Update transform cache with any new transforms from main thread
    # Mesh triangles are cached via targeted broadcast_job (guaranteed delivery)
    # ═══════════════════════════════════════════════════════════════════
    for obj_id, matrix_4x4 in dynamic_transforms_update.items():
        cached = cached_dynamic_meshes.get(obj_id)
        if cached is None:
            continue  # Mesh triangles not cached yet, skip

        local_aabb = cached.get("local_aabb")
        if local_aabb:
            world_aabb = transform_aabb_by_matrix(local_aabb, matrix_4x4)
        else:
            world_aabb = None

        # Compute inverse matrix ONCE when transform changes (not every frame)
        inv_matrix = invert_matrix_4x4(matrix_4x4)

        # Cache the transform + computed world AABB + inverse matrix
        cached_dynamic_transforms[obj_id] = (matrix_4x4, world_aabb, inv_matrix)
        transforms_updated += 1

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Build unified_dynamic_meshes from ALL cached transforms
    # ═══════════════════════════════════════════════════════════════════
    # This is the key change: we test ALL meshes that have cached transforms,
    # not just ones that received updates this frame.
    unified_dynamic_meshes = []
    transform_start = time.perf_counter()

    for obj_id, (matrix_4x4, world_aabb, inv_matrix) in cached_dynamic_transforms.items():
        cached = cached_dynamic_meshes.get(obj_id)
        if cached is None:
            continue  # Shouldn't happen, but be safe

        # Skip if inverse matrix failed (singular matrix)
        if inv_matrix is None:
            continue

        local_triangles = cached["triangles"]

        # Check if this was a fresh update or from cache
        if obj_id not in dynamic_transforms_update:
            transforms_from_cache += 1

        # inv_matrix already cached - no per-frame computation needed!

        # OPTIMIZED: Skip bounding sphere computation - unified_raycast uses AABB first
        # Bounding sphere is only a fallback, and we always have AABB from local_aabb transform
        # This saves ~200 ops per mesh per frame (center calc + sqrt for radius)
        # If AABB is somehow None, fall back to cached radius
        bounding_sphere = None
        if not world_aabb:
            bounding_sphere = ((0, 0, 0), cached.get("radius", 1.0))

        # Add to unified format - triangles stay in LOCAL space!
        unified_dynamic_meshes.append({
            "obj_id": obj_id,
            "triangles": local_triangles,
            "matrix": matrix_4x4,
            "inv_matrix": inv_matrix,
            "bounding_sphere": bounding_sphere,
            "aabb": world_aabb,
            "grid": cached.get("grid")
        })

    transform_end = time.perf_counter()
    dynamic_transform_time_us = (transform_end - transform_start) * 1_000_000

    # ═══════════════════════════════════════════════════════════════════
    # DIAGNOSTIC LOGGING - Unified Dynamic Mesh System
    # ═══════════════════════════════════════════════════════════════════
    total_cached_meshes = len(cached_dynamic_meshes)
    total_cached_transforms = len(cached_dynamic_transforms)
    mesh_count = len(unified_dynamic_meshes)
    total_dyn_tris = sum(len(m["triangles"]) for m in unified_dynamic_meshes)

    if debug_flags.get("engine", False):
        # Cache efficiency: how many transforms came from persistent cache
        cache_hit_pct = (transforms_from_cache / mesh_count * 100) if mesh_count > 0 else 0
        worker_logs.append(("ENGINE",
            f"DYNAMIC: meshes={mesh_count} tris={total_dyn_tris} | "
            f"updated={transforms_updated} cached={transforms_from_cache} ({cache_hit_pct:.0f}% cache hit) | "
            f"transform_time={dynamic_transform_time_us:.0f}us"))

    if debug_flags.get("dynamic_cache", False):
        # Detailed cache state
        worker_logs.append(("DYN-CACHE",
            f"STATE: mesh_cache={total_cached_meshes} transform_cache={total_cached_transforms} active={mesh_count} | "
            f"updated={transforms_updated} from_cache={transforms_from_cache} | "
            f"tris={total_dyn_tris} time={dynamic_transform_time_us:.1f}us"))

    # ─────────────────────────────────────────────────────────────────
    # 1. TIMERS
    # ─────────────────────────────────────────────────────────────────
    coyote_remaining = max(0.0, coyote_remaining - dt)

    # ─────────────────────────────────────────────────────────────────
    # 2. INPUT → VELOCITY (Acceleration)
    # ─────────────────────────────────────────────────────────────────
    target_speed = max_run if is_running else max_walk
    accel = accel_ground if on_ground else accel_air

    # Lerp toward desired velocity
    desired_x = wish_x * target_speed
    desired_y = wish_y * target_speed

    # NOTE: Uphill blocking moved to Step 8 (after ground detection) to use current frame's ground normal

    t = min(1.0, accel * dt)
    vx = vx + (desired_x - vx) * t
    vy = vy + (desired_y - vy) * t

    # ─────────────────────────────────────────────────────────────────
    # 3. GRAVITY
    # ─────────────────────────────────────────────────────────────────
    # Note: Steep slope sliding moved to step 8 (after ground detection)
    if not on_ground:
        vz += gravity * dt
    else:
        if on_walkable:
            vz = max(vz, 0.0)
        # Steep slope sliding handled in step 8 after ground normal is updated

    # ─────────────────────────────────────────────────────────────────
    # 4. JUMP
    # ─────────────────────────────────────────────────────────────────
    can_jump = (on_ground and on_walkable) or (coyote_remaining > 0.0)
    if jump_requested and can_jump:
        vz = jump_speed
        on_ground = False
        jump_consumed = True
        coyote_remaining = 0.0

    # ─────────────────────────────────────────────────────────────────
    # 4.5. UPHILL BLOCKING (BEFORE movement - using last frame's ground normal)
    # ─────────────────────────────────────────────────────────────────
    # CRITICAL: This must happen BEFORE horizontal collision/movement!
    # Block uphill movement on steep slopes (grounded OR airborne)
    # This prevents jump-spam climbing by blocking based on last ground contact

    # Check if we have a steep slope from last ground contact
    slope_angle = 0.0
    if gn_z < 1.0:  # Have a ground normal from previous contact
        slope_angle = math.degrees(math.acos(min(1.0, max(-1.0, gn_z))))

    is_steep = slope_angle > slope_limit_deg

    # Block uphill movement if on steep slope (grounded OR recently airborne from steep slope)
    if is_steep and (on_ground or (not on_ground and vz > -2.0)):  # vz > -2 = recently jumped/airborne
        gn_xy_len = math.sqrt(gn_x*gn_x + gn_y*gn_y)
        if gn_xy_len > 0.001:
            # Calculate uphill direction (negate normal's XY projection)
            uphill_x = -gn_x / gn_xy_len
            uphill_y = -gn_y / gn_xy_len

            # Check uphill velocity
            uphill_vel = vx * uphill_x + vy * uphill_y

            if uphill_vel > 0.0:
                # Remove ALL uphill velocity (no pushback to avoid getting stuck)
                vx = vx - uphill_x * uphill_vel
                vy = vy - uphill_y * uphill_vel

                # Optional: Very gentle nudge downhill ONLY when airborne
                # This helps prevent jump spam without causing stuck-in-mesh issues
                if not on_ground and slope_angle > 65.0:
                    # Gentle downhill nudge only when airborne on very steep slopes
                    downhill_x = -uphill_x
                    downhill_y = -uphill_y
                    vx += downhill_x * 2.0
                    vy += downhill_y * 2.0

                    if debug_slopes:
                        worker_logs.append(("SLOPES", f"PRE-BLOCK AIRBORNE angle={slope_angle:.0f}°"))

    # ─────────────────────────────────────────────────────────────────
    # 4.9. EXTRACT GRID DATA ONCE (performance optimization)
    # ─────────────────────────────────────────────────────────────────
    # Cache grid data to avoid repeated dictionary lookups (4x per frame)
    # Also create grid_data dict for unified_raycast
    grid_bounds_min = None
    grid_bounds_max = None
    grid_cell_size = None
    grid_dims = None
    grid_cells = None
    grid_triangles = None
    grid_min_x = grid_min_y = grid_min_z = 0.0
    grid_max_x = grid_max_y = grid_max_z = 0.0
    grid_nx = grid_ny = grid_nz = 0
    grid_data = None  # For unified_raycast

    if cached_grid is not None:
        grid_bounds_min = cached_grid["bounds_min"]
        grid_bounds_max = cached_grid["bounds_max"]
        grid_cell_size = cached_grid["cell_size"]
        grid_dims = cached_grid["grid_dims"]
        grid_cells = cached_grid["cells"]
        grid_triangles = cached_grid["triangles"]

        grid_min_x, grid_min_y, grid_min_z = grid_bounds_min
        grid_max_x, grid_max_y, grid_max_z = grid_bounds_max
        grid_nx, grid_ny, grid_nz = grid_dims

        # Create grid_data dict for unified_raycast
        grid_data = {
            "cells": grid_cells,
            "triangles": grid_triangles,
            "bounds_min": grid_bounds_min,
            "bounds_max": grid_bounds_max,
            "cell_size": grid_cell_size,
            "grid_dims": grid_dims
        }

    # ─────────────────────────────────────────────────────────────────
    # 4.99. DYNAMIC MESH PROXIMITY + PROACTIVE DETECTION (OPTIMIZED)
    # When mesh is near AND (player stationary OR mesh approaching),
    # cast rays TOWARD mesh to detect contact before normal collision
    #
    # OPTIMIZATIONS:
    # - Reduced ray directions: velocity-opposite + center only (was 8, now 2-4)
    # - Perpendicular rays only for very fast meshes (>8m/s)
    # - Early-out when contact found (< radius)
    # - Stats tracking for performance monitoring
    # ─────────────────────────────────────────────────────────────────
    proximity_meshes = []
    proactive_best_d = None
    proactive_best_n = None
    proactive_obj_id = None
    proactive_mesh_vel = None  # Cache velocity to avoid duplicate lookup

    # Optimization stats
    opt_meshes_total = len(unified_dynamic_meshes) if unified_dynamic_meshes else 0
    opt_meshes_proximity = 0
    opt_meshes_aabb_skip = 0
    opt_meshes_slow_skip = 0
    opt_rays_proactive = 0
    opt_rays_hit = 0
    opt_early_out = False

    if unified_dynamic_meshes:
        # Player capsule AABB
        p_min = (px - radius, py - radius, pz)
        p_max = (px + radius, py + radius, pz + height)
        player_speed = math.sqrt(vx*vx + vy*vy)

        for dyn_mesh in unified_dynamic_meshes:
            # Early-out: if we already found very close contact, stop searching
            if proactive_best_d is not None and proactive_best_d < radius * 0.5:
                opt_early_out = True
                break

            obj_id = dyn_mesh.get("obj_id")
            mesh_aabb = dyn_mesh.get("aabb")
            if mesh_aabb is None:
                continue

            m_min, m_max = mesh_aabb

            # Get mesh velocity FIRST to scale detection range
            mesh_vel = dynamic_velocities.get(obj_id, (0.0, 0.0, 0.0))
            mvx, mvy, mvz = mesh_vel
            mesh_speed = math.sqrt(mvx*mvx + mvy*mvy)

            # AABB overlap check - expand by velocity to catch fast meshes EARLY
            # At 20m/s, mesh travels 0.66m/frame - need to detect 2-3 frames ahead
            speed_expand = mesh_speed * dt * 3.0  # 3 frames of travel
            expand = radius + 0.3 + speed_expand
            overlap = (
                p_min[0] - expand <= m_max[0] and p_max[0] + expand >= m_min[0] and
                p_min[1] - expand <= m_max[1] and p_max[1] + expand >= m_min[1] and
                p_min[2] - expand <= m_max[2] and p_max[2] + expand >= m_min[2]
            )

            if not overlap:
                opt_meshes_aabb_skip += 1
                continue

            opt_meshes_proximity += 1

            # Compute mesh center
            mcx = (m_min[0] + m_max[0]) * 0.5
            mcy = (m_min[1] + m_max[1]) * 0.5

            # Direction from player to mesh center (horizontal only)
            dx = mcx - px
            dy = mcy - py
            dist_xy = math.sqrt(dx*dx + dy*dy)

            proximity_meshes.append(obj_id)

            if debug_horizontal:
                worker_logs.append(("HORIZONTAL",
                    f"PROXIMITY obj={obj_id} dist_xy={dist_xy:.2f}m "
                    f"mesh_vel=({mvx:.2f},{mvy:.2f},{mvz:.2f}) speed={mesh_speed:.2f}m/s "
                    f"player_vel=({vx:.2f},{vy:.2f})"))

            # Check if mesh is moving fast enough to need proactive detection
            if mesh_speed < 0.5 and player_speed >= 1.0:
                opt_meshes_slow_skip += 1
                continue  # Mesh slow and player moving - normal collision handles it

            # OPTIMIZED: Reduced ray directions (was 8 directions, now 2-4)
            # Primary: velocity-opposite (most important)
            # Secondary: toward mesh center (backup)
            # Tertiary: perpendicular only for VERY fast meshes
            ray_directions = []

            if mesh_speed > 0.5:
                # Primary: Cast opposite to mesh velocity (toward approaching face)
                ray_dir_x = -mvx / mesh_speed
                ray_dir_y = -mvy / mesh_speed
                ray_directions.append((ray_dir_x, ray_dir_y, mesh_speed))

                # Tertiary: Perpendicular rays for very fast meshes only (>8m/s)
                # Catches glancing collisions at high speed
                if mesh_speed > 8.0:
                    perp_x = -ray_dir_y  # Perpendicular left
                    perp_y = ray_dir_x
                    ray_directions.append((perp_x, perp_y, radius + 0.5))
                    ray_directions.append((-perp_x, -perp_y, radius + 0.5))

            # Secondary: Also cast toward mesh center
            if dist_xy > 0.1:
                ray_dir_x = dx / dist_xy
                ray_dir_y = dy / dist_xy
                ray_directions.append((ray_dir_x, ray_dir_y, dist_xy))

            # 3 heights: feet, mid, head (head important for beams/overhangs)
            ray_heights = [0.1, height * 0.5, height - radius]

            # Scale ray distance by mesh speed - need to see further ahead for fast meshes
            speed_ray_extend = mesh_speed * dt * 3.0  # Look 3 frames ahead
            contact_range = radius + 0.1 + speed_ray_extend  # Detection triggers this far out

            for ray_dir_x, ray_dir_y, ray_max_hint in ray_directions:
                # Early-out within direction loop
                if proactive_best_d is not None and proactive_best_d < radius * 0.5:
                    break

                ray_max = max(radius + 0.5 + speed_ray_extend, ray_max_hint + speed_ray_extend)

                for ray_z in ray_heights:
                    tris_counter = [0]
                    total_rays += 1
                    opt_rays_proactive += 1
                    ray_result = cast_ray(
                        (px, py, pz + ray_z), (ray_dir_x, ray_dir_y, 0), ray_max,
                        cached_grid, unified_dynamic_meshes, tris_counter
                    )
                    total_tris += tris_counter[0]

                    if ray_result:
                        hit_dist, hit_normal, hit_source, hit_obj_id = ray_result
                        opt_rays_hit += 1
                        # Trigger if within contact range (scaled by speed)
                        if hit_dist < contact_range:
                            if proactive_best_d is None or hit_dist < proactive_best_d:
                                proactive_best_d = hit_dist
                                proactive_best_n = hit_normal
                                proactive_obj_id = hit_obj_id
                                # Cache velocity to avoid duplicate lookup in push response
                                proactive_mesh_vel = (mvx, mvy, mvz, mesh_speed)

                                if debug_horizontal:
                                    worker_logs.append(("HORIZONTAL",
                                        f"PROACTIVE_HIT obj={hit_obj_id} dist={hit_dist:.3f}m "
                                        f"contact_range={contact_range:.2f}m "
                                        f"normal=({hit_normal[0]:.2f},{hit_normal[1]:.2f},{hit_normal[2]:.2f})"))

    # Log optimization stats
    if debug_dynamic_opt and opt_meshes_total > 0:
        worker_logs.append(("DYN-OPT",
            f"PROACTIVE meshes={opt_meshes_proximity}/{opt_meshes_total} "
            f"aabb_skip={opt_meshes_aabb_skip} slow_skip={opt_meshes_slow_skip} "
            f"rays={opt_rays_proactive} hits={opt_rays_hit} "
            f"early_out={opt_early_out} best_d={proactive_best_d if proactive_best_d else 'none'}"))

    # ─────────────────────────────────────────────────────────────────
    # 5. HORIZONTAL COLLISION - UNIFIED for all geometry
    # Single cast_ray() tests BOTH static and dynamic for each ray
    # ─────────────────────────────────────────────────────────────────
    move_x = vx * dt
    move_y = vy * dt
    move_len = math.sqrt(move_x*move_x + move_y*move_y)

    best_d = None
    best_n = None
    per_ray_hits = []

    if move_len > 1e-9:
        # Normalize movement direction
        fwd_x = move_x / move_len
        fwd_y = move_y / move_len

        # Cast rays at 3 heights (feet, mid, head)
        ray_heights = [0.1, min(height * 0.5, height - radius), height - radius]
        ray_len = move_len + radius

        # Main horizontal rays (3 heights)
        for ray_z in ray_heights:
            total_rays += 1
            tris_counter = [0]
            ray_result = cast_ray(
                (px, py, pz + ray_z), (fwd_x, fwd_y, 0), ray_len,
                cached_grid, unified_dynamic_meshes, tris_counter
            )
            total_tris += tris_counter[0]

            if ray_result:
                ray_dist, ray_normal, ray_source, ray_obj_id = ray_result
                per_ray_hits.append(ray_dist)
                if best_d is None or ray_dist < best_d:
                    best_d = ray_dist
                    best_n = ray_normal
            else:
                per_ray_hits.append(None)

        # WIDTH RAYS: Check left/right edges at mid-height for narrow gaps
        perp_x = -fwd_y  # Perpendicular left
        perp_y = fwd_x
        mid_height = height * 0.5
        width_ray_configs = [
            (perp_x * radius, perp_y * radius),   # Left edge
            (-perp_x * radius, -perp_y * radius), # Right edge
        ]

        width_hits = []
        for width_offset_x, width_offset_y in width_ray_configs:
            total_rays += 1
            tris_counter = [0]
            width_result = cast_ray(
                (px + width_offset_x, py + width_offset_y, pz + mid_height),
                (fwd_x, fwd_y, 0), ray_len,
                cached_grid, unified_dynamic_meshes, tris_counter
            )
            total_tris += tris_counter[0]

            if width_result:
                width_hits.append((width_result[0], width_result[1]))
            else:
                width_hits.append((None, None))

        # Only block if BOTH width rays hit (narrow gap detection)
        if len(width_hits) == 2:
            left_dist, left_n = width_hits[0]
            right_dist, right_n = width_hits[1]
            if left_dist is not None and right_dist is not None:
                closest_dist = min(left_dist, right_dist)
                if best_d is None or closest_dist < best_d:
                    best_d = closest_dist
                    best_n = left_n if left_dist < right_dist else right_n

        # SLOPE RAYS: Angled down to catch steep slopes (only when grounded)
        if on_ground:
            slope_angle = 0.5  # ~30 degrees down
            slope_ray_z = radius * 2  # Knee height
            slope_fwd_len = math.sqrt(1.0 / (1.0 + slope_angle * slope_angle))
            slope_down_len = slope_angle * slope_fwd_len

            ray_dx = fwd_x * slope_fwd_len
            ray_dy = fwd_y * slope_fwd_len
            ray_dz = -slope_down_len

            total_rays += 1
            tris_counter = [0]
            slope_result = cast_ray(
                (px, py, pz + slope_ray_z), (ray_dx, ray_dy, ray_dz), ray_len,
                cached_grid, unified_dynamic_meshes, tris_counter
            )
            total_tris += tris_counter[0]

            if slope_result:
                slope_dist, slope_normal, slope_source, slope_obj_id = slope_result
                # Only count steep slopes (not floors)
                if slope_normal[2] < floor_cos and slope_normal[2] > 0.1:
                    horiz_dist = slope_dist * slope_fwd_len
                    if best_d is None or horiz_dist < best_d:
                        best_d = horiz_dist
                        best_n = slope_normal

        # Apply horizontal collision result (static OR dynamic - whichever is closer)
        if best_d is not None:
            h_blocked = True
            allowed = max(0.0, best_d - radius)

            # Move position
            if allowed > 1e-9:
                px += fwd_x * allowed
                py += fwd_y * allowed

            # Remove velocity component into wall
            if best_n is not None:
                bn_x, bn_y, bn_z = best_n
                vn = vx * bn_x + vy * bn_y
                if vn > 0.0:
                    vx -= bn_x * vn
                    vy -= bn_y * vn

                # ─────────────────────────────────────────────────────────
                # AIRBORNE STEEP SLOPE BLOCKING (backup to Step 4.5)
                # ─────────────────────────────────────────────────────────
                # If airborne and hitting a steep surface, block uphill velocity
                # This is a backup - Step 4.5 pre-blocking is the primary defense
                if not on_ground:
                    # Calculate slope angle from normal
                    slope_angle = math.degrees(math.acos(min(1.0, max(-1.0, bn_z))))

                    # If hitting a steep slope (> slope_limit_deg)
                    if slope_angle > slope_limit_deg:
                        # Calculate uphill direction (XY projection of normal, negated)
                        bn_xy_len = math.sqrt(bn_x*bn_x + bn_y*bn_y)
                        if bn_xy_len > 0.001:
                            uphill_x = -bn_x / bn_xy_len
                            uphill_y = -bn_y / bn_xy_len

                            # Check if moving uphill
                            uphill_vel = vx * uphill_x + vy * uphill_y

                            if uphill_vel > 0.0:
                                # Just remove uphill velocity, no pushback
                                # (Pushback causes stuck-in-mesh issues)
                                vx = vx - uphill_x * uphill_vel
                                vy = vy - uphill_y * uphill_vel

            # Debug logging
            if debug_horizontal:
                if best_n is not None:
                    worker_logs.append(("HORIZONTAL", f"blocked dist={best_d:.3f}m allowed={allowed:.3f}m normal=({best_n[0]:.2f},{best_n[1]:.2f},{best_n[2]:.2f}) | {total_rays}rays {total_tris}tris"))
                else:
                    worker_logs.append(("HORIZONTAL", f"blocked dist={best_d:.3f}m allowed={allowed:.3f}m normal=None | {total_rays}rays {total_tris}tris"))

            # ─────────────────────────────────────────────────────────
            # 5a. STEP-UP (if only feet ray hit) - UNIFIED
            # ─────────────────────────────────────────────────────────
            feet_only = (len(per_ray_hits) >= 3 and
                        per_ray_hits[0] is not None and
                        per_ray_hits[1] is None and
                        per_ray_hits[2] is None)

            # Only allow step-up when on walkable ground (prevent step-up on steep slopes)
            if on_ground and on_walkable and feet_only and step_height > 0.0 and best_n is not None:
                bn_x, bn_y, bn_z = best_n
                if bn_z < floor_cos:  # Steep face (step-able)
                    if debug_step:
                        worker_logs.append(("STEP", f"attempting step-up | pos=({px:.2f},{py:.2f},{pz:.2f}) face_nz={bn_z:.3f}"))

                    test_z = pz + step_height
                    head_ray_z = test_z + height

                    # UNIFIED headroom check - tests both static and dynamic
                    tris_counter = [0]
                    headroom_result = cast_ray(
                        (px, py, head_ray_z), (0, 0, 1), step_height,
                        cached_grid, unified_dynamic_meshes, tris_counter
                    )
                    total_tris += tris_counter[0]
                    head_clear = (headroom_result is None)

                    if head_clear:
                        remaining_move = move_len - allowed
                        if remaining_move > 0.01:
                            # Drop down to find ground
                            drop_ox = px + fwd_x * min(remaining_move, radius)
                            drop_oy = py + fwd_y * min(remaining_move, radius)
                            drop_max = step_height + snap_down + 1.0

                            # UNIFIED ground detection at drop position
                            tris_counter = [0]
                            drop_result = cast_ray(
                                (drop_ox, drop_oy, test_z + 1.0), (0, 0, -1), drop_max,
                                cached_grid, unified_dynamic_meshes, tris_counter
                            )
                            total_tris += tris_counter[0]

                            if drop_result:
                                drop_dist, step_ground_n, drop_source, drop_obj_id = drop_result
                                step_ground_z = test_z + 1.0 - drop_dist

                                # Check if walkable
                                gn_check = step_ground_n[2]
                                if gn_check >= floor_cos:
                                    if debug_step:
                                        worker_logs.append(("STEP", f"SUCCESS source={drop_source} | drop_pos=({drop_ox:.2f},{drop_oy:.2f},{step_ground_z:.2f}) gn_z={gn_check:.3f}"))
                                    px = drop_ox
                                    py = drop_oy
                                    pz = step_ground_z
                                    vz = 0.0
                                    on_ground = True
                                    on_walkable = True
                                    gn_x, gn_y, gn_z = step_ground_n
                                    did_step_up = True
                                    coyote_remaining = coyote_time

            # ─────────────────────────────────────────────────────────
            # 5b. WALL SLIDE (if blocked and not step-up) - UNIFIED
            # ─────────────────────────────────────────────────────────
            if not did_step_up and best_n is not None:
                remaining = move_len - allowed
                if remaining > 0.001:
                    bn_x, bn_y, bn_z = best_n
                    is_steep_face = bn_z < floor_cos and bn_z > 0.1

                    # Compute slide direction (tangent to wall)
                    dot = fwd_x * bn_x + fwd_y * bn_y
                    slide_x = fwd_x - bn_x * dot
                    slide_y = fwd_y - bn_y * dot

                    # OPTIMIZED: Check squared length first to avoid sqrt when zeroing
                    # 0.259² ≈ 0.067
                    slide_len_sq = slide_x*slide_x + slide_y*slide_y
                    if slide_len_sq < 0.067:
                        slide_len = 0.0
                    else:
                        slide_len = math.sqrt(slide_len_sq)

                    # Block uphill sliding on steep slopes
                    if is_steep_face and on_ground and slide_len > 1e-9:
                        uphill_xy_len = math.sqrt(bn_x*bn_x + bn_y*bn_y)
                        if uphill_xy_len > 0.001:
                            uphill_x = bn_x / uphill_xy_len
                            uphill_y = bn_y / uphill_xy_len
                            slide_nx = slide_x / slide_len
                            slide_ny = slide_y / slide_len
                            uphill_dot = slide_nx * uphill_x + slide_ny * uphill_y
                            if uphill_dot > -0.1:
                                slide_len = 0.0

                    if slide_len > 1e-9:
                        slide_x /= slide_len
                        slide_y /= slide_len
                        slide_dist = remaining * 0.65

                        # UNIFIED slide collision check
                        tris_counter = [0]
                        slide_result = cast_ray(
                            (px, py, pz + height * 0.5), (slide_x, slide_y, 0),
                            slide_dist + radius, cached_grid, unified_dynamic_meshes, tris_counter
                        )
                        total_tris += tris_counter[0]

                        slide_allowed = slide_dist
                        if slide_result:
                            slide_hit_dist = slide_result[0]
                            slide_allowed = min(slide_allowed, max(0, slide_hit_dist - radius))

                        if slide_allowed > 0.01:
                            px += slide_x * slide_allowed
                            py += slide_y * slide_allowed
                            did_slide = True

                        # Reduce velocity
                        vx *= 0.65
                        vy *= 0.65

                        # SLIDE DIAGNOSTICS (logging only - after slide is applied)
                        if debug_slide:
                            try:
                                slide_applied = slide_allowed if slide_allowed > 0.01 else 0.0
                                slide_requested = slide_dist
                                slide_blocked = slide_result is not None
                                effectiveness = (slide_applied / slide_requested * 100.0) if slide_requested > 0.001 else 0.0
                                worker_logs.append(("SLIDE",
                                    f"applied={slide_applied:.3f}m requested={slide_requested:.3f}m eff={effectiveness:.0f}% "
                                    f"normal=({bn_x:.2f},{bn_y:.2f},{bn_z:.2f}) blocked={slide_blocked}"))
                            except:
                                pass  # Silently ignore any logging errors
        else:
            # No collision - full movement
            px += move_x
            py += move_y
            if debug_horizontal and move_len > 1e-9:
                worker_logs.append(("HORIZONTAL", f"clear move={move_len:.3f}m | {total_rays}rays {total_tris}tris"))

    elif move_len > 1e-9:
        # No grid cached - just move
        px += move_x
        py += move_y

    # ─────────────────────────────────────────────────────────────────
    # 5.1 PROACTIVE COLLISION RESPONSE
    # Always push player away if penetrating a dynamic mesh
    # Normal collision only BLOCKS movement, doesn't PUSH away
    # ─────────────────────────────────────────────────────────────────
    if proactive_best_d is not None:
        pn_x, pn_y, pn_z = proactive_best_n

        # Use cached velocity (OPTIMIZED: avoids duplicate dict lookup + sqrt)
        mesh_vel_toward = 0.0
        mesh_speed_total = 0.0
        if proactive_mesh_vel is not None:
            mvx, mvy, mvz, mesh_speed_total = proactive_mesh_vel
            # How fast is mesh moving toward player? (along normal)
            mesh_vel_toward = -(mvx * pn_x + mvy * pn_y)  # Negative because normal points away from mesh

        # Calculate "safe distance" - how far we need to be to survive next frame
        # At high speeds, need bigger safety margin
        safe_distance = radius + mesh_speed_total * dt * 2.0

        # Penetration relative to safe distance (not just radius)
        effective_penetration = safe_distance - proactive_best_d

        # Always push if within safe distance
        if effective_penetration > -0.1:  # Within 10cm of safe threshold
            h_blocked = True

            # Push amount = get to safe distance + buffer
            # For 20m/s mesh: safe_distance ~= 0.22 + 20*0.033*2 = 1.54m
            push_base = max(0.0, effective_penetration + 0.05)

            # Extra velocity-based push for very fast meshes
            push_velocity = max(0.0, mesh_vel_toward * dt * 2.0)

            push_total = push_base + push_velocity

            if push_total > 0.001:
                px += pn_x * push_total
                py += pn_y * push_total

            # Remove velocity component into wall
            vn = vx * pn_x + vy * pn_y
            if vn > 0.0:
                vx -= pn_x * vn
                vy -= pn_y * vn

            # Inherit mesh velocity (horizontal carry)
            # Uses cached mvx, mvy from proactive_mesh_vel (no extra lookup)
            if mesh_vel_toward > 0.5 and proactive_mesh_vel is not None:
                vx += mvx
                vy += mvy

            if debug_horizontal:
                worker_logs.append(("HORIZONTAL",
                    f"PROACTIVE_PUSH dist={proactive_best_d:.3f}m safe={safe_distance:.2f}m "
                    f"pen={effective_penetration:.3f}m vel={mesh_vel_toward:.1f}m/s push={push_total:.3f}m"))

    # Post-collision diagnostic: did we miss any proximity meshes?
    if debug_horizontal and proximity_meshes and not h_blocked:
        worker_logs.append(("HORIZONTAL",
            f"MISSED? prox_meshes={len(proximity_meshes)} no_collision | "
            f"move_len={move_len:.3f}m player_vel=({vx:.2f},{vy:.2f})"))

    # ─────────────────────────────────────────────────────────────────
    # 5.5 VERTICAL BODY INTEGRITY CHECK - UNIFIED for all geometry
    # ─────────────────────────────────────────────────────────────────
    # Cast vertical ray from feet to head - if blocked, character is embedded in mesh
    body_embedded = False
    embed_distance = None

    # Feet at 0.1m to match horizontal ray height (better low-obstacle detection)
    feet_pos = (px, py, pz + 0.1)
    head_pos = (px, py, pz + height - radius)
    body_height = (height - radius) - 0.1  # Distance from feet (0.1m) to head

    tris_counter = [0]
    total_rays += 1
    body_result = cast_ray(
        feet_pos, (0, 0, 1), body_height,
        cached_grid, unified_dynamic_meshes, tris_counter
    )
    total_tris += tris_counter[0]

    if body_result:
        embed_distance, embed_normal, embed_source, embed_obj_id = body_result
        body_embedded = True

        if debug_body:
            penetration_pct = (embed_distance / body_height) * 100.0
            source_str = f"dynamic_{embed_obj_id}" if embed_source == "dynamic" else "static"
            worker_logs.append(("BODY",
                f"EMBEDDED source={source_str} hit={embed_distance:.3f}m pct={penetration_pct:.1f}%"))

    if debug_body:
        status = "EMBEDDED" if body_embedded else "CLEAR"
        worker_logs.append(("BODY", f"[{status}] feet=({feet_pos[0]:.2f},{feet_pos[1]:.2f},{feet_pos[2]:.2f}) head=({head_pos[0]:.2f},{head_pos[1]:.2f},{head_pos[2]:.2f}) h={body_height:.2f}m"))

    # ─────────────────────────────────────────────────────────────────
    # 5.6 EMBEDDING RESOLUTION (Prevention + Correction)
    # ─────────────────────────────────────────────────────────────────
    # Use vertical integrity ray data to prevent/fix mesh penetration
    if body_embedded and embed_distance is not None:
        # Mesh detected between feet and head at embed_distance from feet
        # embed_distance = distance from feet to the penetrating mesh

        # CASE 1: PREVENTION - Moving downward into mesh (falling scenario)
        # Stop character at mesh surface instead of penetrating through
        if vz < 0:
            # Character is falling and would penetrate mesh
            # Position feet at mesh surface: pz + embed_distance
            correction = embed_distance
            pz += correction
            vz = 0.0  # Kill downward velocity
            on_ground = True  # Treat as landing on surface
            on_walkable = True  # Assume walkable (will be verified by ground detection)

            if debug_body:
                worker_logs.append(("BODY", f"PREVENT-FALL corrected={correction:.3f}m landing on embedded mesh"))

        # CASE 2: CORRECTION - Already embedded from side collision
        # Push character up to clear the penetration
        else:
            # Character entered mesh horizontally (side collision)
            # Need to push up so mesh is no longer between feet and head
            # Add small buffer to ensure full clearance
            correction = embed_distance + 0.05  # 5cm buffer
            pz += correction
            vz = max(0.0, vz)  # Preserve upward velocity if any, kill downward

            if debug_body:
                worker_logs.append(("BODY", f"CORRECT-SIDE corrected={correction:.3f}m pushed up from embedded mesh"))

    # ─────────────────────────────────────────────────────────────────
    # 6. CEILING CHECK (if moving up) - UNIFIED for all geometry
    # ─────────────────────────────────────────────────────────────────
    if vz > 0.0:
        up_dist = vz * dt
        head_z = pz + height

        tris_counter = [0]
        total_rays += 1
        ceiling_result = cast_ray(
            (px, py, head_z), (0, 0, 1), up_dist,
            cached_grid, unified_dynamic_meshes, tris_counter
        )
        total_tris += tris_counter[0]

        if ceiling_result:
            ceil_dist, ceil_normal, ceil_source, ceil_obj_id = ceiling_result
            pz = head_z + ceil_dist - height
            vz = 0.0
            hit_ceiling = True

    # ─────────────────────────────────────────────────────────────────
    # 7. VERTICAL MOVEMENT + GROUND DETECTION (UNIFIED)
    # Single cast_ray() tests BOTH static and dynamic - identical physics
    # ─────────────────────────────────────────────────────────────────
    dz = vz * dt
    was_grounded = on_ground
    ground_hit_source = None  # Track if standing on static or dynamic mesh

    # UNIFIED GROUND RAY - tests ALL geometry in one call
    ray_start_z = pz + 1.0  # Guard above
    ray_max = snap_down + 1.0 + (abs(dz) if dz < 0 else 0)

    ground_hit_z = None
    ground_hit_n = None

    tris_counter = [0]
    total_rays += 1
    ground_result = cast_ray(
        (px, py, ray_start_z), (0, 0, -1), ray_max,
        cached_grid, unified_dynamic_meshes, tris_counter
    )
    total_tris += tris_counter[0]

    if ground_result:
        dist, normal, source, obj_id = ground_result
        # Sanity check: hit must be below player
        hit_z = ray_start_z - dist
        if hit_z <= pz + 0.1:  # Allow small tolerance
            ground_hit_z = hit_z
            ground_hit_n = normal
            if source == "dynamic":
                ground_hit_source = f"dynamic_{obj_id}"
            else:
                ground_hit_source = "static"

    # Debug logging - UNIFIED: Ground detection shows source (static or dynamic)
    if debug_ground and ground_hit_z is not None:
        worker_logs.append(("GROUND",
            f"HIT source={ground_hit_source} z={ground_hit_z:.2f}m "
            f"normal=({ground_hit_n[0]:.2f},{ground_hit_n[1]:.2f},{ground_hit_n[2]:.2f}) | "
            f"player_z={pz:.2f} tris={tris_counter[0]}"))

    # Apply vertical movement and ground snap (unified for all geometry)
    if dz < 0.0:  # Falling
        target_z = pz + dz
        if ground_hit_z is not None and target_z <= ground_hit_z:
            pz = ground_hit_z
            vz = 0.0
            on_ground = True
            if ground_hit_n is not None:
                gn_x, gn_y, gn_z = ground_hit_n
                on_walkable = gn_z >= floor_cos
            coyote_remaining = coyote_time
        else:
            pz = target_z
            on_ground = False
            on_walkable = False
    else:
        pz += dz

    # Ground snap (when grounded) or unground (when no ground found)
    if ground_hit_z is not None and abs(ground_hit_z - pz) <= snap_down and vz <= 0.0:
        # Ground found within snap distance - snap to it
        pz = ground_hit_z
        on_ground = True
        vz = 0.0
        if ground_hit_n is not None:
            gn_x, gn_y, gn_z = ground_hit_n
            on_walkable = gn_z >= floor_cos
        coyote_remaining = coyote_time
        if debug_ground:
            worker_logs.append(("GROUND", f"ON_GROUND snap | dist={abs(ground_hit_z - pz):.3f}m gn_z={gn_z:.3f} walkable={on_walkable}"))
    elif ground_hit_z is None and was_grounded:
        # Was grounded but no ground found - walked off a ledge!
        on_ground = False
        on_walkable = False
        coyote_remaining = coyote_time  # Grant coyote time
        if debug_ground:
            worker_logs.append(("GROUND", f"airborne | walked_off_ledge coyote={coyote_time:.2f}s"))
    elif not on_ground and was_grounded:
        coyote_remaining = coyote_time

    if debug_ground and ground_hit_z is not None and not on_ground:
        worker_logs.append(("GROUND", f"airborne | dist={abs(ground_hit_z - pz):.3f}m too_far | {total_tris}tris"))

    # ─────────────────────────────────────────────────────────────────
    # 8. STEEP SLOPE SLIDING (after ground detection updates the normal)
    # ─────────────────────────────────────────────────────────────────
    # Calculate slope angle from ground normal to check threshold
    steep_slope_detected = False
    slope_angle = 0.0
    if on_ground:
        gn_len = math.sqrt(gn_x*gn_x + gn_y*gn_y + gn_z*gn_z)
        if gn_len > 0.001:
            n_x = gn_x / gn_len
            n_y = gn_y / gn_len
            n_z = gn_z / gn_len
            slope_angle = math.degrees(math.acos(min(1.0, max(-1.0, n_z))))

            # Match slope_limit_deg to eliminate dead zone between walkable and steep
            steep_slope_detected = slope_angle > slope_limit_deg

    # If we're on ground on a steep slope (> slope_limit_deg), apply sliding and blocking
    if on_ground and steep_slope_detected:
        gn_len = math.sqrt(gn_x*gn_x + gn_y*gn_y + gn_z*gn_z)
        if gn_len > 0.001:
            n_x = gn_x / gn_len
            n_y = gn_y / gn_len
            n_z = gn_z / gn_len

            # UPHILL BLOCKING: Remove uphill velocity component (moved here to use current frame's normal)
            gn_xy_len = math.sqrt(n_x*n_x + n_y*n_y)
            if gn_xy_len > 0.001:
                # Normalize uphill direction
                # CRITICAL: Normal points DOWN the slope (outward from surface)
                # To get uphill, we need to NEGATE it!
                uphill_x = -n_x / gn_xy_len
                uphill_y = -n_y / gn_xy_len

                # Project current velocity onto uphill direction
                uphill_vel = vx * uphill_x + vy * uphill_y

                # Slope angle already calculated above

                # POST-MOVEMENT CORRECTION: Gentle backup if Step 4.5 missed anything
                # (Step 4.5 does main blocking BEFORE movement)
                if slope_angle > 65.0 and uphill_vel > 0.0:
                    # Just remove any remaining uphill velocity, minimal force
                    vx = vx - uphill_x * uphill_vel
                    vy = vy - uphill_y * uphill_vel

                    # Very gentle correction only
                    downhill_x = -uphill_x
                    downhill_y = -uphill_y
                    vx += downhill_x * 2.0  # Minimal correction
                    vy += downhill_y * 2.0

                    if debug_slopes:
                        worker_logs.append(("SLOPES", f"POST-CORRECT angle={slope_angle:.0f}° (backup)"))

                # Slopes slope_limit_deg-65°: Gentle correction
                elif slope_angle > slope_limit_deg and uphill_vel > 0.0:
                    # Just remove uphill velocity, no pushback
                    vx = vx - uphill_x * uphill_vel
                    vy = vy - uphill_y * uphill_vel

                # CRITICAL: On slopes > 65°, CLAMP Z position to prevent upward movement
                # This is the nuclear option - directly prevent position from moving up
                if slope_angle > 65.0 and on_ground:
                    # If character somehow moved upward on steep slope, FORCE them back down
                    # Store ground contact Z as maximum allowed Z
                    max_allowed_z = ground_hit_z if ground_hit_z is not None else pz
                    if pz > max_allowed_z:
                        if debug_slopes:
                            worker_logs.append(("SLOPES", f"Z-CLAMP angle={slope_angle:.0f}° prevented {pz - max_allowed_z:.3f}m upward movement"))
                        pz = max_allowed_z  # FORCE character to ground level or below

            # Compute slide direction (down the slope in XY plane)
            # The normal points OUT of the slope surface.
            # For a slope facing up-and-outward, normal XY points UPHILL.
            # To slide DOWNHILL, we go OPPOSITE to normal's XY projection.
            # But wait - if normal points outward from surface, and surface
            # is tilted up, then normal.xy points in the uphill direction.
            # So downhill = +normal.xy (not negative!)
            #
            # Actually: Consider a ramp going up in +X direction.
            # The surface normal would point up and back: (+small, 0, +large)
            # normalized, n_x is positive (pointing in +X, which is uphill)
            # To slide DOWN, we want -X direction, so slide = -n_x
            #
            # Hmm, let me think again with gravity:
            # Gravity pulls down (0,0,-1). Project onto slope plane:
            # g_tangent = g - n*(g.n) = (0,0,-1) - n*(-n_z) = (0,0,-1) + n*n_z
            # = (n_x*n_z, n_y*n_z, -1 + n_z*n_z)
            # The XY components are (n_x*n_z, n_y*n_z) - this IS the downhill direction

            slide_xy_len = math.sqrt(n_x*n_x + n_y*n_y)
            if slide_xy_len > 0.001:
                # Gravity projected onto slope gives downhill direction
                # g_tangent.xy = (n_x * n_z, n_y * n_z)
                slide_x = n_x * n_z
                slide_y = n_y * n_z

                # Normalize the slide direction
                slide_len = math.sqrt(slide_x*slide_x + slide_y*slide_y)
                if slide_len > 0.001:
                    slide_x /= slide_len
                    slide_y /= slide_len

                    # Slope steepness factor (0 = flat, 1 = vertical)
                    steepness = 1.0 - n_z

                    # Apply slide acceleration
                    # Use config parameter instead of hardcoded value
                    # MASSIVE slide force to overpower player input lerp system
                    # Doubled from 800 to 1600 for faster sliding
                    slope_slide_gain = cfg.get("steep_slide_gain", 1600.0)
                    slide_accel = slope_slide_gain * steepness * dt

                    vx += slide_x * slide_accel
                    vy += slide_y * slide_accel

                    # Ensure minimum downhill speed (prevents "sticking" on steep slopes)
                    # Increased from 2.5 to 8.0 for more responsive slide start
                    steep_min_speed = cfg.get("steep_min_speed", 8.0)
                    current_downhill_speed = vx * slide_x + vy * slide_y
                    if current_downhill_speed > 0.0 and current_downhill_speed < steep_min_speed:
                        # Boost velocity to maintain minimum slide speed
                        deficit = steep_min_speed - current_downhill_speed
                        vx += slide_x * deficit
                        vy += slide_y * deficit

                    # Limit maximum slide speed to prevent infinite acceleration
                    # Increased from 30 to 50 m/s for faster steep slope sliding
                    max_slide_speed = cfg.get("max_slide_speed", 50.0)  # m/s
                    slide_speed = math.sqrt(vx*vx + vy*vy)
                    if slide_speed > max_slide_speed:
                        scale = max_slide_speed / slide_speed
                        vx *= scale
                        vy *= scale

                    # Position push removed - was causing character to launch off ground
                    # slide_push = steepness * dt * 8.0
                    # px += slide_x * slide_push
                    # py += slide_y * slide_push

                    # SURFACE TRACKING: Project velocity onto slope to prevent bouncing
                    # When sliding fast on steep slopes, constrain velocity to follow surface
                    # This prevents character from launching off due to horizontal speed
                    if current_downhill_speed > 5.0 and steepness > 0.3:
                        # Project velocity onto slope plane: v_proj = v - (v · n) * n
                        # This removes the component perpendicular to surface
                        vel_dot_normal = vx * n_x + vy * n_y + vz * n_z
                        if vel_dot_normal > 0.0:  # Moving away from surface
                            # Remove perpendicular component to glue to surface
                            vx -= n_x * vel_dot_normal * 0.8  # 0.8 = damping factor
                            vy -= n_y * vel_dot_normal * 0.8
                            vz -= n_z * vel_dot_normal * 0.8

                    # Apply standard gravity (removed 1.2x multiplier that was too aggressive)
                    vz += gravity * dt

                    # Debug logging for steep slope sliding
                    if debug_slopes:
                        slope_angle = math.degrees(math.acos(min(1.0, max(-1.0, n_z))))
                        surface_tracking = "TRACKED" if (current_downhill_speed > 5.0 and steepness > 0.3) else "free"
                        worker_logs.append(("SLOPES", f"GRAVITY-SLIDE angle={slope_angle:.0f}° normal=({n_x:.2f},{n_y:.2f},{n_z:.2f}) "
                                                      f"dir=({slide_x:.2f},{slide_y:.2f}) "
                                                      f"vel_downhill={current_downhill_speed:.2f} max={max_slide_speed:.2f} "
                                                      f"accel={slide_accel:.2f} steep={steepness:.2f} track={surface_tracking}"))

    # ─────────────────────────────────────────────────────────────────
    # BUILD RESULT
    # ─────────────────────────────────────────────────────────────────
    calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

    # ═════════════════════════════════════════════════════════════════
    # PHYSICS SUMMARY - Unified physics (static + dynamic identical)
    # ═════════════════════════════════════════════════════════════════
    # Log unified physics system status (static + dynamic in single path)
    if debug_physics:
        transform_time = dynamic_transform_time_us
        total_time = calc_time_us
        physics_time = total_time - transform_time

        # Cache state
        cached_mesh_count = len(cached_dynamic_meshes)
        active_mesh_count = len(unified_dynamic_meshes)

        # Unified physics summary (single log with all info)
        ground_src = ground_hit_source if ground_hit_source else "none"

        if active_mesh_count > 0:
            # With dynamic meshes: show timing breakdown
            worker_logs.append(("PHYSICS",
                f"total={total_time:.0f}us (xform={transform_time:.0f}us) | "
                f"static+dynamic={active_mesh_count} | "
                f"rays={total_rays} tris={total_tris} | "
                f"ground={ground_src}"))
        else:
            # Static only: simpler log
            worker_logs.append(("PHYSICS",
                f"total={total_time:.0f}us | static_only | "
                f"rays={total_rays} tris={total_tris} | "
                f"ground={ground_src}"))

    # ═════════════════════════════════════════════════════════════════
    # DYNAMIC OPTIMIZATION SUMMARY - Per-frame stats
    # ═════════════════════════════════════════════════════════════════
    if debug_dynamic_opt and opt_meshes_total > 0:
        # Calculate rays used for normal collision (total - proactive)
        normal_rays = total_rays - opt_rays_proactive

        # Estimate efficiency: how many rays we saved vs old system
        # Old: 8 directions × 3 heights = 24 rays per mesh
        # New: 2-4 directions × 2 heights = 4-8 rays per mesh
        old_system_rays = opt_meshes_proximity * 24  # Worst case old
        rays_saved = max(0, old_system_rays - opt_rays_proactive)

        worker_logs.append(("DYN-OPT",
            f"FRAME total={calc_time_us:.0f}us | "
            f"rays: proactive={opt_rays_proactive} normal={normal_rays} total={total_rays} | "
            f"tris={total_tris} | "
            f"est_saved={rays_saved}rays"))

    result_data = {
        "pos": (px, py, pz),
        "vel": (vx, vy, vz),
        "on_ground": on_ground,
        "on_walkable": on_walkable,
        "ground_normal": (gn_x, gn_y, gn_z),
        "ground_hit_source": ground_hit_source,  # "static", "dynamic_ObjectName", or None
        "coyote_remaining": coyote_remaining,
        "jump_consumed": jump_consumed,
        "logs": worker_logs,  # Fast buffer logs (sent to main thread)
        "debug": {
            "rays_cast": total_rays,
            "triangles_tested": total_tris,
            "cells_traversed": total_cells,
            "calc_time_us": calc_time_us,
            "dynamic_meshes_active": len(unified_dynamic_meshes),
            "dynamic_transform_time_us": dynamic_transform_time_us,
            "h_blocked": h_blocked,
            "did_step_up": did_step_up,
            "did_slide": did_slide,
            "hit_ceiling": hit_ceiling,
            "body_embedded": body_embedded,
            "vertical_ray": {
                "origin": (px, py, pz + radius),
                "end": (px, py, pz + height - radius),
                "blocked": body_embedded
            }
        }
    }

    return result_data
