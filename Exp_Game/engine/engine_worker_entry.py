# Exp_Game/engine/engine_worker_entry.py
"""
Worker process entry point - ISOLATED from addon imports.
This module is called directly by multiprocessing and does NOT import bpy.
IMPORTANT: This file has NO relative imports to avoid triggering addon __init__.py
"""

import time
import traceback
from queue import Empty


# ============================================================================
# INLINE DEFINITIONS - Use DICTS instead of dataclasses for pickle safety
# ============================================================================

# Workers receive jobs as objects (sent from main thread)
# but RETURN results as plain dicts (pickle-safe)

# Debug flag (hardcoded to avoid config import)
# Controlled by scene.dev_debug_engine in the Developer Tools panel (main thread)
DEBUG_ENGINE = False

# ============================================================================
# WORKER-SIDE GRID CACHE
# ============================================================================
# Grid is sent once via CACHE_GRID job and stored here for all subsequent raycasts.
# This avoids 3MB serialization per raycast (20ms overhead eliminated).

_cached_grid = None  # Will hold the spatial grid data after CACHE_GRID job


# ============================================================================
# RAY-TRIANGLE INTERSECTION (Möller-Trumbore Algorithm)
# ============================================================================

def ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2):
    """
    Test if a ray intersects a triangle using Möller-Trumbore algorithm.

    Args:
        ray_origin: (x, y, z) ray starting point
        ray_direction: (x, y, z) ray direction (should be normalized)
        v0, v1, v2: Triangle vertices as (x, y, z) tuples

    Returns:
        (hit, distance, hit_point) where:
        - hit: True if intersection found
        - distance: Distance along ray to hit (or None)
        - hit_point: (x, y, z) hit location (or None)
    """
    import math

    EPSILON = 1e-8

    # Unpack vertices
    v0x, v0y, v0z = v0
    v1x, v1y, v1z = v1
    v2x, v2y, v2z = v2

    # Unpack ray
    ox, oy, oz = ray_origin
    dx, dy, dz = ray_direction

    # Edge vectors
    e1x = v1x - v0x
    e1y = v1y - v0y
    e1z = v1z - v0z

    e2x = v2x - v0x
    e2y = v2y - v0y
    e2z = v2z - v0z

    # Cross product: ray_dir × e2
    hx = dy * e2z - dz * e2y
    hy = dz * e2x - dx * e2z
    hz = dx * e2y - dy * e2x

    # Dot product: e1 · h
    a = e1x * hx + e1y * hy + e1z * hz

    # Ray parallel to triangle
    if abs(a) < EPSILON:
        return (False, None, None)

    f = 1.0 / a

    # Vector from v0 to ray origin
    sx = ox - v0x
    sy = oy - v0y
    sz = oz - v0z

    # Barycentric coordinate u
    u = f * (sx * hx + sy * hy + sz * hz)

    # Check bounds
    if u < 0.0 or u > 1.0:
        return (False, None, None)

    # Cross product: s × e1
    qx = sy * e1z - sz * e1y
    qy = sz * e1x - sx * e1z
    qz = sx * e1y - sy * e1x

    # Barycentric coordinate v
    v = f * (dx * qx + dy * qy + dz * qz)

    # Check bounds
    if v < 0.0 or u + v > 1.0:
        return (False, None, None)

    # Distance along ray
    t = f * (e2x * qx + e2y * qy + e2z * qz)

    # Check if hit is in front of ray
    if t < EPSILON:
        return (False, None, None)

    # Calculate hit point
    hit_x = ox + dx * t
    hit_y = oy + dy * t
    hit_z = oz + dz * t

    return (True, t, (hit_x, hit_y, hit_z))


def compute_triangle_normal(v0, v1, v2):
    """
    Compute the normal of a triangle using cross product.

    Args:
        v0, v1, v2: Triangle vertices as (x, y, z) tuples

    Returns:
        (nx, ny, nz) normalized normal vector
    """
    import math

    # Unpack vertices
    v0x, v0y, v0z = v0
    v1x, v1y, v1z = v1
    v2x, v2y, v2z = v2

    # Edge vectors
    e1x = v1x - v0x
    e1y = v1y - v0y
    e1z = v1z - v0z

    e2x = v2x - v0x
    e2y = v2y - v0y
    e2z = v2z - v0z

    # Cross product: e1 × e2
    nx = e1y * e2z - e1z * e2y
    ny = e1z * e2x - e1x * e2z
    nz = e1x * e2y - e1y * e2x

    # Normalize
    length = math.sqrt(nx*nx + ny*ny + nz*nz)
    if length > 1e-8:
        nx /= length
        ny /= length
        nz /= length

    return (nx, ny, nz)


def process_job(job) -> dict:
    """
    Process a single job and return result as a plain dict (pickle-safe).
    IMPORTANT: NO bpy access here!
    """
    global _cached_grid  # Must be at function top, not in elif blocks
    start_time = time.perf_counter()  # Higher precision than time.time()

    try:
        # ===================================================================
        # THIS IS WHERE YOU'LL ADD YOUR LOGIC LATER
        # For now, just echo back to prove it works
        # ===================================================================

        if job.job_type == "ECHO":
            # Simple echo test
            result_data = {
                "echoed": job.data,
                "worker_msg": "Job processed successfully"
            }

        elif job.job_type == "PING":
            # Worker verification ping - used during startup to confirm worker responsiveness
            result_data = {
                "pong": True,
                "worker_id": job.data.get("worker_check", -1),
                "timestamp": time.time(),
                "worker_msg": "Worker alive and responsive"
            }

        elif job.job_type == "CACHE_GRID":
            # Cache spatial grid for subsequent raycast jobs
            # This is sent ONCE at game start to avoid 3MB serialization per raycast
            grid = job.data.get("grid", None)
            if grid is not None:
                _cached_grid = grid
                tri_count = len(grid.get("triangles", []))
                cell_count = len(grid.get("cells", {}))
                if DEBUG_ENGINE:
                    print(f"[Worker] Grid cached: {tri_count:,} triangles, {cell_count:,} cells")
                result_data = {
                    "success": True,
                    "triangles": tri_count,
                    "cells": cell_count,
                    "message": "Grid cached successfully"
                }
            else:
                result_data = {
                    "success": False,
                    "error": "No grid data provided"
                }

        elif job.job_type == "COMPUTE_HEAVY":
            # Stress test - simulate realistic game calculation
            # (e.g., pathfinding, physics prediction, AI decision)
            iterations = job.data.get("iterations", 10)  # Reduced to 10 for realistic 1-5ms jobs
            data = job.data.get("data", [])

            # DIAGNOSTIC: Print actual iteration value being used
            if DEBUG_ENGINE:
                print(f"[Worker] COMPUTE_HEAVY job - iterations={iterations}, data_size={len(data)}")

            # Simulate realistic computation (1-5ms per job)
            # This mimics real game calculations like:
            # - AI pathfinding node evaluation
            # - Physics collision prediction
            # - Batch distance calculations
            total = 0
            for i in range(iterations):
                for val in data:
                    total += val * i
                    # Add some realistic computation
                    total = (total * 31 + val) % 1000000

            result_data = {
                "iterations_completed": iterations,
                "data_size": len(data),
                "result": total,
                "worker_msg": f"Completed {iterations} iterations",
                # Echo back metadata for tracking
                "scenario": job.data.get("scenario", "UNKNOWN"),
                "frame": job.data.get("frame", -1),
            }

        elif job.job_type == "CULL_BATCH":
            # Performance culling - distance-based object visibility
            # This is pure math (NO bpy access) and can run in parallel
            entry_ptr = job.data.get("entry_ptr", 0)
            obj_names = job.data.get("obj_names", [])
            obj_positions = job.data.get("obj_positions", [])
            ref_loc = job.data.get("ref_loc", (0, 0, 0))
            thresh = job.data.get("thresh", 10.0)
            start = job.data.get("start", 0)
            max_count = job.data.get("max_count", 100)

            # Compute distance-based culling (inline to avoid imports)
            rx, ry, rz = ref_loc
            t2 = float(thresh) * float(thresh)
            n = len(obj_names)

            if n == 0:
                result_data = {"entry_ptr": entry_ptr, "next_idx": start, "changes": []}
            else:
                i = 0
                changes = []
                idx = start % n

                while i < n and len(changes) < max_count:
                    name = obj_names[idx]
                    px, py, pz = obj_positions[idx]
                    dx = px - rx
                    dy = py - ry
                    dz = pz - rz
                    # distance^2 compare avoids sqrt
                    far = (dx*dx + dy*dy + dz*dz) > t2
                    changes.append((name, far))
                    i += 1
                    idx = (idx + 1) % n

                result_data = {"entry_ptr": entry_ptr, "next_idx": idx, "changes": changes}

        elif job.job_type == "DYNAMIC_MESH_ACTIVATION":
            # Track worker execution for verification
            calc_start = time.perf_counter()
            # Dynamic mesh proximity checks - distance-based activation gating
            # Pure math (NO bpy access) - determines which meshes should be active
            mesh_positions = job.data.get("mesh_positions", [])
            mesh_objects = job.data.get("mesh_objects", [])  # List of (obj, prev_active) tuples
            player_position = job.data.get("player_position", (0, 0, 0))
            base_distances = job.data.get("base_distances", [])

            px, py, pz = player_position
            activation_decisions = []

            for i, (mesh_pos, (obj_name, prev_active), base_dist) in enumerate(zip(mesh_positions, mesh_objects, base_distances)):
                # Special case: base_dist = 0 means NO distance gating (always active)
                if base_dist <= 0.0:
                    activation_decisions.append((obj_name, True, prev_active))
                    continue

                mx, my, mz = mesh_pos

                # Calculate squared distance (avoid sqrt)
                dx = mx - px
                dy = my - py
                dz = mz - pz
                dist_squared = dx*dx + dy*dy + dz*dz

                # Hysteresis: avoid activation flapping
                # If previously active, add 10% margin before deactivating
                # If previously inactive, subtract 10% margin before activating
                margin = base_dist * 0.10
                if prev_active:
                    threshold = base_dist + margin
                else:
                    threshold = max(0.0, base_dist - margin)

                # Compare squared distances (no sqrt needed)
                should_activate = (dist_squared <= (threshold * threshold))

                activation_decisions.append((obj_name, should_activate, prev_active))

            calc_end = time.perf_counter()
            calc_time_us = (calc_end - calc_start) * 1_000_000  # microseconds

            result_data = {
                "activation_decisions": activation_decisions,
                "count": len(activation_decisions),
                "calc_time_us": calc_time_us  # Prove worker did the work
            }

        elif job.job_type == "INTERACTION_CHECK_BATCH":
            # Interaction proximity & collision checks
            # Pure math (NO bpy access) - determines which interactions are triggered
            calc_start = time.perf_counter()

            interactions = job.data.get("interactions", [])
            player_position = job.data.get("player_position", (0, 0, 0))

            triggered_indices = []
            px, py, pz = player_position

            for i, inter_data in enumerate(interactions):
                inter_type = inter_data.get("type")

                if inter_type == "PROXIMITY":
                    # Distance check
                    obj_a_pos = inter_data.get("obj_a_pos")
                    obj_b_pos = inter_data.get("obj_b_pos")
                    threshold = inter_data.get("threshold", 0.0)

                    if obj_a_pos and obj_b_pos:
                        ax, ay, az = obj_a_pos
                        bx, by, bz = obj_b_pos
                        dx = ax - bx
                        dy = ay - by
                        dz = az - bz
                        dist_squared = dx*dx + dy*dy + dz*dz
                        threshold_squared = threshold * threshold

                        if dist_squared <= threshold_squared:
                            triggered_indices.append(i)

                elif inter_type == "COLLISION":
                    # AABB overlap check
                    aabb_a = inter_data.get("aabb_a")  # (minx, maxx, miny, maxy, minz, maxz)
                    aabb_b = inter_data.get("aabb_b")
                    margin = inter_data.get("margin", 0.0)

                    if aabb_a and aabb_b:
                        # Unpack AABBs
                        a_minx, a_maxx, a_miny, a_maxy, a_minz, a_maxz = aabb_a
                        b_minx, b_maxx, b_miny, b_maxy, b_minz, b_maxz = aabb_b

                        # Apply margin
                        a_minx -= margin
                        a_maxx += margin
                        a_miny -= margin
                        a_maxy += margin
                        a_minz -= margin
                        a_maxz += margin

                        b_minx -= margin
                        b_maxx += margin
                        b_miny -= margin
                        b_maxy += margin
                        b_minz -= margin
                        b_maxz += margin

                        # Check overlap
                        overlap_x = (a_minx <= b_maxx) and (a_maxx >= b_minx)
                        overlap_y = (a_miny <= b_maxy) and (a_maxy >= b_miny)
                        overlap_z = (a_minz <= b_maxz) and (a_maxz >= b_minz)

                        if overlap_x and overlap_y and overlap_z:
                            triggered_indices.append(i)

            calc_end = time.perf_counter()
            calc_time_us = (calc_end - calc_start) * 1_000_000

            result_data = {
                "triggered_indices": triggered_indices,
                "count": len(interactions),
                "calc_time_us": calc_time_us
            }

        elif job.job_type == "KCC_PHYSICS_STEP":
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
            pos = job.data.get("pos", (0.0, 0.0, 0.0))
            vel = job.data.get("vel", (0.0, 0.0, 0.0))
            on_ground = job.data.get("on_ground", False)
            on_walkable = job.data.get("on_walkable", True)
            ground_normal = job.data.get("ground_normal", (0.0, 0.0, 1.0))

            wish_dir = job.data.get("wish_dir", (0.0, 0.0))
            is_running = job.data.get("is_running", False)
            jump_requested = job.data.get("jump_requested", False)

            coyote_remaining = job.data.get("coyote_remaining", 0.0)
            jump_buffer_remaining = job.data.get("jump_buffer_remaining", 0.0)

            dt = job.data.get("dt", 1.0/30.0)

            # Config
            cfg = job.data.get("config", {})
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
            debug_flags = job.data.get("debug_flags", {})
            debug_step_up = debug_flags.get("step_up", False)
            debug_ground = debug_flags.get("ground", False)
            debug_capsule = debug_flags.get("capsule", False)
            debug_slopes = debug_flags.get("slopes", False)
            debug_slide = debug_flags.get("slide", False)
            debug_enhanced = debug_flags.get("enhanced", False)
            debug_body = debug_flags.get("body_integrity", False)

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
                                worker_logs.append(("PHYS-SLOPES", f"PRE-BLOCK AIRBORNE angle={slope_angle:.0f}°"))

            # ─────────────────────────────────────────────────────────────────
            # 4.9. EXTRACT GRID DATA ONCE (performance optimization)
            # ─────────────────────────────────────────────────────────────────
            # Cache grid data to avoid repeated dictionary lookups (4x per frame)
            grid_bounds_min = None
            grid_bounds_max = None
            grid_cell_size = None
            grid_dims = None
            grid_cells = None
            grid_triangles = None
            grid_min_x = grid_min_y = grid_min_z = 0.0
            grid_max_x = grid_max_y = grid_max_z = 0.0
            grid_nx = grid_ny = grid_nz = 0

            if _cached_grid is not None:
                grid_bounds_min = _cached_grid["bounds_min"]
                grid_bounds_max = _cached_grid["bounds_max"]
                grid_cell_size = _cached_grid["cell_size"]
                grid_dims = _cached_grid["grid_dims"]
                grid_cells = _cached_grid["cells"]
                grid_triangles = _cached_grid["triangles"]

                grid_min_x, grid_min_y, grid_min_z = grid_bounds_min
                grid_max_x, grid_max_y, grid_max_z = grid_bounds_max
                grid_nx, grid_ny, grid_nz = grid_dims

            # ─────────────────────────────────────────────────────────────────
            # 5. HORIZONTAL COLLISION (3D DDA on cached grid)
            # ─────────────────────────────────────────────────────────────────
            move_x = vx * dt
            move_y = vy * dt
            move_len = math.sqrt(move_x*move_x + move_y*move_y)

            # For high-speed movement (running), subdivide into substeps to prevent tunneling
            # This is especially important for steep slopes where rays might miss
            max_step_len = radius * 0.8  # Max movement per substep
            num_substeps = max(1, int(math.ceil(move_len / max_step_len)))
            substep_len = move_len / num_substeps if num_substeps > 0 else 0

            if move_len > 1e-9 and _cached_grid is not None:
                # Normalize movement direction
                fwd_x = move_x / move_len
                fwd_y = move_y / move_len

                # Track total movement allowed across all substeps
                total_allowed = 0.0
                blocked_this_frame = False
                final_wall_normal = None

                # Use pre-extracted grid data
                bounds_min = grid_bounds_min
                bounds_max = grid_bounds_max
                cell_size = grid_cell_size
                cells = grid_cells
                triangles = grid_triangles

                min_x, min_y, min_z = grid_min_x, grid_min_y, grid_min_z
                max_x, max_y, max_z = grid_max_x, grid_max_y, grid_max_z
                nx_grid, ny_grid, nz_grid = grid_nx, grid_ny, grid_nz

                # Cast rays at 3 heights (feet, mid, head) + angled rays for steep slopes
                # Feet ray at 0.1m to catch low overhangs and crawl spaces
                ray_heights = [0.1, min(height * 0.5, height - radius), height - radius]
                ray_len = move_len + radius

                # Additional slope detection rays (angled slightly down to catch slopes)
                # These are critical for detecting steep slopes when running
                slope_ray_configs = []
                if on_ground:
                    # Cast rays angled downward at 30 degrees to catch slopes
                    slope_angle = 0.5  # ~30 degrees down
                    slope_ray_z = radius * 2  # Knee height
                    slope_fwd_len = math.sqrt(1.0 / (1.0 + slope_angle * slope_angle))
                    slope_down_len = slope_angle * slope_fwd_len
                    slope_ray_configs.append((slope_ray_z, slope_fwd_len, -slope_down_len))

                best_d = None
                best_n = None
                per_ray_hits = []

                # WIDTH CHECK: Add perpendicular rays to detect narrow gaps
                # Calculate perpendicular direction (90° from forward)
                # This prevents squeezing through gaps narrower than capsule diameter
                perp_x = -fwd_y  # Perpendicular left
                perp_y = fwd_x

                # Width ray positions at mid-height (most critical for gap detection)
                mid_height = height * 0.5
                width_ray_configs = [
                    (perp_x * radius, perp_y * radius),   # Left edge
                    (-perp_x * radius, -perp_y * radius), # Right edge
                ]

                # First, do horizontal rays (forward detection)
                for ray_z in ray_heights:
                    total_rays += 1
                    ox, oy, oz = px, py, pz + ray_z

                    # 3D DDA traversal
                    start_x = max(min_x, min(max_x - 0.001, ox))
                    start_y = max(min_y, min(max_y - 0.001, oy))
                    start_z = max(min_z, min(max_z - 0.001, oz))

                    ix = int((start_x - min_x) / cell_size)
                    iy = int((start_y - min_y) / cell_size)
                    iz = int((start_z - min_z) / cell_size)

                    ix = max(0, min(nx_grid - 1, ix))
                    iy = max(0, min(ny_grid - 1, iy))
                    iz = max(0, min(nz_grid - 1, iz))

                    step_x = 1 if fwd_x >= 0 else -1
                    step_y = 1 if fwd_y >= 0 else -1
                    step_z = 0  # Horizontal only

                    INF = float('inf')

                    if abs(fwd_x) > 1e-12:
                        if fwd_x > 0:
                            t_max_x = ((min_x + (ix + 1) * cell_size) - ox) / fwd_x
                        else:
                            t_max_x = ((min_x + ix * cell_size) - ox) / fwd_x
                        t_delta_x = abs(cell_size / fwd_x)
                    else:
                        t_max_x = INF
                        t_delta_x = INF

                    if abs(fwd_y) > 1e-12:
                        if fwd_y > 0:
                            t_max_y = ((min_y + (iy + 1) * cell_size) - oy) / fwd_y
                        else:
                            t_max_y = ((min_y + iy * cell_size) - oy) / fwd_y
                        t_delta_y = abs(cell_size / fwd_y)
                    else:
                        t_max_y = INF
                        t_delta_y = INF

                    ray_closest_dist = ray_len
                    ray_closest_tri = None
                    tested_triangles = set()
                    cells_traversed = 0
                    max_cells = nx_grid + ny_grid + 10
                    t_current = 0.0

                    while cells_traversed < max_cells:
                        cells_traversed += 1
                        total_cells += 1

                        if t_current > ray_len:
                            break
                        if ix < 0 or ix >= nx_grid or iy < 0 or iy >= ny_grid or iz < 0 or iz >= nz_grid:
                            break

                        cell_key = (ix, iy, iz)
                        if cell_key in cells:
                            for tri_idx in cells[cell_key]:
                                if tri_idx in tested_triangles:
                                    continue
                                tested_triangles.add(tri_idx)
                                total_tris += 1

                                tri = triangles[tri_idx]
                                hit, dist, _ = ray_triangle_intersect((ox, oy, oz), (fwd_x, fwd_y, 0), tri[0], tri[1], tri[2])
                                if hit and dist < ray_closest_dist:
                                    ray_closest_dist = dist
                                    ray_closest_tri = tri

                        t_next = min(t_max_x, t_max_y)
                        if ray_closest_tri is not None and ray_closest_dist <= t_next:
                            break

                        if t_max_x < t_max_y:
                            ix += step_x
                            t_current = t_max_x
                            t_max_x += t_delta_x
                        else:
                            iy += step_y
                            t_current = t_max_y
                            t_max_y += t_delta_y

                    # Track per-ray hits
                    if ray_closest_tri is not None:
                        per_ray_hits.append(ray_closest_dist)
                        if best_d is None or ray_closest_dist < best_d:
                            best_d = ray_closest_dist
                            best_n = compute_triangle_normal(ray_closest_tri[0], ray_closest_tri[1], ray_closest_tri[2])
                    else:
                        per_ray_hits.append(None)

                # WIDTH RAYS: Check left/right edges at mid-height
                # Cast 2 rays perpendicular to movement to detect narrow gaps
                # This prevents squeezing through openings narrower than capsule diameter
                width_hits = []  # Store hits from both width rays
                for width_offset_x, width_offset_y in width_ray_configs:
                    total_rays += 1

                    # Ray origin offset to left/right edge at mid-height
                    ox = px + width_offset_x
                    oy = py + width_offset_y
                    oz = pz + mid_height

                    # Cast ray forward in movement direction (same as center rays)
                    # Using simplified 3-cell check (cheap, like slope rays)
                    start_ix = int((ox - min_x) / cell_size)
                    start_iy = int((oy - min_y) / cell_size)
                    start_iz = int((oz - min_z) / cell_size)
                    start_ix = max(0, min(nx_grid - 1, start_ix))
                    start_iy = max(0, min(ny_grid - 1, start_iy))
                    start_iz = max(0, min(nz_grid - 1, start_iz))

                    width_hit_dist = None
                    width_hit_normal = None

                    tested_width = set()
                    for depth in range(3):  # Check 3 cells forward (like slope rays)
                        check_x = ox + fwd_x * cell_size * (depth + 1)
                        check_y = oy + fwd_y * cell_size * (depth + 1)
                        check_z = oz

                        cix = int((check_x - min_x) / cell_size)
                        ciy = int((check_y - min_y) / cell_size)
                        ciz = int((check_z - min_z) / cell_size)
                        cix = max(0, min(nx_grid - 1, cix))
                        ciy = max(0, min(ny_grid - 1, ciy))
                        ciz = max(0, min(nz_grid - 1, ciz))

                        cell_key = (cix, ciy, ciz)
                        if cell_key in cells:
                            for tri_idx in cells[cell_key]:
                                if tri_idx in tested_width:
                                    continue
                                tested_width.add(tri_idx)
                                total_tris += 1

                                tri = triangles[tri_idx]
                                hit, dist, _ = ray_triangle_intersect(
                                    (ox, oy, oz), (fwd_x, fwd_y, 0),
                                    tri[0], tri[1], tri[2]
                                )
                                if hit and dist < ray_len:
                                    if width_hit_dist is None or dist < width_hit_dist:
                                        width_hit_dist = dist
                                        width_hit_normal = compute_triangle_normal(tri[0], tri[1], tri[2])

                    width_hits.append((width_hit_dist, width_hit_normal))

                # Only block if BOTH width rays hit (narrow gap detection)
                # Don't interfere with normal wall sliding where only one edge hits
                if len(width_hits) == 2:
                    left_dist, left_n = width_hits[0]
                    right_dist, right_n = width_hits[1]

                    if left_dist is not None and right_dist is not None:
                        # Both edges hit - this is a narrow gap
                        # Use the closer hit to block movement
                        closest_dist = min(left_dist, right_dist)
                        if best_d is None or closest_dist < best_d:
                            best_d = closest_dist
                            best_n = left_n if left_dist < right_dist else right_n

                # Second, do angled slope detection rays (only when grounded)
                # These catch steep slopes that horizontal rays might miss
                for slope_cfg in slope_ray_configs:
                    slope_ray_z, slope_fwd_mult, slope_z_mult = slope_cfg
                    total_rays += 1

                    # Ray direction is forward + slightly down
                    ray_dx = fwd_x * slope_fwd_mult
                    ray_dy = fwd_y * slope_fwd_mult
                    ray_dz = slope_z_mult  # Negative = downward

                    ox, oy, oz = px, py, pz + slope_ray_z

                    # Simple brute-force check in nearby cells (cheaper than full DDA for one ray)
                    start_ix = int((ox - min_x) / cell_size)
                    start_iy = int((oy - min_y) / cell_size)
                    start_iz = int((oz - min_z) / cell_size)
                    start_ix = max(0, min(nx_grid - 1, start_ix))
                    start_iy = max(0, min(ny_grid - 1, start_iy))
                    start_iz = max(0, min(nz_grid - 1, start_iz))

                    # Check cells in forward direction (3 cells deep)
                    tested_slope = set()
                    for depth in range(3):
                        check_x = ox + ray_dx * cell_size * (depth + 1)
                        check_y = oy + ray_dy * cell_size * (depth + 1)
                        check_z = oz + ray_dz * cell_size * (depth + 1)

                        cix = int((check_x - min_x) / cell_size)
                        ciy = int((check_y - min_y) / cell_size)
                        ciz = int((check_z - min_z) / cell_size)
                        cix = max(0, min(nx_grid - 1, cix))
                        ciy = max(0, min(ny_grid - 1, ciy))
                        ciz = max(0, min(nz_grid - 1, ciz))

                        cell_key = (cix, ciy, ciz)
                        if cell_key in cells:
                            for tri_idx in cells[cell_key]:
                                if tri_idx in tested_slope:
                                    continue
                                tested_slope.add(tri_idx)
                                total_tris += 1

                                tri = triangles[tri_idx]
                                hit, dist, _ = ray_triangle_intersect(
                                    (ox, oy, oz), (ray_dx, ray_dy, ray_dz),
                                    tri[0], tri[1], tri[2]
                                )
                                if hit and dist < ray_len:
                                    tri_n = compute_triangle_normal(tri[0], tri[1], tri[2])
                                    # Only count steep slopes (not floors)
                                    if tri_n[2] < floor_cos and tri_n[2] > 0.1:
                                        # This is a steep slope - treat as wall
                                        # Convert angled ray hit to equivalent horizontal distance
                                        horiz_dist = dist * slope_fwd_mult
                                        if best_d is None or horiz_dist < best_d:
                                            best_d = horiz_dist
                                            best_n = tri_n

                # Apply horizontal collision result
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
                    if debug_capsule:
                        if best_n is not None:
                            worker_logs.append(("PHYS-CAPSULE", f"blocked dist={best_d:.3f}m allowed={allowed:.3f}m normal=({best_n[0]:.2f},{best_n[1]:.2f},{best_n[2]:.2f}) | {total_rays}rays {total_tris}tris"))
                        else:
                            worker_logs.append(("PHYS-CAPSULE", f"blocked dist={best_d:.3f}m allowed={allowed:.3f}m normal=None | {total_rays}rays {total_tris}tris"))

                    # ─────────────────────────────────────────────────────────
                    # 5a. STEP-UP (if only feet ray hit)
                    # ─────────────────────────────────────────────────────────
                    feet_only = (len(per_ray_hits) >= 3 and
                                per_ray_hits[0] is not None and
                                per_ray_hits[1] is None and
                                per_ray_hits[2] is None)

                    # Only allow step-up when on walkable ground (prevent step-up on steep slopes)
                    if on_ground and on_walkable and feet_only and step_height > 0.0 and best_n is not None:
                        # Check if it's a steep face (step-able)
                        bn_x, bn_y, bn_z = best_n
                        if bn_z < floor_cos:  # Steep face
                            if debug_step_up:
                                worker_logs.append(("PHYS-STEP", f"attempting step-up | pos=({px:.2f},{py:.2f},{pz:.2f}) face_nz={bn_z:.3f}"))
                            # Try step-up: raise, forward, drop
                            test_z = pz + step_height
                            test_ox = px
                            test_oy = py

                            # Check headroom using grid (not brute force)
                            head_clear = True
                            head_ray_z = test_z + height

                            # Grid-based ceiling check at raised position
                            check_ix = int((test_ox - min_x) / cell_size)
                            check_iy = int((test_oy - min_y) / cell_size)
                            check_iz = int((head_ray_z - min_z) / cell_size)
                            check_ix = max(0, min(nx_grid - 1, check_ix))
                            check_iy = max(0, min(ny_grid - 1, check_iy))
                            check_iz = max(0, min(nz_grid - 1, check_iz))

                            # Check a few cells above for ceiling
                            tested_ceil = set()
                            for dz_off in range(3):
                                ceil_iz = min(nz_grid - 1, check_iz + dz_off)
                                ceil_key = (check_ix, check_iy, ceil_iz)
                                if ceil_key in cells:
                                    for tri_idx in cells[ceil_key]:
                                        if tri_idx in tested_ceil:
                                            continue
                                        tested_ceil.add(tri_idx)
                                        tri = triangles[tri_idx]
                                        hit, dist, _ = ray_triangle_intersect(
                                            (test_ox, test_oy, head_ray_z),
                                            (0, 0, 1),
                                            tri[0], tri[1], tri[2]
                                        )
                                        if hit and dist < step_height:
                                            head_clear = False
                                            break
                                if not head_clear:
                                    break

                            if head_clear:
                                # Cast forward ray at raised position
                                remaining_move = move_len - allowed
                                if remaining_move > 0.01:
                                    # Simple forward check at raised height (abbreviated)
                                    can_step = True
                                    if can_step:
                                        # Drop down to find ground
                                        drop_ox = test_ox + fwd_x * min(remaining_move, radius)
                                        drop_oy = test_oy + fwd_y * min(remaining_move, radius)
                                        drop_max = step_height + snap_down

                                        # Grid-based ground raycast at dropped position
                                        step_ground_z = None
                                        step_ground_n = None

                                        # Find cell for drop position
                                        drop_ix = int((drop_ox - min_x) / cell_size)
                                        drop_iy = int((drop_oy - min_y) / cell_size)
                                        drop_iz = int((test_z + 1.0 - min_z) / cell_size)
                                        drop_ix = max(0, min(nx_grid - 1, drop_ix))
                                        drop_iy = max(0, min(ny_grid - 1, drop_iy))
                                        drop_iz = max(0, min(nz_grid - 1, drop_iz))

                                        # Search downward through cells
                                        tested_step = set()
                                        for dz_off in range(min(10, nz_grid)):
                                            step_iz = drop_iz - dz_off
                                            if step_iz < 0:
                                                break
                                            step_key = (drop_ix, drop_iy, step_iz)
                                            if step_key in cells:
                                                for tri_idx in cells[step_key]:
                                                    if tri_idx in tested_step:
                                                        continue
                                                    tested_step.add(tri_idx)
                                                    total_tris += 1
                                                    tri = triangles[tri_idx]
                                                    hit, dist, hp = ray_triangle_intersect(
                                                        (drop_ox, drop_oy, test_z + 1.0),
                                                        (0, 0, -1),
                                                        tri[0], tri[1], tri[2]
                                                    )
                                                    if hit and dist < drop_max + 1.0:
                                                        hit_z = test_z + 1.0 - dist
                                                        if step_ground_z is None or hit_z > step_ground_z:
                                                            step_ground_z = hit_z
                                                            step_ground_n = compute_triangle_normal(tri[0], tri[1], tri[2])

                                        if step_ground_z is not None and step_ground_n is not None:
                                            # Check if walkable
                                            gn_check = step_ground_n[2]
                                            if gn_check >= floor_cos:
                                                if debug_step_up:
                                                    worker_logs.append(("PHYS-STEP", f"SUCCESS | drop_pos=({drop_ox:.2f},{drop_oy:.2f},{step_ground_z:.2f}) gn_z={gn_check:.3f}"))
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
                    # 5b. WALL SLIDE (if blocked and not step-up)
                    # ─────────────────────────────────────────────────────────
                    if not did_step_up and best_n is not None:
                        remaining = move_len - allowed
                        if remaining > 0.001:
                            bn_x, bn_y, bn_z = best_n

                            # Check if this is a steep slope we're trying to climb
                            # If hitting a steep face (bn_z < floor_cos) and we're grounded,
                            # don't allow wall slide that would move us up the slope
                            is_steep_face = bn_z < floor_cos and bn_z > 0.1  # Not a ceiling

                            # Compute slide direction (tangent to wall)
                            dot = fwd_x * bn_x + fwd_y * bn_y

                            slide_x = fwd_x - bn_x * dot
                            slide_y = fwd_y - bn_y * dot
                            slide_len = math.sqrt(slide_x*slide_x + slide_y*slide_y)

                            # Don't slide if moving nearly head-on into wall (within ~15° of perpendicular)
                            # sin(15°) ≈ 0.259 - if tangent component is smaller, we're too perpendicular
                            if slide_len < 0.259:
                                slide_len = 0.0  # Prevent slide for head-on collisions

                            # For steep slopes: check if slide would move us "up" the slope
                            # We detect this by checking if the slide direction aligns with
                            # the uphill direction (opposite of normal's XY projection)
                            if is_steep_face and on_ground and slide_len > 1e-9:
                                # Uphill direction in XY
                                uphill_xy_len = math.sqrt(bn_x*bn_x + bn_y*bn_y)
                                if uphill_xy_len > 0.001:
                                    uphill_x = bn_x / uphill_xy_len
                                    uphill_y = bn_y / uphill_xy_len

                                    # How much is slide aligned with uphill?
                                    slide_nx = slide_x / slide_len
                                    slide_ny = slide_y / slide_len
                                    uphill_dot = slide_nx * uphill_x + slide_ny * uphill_y

                                    # If slide is uphill at all (dot > -0.1), block it
                                    # Changed from 0.3 to -0.1 for much stricter blocking
                                    # This prevents nearly all uphill sliding on steep slopes
                                    if uphill_dot > -0.1:
                                        slide_len = 0.0  # Prevent slide

                            if slide_len > 1e-9:
                                slide_x /= slide_len
                                slide_y /= slide_len
                                slide_dist = remaining * 0.65

                                # Collision check for slide movement
                                # Cast a single ray in slide direction to prevent corner clipping
                                slide_blocked = False
                                slide_allowed = slide_dist

                                slide_ox, slide_oy = px, py
                                slide_oz = pz + height * 0.5  # Mid-height check

                                # Quick cell-based check in slide direction
                                slide_ix = int((slide_ox - min_x) / cell_size)
                                slide_iy = int((slide_oy - min_y) / cell_size)
                                slide_iz = int((slide_oz - min_z) / cell_size)
                                slide_ix = max(0, min(nx_grid - 1, slide_ix))
                                slide_iy = max(0, min(ny_grid - 1, slide_iy))
                                slide_iz = max(0, min(nz_grid - 1, slide_iz))

                                # Check current cell and adjacent cells in slide direction
                                tested_slide = set()
                                for dx_off in range(-1, 2):
                                    for dy_off in range(-1, 2):
                                        check_ix = slide_ix + dx_off
                                        check_iy = slide_iy + dy_off
                                        if check_ix < 0 or check_ix >= nx_grid or check_iy < 0 or check_iy >= ny_grid:
                                            continue
                                        cell_key = (check_ix, check_iy, slide_iz)
                                        if cell_key in cells:
                                            for tri_idx in cells[cell_key]:
                                                if tri_idx in tested_slide:
                                                    continue
                                                tested_slide.add(tri_idx)
                                                tri = triangles[tri_idx]
                                                hit, dist, _ = ray_triangle_intersect(
                                                    (slide_ox, slide_oy, slide_oz),
                                                    (slide_x, slide_y, 0),
                                                    tri[0], tri[1], tri[2]
                                                )
                                                if hit and dist < slide_dist + radius:
                                                    slide_allowed = min(slide_allowed, max(0, dist - radius))
                                                    slide_blocked = True

                                # Apply slide with collision result
                                if slide_allowed > 0.01:
                                    px += slide_x * slide_allowed
                                    py += slide_y * slide_allowed
                                    did_slide = True

                                # Reduce velocity
                                vel_after_factor = 0.65
                                vx *= vel_after_factor
                                vy *= vel_after_factor

                                # SLIDE DIAGNOSTICS (logging only - after slide is applied)
                                if debug_slide:
                                    try:
                                        slide_applied = slide_allowed if slide_allowed > 0.01 else 0.0
                                        slide_requested = slide_dist
                                        effectiveness = (slide_applied / slide_requested * 100.0) if slide_requested > 0.001 else 0.0
                                        worker_logs.append(("PHYS-SLIDE",
                                            f"applied={slide_applied:.3f}m requested={slide_requested:.3f}m eff={effectiveness:.0f}% "
                                            f"normal=({bn_x:.2f},{bn_y:.2f},{bn_z:.2f}) blocked={slide_blocked}"))
                                    except:
                                        pass  # Silently ignore any logging errors
                else:
                    # No collision - full movement
                    px += move_x
                    py += move_y
                    if debug_capsule and move_len > 1e-9:
                        worker_logs.append(("PHYS-CAPSULE", f"clear move={move_len:.3f}m | {total_rays}rays {total_tris}tris"))

            elif move_len > 1e-9:
                # No grid cached - just move (TODO: add dynamic mesh cache to worker)
                px += move_x
                py += move_y

            # ─────────────────────────────────────────────────────────────────
            # 5.5 VERTICAL BODY INTEGRITY CHECK (detect mesh embedding) - Using DDA
            # ─────────────────────────────────────────────────────────────────
            # Cast vertical ray from feet to head - if blocked, character is embedded in mesh
            body_embedded = False
            embed_distance = None

            if _cached_grid is not None:
                # Feet at 0.1m to match horizontal ray height (better low-obstacle detection)
                feet_pos = (px, py, pz + 0.1)
                head_pos = (px, py, pz + height - radius)
                body_height = (height - radius) - 0.1  # Distance from feet (0.1m) to head

                # Use pre-extracted grid data (performance optimization)
                triangles = grid_triangles
                cells = grid_cells
                min_x, min_y, min_z = grid_min_x, grid_min_y, grid_min_z
                max_x, max_y, max_z = grid_max_x, grid_max_y, grid_max_z
                nx_grid, ny_grid, nz_grid = grid_nx, grid_ny, grid_nz
                cell_size = grid_cell_size

                # Vertical DDA traversal from feet to head
                ray_origin = feet_pos
                ray_dir = (0, 0, 1)

                start_x = max(min_x, min(max_x - 0.001, px))
                start_y = max(min_y, min(max_y - 0.001, py))
                start_z = max(min_z, min(max_z - 0.001, feet_pos[2]))

                ix = int((start_x - min_x) / cell_size)
                iy = int((start_y - min_y) / cell_size)
                iz = int((start_z - min_z) / cell_size)

                ix = max(0, min(nx_grid - 1, ix))
                iy = max(0, min(ny_grid - 1, iy))
                iz = max(0, min(nz_grid - 1, iz))

                # Z traversal upward
                t_max_z = ((min_z + (iz + 1) * cell_size) - feet_pos[2]) / 1.0
                t_delta_z = abs(cell_size / 1.0)

                tested_body = set()
                cells_traversed = 0
                max_cells = nz_grid + 10
                t_current = 0.0
                total_rays += 1

                while cells_traversed < max_cells and t_current < body_height:
                    cells_traversed += 1
                    total_cells += 1

                    if iz < 0 or iz >= nz_grid:
                        break

                    cell_key = (ix, iy, iz)
                    if cell_key in cells:
                        for tri_idx in cells[cell_key]:
                            if tri_idx in tested_body:
                                continue
                            tested_body.add(tri_idx)
                            total_tris += 1

                            tri = triangles[tri_idx]
                            hit, dist, _ = ray_triangle_intersect(ray_origin, ray_dir, tri[0], tri[1], tri[2])
                            if hit and dist < body_height:
                                body_embedded = True
                                embed_distance = dist

                                if debug_body:
                                    penetration_pct = (dist / body_height) * 100.0
                                    from_feet = dist
                                    to_head = body_height - dist
                                    worker_logs.append(("PHYS-BODY", f"EMBEDDED! hit={dist:.3f}m pct={penetration_pct:.1f}% feet={from_feet:.3f}m head={to_head:.3f}m z=[{feet_pos[2]:.2f},{head_pos[2]:.2f}]"))
                                break

                    if body_embedded:
                        break

                    # Move to next cell upward
                    iz += 1
                    t_current = t_max_z
                    t_max_z += t_delta_z

                # Single combined log after all checks (avoids frequency gating split)
                if debug_body:
                    status = "EMBEDDED" if body_embedded else "CLEAR"
                    worker_logs.append(("PHYS-BODY", f"[{status}] feet=({feet_pos[0]:.2f},{feet_pos[1]:.2f},{feet_pos[2]:.2f}) head=({head_pos[0]:.2f},{head_pos[1]:.2f},{head_pos[2]:.2f}) h={body_height:.2f}m"))

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
                        worker_logs.append(("PHYS-BODY", f"PREVENT-FALL corrected={correction:.3f}m landing on embedded mesh"))

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
                        worker_logs.append(("PHYS-BODY", f"CORRECT-SIDE corrected={correction:.3f}m pushed up from embedded mesh"))

            # ─────────────────────────────────────────────────────────────────
            # 6. CEILING CHECK (if moving up) - Using DDA spatial grid
            # ─────────────────────────────────────────────────────────────────
            if vz > 0.0 and _cached_grid is not None:
                up_dist = vz * dt
                head_z = pz + height

                # Use pre-extracted grid data (performance optimization)
                cells = grid_cells
                triangles = grid_triangles
                cell_size = grid_cell_size
                min_x, min_y, min_z = grid_min_x, grid_min_y, grid_min_z
                max_x, max_y, max_z = grid_max_x, grid_max_y, grid_max_z
                nx_grid, ny_grid, nz_grid = grid_nx, grid_ny, grid_nz

                # Vertical DDA traversal upward from head
                ray_origin = (px, py, head_z)
                ray_dir = (0, 0, 1)

                start_x = max(min_x, min(max_x - 0.001, px))
                start_y = max(min_y, min(max_y - 0.001, py))
                start_z = max(min_z, min(max_z - 0.001, head_z))

                ix = int((start_x - min_x) / cell_size)
                iy = int((start_y - min_y) / cell_size)
                iz = int((start_z - min_z) / cell_size)

                ix = max(0, min(nx_grid - 1, ix))
                iy = max(0, min(ny_grid - 1, iy))
                iz = max(0, min(nz_grid - 1, iz))

                # Z traversal upward
                INF = float('inf')
                t_max_z = ((min_z + (iz + 1) * cell_size) - head_z) / 1.0
                t_delta_z = abs(cell_size / 1.0)

                tested_ceiling = set()
                cells_traversed = 0
                max_cells = nz_grid + 10
                t_current = 0.0
                total_rays += 1

                ceiling_hit = False

                while cells_traversed < max_cells and t_current < up_dist:
                    cells_traversed += 1
                    total_cells += 1

                    if iz < 0 or iz >= nz_grid:
                        break

                    cell_key = (ix, iy, iz)
                    if cell_key in cells:
                        for tri_idx in cells[cell_key]:
                            if tri_idx in tested_ceiling:
                                continue
                            tested_ceiling.add(tri_idx)
                            total_tris += 1

                            tri = triangles[tri_idx]
                            hit, dist, _ = ray_triangle_intersect(
                                ray_origin, ray_dir,
                                tri[0], tri[1], tri[2]
                            )
                            if hit and dist < up_dist:
                                pz = head_z + dist - height
                                vz = 0.0
                                hit_ceiling = True
                                ceiling_hit = True
                                break

                    if ceiling_hit:
                        break

                    # Move to next cell upward
                    iz += 1
                    t_current = t_max_z
                    t_max_z += t_delta_z

            # ─────────────────────────────────────────────────────────────────
            # 7. VERTICAL MOVEMENT + GROUND DETECTION
            # ─────────────────────────────────────────────────────────────────
            dz = vz * dt
            was_grounded = on_ground

            if _cached_grid is not None:
                # Use pre-extracted grid data (performance optimization)
                triangles = grid_triangles
                cells = grid_cells
                cell_size = grid_cell_size
                min_x, min_y, min_z = grid_min_x, grid_min_y, grid_min_z
                max_x, max_y, max_z = grid_max_x, grid_max_y, grid_max_z
                nx_grid, ny_grid, nz_grid = grid_nx, grid_ny, grid_nz

                # Raycast down for ground
                ray_start_z = pz + 1.0  # Guard above
                ray_max = snap_down + 1.0 + (abs(dz) if dz < 0 else 0)

                ground_hit_z = None
                ground_hit_n = None

                start_x = max(min_x, min(max_x - 0.001, px))
                start_y = max(min_y, min(max_y - 0.001, py))
                start_z = max(min_z, min(max_z - 0.001, ray_start_z))

                ix = int((start_x - min_x) / cell_size)
                iy = int((start_y - min_y) / cell_size)
                iz = int((start_z - min_z) / cell_size)

                ix = max(0, min(nx_grid - 1, ix))
                iy = max(0, min(ny_grid - 1, iy))
                iz = max(0, min(nz_grid - 1, iz))

                INF = float('inf')
                t_max_z = ((min_z + iz * cell_size) - ray_start_z) / (-1.0) if True else INF
                t_delta_z = abs(cell_size / 1.0)

                tested_triangles = set()
                cells_traversed = 0
                max_cells = nz_grid + 10
                t_current = 0.0
                total_rays += 1

                while cells_traversed < max_cells and t_current < ray_max:
                    cells_traversed += 1
                    total_cells += 1

                    if iz < 0 or iz >= nz_grid:
                        break

                    cell_key = (ix, iy, iz)
                    if cell_key in cells:
                        for tri_idx in cells[cell_key]:
                            if tri_idx in tested_triangles:
                                continue
                            tested_triangles.add(tri_idx)
                            total_tris += 1

                            tri = triangles[tri_idx]
                            hit, dist, hp = ray_triangle_intersect(
                                (px, py, ray_start_z), (0, 0, -1),
                                tri[0], tri[1], tri[2]
                            )
                            if hit and dist < ray_max:
                                hit_z = ray_start_z - dist
                                if ground_hit_z is None or hit_z > ground_hit_z:
                                    ground_hit_z = hit_z
                                    ground_hit_n = compute_triangle_normal(tri[0], tri[1], tri[2])

                    iz -= 1
                    t_current = t_max_z
                    t_max_z += t_delta_z

                    if ground_hit_z is not None:
                        break

                # Apply vertical movement and ground snap
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
                        worker_logs.append(("PHYS-GROUND", f"ON_GROUND snap | dist={abs(ground_hit_z - pz):.3f}m gn_z={gn_z:.3f} walkable={on_walkable}"))
                elif ground_hit_z is None and was_grounded:
                    # Was grounded but no ground found - walked off a ledge!
                    on_ground = False
                    on_walkable = False
                    coyote_remaining = coyote_time  # Grant coyote time
                    if debug_ground:
                        worker_logs.append(("PHYS-GROUND", f"airborne | walked_off_ledge coyote={coyote_time:.2f}s"))
                elif not on_ground and was_grounded:
                    coyote_remaining = coyote_time

                if debug_ground and ground_hit_z is not None and not on_ground:
                    worker_logs.append(("PHYS-GROUND", f"airborne | dist={abs(ground_hit_z - pz):.3f}m too_far | {total_tris}tris"))
            else:
                # No grid - just apply vertical movement
                pz += dz
                on_ground = False

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
                                worker_logs.append(("PHYS-SLOPES", f"POST-CORRECT angle={slope_angle:.0f}° (backup)"))

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
                                    worker_logs.append(("PHYS-SLOPES", f"Z-CLAMP angle={slope_angle:.0f}° prevented {pz - max_allowed_z:.3f}m upward movement"))
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
                                worker_logs.append(("PHYS-SLOPES", f"GRAVITY-SLIDE angle={slope_angle:.0f}° normal=({n_x:.2f},{n_y:.2f},{n_z:.2f}) "
                                                                   f"dir=({slide_x:.2f},{slide_y:.2f}) "
                                                                   f"vel_downhill={current_downhill_speed:.2f} max={max_slide_speed:.2f} "
                                                                   f"accel={slide_accel:.2f} steep={steepness:.2f} track={surface_tracking}"))

            # ─────────────────────────────────────────────────────────────────
            # BUILD RESULT
            # ─────────────────────────────────────────────────────────────────
            calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

            # Summary debug log
            if debug_enhanced:
                worker_logs.append(("KCC", f"COMPLETE pos=({px:.2f},{py:.2f},{pz:.2f}) vel=({vx:.2f},{vy:.2f},{vz:.2f}) ground={on_ground} walkable={on_walkable} blocked={h_blocked} step={did_step_up} slide={did_slide} | {calc_time_us:.0f}us {total_rays}rays {total_tris}tris"))

            result_data = {
                "pos": (px, py, pz),
                "vel": (vx, vy, vz),
                "on_ground": on_ground,
                "on_walkable": on_walkable,
                "ground_normal": (gn_x, gn_y, gn_z),
                "coyote_remaining": coyote_remaining,
                "jump_consumed": jump_consumed,
                "logs": worker_logs,  # Fast buffer logs (sent to main thread)
                "debug": {
                    "rays_cast": total_rays,
                    "triangles_tested": total_tris,
                    "cells_traversed": total_cells,
                    "calc_time_us": calc_time_us,
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

        # =====================================================================
        # CAMERA OCCLUSION HANDLERS
        # =====================================================================

        # REMOVED: Old KCC handlers (KCC_INPUT_VECTOR, KCC_RAYCAST, KCC_RAYCAST_GRID, KCC_RAYCAST_CACHED)
        # Now using unified KCC_PHYSICS_STEP handler above

        elif job.job_type == "CAMERA_OCCLUSION_FULL":
            # Full camera occlusion: static grid + dynamic triangles
            # Returns closest hit - LoS+Pushout done on main thread (Blender BVH is faster)
            import math

            calc_start = time.perf_counter()

            # Ray parameters
            ray_origin = job.data.get("ray_origin", (0.0, 0.0, 0.0))
            ray_direction = job.data.get("ray_direction", (0.0, 0.0, -1.0))
            max_distance = job.data.get("max_distance", 10.0)
            dynamic_triangles = job.data.get("dynamic_triangles", [])

            ox, oy, oz = ray_origin
            dx, dy, dz = ray_direction

            # Normalize direction
            d_len = math.sqrt(dx*dx + dy*dy + dz*dz)
            if d_len > 1e-12:
                dx /= d_len
                dy /= d_len
                dz /= d_len

            closest_dist = max_distance
            hit_found = False
            hit_source = None
            static_tris_tested = 0
            static_cells_traversed = 0
            dynamic_tris_tested = 0

            # === STATIC GEOMETRY (cached grid) ===
            if _cached_grid is not None:
                grid = _cached_grid
                bounds_min = grid["bounds_min"]
                bounds_max = grid["bounds_max"]
                cell_size = grid["cell_size"]
                grid_dims = grid["grid_dims"]
                cells = grid["cells"]
                triangles = grid["triangles"]

                nx, ny, nz = grid_dims
                min_x, min_y, min_z = bounds_min
                max_x, max_y, max_z = bounds_max

                start_x = max(min_x, min(max_x - 0.001, ox))
                start_y = max(min_y, min(max_y - 0.001, oy))
                start_z = max(min_z, min(max_z - 0.001, oz))

                ix = int((start_x - min_x) / cell_size)
                iy = int((start_y - min_y) / cell_size)
                iz = int((start_z - min_z) / cell_size)

                ix = max(0, min(nx - 1, ix))
                iy = max(0, min(ny - 1, iy))
                iz = max(0, min(nz - 1, iz))

                step_x = 1 if dx >= 0 else -1
                step_y = 1 if dy >= 0 else -1
                step_z = 1 if dz >= 0 else -1

                INF = float('inf')

                if abs(dx) > 1e-12:
                    if dx > 0:
                        t_max_x = ((min_x + (ix + 1) * cell_size) - ox) / dx
                    else:
                        t_max_x = ((min_x + ix * cell_size) - ox) / dx
                    t_delta_x = abs(cell_size / dx)
                else:
                    t_max_x = INF
                    t_delta_x = INF

                if abs(dy) > 1e-12:
                    if dy > 0:
                        t_max_y = ((min_y + (iy + 1) * cell_size) - oy) / dy
                    else:
                        t_max_y = ((min_y + iy * cell_size) - oy) / dy
                    t_delta_y = abs(cell_size / dy)
                else:
                    t_max_y = INF
                    t_delta_y = INF

                if abs(dz) > 1e-12:
                    if dz > 0:
                        t_max_z = ((min_z + (iz + 1) * cell_size) - oz) / dz
                    else:
                        t_max_z = ((min_z + iz * cell_size) - oz) / dz
                    t_delta_z = abs(cell_size / dz)
                else:
                    t_max_z = INF
                    t_delta_z = INF

                tested_triangles = set()
                max_cells = nx + ny + nz + 10
                t_current = 0.0

                while static_cells_traversed < max_cells:
                    static_cells_traversed += 1

                    if t_current > closest_dist:
                        break
                    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                        break

                    cell_key = (ix, iy, iz)
                    if cell_key in cells:
                        for tri_idx in cells[cell_key]:
                            if tri_idx in tested_triangles:
                                continue
                            tested_triangles.add(tri_idx)
                            static_tris_tested += 1

                            tri = triangles[tri_idx]
                            hit, dist, _ = ray_triangle_intersect(ray_origin, (dx, dy, dz), tri[0], tri[1], tri[2])
                            if hit and dist < closest_dist:
                                closest_dist = dist
                                hit_found = True
                                hit_source = "STATIC"

                    t_next = min(t_max_x, t_max_y, t_max_z)
                    if hit_found and closest_dist <= t_next:
                        break

                    if t_max_x < t_max_y:
                        if t_max_x < t_max_z:
                            ix += step_x
                            t_current = t_max_x
                            t_max_x += t_delta_x
                        else:
                            iz += step_z
                            t_current = t_max_z
                            t_max_z += t_delta_z
                    else:
                        if t_max_y < t_max_z:
                            iy += step_y
                            t_current = t_max_y
                            t_max_y += t_delta_y
                        else:
                            iz += step_z
                            t_current = t_max_z
                            t_max_z += t_delta_z

            # === DYNAMIC GEOMETRY (brute force on provided triangles) ===
            for tri in dynamic_triangles:
                dynamic_tris_tested += 1
                v0, v1, v2 = tri
                hit, dist, _ = ray_triangle_intersect(ray_origin, (dx, dy, dz), v0, v1, v2)
                if hit and dist < closest_dist:
                    closest_dist = dist
                    hit_found = True
                    hit_source = "DYNAMIC"

            calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

            result_data = {
                "hit": hit_found,
                "hit_distance": closest_dist if hit_found else None,
                "hit_source": hit_source,
                "static_triangles_tested": static_tris_tested,
                "static_cells_traversed": static_cells_traversed,
                "dynamic_triangles_tested": dynamic_tris_tested,
                "calc_time_us": calc_time_us,
                "method": "CAMERA_FULL",
                "grid_cached": _cached_grid is not None
            }

        else:
            # Unknown job type - still succeed but note it
            result_data = {
                "message": f"Unknown job type '{job.job_type}' - no handler registered",
                "data": job.data
            }

        processing_time = time.perf_counter() - start_time

        # Return plain dict (pickle-safe)
        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "result": result_data,
            "success": True,
            "error": None,
            "timestamp": time.perf_counter(),
            "processing_time": processing_time
        }

    except Exception as e:
        # Capture any errors and return them safely
        processing_time = time.perf_counter() - start_time

        # Return plain dict (pickle-safe)
        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "result": None,
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            "timestamp": time.perf_counter(),
            "processing_time": processing_time
        }


def worker_loop(job_queue, result_queue, worker_id, shutdown_event):
    """
    Main loop for a worker process.
    This is the entry point called by multiprocessing.Process.
    """
    if DEBUG_ENGINE:
        print(f"[Engine Worker {worker_id}] Started")

    jobs_processed = 0

    try:
        while not shutdown_event.is_set():
            try:
                # Wait for a job (with timeout so we can check shutdown_event)
                job = job_queue.get(timeout=0.1)

                if DEBUG_ENGINE:
                    print(f"[Engine Worker {worker_id}] Processing job {job.job_id} (type: {job.job_type})")

                # Process the job
                result = process_job(job)

                # Add worker_id to result before sending (CRITICAL for grid cache verification)
                result["worker_id"] = worker_id

                # Send result back
                result_queue.put(result)

                jobs_processed += 1

                if DEBUG_ENGINE:
                    print(f"[Engine Worker {worker_id}] Completed job {job.job_id} in {result['processing_time']*1000:.2f}ms")

            except Empty:
                # Queue is empty, just continue
                continue
            except Exception as e:
                # Handle any queue errors or unexpected issues
                if not shutdown_event.is_set():
                    if DEBUG_ENGINE:
                        print(f"[Engine Worker {worker_id}] Error: {e}")
                        traceback.print_exc()
                continue

    finally:
        if DEBUG_ENGINE:
            print(f"[Engine Worker {worker_id}] Shutting down (processed {jobs_processed} jobs)")
            