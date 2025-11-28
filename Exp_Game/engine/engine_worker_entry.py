# Exp_Game/engine/engine_worker_entry.py
"""
Worker process entry point - ISOLATED from addon imports.
This module is called directly by multiprocessing and does NOT import bpy.
IMPORTANT: This file has NO relative imports to avoid triggering addon __init__.py
"""

import time
import traceback
from queue import Empty
from dataclasses import dataclass
from typing import Any, Optional

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

        elif job.job_type == "CACHE_GRID":
            # Cache spatial grid for subsequent raycast jobs
            # This is sent ONCE at game start to avoid 3MB serialization per raycast
            grid = job.data.get("grid", None)
            if grid is not None:
                _cached_grid = grid
                tri_count = len(grid.get("triangles", []))
                cell_count = len(grid.get("cells", {}))
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

        elif job.job_type == "FRAME_SYNC_TEST":
            # Frame synchronization test - lightweight job for latency measurement
            # Echoes back frame number and timestamp to measure round-trip time
            result_data = {
                "frame": job.data.get("frame", -1),
                "submit_timestamp": job.data.get("timestamp", 0.0),
                "process_timestamp": time.time(),
                "worker_id": job.data.get("worker_id", -1),
                "worker_msg": "Sync test completed"
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

        elif job.job_type == "KCC_SLOPE_PLATFORM_MATH":
            # KCC XY slope sliding calculations ONLY
            # NOTE: Z velocity and platform carry calculated on main thread for frame-perfect sync
            # 1-frame latency on Z causes jitter/bouncing due to immediate ground feedback loop
            # Pure math (NO bpy access) - vector projections, normalizations, slope sliding
            import math

            calc_start = time.perf_counter()

            # Extract input data
            vel = job.data.get("vel", (0.0, 0.0, 0.0))  # Current velocity
            on_ground = job.data.get("on_ground", False)
            on_walkable = job.data.get("on_walkable", True)
            ground_norm = job.data.get("ground_norm", (0.0, 0.0, 1.0))

            # Config values
            gravity = job.data.get("gravity", -9.81)
            steep_slide_gain = job.data.get("steep_slide_gain", 18.0)
            steep_min_speed = job.data.get("steep_min_speed", 2.5)
            dt = job.data.get("dt", 0.033)

            # Convert to local variables
            vel_x, vel_y, vel_z = vel
            UP = (0.0, 0.0, 1.0)

            # Initialize output
            slide_xy_x, slide_xy_y = 0.0, 0.0  # XY slide velocity for steep slopes
            is_sliding = False  # Flag indicating character is sliding on steep slope

            # ===== XY SLOPE SLIDING (steep slopes only) =====
            if on_ground and not on_walkable:
                # Steep slope - calculate XY sliding only (Z done on main thread)
                # Calculate downhill direction: downhill = -(UP - n*(UP·n))
                gn_x, gn_y, gn_z = ground_norm

                # Normalize ground normal
                gn_len = math.sqrt(gn_x*gn_x + gn_y*gn_y + gn_z*gn_z)
                if gn_len > 1.0e-12:
                    gn_x /= gn_len
                    gn_y /= gn_len
                    gn_z /= gn_len

                    # uphill = UP - n * (UP·n)
                    up_dot_n = gn_z  # UP·n (UP = (0,0,1))
                    uphill_x = 0.0 - gn_x * up_dot_n
                    uphill_y = 0.0 - gn_y * up_dot_n
                    uphill_z = 1.0 - gn_z * up_dot_n

                    # downhill = -uphill
                    downhill_x = -uphill_x
                    downhill_y = -uphill_y
                    downhill_z = -uphill_z

                    # Normalize downhill
                    dh_len = math.sqrt(downhill_x*downhill_x + downhill_y*downhill_y + downhill_z*downhill_z)
                    if dh_len > 1.0e-12:
                        downhill_x /= dh_len
                        downhill_y /= dh_len
                        downhill_z /= dh_len

                        # Apply gravity-scaled acceleration
                        slide_acc_scale = abs(gravity) * steep_slide_gain

                        # XY component only (velocity to apply)
                        # Calculate target slide velocity (clamped to steep_min_speed minimum)
                        slide_speed = math.sqrt(vel_x*vel_x + vel_y*vel_y)
                        target_speed = max(slide_speed + slide_acc_scale * dt, steep_min_speed)

                        # Slide velocity = downhill direction * target speed (XY only)
                        slide_xy_x = downhill_x * target_speed
                        slide_xy_y = downhill_y * target_speed
                        is_sliding = True

            calc_end = time.perf_counter()
            calc_time_us = (calc_end - calc_start) * 1_000_000

            result_data = {
                "delta_z": 0.0,  # Dummy (calculated on main thread to avoid jitter)
                "slide_xy": (slide_xy_x, slide_xy_y),  # XY slide velocity (replaces movement on steep slopes)
                "is_sliding": is_sliding,  # Flag indicating character is sliding
                "carry": (0.0, 0.0, 0.0),  # Dummy (calculated on main thread for frame-perfect sync)
                "rot_delta_z": 0.0,  # Dummy (calculated on main thread)
                "calc_time_us": calc_time_us
            }

        elif job.job_type == "KCC_INPUT_VECTOR":
            # KCC input vector calculation - camera-relative WASD transformation
            # Pure math (NO bpy access) - matrix rotations, normalizations, slope removal
            import math

            calc_start = time.perf_counter()

            # Extract input data
            keys_pressed = job.data.get("keys_pressed", [])
            camera_yaw = job.data.get("camera_yaw", 0.0)
            on_ground = job.data.get("on_ground", False)
            ground_norm = job.data.get("ground_norm", (0.0, 0.0, 1.0))
            floor_cos = job.data.get("floor_cos", 0.7)

            # Preference keys
            fwd_key = job.data.get("pref_forward", "W")
            back_key = job.data.get("pref_backward", "S")
            left_key = job.data.get("pref_left", "A")
            right_key = job.data.get("pref_right", "D")
            run_key = job.data.get("pref_run", "LEFT_SHIFT")

            # 1) Build raw input vector (local space)
            x = 0.0
            y = 0.0
            if fwd_key in keys_pressed:
                y += 1.0
            if back_key in keys_pressed:
                y -= 1.0
            if right_key in keys_pressed:
                x += 1.0
            if left_key in keys_pressed:
                x -= 1.0

            # 2) Normalize input (camera-plane intent)
            v_len2 = x*x + y*y
            if v_len2 > 1.0e-12:
                inv_len = 1.0 / math.sqrt(v_len2)
                vx = x * inv_len
                vy = y * inv_len
            else:
                vx = vy = 0.0

            # 3) Rotate by camera yaw (matrix rotation about Z)
            # Simplified rotation: world_x = vx * cos(yaw) - vy * sin(yaw)
            #                      world_y = vx * sin(yaw) + vy * cos(yaw)
            cos_yaw = math.cos(camera_yaw)
            sin_yaw = math.sin(camera_yaw)
            world_x = vx * cos_yaw - vy * sin_yaw
            world_y = vx * sin_yaw + vy * cos_yaw

            # 4) Normalize XY result
            xy_len2 = world_x * world_x + world_y * world_y
            if xy_len2 > 1.0e-12:
                inv_xy = 1.0 / math.sqrt(xy_len2)
                xy_x = world_x * inv_xy
                xy_y = world_y * inv_xy
            else:
                xy_x = xy_y = 0.0

            # 5) Steep slope uphill removal (if on non-walkable ground)
            UP = (0.0, 0.0, 1.0)
            if on_ground and ground_norm is not None:
                # Check if slope is walkable (n·up >= floor_cos)
                gn_x, gn_y, gn_z = ground_norm
                gn_len = math.sqrt(gn_x*gn_x + gn_y*gn_y + gn_z*gn_z)
                if gn_len > 1.0e-12:
                    gn_x /= gn_len
                    gn_y /= gn_len
                    gn_z /= gn_len
                    n_dot_up = gn_z  # ground_norm · UP (UP = (0,0,1))

                    # If not walkable (too steep)
                    if n_dot_up < floor_cos:
                        # Compute uphill direction: UP - n * (UP·n)
                        up_dot_n = gn_z  # UP·n
                        uphill_x = 0.0 - gn_x * up_dot_n
                        uphill_y = 0.0 - gn_y * up_dot_n
                        uphill_z = 1.0 - gn_z * up_dot_n

                        # Get XY component of uphill
                        g_xy_len = math.sqrt(uphill_x*uphill_x + uphill_y*uphill_y)
                        if g_xy_len > 1.0e-12:
                            g_xy_x = uphill_x / g_xy_len
                            g_xy_y = uphill_y / g_xy_len

                            # Remove uphill component from input XY
                            comp = xy_x * g_xy_x + xy_y * g_xy_y
                            if comp > 0.0:
                                xy_x -= g_xy_x * comp
                                xy_y -= g_xy_y * comp

            # 6) Check if running
            is_running = run_key in keys_pressed

            calc_end = time.perf_counter()
            calc_time_us = (calc_end - calc_start) * 1_000_000

            result_data = {
                "wish_dir_xy": (xy_x, xy_y),
                "is_running": is_running,
                "calc_time_us": calc_time_us
            }

        elif job.job_type == "KCC_RAYCAST":
            # KCC raycasting - ground detection using mesh soup intersection
            # Pure math (NO bpy access) - ray-triangle intersection (Möller-Trumbore)
            # Phase 1: Brute force mesh soup (slow, 1-5ms expected)
            # Phase 2: TODO - Add spatial acceleration (uniform grid/octree)
            import math

            calc_start = time.perf_counter()

            # Extract input data
            ray_origin = job.data.get("ray_origin", (0.0, 0.0, 0.0))
            ray_direction = job.data.get("ray_direction", (0.0, 0.0, -1.0))
            max_distance = job.data.get("max_distance", 100.0)
            triangles = job.data.get("triangles", [])

            # Brute force mesh soup raycast
            closest_hit = None
            closest_dist = max_distance
            closest_triangle = None

            for tri in triangles:
                v0, v1, v2 = tri
                hit, dist, hit_point = ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2)

                if hit and dist < closest_dist:
                    closest_hit = hit_point
                    closest_dist = dist
                    closest_triangle = tri

            # Compute normal if we hit something
            if closest_hit is not None:
                v0, v1, v2 = closest_triangle
                normal = compute_triangle_normal(v0, v1, v2)
            else:
                normal = None

            calc_end = time.perf_counter()
            calc_time_us = (calc_end - calc_start) * 1_000_000

            result_data = {
                "hit": closest_hit is not None,
                "hit_location": closest_hit,  # (x, y, z) or None
                "hit_normal": normal,  # (nx, ny, nz) or None
                "hit_distance": closest_dist if closest_hit is not None else None,
                "triangles_tested": len(triangles),
                "calc_time_us": calc_time_us,
                "method": "BRUTE_FORCE"
            }

        elif job.job_type == "KCC_RAYCAST_GRID":
            # KCC raycasting with spatial grid acceleration (3D DDA traversal)
            # Dramatically faster than brute force for large scenes
            import math

            # Debug: announce job start
            print(f"[Worker] KCC_RAYCAST_GRID job {job.job_id} starting...")

            calc_start = time.perf_counter()

            # Extract input data
            ray_origin = job.data.get("ray_origin", (0.0, 0.0, 0.0))
            ray_direction = job.data.get("ray_direction", (0.0, 0.0, -1.0))
            max_distance = job.data.get("max_distance", 100.0)
            grid = job.data.get("grid", None)

            if grid is None:
                # No grid provided - fall back to error
                result_data = {
                    "hit": False,
                    "hit_location": None,
                    "hit_normal": None,
                    "hit_distance": None,
                    "triangles_tested": 0,
                    "cells_traversed": 0,
                    "calc_time_us": 0.0,
                    "method": "GRID_ERROR",
                    "error": "No grid data provided"
                }
            else:
                # Extract grid data
                bounds_min = grid["bounds_min"]
                bounds_max = grid["bounds_max"]
                cell_size = grid["cell_size"]
                grid_dims = grid["grid_dims"]
                cells = grid["cells"]
                triangles = grid["triangles"]

                ox, oy, oz = ray_origin
                dx, dy, dz = ray_direction

                # Normalize ray direction
                d_len = math.sqrt(dx*dx + dy*dy + dz*dz)
                if d_len > 1e-12:
                    dx /= d_len
                    dy /= d_len
                    dz /= d_len

                nx, ny, nz = grid_dims
                min_x, min_y, min_z = bounds_min
                max_x, max_y, max_z = bounds_max

                # ========== 3D DDA Ray Traversal ==========
                # Reference: "A Fast Voxel Traversal Algorithm" by Amanatides & Woo

                # Calculate starting cell
                # Clamp ray origin to grid bounds if outside
                start_x = max(min_x, min(max_x - 0.001, ox))
                start_y = max(min_y, min(max_y - 0.001, oy))
                start_z = max(min_z, min(max_z - 0.001, oz))

                ix = int((start_x - min_x) / cell_size)
                iy = int((start_y - min_y) / cell_size)
                iz = int((start_z - min_z) / cell_size)

                # Clamp to valid range
                ix = max(0, min(nx - 1, ix))
                iy = max(0, min(ny - 1, iy))
                iz = max(0, min(nz - 1, iz))

                # Step direction (+1 or -1)
                step_x = 1 if dx >= 0 else -1
                step_y = 1 if dy >= 0 else -1
                step_z = 1 if dz >= 0 else -1

                # Calculate tMax (t value at first cell boundary crossing)
                # and tDelta (t to cross one cell)
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

                # Traversal state
                closest_hit = None
                closest_dist = max_distance
                closest_triangle = None
                triangles_tested = 0
                cells_traversed = 0
                tested_triangles = set()  # Avoid testing same triangle twice

                # Traverse grid using 3D DDA
                max_cells = nx + ny + nz + 10  # Safety limit
                t_current = 0.0

                while cells_traversed < max_cells:
                    cells_traversed += 1

                    # Check if we've gone past max_distance
                    if t_current > max_distance:
                        break

                    # Check if cell is within bounds
                    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                        break

                    # Get triangles in current cell
                    cell_key = (ix, iy, iz)
                    if cell_key in cells:
                        cell_tris = cells[cell_key]

                        for tri_idx in cell_tris:
                            # Skip if already tested this triangle
                            if tri_idx in tested_triangles:
                                continue
                            tested_triangles.add(tri_idx)
                            triangles_tested += 1

                            # Test ray-triangle intersection
                            tri = triangles[tri_idx]
                            v0, v1, v2 = tri
                            hit, dist, hit_point = ray_triangle_intersect(
                                ray_origin, (dx, dy, dz), v0, v1, v2
                            )

                            if hit and dist < closest_dist:
                                closest_hit = hit_point
                                closest_dist = dist
                                closest_triangle = tri

                    # Early exit: if we found a hit and it's within the current cell's t range
                    # we can stop (no closer hit possible in further cells)
                    t_next = min(t_max_x, t_max_y, t_max_z)
                    if closest_hit is not None and closest_dist <= t_next:
                        break

                    # Step to next cell (DDA step)
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

                # Compute normal if we hit something
                if closest_hit is not None:
                    v0, v1, v2 = closest_triangle
                    normal = compute_triangle_normal(v0, v1, v2)
                else:
                    normal = None

                calc_end = time.perf_counter()
                calc_time_us = (calc_end - calc_start) * 1_000_000

                result_data = {
                    "hit": closest_hit is not None,
                    "hit_location": closest_hit,
                    "hit_normal": normal,
                    "hit_distance": closest_dist if closest_hit is not None else None,
                    "triangles_tested": triangles_tested,
                    "cells_traversed": cells_traversed,
                    "calc_time_us": calc_time_us,
                    "method": "GRID_DDA"
                }

                # Debug: announce completion
                print(f"[Worker] KCC_RAYCAST_GRID job {job.job_id} complete: tested={triangles_tested} tris, cells={cells_traversed}, time={calc_time_us:.1f}µs")

        elif job.job_type == "KCC_RAYCAST_CACHED":
            # KCC raycasting using CACHED grid (no serialization overhead!)
            # Grid must be cached first via CACHE_GRID job
            import math

            calc_start = time.perf_counter()

            # Extract ray data (small payload - ~100 bytes)
            ray_origin = job.data.get("ray_origin", (0.0, 0.0, 0.0))
            ray_direction = job.data.get("ray_direction", (0.0, 0.0, -1.0))
            max_distance = job.data.get("max_distance", 100.0)

            # Use cached grid
            if _cached_grid is None:
                result_data = {
                    "hit": False,
                    "hit_location": None,
                    "hit_normal": None,
                    "hit_distance": None,
                    "triangles_tested": 0,
                    "cells_traversed": 0,
                    "calc_time_us": 0.0,
                    "method": "CACHED_ERROR",
                    "error": "No grid cached - send CACHE_GRID job first"
                }
            else:
                grid = _cached_grid

                # Extract grid data
                bounds_min = grid["bounds_min"]
                bounds_max = grid["bounds_max"]
                cell_size = grid["cell_size"]
                grid_dims = grid["grid_dims"]
                cells = grid["cells"]
                triangles = grid["triangles"]

                ox, oy, oz = ray_origin
                dx, dy, dz = ray_direction

                # Normalize ray direction
                d_len = math.sqrt(dx*dx + dy*dy + dz*dz)
                if d_len > 1e-12:
                    dx /= d_len
                    dy /= d_len
                    dz /= d_len

                nx, ny, nz = grid_dims
                min_x, min_y, min_z = bounds_min
                max_x, max_y, max_z = bounds_max

                # 3D DDA Ray Traversal (same algorithm as KCC_RAYCAST_GRID)
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

                closest_hit = None
                closest_dist = max_distance
                closest_triangle = None
                triangles_tested = 0
                cells_traversed = 0
                tested_triangles = set()

                max_cells = nx + ny + nz + 10
                t_current = 0.0

                while cells_traversed < max_cells:
                    cells_traversed += 1

                    if t_current > max_distance:
                        break

                    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                        break

                    cell_key = (ix, iy, iz)
                    if cell_key in cells:
                        cell_tris = cells[cell_key]

                        for tri_idx in cell_tris:
                            if tri_idx in tested_triangles:
                                continue
                            tested_triangles.add(tri_idx)
                            triangles_tested += 1

                            tri = triangles[tri_idx]
                            v0, v1, v2 = tri
                            hit, dist, hit_point = ray_triangle_intersect(
                                ray_origin, (dx, dy, dz), v0, v1, v2
                            )

                            if hit and dist < closest_dist:
                                closest_hit = hit_point
                                closest_dist = dist
                                closest_triangle = tri

                    t_next = min(t_max_x, t_max_y, t_max_z)
                    if closest_hit is not None and closest_dist <= t_next:
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

                if closest_hit is not None:
                    v0, v1, v2 = closest_triangle
                    normal = compute_triangle_normal(v0, v1, v2)
                else:
                    normal = None

                calc_end = time.perf_counter()
                calc_time_us = (calc_end - calc_start) * 1_000_000

                result_data = {
                    "hit": closest_hit is not None,
                    "hit_location": closest_hit,
                    "hit_normal": normal,
                    "hit_distance": closest_dist if closest_hit is not None else None,
                    "triangles_tested": triangles_tested,
                    "cells_traversed": cells_traversed,
                    "calc_time_us": calc_time_us,
                    "method": "CACHED_DDA"
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
