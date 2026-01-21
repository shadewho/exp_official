# Exp_Game/engine/worker/reactions/tracking.py
"""
NPC/Object Tracking - Worker-side movement computation.

TWO NAVIGATION MODES:
- DIRECT: Beeline toward target, stop at walls (fast, simple)
- PATHFINDING: Smart steering around obstacles (zombie-like behavior)

PATHFINDING uses reactive steering:
- Cast rays in a fan pattern to detect obstacles
- Score each direction: clearance + goal alignment
- Pick best direction that makes progress toward goal
- Simple but effective - no navmesh required

Supports BOTH static grid AND dynamic meshes for collision.
Character autopilot stays on main thread (just injects keys).
"""

import math
import time
from ..raycast import unified_raycast


# Pathfinding constants
_PF_NUM_RAYS = 9          # Number of rays in fan (odd = includes center)
_PF_FAN_ANGLE = 2.4       # ~140 degrees total spread (radians)
_PF_PROBE_DIST = 2.5      # How far to probe for obstacles
_PF_CLEARANCE_WEIGHT = 0.4  # Weight for obstacle clearance
_PF_GOAL_WEIGHT = 0.6     # Weight for goal direction alignment


def _compute_pathfinding_direction(
    pos_x: float, pos_y: float, pos_z: float,
    goal_x: float, goal_y: float,
    speed: float, dt: float,
    grid_data, unified_dynamic_meshes
) -> tuple:
    """
    Compute best movement direction using reactive steering.

    Returns (dir_x, dir_y, step_len, blocked)
    """
    # Direct direction to goal
    dx = goal_x - pos_x
    dy = goal_y - pos_y
    dist_sq = dx * dx + dy * dy

    if dist_sq < 0.01:  # Already at goal
        return (0.0, 0.0, 0.0, False)

    dist = math.sqrt(dist_sq)
    direct_x = dx / dist
    direct_y = dy / dist

    # Base step length
    step_len = min(speed * dt, dist)

    # If no collision checking, just go direct
    if not grid_data:
        return (direct_x, direct_y, step_len, False)

    # Cast rays in a fan pattern
    # Angles from -half_fan to +half_fan, centered on direct direction
    half_fan = _PF_FAN_ANGLE * 0.5
    angle_step = _PF_FAN_ANGLE / (_PF_NUM_RAYS - 1) if _PF_NUM_RAYS > 1 else 0

    # Direct angle (atan2)
    base_angle = math.atan2(direct_y, direct_x)

    ray_origin = (pos_x, pos_y, pos_z + 0.5)  # Hip height

    best_score = -999.0
    best_dir = (direct_x, direct_y)
    best_clearance = 0.0
    all_blocked = True

    for i in range(_PF_NUM_RAYS):
        # Angle offset from direct
        offset = -half_fan + i * angle_step
        angle = base_angle + offset

        ray_dx = math.cos(angle)
        ray_dy = math.sin(angle)
        ray_dir = (ray_dx, ray_dy, 0.0)

        # Cast ray
        result = unified_raycast(ray_origin, ray_dir, _PF_PROBE_DIST, grid_data, unified_dynamic_meshes)

        if result["hit"]:
            clearance = result["dist"]
            # Check if it's a wall (mostly horizontal normal)
            nz = result["normal"][2]
            is_wall = (nz * nz) < 0.49  # |nz| < 0.7
            if not is_wall:
                clearance = _PF_PROBE_DIST  # Floor/ceiling don't block
        else:
            clearance = _PF_PROBE_DIST

        # Score this direction
        # - Goal alignment: dot product with direct direction (1.0 = perfect, -1.0 = opposite)
        # - Clearance: how much open space (0 to _PF_PROBE_DIST)
        goal_alignment = ray_dx * direct_x + ray_dy * direct_y  # -1 to 1
        goal_score = (goal_alignment + 1.0) * 0.5  # Normalize to 0-1
        clearance_score = clearance / _PF_PROBE_DIST  # Normalize to 0-1

        # Combined score
        score = goal_score * _PF_GOAL_WEIGHT + clearance_score * _PF_CLEARANCE_WEIGHT

        # Only consider directions with enough clearance to move
        min_clearance = step_len + 0.2
        if clearance >= min_clearance:
            all_blocked = False
            if score > best_score:
                best_score = score
                best_dir = (ray_dx, ray_dy)
                best_clearance = clearance

    if all_blocked:
        # All directions blocked - try to back away or stop
        return (direct_x, direct_y, 0.0, True)

    return (best_dir[0], best_dir[1], step_len, False)


def handle_tracking_batch(job_data: dict, grid_data, cached_dynamic_meshes, cached_dynamic_transforms) -> dict:
    """
    Batch compute tracking movement for all active tracks.

    Supports two modes:
    - DIRECT: Beeline toward target, stop at walls
    - PATHFINDING: Smart steering around obstacles
    """
    start_time = time.perf_counter()

    tracks = job_data.get("tracks", [])
    dynamic_transforms = job_data.get("dynamic_transforms", {})
    debug_logs = job_data.get("debug_logs")

    if not tracks:
        return {
            "results": [],
            "count": 0,
            "rays": 0,
            "tris": 0,
            "dyn_meshes": 0,
            "calc_time_us": 0,
            "worker_logs": [],
        }

    results = []
    total_rays = 0
    total_tris = 0

    # ─────────────────────────────────────────────────────────────────────────
    # PRE-CHECK: Do any tracks need collision/gravity?
    # ─────────────────────────────────────────────────────────────────────────
    any_collision = False
    any_gravity = False
    any_pathfinding = False
    for track in tracks:
        if track.get("use_collision", True):
            any_collision = True
        if track.get("use_gravity", True):
            any_gravity = True
        if track.get("mode", "DIRECT") == "PATHFINDING":
            any_pathfinding = True
        if any_collision and any_gravity and any_pathfinding:
            break

    any_needs_raycast = any_collision or any_gravity or any_pathfinding

    # ─────────────────────────────────────────────────────────────────────────
    # UPDATE DYNAMIC TRANSFORM CACHE
    # ─────────────────────────────────────────────────────────────────────────
    if any_needs_raycast and dynamic_transforms and cached_dynamic_meshes:
        from ..math import transform_aabb_by_matrix, invert_matrix_4x4

        for obj_id, matrix_4x4 in dynamic_transforms.items():
            cached = cached_dynamic_meshes.get(obj_id)
            if cached is None:
                continue

            local_aabb = cached.get("local_aabb")
            world_aabb = transform_aabb_by_matrix(local_aabb, matrix_4x4) if local_aabb else None
            inv_matrix = invert_matrix_4x4(matrix_4x4)
            cached_dynamic_transforms[obj_id] = (matrix_4x4, world_aabb, inv_matrix)

    # ─────────────────────────────────────────────────────────────────────────
    # BUILD UNIFIED DYNAMIC MESHES LIST
    # ─────────────────────────────────────────────────────────────────────────
    unified_dynamic_meshes = None
    if any_needs_raycast and cached_dynamic_meshes and cached_dynamic_transforms:
        unified_dynamic_meshes = []
        for obj_id, transform_data in cached_dynamic_transforms.items():
            matrix_4x4, world_aabb, inv_matrix = transform_data
            if inv_matrix is None:
                continue

            cached = cached_dynamic_meshes.get(obj_id)
            if cached is None:
                continue

            local_triangles = cached.get("triangles")
            if not local_triangles:
                continue

            if world_aabb is None:
                continue

            unified_dynamic_meshes.append({
                "obj_id": obj_id,
                "triangles": local_triangles,
                "matrix": matrix_4x4,
                "inv_matrix": inv_matrix,
                "aabb": world_aabb,
                "bounding_sphere": None,
                "grid": cached.get("grid"),
            })

    dyn_mesh_count = len(unified_dynamic_meshes) if unified_dynamic_meshes else 0

    # ─────────────────────────────────────────────────────────────────────────
    # PROCESS ALL TRACKS
    # ─────────────────────────────────────────────────────────────────────────
    for track in tracks:
        obj_id = track["obj_id"]
        pos_x, pos_y, pos_z = track["current_pos"]
        goal_x, goal_y, goal_z = track["goal_pos"]
        speed = track["speed"]
        dt = track["dt"]
        arrive_radius_sq = track.get("arrive_radius", 0.3) ** 2
        use_gravity = track.get("use_gravity", True)
        use_collision = track.get("use_collision", True)
        mode = track.get("mode", "DIRECT")

        # ─────────────────────────────────────────────────────────────────
        # CHECK ARRIVAL (XY plane)
        # ─────────────────────────────────────────────────────────────────
        dx = goal_x - pos_x
        dy = goal_y - pos_y
        dist_xy_sq = dx * dx + dy * dy

        if dist_xy_sq <= arrive_radius_sq:
            results.append({
                "obj_id": obj_id,
                "new_pos": (pos_x, pos_y, pos_z),
                "arrived": True,
            })
            continue

        # ─────────────────────────────────────────────────────────────────
        # COMPUTE MOVEMENT DIRECTION
        # ─────────────────────────────────────────────────────────────────
        if mode == "PATHFINDING" and use_collision and grid_data:
            # Smart steering around obstacles
            dir_x, dir_y, step_len, blocked = _compute_pathfinding_direction(
                pos_x, pos_y, pos_z,
                goal_x, goal_y,
                speed, dt,
                grid_data, unified_dynamic_meshes
            )
            total_rays += _PF_NUM_RAYS  # Count pathfinding rays

            if blocked:
                # Can't find a path - stay in place
                results.append({
                    "obj_id": obj_id,
                    "new_pos": (pos_x, pos_y, pos_z),
                    "arrived": False,
                })
                continue

            new_x = pos_x + dir_x * step_len
            new_y = pos_y + dir_y * step_len
            new_z = pos_z

        else:
            # DIRECT mode - beeline toward target
            dist_xy = math.sqrt(dist_xy_sq)
            step_len = speed * dt
            if step_len > dist_xy:
                step_len = dist_xy

            inv_dist = 1.0 / dist_xy
            dir_x = dx * inv_dist
            dir_y = dy * inv_dist

            new_x = pos_x + dir_x * step_len
            new_y = pos_y + dir_y * step_len
            new_z = pos_z

            # Collision check for DIRECT mode
            if use_collision and grid_data:
                ray_origin = (pos_x, pos_y, pos_z + 0.5)
                ray_dir = (dir_x, dir_y, 0.0)
                ray_len = step_len + 0.3

                result = unified_raycast(ray_origin, ray_dir, ray_len, grid_data, unified_dynamic_meshes)
                total_rays += 1
                total_tris += result.get("tris_tested", 0)

                if result["hit"]:
                    normal = result["normal"]
                    hit_dist = result["dist"]

                    if normal[2] * normal[2] < 0.49:
                        allow = hit_dist - 0.3
                        if allow < 0.0:
                            allow = 0.0
                        if allow < step_len:
                            new_x = pos_x + dir_x * allow
                            new_y = pos_y + dir_y * allow

        # ─────────────────────────────────────────────────────────────────
        # GRAVITY SNAP
        # ─────────────────────────────────────────────────────────────────
        if use_gravity and grid_data:
            ray_origin = (new_x, new_y, new_z + 1.0)
            ray_dir = (0.0, 0.0, -1.0)
            max_down = 2.0

            result = unified_raycast(ray_origin, ray_dir, max_down, grid_data, unified_dynamic_meshes)
            total_rays += 1
            total_tris += result.get("tris_tested", 0)

            if result["hit"]:
                if result["normal"][2] > 0.5:
                    new_z = result["pos"][2]

        results.append({
            "obj_id": obj_id,
            "new_pos": (new_x, new_y, new_z),
            "arrived": False,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    calc_time_us = int((time.perf_counter() - start_time) * 1_000_000)

    if debug_logs is not None:
        pf_count = sum(1 for t in tracks if t.get("mode") == "PATHFINDING")
        debug_logs.append(("TRACKING", f"BATCH tracks={len(results)} pf={pf_count} rays={total_rays} dyn={dyn_mesh_count} {calc_time_us}us"))

    return {
        "results": results,
        "count": len(results),
        "rays": total_rays,
        "tris": total_tris,
        "dyn_meshes": dyn_mesh_count,
        "calc_time_us": calc_time_us,
        "worker_logs": debug_logs if debug_logs else [],
    }
