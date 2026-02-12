# Exp_Game/engine/worker/reactions/tracking.py
"""
Object Tracking - Worker-side movement + gravity computation.

TWO NAVIGATION MODES:
- DIRECT: Beeline toward target, stop at walls (fast, simple)
- PATHFINDING: Smart steering around obstacles (reactive fan rays)

GRAVITY: Worker-side velocity cache. Each tracked object accumulates
downward velocity (9.81 m/s^2). Ground detection via downward raycast
against static grid AND dynamic meshes. Lands on any surface with
upward-facing normal.

Supports BOTH static grid AND dynamic meshes for collision/gravity.
"""

import math
import time
from ..raycast import unified_raycast


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_GRAVITY = 9.81           # m/s^2
_TERMINAL_VEL = 30.0      # Max fall speed (m/s)
_GROUND_PROBE_BASE = 0.6  # Base downward probe distance (m)
_GROUND_OFFSET = 0.4      # Ray starts this far above object origin (m)

# Pathfinding constants
_PF_NUM_RAYS = 9          # Number of rays in fan (odd = includes center)
_PF_FAN_ANGLE = 2.4       # ~140 degrees total spread (radians)
_PF_PROBE_DIST = 2.5      # How far to probe for obstacles
_PF_CLEARANCE_WEIGHT = 0.4
_PF_GOAL_WEIGHT = 0.6


# ─────────────────────────────────────────────────────────────────────────────
# Worker-side gravity velocity cache (persists across frames)
# ─────────────────────────────────────────────────────────────────────────────
_tracking_velocities = {}  # {obj_id: velocity_z}


# ─────────────────────────────────────────────────────────────────────────────
# Pathfinding
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pathfinding_direction(
    pos_x, pos_y, pos_z,
    goal_x, goal_y,
    speed, dt,
    grid_data, unified_dynamic_meshes
):
    """
    Compute best movement direction using reactive steering.
    Returns (dir_x, dir_y, step_len, blocked).
    """
    dx = goal_x - pos_x
    dy = goal_y - pos_y
    dist_sq = dx * dx + dy * dy

    if dist_sq < 0.01:
        return (0.0, 0.0, 0.0, False)

    dist = math.sqrt(dist_sq)
    direct_x = dx / dist
    direct_y = dy / dist
    step_len = min(speed * dt, dist)

    if not grid_data and not unified_dynamic_meshes:
        return (direct_x, direct_y, step_len, False)

    half_fan = _PF_FAN_ANGLE * 0.5
    angle_step = _PF_FAN_ANGLE / (_PF_NUM_RAYS - 1) if _PF_NUM_RAYS > 1 else 0
    base_angle = math.atan2(direct_y, direct_x)
    ray_origin = (pos_x, pos_y, pos_z + 0.5)

    best_score = -999.0
    best_dir = (direct_x, direct_y)
    all_blocked = True

    for i in range(_PF_NUM_RAYS):
        offset = -half_fan + i * angle_step
        angle = base_angle + offset

        ray_dx = math.cos(angle)
        ray_dy = math.sin(angle)
        ray_dir = (ray_dx, ray_dy, 0.0)

        result = unified_raycast(ray_origin, ray_dir, _PF_PROBE_DIST, grid_data, unified_dynamic_meshes)

        if result["hit"]:
            clearance = result["dist"]
            nz = result["normal"][2]
            is_wall = (nz * nz) < 0.49
            if not is_wall:
                clearance = _PF_PROBE_DIST
        else:
            clearance = _PF_PROBE_DIST

        goal_alignment = ray_dx * direct_x + ray_dy * direct_y
        goal_score = (goal_alignment + 1.0) * 0.5
        clearance_score = clearance / _PF_PROBE_DIST
        score = goal_score * _PF_GOAL_WEIGHT + clearance_score * _PF_CLEARANCE_WEIGHT

        min_clearance = step_len + 0.2
        if clearance >= min_clearance:
            all_blocked = False
            if score > best_score:
                best_score = score
                best_dir = (ray_dx, ray_dy)

    if all_blocked:
        return (direct_x, direct_y, 0.0, True)

    return (best_dir[0], best_dir[1], step_len, False)


# ─────────────────────────────────────────────────────────────────────────────
# Main batch handler
# ─────────────────────────────────────────────────────────────────────────────

def handle_tracking_batch(job_data, grid_data, cached_dynamic_meshes, cached_dynamic_transforms):
    """
    Batch compute tracking movement for all active tracks.
    Handles collision (static + dynamic) and gravity (velocity-based).
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

    # ─────────────────────────────────────────────────────────────────────
    # PRE-CHECK: Do any tracks need raycast features?
    # ─────────────────────────────────────────────────────────────────────
    any_needs_raycast = False
    for track in tracks:
        if track.get("use_collision") or track.get("use_gravity"):
            any_needs_raycast = True
            break

    has_geometry = grid_data or (cached_dynamic_meshes and cached_dynamic_transforms)

    # ─────────────────────────────────────────────────────────────────────
    # UPDATE DYNAMIC TRANSFORM CACHE
    # ─────────────────────────────────────────────────────────────────────
    if any_needs_raycast and has_geometry and dynamic_transforms and cached_dynamic_meshes:
        from ..math import transform_aabb_by_matrix, invert_matrix_4x4

        for obj_id, matrix_4x4 in dynamic_transforms.items():
            cached = cached_dynamic_meshes.get(obj_id)
            if cached is None:
                continue
            local_aabb = cached.get("local_aabb")
            world_aabb = transform_aabb_by_matrix(local_aabb, matrix_4x4) if local_aabb else None
            inv_matrix = invert_matrix_4x4(matrix_4x4)
            cached_dynamic_transforms[obj_id] = (matrix_4x4, world_aabb, inv_matrix)

    # ─────────────────────────────────────────────────────────────────────
    # BUILD UNIFIED DYNAMIC MESHES LIST
    # ─────────────────────────────────────────────────────────────────────
    unified_dynamic_meshes = None
    if any_needs_raycast and cached_dynamic_meshes and cached_dynamic_transforms:
        unified_dynamic_meshes = []
        for obj_id, transform_data in cached_dynamic_transforms.items():
            matrix_4x4, world_aabb, inv_matrix = transform_data
            if inv_matrix is None or world_aabb is None:
                continue
            cached = cached_dynamic_meshes.get(obj_id)
            if cached is None:
                continue
            local_triangles = cached.get("triangles")
            if not local_triangles:
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
    can_raycast = grid_data or unified_dynamic_meshes

    # Track which obj_ids are in this batch (for velocity cache cleanup)
    batch_obj_ids = set()

    # ─────────────────────────────────────────────────────────────────────
    # PROCESS ALL TRACKS
    # ─────────────────────────────────────────────────────────────────────
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

        batch_obj_ids.add(obj_id)

        # ─────────────────────────────────────────────────────────────
        # CHECK ARRIVAL (XY plane)
        # ─────────────────────────────────────────────────────────────
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

        # ─────────────────────────────────────────────────────────────
        # COMPUTE HORIZONTAL MOVEMENT
        # ─────────────────────────────────────────────────────────────
        if mode == "PATHFINDING" and use_collision and can_raycast:
            dir_x, dir_y, step_len, blocked = _compute_pathfinding_direction(
                pos_x, pos_y, pos_z,
                goal_x, goal_y,
                speed, dt,
                grid_data, unified_dynamic_meshes
            )
            total_rays += _PF_NUM_RAYS

            if blocked:
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
            # DIRECT mode
            dist_xy = math.sqrt(dist_xy_sq)
            step_len = min(speed * dt, dist_xy)
            inv_dist = 1.0 / dist_xy
            dir_x = dx * inv_dist
            dir_y = dy * inv_dist

            new_x = pos_x + dir_x * step_len
            new_y = pos_y + dir_y * step_len
            new_z = pos_z

            # Forward collision check
            if use_collision and can_raycast:
                ray_origin = (pos_x, pos_y, pos_z + 0.5)
                ray_dir = (dir_x, dir_y, 0.0)
                ray_len = step_len + 0.3

                result = unified_raycast(ray_origin, ray_dir, ray_len, grid_data, unified_dynamic_meshes)
                total_rays += 1
                total_tris += result.get("tris_tested", 0)

                if result["hit"]:
                    normal = result["normal"]
                    hit_dist = result["dist"]
                    # Only block on walls (not floors/ceilings)
                    if normal[2] * normal[2] < 0.49:
                        allow = max(0.0, hit_dist - 0.3)
                        if allow < step_len:
                            new_x = pos_x + dir_x * allow
                            new_y = pos_y + dir_y * allow

        # ─────────────────────────────────────────────────────────────
        # GRAVITY (velocity-based)
        # ─────────────────────────────────────────────────────────────
        if use_gravity:
            vel_z = _tracking_velocities.get(obj_id, 0.0)

            # Apply gravity acceleration
            vel_z -= _GRAVITY * dt
            if vel_z < -_TERMINAL_VEL:
                vel_z = -_TERMINAL_VEL

            # Apply vertical velocity
            new_z += vel_z * dt

            # Ground detection raycast
            if can_raycast:
                # Probe from above the object's new position downward
                probe_start_z = new_z + _GROUND_OFFSET
                probe_dist = _GROUND_PROBE_BASE + max(0.0, -vel_z * dt)

                ray_origin = (new_x, new_y, probe_start_z)
                ray_dir = (0.0, 0.0, -1.0)

                result = unified_raycast(ray_origin, ray_dir, probe_dist, grid_data, unified_dynamic_meshes)
                total_rays += 1
                total_tris += result.get("tris_tested", 0)

                if result["hit"] and result["normal"][2] > 0.5:
                    ground_z = result["pos"][2]
                    # Land if we've fallen to or below ground level
                    if new_z <= ground_z:
                        new_z = ground_z
                        vel_z = 0.0

            _tracking_velocities[obj_id] = vel_z

        result_entry = {
            "obj_id": obj_id,
            "new_pos": (new_x, new_y, new_z),
            "arrived": False,
        }

        # Face Object rotation computation
        face_target_pos = track.get("face_target_pos")
        if face_target_pos is not None:
            face_axis = track.get("face_axis", "NEG_Y")
            ft_x, ft_y, ft_z = face_target_pos
            fdx = ft_x - new_x
            fdy = ft_y - new_y
            fdz = ft_z - new_z
            dist_xy = math.sqrt(fdx * fdx + fdy * fdy)

            if face_axis == "POS_Z" or face_axis == "NEG_Z":
                # Z-axis variants: no Z rotation, tilt via X rotation
                result_entry["face_euler_z"] = 0.0
                if face_axis == "POS_Z":
                    result_entry["face_euler_x"] = math.atan2(-fdz, dist_xy) if dist_xy > 0.001 else 0.0
                else:
                    result_entry["face_euler_x"] = (math.atan2(fdz, dist_xy) + math.pi) if dist_xy > 0.001 else math.pi
            elif dist_xy > 0.001:
                base_angle = math.atan2(fdy, fdx)
                if face_axis == "POS_X":
                    result_entry["face_euler_z"] = base_angle
                elif face_axis == "NEG_X":
                    result_entry["face_euler_z"] = base_angle + math.pi
                elif face_axis == "POS_Y":
                    result_entry["face_euler_z"] = base_angle - (math.pi * 0.5)
                else:  # NEG_Y
                    result_entry["face_euler_z"] = base_angle + (math.pi * 0.5)

        results.append(result_entry)

    # Clean stale velocity entries (objects no longer tracked)
    if len(_tracking_velocities) > len(batch_obj_ids) * 2:
        stale = [k for k in _tracking_velocities if k not in batch_obj_ids]
        for k in stale:
            del _tracking_velocities[k]

    # ─────────────────────────────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────────────────────────────
    calc_time_us = int((time.perf_counter() - start_time) * 1_000_000)

    if debug_logs is not None:
        pf_count = sum(1 for t in tracks if t.get("mode") == "PATHFINDING")
        grav_count = sum(1 for t in tracks if t.get("use_gravity"))
        debug_logs.append(("TRACKING",
            f"BATCH tracks={len(results)} pf={pf_count} grav={grav_count} "
            f"rays={total_rays} dyn={dyn_mesh_count} {calc_time_us}us"))

    return {
        "results": results,
        "count": len(results),
        "rays": total_rays,
        "tris": total_tris,
        "dyn_meshes": dyn_mesh_count,
        "calc_time_us": calc_time_us,
        "worker_logs": debug_logs if debug_logs else [],
    }
