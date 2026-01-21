# Exp_Game/engine/worker/reactions/tracking.py
"""
NPC/Object Tracking - Worker-side movement computation.

Simplified system that reuses unified_raycast from engine.
- Move toward goal at speed
- Stop at walls (no slide)
- Snap to ground with gravity

Character autopilot stays on main thread (just injects keys).

OPTIMIZATIONS:
  - Only build unified_dynamic_meshes if any track needs collision
  - Early-out for arrived tracks (no raycast)
  - Track ray/tri stats for diagnostics
  - Minimal dict lookups in hot loop
"""

import math
import time
from ..raycast import unified_raycast


def handle_tracking_batch(job_data: dict, grid_data, cached_dynamic_meshes, cached_dynamic_transforms) -> dict:
    """
    Batch compute tracking movement for all active tracks.

    Simplified physics:
    - XY movement toward goal
    - Forward collision check (stop at walls)
    - Gravity snap to ground
    """
    start_time = time.perf_counter()

    tracks = job_data.get("tracks", [])
    debug_logs = job_data.get("debug_logs")
    results = []

    # Stats
    total_rays = 0
    total_tris = 0

    # ─────────────────────────────────────────────────────────────────────────
    # PRE-CHECK: Do any tracks need collision/gravity?
    # Only build unified_dynamic_meshes if needed
    # ─────────────────────────────────────────────────────────────────────────
    any_needs_raycast = False
    for track in tracks:
        if track.get("use_collision", True) or track.get("use_gravity", True):
            any_needs_raycast = True
            break

    # ─────────────────────────────────────────────────────────────────────────
    # BUILD UNIFIED DYNAMIC MESHES LIST FROM ALL (only if needed)
    # ─────────────────────────────────────────────────────────────────────────
    unified_dynamic_meshes = None
    if any_needs_raycast and cached_dynamic_meshes and cached_dynamic_transforms:
        unified_dynamic_meshes = []
        for obj_id, (matrix_4x4, world_aabb, inv_matrix) in cached_dynamic_transforms.items():
            cached = cached_dynamic_meshes.get(obj_id)
            if cached is None:
                continue

            local_triangles = cached.get("triangles", [])
            if not local_triangles:
                continue  # Skip meshes with no triangles

            local_grid = cached.get("grid")

            # AABB is required - no fallback
            if world_aabb is None:
                continue
            bounding_sphere = None

            unified_dynamic_meshes.append({
                "obj_id": obj_id,
                "triangles": local_triangles,
                "matrix": matrix_4x4,
                "inv_matrix": inv_matrix,
                "aabb": world_aabb,
                "bounding_sphere": bounding_sphere,
                "grid": local_grid,
            })

    # ─────────────────────────────────────────────────────────────────────────
    # PROCESS EACH TRACK
    # ─────────────────────────────────────────────────────────────────────────
    for track in tracks:
        obj_id = track["obj_id"]
        pos_x, pos_y, pos_z = track["current_pos"]
        goal_x, goal_y, goal_z = track["goal_pos"]
        speed = track["speed"]
        dt = track["dt"]
        arrive_radius = track.get("arrive_radius", 0.3)
        use_gravity = track.get("use_gravity", True)
        use_collision = track.get("use_collision", True)

        # ─────────────────────────────────────────────────────────────────
        # DIRECTION TO GOAL (XY plane only)
        # ─────────────────────────────────────────────────────────────────
        dx = goal_x - pos_x
        dy = goal_y - pos_y
        dist_xy_sq = dx * dx + dy * dy

        # Check arrival (squared distance comparison - faster)
        arrive_radius_sq = arrive_radius * arrive_radius
        if dist_xy_sq <= arrive_radius_sq:
            results.append({
                "obj_id": obj_id,
                "new_pos": (pos_x, pos_y, pos_z),
                "arrived": True,
            })
            if debug_logs is not None:
                debug_logs.append(("TRACKING", f"  obj={obj_id} ARRIVED"))
            continue

        # Movement step
        dist_xy = math.sqrt(dist_xy_sq)
        step_len = min(speed * dt, dist_xy)
        dir_x = dx / dist_xy
        dir_y = dy / dist_xy

        # ─────────────────────────────────────────────────────────────────
        # COLLISION CHECK (simple - just stop at walls)
        # ─────────────────────────────────────────────────────────────────
        new_x = pos_x + dir_x * step_len
        new_y = pos_y + dir_y * step_len
        new_z = pos_z

        if use_collision and grid_data:
            # Cast ray forward at a reasonable height (0.5m above origin)
            ray_origin = (pos_x, pos_y, pos_z + 0.5)
            ray_dir = (dir_x, dir_y, 0.0)
            ray_len = step_len + 0.3  # step + buffer

            result = unified_raycast(ray_origin, ray_dir, ray_len, grid_data, unified_dynamic_meshes)
            total_rays += 1
            total_tris += result.get("tris_tested", 0)

            if result["hit"]:
                normal_z = result["normal"][2]
                hit_dist = result["dist"]

                # Only block on walls (normal mostly horizontal)
                if abs(normal_z) < 0.7:
                    # Wall - stop before it
                    allow = max(0.0, hit_dist - 0.3)
                    if allow < step_len:
                        new_x = pos_x + dir_x * allow
                        new_y = pos_y + dir_y * allow

                    if debug_logs is not None:
                        debug_logs.append(("TRACKING", f"  WALL at dist={hit_dist:.2f} allow={allow:.2f}"))

        # ─────────────────────────────────────────────────────────────────
        # GRAVITY SNAP
        # ─────────────────────────────────────────────────────────────────
        if use_gravity and grid_data:
            # Cast down from above to find ground
            ray_origin = (new_x, new_y, new_z + 1.0)
            ray_dir = (0.0, 0.0, -1.0)
            max_down = 2.0

            result = unified_raycast(ray_origin, ray_dir, max_down, grid_data, unified_dynamic_meshes)
            total_rays += 1
            total_tris += result.get("tris_tested", 0)

            if result["hit"]:
                normal_z = result["normal"][2]
                # Only snap to floors (normal pointing up)
                if normal_z > 0.5:
                    new_z = result["pos"][2]

                    if debug_logs is not None:
                        debug_logs.append(("TRACKING", f"  GROUND at z={new_z:.2f}"))

        results.append({
            "obj_id": obj_id,
            "new_pos": (new_x, new_y, new_z),
            "arrived": False,
        })

        if debug_logs is not None:
            debug_logs.append(("TRACKING", f"  obj={obj_id} moved ({pos_x:.2f},{pos_y:.2f},{pos_z:.2f}) -> ({new_x:.2f},{new_y:.2f},{new_z:.2f})"))

    # Summary
    calc_time_us = int((time.perf_counter() - start_time) * 1_000_000)

    if debug_logs is not None:
        dyn_count = len(unified_dynamic_meshes) if unified_dynamic_meshes else 0
        debug_logs.append(("TRACKING", f"WORKER processed {len(results)} tracks | {total_rays}rays {total_tris}tris {dyn_count}dyn | {calc_time_us}us"))

    return {
        "results": results,
        "count": len(results),
        "rays": total_rays,
        "tris": total_tris,
        "calc_time_us": calc_time_us,
        "worker_logs": debug_logs if debug_logs else [],
    }
