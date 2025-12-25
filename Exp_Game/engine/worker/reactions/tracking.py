# Exp_Game/engine/worker/reactions/tracking.py
"""
Tracking movement computation - runs in worker process (NO bpy).

Handles sweep/slide/gravity for non-character object movement.
Character autopilot stays on main thread (just injects keys).

Reuses unified_raycast from worker/raycast.py for collision detection.
"""

import math
from ..raycast import unified_raycast


def handle_tracking_batch(job_data: dict, grid_data, dynamic_meshes, dynamic_transforms) -> dict:
    """
    Batch compute tracking movement with collision.

    Input:
        tracks: [
            {
                "obj_id": int,
                "current_pos": (x, y, z),
                "goal_pos": (x, y, z),
                "speed": float,
                "dt": float,
                "radius": float,
                "height": float,
                "arrive_radius": float,
                "use_gravity": bool,
                "respect_proxy": bool,
            }
        ]

    Output:
        results: [
            {
                "obj_id": int,
                "new_pos": (x, y, z),
                "arrived": bool,
            }
        ]
    """
    tracks = job_data.get("tracks", [])
    debug_logs = job_data.get("debug_logs")  # None if not debugging
    results = []
    total_rays = 0
    total_tris = 0
    walls_hit = 0
    floors_skipped = 0

    for track in tracks:
        obj_id = track["obj_id"]
        pos = list(track["current_pos"])
        goal = track["goal_pos"]
        speed = track["speed"]
        dt = track["dt"]
        radius = track.get("radius", 0.22)
        height = track.get("height", 1.8)
        arrive_radius = track.get("arrive_radius", 0.3)
        use_gravity = track.get("use_gravity", True)
        respect_proxy = track.get("respect_proxy", True)

        # Direction to goal (XY only)
        dx = goal[0] - pos[0]
        dy = goal[1] - pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        # Check arrival
        if dist <= arrive_radius:
            results.append({
                "obj_id": obj_id,
                "new_pos": tuple(pos),
                "arrived": True,
            })
            continue

        # Normalize direction and compute step
        step_len = min(speed * dt, dist)
        fwd = (dx / dist, dy / dist, 0.0)

        # ─────────────────────────────────────────────────────────────────
        # SWEEP FORWARD WITH COLLISION (3 vertical rays: feet/mid/head)
        # ─────────────────────────────────────────────────────────────────
        new_pos = list(pos)

        if respect_proxy and grid_data:
            ray_len = step_len + radius
            best_d = None
            best_n = None

            # 3 vertical rays - raised well above ground to avoid terrain grazing
            # Use minimum 0.4m to clear most terrain geometry
            feet_z = max(0.4, radius * 2)
            mid_z = max(feet_z + 0.2, 0.5 * height)
            head_z = max(mid_z + 0.2, height - radius)

            for z in (feet_z, mid_z, head_z):
                origin = (pos[0], pos[1], pos[2] + z)
                result = unified_raycast(origin, fwd, ray_len, grid_data, dynamic_meshes)
                total_rays += 1
                total_tris += result.get("tris_tested", 0)

                if result["hit"]:
                    # Only count as wall if normal is mostly horizontal
                    # (skip floors/ramps where normal.z > 0.7)
                    n = result["normal"]
                    if abs(n[2]) < 0.7:  # Wall-like surface
                        walls_hit += 1
                        d = result["dist"]
                        if best_d is None or d < best_d:
                            best_d = d
                            best_n = n
                    else:
                        floors_skipped += 1

            # Advance until contact
            if best_d is None:
                # No wall - full step
                new_pos[0] = pos[0] + fwd[0] * step_len
                new_pos[1] = pos[1] + fwd[1] * step_len
            else:
                # Wall hit - advance to wall, then slide
                allow = max(0.0, best_d - radius)
                moved = min(step_len, allow)
                new_pos[0] = pos[0] + fwd[0] * moved
                new_pos[1] = pos[1] + fwd[1] * moved

                # Slide along wall if remainder
                remain = step_len - moved
                if remain > (0.15 * radius) and best_n is not None:
                    # Slide direction: remove normal component from forward (XY plane only)
                    # Normalize the XY part of the normal for proper projection
                    nx, ny = best_n[0], best_n[1]
                    n_xy_len = math.sqrt(nx * nx + ny * ny)
                    if n_xy_len > 1e-9:
                        nx, ny = nx / n_xy_len, ny / n_xy_len
                        dot = fwd[0] * nx + fwd[1] * ny
                        slide = (fwd[0] - nx * dot, fwd[1] - ny * dot, 0.0)
                    else:
                        # Normal is vertical, just keep moving forward
                        slide = (fwd[0], fwd[1], 0.0)

                    slide_len = math.sqrt(slide[0] ** 2 + slide[1] ** 2)

                    if slide_len > 1e-9:
                        slide = (slide[0] / slide_len, slide[1] / slide_len, 0.0)

                        # Check slide direction for walls (3 more rays)
                        best_d2 = None
                        for z in (feet_z, mid_z, head_z):
                            origin2 = (new_pos[0], new_pos[1], new_pos[2] + z)
                            result2 = unified_raycast(origin2, slide, remain + radius, grid_data, dynamic_meshes)
                            total_rays += 1
                            total_tris += result2.get("tris_tested", 0)

                            if result2["hit"]:
                                # Only walls block sliding
                                n2 = result2["normal"]
                                if abs(n2[2]) < 0.7:
                                    walls_hit += 1
                                    d2 = result2["dist"]
                                    if best_d2 is None or d2 < best_d2:
                                        best_d2 = d2
                                else:
                                    floors_skipped += 1

                        allow2 = remain if best_d2 is None else max(0.0, best_d2 - radius)
                        new_pos[0] += slide[0] * allow2
                        new_pos[1] += slide[1] * allow2
        else:
            # No collision - simple move
            new_pos[0] = pos[0] + fwd[0] * step_len
            new_pos[1] = pos[1] + fwd[1] * step_len

        # ─────────────────────────────────────────────────────────────────
        # GRAVITY SNAP (downward raycast)
        # ─────────────────────────────────────────────────────────────────
        if use_gravity and grid_data:
            # Cast from well above object to find ground below
            origin_down = (new_pos[0], new_pos[1], new_pos[2] + height * 0.5)
            direction_down = (0.0, 0.0, -1.0)
            max_down = height + 0.5  # Search down past object height

            result_down = unified_raycast(origin_down, direction_down, max_down, grid_data, dynamic_meshes)
            total_rays += 1
            total_tris += result_down.get("tris_tested", 0)

            if result_down["hit"]:
                # Only snap if ground is floor-like (normal pointing up)
                n = result_down["normal"]
                if n[2] > 0.5:  # Floor surface
                    new_pos[2] = result_down["pos"][2]

        results.append({
            "obj_id": obj_id,
            "new_pos": tuple(new_pos),
            "arrived": False,
        })

    # Add summary log (only if debugging)
    if debug_logs is not None:
        debug_logs.append(("TRACKING", f"WORKER count={len(results)} rays={total_rays} tris={total_tris} walls_hit={walls_hit} floors_skipped={floors_skipped}"))

    return {
        "results": results,
        "count": len(results),
        "rays": total_rays,
        "tris": total_tris,
        "worker_logs": debug_logs if debug_logs else [],
    }
