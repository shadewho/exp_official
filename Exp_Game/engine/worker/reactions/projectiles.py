# Exp_Game/engine/worker/reactions/projectiles.py
"""
Projectile physics simulation - runs in worker process (NO bpy).

Handles:
- Gravity integration
- Sweep raycasting against static + dynamic geometry
- Impact detection and location reporting
"""

import time
import math

# Import raycast from sibling module
from ..raycast import unified_raycast


# Worker-side projectile state
# Each entry: {
#     "id": int,           # unique projectile ID
#     "pos": (x, y, z),    # current position
#     "vel": (vx, vy, vz), # current velocity
#     "gravity": float,    # gravity acceleration (negative = down)
#     "end_time": float,   # when projectile expires
#     "stop_on_contact": bool,
#     "owner_key": int,    # reaction owner for impact events
# }
_active_projectiles = []
_next_projectile_id = 0


def reset_projectile_state():
    """Reset all projectile state. Called on game reset."""
    global _active_projectiles, _next_projectile_id
    _active_projectiles = []
    _next_projectile_id = 0


def handle_projectile_update_batch(job_data: dict, cached_grid, cached_dynamic_meshes, cached_dynamic_transforms) -> dict:
    """
    Handle PROJECTILE_UPDATE_BATCH job.
    Advances all active projectiles by dt, performs sweep raycasts.

    Input job_data:
        {
            "dt": float,                    # time step in seconds
            "game_time": float,             # current game time
            "new_projectiles": [            # projectiles to spawn this frame
                {
                    "pos": (x, y, z),
                    "vel": (vx, vy, vz),
                    "gravity": float,
                    "lifetime": float,
                    "stop_on_contact": bool,
                    "owner_key": int,
                },
                ...
            ],
            "dynamic_transforms": {         # updated transforms for dynamic meshes
                obj_id: matrix_16_tuple,
                ...
            },
        }

    Returns:
        {
            "success": bool,
            "updated_projectiles": [        # all active projectiles after update
                {
                    "id": int,
                    "pos": (x, y, z),
                    "vel": (vx, vy, vz),
                    "active": bool,
                },
                ...
            ],
            "impacts": [                    # projectiles that hit something
                {
                    "id": int,
                    "owner_key": int,
                    "pos": (x, y, z),
                    "normal": (nx, ny, nz),
                    "source": "static" | "dynamic",
                    "obj_id": int | None,
                },
                ...
            ],
            "expired": [int, ...],          # IDs of projectiles that expired
            "active_count": int,
            "calc_time_us": float,
            "rays_cast": int,
            "tris_tested": int,
            "logs": [(category, message), ...],
        }
    """
    global _active_projectiles, _next_projectile_id

    calc_start = time.perf_counter()
    logs = []

    dt = job_data.get("dt", 1/30)
    game_time = job_data.get("game_time", 0.0)
    new_projectiles = job_data.get("new_projectiles", [])
    dynamic_transforms = job_data.get("dynamic_transforms", {})

    # Update dynamic transform cache
    from ..math import transform_aabb_by_matrix, invert_matrix_4x4
    for obj_id, matrix_4x4 in dynamic_transforms.items():
        cached = cached_dynamic_meshes.get(obj_id)
        if cached is None:
            continue
        local_aabb = cached.get("local_aabb")
        world_aabb = transform_aabb_by_matrix(local_aabb, matrix_4x4) if local_aabb else None
        inv_matrix = invert_matrix_4x4(matrix_4x4)
        cached_dynamic_transforms[obj_id] = (matrix_4x4, world_aabb, inv_matrix)

    # Spawn new projectiles (preserve client_id for matching)
    for proj in new_projectiles:
        _next_projectile_id += 1
        _active_projectiles.append({
            "id": _next_projectile_id,
            "client_id": int(proj.get("client_id", 0)),  # For matching with main thread
            "pos": tuple(proj.get("pos", (0, 0, 0))),
            "vel": tuple(proj.get("vel", (0, 0, 0))),
            "gravity": float(proj.get("gravity", -9.8)),
            "end_time": game_time + float(proj.get("lifetime", 3.0)),
            "stop_on_contact": bool(proj.get("stop_on_contact", True)),
            "owner_key": int(proj.get("owner_key", 0)),
        })

    if new_projectiles:
        logs.append(("PROJECTILE", f"SPAWNED {len(new_projectiles)} projectiles, total={len(_active_projectiles)}"))

    # Build grid data for raycast
    grid_data = None
    if cached_grid is not None:
        grid_data = {
            "cells": cached_grid.get("cells", {}),
            "triangles": cached_grid.get("triangles", []),
            "bounds_min": cached_grid.get("bounds_min", (0, 0, 0)),
            "bounds_max": cached_grid.get("bounds_max", (0, 0, 0)),
            "cell_size": cached_grid.get("cell_size", 1.0),
            "grid_dims": cached_grid.get("grid_dims", (1, 1, 1)),
        }

    # Build dynamic mesh list for raycast
    unified_dynamic_meshes = []
    for obj_id, (matrix_4x4, world_aabb, inv_matrix) in cached_dynamic_transforms.items():
        cached = cached_dynamic_meshes.get(obj_id)
        if cached is None or inv_matrix is None:
            continue

        unified_dynamic_meshes.append({
            "obj_id": obj_id,
            "triangles": cached["triangles"],
            "matrix": matrix_4x4,
            "inv_matrix": inv_matrix,
            "bounding_sphere": None,
            "aabb": world_aabb,
            "grid": cached.get("grid"),
        })

    # Update projectiles
    updated_projectiles = []
    impacts = []
    expired = []
    rays_cast = 0
    total_tris = 0

    keep = []
    for proj in _active_projectiles:
        proj_id = proj["id"]

        # Check expiration
        if game_time >= proj["end_time"]:
            expired.append({
                "id": proj_id,
                "client_id": proj.get("client_id", 0),
            })
            continue

        # Get current state
        px, py, pz = proj["pos"]
        vx, vy, vz = proj["vel"]
        g = proj["gravity"]

        # Integrate gravity
        vz_new = vz + g * dt

        # Predict new position
        new_px = px + vx * dt
        new_py = py + vy * dt
        new_pz = pz + vz_new * dt

        # Sweep raycast from old to new position
        seg_x = new_px - px
        seg_y = new_py - py
        seg_z = new_pz - pz
        seg_len = math.sqrt(seg_x*seg_x + seg_y*seg_y + seg_z*seg_z)

        hit = False
        hit_result = None

        if seg_len > 1e-7:
            rays_cast += 1
            ray_dir = (seg_x / seg_len, seg_y / seg_len, seg_z / seg_len)
            result = unified_raycast(
                (px, py, pz), ray_dir, seg_len,
                grid_data, unified_dynamic_meshes
            )
            total_tris += result.get("tris_tested", 0)

            if result.get("hit"):
                hit = True
                hit_result = result

        if hit and proj["stop_on_contact"]:
            # Record impact (include client_id for matching)
            impacts.append({
                "id": proj_id,
                "client_id": proj.get("client_id", 0),
                "owner_key": proj["owner_key"],
                "pos": hit_result["pos"],
                "normal": hit_result["normal"],
                "source": hit_result["source"],
                "obj_id": hit_result.get("obj_id"),
            })

            # Report final position at impact
            updated_projectiles.append({
                "id": proj_id,
                "client_id": proj.get("client_id", 0),
                "pos": hit_result["pos"],
                "vel": (vx, vy, vz_new),
                "active": False,
            })
            # Don't keep this projectile
            continue

        # Update state and keep
        proj["pos"] = (new_px, new_py, new_pz)
        proj["vel"] = (vx, vy, vz_new)
        keep.append(proj)

        updated_projectiles.append({
            "id": proj_id,
            "client_id": proj.get("client_id", 0),
            "pos": (new_px, new_py, new_pz),
            "vel": (vx, vy, vz_new),
            "active": True,
        })

    _active_projectiles = keep

    calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

    if rays_cast > 0 or impacts:
        logs.append(("PROJECTILE", f"UPDATE active={len(_active_projectiles)} rays={rays_cast} tris={total_tris} impacts={len(impacts)} {calc_time_us:.0f}us"))

    return {
        "success": True,
        "updated_projectiles": updated_projectiles,
        "impacts": impacts,
        "expired": expired,
        "active_count": len(_active_projectiles),
        "calc_time_us": calc_time_us,
        "rays_cast": rays_cast,
        "tris_tested": total_tris,
        "logs": logs,
    }
