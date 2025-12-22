# Exp_Game/engine/worker/reactions/hitscan.py
"""
Hitscan raycast batching - runs in worker process (NO bpy).

Handles:
- Batch raycasting for instant-hit weapons
- Returns hit locations and normals for each ray
"""

import time
import math

# Import raycast from sibling module
from ..raycast import unified_raycast


def handle_hitscan_batch(job_data: dict, cached_grid, cached_dynamic_meshes, cached_dynamic_transforms) -> dict:
    """
    Handle HITSCAN_BATCH job.
    Performs multiple instant raycasts in a single job.

    Input job_data:
        {
            "rays": [
                {
                    "id": int,              # unique ray ID for result mapping
                    "origin": (x, y, z),
                    "direction": (dx, dy, dz),
                    "max_range": float,
                    "owner_key": int,       # reaction owner for impact events
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
            "results": [
                {
                    "id": int,
                    "owner_key": int,
                    "hit": bool,
                    "pos": (x, y, z) | None,
                    "normal": (nx, ny, nz) | None,
                    "distance": float | None,
                    "source": "static" | "dynamic" | None,
                    "obj_id": int | None,
                },
                ...
            ],
            "hits": int,
            "misses": int,
            "calc_time_us": float,
            "tris_tested": int,
            "logs": [(category, message), ...],
        }
    """
    calc_start = time.perf_counter()
    logs = []

    rays = job_data.get("rays", [])
    dynamic_transforms = job_data.get("dynamic_transforms", {})

    if not rays:
        return {
            "success": True,
            "results": [],
            "hits": 0,
            "misses": 0,
            "calc_time_us": 0,
            "tris_tested": 0,
            "logs": [],
        }

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
    transforms_received = len(dynamic_transforms)
    transforms_matched = 0
    for obj_id, (matrix_4x4, world_aabb, inv_matrix) in cached_dynamic_transforms.items():
        cached = cached_dynamic_meshes.get(obj_id)
        if cached is None or inv_matrix is None:
            continue
        transforms_matched += 1

        unified_dynamic_meshes.append({
            "obj_id": obj_id,
            "triangles": cached["triangles"],
            "matrix": matrix_4x4,
            "inv_matrix": inv_matrix,
            "bounding_sphere": None,
            "aabb": world_aabb,
            "grid": cached.get("grid"),
        })

    # Debug: log dynamic mesh availability
    if transforms_received > 0 or len(cached_dynamic_meshes) > 0:
        logs.append(("HITSCAN", f"DYNAMIC_MESHES transforms_received={transforms_received} cached_meshes={len(cached_dynamic_meshes)} matched={transforms_matched} available={len(unified_dynamic_meshes)}"))

    # Process rays
    results = []
    hits = 0
    misses = 0
    total_tris = 0

    for ray in rays:
        ray_id = ray.get("id", 0)
        owner_key = ray.get("owner_key", 0)
        origin = ray.get("origin", (0, 0, 0))
        direction = ray.get("direction", (0, 0, 1))
        max_range = ray.get("max_range", 100.0)

        # Normalize direction
        dx, dy, dz = direction
        d_len = math.sqrt(dx*dx + dy*dy + dz*dz)
        if d_len > 1e-12:
            direction = (dx / d_len, dy / d_len, dz / d_len)

        # Cast ray
        result = unified_raycast(
            origin, direction, max_range,
            grid_data, unified_dynamic_meshes
        )
        total_tris += result.get("tris_tested", 0)

        if result.get("hit"):
            hits += 1
            results.append({
                "id": ray_id,
                "owner_key": owner_key,
                "hit": True,
                "pos": result["pos"],
                "normal": result["normal"],
                "distance": result["dist"],
                "source": result["source"],
                "obj_id": result.get("obj_id"),
            })
        else:
            misses += 1
            # Calculate miss endpoint
            miss_pos = (
                origin[0] + direction[0] * max_range,
                origin[1] + direction[1] * max_range,
                origin[2] + direction[2] * max_range,
            )
            results.append({
                "id": ray_id,
                "owner_key": owner_key,
                "hit": False,
                "pos": miss_pos,
                "normal": None,
                "distance": max_range,
                "source": None,
                "obj_id": None,
            })

    calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

    logs.append(("HITSCAN", f"BATCH rays={len(rays)} hits={hits} misses={misses} tris={total_tris} {calc_time_us:.0f}us"))

    return {
        "success": True,
        "results": results,
        "hits": hits,
        "misses": misses,
        "calc_time_us": calc_time_us,
        "tris_tested": total_tris,
        "logs": logs,
    }
