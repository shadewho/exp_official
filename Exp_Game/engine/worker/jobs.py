# Exp_Game/engine/worker/jobs.py
"""
Job handlers for worker process.
Contains handler functions for various job types.
"""

import math
import time

# Import raycast functions from sibling module
from .raycast import unified_raycast

# Import math utilities
from .math import (
    transform_aabb_by_matrix,
    invert_matrix_4x4,
)


def handle_camera_occlusion(job_data, cached_grid, cached_dynamic_meshes, cached_dynamic_transforms):
    """
    Handle CAMERA_OCCLUSION_FULL job.
    Uses unified_raycast (same as KCC physics) for camera collision detection.

    Args:
        job_data: dict with ray_origin, ray_direction, max_distance, dynamic_transforms
        cached_grid: static collision grid data
        cached_dynamic_meshes: dict of dynamic mesh data
        cached_dynamic_transforms: dict of {obj_id: (matrix, aabb, inv_matrix)}

    Returns:
        dict with hit info and timing data
    """
    calc_start = time.perf_counter()

    # Ray parameters
    ray_origin = job_data.get("ray_origin", (0.0, 0.0, 0.0))
    ray_direction = job_data.get("ray_direction", (0.0, 0.0, -1.0))
    max_distance = job_data.get("max_distance", 10.0)
    dynamic_transforms_update = job_data.get("dynamic_transforms", {})

    # Normalize direction
    dx, dy, dz = ray_direction
    d_len = math.sqrt(dx*dx + dy*dy + dz*dz)
    if d_len > 1e-12:
        dx /= d_len
        dy /= d_len
        dz /= d_len
    ray_direction_norm = (dx, dy, dz)

    # ─────────────────────────────────────────────────────────────────
    # Build grid_data for unified_raycast (same format as KCC)
    # ─────────────────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────────────────
    # Build unified_dynamic_meshes FROM PERSISTENT CACHE
    # ─────────────────────────────────────────────────────────────────
    # Camera uses the same transform cache as KCC physics.
    # If transforms are sent in the job, update the cache first.
    # Then build mesh list from ALL cached transforms.
    # ─────────────────────────────────────────────────────────────────

    # Update cache with any fresh transforms
    # Mesh triangles are cached via targeted broadcast_job (guaranteed delivery)
    for obj_id, matrix_4x4 in dynamic_transforms_update.items():
        cached = cached_dynamic_meshes.get(obj_id)
        if cached is None:
            continue
        local_aabb = cached.get("local_aabb")
        world_aabb = transform_aabb_by_matrix(local_aabb, matrix_4x4) if local_aabb else None
        # Compute inverse matrix ONCE when transform changes
        inv_matrix = invert_matrix_4x4(matrix_4x4)
        cached_dynamic_transforms[obj_id] = (matrix_4x4, world_aabb, inv_matrix)

    # Build mesh list from ALL cached transforms
    unified_dynamic_meshes = []

    for obj_id, (matrix_4x4, world_aabb, inv_matrix) in cached_dynamic_transforms.items():
        cached = cached_dynamic_meshes.get(obj_id)
        if cached is None:
            continue

        # Skip if inverse matrix failed (singular matrix)
        if inv_matrix is None:
            continue

        local_triangles = cached["triangles"]

        # inv_matrix already cached - no per-frame computation needed!

        # OPTIMIZED: Skip bounding sphere - unified_raycast uses AABB first
        bounding_sphere = None
        if not world_aabb:
            bounding_sphere = ((0, 0, 0), cached.get("radius", 1.0))

        # Add to unified format (same as KCC)
        unified_dynamic_meshes.append({
            "obj_id": obj_id,
            "triangles": local_triangles,
            "matrix": matrix_4x4,
            "inv_matrix": inv_matrix,
            "bounding_sphere": bounding_sphere,
            "aabb": world_aabb,
            "grid": cached.get("grid"),
        })

    # ─────────────────────────────────────────────────────────────────
    # Call unified_raycast - same function used by KCC physics
    # ─────────────────────────────────────────────────────────────────
    result = unified_raycast(
        ray_origin, ray_direction_norm, max_distance,
        grid_data, unified_dynamic_meshes
    )

    calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

    # Extract result
    hit_found = result.get("hit", False)
    hit_distance = result.get("dist") if hit_found else None
    hit_source = result.get("source", "").upper() if hit_found else None

    return {
        "hit": hit_found,
        "hit_distance": hit_distance,
        "hit_source": hit_source,
        "static_triangles_tested": result.get("tris_tested", 0),
        "static_cells_traversed": result.get("cells_traversed", 0),
        "dynamic_triangles_tested": 0,  # Included in tris_tested
        "calc_time_us": calc_time_us,
        "method": "CAMERA_UNIFIED",
        "grid_cached": cached_grid is not None,
        "dynamic_meshes_tested": len(unified_dynamic_meshes),
    }
