# Exp_Game/engine/worker/raycast.py
"""
Raycast implementations for worker process.
Handles unified raycasting against static grids and dynamic meshes.
"""

import math

# Import math utilities from sibling module
from .math import (
    ray_triangle_intersect,
    compute_facing_normal,
    ray_aabb_intersect,
    ray_sphere_intersect,
    transform_ray_to_local,
    ray_grid_traverse,
)


# ============================================================================
# UNIFIED RAYCAST - Tests both static grid and dynamic meshes
# ============================================================================

def unified_raycast(ray_origin, ray_direction, max_dist, grid_data, dynamic_meshes,
                    debug_logs=None, debug_category=None):
    """
    Test ray against ALL geometry (static grid + dynamic meshes).
    Returns closest hit with source metadata.

    This is the core of the unified physics system - same code path for all geometry.

    Args:
        ray_origin: (x, y, z) ray starting point
        ray_direction: (x, y, z) ray direction (should be normalized for best results)
        max_dist: maximum ray distance
        grid_data: dict with cached static grid data (or None)
                   Expected keys: cells, triangles, bounds_min, bounds_max, cell_size, grid_dims
        dynamic_meshes: list of dicts with transformed dynamic mesh data
                       Each: {"obj_id": id, "triangles": [...], "bounding_sphere": ((cx,cy,cz), radius)}
        debug_logs: optional list to append debug messages
        debug_category: category name for debug logs (e.g., "UNIFIED-RAY")

    Returns:
        dict with:
            "hit": True/False
            "dist": float (distance to hit)
            "normal": (nx, ny, nz) surface normal
            "pos": (x, y, z) hit position
            "source": "static" or "dynamic"
            "obj_id": object ID if dynamic hit (None for static)
            "tris_tested": int (number of triangles tested)
            "cells_traversed": int (number of grid cells checked)
    """
    best_hit = None
    best_dist = max_dist
    total_tris = 0
    total_cells = 0

    ox, oy, oz = ray_origin
    dx, dy, dz = ray_direction

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Test Static Grid (3D DDA traversal - efficient)
    # ─────────────────────────────────────────────────────────────────────────
    if grid_data is not None:
        cells = grid_data.get("cells", {})
        triangles = grid_data.get("triangles", [])
        bounds_min = grid_data.get("bounds_min", (0, 0, 0))
        bounds_max = grid_data.get("bounds_max", (0, 0, 0))
        cell_size = grid_data.get("cell_size", 1.0)
        grid_dims = grid_data.get("grid_dims", (1, 1, 1))

        min_x, min_y, min_z = bounds_min
        max_x, max_y, max_z = bounds_max
        nx_grid, ny_grid, nz_grid = grid_dims

        # Clamp start position to grid bounds
        start_x = max(min_x, min(max_x - 0.001, ox))
        start_y = max(min_y, min(max_y - 0.001, oy))
        start_z = max(min_z, min(max_z - 0.001, oz))

        ix = int((start_x - min_x) / cell_size)
        iy = int((start_y - min_y) / cell_size)
        iz = int((start_z - min_z) / cell_size)

        ix = max(0, min(nx_grid - 1, ix))
        iy = max(0, min(ny_grid - 1, iy))
        iz = max(0, min(nz_grid - 1, iz))

        # DDA setup
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
        cells_traversed = 0
        max_cells = nx_grid + ny_grid + nz_grid + 20
        t_current = 0.0

        while cells_traversed < max_cells:
            cells_traversed += 1
            total_cells += 1

            if t_current > best_dist:
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
                    hit, dist, hit_pos = ray_triangle_intersect(
                        ray_origin, ray_direction, tri[0], tri[1], tri[2]
                    )

                    if hit and dist < best_dist:
                        best_dist = dist
                        # Use facing normal (handles backface hits correctly)
                        normal = compute_facing_normal(tri[0], tri[1], tri[2], ray_direction)
                        best_hit = {
                            "hit": True,
                            "dist": dist,
                            "normal": normal,
                            "pos": hit_pos,
                            "source": "static",
                            "obj_id": None
                        }

            # Find next cell
            t_next = min(t_max_x, t_max_y, t_max_z)

            # Early out if we found a hit closer than next cell
            if best_hit is not None and best_dist <= t_next:
                break

            if t_max_x <= t_max_y and t_max_x <= t_max_z:
                ix += step_x
                t_current = t_max_x
                t_max_x += t_delta_x
            elif t_max_y <= t_max_z:
                iy += step_y
                t_current = t_max_y
                t_max_y += t_delta_y
            else:
                iz += step_z
                t_current = t_max_z
                t_max_z += t_delta_z

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: Test Dynamic Meshes (LOCAL-SPACE + GRID ACCELERATED)
    # Same optimization as static: transform ray, use spatial grid
    # ─────────────────────────────────────────────────────────────────────────
    if dynamic_meshes:
        for dyn_mesh in dynamic_meshes:
            obj_id = dyn_mesh.get("obj_id")
            triangles = dyn_mesh.get("triangles", [])
            aabb = dyn_mesh.get("aabb")
            bounding_sphere = dyn_mesh.get("bounding_sphere")
            inv_matrix = dyn_mesh.get("inv_matrix")
            matrix = dyn_mesh.get("matrix")
            grid = dyn_mesh.get("grid")

            # AABB rejection first (world-space AABB vs world-space ray)
            if not aabb:
                continue  # Skip meshes without AABB - no fallback
            aabb_min, aabb_max = aabb
            if not ray_aabb_intersect(ray_origin, ray_direction,
                                      aabb_min, aabb_max, best_dist):
                continue  # Skip ALL triangles in this mesh!

            # Transform ray to LOCAL space (O(1) regardless of mesh complexity)
            if inv_matrix is None:
                continue  # Can't test without inverse matrix

            local_origin, local_direction, dir_len = transform_ray_to_local(ray_origin, ray_direction, inv_matrix)

            # Convert best_dist (world) to local space for comparisons
            # t_local = t_world * dir_len
            best_dist_local = best_dist * dir_len

            # Test triangles in LOCAL space (grid-accelerated if available)
            hit_dist = None
            hit_tri_idx = None

            if not grid:
                continue  # Skip meshes without spatial grid - no fallback

            # GRID-ACCELERATED: O(cells) instead of O(N)
            tris_counter = [0]
            grid_result = ray_grid_traverse(
                local_origin, local_direction, best_dist_local, grid, triangles, tris_counter
            )
            total_tris += tris_counter[0]
            if grid_result:
                hit_dist, hit_tri_idx = grid_result

            # Process hit - transform normal back to world space
            if hit_dist is not None and hit_tri_idx is not None:
                # Convert hit distance from local to world space
                # t_world = t_local / dir_len
                hit_dist_world = hit_dist / dir_len

                if hit_dist_world < best_dist:
                    best_dist = hit_dist_world
                    tri = triangles[hit_tri_idx]
                    # Use facing normal (handles backface hits correctly)
                    # Use local_direction since triangles are in local space
                    local_normal = compute_facing_normal(tri[0], tri[1], tri[2], local_direction)

                    # Transform normal to world space using INVERSE TRANSPOSE
                    # N_world = normalize((M^-1)^T @ N_local)
                    # For row-major matrix, transpose means using columns instead of rows
                    if inv_matrix:
                        inv = inv_matrix
                        nx, ny, nz = local_normal
                        # Use columns of inv_matrix (= rows of inv_matrix transposed)
                        wnx = inv[0]*nx + inv[4]*ny + inv[8]*nz
                        wny = inv[1]*nx + inv[5]*ny + inv[9]*nz
                        wnz = inv[2]*nx + inv[6]*ny + inv[10]*nz
                        n_len = math.sqrt(wnx*wnx + wny*wny + wnz*wnz)
                        if n_len > 1e-10:
                            wnx /= n_len
                            wny /= n_len
                            wnz /= n_len
                        world_normal = (wnx, wny, wnz)
                    else:
                        world_normal = local_normal

                    # Compute world-space hit position using world distance
                    hit_pos = (
                        ray_origin[0] + ray_direction[0] * hit_dist_world,
                        ray_origin[1] + ray_direction[1] * hit_dist_world,
                        ray_origin[2] + ray_direction[2] * hit_dist_world
                    )

                    best_hit = {
                        "hit": True,
                        "dist": hit_dist_world,
                        "normal": world_normal,
                        "pos": hit_pos,
                        "source": "dynamic",
                        "obj_id": obj_id
                    }

    # ─────────────────────────────────────────────────────────────────────────
    # Build result
    # ─────────────────────────────────────────────────────────────────────────
    if best_hit is None:
        result = {
            "hit": False,
            "dist": None,
            "normal": None,
            "pos": None,
            "source": None,
            "obj_id": None,
            "tris_tested": total_tris,
            "cells_traversed": total_cells
        }
    else:
        best_hit["tris_tested"] = total_tris
        best_hit["cells_traversed"] = total_cells
        result = best_hit

    # Debug logging
    if debug_logs is not None and debug_category:
        if result["hit"]:
            debug_logs.append((debug_category,
                f"HIT source={result['source']} dist={result['dist']:.3f}m "
                f"tris={total_tris} cells={total_cells}"))
        else:
            debug_logs.append((debug_category,
                f"MISS tris={total_tris} cells={total_cells}"))

    return result


# ============================================================================
# UNIFIED RAY CAST - THE SINGLE FUNCTION FOR ALL PHYSICS
# ============================================================================

def cast_ray(ray_origin, ray_direction, max_dist, grid_data, dynamic_meshes, tris_counter=None):
    """
    THE SINGLE UNIFIED RAY FUNCTION - Tests ALL geometry (static + dynamic).

    Both static and dynamic use identical physics:
    - Same ray-triangle intersection
    - Same spatial grid acceleration
    - Same normal computation
    - Closest hit wins regardless of source

    Args:
        ray_origin: (x, y, z) world-space ray start
        ray_direction: (x, y, z) world-space direction (should be normalized)
        max_dist: maximum ray distance
        grid_data: static grid cache (or None)
        dynamic_meshes: list of dynamic mesh dicts (or None/empty)
        tris_counter: optional [count] list to track triangles tested

    Returns:
        (dist, normal, source, obj_id) if hit:
            dist: float distance to hit
            normal: (nx, ny, nz) surface normal
            source: "static" or "dynamic"
            obj_id: object ID if dynamic (None if static)
        None if no hit
    """
    result = unified_raycast(ray_origin, ray_direction, max_dist, grid_data, dynamic_meshes)

    if tris_counter is not None:
        tris_counter[0] += result.get("tris_tested", 0)

    if result["hit"]:
        return (
            result["dist"],
            result["normal"],
            result["source"],
            result["obj_id"]
        )
    return None


# ============================================================================
# SIMPLE DYNAMIC MESH RAY HELPER (for integration with existing ray loops)
# ============================================================================

def test_dynamic_meshes_ray(ray_origin, ray_direction, max_dist, dynamic_meshes, tris_tested_counter=None, debug_log=None):
    """
    Test a single ray against dynamic meshes with AABB + sphere culling.
    Uses LOCAL-SPACE ray testing for O(1) transform cost per mesh regardless of tri count.

    OPTIMIZATION: Triangles are stored in LOCAL space. We transform the ray into
    local space (O(1)) instead of transforming all triangles to world space (O(N)).
    This makes performance independent of mesh complexity.

    Args:
        ray_origin: (x, y, z) ray starting point (WORLD space)
        ray_direction: (x, y, z) ray direction (WORLD space, should be normalized)
        max_dist: maximum ray distance
        dynamic_meshes: list of dicts with {"obj_id", "triangles", "inv_matrix", "matrix", "aabb"}
        tris_tested_counter: optional list [count] to increment for tracking
        debug_log: optional list to append debug info for diagnosis

    Returns:
        (dist, normal, obj_id) if hit, None if no hit
    """
    best_dist = max_dist
    best_normal = None
    best_obj_id = None

    for dyn_mesh in dynamic_meshes:
        obj_id = dyn_mesh.get("obj_id")
        triangles = dyn_mesh.get("triangles", [])
        aabb = dyn_mesh.get("aabb")
        bounding_sphere = dyn_mesh.get("bounding_sphere")
        inv_matrix = dyn_mesh.get("inv_matrix")
        matrix = dyn_mesh.get("matrix")

        # AABB rejection first (world-space AABB, world-space ray)
        if not aabb:
            continue  # Skip meshes without AABB - no fallback
        aabb_min, aabb_max = aabb
        if not ray_aabb_intersect(ray_origin, ray_direction, aabb_min, aabb_max, best_dist):
            if debug_log is not None:
                debug_log.append(f"AABB_REJECT ray=({ray_origin[0]:.1f},{ray_origin[1]:.1f},{ray_origin[2]:.1f}) dir=({ray_direction[0]:.2f},{ray_direction[1]:.2f},{ray_direction[2]:.2f}) aabb=[({aabb_min[0]:.1f},{aabb_min[1]:.1f},{aabb_min[2]:.1f})->({aabb_max[0]:.1f},{aabb_max[1]:.1f},{aabb_max[2]:.1f})]")
            continue  # Skip ALL triangles in this mesh!

        # ═══════════════════════════════════════════════════════════════════
        # OPTIMIZED: Transform ray to LOCAL space (O(1) per mesh)
        # Instead of transforming N triangles to world space (O(N))
        # ═══════════════════════════════════════════════════════════════════
        if inv_matrix is None:
            # No inverse matrix - skip (shouldn't happen with new caches)
            continue

        local_origin, local_direction, dir_len = transform_ray_to_local(ray_origin, ray_direction, inv_matrix)

        # Convert best_dist (world) to local space for comparisons
        # t_local = t_world * dir_len
        best_dist_local = best_dist * dir_len

        # Get spatial grid if available
        grid = dyn_mesh.get("grid")

        hit_dist = None
        hit_tri_idx = None

        # ═══════════════════════════════════════════════════════════════════
        # GRID-ACCELERATED: Use 3D-DDA (O(cells) vs O(N))
        # ═══════════════════════════════════════════════════════════════════
        if not grid:
            continue  # Skip meshes without spatial grid - no fallback

        grid_result = ray_grid_traverse(
            local_origin, local_direction, best_dist_local, grid, triangles, tris_tested_counter
        )
        if grid_result:
            hit_dist, hit_tri_idx = grid_result

        # Process hit result
        if hit_dist is not None and hit_tri_idx is not None:
            # Convert hit distance from local to world space
            # t_world = t_local / dir_len
            hit_dist_world = hit_dist / dir_len

            if hit_dist_world < best_dist:
                best_dist = hit_dist_world
                tri = triangles[hit_tri_idx]
                # Use facing normal (handles backface hits correctly)
                # Use local_direction since triangles are in local space
                local_normal = compute_facing_normal(tri[0], tri[1], tri[2], local_direction)
                # Transform normal to world space using INVERSE TRANSPOSE
                # N_world = normalize((M^-1)^T @ N_local)
                # For row-major matrix, transpose means using columns instead of rows
                if inv_matrix:
                    inv = inv_matrix
                    nx, ny, nz = local_normal
                    # Use columns of inv_matrix (= rows of inv_matrix transposed)
                    wnx = inv[0]*nx + inv[4]*ny + inv[8]*nz
                    wny = inv[1]*nx + inv[5]*ny + inv[9]*nz
                    wnz = inv[2]*nx + inv[6]*ny + inv[10]*nz
                    # Normalize
                    n_len = math.sqrt(wnx*wnx + wny*wny + wnz*wnz)
                    if n_len > 1e-10:
                        wnx /= n_len
                        wny /= n_len
                        wnz /= n_len
                    best_normal = (wnx, wny, wnz)
                else:
                    best_normal = local_normal
                best_obj_id = obj_id

    if best_normal is not None:
        return (best_dist, best_normal, best_obj_id)
    return None
