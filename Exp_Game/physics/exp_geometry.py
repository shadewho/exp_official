# Exp_Game/physics/exp_geometry.py
"""
Geometry Utilities for Raycast Offloading

Extracts and processes mesh geometry for worker-based raycasting.
Includes triangle extraction and spatial acceleration structures.
"""

import bpy
from mathutils import Vector


# ============================================================================
# Triangle Extraction
# ============================================================================

def extract_static_triangles(static_bvh_source_objects):
    """
    Extract raw triangle data from static mesh objects.

    Converts Blender mesh objects into world-space triangles that can be
    serialized and sent to worker processes for raycasting.

    Args:
        static_bvh_source_objects: List of Blender objects to extract from

    Returns:
        List of triangles, where each triangle is ((v0_x, v0_y, v0_z), (v1_x, v1_y, v1_z), (v2_x, v2_y, v2_z))
    """
    triangles = []

    for obj in static_bvh_source_objects:
        if obj.type != 'MESH':
            continue

        mesh = obj.data
        matrix = obj.matrix_world

        # Ensure mesh has polygons
        if not mesh.polygons:
            continue

        for poly in mesh.polygons:
            # Get world-space vertex positions
            verts = [matrix @ mesh.vertices[i].co for i in poly.vertices]

            # Triangulate polygon (handle quads and n-gons)
            if len(verts) == 3:
                # Already a triangle
                triangles.append((
                    (verts[0].x, verts[0].y, verts[0].z),
                    (verts[1].x, verts[1].y, verts[1].z),
                    (verts[2].x, verts[2].y, verts[2].z)
                ))
            elif len(verts) == 4:
                # Quad - split into 2 triangles (fan triangulation)
                triangles.append((
                    (verts[0].x, verts[0].y, verts[0].z),
                    (verts[1].x, verts[1].y, verts[1].z),
                    (verts[2].x, verts[2].y, verts[2].z)
                ))
                triangles.append((
                    (verts[0].x, verts[0].y, verts[0].z),
                    (verts[2].x, verts[2].y, verts[2].z),
                    (verts[3].x, verts[3].y, verts[3].z)
                ))
            else:
                # N-gon - fan triangulation from first vertex
                for i in range(1, len(verts) - 1):
                    triangles.append((
                        (verts[0].x, verts[0].y, verts[0].z),
                        (verts[i].x, verts[i].y, verts[i].z),
                        (verts[i + 1].x, verts[i + 1].y, verts[i + 1].z)
                    ))

    return triangles


def get_triangle_stats(triangles):
    """
    Calculate statistics about triangle data.

    Args:
        triangles: List of triangles

    Returns:
        Dictionary with stats: count, bounds, memory estimate
    """
    if not triangles:
        return {
            "count": 0,
            "bounds": None,
            "memory_mb": 0.0
        }

    # Calculate bounds
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    for tri in triangles:
        for v in tri:
            min_x = min(min_x, v[0])
            max_x = max(max_x, v[0])
            min_y = min(min_y, v[1])
            max_y = max(max_y, v[1])
            min_z = min(min_z, v[2])
            max_z = max(max_z, v[2])

    # Estimate memory (each triangle = 9 floats * 8 bytes)
    memory_bytes = len(triangles) * 9 * 8
    memory_mb = memory_bytes / (1024 * 1024)

    return {
        "count": len(triangles),
        "bounds": {
            "min": (min_x, min_y, min_z),
            "max": (max_x, max_y, max_z),
            "size": (max_x - min_x, max_y - min_y, max_z - min_z)
        },
        "memory_mb": memory_mb
    }


# ============================================================================
# Spatial Acceleration (Phase 2 - Uniform Grid)
# ============================================================================

def build_uniform_grid(triangles, cell_size=2.0, context=None):
    """
    Build a uniform 3D grid for spatial acceleration.

    Divides the scene into a 3D grid of cells and assigns triangles
    to all cells they overlap. This reduces raycast complexity from
    O(n) to O(k) where k << n is the number of triangles in traversed cells.

    Args:
        triangles: List of triangles (picklable format)
        cell_size: Size of each grid cell in Blender units (default 2.0m)
        context: Blender context for debug output

    Returns:
        Grid data structure (picklable dict):
        {
            "bounds_min": (x, y, z),
            "bounds_max": (x, y, z),
            "cell_size": float,
            "grid_dims": (nx, ny, nz),
            "cells": {(ix, iy, iz): [tri_indices], ...},
            "triangles": original triangles list,
            "stats": build statistics
        }
    """
    import time
    build_start = time.perf_counter()

    if not triangles:
        return None

    # Get debug flag
    debug = False
    if context:
        debug = getattr(context.scene, 'dev_debug_kcc_physics', False)

    # ========== Step 1: Calculate Bounds ==========
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    for tri in triangles:
        for v in tri:
            min_x = min(min_x, v[0])
            max_x = max(max_x, v[0])
            min_y = min(min_y, v[1])
            max_y = max(max_y, v[1])
            min_z = min(min_z, v[2])
            max_z = max(max_z, v[2])

    # Add small padding to avoid edge cases
    padding = cell_size * 0.01
    min_x -= padding
    min_y -= padding
    min_z -= padding
    max_x += padding
    max_y += padding
    max_z += padding

    bounds_min = (min_x, min_y, min_z)
    bounds_max = (max_x, max_y, max_z)

    # ========== Step 2: Calculate Grid Dimensions ==========
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z

    # Grid dimensions (at least 1 cell per axis)
    nx = max(1, int(size_x / cell_size) + 1)
    ny = max(1, int(size_y / cell_size) + 1)
    nz = max(1, int(size_z / cell_size) + 1)

    grid_dims = (nx, ny, nz)
    total_cells = nx * ny * nz

    if debug:
        print(f"\n[Grid Build] ========== BUILDING SPATIAL GRID ==========")
        print(f"[Grid Build] Cell size: {cell_size:.2f}m")
        print(f"[Grid Build] Scene size: ({size_x:.2f}, {size_y:.2f}, {size_z:.2f})")
        print(f"[Grid Build] Grid dimensions: {nx} x {ny} x {nz} = {total_cells:,} cells")

    # ========== Step 3: Assign Triangles to Cells ==========
    cells = {}  # (ix, iy, iz) -> [triangle indices]

    for tri_idx, tri in enumerate(triangles):
        # Get triangle AABB
        tri_min_x = min(tri[0][0], tri[1][0], tri[2][0])
        tri_max_x = max(tri[0][0], tri[1][0], tri[2][0])
        tri_min_y = min(tri[0][1], tri[1][1], tri[2][1])
        tri_max_y = max(tri[0][1], tri[1][1], tri[2][1])
        tri_min_z = min(tri[0][2], tri[1][2], tri[2][2])
        tri_max_z = max(tri[0][2], tri[1][2], tri[2][2])

        # Find cell range this triangle overlaps
        ix_min = max(0, int((tri_min_x - min_x) / cell_size))
        ix_max = min(nx - 1, int((tri_max_x - min_x) / cell_size))
        iy_min = max(0, int((tri_min_y - min_y) / cell_size))
        iy_max = min(ny - 1, int((tri_max_y - min_y) / cell_size))
        iz_min = max(0, int((tri_min_z - min_z) / cell_size))
        iz_max = min(nz - 1, int((tri_max_z - min_z) / cell_size))

        # Add triangle to all overlapping cells
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                for iz in range(iz_min, iz_max + 1):
                    key = (ix, iy, iz)
                    if key not in cells:
                        cells[key] = []
                    cells[key].append(tri_idx)

    build_end = time.perf_counter()
    build_time_ms = (build_end - build_start) * 1000

    # ========== Step 4: Calculate Statistics ==========
    non_empty_cells = len(cells)
    total_refs = sum(len(tris) for tris in cells.values())
    avg_tris_per_cell = total_refs / non_empty_cells if non_empty_cells > 0 else 0
    max_tris_in_cell = max(len(tris) for tris in cells.values()) if cells else 0
    min_tris_in_cell = min(len(tris) for tris in cells.values()) if cells else 0

    # Memory estimate: dict overhead + list overhead + int indices
    # Rough estimate: 100 bytes per cell + 8 bytes per triangle reference
    memory_bytes = non_empty_cells * 100 + total_refs * 8
    memory_kb = memory_bytes / 1024

    stats = {
        "build_time_ms": build_time_ms,
        "total_cells": total_cells,
        "non_empty_cells": non_empty_cells,
        "empty_cells": total_cells - non_empty_cells,
        "fill_ratio": non_empty_cells / total_cells if total_cells > 0 else 0,
        "total_triangle_refs": total_refs,
        "avg_tris_per_cell": avg_tris_per_cell,
        "max_tris_in_cell": max_tris_in_cell,
        "min_tris_in_cell": min_tris_in_cell,
        "memory_kb": memory_kb,
        "triangle_count": len(triangles)
    }

    if debug:
        print(f"[Grid Build] ----- Cell Statistics -----")
        print(f"[Grid Build] Non-empty cells: {non_empty_cells:,} / {total_cells:,} ({stats['fill_ratio']*100:.1f}% fill)")
        print(f"[Grid Build] Triangle references: {total_refs:,} (avg {avg_tris_per_cell:.1f} per cell)")
        print(f"[Grid Build] Triangles per cell: min={min_tris_in_cell}, max={max_tris_in_cell}")
        print(f"[Grid Build] Grid memory: {memory_kb:.2f} KB")
        print(f"[Grid Build] Build time: {build_time_ms:.2f}ms")
        print(f"[Grid Build] ============================================\n")

    # ========== Return Picklable Structure ==========
    return {
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "cell_size": cell_size,
        "grid_dims": grid_dims,
        "cells": cells,
        "triangles": triangles,
        "stats": stats
    }


def print_grid_report(grid_data, context=None):
    """
    Print a detailed report about the spatial grid.

    Args:
        grid_data: Grid data structure from build_uniform_grid()
        context: Blender context (for debug gate)
    """
    if not grid_data:
        print("[Grid Report] No grid data available")
        return

    # Check debug flag
    if context:
        if not getattr(context.scene, 'dev_debug_kcc_physics', False):
            return

    stats = grid_data["stats"]
    dims = grid_data["grid_dims"]

    print("\n" + "="*60)
    print("[Grid Report] Spatial Acceleration Grid")
    print("="*60)
    print(f"Grid Dimensions: {dims[0]} x {dims[1]} x {dims[2]}")
    print(f"Cell Size: {grid_data['cell_size']:.2f}m")
    print(f"Bounds: ({grid_data['bounds_min'][0]:.2f}, {grid_data['bounds_min'][1]:.2f}, {grid_data['bounds_min'][2]:.2f})")
    print(f"     to ({grid_data['bounds_max'][0]:.2f}, {grid_data['bounds_max'][1]:.2f}, {grid_data['bounds_max'][2]:.2f})")
    print("-"*60)
    print(f"Total Cells: {stats['total_cells']:,}")
    print(f"Non-Empty Cells: {stats['non_empty_cells']:,} ({stats['fill_ratio']*100:.1f}%)")
    print(f"Triangle References: {stats['total_triangle_refs']:,}")
    print(f"Avg Triangles/Cell: {stats['avg_tris_per_cell']:.1f}")
    print(f"Max Triangles/Cell: {stats['max_tris_in_cell']}")
    print("-"*60)
    print(f"Build Time: {stats['build_time_ms']:.2f}ms")
    print(f"Grid Memory: {stats['memory_kb']:.2f} KB")
    print(f"Original Triangles: {stats['triangle_count']:,}")

    # Performance prediction
    expected_tests = stats['avg_tris_per_cell'] * 3  # ~3 cells per typical ray
    brute_force = stats['triangle_count']
    speedup = brute_force / expected_tests if expected_tests > 0 else 0

    print("-"*60)
    print(f"[Performance Prediction]")
    print(f"  Brute force: ~{brute_force:,} triangle tests/ray")
    print(f"  With grid: ~{int(expected_tests)} triangle tests/ray (est.)")
    print(f"  Expected speedup: ~{speedup:.0f}x")
    print("="*60 + "\n")


# ============================================================================
# Diagnostic Functions
# ============================================================================

def print_geometry_report(triangles, context=None):
    """
    Print a diagnostic report about extracted geometry.

    Args:
        triangles: List of triangles
        context: Blender context (optional, for debug gate)
    """
    stats = get_triangle_stats(triangles)

    # Check if debug output is enabled
    if context:
        from ..developer.dev_debug_gate import should_print_debug
        if not should_print_debug("raycast_offload"):
            return

    print("\n" + "="*60)
    print("[Geometry] Triangle Extraction Report")
    print("="*60)
    print(f"Triangle Count: {stats['count']:,}")

    if stats['bounds']:
        bounds = stats['bounds']
        print(f"Scene Bounds:")
        print(f"  Min: ({bounds['min'][0]:.2f}, {bounds['min'][1]:.2f}, {bounds['min'][2]:.2f})")
        print(f"  Max: ({bounds['max'][0]:.2f}, {bounds['max'][1]:.2f}, {bounds['max'][2]:.2f})")
        print(f"  Size: ({bounds['size'][0]:.2f}, {bounds['size'][1]:.2f}, {bounds['size'][2]:.2f})")

    print(f"Memory Estimate: {stats['memory_mb']:.2f} MB")

    # Performance recommendation
    if stats['count'] < 5000:
        print("✅ Scene size: SMALL - Mesh soup raycasting will work well")
    elif stats['count'] < 20000:
        print("⚠️  Scene size: MEDIUM - Consider spatial acceleration (Phase 2)")
    else:
        print("❌ Scene size: LARGE - Spatial acceleration required (Phase 2)")

    print("="*60 + "\n")
