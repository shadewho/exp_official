# Exp_Game/engine/engine_worker_entry.py
"""
Worker process entry point - ISOLATED from addon imports.
This module is called directly by multiprocessing and does NOT import bpy.
IMPORTANT: This file has NO relative imports to avoid triggering addon __init__.py
"""

import time
import traceback
from queue import Empty


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
# WORKER-SIDE DYNAMIC MESH
# ============================================================================
_cached_dynamic_meshes = {}  # Will hold dynamic mesh data after CACHE_DYNAMIC_MESH jobs

# ============================================================================
# PERSISTENT TRANSFORM CACHE
# ============================================================================
# Main thread sends transform updates only when meshes MOVE.
# Worker caches last known transform for each mesh, enabling:
# - Stationary meshes: zero main-thread work, worker uses cached transform
# - Moving meshes: main thread sends update, worker caches it
# - No activation logic needed - worker always knows where all meshes are
# ============================================================================
_cached_dynamic_transforms = {}  # {obj_id: (matrix_16_tuple, world_aabb)}



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

    # Ray parallel to triangle (test both front and back faces for proper collision)
    # NOTE: Backface culling (a < EPSILON) breaks body integrity ray which shoots upward
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


def compute_facing_normal(v0, v1, v2, ray_direction):
    """
    Compute triangle normal that faces TOWARD the ray origin (not away).

    This handles backface hits correctly: if the geometric normal points
    in the same direction as the ray (backface hit), we flip it.

    Args:
        v0, v1, v2: Triangle vertices as (x, y, z) tuples
        ray_direction: (dx, dy, dz) ray direction (should be normalized)

    Returns:
        (nx, ny, nz) normalized normal vector facing the ray origin
    """
    # Get geometric normal from winding order
    nx, ny, nz = compute_triangle_normal(v0, v1, v2)

    # Check if normal faces away from ray (backface hit)
    # dot(normal, ray_direction) > 0 means they point in same direction
    dx, dy, dz = ray_direction
    dot = nx * dx + ny * dy + nz * dz

    if dot > 0:
        # Backface hit - flip normal to face the ray
        nx = -nx
        ny = -ny
        nz = -nz

    return (nx, ny, nz)


# ============================================================================
# BOUNDING SPHERE INTERSECTION (for quick dynamic mesh rejection)
# ============================================================================

def ray_sphere_intersect(ray_origin, ray_direction, sphere_center, sphere_radius, max_dist):
    """
    Quick test if a ray might hit a sphere (conservative - may return true for near misses).
    Used for early rejection of dynamic meshes that are clearly not in the ray's path.

    Args:
        ray_origin: (x, y, z) ray starting point
        ray_direction: (x, y, z) ray direction (should be normalized)
        sphere_center: (x, y, z) center of bounding sphere
        sphere_radius: radius of bounding sphere
        max_dist: maximum ray distance to check

    Returns:
        True if ray might hit sphere, False if definitely misses
    """
    # Vector from ray origin to sphere center
    ox, oy, oz = ray_origin
    cx, cy, cz = sphere_center
    dx, dy, dz = ray_direction

    # Vector from origin to center
    ocx = cx - ox
    ocy = cy - oy
    ocz = cz - oz

    # Project center onto ray
    t_closest = ocx * dx + ocy * dy + ocz * dz

    # If sphere is behind ray and ray doesn't start inside sphere
    oc_len_sq = ocx*ocx + ocy*ocy + ocz*ocz
    if t_closest < 0:
        # Check if ray origin is inside sphere
        return oc_len_sq <= sphere_radius * sphere_radius

    # Check if closest approach is beyond max distance
    if t_closest > max_dist + sphere_radius:
        return False

    # Find closest point on ray to sphere center
    closest_x = ox + dx * t_closest
    closest_y = oy + dy * t_closest
    closest_z = oz + dz * t_closest

    # Distance from closest point to sphere center
    dist_sq = (closest_x - cx)**2 + (closest_y - cy)**2 + (closest_z - cz)**2

    # Hit if distance is less than radius (with small margin for numerical safety)
    return dist_sq <= (sphere_radius + 0.1) ** 2


def compute_bounding_sphere(triangles):
    """
    Compute a simple bounding sphere for a list of triangles (world space).
    Uses centroid + max distance method (not optimal but fast).

    Args:
        triangles: list of ((x,y,z), (x,y,z), (x,y,z)) triangles

    Returns:
        (center, radius) where center is (x,y,z) and radius is float
    """
    import math

    if not triangles:
        return ((0, 0, 0), 0.0)

    # Collect all vertices
    sum_x = sum_y = sum_z = 0.0
    count = 0

    all_verts = []
    for v0, v1, v2 in triangles:
        for v in (v0, v1, v2):
            sum_x += v[0]
            sum_y += v[1]
            sum_z += v[2]
            count += 1
            all_verts.append(v)

    # Centroid
    if count == 0:
        return ((0, 0, 0), 0.0)

    cx = sum_x / count
    cy = sum_y / count
    cz = sum_z / count

    # Max distance from centroid
    max_dist_sq = 0.0
    for v in all_verts:
        dx = v[0] - cx
        dy = v[1] - cy
        dz = v[2] - cz
        dist_sq = dx*dx + dy*dy + dz*dz
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq

    radius = math.sqrt(max_dist_sq)

    return ((cx, cy, cz), radius)


def compute_aabb(triangles):
    """
    Compute axis-aligned bounding box for triangles.
    AABB is tighter than sphere for elongated meshes.

    Args:
        triangles: list of ((x,y,z), (x,y,z), (x,y,z)) triangles

    Returns:
        ((min_x, min_y, min_z), (max_x, max_y, max_z)) or None if empty
    """
    if not triangles:
        return None

    # Initialize with first vertex
    v = triangles[0][0]
    min_x = max_x = v[0]
    min_y = max_y = v[1]
    min_z = max_z = v[2]

    # Expand to fit all vertices
    for tri in triangles:
        for v in tri:
            x, y, z = v
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y
            if z < min_z: min_z = z
            if z > max_z: max_z = z

    return ((min_x, min_y, min_z), (max_x, max_y, max_z))


# ============================================================================
# SPATIAL ACCELERATION: Grid for fast ray-triangle culling
# ============================================================================

# Dynamic mesh grid resolution tiers based on triangle count
# Goal: ~30-50 triangles per cell for optimal ray culling
def get_adaptive_grid_resolution(tri_count):
    """
    Compute adaptive grid resolution based on triangle count.

    Fixed 8x8x8 was too coarse for dense meshes (2500+ tris).
    Now scales based on complexity:
    - <200 tris: 4x4x4 = 64 cells (~3 tris/cell)
    - <500 tris: 6x6x6 = 216 cells (~2 tris/cell)
    - <1500 tris: 8x8x8 = 512 cells (~3 tris/cell)
    - <4000 tris: 12x12x12 = 1728 cells (~2 tris/cell)
    - <10000 tris: 16x16x16 = 4096 cells (~2 tris/cell)
    - 10000+ tris: 20x20x20 = 8000 cells
    """
    if tri_count < 200:
        return 4
    elif tri_count < 500:
        return 6
    elif tri_count < 1500:
        return 8
    elif tri_count < 4000:
        return 12
    elif tri_count < 10000:
        return 16
    else:
        return 20


def build_triangle_grid(triangles, aabb):
    """
    Build a uniform 3D grid for spatial acceleration of ray-triangle tests.
    Each cell contains indices of triangles that overlap it.

    Built ONCE during caching - O(N) build, O(cells_traversed) query.

    ADAPTIVE: Grid resolution now scales with triangle count to maintain
    ~30-50 triangles per cell even for dense meshes.

    Args:
        triangles: list of ((x,y,z), (x,y,z), (x,y,z)) triangles
        aabb: ((min_x, min_y, min_z), (max_x, max_y, max_z))

    Returns:
        dict with grid data for fast ray traversal
    """
    if not triangles or not aabb:
        return None

    aabb_min, aabb_max = aabb
    min_x, min_y, min_z = aabb_min
    max_x, max_y, max_z = aabb_max

    # Grid dimensions with small epsilon to avoid edge cases
    eps = 0.001
    size_x = max(max_x - min_x, eps)
    size_y = max(max_y - min_y, eps)
    size_z = max(max_z - min_z, eps)

    # ADAPTIVE: Scale resolution based on triangle count
    grid_resolution = get_adaptive_grid_resolution(len(triangles))

    cell_size_x = size_x / grid_resolution
    cell_size_y = size_y / grid_resolution
    cell_size_z = size_z / grid_resolution

    # Grid cells: dict of (ix, iy, iz) -> list of triangle indices
    cells = {}

    for tri_idx, tri in enumerate(triangles):
        # Find AABB of this triangle
        v0, v1, v2 = tri
        tri_min_x = min(v0[0], v1[0], v2[0])
        tri_max_x = max(v0[0], v1[0], v2[0])
        tri_min_y = min(v0[1], v1[1], v2[1])
        tri_max_y = max(v0[1], v1[1], v2[1])
        tri_min_z = min(v0[2], v1[2], v2[2])
        tri_max_z = max(v0[2], v1[2], v2[2])

        # Find cell range this triangle overlaps
        # CRITICAL: Clamp BOTH min and max to valid range [0, grid_resolution-1]
        # Without upper-bound clamping on _min, flat faces at mesh boundary
        # (e.g., cylinder tops where all verts have z=max_z) get:
        #   iz_min = grid_resolution, iz_max = grid_resolution - 1
        #   range(iz_min, iz_max+1) = EMPTY! Triangle never added to any cell!
        ix_min = max(0, min(grid_resolution - 1, int((tri_min_x - min_x) / cell_size_x)))
        ix_max = max(0, min(grid_resolution - 1, int((tri_max_x - min_x) / cell_size_x)))
        iy_min = max(0, min(grid_resolution - 1, int((tri_min_y - min_y) / cell_size_y)))
        iy_max = max(0, min(grid_resolution - 1, int((tri_max_y - min_y) / cell_size_y)))
        iz_min = max(0, min(grid_resolution - 1, int((tri_min_z - min_z) / cell_size_z)))
        iz_max = max(0, min(grid_resolution - 1, int((tri_max_z - min_z) / cell_size_z)))

        # Add triangle to all overlapping cells
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                for iz in range(iz_min, iz_max + 1):
                    key = (ix, iy, iz)
                    if key not in cells:
                        cells[key] = []
                    cells[key].append(tri_idx)

    return {
        "cells": cells,
        "aabb_min": aabb_min,
        "aabb_max": aabb_max,
        "cell_size": (cell_size_x, cell_size_y, cell_size_z),
        "resolution": grid_resolution
    }


def ray_grid_traverse(ray_origin, ray_direction, max_dist, grid, triangles, tris_tested_counter=None):
    """
    Traverse grid cells along ray path and test only triangles in those cells.
    Uses 3D-DDA algorithm for efficient grid traversal.

    Args:
        ray_origin: (x, y, z) in LOCAL space
        ray_direction: (x, y, z) normalized direction in LOCAL space
        max_dist: maximum ray distance
        grid: grid data from build_triangle_grid
        triangles: list of triangles (for actual intersection test)
        tris_tested_counter: optional [count] to track tests

    Returns:
        (dist, tri_idx) if hit, None if no hit
    """
    if not grid:
        return None

    cells = grid["cells"]
    aabb_min = grid["aabb_min"]
    aabb_max = grid["aabb_max"]
    cell_size_x, cell_size_y, cell_size_z = grid["cell_size"]
    resolution = grid["resolution"]

    ox, oy, oz = ray_origin
    dx, dy, dz = ray_direction

    # Check if ray intersects grid AABB at all
    if not ray_aabb_intersect(ray_origin, ray_direction, aabb_min, aabb_max, max_dist):
        return None

    # Find entry point into grid
    min_x, min_y, min_z = aabb_min
    max_x, max_y, max_z = aabb_max

    # Clamp ray origin to grid bounds for starting cell
    t_start = 0.0

    # If origin is outside AABB, find entry point
    if ox < min_x or ox > max_x or oy < min_y or oy > max_y or oz < min_z or oz > max_z:
        # Use AABB intersection to find entry t
        EPSILON = 1e-8
        t_min = 0.0
        t_max = max_dist

        for i, (o, d, lo, hi) in enumerate([(ox, dx, min_x, max_x),
                                             (oy, dy, min_y, max_y),
                                             (oz, dz, min_z, max_z)]):
            if abs(d) < EPSILON:
                if o < lo or o > hi:
                    return None  # Parallel and outside
            else:
                t1 = (lo - o) / d
                t2 = (hi - o) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return None

        # Small offset to be inside - but direction matters!
        # For rays entering from above going down, we want to start at the TOP of the AABB
        # to ensure we check the topmost cells first
        t_start = t_min + 1e-6  # Tiny offset (was 0.001 which skipped top cells)

    # Starting point in grid
    start_x = ox + dx * t_start
    start_y = oy + dy * t_start
    start_z = oz + dz * t_start

    # Current cell
    ix = int((start_x - min_x) / cell_size_x)
    iy = int((start_y - min_y) / cell_size_y)
    iz = int((start_z - min_z) / cell_size_z)

    # Clamp to valid range
    ix = max(0, min(resolution - 1, ix))
    iy = max(0, min(resolution - 1, iy))
    iz = max(0, min(resolution - 1, iz))

    # Step direction
    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1

    # Distance to next cell boundary
    EPSILON = 1e-8
    if abs(dx) < EPSILON:
        t_delta_x = float('inf')
        t_max_x = float('inf')
    else:
        t_delta_x = abs(cell_size_x / dx)
        if dx > 0:
            next_x = min_x + (ix + 1) * cell_size_x
        else:
            next_x = min_x + ix * cell_size_x
        t_max_x = abs((next_x - start_x) / dx)

    if abs(dy) < EPSILON:
        t_delta_y = float('inf')
        t_max_y = float('inf')
    else:
        t_delta_y = abs(cell_size_y / dy)
        if dy > 0:
            next_y = min_y + (iy + 1) * cell_size_y
        else:
            next_y = min_y + iy * cell_size_y
        t_max_y = abs((next_y - start_y) / dy)

    if abs(dz) < EPSILON:
        t_delta_z = float('inf')
        t_max_z = float('inf')
    else:
        t_delta_z = abs(cell_size_z / dz)
        if dz > 0:
            next_z = min_z + (iz + 1) * cell_size_z
        else:
            next_z = min_z + iz * cell_size_z
        t_max_z = abs((next_z - start_z) / dz)

    # Track tested triangles to avoid duplicates
    tested = set()
    best_dist = max_dist
    best_tri_idx = None

    # Traverse grid (limit iterations to prevent infinite loops)
    max_steps = resolution * 3
    for _ in range(max_steps):
        # Check if we're still in grid
        if ix < 0 or ix >= resolution or iy < 0 or iy >= resolution or iz < 0 or iz >= resolution:
            break

        # Test triangles in current cell
        cell_key = (ix, iy, iz)
        if cell_key in cells:
            for tri_idx in cells[cell_key]:
                if tri_idx in tested:
                    continue
                tested.add(tri_idx)

                if tris_tested_counter is not None:
                    tris_tested_counter[0] += 1

                tri = triangles[tri_idx]
                hit, dist, _ = ray_triangle_intersect(
                    ray_origin, ray_direction, tri[0], tri[1], tri[2]
                )
                if hit and dist < best_dist and dist > 0:
                    best_dist = dist
                    best_tri_idx = tri_idx

        # If we found a hit, check if it's in current cell (early exit)
        if best_tri_idx is not None:
            # Calculate t where we exit this cell
            t_exit = min(t_max_x, t_max_y, t_max_z)
            if best_dist < t_start + t_exit:
                # Hit is in a cell we've already checked
                break

        # Move to next cell
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                ix += step_x
                t_max_x += t_delta_x
            else:
                iz += step_z
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                iy += step_y
                t_max_y += t_delta_y
            else:
                iz += step_z
                t_max_z += t_delta_z

    if best_tri_idx is not None:
        return (best_dist, best_tri_idx)
    return None


def transform_aabb_by_matrix(local_aabb, matrix_4x4):
    """
    Transform a local-space AABB to world space by transforming its 8 corners.
    O(8) operations instead of O(N) for all vertices.

    Args:
        local_aabb: ((min_x, min_y, min_z), (max_x, max_y, max_z))
        matrix_4x4: 4x4 transform matrix as flat 16-element tuple/list

    Returns:
        ((min_x, min_y, min_z), (max_x, max_y, max_z)) world-space AABB
    """
    if local_aabb is None:
        return None

    aabb_min, aabb_max = local_aabb
    min_x, min_y, min_z = aabb_min
    max_x, max_y, max_z = aabb_max

    # Generate 8 corners
    corners = [
        (min_x, min_y, min_z),
        (max_x, min_y, min_z),
        (min_x, max_y, min_z),
        (max_x, max_y, min_z),
        (min_x, min_y, max_z),
        (max_x, min_y, max_z),
        (min_x, max_y, max_z),
        (max_x, max_y, max_z),
    ]

    # Transform corners and find new AABB
    m = matrix_4x4
    first = True
    for cx, cy, cz in corners:
        # Transform point: M @ [x, y, z, 1]
        wx = m[0]*cx + m[1]*cy + m[2]*cz + m[3]
        wy = m[4]*cx + m[5]*cy + m[6]*cz + m[7]
        wz = m[8]*cx + m[9]*cy + m[10]*cz + m[11]

        if first:
            new_min_x = new_max_x = wx
            new_min_y = new_max_y = wy
            new_min_z = new_max_z = wz
            first = False
        else:
            if wx < new_min_x: new_min_x = wx
            if wx > new_max_x: new_max_x = wx
            if wy < new_min_y: new_min_y = wy
            if wy > new_max_y: new_max_y = wy
            if wz < new_min_z: new_min_z = wz
            if wz > new_max_z: new_max_z = wz

    return ((new_min_x, new_min_y, new_min_z), (new_max_x, new_max_y, new_max_z))


def invert_matrix_4x4(m):
    """
    Invert a 4x4 transformation matrix.
    Used to transform rays into local space for collision testing.

    Args:
        m: 16-element flat tuple/list (row-major 4x4 matrix)

    Returns:
        16-element tuple (inverted matrix) or None if singular
    """
    # For typical rigid transforms (rotation + translation + uniform scale),
    # we can use a faster method, but for safety we use the general inverse.

    # Compute 2x2 minors
    s0 = m[0]*m[5] - m[1]*m[4]
    s1 = m[0]*m[6] - m[2]*m[4]
    s2 = m[0]*m[7] - m[3]*m[4]
    s3 = m[1]*m[6] - m[2]*m[5]
    s4 = m[1]*m[7] - m[3]*m[5]
    s5 = m[2]*m[7] - m[3]*m[6]

    c5 = m[10]*m[15] - m[11]*m[14]
    c4 = m[9]*m[15] - m[11]*m[13]
    c3 = m[9]*m[14] - m[10]*m[13]
    c2 = m[8]*m[15] - m[11]*m[12]
    c1 = m[8]*m[14] - m[10]*m[12]
    c0 = m[8]*m[13] - m[9]*m[12]

    # Determinant
    det = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0
    if abs(det) < 1e-10:
        return None  # Singular matrix

    inv_det = 1.0 / det

    # Compute adjugate and multiply by 1/det
    return (
        ( m[5]*c5 - m[6]*c4 + m[7]*c3) * inv_det,
        (-m[1]*c5 + m[2]*c4 - m[3]*c3) * inv_det,
        ( m[13]*s5 - m[14]*s4 + m[15]*s3) * inv_det,
        (-m[9]*s5 + m[10]*s4 - m[11]*s3) * inv_det,

        (-m[4]*c5 + m[6]*c2 - m[7]*c1) * inv_det,
        ( m[0]*c5 - m[2]*c2 + m[3]*c1) * inv_det,
        (-m[12]*s5 + m[14]*s2 - m[15]*s1) * inv_det,
        ( m[8]*s5 - m[10]*s2 + m[11]*s1) * inv_det,

        ( m[4]*c4 - m[5]*c2 + m[7]*c0) * inv_det,
        (-m[0]*c4 + m[1]*c2 - m[3]*c0) * inv_det,
        ( m[12]*s4 - m[13]*s2 + m[15]*s0) * inv_det,
        (-m[8]*s4 + m[9]*s2 - m[11]*s0) * inv_det,

        (-m[4]*c3 + m[5]*c1 - m[6]*c0) * inv_det,
        ( m[0]*c3 - m[1]*c1 + m[2]*c0) * inv_det,
        (-m[12]*s3 + m[13]*s1 - m[14]*s0) * inv_det,
        ( m[8]*s3 - m[9]*s1 + m[10]*s0) * inv_det,
    )


def transform_ray_to_local(ray_origin, ray_direction, inv_matrix):
    """
    Transform a world-space ray into local (object) space using inverse matrix.

    Args:
        ray_origin: (x, y, z) world-space origin
        ray_direction: (x, y, z) world-space direction (should be normalized)
        inv_matrix: 16-element inverted 4x4 matrix

    Returns:
        (local_origin, local_direction, dir_len) - direction is re-normalized
        dir_len is the scale factor for converting local t-values to world t-values:
            t_world = t_local / dir_len
            t_local = t_world * dir_len
    """
    m = inv_matrix
    ox, oy, oz = ray_origin
    dx, dy, dz = ray_direction

    # Transform origin (point): M^-1 @ [ox, oy, oz, 1]
    lox = m[0]*ox + m[1]*oy + m[2]*oz + m[3]
    loy = m[4]*ox + m[5]*oy + m[6]*oz + m[7]
    loz = m[8]*ox + m[9]*oy + m[10]*oz + m[11]

    # Transform direction (vector, no translation): M^-1 @ [dx, dy, dz, 0]
    ldx = m[0]*dx + m[1]*dy + m[2]*dz
    ldy = m[4]*dx + m[5]*dy + m[6]*dz
    ldz = m[8]*dx + m[9]*dy + m[10]*dz

    # Compute direction length BEFORE normalizing - this is the scale factor
    # for converting between local and world t-values
    import math
    dir_len = math.sqrt(ldx*ldx + ldy*ldy + ldz*ldz)
    if dir_len > 1e-10:
        ldx /= dir_len
        ldy /= dir_len
        ldz /= dir_len
    else:
        dir_len = 1.0  # Avoid division by zero

    return ((lox, loy, loz), (ldx, ldy, ldz), dir_len)


def ray_aabb_intersect(ray_origin, ray_dir, aabb_min, aabb_max, max_dist):
    """
    Fast ray-AABB intersection test (slab method).

    Args:
        ray_origin: (x, y, z) ray starting point
        ray_dir: (x, y, z) ray direction (should be normalized)
        aabb_min: (x, y, z) AABB minimum corner
        aabb_max: (x, y, z) AABB maximum corner
        max_dist: maximum ray distance

    Returns:
        True if ray intersects AABB within max_dist
    """
    EPSILON = 1e-8
    ox, oy, oz = ray_origin
    dx, dy, dz = ray_dir

    t_min = 0.0
    t_max = max_dist

    # X slab
    if abs(dx) > EPSILON:
        inv_d = 1.0 / dx
        t1 = (aabb_min[0] - ox) * inv_d
        t2 = (aabb_max[0] - ox) * inv_d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    else:
        if ox < aabb_min[0] or ox > aabb_max[0]:
            return False

    # Y slab
    if abs(dy) > EPSILON:
        inv_d = 1.0 / dy
        t1 = (aabb_min[1] - oy) * inv_d
        t2 = (aabb_max[1] - oy) * inv_d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    else:
        if oy < aabb_min[1] or oy > aabb_max[1]:
            return False

    # Z slab
    if abs(dz) > EPSILON:
        inv_d = 1.0 / dz
        t1 = (aabb_min[2] - oz) * inv_d
        t2 = (aabb_max[2] - oz) * inv_d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    else:
        if oz < aabb_min[2] or oz > aabb_max[2]:
            return False

    return True


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
    import math

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
            if aabb:
                aabb_min, aabb_max = aabb
                if not ray_aabb_intersect(ray_origin, ray_direction,
                                          aabb_min, aabb_max, best_dist):
                    continue  # Skip ALL triangles in this mesh!
            # Fallback to sphere if no AABB
            elif bounding_sphere:
                sphere_center, sphere_radius = bounding_sphere
                if not ray_sphere_intersect(ray_origin, ray_direction,
                                           sphere_center, sphere_radius, best_dist):
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

            if grid:
                # GRID-ACCELERATED: O(cells) instead of O(N)
                tris_counter = [0]
                grid_result = ray_grid_traverse(
                    local_origin, local_direction, best_dist_local, grid, triangles, tris_counter
                )
                total_tris += tris_counter[0]
                if grid_result:
                    hit_dist, hit_tri_idx = grid_result
            else:
                # Fallback: brute-force (for meshes without grid)
                for tri_idx, tri in enumerate(triangles):
                    total_tris += 1
                    hit, dist, _ = ray_triangle_intersect(
                        local_origin, local_direction, tri[0], tri[1], tri[2]
                    )
                    if hit and dist < best_dist_local:
                        hit_dist = dist
                        hit_tri_idx = tri_idx

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
        if aabb:
            aabb_min, aabb_max = aabb
            if not ray_aabb_intersect(ray_origin, ray_direction, aabb_min, aabb_max, best_dist):
                if debug_log is not None:
                    debug_log.append(f"AABB_REJECT ray=({ray_origin[0]:.1f},{ray_origin[1]:.1f},{ray_origin[2]:.1f}) dir=({ray_direction[0]:.2f},{ray_direction[1]:.2f},{ray_direction[2]:.2f}) aabb=[({aabb_min[0]:.1f},{aabb_min[1]:.1f},{aabb_min[2]:.1f})->({aabb_max[0]:.1f},{aabb_max[1]:.1f},{aabb_max[2]:.1f})]")
                continue  # Skip ALL triangles in this mesh!
        # Fallback to sphere if no AABB
        elif bounding_sphere:
            center, radius = bounding_sphere
            if not ray_sphere_intersect(ray_origin, ray_direction, center, radius, best_dist):
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
        # GRID-ACCELERATED: Use 3D-DDA if grid available (O(cells) vs O(N))
        # ═══════════════════════════════════════════════════════════════════
        if grid:
            grid_result = ray_grid_traverse(
                local_origin, local_direction, best_dist_local, grid, triangles, tris_tested_counter
            )
            if grid_result:
                hit_dist, hit_tri_idx = grid_result
        else:
            # Fallback: brute-force test all triangles (for meshes without grid)
            for tri_idx, tri in enumerate(triangles):
                if tris_tested_counter is not None:
                    tris_tested_counter[0] += 1

                hit, dist, _ = ray_triangle_intersect(
                    local_origin, local_direction, tri[0], tri[1], tri[2]
                )
                if hit and dist < best_dist_local:
                    hit_dist = dist
                    hit_tri_idx = tri_idx

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
                    import math
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


# ============================================================================
# DYNAMIC MESH TRANSFORM HELPERS
# ============================================================================

def transform_point(point, matrix_4x4):
    """
    Transform a point from local to world space using 4x4 matrix.

    Args:
        point: (x, y, z) point in local space
        matrix_4x4: 16-element tuple representing row-major 4x4 matrix:
                   (m00, m01, m02, m03,
                    m10, m11, m12, m13,
                    m20, m21, m22, m23,
                    m30, m31, m32, m33)

    Returns:
        (x, y, z) point in world space
    """
    px, py, pz = point

    # Extract matrix elements (row-major)
    m00, m01, m02, m03 = matrix_4x4[0:4]
    m10, m11, m12, m13 = matrix_4x4[4:8]
    m20, m21, m22, m23 = matrix_4x4[8:12]
    # m30, m31, m32, m33 = matrix_4x4[12:16]  # Not needed for point transform

    # Apply transformation (assumes w=1 for points)
    wx = m00 * px + m01 * py + m02 * pz + m03
    wy = m10 * px + m11 * py + m12 * pz + m13
    wz = m20 * px + m21 * py + m22 * pz + m23

    return (wx, wy, wz)


def transform_triangle(tri_local, matrix_4x4):
    """
    Transform a triangle from local to world space.

    Args:
        tri_local: (v0, v1, v2) triangle in local space
        matrix_4x4: 16-element tuple (row-major 4x4 matrix)

    Returns:
        (v0_world, v1_world, v2_world) triangle in world space
    """
    v0_local, v1_local, v2_local = tri_local

    v0_world = transform_point(v0_local, matrix_4x4)
    v1_world = transform_point(v1_local, matrix_4x4)
    v2_world = transform_point(v2_local, matrix_4x4)

    return (v0_world, v1_world, v2_world)


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

        elif job.job_type == "PING":
            # Worker verification ping - used during startup to confirm worker responsiveness
            result_data = {
                "pong": True,
                "worker_id": job.data.get("worker_check", -1),
                "timestamp": time.time(),
                "worker_msg": "Worker alive and responsive"
            }

        elif job.job_type == "CACHE_GRID":
            # Cache spatial grid for subsequent raycast jobs
            # This is sent ONCE at game start to avoid 3MB serialization per raycast
            grid = job.data.get("grid", None)
            if grid is not None:
                _cached_grid = grid
                tri_count = len(grid.get("triangles", []))
                cell_count = len(grid.get("cells", {}))
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

        elif job.job_type == "CACHE_DYNAMIC_MESH":
            # Cache dynamic mesh triangles in LOCAL space
            # This is sent ONCE per dynamic mesh (or when mesh changes)
            # Per-frame, only transform matrices are sent (64 bytes vs 3MB!)
            #
            # OPTIMIZATION: Pre-compute local AABB once here.
            # Per-frame we only transform 8 AABB corners instead of N*3 vertices.
            global _cached_dynamic_meshes

            obj_id = job.data.get("obj_id")
            triangles = job.data.get("triangles", [])
            radius = job.data.get("radius", 1.0)

            if obj_id is not None and triangles:
                # DIAGNOSTIC: Check if already cached (potential duplicate caching)
                was_cached = obj_id in _cached_dynamic_meshes

                # Compute local-space AABB ONCE (O(N) here, O(8) per frame)
                local_aabb = compute_aabb(triangles)

                # Build spatial grid for O(cells) ray testing instead of O(N)
                # This is the key optimization for high-poly meshes!
                tri_grid = build_triangle_grid(triangles, local_aabb)
                grid_cells = len(tri_grid["cells"]) if tri_grid else 0

                _cached_dynamic_meshes[obj_id] = {
                    "triangles": triangles,   # List of (v0, v1, v2) tuples in local space
                    "local_aabb": local_aabb, # Pre-computed local AABB for fast world transform
                    "radius": radius,         # Bounding sphere radius for quick rejection
                    "grid": tri_grid          # Spatial grid for fast ray-triangle culling
                }
                tri_count = len(triangles)
                grid_res = tri_grid["resolution"] if tri_grid else 0
                if DEBUG_ENGINE:
                    print(f"[Worker] Dynamic mesh cached: obj_id={obj_id} tris={tri_count:,} radius={radius:.2f}m grid={grid_res}³={grid_cells}cells")

                # Log to diagnostics (one-time cache event)
                status = "RE-CACHED" if was_cached else "CACHED"
                total_cached = len(_cached_dynamic_meshes)
                cache_log = ("DYN-CACHE", f"{status} obj_id={obj_id} tris={tri_count} grid={grid_res}³={grid_cells}cells radius={radius:.2f}m total={total_cached}")
                logs = [cache_log]

                result_data = {
                    "success": True,
                    "obj_id": obj_id,
                    "triangle_count": tri_count,
                    "radius": radius,
                    "message": "Dynamic mesh cached successfully",
                    "logs": logs  # Return all logs to main thread
                }
            else:
                result_data = {
                    "success": False,
                    "error": "Missing obj_id or triangles data"
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

        elif job.job_type == "KCC_PHYSICS_STEP":
            # =================================================================
            # FULL KCC PHYSICS STEP - Worker computes entire physics frame
            # =================================================================
            # This is the new architecture: worker does ALL physics computation,
            # main thread only applies results to Blender objects.
            # NO prediction needed - worker computes actual result.
            # =================================================================
            import math

            calc_start = time.perf_counter()

            # ─────────────────────────────────────────────────────────────────
            # UNPACK INPUT
            # ─────────────────────────────────────────────────────────────────
            pos = job.data.get("pos", (0.0, 0.0, 0.0))
            vel = job.data.get("vel", (0.0, 0.0, 0.0))
            on_ground = job.data.get("on_ground", False)
            on_walkable = job.data.get("on_walkable", True)
            ground_normal = job.data.get("ground_normal", (0.0, 0.0, 1.0))

            wish_dir = job.data.get("wish_dir", (0.0, 0.0))
            is_running = job.data.get("is_running", False)
            jump_requested = job.data.get("jump_requested", False)

            coyote_remaining = job.data.get("coyote_remaining", 0.0)
            jump_buffer_remaining = job.data.get("jump_buffer_remaining", 0.0)

            dt = job.data.get("dt", 1.0/30.0)

            # Config
            cfg = job.data.get("config", {})
            radius = cfg.get("radius", 0.22)
            height = cfg.get("height", 1.8)
            gravity = cfg.get("gravity", -9.81)
            max_walk = cfg.get("max_walk", 2.5)
            max_run = cfg.get("max_run", 5.5)
            accel_ground = cfg.get("accel_ground", 20.0)
            accel_air = cfg.get("accel_air", 5.0)
            step_height = cfg.get("step_height", 0.4)
            snap_down = cfg.get("snap_down", 0.5)
            slope_limit_deg = cfg.get("slope_limit_deg", 50.0)
            jump_speed = cfg.get("jump_speed", 7.0)
            coyote_time = cfg.get("coyote_time", 0.08)

            floor_cos = math.cos(math.radians(slope_limit_deg))

            # Debug flags
            debug_flags = job.data.get("debug_flags", {})
            # UNIFIED PHYSICS: All flags control unified physics (static + dynamic identical)
            debug_physics = debug_flags.get("physics", False)      # Physics summary
            debug_ground = debug_flags.get("ground", False)        # Ground detection
            debug_horizontal = debug_flags.get("horizontal", False) # Horizontal collision
            debug_body = debug_flags.get("body", False)            # Body integrity
            debug_ceiling = debug_flags.get("ceiling", False)      # Ceiling check
            debug_step = debug_flags.get("step", False)            # Step-up
            debug_slide = debug_flags.get("slide", False)          # Wall slide
            debug_slopes = debug_flags.get("slopes", False)        # Slopes

            # Worker log buffer (collected during computation, returned to main thread)
            worker_logs = []

            # Convert to mutable lists for computation
            px, py, pz = pos
            vx, vy, vz = vel
            gn_x, gn_y, gn_z = ground_normal
            wish_x, wish_y = wish_dir

            # Debug counters
            total_rays = 0
            total_tris = 0
            total_cells = 0
            h_blocked = False
            did_step_up = False
            did_slide = False
            hit_ceiling = False
            jump_consumed = False

            # ─────────────────────────────────────────────────────────────────
            # PERSISTENT TRANSFORM CACHE - UNIFIED ARCHITECTURE
            # ─────────────────────────────────────────────────────────────────
            # Worker maintains last-known transform for ALL dynamic meshes.
            # Main thread only sends transforms when meshes MOVE (thin/efficient).
            # Worker uses cached transforms for stationary meshes (zero main-thread cost).
            #
            # This eliminates the "activation" concept entirely:
            # - No AABB checks on main thread
            # - No chicken-and-egg timing issues
            # - Dynamic meshes behave like static: always available for testing
            # ─────────────────────────────────────────────────────────────────

            global _cached_dynamic_transforms
            dynamic_transforms_update = job.data.get("dynamic_transforms", {})
            dynamic_velocities = job.data.get("dynamic_velocities", {})  # For diagnostics
            dynamic_transform_time_us = 0.0
            transforms_updated = 0
            transforms_from_cache = 0

            # ═══════════════════════════════════════════════════════════════════
            # STEP 1: Update transform cache with any new transforms from main thread
            # Mesh triangles are cached via targeted broadcast_job (guaranteed delivery)
            # ═══════════════════════════════════════════════════════════════════
            for obj_id, matrix_4x4 in dynamic_transforms_update.items():
                cached = _cached_dynamic_meshes.get(obj_id)
                if cached is None:
                    continue  # Mesh triangles not cached yet, skip

                local_aabb = cached.get("local_aabb")
                if local_aabb:
                    world_aabb = transform_aabb_by_matrix(local_aabb, matrix_4x4)
                else:
                    world_aabb = None

                # Cache the transform + computed world AABB
                _cached_dynamic_transforms[obj_id] = (matrix_4x4, world_aabb)
                transforms_updated += 1

            # ═══════════════════════════════════════════════════════════════════
            # STEP 2: Build unified_dynamic_meshes from ALL cached transforms
            # ═══════════════════════════════════════════════════════════════════
            # This is the key change: we test ALL meshes that have cached transforms,
            # not just ones that received updates this frame.
            unified_dynamic_meshes = []
            transform_start = time.perf_counter()

            for obj_id, (matrix_4x4, world_aabb) in _cached_dynamic_transforms.items():
                cached = _cached_dynamic_meshes.get(obj_id)
                if cached is None:
                    continue  # Shouldn't happen, but be safe

                local_triangles = cached["triangles"]

                # Check if this was a fresh update or from cache
                if obj_id not in dynamic_transforms_update:
                    transforms_from_cache += 1

                # Compute inverse matrix for local-space ray testing
                inv_matrix = invert_matrix_4x4(matrix_4x4)

                # Compute bounding sphere center from AABB (fast approximation)
                if world_aabb:
                    aabb_min, aabb_max = world_aabb
                    center = (
                        (aabb_min[0] + aabb_max[0]) * 0.5,
                        (aabb_min[1] + aabb_max[1]) * 0.5,
                        (aabb_min[2] + aabb_max[2]) * 0.5
                    )
                    # Radius is half-diagonal of AABB (conservative bound)
                    import math
                    half_diag = math.sqrt(
                        (aabb_max[0] - aabb_min[0])**2 +
                        (aabb_max[1] - aabb_min[1])**2 +
                        (aabb_max[2] - aabb_min[2])**2
                    ) * 0.5
                    bounding_sphere = (center, half_diag)
                else:
                    bounding_sphere = ((0, 0, 0), cached.get("radius", 1.0))

                # Add to unified format - triangles stay in LOCAL space!
                unified_dynamic_meshes.append({
                    "obj_id": obj_id,
                    "triangles": local_triangles,
                    "matrix": matrix_4x4,
                    "inv_matrix": inv_matrix,
                    "bounding_sphere": bounding_sphere,
                    "aabb": world_aabb,
                    "grid": cached.get("grid")
                })

            transform_end = time.perf_counter()
            dynamic_transform_time_us = (transform_end - transform_start) * 1_000_000

            # ═══════════════════════════════════════════════════════════════════
            # DIAGNOSTIC LOGGING - Unified Dynamic Mesh System
            # ═══════════════════════════════════════════════════════════════════
            total_cached_meshes = len(_cached_dynamic_meshes)
            total_cached_transforms = len(_cached_dynamic_transforms)
            mesh_count = len(unified_dynamic_meshes)
            total_dyn_tris = sum(len(m["triangles"]) for m in unified_dynamic_meshes)

            if debug_flags.get("engine", False):
                # Cache efficiency: how many transforms came from persistent cache
                cache_hit_pct = (transforms_from_cache / mesh_count * 100) if mesh_count > 0 else 0
                worker_logs.append(("ENGINE",
                    f"DYNAMIC: meshes={mesh_count} tris={total_dyn_tris} | "
                    f"updated={transforms_updated} cached={transforms_from_cache} ({cache_hit_pct:.0f}% cache hit) | "
                    f"transform_time={dynamic_transform_time_us:.0f}us"))

            if debug_flags.get("dynamic_cache", False):
                # Detailed cache state
                worker_logs.append(("DYN-CACHE",
                    f"STATE: mesh_cache={total_cached_meshes} transform_cache={total_cached_transforms} active={mesh_count} | "
                    f"updated={transforms_updated} from_cache={transforms_from_cache} | "
                    f"tris={total_dyn_tris} time={dynamic_transform_time_us:.1f}us"))

            # ─────────────────────────────────────────────────────────────────
            # 1. TIMERS
            # ─────────────────────────────────────────────────────────────────
            coyote_remaining = max(0.0, coyote_remaining - dt)

            # ─────────────────────────────────────────────────────────────────
            # 2. INPUT → VELOCITY (Acceleration)
            # ─────────────────────────────────────────────────────────────────
            target_speed = max_run if is_running else max_walk
            accel = accel_ground if on_ground else accel_air

            # Lerp toward desired velocity
            desired_x = wish_x * target_speed
            desired_y = wish_y * target_speed

            # NOTE: Uphill blocking moved to Step 8 (after ground detection) to use current frame's ground normal

            t = min(1.0, accel * dt)
            vx = vx + (desired_x - vx) * t
            vy = vy + (desired_y - vy) * t

            # ─────────────────────────────────────────────────────────────────
            # 3. GRAVITY
            # ─────────────────────────────────────────────────────────────────
            # Note: Steep slope sliding moved to step 8 (after ground detection)
            if not on_ground:
                vz += gravity * dt
            else:
                if on_walkable:
                    vz = max(vz, 0.0)
                # Steep slope sliding handled in step 8 after ground normal is updated

            # ─────────────────────────────────────────────────────────────────
            # 4. JUMP
            # ─────────────────────────────────────────────────────────────────
            can_jump = (on_ground and on_walkable) or (coyote_remaining > 0.0)
            if jump_requested and can_jump:
                vz = jump_speed
                on_ground = False
                jump_consumed = True
                coyote_remaining = 0.0

            # ─────────────────────────────────────────────────────────────────
            # 4.5. UPHILL BLOCKING (BEFORE movement - using last frame's ground normal)
            # ─────────────────────────────────────────────────────────────────
            # CRITICAL: This must happen BEFORE horizontal collision/movement!
            # Block uphill movement on steep slopes (grounded OR airborne)
            # This prevents jump-spam climbing by blocking based on last ground contact

            # Check if we have a steep slope from last ground contact
            slope_angle = 0.0
            if gn_z < 1.0:  # Have a ground normal from previous contact
                slope_angle = math.degrees(math.acos(min(1.0, max(-1.0, gn_z))))

            is_steep = slope_angle > slope_limit_deg

            # Block uphill movement if on steep slope (grounded OR recently airborne from steep slope)
            if is_steep and (on_ground or (not on_ground and vz > -2.0)):  # vz > -2 = recently jumped/airborne
                gn_xy_len = math.sqrt(gn_x*gn_x + gn_y*gn_y)
                if gn_xy_len > 0.001:
                    # Calculate uphill direction (negate normal's XY projection)
                    uphill_x = -gn_x / gn_xy_len
                    uphill_y = -gn_y / gn_xy_len

                    # Check uphill velocity
                    uphill_vel = vx * uphill_x + vy * uphill_y

                    if uphill_vel > 0.0:
                        # Remove ALL uphill velocity (no pushback to avoid getting stuck)
                        vx = vx - uphill_x * uphill_vel
                        vy = vy - uphill_y * uphill_vel

                        # Optional: Very gentle nudge downhill ONLY when airborne
                        # This helps prevent jump spam without causing stuck-in-mesh issues
                        if not on_ground and slope_angle > 65.0:
                            # Gentle downhill nudge only when airborne on very steep slopes
                            downhill_x = -uphill_x
                            downhill_y = -uphill_y
                            vx += downhill_x * 2.0
                            vy += downhill_y * 2.0

                            if debug_slopes:
                                worker_logs.append(("SLOPES", f"PRE-BLOCK AIRBORNE angle={slope_angle:.0f}°"))

            # ─────────────────────────────────────────────────────────────────
            # 4.9. EXTRACT GRID DATA ONCE (performance optimization)
            # ─────────────────────────────────────────────────────────────────
            # Cache grid data to avoid repeated dictionary lookups (4x per frame)
            # Also create grid_data dict for unified_raycast
            grid_bounds_min = None
            grid_bounds_max = None
            grid_cell_size = None
            grid_dims = None
            grid_cells = None
            grid_triangles = None
            grid_min_x = grid_min_y = grid_min_z = 0.0
            grid_max_x = grid_max_y = grid_max_z = 0.0
            grid_nx = grid_ny = grid_nz = 0
            grid_data = None  # For unified_raycast

            if _cached_grid is not None:
                grid_bounds_min = _cached_grid["bounds_min"]
                grid_bounds_max = _cached_grid["bounds_max"]
                grid_cell_size = _cached_grid["cell_size"]
                grid_dims = _cached_grid["grid_dims"]
                grid_cells = _cached_grid["cells"]
                grid_triangles = _cached_grid["triangles"]

                grid_min_x, grid_min_y, grid_min_z = grid_bounds_min
                grid_max_x, grid_max_y, grid_max_z = grid_bounds_max
                grid_nx, grid_ny, grid_nz = grid_dims

                # Create grid_data dict for unified_raycast
                grid_data = {
                    "cells": grid_cells,
                    "triangles": grid_triangles,
                    "bounds_min": grid_bounds_min,
                    "bounds_max": grid_bounds_max,
                    "cell_size": grid_cell_size,
                    "grid_dims": grid_dims
                }

            # ─────────────────────────────────────────────────────────────────
            # 4.99. DYNAMIC MESH PROXIMITY + PROACTIVE DETECTION
            # When mesh is near AND (player stationary OR mesh approaching),
            # cast rays TOWARD mesh to detect contact before normal collision
            # ─────────────────────────────────────────────────────────────────
            proximity_meshes = []
            proactive_best_d = None
            proactive_best_n = None
            proactive_obj_id = None

            if unified_dynamic_meshes:
                # Player capsule AABB
                p_min = (px - radius, py - radius, pz)
                p_max = (px + radius, py + radius, pz + height)
                player_speed = math.sqrt(vx*vx + vy*vy)

                for dyn_mesh in unified_dynamic_meshes:
                    obj_id = dyn_mesh.get("obj_id")
                    mesh_aabb = dyn_mesh.get("aabb")
                    if mesh_aabb is None:
                        continue

                    m_min, m_max = mesh_aabb

                    # Get mesh velocity FIRST to scale detection range
                    mesh_vel = dynamic_velocities.get(obj_id, (0.0, 0.0, 0.0))
                    mvx, mvy, mvz = mesh_vel
                    mesh_speed = math.sqrt(mvx*mvx + mvy*mvy)

                    # AABB overlap check - expand by velocity to catch fast meshes EARLY
                    # At 20m/s, mesh travels 0.66m/frame - need to detect 2-3 frames ahead
                    speed_expand = mesh_speed * dt * 3.0  # 3 frames of travel
                    expand = radius + 0.3 + speed_expand
                    overlap = (
                        p_min[0] - expand <= m_max[0] and p_max[0] + expand >= m_min[0] and
                        p_min[1] - expand <= m_max[1] and p_max[1] + expand >= m_min[1] and
                        p_min[2] - expand <= m_max[2] and p_max[2] + expand >= m_min[2]
                    )

                    if not overlap:
                        continue

                    # Compute mesh center
                    mcx = (m_min[0] + m_max[0]) * 0.5
                    mcy = (m_min[1] + m_max[1]) * 0.5

                    # Direction from player to mesh center (horizontal only)
                    dx = mcx - px
                    dy = mcy - py
                    dist_xy = math.sqrt(dx*dx + dy*dy)

                    proximity_meshes.append(obj_id)

                    if debug_horizontal:
                        worker_logs.append(("HORIZONTAL",
                            f"PROXIMITY obj={obj_id} dist_xy={dist_xy:.2f}m "
                            f"mesh_vel=({mvx:.2f},{mvy:.2f},{mvz:.2f}) speed={mesh_speed:.2f}m/s "
                            f"player_vel=({vx:.2f},{vy:.2f})"))

                    # Check if mesh is moving fast enough to need proactive detection
                    if mesh_speed < 0.5 and player_speed >= 1.0:
                        continue  # Mesh slow and player moving - normal collision handles it

                    # Determine ray direction based on mesh velocity (cast OPPOSITE to mesh movement)
                    # This finds the surface that's approaching the player
                    ray_directions = []

                    if mesh_speed > 0.5:
                        # Primary: Cast opposite to mesh velocity (toward approaching face)
                        ray_dir_x = -mvx / mesh_speed
                        ray_dir_y = -mvy / mesh_speed
                        ray_directions.append((ray_dir_x, ray_dir_y, mesh_speed))

                    # Secondary: Also cast toward mesh center for stationary meshes
                    if dist_xy > 0.1:
                        ray_dir_x = dx / dist_xy
                        ray_dir_y = dy / dist_xy
                        ray_directions.append((ray_dir_x, ray_dir_y, dist_xy))

                    # Tertiary: Cardinal directions for robustness
                    if mesh_speed > 2.0 or player_speed < 0.5:
                        ray_directions.extend([
                            (1.0, 0.0, 2.0),
                            (-1.0, 0.0, 2.0),
                            (0.0, 1.0, 2.0),
                            (0.0, -1.0, 2.0),
                        ])

                    ray_heights = [0.1, height * 0.5, height - radius]

                    # Scale ray distance by mesh speed - need to see further ahead for fast meshes
                    speed_ray_extend = mesh_speed * dt * 3.0  # Look 3 frames ahead
                    contact_range = radius + 0.1 + speed_ray_extend  # Detection triggers this far out

                    for ray_dir_x, ray_dir_y, ray_max_hint in ray_directions:
                        ray_max = max(radius + 0.5 + speed_ray_extend, ray_max_hint + speed_ray_extend)

                        for ray_z in ray_heights:
                            tris_counter = [0]
                            total_rays += 1
                            ray_result = cast_ray(
                                (px, py, pz + ray_z), (ray_dir_x, ray_dir_y, 0), ray_max,
                                _cached_grid, unified_dynamic_meshes, tris_counter
                            )
                            total_tris += tris_counter[0]

                            if ray_result:
                                hit_dist, hit_normal, hit_source, hit_obj_id = ray_result
                                # Trigger if within contact range (scaled by speed)
                                if hit_dist < contact_range:
                                    if proactive_best_d is None or hit_dist < proactive_best_d:
                                        proactive_best_d = hit_dist
                                        proactive_best_n = hit_normal
                                        proactive_obj_id = hit_obj_id

                                        if debug_horizontal:
                                            worker_logs.append(("HORIZONTAL",
                                                f"PROACTIVE_HIT obj={hit_obj_id} dist={hit_dist:.3f}m "
                                                f"contact_range={contact_range:.2f}m "
                                                f"normal=({hit_normal[0]:.2f},{hit_normal[1]:.2f},{hit_normal[2]:.2f})"))

            # ─────────────────────────────────────────────────────────────────
            # 5. HORIZONTAL COLLISION - UNIFIED for all geometry
            # Single cast_ray() tests BOTH static and dynamic for each ray
            # ─────────────────────────────────────────────────────────────────
            move_x = vx * dt
            move_y = vy * dt
            move_len = math.sqrt(move_x*move_x + move_y*move_y)

            best_d = None
            best_n = None
            per_ray_hits = []

            if move_len > 1e-9:
                # Normalize movement direction
                fwd_x = move_x / move_len
                fwd_y = move_y / move_len

                # Cast rays at 3 heights (feet, mid, head)
                ray_heights = [0.1, min(height * 0.5, height - radius), height - radius]
                ray_len = move_len + radius

                # Main horizontal rays (3 heights)
                for ray_z in ray_heights:
                    total_rays += 1
                    tris_counter = [0]
                    ray_result = cast_ray(
                        (px, py, pz + ray_z), (fwd_x, fwd_y, 0), ray_len,
                        _cached_grid, unified_dynamic_meshes, tris_counter
                    )
                    total_tris += tris_counter[0]

                    if ray_result:
                        ray_dist, ray_normal, ray_source, ray_obj_id = ray_result
                        per_ray_hits.append(ray_dist)
                        if best_d is None or ray_dist < best_d:
                            best_d = ray_dist
                            best_n = ray_normal
                    else:
                        per_ray_hits.append(None)

                # WIDTH RAYS: Check left/right edges at mid-height for narrow gaps
                perp_x = -fwd_y  # Perpendicular left
                perp_y = fwd_x
                mid_height = height * 0.5
                width_ray_configs = [
                    (perp_x * radius, perp_y * radius),   # Left edge
                    (-perp_x * radius, -perp_y * radius), # Right edge
                ]

                width_hits = []
                for width_offset_x, width_offset_y in width_ray_configs:
                    total_rays += 1
                    tris_counter = [0]
                    width_result = cast_ray(
                        (px + width_offset_x, py + width_offset_y, pz + mid_height),
                        (fwd_x, fwd_y, 0), ray_len,
                        _cached_grid, unified_dynamic_meshes, tris_counter
                    )
                    total_tris += tris_counter[0]

                    if width_result:
                        width_hits.append((width_result[0], width_result[1]))
                    else:
                        width_hits.append((None, None))

                # Only block if BOTH width rays hit (narrow gap detection)
                if len(width_hits) == 2:
                    left_dist, left_n = width_hits[0]
                    right_dist, right_n = width_hits[1]
                    if left_dist is not None and right_dist is not None:
                        closest_dist = min(left_dist, right_dist)
                        if best_d is None or closest_dist < best_d:
                            best_d = closest_dist
                            best_n = left_n if left_dist < right_dist else right_n

                # SLOPE RAYS: Angled down to catch steep slopes (only when grounded)
                if on_ground:
                    slope_angle = 0.5  # ~30 degrees down
                    slope_ray_z = radius * 2  # Knee height
                    slope_fwd_len = math.sqrt(1.0 / (1.0 + slope_angle * slope_angle))
                    slope_down_len = slope_angle * slope_fwd_len

                    ray_dx = fwd_x * slope_fwd_len
                    ray_dy = fwd_y * slope_fwd_len
                    ray_dz = -slope_down_len

                    total_rays += 1
                    tris_counter = [0]
                    slope_result = cast_ray(
                        (px, py, pz + slope_ray_z), (ray_dx, ray_dy, ray_dz), ray_len,
                        _cached_grid, unified_dynamic_meshes, tris_counter
                    )
                    total_tris += tris_counter[0]

                    if slope_result:
                        slope_dist, slope_normal, slope_source, slope_obj_id = slope_result
                        # Only count steep slopes (not floors)
                        if slope_normal[2] < floor_cos and slope_normal[2] > 0.1:
                            horiz_dist = slope_dist * slope_fwd_len
                            if best_d is None or horiz_dist < best_d:
                                best_d = horiz_dist
                                best_n = slope_normal

                # Apply horizontal collision result (static OR dynamic - whichever is closer)
                if best_d is not None:
                    h_blocked = True
                    allowed = max(0.0, best_d - radius)

                    # Move position
                    if allowed > 1e-9:
                        px += fwd_x * allowed
                        py += fwd_y * allowed

                    # Remove velocity component into wall
                    if best_n is not None:
                        bn_x, bn_y, bn_z = best_n
                        vn = vx * bn_x + vy * bn_y
                        if vn > 0.0:
                            vx -= bn_x * vn
                            vy -= bn_y * vn

                        # ─────────────────────────────────────────────────────────
                        # AIRBORNE STEEP SLOPE BLOCKING (backup to Step 4.5)
                        # ─────────────────────────────────────────────────────────
                        # If airborne and hitting a steep surface, block uphill velocity
                        # This is a backup - Step 4.5 pre-blocking is the primary defense
                        if not on_ground:
                            # Calculate slope angle from normal
                            slope_angle = math.degrees(math.acos(min(1.0, max(-1.0, bn_z))))

                            # If hitting a steep slope (> slope_limit_deg)
                            if slope_angle > slope_limit_deg:
                                # Calculate uphill direction (XY projection of normal, negated)
                                bn_xy_len = math.sqrt(bn_x*bn_x + bn_y*bn_y)
                                if bn_xy_len > 0.001:
                                    uphill_x = -bn_x / bn_xy_len
                                    uphill_y = -bn_y / bn_xy_len

                                    # Check if moving uphill
                                    uphill_vel = vx * uphill_x + vy * uphill_y

                                    if uphill_vel > 0.0:
                                        # Just remove uphill velocity, no pushback
                                        # (Pushback causes stuck-in-mesh issues)
                                        vx = vx - uphill_x * uphill_vel
                                        vy = vy - uphill_y * uphill_vel

                    # Debug logging
                    if debug_horizontal:
                        if best_n is not None:
                            worker_logs.append(("HORIZONTAL", f"blocked dist={best_d:.3f}m allowed={allowed:.3f}m normal=({best_n[0]:.2f},{best_n[1]:.2f},{best_n[2]:.2f}) | {total_rays}rays {total_tris}tris"))
                        else:
                            worker_logs.append(("HORIZONTAL", f"blocked dist={best_d:.3f}m allowed={allowed:.3f}m normal=None | {total_rays}rays {total_tris}tris"))

                    # ─────────────────────────────────────────────────────────
                    # 5a. STEP-UP (if only feet ray hit) - UNIFIED
                    # ─────────────────────────────────────────────────────────
                    feet_only = (len(per_ray_hits) >= 3 and
                                per_ray_hits[0] is not None and
                                per_ray_hits[1] is None and
                                per_ray_hits[2] is None)

                    # Only allow step-up when on walkable ground (prevent step-up on steep slopes)
                    if on_ground and on_walkable and feet_only and step_height > 0.0 and best_n is not None:
                        bn_x, bn_y, bn_z = best_n
                        if bn_z < floor_cos:  # Steep face (step-able)
                            if debug_step:
                                worker_logs.append(("STEP", f"attempting step-up | pos=({px:.2f},{py:.2f},{pz:.2f}) face_nz={bn_z:.3f}"))

                            test_z = pz + step_height
                            head_ray_z = test_z + height

                            # UNIFIED headroom check - tests both static and dynamic
                            tris_counter = [0]
                            headroom_result = cast_ray(
                                (px, py, head_ray_z), (0, 0, 1), step_height,
                                _cached_grid, unified_dynamic_meshes, tris_counter
                            )
                            total_tris += tris_counter[0]
                            head_clear = (headroom_result is None)

                            if head_clear:
                                remaining_move = move_len - allowed
                                if remaining_move > 0.01:
                                    # Drop down to find ground
                                    drop_ox = px + fwd_x * min(remaining_move, radius)
                                    drop_oy = py + fwd_y * min(remaining_move, radius)
                                    drop_max = step_height + snap_down + 1.0

                                    # UNIFIED ground detection at drop position
                                    tris_counter = [0]
                                    drop_result = cast_ray(
                                        (drop_ox, drop_oy, test_z + 1.0), (0, 0, -1), drop_max,
                                        _cached_grid, unified_dynamic_meshes, tris_counter
                                    )
                                    total_tris += tris_counter[0]

                                    if drop_result:
                                        drop_dist, step_ground_n, drop_source, drop_obj_id = drop_result
                                        step_ground_z = test_z + 1.0 - drop_dist

                                        # Check if walkable
                                        gn_check = step_ground_n[2]
                                        if gn_check >= floor_cos:
                                            if debug_step:
                                                worker_logs.append(("STEP", f"SUCCESS source={drop_source} | drop_pos=({drop_ox:.2f},{drop_oy:.2f},{step_ground_z:.2f}) gn_z={gn_check:.3f}"))
                                            px = drop_ox
                                            py = drop_oy
                                            pz = step_ground_z
                                            vz = 0.0
                                            on_ground = True
                                            on_walkable = True
                                            gn_x, gn_y, gn_z = step_ground_n
                                            did_step_up = True
                                            coyote_remaining = coyote_time

                    # ─────────────────────────────────────────────────────────
                    # 5b. WALL SLIDE (if blocked and not step-up) - UNIFIED
                    # ─────────────────────────────────────────────────────────
                    if not did_step_up and best_n is not None:
                        remaining = move_len - allowed
                        if remaining > 0.001:
                            bn_x, bn_y, bn_z = best_n
                            is_steep_face = bn_z < floor_cos and bn_z > 0.1

                            # Compute slide direction (tangent to wall)
                            dot = fwd_x * bn_x + fwd_y * bn_y
                            slide_x = fwd_x - bn_x * dot
                            slide_y = fwd_y - bn_y * dot
                            slide_len = math.sqrt(slide_x*slide_x + slide_y*slide_y)

                            # Don't slide if too perpendicular
                            if slide_len < 0.259:
                                slide_len = 0.0

                            # Block uphill sliding on steep slopes
                            if is_steep_face and on_ground and slide_len > 1e-9:
                                uphill_xy_len = math.sqrt(bn_x*bn_x + bn_y*bn_y)
                                if uphill_xy_len > 0.001:
                                    uphill_x = bn_x / uphill_xy_len
                                    uphill_y = bn_y / uphill_xy_len
                                    slide_nx = slide_x / slide_len
                                    slide_ny = slide_y / slide_len
                                    uphill_dot = slide_nx * uphill_x + slide_ny * uphill_y
                                    if uphill_dot > -0.1:
                                        slide_len = 0.0

                            if slide_len > 1e-9:
                                slide_x /= slide_len
                                slide_y /= slide_len
                                slide_dist = remaining * 0.65

                                # UNIFIED slide collision check
                                tris_counter = [0]
                                slide_result = cast_ray(
                                    (px, py, pz + height * 0.5), (slide_x, slide_y, 0),
                                    slide_dist + radius, _cached_grid, unified_dynamic_meshes, tris_counter
                                )
                                total_tris += tris_counter[0]

                                slide_allowed = slide_dist
                                if slide_result:
                                    slide_hit_dist = slide_result[0]
                                    slide_allowed = min(slide_allowed, max(0, slide_hit_dist - radius))

                                if slide_allowed > 0.01:
                                    px += slide_x * slide_allowed
                                    py += slide_y * slide_allowed
                                    did_slide = True

                                # Reduce velocity
                                vx *= 0.65
                                vy *= 0.65

                                # SLIDE DIAGNOSTICS (logging only - after slide is applied)
                                if debug_slide:
                                    try:
                                        slide_applied = slide_allowed if slide_allowed > 0.01 else 0.0
                                        slide_requested = slide_dist
                                        slide_blocked = slide_result is not None
                                        effectiveness = (slide_applied / slide_requested * 100.0) if slide_requested > 0.001 else 0.0
                                        worker_logs.append(("SLIDE",
                                            f"applied={slide_applied:.3f}m requested={slide_requested:.3f}m eff={effectiveness:.0f}% "
                                            f"normal=({bn_x:.2f},{bn_y:.2f},{bn_z:.2f}) blocked={slide_blocked}"))
                                    except:
                                        pass  # Silently ignore any logging errors
                else:
                    # No collision - full movement
                    px += move_x
                    py += move_y
                    if debug_horizontal and move_len > 1e-9:
                        worker_logs.append(("HORIZONTAL", f"clear move={move_len:.3f}m | {total_rays}rays {total_tris}tris"))

            elif move_len > 1e-9:
                # No grid cached - just move
                px += move_x
                py += move_y

            # ─────────────────────────────────────────────────────────────────
            # 5.1 PROACTIVE COLLISION RESPONSE
            # Always push player away if penetrating a dynamic mesh
            # Normal collision only BLOCKS movement, doesn't PUSH away
            # ─────────────────────────────────────────────────────────────────
            if proactive_best_d is not None:
                pn_x, pn_y, pn_z = proactive_best_n

                # Get mesh velocity
                mesh_vel_toward = 0.0
                mesh_speed_total = 0.0
                if proactive_obj_id in dynamic_velocities:
                    mvx, mvy, mvz = dynamic_velocities[proactive_obj_id]
                    mesh_speed_total = math.sqrt(mvx*mvx + mvy*mvy)
                    # How fast is mesh moving toward player? (along normal)
                    mesh_vel_toward = -(mvx * pn_x + mvy * pn_y)  # Negative because normal points away from mesh

                # Calculate "safe distance" - how far we need to be to survive next frame
                # At high speeds, need bigger safety margin
                safe_distance = radius + mesh_speed_total * dt * 2.0

                # Penetration relative to safe distance (not just radius)
                effective_penetration = safe_distance - proactive_best_d

                # Always push if within safe distance
                if effective_penetration > -0.1:  # Within 10cm of safe threshold
                    h_blocked = True

                    # Push amount = get to safe distance + buffer
                    # For 20m/s mesh: safe_distance ~= 0.22 + 20*0.033*2 = 1.54m
                    push_base = max(0.0, effective_penetration + 0.05)

                    # Extra velocity-based push for very fast meshes
                    push_velocity = max(0.0, mesh_vel_toward * dt * 2.0)

                    push_total = push_base + push_velocity

                    if push_total > 0.001:
                        px += pn_x * push_total
                        py += pn_y * push_total

                    # Remove velocity component into wall
                    vn = vx * pn_x + vy * pn_y
                    if vn > 0.0:
                        vx -= pn_x * vn
                        vy -= pn_y * vn

                    # Inherit mesh velocity (horizontal carry)
                    if mesh_vel_toward > 0.5 and proactive_obj_id in dynamic_velocities:
                        mvx, mvy, mvz = dynamic_velocities[proactive_obj_id]
                        vx += mvx
                        vy += mvy

                    if debug_horizontal:
                        worker_logs.append(("HORIZONTAL",
                            f"PROACTIVE_PUSH dist={proactive_best_d:.3f}m safe={safe_distance:.2f}m "
                            f"pen={effective_penetration:.3f}m vel={mesh_vel_toward:.1f}m/s push={push_total:.3f}m"))

            # Post-collision diagnostic: did we miss any proximity meshes?
            if debug_horizontal and proximity_meshes and not h_blocked:
                worker_logs.append(("HORIZONTAL",
                    f"MISSED? prox_meshes={len(proximity_meshes)} no_collision | "
                    f"move_len={move_len:.3f}m player_vel=({vx:.2f},{vy:.2f})"))

            # ─────────────────────────────────────────────────────────────────
            # 5.5 VERTICAL BODY INTEGRITY CHECK - UNIFIED for all geometry
            # ─────────────────────────────────────────────────────────────────
            # Cast vertical ray from feet to head - if blocked, character is embedded in mesh
            body_embedded = False
            embed_distance = None

            # Feet at 0.1m to match horizontal ray height (better low-obstacle detection)
            feet_pos = (px, py, pz + 0.1)
            head_pos = (px, py, pz + height - radius)
            body_height = (height - radius) - 0.1  # Distance from feet (0.1m) to head

            tris_counter = [0]
            total_rays += 1
            body_result = cast_ray(
                feet_pos, (0, 0, 1), body_height,
                _cached_grid, unified_dynamic_meshes, tris_counter
            )
            total_tris += tris_counter[0]

            if body_result:
                embed_distance, embed_normal, embed_source, embed_obj_id = body_result
                body_embedded = True

                if debug_body:
                    penetration_pct = (embed_distance / body_height) * 100.0
                    source_str = f"dynamic_{embed_obj_id}" if embed_source == "dynamic" else "static"
                    worker_logs.append(("BODY",
                        f"EMBEDDED source={source_str} hit={embed_distance:.3f}m pct={penetration_pct:.1f}%"))

            if debug_body:
                status = "EMBEDDED" if body_embedded else "CLEAR"
                worker_logs.append(("BODY", f"[{status}] feet=({feet_pos[0]:.2f},{feet_pos[1]:.2f},{feet_pos[2]:.2f}) head=({head_pos[0]:.2f},{head_pos[1]:.2f},{head_pos[2]:.2f}) h={body_height:.2f}m"))

            # ─────────────────────────────────────────────────────────────────
            # 5.6 EMBEDDING RESOLUTION (Prevention + Correction)
            # ─────────────────────────────────────────────────────────────────
            # Use vertical integrity ray data to prevent/fix mesh penetration
            if body_embedded and embed_distance is not None:
                # Mesh detected between feet and head at embed_distance from feet
                # embed_distance = distance from feet to the penetrating mesh

                # CASE 1: PREVENTION - Moving downward into mesh (falling scenario)
                # Stop character at mesh surface instead of penetrating through
                if vz < 0:
                    # Character is falling and would penetrate mesh
                    # Position feet at mesh surface: pz + embed_distance
                    correction = embed_distance
                    pz += correction
                    vz = 0.0  # Kill downward velocity
                    on_ground = True  # Treat as landing on surface
                    on_walkable = True  # Assume walkable (will be verified by ground detection)

                    if debug_body:
                        worker_logs.append(("BODY", f"PREVENT-FALL corrected={correction:.3f}m landing on embedded mesh"))

                # CASE 2: CORRECTION - Already embedded from side collision
                # Push character up to clear the penetration
                else:
                    # Character entered mesh horizontally (side collision)
                    # Need to push up so mesh is no longer between feet and head
                    # Add small buffer to ensure full clearance
                    correction = embed_distance + 0.05  # 5cm buffer
                    pz += correction
                    vz = max(0.0, vz)  # Preserve upward velocity if any, kill downward

                    if debug_body:
                        worker_logs.append(("BODY", f"CORRECT-SIDE corrected={correction:.3f}m pushed up from embedded mesh"))

            # ─────────────────────────────────────────────────────────────────
            # 6. CEILING CHECK (if moving up) - UNIFIED for all geometry
            # ─────────────────────────────────────────────────────────────────
            if vz > 0.0:
                up_dist = vz * dt
                head_z = pz + height

                tris_counter = [0]
                total_rays += 1
                ceiling_result = cast_ray(
                    (px, py, head_z), (0, 0, 1), up_dist,
                    _cached_grid, unified_dynamic_meshes, tris_counter
                )
                total_tris += tris_counter[0]

                if ceiling_result:
                    ceil_dist, ceil_normal, ceil_source, ceil_obj_id = ceiling_result
                    pz = head_z + ceil_dist - height
                    vz = 0.0
                    hit_ceiling = True

            # ─────────────────────────────────────────────────────────────────
            # 7. VERTICAL MOVEMENT + GROUND DETECTION (UNIFIED)
            # Single cast_ray() tests BOTH static and dynamic - identical physics
            # ─────────────────────────────────────────────────────────────────
            dz = vz * dt
            was_grounded = on_ground
            ground_hit_source = None  # Track if standing on static or dynamic mesh

            # UNIFIED GROUND RAY - tests ALL geometry in one call
            ray_start_z = pz + 1.0  # Guard above
            ray_max = snap_down + 1.0 + (abs(dz) if dz < 0 else 0)

            ground_hit_z = None
            ground_hit_n = None

            tris_counter = [0]
            total_rays += 1
            ground_result = cast_ray(
                (px, py, ray_start_z), (0, 0, -1), ray_max,
                _cached_grid, unified_dynamic_meshes, tris_counter
            )
            total_tris += tris_counter[0]

            if ground_result:
                dist, normal, source, obj_id = ground_result
                # Sanity check: hit must be below player
                hit_z = ray_start_z - dist
                if hit_z <= pz + 0.1:  # Allow small tolerance
                    ground_hit_z = hit_z
                    ground_hit_n = normal
                    if source == "dynamic":
                        ground_hit_source = f"dynamic_{obj_id}"
                    else:
                        ground_hit_source = "static"

            # Debug logging - UNIFIED: Ground detection shows source (static or dynamic)
            if debug_ground and ground_hit_z is not None:
                worker_logs.append(("GROUND",
                    f"HIT source={ground_hit_source} z={ground_hit_z:.2f}m "
                    f"normal=({ground_hit_n[0]:.2f},{ground_hit_n[1]:.2f},{ground_hit_n[2]:.2f}) | "
                    f"player_z={pz:.2f} tris={tris_counter[0]}"))

            # Apply vertical movement and ground snap (unified for all geometry)
            if dz < 0.0:  # Falling
                target_z = pz + dz
                if ground_hit_z is not None and target_z <= ground_hit_z:
                    pz = ground_hit_z
                    vz = 0.0
                    on_ground = True
                    if ground_hit_n is not None:
                        gn_x, gn_y, gn_z = ground_hit_n
                        on_walkable = gn_z >= floor_cos
                    coyote_remaining = coyote_time
                else:
                    pz = target_z
                    on_ground = False
                    on_walkable = False
            else:
                pz += dz

            # Ground snap (when grounded) or unground (when no ground found)
            if ground_hit_z is not None and abs(ground_hit_z - pz) <= snap_down and vz <= 0.0:
                # Ground found within snap distance - snap to it
                pz = ground_hit_z
                on_ground = True
                vz = 0.0
                if ground_hit_n is not None:
                    gn_x, gn_y, gn_z = ground_hit_n
                    on_walkable = gn_z >= floor_cos
                coyote_remaining = coyote_time
                if debug_ground:
                    worker_logs.append(("GROUND", f"ON_GROUND snap | dist={abs(ground_hit_z - pz):.3f}m gn_z={gn_z:.3f} walkable={on_walkable}"))
            elif ground_hit_z is None and was_grounded:
                # Was grounded but no ground found - walked off a ledge!
                on_ground = False
                on_walkable = False
                coyote_remaining = coyote_time  # Grant coyote time
                if debug_ground:
                    worker_logs.append(("GROUND", f"airborne | walked_off_ledge coyote={coyote_time:.2f}s"))
            elif not on_ground and was_grounded:
                coyote_remaining = coyote_time

            if debug_ground and ground_hit_z is not None and not on_ground:
                worker_logs.append(("GROUND", f"airborne | dist={abs(ground_hit_z - pz):.3f}m too_far | {total_tris}tris"))

            # ─────────────────────────────────────────────────────────────────
            # 8. STEEP SLOPE SLIDING (after ground detection updates the normal)
            # ─────────────────────────────────────────────────────────────────
            # Calculate slope angle from ground normal to check threshold
            steep_slope_detected = False
            slope_angle = 0.0
            if on_ground:
                gn_len = math.sqrt(gn_x*gn_x + gn_y*gn_y + gn_z*gn_z)
                if gn_len > 0.001:
                    n_x = gn_x / gn_len
                    n_y = gn_y / gn_len
                    n_z = gn_z / gn_len
                    slope_angle = math.degrees(math.acos(min(1.0, max(-1.0, n_z))))

                    # Match slope_limit_deg to eliminate dead zone between walkable and steep
                    steep_slope_detected = slope_angle > slope_limit_deg

            # If we're on ground on a steep slope (> slope_limit_deg), apply sliding and blocking
            if on_ground and steep_slope_detected:
                gn_len = math.sqrt(gn_x*gn_x + gn_y*gn_y + gn_z*gn_z)
                if gn_len > 0.001:
                    n_x = gn_x / gn_len
                    n_y = gn_y / gn_len
                    n_z = gn_z / gn_len

                    # UPHILL BLOCKING: Remove uphill velocity component (moved here to use current frame's normal)
                    gn_xy_len = math.sqrt(n_x*n_x + n_y*n_y)
                    if gn_xy_len > 0.001:
                        # Normalize uphill direction
                        # CRITICAL: Normal points DOWN the slope (outward from surface)
                        # To get uphill, we need to NEGATE it!
                        uphill_x = -n_x / gn_xy_len
                        uphill_y = -n_y / gn_xy_len

                        # Project current velocity onto uphill direction
                        uphill_vel = vx * uphill_x + vy * uphill_y

                        # Slope angle already calculated above

                        # POST-MOVEMENT CORRECTION: Gentle backup if Step 4.5 missed anything
                        # (Step 4.5 does main blocking BEFORE movement)
                        if slope_angle > 65.0 and uphill_vel > 0.0:
                            # Just remove any remaining uphill velocity, minimal force
                            vx = vx - uphill_x * uphill_vel
                            vy = vy - uphill_y * uphill_vel

                            # Very gentle correction only
                            downhill_x = -uphill_x
                            downhill_y = -uphill_y
                            vx += downhill_x * 2.0  # Minimal correction
                            vy += downhill_y * 2.0

                            if debug_slopes:
                                worker_logs.append(("SLOPES", f"POST-CORRECT angle={slope_angle:.0f}° (backup)"))

                        # Slopes slope_limit_deg-65°: Gentle correction
                        elif slope_angle > slope_limit_deg and uphill_vel > 0.0:
                            # Just remove uphill velocity, no pushback
                            vx = vx - uphill_x * uphill_vel
                            vy = vy - uphill_y * uphill_vel

                        # CRITICAL: On slopes > 65°, CLAMP Z position to prevent upward movement
                        # This is the nuclear option - directly prevent position from moving up
                        if slope_angle > 65.0 and on_ground:
                            # If character somehow moved upward on steep slope, FORCE them back down
                            # Store ground contact Z as maximum allowed Z
                            max_allowed_z = ground_hit_z if ground_hit_z is not None else pz
                            if pz > max_allowed_z:
                                if debug_slopes:
                                    worker_logs.append(("SLOPES", f"Z-CLAMP angle={slope_angle:.0f}° prevented {pz - max_allowed_z:.3f}m upward movement"))
                                pz = max_allowed_z  # FORCE character to ground level or below

                    # Compute slide direction (down the slope in XY plane)
                    # The normal points OUT of the slope surface.
                    # For a slope facing up-and-outward, normal XY points UPHILL.
                    # To slide DOWNHILL, we go OPPOSITE to normal's XY projection.
                    # But wait - if normal points outward from surface, and surface
                    # is tilted up, then normal.xy points in the uphill direction.
                    # So downhill = +normal.xy (not negative!)
                    #
                    # Actually: Consider a ramp going up in +X direction.
                    # The surface normal would point up and back: (+small, 0, +large)
                    # normalized, n_x is positive (pointing in +X, which is uphill)
                    # To slide DOWN, we want -X direction, so slide = -n_x
                    #
                    # Hmm, let me think again with gravity:
                    # Gravity pulls down (0,0,-1). Project onto slope plane:
                    # g_tangent = g - n*(g.n) = (0,0,-1) - n*(-n_z) = (0,0,-1) + n*n_z
                    # = (n_x*n_z, n_y*n_z, -1 + n_z*n_z)
                    # The XY components are (n_x*n_z, n_y*n_z) - this IS the downhill direction

                    slide_xy_len = math.sqrt(n_x*n_x + n_y*n_y)
                    if slide_xy_len > 0.001:
                        # Gravity projected onto slope gives downhill direction
                        # g_tangent.xy = (n_x * n_z, n_y * n_z)
                        slide_x = n_x * n_z
                        slide_y = n_y * n_z

                        # Normalize the slide direction
                        slide_len = math.sqrt(slide_x*slide_x + slide_y*slide_y)
                        if slide_len > 0.001:
                            slide_x /= slide_len
                            slide_y /= slide_len

                            # Slope steepness factor (0 = flat, 1 = vertical)
                            steepness = 1.0 - n_z

                            # Apply slide acceleration
                            # Use config parameter instead of hardcoded value
                            # MASSIVE slide force to overpower player input lerp system
                            # Doubled from 800 to 1600 for faster sliding
                            slope_slide_gain = cfg.get("steep_slide_gain", 1600.0)
                            slide_accel = slope_slide_gain * steepness * dt

                            vx += slide_x * slide_accel
                            vy += slide_y * slide_accel

                            # Ensure minimum downhill speed (prevents "sticking" on steep slopes)
                            # Increased from 2.5 to 8.0 for more responsive slide start
                            steep_min_speed = cfg.get("steep_min_speed", 8.0)
                            current_downhill_speed = vx * slide_x + vy * slide_y
                            if current_downhill_speed > 0.0 and current_downhill_speed < steep_min_speed:
                                # Boost velocity to maintain minimum slide speed
                                deficit = steep_min_speed - current_downhill_speed
                                vx += slide_x * deficit
                                vy += slide_y * deficit

                            # Limit maximum slide speed to prevent infinite acceleration
                            # Increased from 30 to 50 m/s for faster steep slope sliding
                            max_slide_speed = cfg.get("max_slide_speed", 50.0)  # m/s
                            slide_speed = math.sqrt(vx*vx + vy*vy)
                            if slide_speed > max_slide_speed:
                                scale = max_slide_speed / slide_speed
                                vx *= scale
                                vy *= scale

                            # Position push removed - was causing character to launch off ground
                            # slide_push = steepness * dt * 8.0
                            # px += slide_x * slide_push
                            # py += slide_y * slide_push

                            # SURFACE TRACKING: Project velocity onto slope to prevent bouncing
                            # When sliding fast on steep slopes, constrain velocity to follow surface
                            # This prevents character from launching off due to horizontal speed
                            if current_downhill_speed > 5.0 and steepness > 0.3:
                                # Project velocity onto slope plane: v_proj = v - (v · n) * n
                                # This removes the component perpendicular to surface
                                vel_dot_normal = vx * n_x + vy * n_y + vz * n_z
                                if vel_dot_normal > 0.0:  # Moving away from surface
                                    # Remove perpendicular component to glue to surface
                                    vx -= n_x * vel_dot_normal * 0.8  # 0.8 = damping factor
                                    vy -= n_y * vel_dot_normal * 0.8
                                    vz -= n_z * vel_dot_normal * 0.8

                            # Apply standard gravity (removed 1.2x multiplier that was too aggressive)
                            vz += gravity * dt

                            # Debug logging for steep slope sliding
                            if debug_slopes:
                                slope_angle = math.degrees(math.acos(min(1.0, max(-1.0, n_z))))
                                surface_tracking = "TRACKED" if (current_downhill_speed > 5.0 and steepness > 0.3) else "free"
                                worker_logs.append(("SLOPES", f"GRAVITY-SLIDE angle={slope_angle:.0f}° normal=({n_x:.2f},{n_y:.2f},{n_z:.2f}) "
                                                              f"dir=({slide_x:.2f},{slide_y:.2f}) "
                                                              f"vel_downhill={current_downhill_speed:.2f} max={max_slide_speed:.2f} "
                                                              f"accel={slide_accel:.2f} steep={steepness:.2f} track={surface_tracking}"))

            # ─────────────────────────────────────────────────────────────────
            # BUILD RESULT
            # ─────────────────────────────────────────────────────────────────
            calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

            # ═════════════════════════════════════════════════════════════════
            # PHYSICS SUMMARY - Unified physics (static + dynamic identical)
            # ═════════════════════════════════════════════════════════════════
            # Log unified physics system status (static + dynamic in single path)
            if debug_physics:
                transform_time = dynamic_transform_time_us
                total_time = calc_time_us
                physics_time = total_time - transform_time

                # Cache state
                cached_mesh_count = len(_cached_dynamic_meshes)
                active_mesh_count = len(unified_dynamic_meshes)

                # Unified physics summary (single log with all info)
                ground_src = ground_hit_source if ground_hit_source else "none"

                if active_mesh_count > 0:
                    # With dynamic meshes: show timing breakdown
                    worker_logs.append(("PHYSICS",
                        f"total={total_time:.0f}us (xform={transform_time:.0f}us) | "
                        f"static+dynamic={active_mesh_count} | "
                        f"rays={total_rays} tris={total_tris} | "
                        f"ground={ground_src}"))
                else:
                    # Static only: simpler log
                    worker_logs.append(("PHYSICS",
                        f"total={total_time:.0f}us | static_only | "
                        f"rays={total_rays} tris={total_tris} | "
                        f"ground={ground_src}"))

            result_data = {
                "pos": (px, py, pz),
                "vel": (vx, vy, vz),
                "on_ground": on_ground,
                "on_walkable": on_walkable,
                "ground_normal": (gn_x, gn_y, gn_z),
                "ground_hit_source": ground_hit_source,  # "static", "dynamic_ObjectName", or None
                "coyote_remaining": coyote_remaining,
                "jump_consumed": jump_consumed,
                "logs": worker_logs,  # Fast buffer logs (sent to main thread)
                "debug": {
                    "rays_cast": total_rays,
                    "triangles_tested": total_tris,
                    "cells_traversed": total_cells,
                    "calc_time_us": calc_time_us,
                    "dynamic_meshes_active": len(unified_dynamic_meshes),
                    "dynamic_transform_time_us": dynamic_transform_time_us,
                    "h_blocked": h_blocked,
                    "did_step_up": did_step_up,
                    "did_slide": did_slide,
                    "hit_ceiling": hit_ceiling,
                    "body_embedded": body_embedded,
                    "vertical_ray": {
                        "origin": (px, py, pz + radius),
                        "end": (px, py, pz + height - radius),
                        "blocked": body_embedded
                    }
                }
            }

        # =====================================================================
        # CAMERA OCCLUSION HANDLERS
        # =====================================================================

        # REMOVED: Old KCC handlers (KCC_INPUT_VECTOR, KCC_RAYCAST, KCC_RAYCAST_GRID, KCC_RAYCAST_CACHED)
        # Now using unified KCC_PHYSICS_STEP handler above

        elif job.job_type == "CAMERA_OCCLUSION_FULL":
            # ═══════════════════════════════════════════════════════════════════
            # UNIFIED CAMERA OCCLUSION - Uses same raycast as KCC physics
            # Static grid + dynamic meshes go through unified_raycast
            # ═══════════════════════════════════════════════════════════════════
            import math

            calc_start = time.perf_counter()

            # Ray parameters
            ray_origin = job.data.get("ray_origin", (0.0, 0.0, 0.0))
            ray_direction = job.data.get("ray_direction", (0.0, 0.0, -1.0))
            max_distance = job.data.get("max_distance", 10.0)
            dynamic_transforms = job.data.get("dynamic_transforms", {})

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
            if _cached_grid is not None:
                grid_data = {
                    "cells": _cached_grid.get("cells", {}),
                    "triangles": _cached_grid.get("triangles", []),
                    "bounds_min": _cached_grid.get("bounds_min", (0, 0, 0)),
                    "bounds_max": _cached_grid.get("bounds_max", (0, 0, 0)),
                    "cell_size": _cached_grid.get("cell_size", 1.0),
                    "grid_dims": _cached_grid.get("grid_dims", (1, 1, 1)),
                }

            # ─────────────────────────────────────────────────────────────────
            # Build unified_dynamic_meshes FROM PERSISTENT CACHE
            # ─────────────────────────────────────────────────────────────────
            # Camera uses the same transform cache as KCC physics.
            # If transforms are sent in the job, update the cache first.
            # Then build mesh list from ALL cached transforms.
            # Note: _cached_dynamic_transforms is already declared global in KCC handler
            # ─────────────────────────────────────────────────────────────────
            dynamic_transforms_update = job.data.get("dynamic_transforms", {})

            # Update cache with any fresh transforms
            # Mesh triangles are cached via targeted broadcast_job (guaranteed delivery)
            for obj_id, matrix_4x4 in dynamic_transforms_update.items():
                cached = _cached_dynamic_meshes.get(obj_id)
                if cached is None:
                    continue
                local_aabb = cached.get("local_aabb")
                world_aabb = transform_aabb_by_matrix(local_aabb, matrix_4x4) if local_aabb else None
                _cached_dynamic_transforms[obj_id] = (matrix_4x4, world_aabb)

            # Build mesh list from ALL cached transforms
            unified_dynamic_meshes = []

            for obj_id, (matrix_4x4, world_aabb) in _cached_dynamic_transforms.items():
                cached = _cached_dynamic_meshes.get(obj_id)
                if cached is None:
                    continue

                local_triangles = cached["triangles"]

                # Compute inverse matrix for local-space ray testing
                inv_matrix = invert_matrix_4x4(matrix_4x4)
                if inv_matrix is None:
                    continue

                # Compute bounding sphere from AABB
                if world_aabb:
                    aabb_min, aabb_max = world_aabb
                    center = (
                        (aabb_min[0] + aabb_max[0]) * 0.5,
                        (aabb_min[1] + aabb_max[1]) * 0.5,
                        (aabb_min[2] + aabb_max[2]) * 0.5
                    )
                    half_diag = math.sqrt(
                        (aabb_max[0] - aabb_min[0])**2 +
                        (aabb_max[1] - aabb_min[1])**2 +
                        (aabb_max[2] - aabb_min[2])**2
                    ) * 0.5
                    bounding_sphere = (center, half_diag)
                else:
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

            result_data = {
                "hit": hit_found,
                "hit_distance": hit_distance,
                "hit_source": hit_source,
                "static_triangles_tested": result.get("tris_tested", 0),
                "static_cells_traversed": result.get("cells_traversed", 0),
                "dynamic_triangles_tested": 0,  # Included in tris_tested
                "calc_time_us": calc_time_us,
                "method": "CAMERA_UNIFIED",
                "grid_cached": _cached_grid is not None,
                "dynamic_meshes_tested": len(unified_dynamic_meshes),
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

                # Check if this job is targeted at a specific worker
                target = getattr(job, 'target_worker', -1)
                if target >= 0 and target != worker_id:
                    # This job is for a different worker - put it back and try again
                    try:
                        job_queue.put_nowait(job)
                    except Exception:
                        pass  # Queue full, job will be lost (shouldn't happen)
                    continue

                if DEBUG_ENGINE:
                    print(f"[Engine Worker {worker_id}] Processing job {job.job_id} (type: {job.job_type})")

                # Process the job
                result = process_job(job)

                # Add worker_id to result before sending (CRITICAL for grid cache verification)
                result["worker_id"] = worker_id

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
            