# Exp_Game/engine/worker/math.py
"""
Geometric math utilities for worker process.
All functions are pure Python with no external dependencies (except math module).
"""

import math


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


# ============================================================================
# AABB (Axis-Aligned Bounding Box) Operations
# ============================================================================

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
# MATRIX OPERATIONS
# ============================================================================

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
    dir_len = math.sqrt(ldx*ldx + ldy*ldy + ldz*ldz)
    if dir_len > 1e-10:
        ldx /= dir_len
        ldy /= dir_len
        ldz /= dir_len
    else:
        dir_len = 1.0  # Avoid division by zero

    return ((lox, loy, loz), (ldx, ldy, ldz), dir_len)


def transform_point(point, matrix):
    """
    Transform a single point by a 4x4 matrix.

    Args:
        point: (x, y, z) point
        matrix: 16-element flat 4x4 matrix

    Returns:
        (x, y, z) transformed point
    """
    px, py, pz = point
    m = matrix
    return (
        m[0]*px + m[1]*py + m[2]*pz + m[3],
        m[4]*px + m[5]*py + m[6]*pz + m[7],
        m[8]*px + m[9]*py + m[10]*pz + m[11]
    )


def transform_triangle(tri, matrix):
    """
    Transform a triangle by a 4x4 matrix.

    Args:
        tri: ((x,y,z), (x,y,z), (x,y,z)) triangle vertices
        matrix: 16-element flat 4x4 matrix

    Returns:
        ((x,y,z), (x,y,z), (x,y,z)) transformed triangle
    """
    return (
        transform_point(tri[0], matrix),
        transform_point(tri[1], matrix),
        transform_point(tri[2], matrix)
    )


# ============================================================================
# SPATIAL ACCELERATION: Grid for fast ray-triangle culling
# ============================================================================

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


# ============================================================================
# MATRIX TO EULER Z (YAW EXTRACTION)
# ============================================================================

def matrix_to_euler_z(matrix_16):
    """
    Extract Z rotation (yaw) from a 4x4 matrix stored as 16-element tuple.

    IMPORTANT: Blender matrices use column-major indexing: matrix[col][row]
    The serialization in exp_kcc.py stores columns sequentially:
        matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3],  # Column 0
        matrix[1][0], matrix[1][1], ...

    Args:
        matrix_16: 16-element tuple in column-major order
                   (m00, m10, m20, m30, m01, m11, m21, m31, ...)

    Returns:
        float: Z rotation in radians
    """
    # Matrix layout (COLUMN-MAJOR from Blender):
    # [0]  [1]  [2]  [3]    m00 m10 m20 m30  (column 0)
    # [4]  [5]  [6]  [7]    m01 m11 m21 m31  (column 1)
    # [8]  [9]  [10] [11]   m02 m12 m22 m32  (column 2)
    # [12] [13] [14] [15]   m03 m13 m23 m33  (column 3)
    #
    # For XYZ euler, yaw (Z rotation) can be extracted from:
    # yaw = atan2(m10, m00) = atan2(matrix[1], matrix[0])
    m00 = matrix_16[0]   # Column 0, row 0
    m10 = matrix_16[1]   # Column 0, row 1
    return math.atan2(m10, m00)
