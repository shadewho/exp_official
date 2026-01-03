# Exp_Game/engine/worker/reactions/ragdoll.py
"""
Verlet Particle Ragdoll - Worker Side

Each bone joint is a particle. Bones are distance constraints.
Particles collide with static/dynamic meshes.

This runs in the worker process - NO bpy access!
"""

import time
import math

# =============================================================================
# PHYSICS CONSTANTS
# =============================================================================

GRAVITY = (0.0, 0.0, -9.8)
GRAVITY_SCALE = 2.0           # Multiplier for gravity (was 1.0 - too floaty)

CONSTRAINT_ITERATIONS = 4     # Fewer = looser/floppier (was 8)
CONSTRAINT_STIFFNESS = 0.5    # 0-1, lower = looser joints (was 0.9)

DAMPING = 0.995               # Velocity retention (was 0.98 - too much drag)
GROUND_FRICTION = 0.5         # Friction when touching ground

COLLISION_RADIUS = 0.05       # Particle collision sphere radius
COLLISION_PUSH = 1.02         # Push multiplier when colliding (slightly over 1)

# Floor is the minimum - particles can't go below this
FLOOR_OFFSET = 0.02           # Small offset above actual floor


# =============================================================================
# MATH HELPERS (no numpy in workers)
# =============================================================================

def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def vec_scale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)

def vec_length(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def vec_normalize(v):
    l = vec_length(v)
    if l < 0.0001:
        return (0.0, 0.0, 0.0)
    return (v[0]/l, v[1]/l, v[2]/l)

def vec_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vec_cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    )

def vec_lerp(a, b, t):
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t
    )


# =============================================================================
# QUATERNION MATH (for bone rotations)
# =============================================================================

def quat_multiply(q1, q2):
    """Multiply two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    )

def quat_conjugate(q):
    """Conjugate (inverse for unit quaternion)."""
    return (q[0], -q[1], -q[2], -q[3])

def quat_from_two_vectors(v1, v2):
    """
    Create quaternion that rotates v1 to v2.
    Both vectors should be normalized.
    """
    # Cross product gives rotation axis
    cross = vec_cross(v1, v2)
    dot = vec_dot(v1, v2)

    # Handle parallel/anti-parallel cases
    if dot > 0.9999:
        return (1.0, 0.0, 0.0, 0.0)  # Identity
    if dot < -0.9999:
        # 180 degree rotation - pick arbitrary perpendicular axis
        if abs(v1[0]) < 0.9:
            axis = vec_normalize(vec_cross(v1, (1, 0, 0)))
        else:
            axis = vec_normalize(vec_cross(v1, (0, 1, 0)))
        return (0.0, axis[0], axis[1], axis[2])

    # Standard case
    w = 1.0 + dot
    length = math.sqrt(w * 2.0)
    return (
        length / 2.0,
        cross[0] / length,
        cross[1] / length,
        cross[2] / length
    )

def quat_normalize(q):
    """Normalize quaternion."""
    length = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if length < 0.0001:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0]/length, q[1]/length, q[2]/length, q[3]/length)

def matrix_to_quat(m):
    """Convert 4x4 matrix (flat 16 floats) to quaternion."""
    # Extract rotation part (3x3)
    trace = m[0] + m[5] + m[10]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[6] - m[9]) * s
        y = (m[8] - m[2]) * s
        z = (m[1] - m[4]) * s
    elif m[0] > m[5] and m[0] > m[10]:
        s = 2.0 * math.sqrt(1.0 + m[0] - m[5] - m[10])
        w = (m[6] - m[9]) / s
        x = 0.25 * s
        y = (m[4] + m[1]) / s
        z = (m[8] + m[2]) / s
    elif m[5] > m[10]:
        s = 2.0 * math.sqrt(1.0 + m[5] - m[0] - m[10])
        w = (m[8] - m[2]) / s
        x = (m[4] + m[1]) / s
        y = 0.25 * s
        z = (m[9] + m[6]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[10] - m[0] - m[5])
        w = (m[1] - m[4]) / s
        x = (m[8] + m[2]) / s
        y = (m[9] + m[6]) / s
        z = 0.25 * s

    return quat_normalize((w, x, y, z))

def rotate_vector_by_quat(v, q):
    """Rotate vector by quaternion."""
    # v' = q * v * q^-1
    qv = (0.0, v[0], v[1], v[2])
    q_conj = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, qv), q_conj)
    return (result[1], result[2], result[3])


# =============================================================================
# COLLISION DETECTION
# =============================================================================

def point_in_triangle(p, v0, v1, v2):
    """Check if point p projects inside triangle v0,v1,v2 (XY plane check)."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def closest_point_on_triangle(p, v0, v1, v2):
    """Find closest point on triangle to point p."""
    # Edge vectors
    edge0 = vec_sub(v1, v0)
    edge1 = vec_sub(v2, v0)
    v0_to_p = vec_sub(p, v0)

    # Compute dot products
    a = vec_dot(edge0, edge0)
    b = vec_dot(edge0, edge1)
    c = vec_dot(edge1, edge1)
    d = vec_dot(edge0, v0_to_p)
    e = vec_dot(edge1, v0_to_p)

    det = a * c - b * b
    if abs(det) < 0.0001:
        return v0  # Degenerate triangle

    s = (c * d - b * e) / det
    t = (a * e - b * d) / det

    # Clamp to triangle
    s = max(0.0, min(1.0, s))
    t = max(0.0, min(1.0, t))
    if s + t > 1.0:
        # On edge v1-v2
        scale = 1.0 / (s + t)
        s *= scale
        t *= scale

    # Compute closest point
    return vec_add(v0, vec_add(vec_scale(edge0, s), vec_scale(edge1, t)))


def triangle_normal(v0, v1, v2):
    """Get triangle normal."""
    edge1 = vec_sub(v1, v0)
    edge2 = vec_sub(v2, v0)
    n = vec_cross(edge1, edge2)
    return vec_normalize(n)


def collide_particle_with_triangles(pos, triangles, radius):
    """
    Check if particle at pos collides with any triangle.
    Returns (collided, new_pos, normal)
    """
    closest_dist = float('inf')
    closest_point = None
    closest_normal = None

    for tri in triangles:
        if len(tri) < 9:
            continue

        v0 = (tri[0], tri[1], tri[2])
        v1 = (tri[3], tri[4], tri[5])
        v2 = (tri[6], tri[7], tri[8])

        # Find closest point on triangle
        cp = closest_point_on_triangle(pos, v0, v1, v2)

        # Distance to closest point
        delta = vec_sub(pos, cp)
        dist = vec_length(delta)

        if dist < closest_dist and dist < radius:
            closest_dist = dist
            closest_point = cp
            closest_normal = triangle_normal(v0, v1, v2)

    if closest_point is not None:
        # Push particle out along normal
        push_dist = radius - closest_dist
        if push_dist > 0:
            new_pos = vec_add(pos, vec_scale(closest_normal, push_dist * COLLISION_PUSH))
            return (True, new_pos, closest_normal)

    return (False, pos, None)


def get_nearby_triangles(pos, cached_grid, cached_dynamic_meshes, cached_dynamic_transforms, search_radius=1.0):
    """Get triangles near a position from static grid and dynamic meshes."""
    triangles = []

    # Static grid triangles
    # Grid stores: cells = {cell_key: [tri_index, ...]} and triangles = [[v0, v1, v2], ...]
    # Each vertex is (x, y, z), so we need to convert to flat 9-tuple
    if cached_grid:
        cell_size = cached_grid.get("cell_size", 1.0)
        cells = cached_grid.get("cells", {})
        grid_triangles = cached_grid.get("triangles", [])  # Actual triangle data
        bounds_min = cached_grid.get("bounds_min", (0, 0, 0))

        # Calculate cell index based on grid bounds
        min_x, min_y, min_z = bounds_min
        cx = int((pos[0] - min_x) / cell_size)
        cy = int((pos[1] - min_y) / cell_size)
        cz = int((pos[2] - min_z) / cell_size)

        tested_indices = set()  # Avoid duplicate triangles
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    cell_key = (cx + dx, cy + dy, cz + dz)
                    tri_indices = cells.get(cell_key, [])
                    for tri_idx in tri_indices:
                        if tri_idx in tested_indices:
                            continue
                        tested_indices.add(tri_idx)
                        # Look up actual triangle and convert to flat format
                        if 0 <= tri_idx < len(grid_triangles):
                            tri = grid_triangles[tri_idx]  # [v0, v1, v2]
                            v0, v1, v2 = tri[0], tri[1], tri[2]
                            # Convert to flat 9-tuple: (x0,y0,z0, x1,y1,z1, x2,y2,z2)
                            flat_tri = (v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], v2[0], v2[1], v2[2])
                            triangles.append(flat_tri)

    # Dynamic mesh triangles
    # Dynamic meshes store triangles as [(v0, v1, v2), ...] where each v is (x, y, z)
    if cached_dynamic_meshes and cached_dynamic_transforms:
        for mesh_name, mesh_data in cached_dynamic_meshes.items():
            transform = cached_dynamic_transforms.get(mesh_name)
            if not transform or not mesh_data:
                continue

            local_tris = mesh_data.get("triangles", [])

            # Transform triangles to world space
            for tri in local_tris:
                if len(tri) < 3:
                    continue

                v0, v1, v2 = tri[0], tri[1], tri[2]

                # Apply transform (assumes transform is 4x4 matrix flat list)
                if len(transform) >= 16:
                    v0_w = transform_point(v0, transform)
                    v1_w = transform_point(v1, transform)
                    v2_w = transform_point(v2, transform)
                else:
                    v0_w, v1_w, v2_w = v0, v1, v2

                # Convert to flat 9-tuple
                flat_tri = (v0_w[0], v0_w[1], v0_w[2],
                            v1_w[0], v1_w[1], v1_w[2],
                            v2_w[0], v2_w[1], v2_w[2])
                triangles.append(flat_tri)

    return triangles


def transform_point(p, m):
    """Transform point by 4x4 matrix (flat list)."""
    x = m[0]*p[0] + m[4]*p[1] + m[8]*p[2] + m[12]
    y = m[1]*p[0] + m[5]*p[1] + m[9]*p[2] + m[13]
    z = m[2]*p[0] + m[6]*p[1] + m[10]*p[2] + m[14]
    return (x, y, z)


# =============================================================================
# VERLET INTEGRATION
# =============================================================================

def verlet_integrate(particles, prev_particles, dt, fixed_mask, floor_z, logs):
    """
    Verlet integration step.
    Updates particles in place, returns new prev_particles.
    """
    new_prev = []
    gravity_step = vec_scale(GRAVITY, GRAVITY_SCALE * dt * dt)

    floor_hits = 0
    max_vel_z = 0.0
    min_z_before = float('inf')
    min_z_after = float('inf')

    for i, (pos, prev, fixed) in enumerate(zip(particles, prev_particles, fixed_mask)):
        new_prev.append(pos)

        if fixed:
            continue

        min_z_before = min(min_z_before, pos[2])

        # Velocity = current - previous (Verlet style)
        vel = vec_sub(pos, prev)
        max_vel_z = max(max_vel_z, abs(vel[2]))

        # Apply damping
        vel = vec_scale(vel, DAMPING)

        # New position = current + velocity + gravity
        new_pos = vec_add(vec_add(pos, vel), gravity_step)

        # Floor collision (simple)
        if new_pos[2] < floor_z + FLOOR_OFFSET:
            new_pos = (new_pos[0], new_pos[1], floor_z + FLOOR_OFFSET)
            # Apply friction
            vel_xz = (vel[0] * GROUND_FRICTION, vel[1] * GROUND_FRICTION, 0.0)
            new_prev[i] = vec_sub(new_pos, vel_xz)
            floor_hits += 1

        min_z_after = min(min_z_after, new_pos[2])
        particles[i] = new_pos

    # Calculate average velocity for logging
    total_vel_z = 0.0
    for i, (pos, prev) in enumerate(zip(particles, new_prev)):
        if not fixed_mask[i]:
            vel_z = pos[2] - prev[2]
            total_vel_z += vel_z
    avg_vel_z = total_vel_z / max(1, len(particles) - sum(fixed_mask))

    # Log integration summary
    logs.append(("RAGDOLL", f"INTEGRATE grav_z={gravity_step[2]:.4f} floor={floor_z:.3f}"))
    logs.append(("RAGDOLL", f"VELOCITY avg_z={avg_vel_z:.4f} max_z={max_vel_z:.4f} floor_hits={floor_hits}"))
    logs.append(("RAGDOLL", f"FALL min_z: {min_z_before:.3f}->{min_z_after:.3f} (delta={min_z_after-min_z_before:.4f})"))

    return new_prev


def satisfy_constraints(particles, constraints, fixed_mask, logs=None, log_label=""):
    """
    Satisfy distance constraints between particles.
    Iterates multiple times for stability.
    """
    # Track constraint stretch for logging
    max_stretch = 0.0
    total_stretch = 0.0

    for _ in range(CONSTRAINT_ITERATIONS):
        for p1_idx, p2_idx, rest_length in constraints:
            p1 = particles[p1_idx]
            p2 = particles[p2_idx]

            delta = vec_sub(p2, p1)
            current_length = vec_length(delta)

            if current_length < 0.0001:
                continue

            # Track stretch
            stretch = abs(current_length - rest_length)
            max_stretch = max(max_stretch, stretch)
            total_stretch += stretch

            # How much to correct
            diff = (current_length - rest_length) / current_length
            correction = vec_scale(delta, diff * 0.5 * CONSTRAINT_STIFFNESS)

            # Apply correction based on fixed status
            p1_fixed = fixed_mask[p1_idx]
            p2_fixed = fixed_mask[p2_idx]

            if p1_fixed and p2_fixed:
                continue
            elif p1_fixed:
                particles[p2_idx] = vec_sub(p2, vec_scale(correction, 2.0))
            elif p2_fixed:
                particles[p1_idx] = vec_add(p1, vec_scale(correction, 2.0))
            else:
                particles[p1_idx] = vec_add(p1, correction)
                particles[p2_idx] = vec_sub(p2, correction)

    if logs is not None and constraints:
        avg_stretch = total_stretch / (len(constraints) * CONSTRAINT_ITERATIONS)
        logs.append(("RAGDOLL", f"CONSTRAINTS{log_label} max_stretch={max_stretch:.4f} avg_stretch={avg_stretch:.4f} iters={CONSTRAINT_ITERATIONS}"))


def apply_mesh_collisions(particles, prev_particles, fixed_mask, cached_grid, cached_dynamic_meshes, cached_dynamic_transforms, floor_z, logs):
    """
    Collide particles with static and dynamic meshes.
    """
    collision_count = 0
    total_triangles_checked = 0
    particles_with_nearby_tris = 0

    for i, (pos, fixed) in enumerate(zip(particles, fixed_mask)):
        if fixed:
            continue

        # Get nearby triangles
        triangles = get_nearby_triangles(
            pos, cached_grid, cached_dynamic_meshes, cached_dynamic_transforms
        )

        if triangles:
            particles_with_nearby_tris += 1
            total_triangles_checked += len(triangles)

        if not triangles:
            continue

        # Check collision
        collided, new_pos, normal = collide_particle_with_triangles(pos, triangles, COLLISION_RADIUS)

        if collided:
            particles[i] = new_pos
            collision_count += 1

            # Reflect velocity for bounce
            vel = vec_sub(pos, prev_particles[i])
            vel_normal = vec_scale(normal, vec_dot(vel, normal))
            vel_tangent = vec_sub(vel, vel_normal)
            # Damped reflection
            new_vel = vec_add(vec_scale(vel_tangent, GROUND_FRICTION), vec_scale(vel_normal, -0.3))
            prev_particles[i] = vec_sub(new_pos, new_vel)

    logs.append(("RAGDOLL", f"MESH_COLLISION particles_near_tris={particles_with_nearby_tris} total_tris={total_triangles_checked} hits={collision_count}"))

    return collision_count


# =============================================================================
# BONE ROTATION CALCULATION (moved from main thread!)
# =============================================================================

def calculate_bone_rotations(particles, bone_map, bone_rest_data, armature_matrix, logs):
    """
    Calculate bone quaternions from particle positions.

    This was moved from main thread to worker for performance.

    Args:
        particles: List of (x, y, z) particle positions
        bone_map: Dict mapping bone_name -> (head_idx, tail_idx)
        bone_rest_data: Dict with:
            - rest_dirs: {bone_name: (x, y, z)} rest direction in armature space
            - hierarchy: [bone_names] in hierarchy order (parents first)
            - parents: {bone_name: parent_name or None}
        armature_matrix: Flat 16-float armature world matrix
        logs: List to append log messages

    Returns:
        Dict of {bone_name: (w, x, y, z)} quaternions ready to apply
    """
    if not bone_rest_data:
        return {}

    rest_dirs = bone_rest_data.get("rest_dirs", {})
    hierarchy = bone_rest_data.get("hierarchy", [])
    parents = bone_rest_data.get("parents", {})

    if not hierarchy:
        return {}

    # Get armature rotation quaternion from matrix
    arm_quat = matrix_to_quat(armature_matrix) if armature_matrix else (1, 0, 0, 0)

    # Calculate armature's 3x3 rotation matrix for transforming rest dirs
    # Matrix layout: [0,1,2,3] = col0, [4,5,6,7] = col1, etc (column-major)
    # We need the upper 3x3 for direction transforms
    m = armature_matrix if armature_matrix else [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]

    bone_rotations = {}
    computed_world_rots = {}  # Cache world rotations for parent lookups

    for bone_name in hierarchy:
        if bone_name not in bone_map:
            continue
        if bone_name not in rest_dirs:
            continue

        head_idx, tail_idx = bone_map[bone_name]
        if head_idx >= len(particles) or tail_idx >= len(particles):
            continue

        # Get particle positions (world space)
        head = particles[head_idx]
        tail = particles[tail_idx]

        # Target direction (world space)
        target_dir = vec_sub(tail, head)
        target_dir = vec_normalize(target_dir)

        if vec_length(target_dir) < 0.001:
            bone_rotations[bone_name] = (1.0, 0.0, 0.0, 0.0)
            continue

        # Get rest direction (armature local space)
        rest_dir_local = rest_dirs[bone_name]

        # Transform rest direction to world space using armature matrix
        # rest_dir_world = matrix_3x3 @ rest_dir_local
        rest_dir_world = (
            m[0]*rest_dir_local[0] + m[4]*rest_dir_local[1] + m[8]*rest_dir_local[2],
            m[1]*rest_dir_local[0] + m[5]*rest_dir_local[1] + m[9]*rest_dir_local[2],
            m[2]*rest_dir_local[0] + m[6]*rest_dir_local[1] + m[10]*rest_dir_local[2]
        )
        rest_dir_world = vec_normalize(rest_dir_world)

        # Calculate rotation from rest to target (world space)
        world_rot = quat_from_two_vectors(rest_dir_world, target_dir)
        world_rot = quat_normalize(world_rot)

        # Store world rotation for children to use
        computed_world_rots[bone_name] = world_rot

        # Convert to bone-local rotation (relative to parent)
        parent_name = parents.get(bone_name)

        if parent_name and parent_name in computed_world_rots:
            # Get parent's world rotation
            parent_world_rot = computed_world_rots[parent_name]
            # local_rot = parent^-1 * world_rot * parent (conjugate sandwich)
            parent_inv = quat_conjugate(parent_world_rot)
            local_rot = quat_multiply(quat_multiply(parent_inv, world_rot), parent_world_rot)
        else:
            # Root bone - rotation is relative to armature
            arm_inv = quat_conjugate(arm_quat)
            local_rot = quat_multiply(quat_multiply(arm_inv, world_rot), arm_quat)

        local_rot = quat_normalize(local_rot)
        bone_rotations[bone_name] = local_rot

    logs.append(("RAGDOLL", f"BONE_CALC computed {len(bone_rotations)} bone rotations"))

    return bone_rotations


# =============================================================================
# MAIN HANDLER
# =============================================================================

def handle_ragdoll_update_batch(job_data: dict, cached_grid, cached_dynamic_meshes, cached_dynamic_transforms) -> dict:
    """
    Verlet particle ragdoll physics.

    Each ragdoll has:
    - particles: list of (x, y, z) positions
    - prev_particles: previous frame positions
    - constraints: list of (p1_idx, p2_idx, rest_length)
    - fixed_mask: list of bools (True = particle doesn't move)
    - bone_map: dict mapping bone_name -> (head_idx, tail_idx)
    """
    calc_start = time.perf_counter()
    logs = []

    # Immediately log that we received the job
    logs.append(("RAGDOLL", "WORKER: Job received"))

    dt = job_data.get("dt", 1/30)
    ragdolls = job_data.get("ragdolls", [])

    logs.append(("RAGDOLL", f"WORKER: dt={dt:.4f} ragdolls={len(ragdolls)}"))

    # Log cache status with more detail
    grid_cells = len(cached_grid.get("cells", {})) if cached_grid else 0
    grid_cell_size = cached_grid.get("cell_size", 0) if cached_grid else 0
    dyn_count = len(cached_dynamic_meshes) if cached_dynamic_meshes else 0
    dyn_names = list(cached_dynamic_meshes.keys())[:3] if cached_dynamic_meshes else []
    transform_count = len(cached_dynamic_transforms) if cached_dynamic_transforms else 0

    logs.append(("RAGDOLL", f"WORKER: grid_cells={grid_cells} cell_size={grid_cell_size}"))
    logs.append(("RAGDOLL", f"WORKER: dynamic_meshes={dyn_count} transforms={transform_count} names={dyn_names}"))

    # Log total triangle count in grid
    if cached_grid:
        total_tris = sum(len(tris) for tris in cached_grid.get("cells", {}).values())
        logs.append(("RAGDOLL", f"WORKER: grid_total_tris={total_tris}"))

    updated_ragdolls = []

    for ragdoll in ragdolls:
        ragdoll_id = ragdoll.get("id", 0)
        seq = ragdoll.get("seq", 0)  # Sequence number for ordering
        time_remaining = ragdoll.get("time_remaining", 0.0)
        floor_z = ragdoll.get("floor_z", 0.0)

        # Particle data
        particles = [tuple(p) for p in ragdoll.get("particles", [])]
        prev_particles = [tuple(p) for p in ragdoll.get("prev_particles", [])]
        constraints = ragdoll.get("constraints", [])
        fixed_mask = ragdoll.get("fixed_mask", [])
        bone_map = ragdoll.get("bone_map", {})

        # Bone rotation data (for worker-side bone calculation)
        bone_rest_data = ragdoll.get("bone_rest_data", {})
        armature_matrix = ragdoll.get("armature_matrix", None)

        if not particles:
            logs.append(("RAGDOLL", f"SKIP ragdoll {ragdoll_id}: no particles"))
            continue

        logs.append(("RAGDOLL", f"UPDATE ragdoll {ragdoll_id}: {len(particles)} particles, {len(constraints)} constraints, time_left={time_remaining:.2f}s"))

        # Ensure prev_particles exists
        if len(prev_particles) != len(particles):
            prev_particles = list(particles)
            logs.append(("RAGDOLL", f"  INIT prev_particles (first frame)"))

        # Ensure fixed_mask exists
        if len(fixed_mask) != len(particles):
            fixed_mask = [False] * len(particles)

        # Convert to mutable lists
        particles = list(particles)
        prev_particles = list(prev_particles)

        # Log initial state BEFORE physics
        zs_before = [p[2] for p in particles]
        logs.append(("RAGDOLL", f"  BEFORE_PHYSICS z_range=[{min(zs_before):.3f},{max(zs_before):.3f}]"))

        # === PHYSICS STEPS ===

        # 1. Verlet integration (gravity + velocity)
        prev_particles = verlet_integrate(particles, prev_particles, dt, fixed_mask, floor_z, logs)

        # 2. Satisfy distance constraints
        satisfy_constraints(particles, constraints, fixed_mask, logs, "_PRE")

        # 3. Mesh collisions
        apply_mesh_collisions(
            particles, prev_particles, fixed_mask,
            cached_grid, cached_dynamic_meshes, cached_dynamic_transforms,
            floor_z, logs
        )

        # 4. Re-satisfy constraints after collision
        satisfy_constraints(particles, constraints, fixed_mask, logs, "_POST")

        # 5. Final floor clamp
        floor_clamp_count = 0
        for i, pos in enumerate(particles):
            if not fixed_mask[i] and pos[2] < floor_z + FLOOR_OFFSET:
                particles[i] = (pos[0], pos[1], floor_z + FLOOR_OFFSET)
                floor_clamp_count += 1

        if floor_clamp_count > 0:
            logs.append(("RAGDOLL", f"FLOOR_CLAMP {floor_clamp_count} particles clamped to floor_z={floor_z:.3f}"))

        # Log particle Z range and sample positions
        zs = [p[2] for p in particles]
        logs.append(("RAGDOLL", f"FINAL_Z range=[{min(zs):.3f},{max(zs):.3f}] floor={floor_z:.3f}"))

        # Log some particle positions for debugging
        if len(particles) >= 3:
            p0 = particles[0]
            p1 = particles[min(1, len(particles)-1)]
            p2 = particles[min(2, len(particles)-1)]
            logs.append(("RAGDOLL", f"  P0=({p0[0]:.2f},{p0[1]:.2f},{p0[2]:.2f}) P1=({p1[0]:.2f},{p1[1]:.2f},{p1[2]:.2f}) P2=({p2[0]:.2f},{p2[1]:.2f},{p2[2]:.2f})"))

        # === BONE ROTATIONS (moved from main thread!) ===
        # Calculate bone quaternions from final particle positions
        bone_rotations = {}
        if bone_rest_data and armature_matrix:
            bone_rotations = calculate_bone_rotations(
                particles, bone_map, bone_rest_data, armature_matrix, logs
            )

        # Check if finished
        new_time = time_remaining - dt
        finished = new_time <= 0

        if finished:
            logs.append(("RAGDOLL", f"FINISHED ragdoll {ragdoll_id}"))

        updated_ragdolls.append({
            "id": ragdoll_id,
            "seq": seq,  # Pass through sequence number
            "particles": particles,
            "prev_particles": prev_particles,
            "bone_rotations": bone_rotations,  # Pre-computed quaternions!
            "time_remaining": max(0, new_time),
            "finished": finished,
        })

    calc_time = (time.perf_counter() - calc_start) * 1_000_000
    total_particles = sum(len(r.get("particles", [])) for r in ragdolls)
    logs.append(("RAGDOLL", f"WORKER: {len(ragdolls)} ragdolls, {total_particles} particles, {calc_time:.0f}us"))

    return {
        "success": True,
        "updated_ragdolls": updated_ragdolls,
        "calc_time_us": calc_time,
        "logs": logs,
    }
