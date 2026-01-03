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
GRAVITY_SCALE = 1.0           # Multiplier for gravity strength

CONSTRAINT_ITERATIONS = 8     # More = stiffer bones, slower
CONSTRAINT_STIFFNESS = 0.9    # 0-1, how rigidly bones maintain length

DAMPING = 0.98                # Velocity retention (1.0 = no damping)
GROUND_FRICTION = 0.7         # Friction when touching ground

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
    if cached_grid:
        cell_size = cached_grid.get("cell_size", 1.0)
        cells = cached_grid.get("cells", {})

        # Check nearby cells
        cx = int(pos[0] / cell_size)
        cy = int(pos[1] / cell_size)
        cz = int(pos[2] / cell_size)

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    cell_key = (cx + dx, cy + dy, cz + dz)
                    cell_tris = cells.get(cell_key, [])
                    triangles.extend(cell_tris)

    # Dynamic mesh triangles
    if cached_dynamic_meshes and cached_dynamic_transforms:
        for mesh_name, mesh_data in cached_dynamic_meshes.items():
            transform = cached_dynamic_transforms.get(mesh_name)
            if not transform or not mesh_data:
                continue

            local_tris = mesh_data.get("triangles", [])

            # Transform triangles to world space
            for tri in local_tris:
                if len(tri) < 9:
                    continue

                # Apply transform (simplified - assumes transform is 4x4 matrix flat)
                if len(transform) >= 16:
                    world_tri = transform_triangle(tri, transform)
                    triangles.append(world_tri)
                else:
                    triangles.append(tri)

    return triangles


def transform_triangle(tri, matrix):
    """Transform triangle vertices by 4x4 matrix."""
    def transform_point(p, m):
        x = m[0]*p[0] + m[4]*p[1] + m[8]*p[2] + m[12]
        y = m[1]*p[0] + m[5]*p[1] + m[9]*p[2] + m[13]
        z = m[2]*p[0] + m[6]*p[1] + m[10]*p[2] + m[14]
        return (x, y, z)

    v0 = transform_point((tri[0], tri[1], tri[2]), matrix)
    v1 = transform_point((tri[3], tri[4], tri[5]), matrix)
    v2 = transform_point((tri[6], tri[7], tri[8]), matrix)

    return (v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], v2[0], v2[1], v2[2])


# =============================================================================
# VERLET INTEGRATION
# =============================================================================

def verlet_integrate(particles, prev_particles, dt, fixed_mask, floor_z):
    """
    Verlet integration step.
    Updates particles in place, returns new prev_particles.
    """
    new_prev = []
    gravity_step = vec_scale(GRAVITY, GRAVITY_SCALE * dt * dt)

    for i, (pos, prev, fixed) in enumerate(zip(particles, prev_particles, fixed_mask)):
        new_prev.append(pos)

        if fixed:
            continue

        # Velocity = current - previous (Verlet style)
        vel = vec_sub(pos, prev)

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

        particles[i] = new_pos

    return new_prev


def satisfy_constraints(particles, constraints, fixed_mask):
    """
    Satisfy distance constraints between particles.
    Iterates multiple times for stability.
    """
    for _ in range(CONSTRAINT_ITERATIONS):
        for p1_idx, p2_idx, rest_length in constraints:
            p1 = particles[p1_idx]
            p2 = particles[p2_idx]

            delta = vec_sub(p2, p1)
            current_length = vec_length(delta)

            if current_length < 0.0001:
                continue

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


def apply_mesh_collisions(particles, prev_particles, fixed_mask, cached_grid, cached_dynamic_meshes, cached_dynamic_transforms, floor_z, logs):
    """
    Collide particles with static and dynamic meshes.
    """
    collision_count = 0

    for i, (pos, fixed) in enumerate(zip(particles, fixed_mask)):
        if fixed:
            continue

        # Get nearby triangles
        triangles = get_nearby_triangles(
            pos, cached_grid, cached_dynamic_meshes, cached_dynamic_transforms
        )

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

    if collision_count > 0:
        logs.append(("RAGDOLL", f"COLLISION: {collision_count} particles hit mesh"))

    return collision_count


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

    # Log cache status
    grid_cells = len(cached_grid.get("cells", {})) if cached_grid else 0
    dyn_count = len(cached_dynamic_meshes) if cached_dynamic_meshes else 0
    logs.append(("RAGDOLL", f"WORKER: grid_cells={grid_cells} dynamic_meshes={dyn_count}"))

    updated_ragdolls = []

    for ragdoll in ragdolls:
        ragdoll_id = ragdoll.get("id", 0)
        time_remaining = ragdoll.get("time_remaining", 0.0)
        floor_z = ragdoll.get("floor_z", 0.0)

        # Particle data
        particles = [tuple(p) for p in ragdoll.get("particles", [])]
        prev_particles = [tuple(p) for p in ragdoll.get("prev_particles", [])]
        constraints = ragdoll.get("constraints", [])
        fixed_mask = ragdoll.get("fixed_mask", [])
        bone_map = ragdoll.get("bone_map", {})

        if not particles:
            logs.append(("RAGDOLL", f"SKIP ragdoll {ragdoll_id}: no particles"))
            continue

        logs.append(("RAGDOLL", f"UPDATE ragdoll {ragdoll_id}: {len(particles)} particles, {len(constraints)} constraints"))

        # Ensure prev_particles exists
        if len(prev_particles) != len(particles):
            prev_particles = list(particles)

        # Ensure fixed_mask exists
        if len(fixed_mask) != len(particles):
            fixed_mask = [False] * len(particles)

        # Convert to mutable lists
        particles = list(particles)
        prev_particles = list(prev_particles)

        # === PHYSICS STEPS ===

        # 1. Verlet integration (gravity + velocity)
        prev_particles = verlet_integrate(particles, prev_particles, dt, fixed_mask, floor_z)

        # 2. Satisfy distance constraints
        satisfy_constraints(particles, constraints, fixed_mask)

        # 3. Mesh collisions
        apply_mesh_collisions(
            particles, prev_particles, fixed_mask,
            cached_grid, cached_dynamic_meshes, cached_dynamic_transforms,
            floor_z, logs
        )

        # 4. Re-satisfy constraints after collision
        satisfy_constraints(particles, constraints, fixed_mask)

        # 5. Final floor clamp
        for i, pos in enumerate(particles):
            if not fixed_mask[i] and pos[2] < floor_z + FLOOR_OFFSET:
                particles[i] = (pos[0], pos[1], floor_z + FLOOR_OFFSET)

        # Log some particle positions for debugging
        if len(particles) >= 3:
            p0 = particles[0]
            p1 = particles[min(1, len(particles)-1)]
            p2 = particles[min(2, len(particles)-1)]
            logs.append(("RAGDOLL", f"  P0=({p0[0]:.2f},{p0[1]:.2f},{p0[2]:.2f}) P1=({p1[0]:.2f},{p1[1]:.2f},{p1[2]:.2f}) P2=({p2[0]:.2f},{p2[1]:.2f},{p2[2]:.2f})"))

        # Check if finished
        new_time = time_remaining - dt
        finished = new_time <= 0

        if finished:
            logs.append(("RAGDOLL", f"FINISHED ragdoll {ragdoll_id}"))

        updated_ragdolls.append({
            "id": ragdoll_id,
            "particles": particles,
            "prev_particles": prev_particles,
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
