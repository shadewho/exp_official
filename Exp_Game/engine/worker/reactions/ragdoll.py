# Exp_Game/engine/worker/reactions/ragdoll.py
"""
Articulated-Rod Ragdoll Physics - Rig Agnostic

Real physics-based bone rotation that works on ANY armature.
- Gravity torque projected into bone local space (strong influence)
- Parent-alignment torque prevents curling into a ball
- Per-bone stiffness/damping/limits based on role
- Angular velocity clamping prevents runaway

NO HARDCODED BONE NAMES OR DIRECTIONS.
"""

import time
import math
import random


# =============================================================================
# PHYSICS CONSTANTS (ALL TUNABLES HERE)
# =============================================================================

WORLD_GRAVITY = (0.0, 0.0, -9.8)  # World-space gravity vector

# Gravity torque multipliers (higher = more gravity influence)
GRAVITY_TORQUE_PRIMARY = 0.6    # X/Z axes - main bend
GRAVITY_TORQUE_TWIST = 0.05     # Y axis - twist (keep tiny)

# Role-based parameters
# STIFFNESS: How strongly bone returns to rest (LOW = floppy)
# DAMPING: How quickly motion dies out
# INERTIA: Resistance to rotation
ROLE_PARAMS = {
    # role: (stiffness, damping, inertia)
    "core":  (0.8, 0.90, 1.2),   # Spine, hips - soft so they can slump
    "limb":  (0.3, 0.88, 1.0),   # Arms, legs - floppy
    "head":  (0.4, 0.90, 0.7),   # Head, neck - slightly damped
    "hand":  (0.15, 0.85, 0.5),  # Hands, feet, fingers - very loose
}

# Twist axis gets reduced stiffness (0.25x)
TWIST_STIFFNESS_MULT = 0.25

# Default joint limits (radians) if not provided
DEFAULT_LIMIT = 1.4           # ~80 degrees
DEFAULT_SECONDARY_LIMIT = 0.7  # ~40 degrees

# Angular velocity clamp (prevents runaway curl)
MAX_ANGULAR_VELOCITY = 6.0  # rad/s

# Parent alignment (prevents folding into ball)
PARENT_ALIGN_STRENGTH = 0.4  # How strongly child aligns with parent direction

# Ground contact
GROUND_PUSH_STRENGTH = 5.0
GROUND_DAMPING = 0.6

# Initial impulse variance
INITIAL_IMPULSE_RANGE = 1.0


# =============================================================================
# MATH HELPERS
# =============================================================================

def mat3_from_flat(m):
    """Convert flat 9-element list to 3x3 matrix (row-major)."""
    return [
        [m[0], m[1], m[2]],
        [m[3], m[4], m[5]],
        [m[6], m[7], m[8]],
    ]


def mat3_transpose(m):
    """Transpose 3x3 matrix."""
    return [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]


def mat3_mul_vec(m, v):
    """Multiply 3x3 matrix by 3-vector."""
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


# =============================================================================
# MAIN HANDLER
# =============================================================================

def handle_ragdoll_update_batch(job_data: dict, cached_grid, cached_dynamic_meshes, cached_dynamic_transforms) -> dict:
    """
    Articulated-rod ragdoll physics.

    For each bone:
    1. Project world gravity into bone local space
    2. Apply gravity torque on PRIMARY bend axes (X/Z), tiny on twist (Y)
    3. Add parent-alignment torque to prevent curling
    4. Add weak spring-to-rest
    5. Integrate with damping and velocity clamping
    6. Clamp to joint limits
    """
    calc_start = time.perf_counter()
    logs = []

    dt = job_data.get("dt", 1/30)
    ragdolls = job_data.get("ragdolls", [])

    updated_ragdolls = []

    for ragdoll in ragdolls:
        ragdoll_id = ragdoll.get("id", 0)
        time_remaining = ragdoll.get("time_remaining", 0.0)
        bone_data = ragdoll.get("bone_data", {})
        bone_physics = ragdoll.get("bone_physics", {})
        armature_matrix = ragdoll.get("armature_matrix", None)
        ground_z = ragdoll.get("ground_z", 0.0)
        initialized = ragdoll.get("initialized", False)

        new_bone_physics = {}

        # Get armature rotation matrix (3x3 upper-left of 4x4)
        if armature_matrix and len(armature_matrix) >= 12:
            arm_rot = mat3_from_flat([
                armature_matrix[0], armature_matrix[1], armature_matrix[2],
                armature_matrix[4], armature_matrix[5], armature_matrix[6],
                armature_matrix[8], armature_matrix[9], armature_matrix[10],
            ])
            arm_rot_inv = mat3_transpose(arm_rot)
        else:
            arm_rot_inv = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # Transform world gravity into armature space
        armature_gravity = mat3_mul_vec(arm_rot_inv, WORLD_GRAVITY)

        for bone_name, bdata in bone_data.items():
            # Get static bone data
            rest_matrix = bdata.get("rest_matrix")
            role = bdata.get("role", "limb")
            limits = bdata.get("limits", {})
            is_ground_bone = bdata.get("is_ground_bone", False)
            bone_tip_local = bdata.get("bone_tip_local", (0, 0, 0))

            # Get dynamic state
            bp = bone_physics.get(bone_name, {"rot": (0.0, 0.0, 0.0), "ang_vel": (0.0, 0.0, 0.0)})
            rot = list(bp.get("rot", (0.0, 0.0, 0.0)))
            ang_vel = list(bp.get("ang_vel", (0.0, 0.0, 0.0)))

            # Get role-based parameters
            stiffness, damping, inertia = ROLE_PARAMS.get(role, ROLE_PARAMS["limb"])

            # Get joint limits
            lim_x = limits.get("x", [-DEFAULT_LIMIT, DEFAULT_LIMIT])
            lim_y = limits.get("y", [-DEFAULT_SECONDARY_LIMIT, DEFAULT_SECONDARY_LIMIT])
            lim_z = limits.get("z", [-DEFAULT_LIMIT, DEFAULT_LIMIT])

            # Initialize with random impulse on first frame
            if not initialized:
                ang_vel[0] += random.uniform(-INITIAL_IMPULSE_RANGE, INITIAL_IMPULSE_RANGE)
                ang_vel[1] += random.uniform(-INITIAL_IMPULSE_RANGE * 0.3, INITIAL_IMPULSE_RANGE * 0.3)
                ang_vel[2] += random.uniform(-INITIAL_IMPULSE_RANGE, INITIAL_IMPULSE_RANGE)

            # Transform gravity into bone local space
            if rest_matrix and len(rest_matrix) >= 9:
                bone_rest = mat3_from_flat(rest_matrix)
                bone_rest_inv = mat3_transpose(bone_rest)
                local_gravity = mat3_mul_vec(bone_rest_inv, armature_gravity)
            else:
                local_gravity = armature_gravity

            # Compute gravity torque
            # Gravity in local Z creates torque around X, gravity in X creates torque around Z
            # Scale by cos(angle) for pendulum behavior (max at horizontal, zero at vertical)
            grav_mag = math.sqrt(local_gravity[0]**2 + local_gravity[1]**2 + local_gravity[2]**2)

            if grav_mag > 0.01:
                # Torque from gravity - PRIMARY axes get strong influence
                # torque_x from gravity_z (forward/back tilt)
                torque_x = -local_gravity[2] * GRAVITY_TORQUE_PRIMARY * math.cos(rot[0])
                # torque_z from gravity_x (side tilt)
                torque_z = local_gravity[0] * GRAVITY_TORQUE_PRIMARY * math.cos(rot[2])
                # torque_y (twist) - keep very small to prevent winding up
                torque_y = local_gravity[1] * GRAVITY_TORQUE_TWIST * math.cos(rot[1])
            else:
                torque_x, torque_y, torque_z = 0.0, 0.0, 0.0

            # Parent alignment torque - prevents curling into a ball
            # Pushes bone back toward its rest direction relative to parent
            # This is approximated by a torque proportional to current rotation
            # but weaker than spring-to-rest
            align_torque_x = -PARENT_ALIGN_STRENGTH * math.sin(rot[0])
            align_torque_z = -PARENT_ALIGN_STRENGTH * math.sin(rot[2])
            torque_x += align_torque_x
            torque_z += align_torque_z

            # Spring-to-rest torque (weak - just prevents extreme poses)
            torque_x -= stiffness * rot[0]
            torque_y -= stiffness * TWIST_STIFFNESS_MULT * rot[1]  # Reduced twist stiffness
            torque_z -= stiffness * rot[2]

            # Integrate angular velocity
            ang_vel[0] += (torque_x / inertia) * dt
            ang_vel[1] += (torque_y / inertia) * dt
            ang_vel[2] += (torque_z / inertia) * dt

            # Apply damping
            ang_vel[0] *= damping
            ang_vel[1] *= damping
            ang_vel[2] *= damping

            # Clamp angular velocity to prevent runaway
            ang_vel[0] = clamp(ang_vel[0], -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
            ang_vel[1] = clamp(ang_vel[1], -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
            ang_vel[2] = clamp(ang_vel[2], -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)

            # Integrate rotation
            rot[0] += ang_vel[0] * dt
            rot[1] += ang_vel[1] * dt
            rot[2] += ang_vel[2] * dt

            # Clamp to joint limits and zero velocity on clamped axes
            if rot[0] < lim_x[0]:
                rot[0] = lim_x[0]
                ang_vel[0] = max(0, ang_vel[0])
            elif rot[0] > lim_x[1]:
                rot[0] = lim_x[1]
                ang_vel[0] = min(0, ang_vel[0])

            if rot[1] < lim_y[0]:
                rot[1] = lim_y[0]
                ang_vel[1] = max(0, ang_vel[1])
            elif rot[1] > lim_y[1]:
                rot[1] = lim_y[1]
                ang_vel[1] = min(0, ang_vel[1])

            if rot[2] < lim_z[0]:
                rot[2] = lim_z[0]
                ang_vel[2] = max(0, ang_vel[2])
            elif rot[2] > lim_z[1]:
                rot[2] = lim_z[1]
                ang_vel[2] = min(0, ang_vel[2])

            # Ground contact for ground bones
            if is_ground_bone and armature_matrix:
                arm_z = armature_matrix[14] if len(armature_matrix) > 14 else 0
                tip_z = arm_z + bone_tip_local[2]

                if tip_z < ground_z:
                    penetration = ground_z - tip_z
                    # Soft spring toward ground + damp all axes
                    push = min(GROUND_PUSH_STRENGTH * penetration * dt, 0.5)
                    rot[0] -= push
                    ang_vel[0] *= GROUND_DAMPING
                    ang_vel[1] *= GROUND_DAMPING
                    ang_vel[2] *= GROUND_DAMPING

            new_bone_physics[bone_name] = {
                "rot": tuple(rot),
                "ang_vel": tuple(ang_vel),
            }

        new_time = time_remaining - dt
        finished = new_time <= 0

        if finished:
            logs.append(("RAGDOLL", f"WORKER: Ragdoll {ragdoll_id} FINISHED"))

        updated_ragdolls.append({
            "id": ragdoll_id,
            "bone_physics": new_bone_physics,
            "time_remaining": max(0, new_time),
            "finished": finished,
            "initialized": True,
        })

    calc_time = (time.perf_counter() - calc_start) * 1_000_000
    if ragdolls:
        logs.append(("RAGDOLL", f"WORKER: {len(ragdolls)} ragdolls, {len(bone_data)} bones, {calc_time:.0f}us"))

    return {
        "success": True,
        "updated_ragdolls": updated_ragdolls,
        "calc_time_us": calc_time,
        "logs": logs,
    }
