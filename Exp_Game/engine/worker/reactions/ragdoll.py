# Exp_Game/engine/worker/reactions/ragdoll.py
"""
Ragdoll Physics - Rig Agnostic

Simple pendulum physics - bones fall toward gravity.
No hardcoded bone names, no special cases.

Works on ANY armature.
"""

import time
import math

# =============================================================================
# PHYSICS CONSTANTS
# =============================================================================

WORLD_GRAVITY = (0.0, 0.0, -9.8)
GRAVITY_STRENGTH = 8.0

# Per-role physics: (damping, limit_radians)
# damping: velocity decay per frame (lower = faster settle)
# limit: max rotation from rest pose
ROLE_PHYSICS = {
    "core": (0.90, 0.8),    # Spine/chest/hips - fairly stable
    "limb": (0.92, 2.0),    # Arms/legs - loose
    "head": (0.90, 1.0),    # Head/neck
    "hand": (0.94, 2.2),    # Hands/feet/fingers - very loose
}


# =============================================================================
# MATH HELPERS
# =============================================================================

def mat3_from_flat(m):
    return [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]]

def mat3_transpose(m):
    return [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]

def mat3_mul_vec(m, v):
    return (
        m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
        m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
        m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2],
    )

def euler_to_mat3(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return [
        [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz],
        [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
        [-sy, sx*cy, cx*cy]
    ]

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# =============================================================================
# MAIN HANDLER
# =============================================================================

def handle_ragdoll_update_batch(job_data: dict, cached_grid, cached_dynamic_meshes, cached_dynamic_transforms) -> dict:
    """
    Simple pendulum ragdoll - every bone falls toward gravity.
    No special cases, no hardcoded anything.
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
        initialized = ragdoll.get("initialized", False)

        new_bone_physics = {}

        # Get armature rotation to transform gravity
        if armature_matrix and len(armature_matrix) >= 12:
            arm_rot = mat3_from_flat([
                armature_matrix[0], armature_matrix[1], armature_matrix[2],
                armature_matrix[4], armature_matrix[5], armature_matrix[6],
                armature_matrix[8], armature_matrix[9], armature_matrix[10],
            ])
            arm_rot_inv = mat3_transpose(arm_rot)
        else:
            arm_rot_inv = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # Gravity in armature space
        armature_gravity = mat3_mul_vec(arm_rot_inv, WORLD_GRAVITY)

        logs.append(("RAGDOLL", f"GRAV g=({armature_gravity[0]:.1f},{armature_gravity[1]:.1f},{armature_gravity[2]:.1f})"))

        bone_count = 0
        for bone_name, bdata in bone_data.items():
            rest_matrix = bdata.get("rest_matrix")
            role = bdata.get("role", "limb")

            # Get per-role physics
            damping, limit = ROLE_PHYSICS.get(role, ROLE_PHYSICS["limb"])

            # Get current physics state
            bp = bone_physics.get(bone_name, {"rot": (0.0, 0.0, 0.0), "ang_vel": (0.0, 0.0, 0.0)})
            rot = list(bp.get("rot", (0.0, 0.0, 0.0)))
            ang_vel = list(bp.get("ang_vel", (0.0, 0.0, 0.0)))

            # Transform gravity into bone's REST local space
            if rest_matrix and len(rest_matrix) >= 9:
                bone_rest = mat3_from_flat(rest_matrix)
                bone_rest_inv = mat3_transpose(bone_rest)
                rest_local_grav = mat3_mul_vec(bone_rest_inv, armature_gravity)
            else:
                rest_local_grav = armature_gravity

            # Transform into CURRENT local space (after physics rotation)
            phys_rot = euler_to_mat3(rot[0], rot[1], rot[2])
            phys_rot_inv = mat3_transpose(phys_rot)
            local_grav = mat3_mul_vec(phys_rot_inv, rest_local_grav)

            # Normalize
            grav_len = math.sqrt(local_grav[0]**2 + local_grav[1]**2 + local_grav[2]**2)
            if grav_len > 0.01:
                gx, gy, gz = local_grav[0]/grav_len, local_grav[1]/grav_len, local_grav[2]/grav_len
            else:
                gx, gy, gz = 0, 0, -1

            # Pendulum torque: bone_Y × gravity = (0,1,0) × (gx,gy,gz) = (gz, 0, -gx)
            torque_x = gz * GRAVITY_STRENGTH
            torque_z = -gx * GRAVITY_STRENGTH

            # Integrate velocity
            ang_vel[0] += torque_x * dt
            ang_vel[2] += torque_z * dt

            # Damping
            ang_vel[0] *= damping
            ang_vel[1] *= damping
            ang_vel[2] *= damping

            # Clamp velocity
            max_vel = 10.0
            ang_vel[0] = clamp(ang_vel[0], -max_vel, max_vel)
            ang_vel[1] = clamp(ang_vel[1], -max_vel, max_vel)
            ang_vel[2] = clamp(ang_vel[2], -max_vel, max_vel)

            # Integrate rotation
            rot[0] += ang_vel[0] * dt
            rot[1] += ang_vel[1] * dt
            rot[2] += ang_vel[2] * dt

            # Clamp to limits
            rot[0] = clamp(rot[0], -limit, limit)
            rot[1] = clamp(rot[1], -limit * 0.4, limit * 0.4)  # Twist limited
            rot[2] = clamp(rot[2], -limit, limit)

            new_bone_physics[bone_name] = {
                "rot": tuple(rot),
                "ang_vel": tuple(ang_vel),
            }

            if bone_count < 4:
                logs.append(("RAGDOLL", f"BONE{bone_count}:{bone_name} role={role} T=({torque_x:.1f},{torque_z:.1f}) R=({rot[0]:.2f},{rot[1]:.2f},{rot[2]:.2f})"))
                bone_count += 1

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
        total_bones = sum(len(r.get("bone_data", {})) for r in ragdolls)
        logs.append(("RAGDOLL", f"WORKER: {len(ragdolls)} ragdolls, {total_bones} bones, {calc_time:.0f}us"))

    return {
        "success": True,
        "updated_ragdolls": updated_ragdolls,
        "calc_time_us": calc_time,
        "logs": logs,
    }
