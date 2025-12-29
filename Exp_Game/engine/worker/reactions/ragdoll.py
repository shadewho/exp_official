# Exp_Game/engine/worker/reactions/ragdoll.py
"""
SIMPLE Ragdoll Physics - Worker Side (NO bpy)

Each bone is a point mass that:
1. Falls with gravity
2. Has velocity damping
3. Stays connected to parent (distance constraint)
4. Bounces off floor

Output: New world positions for each bone head.
Main thread converts to pose transforms.
"""

import time
import math


def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_scale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def _vec_length(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def _vec_normalize(v):
    length = _vec_length(v)
    if length < 1e-9:
        return (0.0, 0.0, 0.0)
    return (v[0]/length, v[1]/length, v[2]/length)


def handle_ragdoll_update_batch(job_data: dict, cached_grid, cached_dynamic_meshes, cached_dynamic_transforms) -> dict:
    """
    SIMPLE ragdoll - compute rotation offsets for loose body effect.

    Instead of world positions, output rotation deltas for each bone.
    Root gets position offset, other bones get rotation sway.
    """
    calc_start = time.perf_counter()
    logs = []

    dt = job_data.get("dt", 1/30)
    ragdolls = job_data.get("ragdolls", [])

    logs.append(("RAGDOLL", f"WORKER: {len(ragdolls)} ragdolls"))

    updated_ragdolls = []

    for ragdoll in ragdolls:
        ragdoll_id = ragdoll.get("id", 0)
        time_remaining = ragdoll.get("time_remaining", 0.0)
        bone_order = ragdoll.get("bone_order", [])
        bone_states = ragdoll.get("bone_states", {})
        gravity = ragdoll.get("gravity", (0.0, 0.0, -20.0))

        impulse = ragdoll.get("impulse")
        impulse_bone = ragdoll.get("impulse_bone")

        logs.append(("RAGDOLL", f"  Ragdoll {ragdoll_id}: {len(bone_order)} bones, t={time_remaining:.1f}s"))

        # Apply impulse to root velocity on first frame
        if impulse and impulse_bone and impulse_bone in bone_states:
            state = bone_states[impulse_bone]
            old_vel = state.get("vel", (0, 0, 0))
            state["vel"] = _vec_add(old_vel, impulse)
            logs.append(("RAGDOLL", f"    Impulse applied to {impulse_bone}"))

        new_bone_states = {}
        bone_rotations = {}  # bone_name -> (rx, ry, rz) rotation offset in radians

        for i, bone_name in enumerate(bone_order):
            if bone_name not in bone_states:
                continue

            state = bone_states[bone_name]
            rot = tuple(state.get("rot", (0.0, 0.0, 0.0)))  # rotation offset
            rot_vel = tuple(state.get("rot_vel", (0.0, 0.0, 0.0)))  # angular velocity

            is_root = (i == 0)

            if is_root:
                # Root: apply gravity to position
                pos = tuple(state.get("pos", (0, 0, 0)))
                vel = tuple(state.get("vel", (0, 0, 0)))

                # Gentle gravity (30% of scene gravity)
                grav_scale = 0.3
                new_vel = (
                    vel[0] + gravity[0] * grav_scale * dt,
                    vel[1] + gravity[1] * grav_scale * dt,
                    vel[2] + gravity[2] * grav_scale * dt,
                )
                new_vel = _vec_scale(new_vel, 0.95)  # damping

                new_pos = (
                    pos[0] + new_vel[0] * dt,
                    pos[1] + new_vel[1] * dt,
                    pos[2] + new_vel[2] * dt,
                )

                # Floor constraint at Z=0
                if new_pos[2] < 0:
                    new_pos = (new_pos[0], new_pos[1], 0.0)
                    if new_vel[2] < 0:
                        new_vel = (new_vel[0] * 0.5, new_vel[1] * 0.5, -new_vel[2] * 0.2)

                new_bone_states[bone_name] = {
                    "pos": new_pos,
                    "vel": new_vel,
                    "rot": (0, 0, 0),
                    "rot_vel": (0, 0, 0),
                }
                bone_rotations[bone_name] = new_pos  # For root, store position offset

                if i < 2:
                    logs.append(("RAGDOLL", f"    {bone_name}: Z {pos[2]:.2f} -> {new_pos[2]:.2f}"))
            else:
                # Non-root: wobble rotation
                # Apply angular "gravity" - bones tend to droop
                droop = -0.5  # radians/sec^2 tendency to rotate forward/down
                new_rot_vel = (
                    rot_vel[0] + droop * dt,
                    rot_vel[1],
                    rot_vel[2],
                )
                # Strong damping on rotation
                new_rot_vel = _vec_scale(new_rot_vel, 0.92)

                # Update rotation
                new_rot = (
                    rot[0] + new_rot_vel[0] * dt,
                    rot[1] + new_rot_vel[1] * dt,
                    rot[2] + new_rot_vel[2] * dt,
                )

                # Clamp rotation to reasonable range (-45 to +45 degrees)
                max_rot = 0.8  # ~45 degrees
                new_rot = (
                    max(-max_rot, min(max_rot, new_rot[0])),
                    max(-max_rot, min(max_rot, new_rot[1])),
                    max(-max_rot, min(max_rot, new_rot[2])),
                )

                new_bone_states[bone_name] = {
                    "pos": (0, 0, 0),
                    "vel": (0, 0, 0),
                    "rot": new_rot,
                    "rot_vel": new_rot_vel,
                }
                bone_rotations[bone_name] = new_rot

                if i < 3:
                    logs.append(("RAGDOLL", f"    {bone_name}: rot {rot[0]:.2f} -> {new_rot[0]:.2f}"))

        new_time = time_remaining - dt
        finished = new_time <= 0

        logs.append(("RAGDOLL", f"    {len(bone_rotations)} bones"))

        updated_ragdolls.append({
            "id": ragdoll_id,
            "bone_rotations": bone_rotations,  # root=position, others=rotation
            "bone_states": new_bone_states,
            "time_remaining": max(0, new_time),
            "finished": finished,
        })

    calc_time = (time.perf_counter() - calc_start) * 1_000_000
    logs.append(("RAGDOLL", f"WORKER: Done in {calc_time:.0f}μs"))

    return {
        "success": True,
        "updated_ragdolls": updated_ragdolls,
        "calc_time_us": calc_time,
        "logs": logs,
    }
