# Exp_Game/engine/worker/reactions/transforms.py
"""
Transform interpolation - runs in worker process (NO bpy).

Handles:
- Location lerp
- Rotation interpolation (euler lerp, quaternion slerp, local_delta)
- Scale lerp

All computation offloaded from main thread. Main thread only applies results.
"""

import time

from ..math import (
    euler_to_quaternion,
    quaternion_to_euler,
    quaternion_multiply,
    slerp_quaternion,
)


def handle_transform_batch(job_data: dict) -> dict:
    """
    Handle TRANSFORM_BATCH job - compute transform interpolations.

    Input job_data:
        {
            "transforms": [
                {
                    "obj_id": int,              # id(obj) for result matching
                    "t": float,                 # interpolation factor [0, 1]
                    "start_loc": (x, y, z),
                    "end_loc": (x, y, z),
                    "start_rot_q": (w, x, y, z),  # quaternion
                    "end_rot_q": (w, x, y, z),    # quaternion
                    "start_scl": (x, y, z),
                    "end_scl": (x, y, z),
                    "rot_mode": "euler" | "quat" | "local_delta",
                    "start_rot_e": (x, y, z),    # euler (for euler mode)
                    "end_rot_e": (x, y, z),      # euler (for euler mode)
                    "delta_euler": (x, y, z),    # for local_delta mode
                },
                ...
            ]
        }

    Returns:
        {
            "success": bool,
            "results": [
                {
                    "obj_id": int,
                    "loc": (x, y, z),
                    "rot_euler": (x, y, z),
                    "scl": (x, y, z),
                    "finished": bool,
                },
                ...
            ],
            "count": int,
            "calc_time_us": float,
            "logs": [(category, message), ...],
        }
    """
    import math
    calc_start = time.perf_counter()
    logs = []

    transforms = job_data.get("transforms", [])

    # Count rotation modes for summary
    mode_counts = {"euler": 0, "quat": 0, "local_delta": 0}
    results = []

    for tf in transforms:
        obj_id = tf["obj_id"]
        t = tf["t"]
        finished = t >= 1.0

        # Clamp t
        t = max(0.0, min(1.0, t))

        # --- Location lerp ---
        start_loc = tf["start_loc"]
        end_loc = tf["end_loc"]
        loc = (
            start_loc[0] + (end_loc[0] - start_loc[0]) * t,
            start_loc[1] + (end_loc[1] - start_loc[1]) * t,
            start_loc[2] + (end_loc[2] - start_loc[2]) * t,
        )

        # --- Rotation ---
        rot_mode = tf.get("rot_mode", "quat")
        mode_counts[rot_mode] = mode_counts.get(rot_mode, 0) + 1

        if rot_mode == "local_delta":
            # q(t) = q_start @ quat(euler(t * delta))
            # Supports multi-turn rotations (e.g., 720 degrees)
            delta = tf.get("delta_euler", (0.0, 0.0, 0.0))
            delta_t = (delta[0] * t, delta[1] * t, delta[2] * t)
            q_delta = euler_to_quaternion(delta_t)
            q_start = tf["start_rot_q"]
            q_result = quaternion_multiply(q_start, q_delta)
            rot_euler = quaternion_to_euler(q_result)

        elif rot_mode == "quat":
            # Spherical interpolation (smooth, shortest path)
            q_start = tf["start_rot_q"]
            q_end = tf["end_rot_q"]
            q_result = slerp_quaternion(q_start, q_end, t)
            rot_euler = quaternion_to_euler(q_result)

        else:  # "euler"
            # Per-channel Euler lerp (can cause gimbal issues but preserves large angles)
            start_rot = tf.get("start_rot_e", (0.0, 0.0, 0.0))
            end_rot = tf.get("end_rot_e", (0.0, 0.0, 0.0))
            rot_euler = (
                start_rot[0] + (end_rot[0] - start_rot[0]) * t,
                start_rot[1] + (end_rot[1] - start_rot[1]) * t,
                start_rot[2] + (end_rot[2] - start_rot[2]) * t,
            )

        # --- Scale lerp ---
        start_scl = tf["start_scl"]
        end_scl = tf["end_scl"]
        scl = (
            start_scl[0] + (end_scl[0] - start_scl[0]) * t,
            start_scl[1] + (end_scl[1] - start_scl[1]) * t,
            start_scl[2] + (end_scl[2] - start_scl[2]) * t,
        )

        results.append({
            "obj_id": obj_id,
            "loc": loc,
            "rot_euler": rot_euler,
            "scl": scl,
            "finished": finished,
        })

    calc_time_us = (time.perf_counter() - calc_start) * 1_000_000
    count = len(results)

    if count > 0:
        # Build mode breakdown string
        mode_parts = []
        if mode_counts.get("euler", 0) > 0:
            mode_parts.append(f"euler={mode_counts['euler']}")
        if mode_counts.get("quat", 0) > 0:
            mode_parts.append(f"quat={mode_counts['quat']}")
        if mode_counts.get("local_delta", 0) > 0:
            mode_parts.append(f"local_delta={mode_counts['local_delta']}")
        mode_str = " ".join(mode_parts) if mode_parts else "none"

        # Count finished transforms
        finished_count = sum(1 for r in results if r["finished"])

        logs.append((
            "TRANSFORMS",
            f"BATCH count={count} modes=[{mode_str}] finished={finished_count} calc_time={calc_time_us:.0f}us"
        ))

    return {
        "success": True,
        "results": results,
        "count": count,
        "calc_time_us": calc_time_us,
        "logs": logs,
    }
