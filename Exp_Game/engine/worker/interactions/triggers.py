# Exp_Game/engine/worker/interactions/triggers.py
"""
Trigger evaluation handlers for PROXIMITY and COLLISION interactions.
Runs in worker process - NO bpy access.
"""

import time


def handle_interaction_check_batch(job_data: dict) -> dict:
    """
    Handle INTERACTION_CHECK_BATCH job.
    Evaluates PROXIMITY and COLLISION triggers in batch.

    Input job_data:
        {
            "interactions": [
                {
                    "type": "PROXIMITY" | "COLLISION",
                    "index": int,  # interaction index for result mapping
                    # PROXIMITY fields:
                    "obj_a_pos": (x, y, z),
                    "obj_b_pos": (x, y, z),
                    "threshold": float,
                    # COLLISION fields:
                    "aabb_a": (min_x, max_x, min_y, max_y, min_z, max_z),
                    "aabb_b": (min_x, max_x, min_y, max_y, min_z, max_z),
                    "margin": float,
                },
                ...
            ],
            "player_position": (x, y, z),  # for reference
        }

    Returns:
        {
            "triggered_indices": [int, ...],  # indices that triggered
            "count": int,  # total interactions checked
            "calc_time_us": float,
        }
    """
    calc_start = time.perf_counter()

    interactions = job_data.get("interactions", [])
    player_position = job_data.get("player_position", (0, 0, 0))

    triggered_indices = []
    px, py, pz = player_position

    for i, inter_data in enumerate(interactions):
        inter_type = inter_data.get("type")
        inter_index = inter_data.get("index", i)

        if inter_type == "PROXIMITY":
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
                    triggered_indices.append(inter_index)

        elif inter_type == "COLLISION":
            aabb_a = inter_data.get("aabb_a")
            aabb_b = inter_data.get("aabb_b")
            margin = inter_data.get("margin", 0.0)

            if aabb_a and aabb_b:
                a_minx, a_maxx, a_miny, a_maxy, a_minz, a_maxz = aabb_a
                b_minx, b_maxx, b_miny, b_maxy, b_minz, b_maxz = aabb_b

                # Expand AABBs by margin
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

                # Check overlap on all axes
                overlap_x = (a_minx <= b_maxx) and (a_maxx >= b_minx)
                overlap_y = (a_miny <= b_maxy) and (a_maxy >= b_miny)
                overlap_z = (a_minz <= b_maxz) and (a_maxz >= b_minz)

                if overlap_x and overlap_y and overlap_z:
                    triggered_indices.append(inter_index)

    calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

    return {
        "triggered_indices": triggered_indices,
        "count": len(interactions),
        "calc_time_us": calc_time_us,
    }
