# Exp_Game/engine/worker/interactions/triggers.py
"""
Trigger evaluation handlers for PROXIMITY and COLLISION interactions.
Runs in worker process - NO bpy access.

State-based trigger logic:
- Worker receives current zone state and trigger mode
- Worker determines if condition is met (in zone)
- Worker decides transition (ENTERED, EXITED, STILL_IN, STILL_OUT)
- Worker decides if trigger should fire based on mode
- Main thread applies state updates and fires reactions
"""

import time


def handle_interaction_check_batch(job_data: dict) -> dict:
    """
    Handle INTERACTION_CHECK_BATCH job.
    Evaluates PROXIMITY and COLLISION triggers with full state-based logic.

    Input job_data:
        {
            "interactions": [
                {
                    "type": "PROXIMITY" | "COLLISION",
                    # PROXIMITY fields:
                    "obj_a_pos": (x, y, z),
                    "obj_b_pos": (x, y, z),
                    "threshold": float,
                    # COLLISION fields:
                    "aabb_a": (min_x, max_x, min_y, max_y, min_z, max_z),
                    "aabb_b": (min_x, max_x, min_y, max_y, min_z, max_z),
                    "margin": float,
                    # State data (all types):
                    "is_in_zone": bool,        # Previous zone state
                    "has_fired": bool,         # Has already fired
                    "last_trigger_time": float,
                    "trigger_mode": str,       # ENTER_ONLY, CONTINUOUS, COOLDOWN
                    "trigger_cooldown": float,
                },
                ...
            ],
            "current_time": float,  # For cooldown checks
        }

    Returns:
        {
            "triggered_indices": [int, ...],  # indices that triggered
            "state_updates": {                # Per-interaction state changes
                "0": {
                    "is_in_zone": bool,
                    "should_fire": bool,
                    "transition": str,  # ENTERED, EXITED, STILL_IN, STILL_OUT
                    "should_update_time": bool,
                },
                ...
            },
            "count": int,
            "calc_time_us": float,
        }
    """
    calc_start = time.perf_counter()

    interactions = job_data.get("interactions", [])
    current_time = job_data.get("current_time", 0.0)

    triggered_indices = []
    state_updates = {}

    for i, inter_data in enumerate(interactions):
        inter_type = inter_data.get("type")

        # Get state data
        was_in_zone = inter_data.get("is_in_zone", False)
        has_fired = inter_data.get("has_fired", False)
        last_trigger_time = inter_data.get("last_trigger_time", 0.0)
        trigger_mode = inter_data.get("trigger_mode", "ENTER_ONLY")
        trigger_cooldown = inter_data.get("trigger_cooldown", 0.0)

        # Check if currently in zone
        now_in_zone = False

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
                now_in_zone = dist_squared <= threshold_squared

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
                now_in_zone = overlap_x and overlap_y and overlap_z

        # Determine transition
        if now_in_zone and not was_in_zone:
            transition = "ENTERED"
        elif not now_in_zone and was_in_zone:
            transition = "EXITED"
        elif now_in_zone:
            transition = "STILL_IN"
        else:
            transition = "STILL_OUT"

        # Determine if should fire based on trigger_mode
        should_fire = False
        should_update_time = False

        if trigger_mode == "ONE_SHOT":
            # Fire once on first entry, never again (no reset on exit)
            if transition == "ENTERED" and not has_fired:
                should_fire = True
                should_update_time = True

        elif trigger_mode == "ENTER_ONLY":
            # Fire on entry, resets when exiting (can re-trigger on next entry)
            if transition == "ENTERED" and not has_fired:
                should_fire = True
                should_update_time = True

        elif trigger_mode == "CONTINUOUS":
            # Fire every frame while in zone
            if now_in_zone:
                should_fire = True
                should_update_time = True

        elif trigger_mode == "COOLDOWN":
            # Fire on entry, then respect cooldown for re-fires
            if now_in_zone:
                time_since_last = current_time - last_trigger_time
                if not has_fired or time_since_last >= trigger_cooldown:
                    should_fire = True
                    should_update_time = True

        # Record state update
        state_updates[str(i)] = {
            "is_in_zone": now_in_zone,
            "should_fire": should_fire,
            "transition": transition,
            "should_update_time": should_update_time,
        }

        if should_fire:
            triggered_indices.append(i)

    calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

    return {
        "triggered_indices": triggered_indices,
        "state_updates": state_updates,
        "count": len(interactions),
        "calc_time_us": calc_time_us,
    }
