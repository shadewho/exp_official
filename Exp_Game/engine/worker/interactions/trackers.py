# Exp_Game/engine/worker/interactions/trackers.py
"""
Tracker condition tree evaluation for EXTERNAL triggers.
Runs in worker process - NO bpy access.
"""

import time


# Worker-side caches
_cached_trackers = []
_tracker_states = {}  # {interaction_index: last_bool_value}
_tracker_last_eval = {}  # {interaction_index: last_eval_time}
_tracker_primed = set()  # Trackers that have been evaluated at least once (prevents false trigger on first frame)


def reset_tracker_state():
    """Reset all tracker state. Called on game reset."""
    global _cached_trackers, _tracker_states, _tracker_last_eval, _tracker_primed
    _cached_trackers = []
    _tracker_states.clear()
    _tracker_last_eval.clear()
    _tracker_primed.clear()


def _compare(value: float, op: str, threshold: float) -> bool:
    """Compare value against threshold using operator."""
    if op == 'LT': return value < threshold
    if op == 'LE': return value <= threshold
    if op == 'EQ': return abs(value - threshold) < 0.001
    if op == 'NE': return abs(value - threshold) >= 0.001
    if op == 'GE': return value >= threshold
    if op == 'GT': return value > threshold
    return False


def _get_max_eval_hz(tree: dict) -> int:
    """
    Find the maximum eval_hz in the entire condition tree.
    Logic gates don't have eval_hz, so we check all children.
    The fastest tracker wins (highest Hz = most responsive).
    """
    if not tree:
        return 10

    # This node's Hz (if it has one)
    max_hz = tree.get('eval_hz', 0)

    # Check children (for logic gates)
    for child in tree.get('inputs', []):
        child_hz = _get_max_eval_hz(child)
        if child_hz > max_hz:
            max_hz = child_hz

    return max_hz if max_hz > 0 else 10


def _eval_condition_tree(tree: dict, world_state: dict) -> bool:
    """
    Recursively evaluate a serialized condition tree.
    This is the WORKER-SIDE evaluation - no bpy access!
    """
    node_type = tree.get('type', '')
    positions = world_state.get('positions', {})
    inputs = world_state.get('inputs', {})
    char_state = world_state.get('char_state', 'IDLE')
    game_time = world_state.get('game_time', 0.0)

    # Distance Tracker
    if node_type == 'DistanceTrackerNodeType':
        obj_a = tree.get('object_a', '')
        obj_b = tree.get('object_b', '')
        pos_a = positions.get(obj_a)
        pos_b = positions.get(obj_b)

        if not pos_a or not pos_b:
            return False

        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        dz = pos_a[2] - pos_b[2]
        dist = (dx*dx + dy*dy + dz*dz) ** 0.5

        return _compare(dist, tree.get('op', 'LT'), tree.get('value', 5.0))

    # State Tracker
    elif node_type == 'StateTrackerNodeType':
        target_state = tree.get('state', 'GROUNDED')
        equals = tree.get('equals', True)

        if target_state == 'GROUNDED':
            is_match = char_state in ('IDLE', 'WALKING', 'RUNNING')
        elif target_state == 'AIRBORNE':
            is_match = char_state in ('JUMPING', 'FALLING', 'AIRBORNE')
        else:
            is_match = (char_state == target_state)

        return is_match if equals else not is_match

    # Contact Tracker - AABB surface-distance contact check
    elif node_type == 'ContactTrackerNodeType':
        obj = tree.get('object', '')
        target = tree.get('target', '')
        threshold_sq = tree.get('threshold_sq', 0.25)

        pos_obj = positions.get(obj)
        if not pos_obj:
            return False

        aabb_min = tree.get('target_aabb_min')
        aabb_max = tree.get('target_aabb_max')

        if aabb_min and aabb_max:
            # AABB contact: distance from point to nearest surface of bounding box
            # For moving targets, offset AABB by position delta from serialization
            pos_target = positions.get(target)
            init_pos = tree.get('target_initial_pos')
            if pos_target and init_pos:
                # Moving object: shift AABB by delta
                dx = pos_target[0] - init_pos[0]
                dy = pos_target[1] - init_pos[1]
                dz = pos_target[2] - init_pos[2]
                mn0 = aabb_min[0] + dx; mn1 = aabb_min[1] + dy; mn2 = aabb_min[2] + dz
                mx0 = aabb_max[0] + dx; mx1 = aabb_max[1] + dy; mx2 = aabb_max[2] + dz
            else:
                # Static object: use serialized AABB directly
                mn0 = aabb_min[0]; mn1 = aabb_min[1]; mn2 = aabb_min[2]
                mx0 = aabb_max[0]; mx1 = aabb_max[1]; mx2 = aabb_max[2]

            # Nearest point on AABB to character position
            ox, oy, oz = pos_obj
            nx = mn0 if ox < mn0 else (mx0 if ox > mx0 else ox)
            ny = mn1 if oy < mn1 else (mx1 if oy > mx1 else oy)
            nz = mn2 if oz < mn2 else (mx2 if oz > mx2 else oz)

            ddx = ox - nx
            ddy = oy - ny
            ddz = oz - nz
            return (ddx*ddx + ddy*ddy + ddz*ddz) < threshold_sq
        else:
            # Fallback: origin-to-origin distance (no AABB data)
            pos_target = positions.get(target)
            if not pos_target:
                return False
            dx = pos_obj[0] - pos_target[0]
            dy = pos_obj[1] - pos_target[1]
            dz = pos_obj[2] - pos_target[2]
            return (dx*dx + dy*dy + dz*dz) < threshold_sq

    # Input Tracker
    elif node_type == 'InputTrackerNodeType':
        action = tree.get('action', '')
        is_pressed = tree.get('is_pressed', True)
        current = inputs.get(action, False)
        return current if is_pressed else not current

    # Game Time Tracker
    elif node_type == 'GameTimeTrackerNodeType':
        if not tree.get('compare_enabled', True):
            return True

        return _compare(game_time, tree.get('op', 'GE'), tree.get('value', 10.0))

    # Logic AND
    elif node_type == 'LogicAndNodeType':
        children = tree.get('inputs', [])
        if not children:
            return True
        for child in children:
            if not _eval_condition_tree(child, world_state):
                return False
        return True

    # Logic OR
    elif node_type == 'LogicOrNodeType':
        children = tree.get('inputs', [])
        if not children:
            return False
        for child in children:
            if _eval_condition_tree(child, world_state):
                return True
        return False

    # Logic NOT
    elif node_type == 'LogicNotNodeType':
        children = tree.get('inputs', [])
        if not children:
            return False
        return not _eval_condition_tree(children[0], world_state)

    return False


def handle_cache_trackers(job_data: dict) -> dict:
    """
    Handle CACHE_TRACKERS job - store tracker definitions from main thread.
    Called once at game start.

    Input job_data:
        {
            "trackers": [
                {
                    "interaction_index": int,
                    "condition_tree": {...},
                },
                ...
            ]
        }

    Returns:
        {
            "success": bool,
            "tracker_count": int,
            "message": str,
            "logs": [(category, message), ...],
        }
    """
    global _cached_trackers, _tracker_states, _tracker_last_eval, _tracker_primed

    trackers = job_data.get("trackers", [])
    _cached_trackers = list(trackers)
    _tracker_states.clear()
    _tracker_last_eval.clear()
    _tracker_primed.clear()  # Reset priming flags

    logs = [("TRACKERS", f"WORKER_CACHED {len(_cached_trackers)} tracker chains")]

    return {
        "success": True,
        "tracker_count": len(_cached_trackers),
        "message": "Trackers cached successfully",
        "logs": logs,
    }


def handle_evaluate_trackers(job_data: dict) -> dict:
    """
    Handle EVALUATE_TRACKERS job - evaluate all cached trackers with world state.
    Called each frame from main thread.

    Input job_data:
        {
            "world_state": {
                "positions": {obj_name: (x, y, z), ...},
                "inputs": {action_name: bool, ...},
                "char_state": str,
                "game_time": float,
                "contacts": {obj_name: [contacted_obj_names], ...},
            },
            "game_time": float,
        }

    Returns:
        {
            "success": bool,
            "signal_updates": {str(inter_idx): bool, ...},
            "fired_indices": [int, ...],
            "trackers_evaluated": int,
            "calc_time_us": float,
            "logs": [(category, message), ...],
        }
    """
    global _cached_trackers, _tracker_states, _tracker_last_eval, _tracker_primed

    calc_start = time.perf_counter()
    logs = []

    world_state = job_data.get("world_state", {})
    game_time = job_data.get("game_time", 0.0)
    generation = job_data.get("generation", 0)  # Echo back for stale result detection

    if not _cached_trackers:
        return {
            "success": True,
            "signal_updates": {},
            "fired_indices": [],
            "trackers_evaluated": 0,
            "calc_time_us": 0,
            "logs": [],
            "generation": generation,
        }

    signal_updates = {}
    fired_indices = []
    evaluated = 0

    for tracker in _cached_trackers:
        inter_idx = tracker.get('interaction_index', -1)
        if inter_idx < 0:
            continue

        # Hz throttling - use max Hz from entire tree (fastest child wins)
        tree = tracker.get('condition_tree', {})
        eval_hz = _get_max_eval_hz(tree)

        eval_interval = 1.0 / max(1, eval_hz)
        last_eval = _tracker_last_eval.get(inter_idx, 0.0)

        if game_time - last_eval < eval_interval:
            continue

        _tracker_last_eval[inter_idx] = game_time

        # Evaluate condition tree
        new_value = _eval_condition_tree(tree, world_state)
        evaluated += 1

        # PRIMING: First evaluation stores initial state WITHOUT firing
        # This prevents false triggers when condition starts as True
        if inter_idx not in _tracker_primed:
            _tracker_primed.add(inter_idx)
            _tracker_states[inter_idx] = new_value
            state_str = "TRUE" if new_value else "FALSE"
            logs.append(("TRACKERS", f"PRIME inter={inter_idx} initial={state_str}"))
            continue  # Don't fire on first evaluation

        old_value = _tracker_states.get(inter_idx, False)

        # Track state change (edge detection)
        if new_value != old_value:
            _tracker_states[inter_idx] = new_value
            signal_updates[str(inter_idx)] = new_value

            if new_value:
                fired_indices.append(inter_idx)

            state_str = "TRUE" if new_value else "FALSE"
            logs.append(("TRACKERS", f"FIRE inter={inter_idx} -> {state_str}"))

    calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

    if evaluated > 0:
        logs.append(("TRACKERS", f"EVAL {evaluated} trackers, {len(signal_updates)} changed, {calc_time_us:.0f}us"))

    return {
        "success": True,
        "signal_updates": signal_updates,
        "fired_indices": fired_indices,
        "trackers_evaluated": evaluated,
        "calc_time_us": calc_time_us,
        "logs": logs,
        "generation": generation,
    }
