# Exp_Game/interactions/exp_tracker_eval.py
"""
Tracker Runtime Evaluation - WORKER OFFLOADED

The node graph is just the visual representation - actual evaluation
happens on the WORKER THREAD for performance.

Architecture:
    1. Game start: serialize_tracker_graph() → CACHE_TRACKERS job to worker
    2. Each frame: collect_world_state() → EVALUATE_TRACKERS job
    3. Worker evaluates all conditions using cached graph + world state
    4. Worker returns fired tracker indices
    5. Main thread: apply_tracker_results() sets external_signal on triggers
"""

import bpy
from ..developer.dev_logger import log_game
from ..props_and_utils.exp_time import get_game_time

EXPL_TREE_ID = "ExploratoryNodesTreeType"

# Cache for operator reference (set by game loop)
_current_operator = None

# Pending tracker job tracking
_pending_tracker_job = None


def set_current_operator(op):
    """Set the current operator reference for input state access."""
    global _current_operator
    _current_operator = op


# ══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION (Main Thread → Worker at game start)
# ══════════════════════════════════════════════════════════════════════════════

def _serialize_node(node, visited: set) -> dict:
    """
    Recursively serialize a tracker/logic node for worker.
    Returns serializable dict with node type and parameters.
    """
    node_id = id(node)
    if node_id in visited:
        return None  # Prevent cycles
    visited.add(node_id)

    node_type = getattr(node, 'bl_idname', '')
    result = {'type': node_type, 'name': node.name}

    # ─── Distance Tracker ───
    if node_type == 'DistanceTrackerNodeType':
        result['object_a'] = node.object_a.name if node.object_a else ""
        result['object_b'] = node.object_b.name if node.object_b else ""
        result['op'] = node.operator
        result['value'] = node.distance
        result['eval_hz'] = node.eval_hz

    # ─── State Tracker ───
    elif node_type == 'StateTrackerNodeType':
        result['state'] = node.state
        result['equals'] = node.equals
        result['eval_hz'] = node.eval_hz

    # ─── Contact Tracker ───
    elif node_type == 'ContactTrackerNodeType':
        result['object'] = node.contact_object.name if node.contact_object else ""
        if node.use_collection and node.contact_collection:
            result['targets'] = [o.name for o in node.contact_collection.objects]
        elif node.contact_target:
            result['targets'] = [node.contact_target.name]
        else:
            result['targets'] = []
        result['eval_hz'] = node.eval_hz

    # ─── Input Tracker ───
    elif node_type == 'InputTrackerNodeType':
        result['action'] = node.input_action
        result['is_pressed'] = node.is_pressed
        result['eval_hz'] = node.eval_hz

    # ─── Game Time Tracker ───
    elif node_type == 'GameTimeTrackerNodeType':
        result['compare_enabled'] = node.compare_enabled
        result['op'] = node.operator
        result['value'] = node.time_threshold
        result['eval_hz'] = node.eval_hz

    # ─── Logic Gates ───
    elif node_type in ('LogicAndNodeType', 'LogicOrNodeType', 'LogicNotNodeType'):
        # Serialize connected input nodes
        result['inputs'] = []
        for inp in node.inputs:
            if inp.is_linked:
                src_node = inp.links[0].from_node
                child = _serialize_node(src_node, visited)
                if child:
                    result['inputs'].append(child)

    return result


def serialize_tracker_graph(scene) -> list:
    """
    Serialize all tracker chains connected to ExternalTriggerNodes.
    Called once at game start. Returns list of tracker definitions for worker.

    Each entry: {
        'interaction_index': int,
        'trigger_node': str,
        'condition_tree': {...}  # Serialized node tree
    }
    """
    trackers = []

    for ng in bpy.data.node_groups:
        if getattr(ng, 'bl_idname', '') != EXPL_TREE_ID:
            continue

        # Only process trees for this scene
        tree_scene = getattr(ng, 'scene', None)
        if tree_scene and tree_scene != scene:
            continue

        for node in ng.nodes:
            if getattr(node, 'bl_idname', '') != 'ExternalTriggerNodeType':
                continue

            inter_idx = getattr(node, 'interaction_index', -1)
            if inter_idx < 0:
                continue

            # Check for connected condition
            condition_socket = node.inputs.get("Condition")
            if not condition_socket or not condition_socket.is_linked:
                continue

            # Serialize the condition tree
            src_node = condition_socket.links[0].from_node
            visited = set()
            condition_tree = _serialize_node(src_node, visited)

            if condition_tree:
                trackers.append({
                    'interaction_index': inter_idx,
                    'trigger_node': node.name,
                    'tree_name': ng.name,
                    'condition_tree': condition_tree,
                })

    log_game("TRACKERS", f"SERIALIZED {len(trackers)} tracker chains for worker")
    return trackers


# ══════════════════════════════════════════════════════════════════════════════
# WORLD STATE COLLECTION (Main Thread → Worker each frame)
# ══════════════════════════════════════════════════════════════════════════════

def collect_world_state(context) -> dict:
    """
    Collect minimal world state needed for tracker evaluation.
    Sent to worker each frame with EVALUATE_TRACKERS job.
    """
    global _current_operator
    scn = context.scene

    # Object positions (only for objects referenced by trackers)
    positions = {}
    for obj in bpy.data.objects:
        if obj.type in {'MESH', 'ARMATURE', 'EMPTY'}:
            loc = obj.matrix_world.translation
            positions[obj.name] = (loc.x, loc.y, loc.z)

    # Character state
    char_state = _get_character_state(context)

    # Input states from operator
    inputs = _get_input_states()

    # Game time
    game_time = get_game_time()

    # Contacts (from physics system if available)
    contacts = getattr(scn, 'exp_physics_contacts', {})

    return {
        'positions': positions,
        'char_state': char_state,
        'inputs': inputs,
        'game_time': game_time,
        'contacts': dict(contacts) if contacts else {},
    }


def _get_character_state(context) -> str:
    """Get current character state for StateTrackerNode evaluation."""
    scn = context.scene

    is_grounded = getattr(scn, 'exp_physics_grounded', True)
    is_sprinting = getattr(scn, 'exp_physics_sprinting', False)
    is_crouching = getattr(scn, 'exp_physics_crouching', False)

    if not is_grounded:
        vel_z = getattr(scn, 'exp_physics_velocity_z', 0.0)
        if vel_z > 0.5:
            return 'JUMPING'
        elif vel_z < -0.5:
            return 'FALLING'
        return 'AIRBORNE'

    if is_crouching:
        return 'CROUCHING'
    if is_sprinting:
        return 'SPRINTING'

    vel_xz = getattr(scn, 'exp_physics_velocity_xz', 0.0)
    if vel_xz < 0.1:
        return 'IDLE'
    elif vel_xz < 2.0:
        return 'WALKING'
    else:
        return 'RUNNING'


def _get_input_states() -> dict:
    """Get current input states from operator."""
    global _current_operator

    if not _current_operator:
        return {}

    op = _current_operator
    keys_pressed = getattr(op, 'keys_pressed', set())

    return {
        'FORWARD': getattr(op, 'pref_forward_key', 'W') in keys_pressed,
        'BACKWARD': getattr(op, 'pref_backward_key', 'S') in keys_pressed,
        'LEFT': getattr(op, 'pref_left_key', 'A') in keys_pressed,
        'RIGHT': getattr(op, 'pref_right_key', 'D') in keys_pressed,
        'JUMP': getattr(op, 'pref_jump_key', 'SPACE') in keys_pressed,
        'RUN': getattr(op, 'pref_run_key', 'LEFT_SHIFT') in keys_pressed,
        'ACTION': getattr(op, 'pref_action_key', 'E') in keys_pressed,
        'INTERACT': getattr(op, 'pref_interact_key', 'F') in keys_pressed,
        'RESET': getattr(op, 'pref_reset_key', 'R') in keys_pressed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# WORKER JOB SUBMISSION & RESULT HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def cache_trackers_in_worker(modal, context) -> bool:
    """
    Serialize tracker graph and send to worker at game start.
    Called after engine is ready.
    """
    if not hasattr(modal, 'engine') or not modal.engine:
        log_game("TRACKERS", "CACHE_SKIP engine not available")
        return False

    # Serialize tracker graph
    tracker_data = serialize_tracker_graph(context.scene)

    if not tracker_data:
        log_game("TRACKERS", "CACHE_SKIP no tracker chains found")
        return True  # Not an error, just no trackers

    # Send to all workers using broadcast_job (sends to all workers)
    num_sent = modal.engine.broadcast_job("CACHE_TRACKERS", {"trackers": tracker_data})

    log_game("TRACKERS", f"CACHE_BROADCAST {len(tracker_data)} chains to {num_sent} workers")
    return True


def submit_tracker_evaluation(modal, context) -> int:
    """
    Submit EVALUATE_TRACKERS job to worker with current world state.
    Called each frame.

    Returns job_id or -1 if not submitted.
    """
    if not hasattr(modal, 'engine') or not modal.engine:
        return -1

    # Collect world state
    world_state = collect_world_state(context)

    # Submit job (no pending tracking - just submit every frame)
    job_id = modal.engine.submit_job("EVALUATE_TRACKERS", {
        "world_state": world_state,
        "game_time": get_game_time(),
    })

    return job_id if job_id is not None else -1


def apply_tracker_results(result) -> int:
    """
    Apply tracker evaluation results from worker.
    Sets external_signal on triggered interactions.

    Returns number of signals changed.
    """
    if not result.success:
        log_game("TRACKERS", f"EVAL_FAIL error={getattr(result, 'error', 'unknown')}")
        return 0

    result_data = result.result
    if not result_data:
        return 0

    # Forward worker logs to main thread logging
    logs = result_data.get("logs", [])
    for category, message in logs:
        log_game(category, message)

    signal_updates = result_data.get("signal_updates", {})

    if not signal_updates:
        return 0

    scn = bpy.context.scene
    if not hasattr(scn, 'custom_interactions'):
        return 0

    changes = 0
    for inter_idx_str, new_value in signal_updates.items():
        inter_idx = int(inter_idx_str)
        if 0 <= inter_idx < len(scn.custom_interactions):
            inter = scn.custom_interactions[inter_idx]
            inter.external_signal = new_value
            changes += 1
            state_str = "TRUE" if new_value else "FALSE"
            log_game("TRACKERS", f"MAIN_SIGNAL inter={inter_idx} -> {state_str}")

    return changes


def process_tracker_result(result) -> bool:
    """
    Check if result is a tracker evaluation and process it.
    Returns True if it was a tracker result.
    """
    if getattr(result, 'job_type', '') != "EVALUATE_TRACKERS":
        return False

    apply_tracker_results(result)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def reset_tracker_state():
    """Reset tracker state cache. Call on game start/reset."""
    global _current_operator
    _current_operator = None
    log_game("TRACKERS", "STATE_RESET")
