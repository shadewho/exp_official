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
from ..developer.dev_debug_gate import should_print_debug
from ..props_and_utils.exp_time import get_game_time

EXPL_TREE_ID = "ExploratoryNodesTreeType"

# Cache for operator reference (set by game loop)
_current_operator = None

# Pending tracker job tracking
_pending_tracker_job = None

# Filtered object names - only objects referenced by trackers (Phase 1.1 optimization)
_tracked_object_names: set = set()

# Cached object references - avoids O(n) bpy.data.objects.get() lookups each frame
_tracked_object_refs: dict = {}  # {name: bpy.types.Object}

# Generation counter - increments on reset to invalidate stale results
# Results with old generation are discarded to prevent false triggers after reset
_tracker_generation: int = 0

# Main-thread Compare evaluation (trees containing CompareNode can't serialize to worker)
_compare_chains: list = []         # [{interaction_index, condition_node, trigger_node_name, tree_name}]
_compare_last_values: dict = {}    # {interaction_index: last_bool_value}
_compare_primed: set = set()       # Interaction indices evaluated at least once


def set_current_operator(op):
    """Set the current operator reference for input state access."""
    global _current_operator
    _current_operator = op


# ══════════════════════════════════════════════════════════════════════════════
# COMPARE NODE DETECTION & MAIN-THREAD EVALUATION
# Trees containing CompareNode can't be serialized to the worker because
# CompareNode reads live bpy objects (PointerProperty). These trees are
# evaluated on the main thread each frame instead.
# ══════════════════════════════════════════════════════════════════════════════

def _tree_has_compare_node(node, visited=None):
    """
    Walk the bpy node graph upstream from 'node' and return True if any
    node in the tree is a CompareNode.  Used at game start to decide
    whether to send a condition tree to the worker or keep it main-thread.
    """
    if visited is None:
        visited = set()
    node_id = id(node)
    if node_id in visited:
        return False
    visited.add(node_id)

    if getattr(node, 'bl_idname', '') == 'CompareNodeType':
        return True

    for inp in node.inputs:
        if inp.is_linked:
            src = inp.links[0].from_node
            if _tree_has_compare_node(src, visited):
                return True
    return False


def _evaluate_bool_tree(node, visited=None):
    """
    Evaluate a boolean condition tree on the MAIN THREAD by walking live
    bpy node objects.  Handles CompareNode, Logic gates, BoolDataNode,
    and any node with export_bool().

    This is intentionally lightweight — typical trees are 1-5 nodes.
    Called at 30 Hz so must stay fast.
    """
    if visited is None:
        visited = set()
    node_id = id(node)
    if node_id in visited:
        return False
    visited.add(node_id)

    bl_id = getattr(node, 'bl_idname', '')

    # CompareNode — the primary reason this evaluator exists
    if bl_id == 'CompareNodeType':
        return node.export_bool()

    # Logic AND — all connected inputs must be True
    if bl_id == 'LogicAndNodeType':
        for inp in node.inputs:
            if inp.is_linked:
                src = inp.links[0].from_node
                if not _evaluate_bool_tree(src, visited):
                    return False
        return True

    # Logic OR — any connected input must be True
    if bl_id == 'LogicOrNodeType':
        for inp in node.inputs:
            if inp.is_linked:
                src = inp.links[0].from_node
                if _evaluate_bool_tree(src, visited):
                    return True
        return False

    # Logic NOT — invert first connected input
    if bl_id == 'LogicNotNodeType':
        for inp in node.inputs:
            if inp.is_linked:
                src = inp.links[0].from_node
                return not _evaluate_bool_tree(src, visited)
        return False

    # BoolDataNode — read live store value
    if bl_id == 'BoolDataNodeType':
        if hasattr(node, 'export_bool'):
            return node.export_bool()
        return False

    # Generic fallback — any node that exposes export_bool
    if hasattr(node, 'export_bool'):
        return node.export_bool()

    return False


# ══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION (Main Thread → Worker at game start)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_object_name(node, socket_name: str, fallback_prop: str) -> str:
    """
    Resolve object name from either a connected ObjectDataNode or inline property.

    Args:
        node: The tracker node
        socket_name: Name of the input socket (e.g., "Object A")
        fallback_prop: Name of the inline property (e.g., "object_a")

    Returns:
        Object name string for worker serialization
    """
    # Check if socket is connected
    socket = node.inputs.get(socket_name)
    if socket and socket.is_linked:
        src_node = socket.links[0].from_node
        # Check if source node has export_object_name (ObjectDataNode)
        if hasattr(src_node, 'export_object_name'):
            return src_node.export_object_name()

    # Fall back to inline property
    obj = getattr(node, fallback_prop, None)
    return obj.name if obj else ""


def _resolve_object_ref(node, socket_name: str, fallback_prop: str):
    """
    Resolve actual bpy.Object reference from socket or inline property.
    Used for reading bounding box data at serialization time.
    """
    socket = node.inputs.get(socket_name)
    if socket and socket.is_linked:
        src_node = socket.links[0].from_node
        if hasattr(src_node, 'export_object'):
            return src_node.export_object()
    return getattr(node, fallback_prop, None)


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
        result['object_a'] = _resolve_object_name(node, "Object A", "object_a")
        result['object_b'] = _resolve_object_name(node, "Object B", "object_b")
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
        obj_name = _resolve_object_name(node, "Object", "contact_object")
        # Auto-default to player character if Object not connected/set
        if not obj_name:
            scn = bpy.context.scene
            char = getattr(scn, 'target_armature', None)
            if char:
                obj_name = char.name
        result['object'] = obj_name
        result['target'] = _resolve_object_name(node, "Target", "contact_target")

        # Threshold with precomputed squared value for fast distance checks
        threshold = getattr(node, 'contact_threshold', 0.5)
        result['threshold_sq'] = threshold * threshold
        result['eval_hz'] = node.eval_hz

        # Pre-compute target AABB for surface-distance contact detection
        # Transform local bound_box corners to world space, then compute AABB
        target_obj = _resolve_object_ref(node, "Target", "contact_target")
        if target_obj and hasattr(target_obj, 'bound_box'):
            import mathutils
            bb = target_obj.bound_box
            mtx = target_obj.matrix_world
            corners = [mtx @ mathutils.Vector(bb[i]) for i in range(8)]
            aabb_min = (
                min(c[0] for c in corners),
                min(c[1] for c in corners),
                min(c[2] for c in corners),
            )
            aabb_max = (
                max(c[0] for c in corners),
                max(c[1] for c in corners),
                max(c[2] for c in corners),
            )
            # Store world-space AABB + initial position for moving-object delta
            loc = target_obj.matrix_world.translation
            result['target_aabb_min'] = aabb_min
            result['target_aabb_max'] = aabb_max
            result['target_initial_pos'] = (loc.x, loc.y, loc.z)

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

    # ─── Compare Node ───
    # Compare nodes connected to hitscan/projectile outputs are evaluated
    # at impact time by the impact cache system, not by the tracker worker.
    elif node_type == 'CompareNodeType':
        return None

    # ─── Data Node Passthrough ───
    # Data nodes (Boolean, Float, etc.) act as passthrough - follow their input
    # This allows chaining: Tracker → Boolean → Boolean → Trigger
    elif node_type in ('BoolDataNodeType', 'FloatDataNodeType', 'IntDataNodeType'):
        # Check if the Input socket is connected
        input_socket = node.inputs.get("Input")
        if input_socket and input_socket.is_linked:
            # Follow through to the source node
            src_node = input_socket.links[0].from_node
            return _serialize_node(src_node, visited)
        else:
            # No input connected - this is a static value, not useful for conditions
            # Return None to indicate no valid condition source
            return None

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
                log_game("TRACKERS", f"SERIALIZE_SKIP {node.name} no interaction_index")
                continue

            # Check for connected condition
            condition_socket = node.inputs.get("Condition")
            if not condition_socket or not condition_socket.is_linked:
                log_game("TRACKERS", f"SERIALIZE_SKIP {node.name} no Condition connected")
                continue

            # Check condition tree for CompareNode — those are evaluated
            # on the main thread, not serialized to the worker.
            src_node = condition_socket.links[0].from_node
            if _tree_has_compare_node(src_node):
                log_game("TRACKERS", f"SERIALIZE_SKIP {node.name} contains CompareNode (main-thread eval)")
                continue

            # Serialize the condition tree
            visited = set()
            try:
                condition_tree = _serialize_node(src_node, visited)
            except Exception as e:
                log_game("TRACKERS", f"SERIALIZE_ERROR {node.name}: {e}")
                condition_tree = None

            if condition_tree:
                trackers.append({
                    'interaction_index': inter_idx,
                    'trigger_node': node.name,
                    'tree_name': ng.name,
                    'condition_tree': condition_tree,
                })
                # DEBUG: Log what we serialized
                log_game("TRACKERS", f"SERIALIZE_OK {node.name} obj={condition_tree.get('object','')} tgt={condition_tree.get('targets',[])}")
            else:
                log_game("TRACKERS", f"SERIALIZE_FAIL {node.name} condition_tree=None")

    log_game("TRACKERS", f"SERIALIZED {len(trackers)} tracker chains for worker")
    return trackers


# ══════════════════════════════════════════════════════════════════════════════
# OBJECT NAME EXTRACTION (Phase 1.1 - Filter world state to tracked objects only)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_from_tree(node: dict):
    """
    Recursively extract object references from a serialized node tree.
    Adds found object names to _tracked_object_names.
    """
    global _tracked_object_names

    if not node:
        return

    # Distance tracker - has object_a and object_b
    obj_a = node.get("object_a")
    if obj_a:
        _tracked_object_names.add(obj_a)

    obj_b = node.get("object_b")
    if obj_b:
        _tracked_object_names.add(obj_b)

    # Contact tracker - has object and target
    contact_obj = node.get("object")
    if contact_obj:
        _tracked_object_names.add(contact_obj)

    contact_target = node.get("target")
    if contact_target:
        _tracked_object_names.add(contact_target)

    # Recurse into logic gate inputs
    for child in node.get("inputs", []):
        _extract_from_tree(child)


def _extract_referenced_objects(tracker_data: list) -> int:
    """
    Extract all object names referenced by tracker nodes.
    Called after serialize_tracker_graph() to build the filter set.

    Also caches object references for O(1) lookup during gameplay
    (avoids O(n) bpy.data.objects.get() calls each frame).

    Returns the number of unique objects found.
    """
    global _tracked_object_names, _tracked_object_refs
    _tracked_object_names.clear()
    _tracked_object_refs.clear()

    for tracker in tracker_data:
        tree = tracker.get("condition_tree")
        if tree:
            _extract_from_tree(tree)

    # Cache object references for O(1) lookup during gameplay
    for name in _tracked_object_names:
        obj = bpy.data.objects.get(name)
        if obj:
            _tracked_object_refs[name] = obj

    count = len(_tracked_object_names)
    cached = len(_tracked_object_refs)
    if count > 0:
        log_game("TRACKERS", f"FILTER_EXTRACTED {count} tracked objects, {cached} cached refs: {sorted(_tracked_object_names)}")
    else:
        log_game("TRACKERS", "FILTER_EXTRACTED 0 objects (no position-based trackers)")

    return count


# ══════════════════════════════════════════════════════════════════════════════
# MAIN-THREAD COMPARE CHAIN BUILD / EVALUATE / RESET
# ══════════════════════════════════════════════════════════════════════════════

def build_compare_chains(scene):
    """
    Scan ExternalTriggerNodes for condition trees that contain a CompareNode.
    Those trees are kept for per-frame main-thread evaluation instead of
    being sent to the worker.  Called once at game start.
    """
    global _compare_chains, _compare_last_values, _compare_primed
    _compare_chains = []
    _compare_last_values.clear()
    _compare_primed.clear()

    for ng in bpy.data.node_groups:
        if getattr(ng, 'bl_idname', '') != EXPL_TREE_ID:
            continue
        tree_scene = getattr(ng, 'scene', None)
        if tree_scene and tree_scene != scene:
            continue

        for node in ng.nodes:
            if getattr(node, 'bl_idname', '') != 'ExternalTriggerNodeType':
                continue

            inter_idx = getattr(node, 'interaction_index', -1)
            if inter_idx < 0:
                continue

            condition_socket = node.inputs.get("Condition")
            if not condition_socket or not condition_socket.is_linked:
                continue

            src_node = condition_socket.links[0].from_node
            if _tree_has_compare_node(src_node):
                _compare_chains.append({
                    'interaction_index': inter_idx,
                    'condition_node': src_node,
                    'trigger_node_name': node.name,
                    'tree_name': ng.name,
                })
                log_game("TRACKERS",
                         f"COMPARE_CHAIN inter={inter_idx} trigger={node.name} tree={ng.name}")

    if _compare_chains:
        log_game("TRACKERS",
                 f"BUILT {len(_compare_chains)} compare chain(s) for main-thread evaluation")


def evaluate_compare_chains():
    """
    Per-frame main-thread evaluation of Compare condition trees.
    Uses edge detection + priming (mirrors the worker pattern) so
    external_signal only changes on actual state transitions.
    """
    global _compare_chains, _compare_last_values, _compare_primed

    if not _compare_chains:
        return

    scn = bpy.context.scene
    interactions = getattr(scn, 'custom_interactions', None)
    if not interactions:
        return

    for chain in _compare_chains:
        inter_idx = chain['interaction_index']
        condition_node = chain['condition_node']

        # Evaluate the boolean tree on main thread
        try:
            new_value = _evaluate_bool_tree(condition_node)
        except Exception:
            continue

        # Priming: store initial state WITHOUT firing to avoid false triggers
        if inter_idx not in _compare_primed:
            _compare_primed.add(inter_idx)
            _compare_last_values[inter_idx] = new_value
            log_game("TRACKERS",
                     f"COMPARE_PRIME inter={inter_idx} initial={'TRUE' if new_value else 'FALSE'}")
            continue

        old_value = _compare_last_values.get(inter_idx, False)

        # Edge detection: only update external_signal on state change
        if new_value != old_value:
            _compare_last_values[inter_idx] = new_value
            if 0 <= inter_idx < len(interactions):
                interactions[inter_idx].external_signal = new_value
                log_game("TRACKERS",
                         f"COMPARE_SIGNAL inter={inter_idx} -> {'TRUE' if new_value else 'FALSE'}")


def reset_compare_chains():
    """Reset Compare chain state. Called on game end / reset."""
    global _compare_chains, _compare_last_values, _compare_primed
    _compare_chains = []
    _compare_last_values.clear()
    _compare_primed.clear()


# ══════════════════════════════════════════════════════════════════════════════
# WORLD STATE COLLECTION (Main Thread → Worker each frame)
# ══════════════════════════════════════════════════════════════════════════════

def collect_world_state(context) -> dict:
    """
    Collect minimal world state needed for tracker evaluation.
    Sent to worker each frame with EVALUATE_TRACKERS job.

    Phase 1.1 Optimization: Only collects positions for objects
    referenced by trackers, not all objects in scene.
    """
    global _current_operator, _tracked_object_names
    scn = context.scene

    # Object positions - FILTERED to only tracked objects (Phase 1.1)
    positions = {}

    # Always include character (needed for distance/contact checks)
    char = scn.target_armature
    if char:
        loc = char.matrix_world.translation
        positions[char.name] = (loc.x, loc.y, loc.z)

    # Only collect positions for objects referenced by trackers
    # Uses cached object refs for O(1) lookup instead of O(n) bpy.data.objects.get()
    for name in _tracked_object_names:
        if name in positions:
            continue  # Already added (e.g., character)
        obj = _tracked_object_refs.get(name)
        if obj:
            loc = obj.matrix_world.translation
            positions[name] = (loc.x, loc.y, loc.z)

    # Log world state filtering stats (Phase 1.1) - only compute if logging enabled
    if should_print_debug("world_state"):
        total_objects = sum(1 for o in bpy.data.objects if o.type in {'MESH', 'ARMATURE', 'EMPTY'})
        collected = len(positions)
        if total_objects > 0:
            reduction = ((total_objects - collected) / total_objects) * 100
            log_game("WORLD-STATE", f"COLLECT filtered={collected} total={total_objects} reduction={reduction:.0f}%")

    # Character state
    char_state = _get_character_state(context)

    # Input states from operator
    inputs = _get_input_states()

    # Game time
    game_time = get_game_time()

    return {
        'positions': positions,
        'char_state': char_state,
        'inputs': inputs,
        'game_time': game_time,
    }


def _get_character_state(context) -> str:
    """Get current character state for StateTrackerNode evaluation.

    Reads live physics state from the modal operator (set each frame
    after the KCC physics step).
    """
    global _current_operator
    op = _current_operator
    if not op:
        return 'IDLE'

    is_grounded = getattr(op, 'is_grounded', True)

    if not is_grounded:
        vel_z = getattr(op, 'z_velocity', 0.0)
        if vel_z > 0.5:
            return 'JUMPING'
        elif vel_z < -0.5:
            return 'FALLING'
        return 'AIRBORNE'

    # Grounded — determine locomotion state from speed and sprint key
    vel_xz = getattr(op, 'horizontal_speed', 0.0)
    if vel_xz < 0.1:
        return 'IDLE'

    keys = getattr(op, 'keys_pressed', set())
    run_key = getattr(op, 'pref_run_key', 'LEFT_SHIFT')
    if run_key in keys:
        return 'RUNNING'

    return 'WALKING'


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

    Also extracts referenced object names for Phase 1.1 world state filtering.
    """
    # PERFORMANCE: Direct access - class-level default is None
    if not modal.engine:
        log_game("TRACKERS", "CACHE_SKIP engine not available")
        return False

    # Build main-thread Compare chains BEFORE serializing (so serialize can skip them)
    build_compare_chains(context.scene)

    # Serialize tracker graph (skips trees containing CompareNode)
    tracker_data = serialize_tracker_graph(context.scene)

    # Phase 1.1: Extract referenced object names for world state filtering
    # This builds the filter set even if no trackers, so we don't collect all objects
    _extract_referenced_objects(tracker_data)

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
    Also evaluates main-thread Compare chains each frame.
    Called each frame.

    Returns job_id or -1 if not submitted.
    """
    global _tracker_generation

    # Evaluate main-thread Compare chains (trees containing CompareNode)
    evaluate_compare_chains()

    # PERFORMANCE: Direct access - class-level default is None
    if not modal.engine:
        return -1

    # Collect world state
    world_state = collect_world_state(context)

    # Submit job with generation to detect stale results after reset
    job_id = modal.engine.submit_job("EVALUATE_TRACKERS", {
        "world_state": world_state,
        "game_time": get_game_time(),
        "generation": _tracker_generation,
    })

    return job_id if job_id is not None else -1


def apply_tracker_results(result) -> int:
    """
    Apply tracker evaluation results from worker.
    Sets external_signal on triggered interactions.

    Returns number of signals changed.
    """
    global _tracker_generation

    if not result.success:
        log_game("TRACKERS", f"EVAL_FAIL error={getattr(result, 'error', 'unknown')}")
        return 0

    result_data = result.result
    if not result_data:
        return 0

    # Check generation to discard stale results from before reset
    result_generation = result_data.get("generation", -1)
    if result_generation != _tracker_generation:
        log_game("TRACKERS", f"DISCARD_STALE gen={result_generation} current={_tracker_generation}")
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
    global _current_operator, _tracked_object_names, _tracked_object_refs
    _current_operator = None
    _tracked_object_names.clear()
    _tracked_object_refs.clear()
    reset_compare_chains()
    log_game("TRACKERS", "STATE_RESET (filter, refs, and compare chains cleared)")


def reset_worker_trackers(modal, context) -> bool:
    """
    Reset worker-side tracker state by re-caching trackers.
    This clears _tracker_primed, _tracker_states, etc. in workers.
    Call on game reset (not just game start).

    Also increments generation counter to invalidate any stale results
    that were computed before the reset.

    Returns True if successful.
    """
    global _tracker_generation

    # Increment generation to invalidate stale results from before reset
    _tracker_generation += 1
    log_game("TRACKERS", f"GENERATION_INCREMENT now={_tracker_generation}")

    if not modal or not hasattr(modal, 'engine') or not modal.engine:
        log_game("TRACKERS", "WORKER_RESET_SKIP no engine available")
        return False

    # Re-cache triggers the worker to clear all state
    return cache_trackers_in_worker(modal, context)
