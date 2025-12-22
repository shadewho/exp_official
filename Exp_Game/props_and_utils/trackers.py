# Exp_Game/props_and_utils/trackers.py
"""
Tracker System - Worker-Offloaded Conditional Logic

PERFORMANCE CRITICAL:
- All condition evaluation happens on the WORKER thread
- Main thread only receives "fire" notifications
- Conditions checked at configurable Hz (not every frame)
- Edge detection on worker (only fires on state transitions)

Architecture:
    1. Main thread serializes tracker definitions at game start
    2. Worker receives tracker configs and evaluates them
    3. Worker does edge detection (false->true transitions)
    4. Worker sends minimal "fire" messages back
    5. Main thread dispatches to reaction chains

Condition Types:
    - DISTANCE: Distance between two objects </>/= value
    - CONTACT: Mesh collision/overlap detection
    - CHAR_STATE: Character state (run, walk, jump, grounded, etc.)
    - PROPERTY: Object custom property comparison
    - COMPARE: Generic value comparison (float/int/bool)

Fire Modes:
    - ON_BECOME_TRUE: Fire once when condition transitions false->true
    - ON_BECOME_FALSE: Fire once when condition transitions true->false
    - WHILE_TRUE: Fire repeatedly while true (respects cooldown)
    - ON_CHANGE: Fire on any transition
"""

import bpy
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from bpy.props import (
    StringProperty, IntProperty, FloatProperty, BoolProperty,
    EnumProperty, PointerProperty, CollectionProperty
)
from bpy.types import PropertyGroup


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CONDITION_TYPES = [
    ('DISTANCE', "Distance", "Distance between two objects"),
    ('CONTACT', "Contact", "Mesh collision/overlap"),
    ('CHAR_STATE', "Character State", "Character movement state"),
    ('PROPERTY', "Property", "Object custom property"),
    ('COMPARE', "Compare", "Compare two values"),
]

COMPARE_OPERATORS = [
    ('LT', "<", "Less than"),
    ('LE', "<=", "Less than or equal"),
    ('EQ', "==", "Equal"),
    ('NE', "!=", "Not equal"),
    ('GE', ">=", "Greater than or equal"),
    ('GT', ">", "Greater than"),
]

CHAR_STATES = [
    ('GROUNDED', "Grounded", "Character is on ground"),
    ('AIRBORNE', "Airborne", "Character is in air"),
    ('SPRINTING', "Sprinting", "Character is sprinting"),
    ('WALKING', "Walking", "Character is walking"),
    ('RUNNING', "Running", "Character is running"),
    ('JUMPING', "Jumping", "Character is jumping"),
    ('FALLING', "Falling", "Character is falling"),
    ('CROUCHING', "Crouching", "Character is crouching"),
    ('IDLE', "Idle", "Character is idle"),
]

FIRE_MODES = [
    ('ON_BECOME_TRUE', "On Become True", "Fire once when condition becomes true"),
    ('ON_BECOME_FALSE', "On Become False", "Fire once when condition becomes false"),
    ('WHILE_TRUE', "While True", "Fire repeatedly while condition is true"),
    ('ON_CHANGE', "On Change", "Fire on any state change"),
]

LOGIC_MODES = [
    ('AND', "AND", "All conditions must be true"),
    ('OR', "OR", "Any condition must be true"),
]

# Input actions (logical key actions from preferences)
INPUT_ACTIONS = [
    ('FORWARD', "Forward", "Forward movement key"),
    ('BACKWARD', "Backward", "Backward movement key"),
    ('LEFT', "Left", "Left strafe key"),
    ('RIGHT', "Right", "Right strafe key"),
    ('JUMP', "Jump", "Jump key"),
    ('RUN', "Run", "Run/sprint modifier"),
    ('ACTION', "Action", "Primary action key"),
    ('INTERACT', "Interact", "Interact key"),
    ('RESET', "Reset", "Reset key"),
]


# ══════════════════════════════════════════════════════════════════════════════
# PROPERTY GROUPS
# ══════════════════════════════════════════════════════════════════════════════

class TrackerConditionPG(PropertyGroup):
    """Single condition within a tracker."""

    uid: StringProperty(name="UID", default="")

    condition_type: EnumProperty(
        name="Type",
        items=CONDITION_TYPES,
        default='DISTANCE'
    )

    # Distance condition
    object_a: PointerProperty(type=bpy.types.Object, name="Object A")
    object_b: PointerProperty(type=bpy.types.Object, name="Object B")
    distance_op: EnumProperty(name="Operator", items=COMPARE_OPERATORS, default='LT')
    distance_value: FloatProperty(name="Distance", default=5.0, min=0.0)

    # Contact condition
    contact_object: PointerProperty(type=bpy.types.Object, name="Object")
    contact_target: PointerProperty(type=bpy.types.Object, name="Target")
    contact_collection: PointerProperty(type=bpy.types.Collection, name="Collection")
    use_collection: BoolProperty(name="Use Collection", default=False)

    # Character state condition
    char_state: EnumProperty(name="State", items=CHAR_STATES, default='GROUNDED')
    char_state_equals: BoolProperty(name="Is", default=True)

    # Property condition
    prop_object: PointerProperty(type=bpy.types.Object, name="Object")
    prop_name: StringProperty(name="Property", default="")
    prop_op: EnumProperty(name="Operator", items=COMPARE_OPERATORS, default='EQ')
    prop_value: FloatProperty(name="Value", default=0.0)

    # Compare condition (uses data node inputs via node sockets)
    compare_op: EnumProperty(name="Operator", items=COMPARE_OPERATORS, default='EQ')


class TrackerDefinitionPG(PropertyGroup):
    """Complete tracker definition with conditions and fire settings."""

    uid: StringProperty(name="UID", default="")
    name: StringProperty(name="Name", default="Tracker")
    enabled: BoolProperty(name="Enabled", default=True)

    # Conditions stored in collection
    conditions: CollectionProperty(type=TrackerConditionPG)
    active_condition_index: IntProperty(name="Active Condition", default=0)

    # Logic mode for combining conditions
    logic_mode: EnumProperty(name="Logic", items=LOGIC_MODES, default='AND')

    # Fire settings
    fire_mode: EnumProperty(name="Fire Mode", items=FIRE_MODES, default='ON_BECOME_TRUE')
    cooldown: FloatProperty(name="Cooldown", default=0.0, min=0.0, description="Seconds between fires")
    max_fires: IntProperty(name="Max Fires", default=0, min=0, description="0 = unlimited")
    eval_hz: IntProperty(name="Eval Rate", default=10, min=1, max=30, description="Checks per second")

    # Runtime state (not serialized to worker - tracked on worker)
    # These are for UI display only
    runtime_last_result: BoolProperty(default=False)
    runtime_fire_count: IntProperty(default=0)


# ══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION (Main Thread -> Worker)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_uid(item) -> str:
    """Ensure item has a UID, generate if missing."""
    if not item.uid:
        item.uid = str(uuid.uuid4())
    return item.uid


def serialize_condition(cond: TrackerConditionPG) -> Dict[str, Any]:
    """Serialize a single condition for worker."""
    _ensure_uid(cond)

    data = {
        'uid': cond.uid,
        'type': cond.condition_type,
    }

    if cond.condition_type == 'DISTANCE':
        data['object_a'] = cond.object_a.name if cond.object_a else ""
        data['object_b'] = cond.object_b.name if cond.object_b else ""
        data['op'] = cond.distance_op
        data['value'] = cond.distance_value

    elif cond.condition_type == 'CONTACT':
        data['object'] = cond.contact_object.name if cond.contact_object else ""
        if cond.use_collection and cond.contact_collection:
            data['targets'] = [obj.name for obj in cond.contact_collection.objects]
        else:
            data['targets'] = [cond.contact_target.name] if cond.contact_target else []

    elif cond.condition_type == 'CHAR_STATE':
        data['state'] = cond.char_state
        data['equals'] = cond.char_state_equals

    elif cond.condition_type == 'PROPERTY':
        data['object'] = cond.prop_object.name if cond.prop_object else ""
        data['property'] = cond.prop_name
        data['op'] = cond.prop_op
        data['value'] = cond.prop_value

    elif cond.condition_type == 'COMPARE':
        data['op'] = cond.compare_op
        # Values come from node socket connections at runtime

    return data


def serialize_tracker(tracker: TrackerDefinitionPG) -> Dict[str, Any]:
    """Serialize complete tracker for worker."""
    _ensure_uid(tracker)

    return {
        'uid': tracker.uid,
        'name': tracker.name,
        'enabled': tracker.enabled,
        'conditions': [serialize_condition(c) for c in tracker.conditions],
        'logic': tracker.logic_mode,
        'fire_mode': tracker.fire_mode,
        'cooldown': tracker.cooldown,
        'max_fires': tracker.max_fires,
        'eval_hz': tracker.eval_hz,
        # Worker runtime state (initialized)
        '_last_result': False,
        '_fire_count': 0,
        '_cooldown_timer': 0.0,
        '_eval_timer': 0.0,
    }


def serialize_all_trackers(scene) -> List[Dict[str, Any]]:
    """Serialize all enabled trackers for worker."""
    trackers = getattr(scene, 'trackers', None)
    if not trackers:
        return []
    return [serialize_tracker(t) for t in trackers if t.enabled]


# ══════════════════════════════════════════════════════════════════════════════
# WORKER-SIDE EVALUATION (Called from worker process)
# ══════════════════════════════════════════════════════════════════════════════

def _compare(a: float, op: str, b: float) -> bool:
    """Compare two values with operator."""
    if op == 'LT': return a < b
    if op == 'LE': return a <= b
    if op == 'EQ': return abs(a - b) < 0.0001
    if op == 'NE': return abs(a - b) >= 0.0001
    if op == 'GE': return a >= b
    if op == 'GT': return a > b
    return False


def eval_condition_worker(cond: Dict, world_state: Dict) -> bool:
    """
    Evaluate a single condition on the worker.

    Args:
        cond: Serialized condition dict
        world_state: Current world state from worker
            - 'positions': {obj_name: (x, y, z)}
            - 'char_state': current character state string
            - 'contacts': {obj_name: [touching_obj_names]}
            - 'properties': {obj_name: {prop_name: value}}
            - 'inputs': {action_name: bool} - current input states

    Returns:
        Boolean result of condition
    """
    ctype = cond.get('type', '')

    if ctype == 'DISTANCE':
        obj_a = cond.get('object_a', '')
        obj_b = cond.get('object_b', '')
        positions = world_state.get('positions', {})

        pos_a = positions.get(obj_a)
        pos_b = positions.get(obj_b)

        if pos_a is None or pos_b is None:
            return False

        # Calculate distance
        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        dz = pos_a[2] - pos_b[2]
        dist = (dx*dx + dy*dy + dz*dz) ** 0.5

        return _compare(dist, cond.get('op', 'LT'), cond.get('value', 5.0))

    elif ctype == 'CONTACT':
        obj = cond.get('object', '')
        targets = cond.get('targets', [])
        contacts = world_state.get('contacts', {})

        touching = contacts.get(obj, [])
        for target in targets:
            if target in touching:
                return True
        return False

    elif ctype == 'CHAR_STATE':
        state = cond.get('state', '')
        equals = cond.get('equals', True)
        current = world_state.get('char_state', '')

        is_match = (current == state)
        return is_match if equals else not is_match

    elif ctype == 'PROPERTY':
        obj = cond.get('object', '')
        prop = cond.get('property', '')
        props = world_state.get('properties', {})

        obj_props = props.get(obj, {})
        value = obj_props.get(prop, 0.0)

        return _compare(value, cond.get('op', 'EQ'), cond.get('value', 0.0))

    elif ctype == 'COMPARE':
        # Values injected at runtime from node connections
        val_a = cond.get('_value_a', 0.0)
        val_b = cond.get('_value_b', 0.0)
        return _compare(val_a, cond.get('op', 'EQ'), val_b)

    elif ctype == 'INPUT':
        # Check if an input action is pressed/held
        action = cond.get('action', '')
        is_pressed = cond.get('is_pressed', True)
        inputs = world_state.get('inputs', {})

        current = inputs.get(action, False)
        return current if is_pressed else not current

    elif ctype == 'GAME_TIME':
        # Check elapsed game time against threshold
        game_time = world_state.get('game_time', 0.0)
        compare_enabled = cond.get('compare_enabled', True)

        if not compare_enabled:
            return True  # No comparison, always true (use float output)

        return _compare(game_time, cond.get('op', 'GE'), cond.get('value', 10.0))

    return False


def eval_tracker_worker(tracker: Dict, world_state: Dict, dt: float) -> Optional[str]:
    """
    Evaluate a tracker on the worker. Returns tracker UID if should fire.

    Args:
        tracker: Serialized tracker dict (with runtime state)
        world_state: Current world state
        dt: Delta time since last eval

    Returns:
        Tracker UID if should fire, None otherwise
    """
    if not tracker.get('enabled', True):
        return None

    # Update eval timer
    tracker['_eval_timer'] = tracker.get('_eval_timer', 0.0) + dt
    eval_interval = 1.0 / tracker.get('eval_hz', 10)

    if tracker['_eval_timer'] < eval_interval:
        return None  # Not time to check yet
    tracker['_eval_timer'] = 0.0

    # Update cooldown timer
    if tracker.get('_cooldown_timer', 0.0) > 0:
        tracker['_cooldown_timer'] -= eval_interval

    # Evaluate all conditions
    conditions = tracker.get('conditions', [])
    if not conditions:
        return None

    logic = tracker.get('logic', 'AND')

    if logic == 'AND':
        result = all(eval_condition_worker(c, world_state) for c in conditions)
    else:  # OR
        result = any(eval_condition_worker(c, world_state) for c in conditions)

    # Edge detection
    prev = tracker.get('_last_result', False)
    tracker['_last_result'] = result

    fire_mode = tracker.get('fire_mode', 'ON_BECOME_TRUE')
    should_fire = False

    if fire_mode == 'ON_BECOME_TRUE':
        should_fire = result and not prev
    elif fire_mode == 'ON_BECOME_FALSE':
        should_fire = not result and prev
    elif fire_mode == 'WHILE_TRUE':
        should_fire = result
    elif fire_mode == 'ON_CHANGE':
        should_fire = result != prev

    if not should_fire:
        return None

    # Check cooldown
    if tracker.get('_cooldown_timer', 0.0) > 0:
        return None

    # Check max fires
    max_fires = tracker.get('max_fires', 0)
    fire_count = tracker.get('_fire_count', 0)
    if max_fires > 0 and fire_count >= max_fires:
        return None

    # Fire!
    tracker['_fire_count'] = fire_count + 1
    tracker['_cooldown_timer'] = tracker.get('cooldown', 0.0)

    return tracker.get('uid')


def eval_all_trackers_worker(trackers: List[Dict], world_state: Dict, dt: float) -> List[str]:
    """
    Evaluate all trackers on worker. Returns list of tracker UIDs to fire.

    This is the main entry point called from the worker's tick loop.
    """
    fired = []
    for tracker in trackers:
        uid = eval_tracker_worker(tracker, world_state, dt)
        if uid:
            fired.append(uid)
    return fired


# ══════════════════════════════════════════════════════════════════════════════
# MAIN THREAD DISPATCH (Receives fire messages from worker)
# ══════════════════════════════════════════════════════════════════════════════

_tracker_callbacks: Dict[str, callable] = {}


def register_tracker_callback(tracker_uid: str, callback: callable):
    """Register a callback for when a tracker fires."""
    _tracker_callbacks[tracker_uid] = callback


def unregister_tracker_callback(tracker_uid: str):
    """Unregister a tracker callback."""
    _tracker_callbacks.pop(tracker_uid, None)


def clear_tracker_callbacks():
    """Clear all tracker callbacks."""
    _tracker_callbacks.clear()


def dispatch_tracker_fires(fired_uids: List[str]):
    """
    Called on main thread when worker reports tracker fires.

    Args:
        fired_uids: List of tracker UIDs that fired
    """
    from ..developer.dev_logger import log_game

    for uid in fired_uids:
        callback = _tracker_callbacks.get(uid)
        if callback:
            try:
                log_game("TRACKERS", f"FIRE uid={uid[:8]}...")
                callback()
            except Exception as e:
                log_game("TRACKERS", f"ERROR uid={uid[:8]} err={e}")


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRATION
# ══════════════════════════════════════════════════════════════════════════════

_CLASSES = [
    TrackerConditionPG,
    TrackerDefinitionPG,
]


def register_tracker_properties():
    """Register tracker property groups."""
    for cls in _CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Scene.trackers = CollectionProperty(type=TrackerDefinitionPG)
    bpy.types.Scene.active_tracker_index = IntProperty(name="Active Tracker", default=0)


def unregister_tracker_properties():
    """Unregister tracker property groups."""
    if hasattr(bpy.types.Scene, 'trackers'):
        del bpy.types.Scene.trackers
    if hasattr(bpy.types.Scene, 'active_tracker_index'):
        del bpy.types.Scene.active_tracker_index

    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
