# Exp_Game/props_and_utils/trackers.py
"""
Tracker System Constants & Utilities

The actual tracker evaluation is handled by:
- Serialization: exp_tracker_eval.py (node graph -> flat data)
- Evaluation: engine/worker/interactions/trackers.py (worker-side)

This file provides shared constants and callback utilities.
"""

from typing import Dict, List

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS (Used by tracker_nodes.py)
# ══════════════════════════════════════════════════════════════════════════════

COMPARE_OPERATORS = [
    ('LT', "<", "Less than"),
    ('LE', "<=", "Less than or equal"),
    ('EQ', "==", "Equal"),
    ('NE', "!=", "Not equal"),
    ('GE', ">=", "Greater than or equal"),
    ('GT', ">", "Greater than"),
]

EQUALITY_ONLY_OPERATORS = [
    ('EQ', "==", "Equal"),
    ('NE', "!=", "Not equal"),
]

CHAR_STATES = [
    ('GROUNDED', "Grounded", "Character is on ground"),
    ('AIRBORNE', "Airborne", "Character is in air"),
    ('IDLE', "Idle", "Character is idle"),
    ('WALKING', "Walking", "Character is walking"),
    ('RUNNING', "Running", "Character is running"),
    ('JUMPING', "Jumping", "Character is jumping"),
    ('FALLING', "Falling", "Character is falling"),
]

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
# CALLBACK DISPATCH (Main thread receives fire messages from worker)
# ══════════════════════════════════════════════════════════════════════════════

_tracker_callbacks: Dict[str, callable] = {}


def register_tracker_callback(tracker_uid: str, callback: callable):
    """Register a callback for when a tracker fires."""
    _tracker_callbacks[tracker_uid] = callback


def unregister_tracker_callback(tracker_uid: str):
    """Unregister a tracker callback."""
    _tracker_callbacks.pop(tracker_uid, None)


def clear_tracker_callbacks():
    """Clear all tracker callbacks. Called on game end."""
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
