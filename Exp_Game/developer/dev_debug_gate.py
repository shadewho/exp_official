# Exp_Game/developer/dev_debug_gate.py
"""
Debug output frequency gating system.

Prevents console spam by limiting debug output to specified Hz (1-30).

IMPORTANT: Gating is per MESSAGE KEY, not per category. This ensures that
different log types within the same category (e.g., BATCH_SUBMIT vs BATCH_RESULT)
don't block each other.
"""

import time
import bpy

# Track last print time for each unique message key
# Key format: "category" for legacy calls, "category:prefix" for log_game calls
_last_print_times = {}


def should_print_debug(category: str, message_key: str = None) -> bool:
    """
    Check if debug output should be printed based on master frequency gate.

    Args:
        category: Debug category name (matches scene property suffix, e.g., "engine", "kcc_physics")
                  Property is built as: dev_debug_{category}
        message_key: Optional unique key for this message type. If provided, gating is
                     per message_key instead of per category. This prevents different
                     log types from blocking each other.

    Returns:
        True if enough time has passed since last print, False otherwise
    """
    scene = bpy.context.scene

    # Get the enabled flag for this category
    debug_enabled_prop = f"dev_debug_{category}"

    # Check if debug is enabled for this category
    if not hasattr(scene, debug_enabled_prop) or not getattr(scene, debug_enabled_prop):
        return False

    # Get master frequency (default to 30Hz if property doesn't exist)
    if not hasattr(scene, 'dev_debug_master_hz'):
        frequency = 30
    else:
        frequency = getattr(scene, 'dev_debug_master_hz')

    # Special case: 30Hz = every frame (no gating)
    if frequency >= 30:
        return True

    # Calculate time threshold
    time_threshold = 1.0 / frequency

    # Use message_key if provided, otherwise fall back to category
    gate_key = message_key if message_key else category

    # Check if enough time has passed
    current_time = time.perf_counter()
    last_time = _last_print_times.get(gate_key, 0.0)

    if (current_time - last_time) >= time_threshold:
        _last_print_times[gate_key] = current_time
        return True

    return False


def reset_debug_timers():
    """Reset all debug print timers (called on game start/end)."""
    global _last_print_times
    _last_print_times.clear()