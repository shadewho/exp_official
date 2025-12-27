"""
Developer Logger - Fast Memory Buffer Logging System

UNIFIED PHYSICS ARCHITECTURE:
All physics logs show source (static/dynamic) in a consistent format.
Static and dynamic use identical code paths - there is ONE physics system.

Performance: ~1μs per log call (vs ~1000μs+ for console print)
Export: Batch write to file when game stops.

Usage:
    from Exp_Game.developer.dev_logger import log_game, export_game_log, clear_log

    # During gameplay (fast - just appends to buffer)
    log_game("GROUND", "source=static z=5.0 normal=(0,0,1)")
    log_game("GROUND", "source=dynamic_12345 z=5.5 normal=(0,0,1)")

    # When game stops
    export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
    clear_log()
"""

import time
from typing import List, Dict, Optional
from .dev_debug_gate import should_print_debug

# Global log buffer
_log_buffer: List[Dict] = []

# Session tracking
_session_start_time: Optional[float] = None
_current_frame: int = 0

# Performance stats
_total_logs: int = 0
_logs_per_category: Dict[str, int] = {}

# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY MAPPING (log category -> debug property name)
# ══════════════════════════════════════════════════════════════════════════════
# Property names match dev_properties.py: dev_debug_{property_name}
#
# UNIFIED PHYSICS: All physics categories show source (static/dynamic).
# There is NO separation between static and dynamic physics code.

_CATEGORY_MAP = {
    # Engine
    'ENGINE': 'engine',

    # Offload Systems
    'KCC': 'kcc_physics',
    'CAMERA': 'camera',
    'FRAME': 'frame_numbers',
    'CULLING': 'culling',

    # Spatial Grid System
    'GRID': 'grid',                       # Grid build and raycast stats

    # Unified Physics (all show source: static/dynamic)
    'PHYSICS': 'physics',                 # Summary
    'GROUND': 'physics_ground',           # Ground detection
    'HORIZONTAL': 'physics_horizontal',   # Horizontal collision
    'BODY': 'physics_body',               # Body integrity
    'CEILING': 'physics_ceiling',         # Ceiling check
    'STEP': 'physics_step',               # Step-up
    'SLIDE': 'physics_slide',             # Wall slide
    'SLOPES': 'physics_slopes',           # Slope handling

    # Dynamic Mesh System (unified with static)
    'DYN-CACHE': 'dynamic_cache',
    'PLATFORM': 'dynamic_cache',  # Platform attach/detach (relative position system)
    'DYN-OPT': 'dynamic_opt',     # Dynamic mesh optimization stats

    # Game Systems
    'INTERACTIONS': 'interactions',
    'AUDIO': 'audio',
    'ANIMATIONS': 'animations',
    'ANIM-CACHE': 'anim_cache',
    'ANIM-WORKER': 'anim_worker',
    'TEST_MODAL': 'animations',  # Animation 2.0 test modal
    'PROJECTILE': 'projectiles',
    'HITSCAN': 'hitscans',

    # Unified IK System (formerly separate full-body and runtime IK)
    'IK': 'ik',                      # Main IK logs (unified system)
    'IK-SOLVE': 'ik_solve',          # Detailed solver math (step-by-step)
    'FULL-BODY-IK': 'ik',            # Legacy alias → main IK logs

    # Rig State Logging (bone transforms, violations, collisions)
    'RIG': 'rig_state',

    # Pose Library System
    'POSES': 'poses',
    'POSE-CACHE': 'poses',
    'POSE-BLEND': 'pose_blend',  # Pose-to-pose blending diagnostics

    # Tracker System
    'TRACKERS': 'trackers',

    # World State Optimization (Phase 1.1)
    'WORLD-STATE': 'world_state',

    # AABB Cache Optimization (Phase 1.2)
    'AABB-CACHE': 'aabb_cache',
}


def start_session():
    """Call when game starts - resets session tracking."""
    global _session_start_time, _current_frame, _total_logs, _logs_per_category, _log_buffer
    _session_start_time = time.perf_counter()
    _current_frame = 0
    _total_logs = 0
    _logs_per_category.clear()
    _log_buffer.clear()

    print("[DevLogger] Session started - fast logging active")


def increment_frame():
    """Call once per frame to track frame numbers."""
    global _current_frame
    _current_frame += 1


def _extract_message_key(category: str, message: str) -> str:
    """
    Extract a unique key for frequency gating from a log message.

    The key is: "category:PREFIX" where PREFIX is the first word of the message.
    This ensures different log types (e.g., BATCH_SUBMIT vs BATCH_RESULT) have
    separate frequency gates and don't block each other.

    Examples:
        ("ANIMATIONS", "BATCH_SUBMIT job=123") -> "animations:BATCH_SUBMIT"
        ("ANIMATIONS", "BATCH_RESULT job=123") -> "animations:BATCH_RESULT"
        ("KCC", "GROUND pos=(1,2,3)") -> "kcc_physics:GROUND"
    """
    # Get first word (up to first space or end of string)
    first_space = message.find(' ')
    if first_space > 0:
        prefix = message[:first_space]
    else:
        prefix = message[:20] if len(message) > 20 else message

    # Build unique key
    debug_property = _CATEGORY_MAP.get(category, category.lower())
    return f"{debug_property}:{prefix}"


def log_game(category: str, message: str):
    """
    Fast in-memory logging with frequency gating. Zero I/O during gameplay.

    Args:
        category: Log category (e.g., "GROUND", "HORIZONTAL", "BODY")
        message: The log message (should include source=static/dynamic for physics)

    Performance: ~1μs (just list append + dict creation) + frequency check

    Note: Frequency gating is per MESSAGE TYPE, not per category. This ensures
    that different log types (e.g., BATCH_SUBMIT vs BATCH_RESULT) don't block
    each other even when they occur in the same frame.
    """
    global _log_buffer, _total_logs, _logs_per_category

    # Map category to debug property and check frequency gate
    debug_property = _CATEGORY_MAP.get(category)
    if debug_property:
        # Extract message key for per-type gating
        message_key = _extract_message_key(category, message)
        if not should_print_debug(debug_property, message_key):
            return  # Frequency gate blocked

    # Fast: just append to list (no I/O)
    _log_buffer.append({
        'frame': _current_frame,
        'time': time.perf_counter(),
        'category': category,
        'message': message
    })

    _total_logs += 1
    _logs_per_category[category] = _logs_per_category.get(category, 0) + 1


def log_worker_messages(worker_logs: list):
    """
    Log messages from worker processes with frequency gating.

    Workers collect logs during computation and return them in the result.
    Main thread calls this to add them to the buffer.

    Args:
        worker_logs: List of (category, message) tuples from worker

    Note: Uses per-message-type gating (same as log_game) to prevent different
    log types from blocking each other.
    """
    global _log_buffer, _total_logs, _logs_per_category

    for category, message in worker_logs:
        debug_property = _CATEGORY_MAP.get(category)
        if debug_property:
            # Extract message key for per-type gating
            message_key = _extract_message_key(category, message)
            if not should_print_debug(debug_property, message_key):
                continue  # Frequency gate blocked

        _log_buffer.append({
            'frame': _current_frame,
            'time': time.perf_counter(),
            'category': category,
            'message': message
        })
        _total_logs += 1
        _logs_per_category[category] = _logs_per_category.get(category, 0) + 1


def export_game_log(filepath: str) -> bool:
    """
    Write entire buffer to file. Call when game stops.

    Returns:
        True if successful, False if error
    """
    global _log_buffer, _session_start_time

    if not _log_buffer:
        print("[DevLogger] No logs to export (buffer empty)")
        return False

    try:
        start_time = _session_start_time if _session_start_time else _log_buffer[0]['time']

        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("GAME DIAGNOSTICS LOG (UNIFIED PHYSICS)\n")
            f.write(f"Total Logs: {_total_logs}\n")
            f.write(f"Total Frames: {_current_frame}\n")
            f.write(f"Duration: {_log_buffer[-1]['time'] - start_time:.3f}s\n")
            f.write("=" * 80 + "\n\n")

            # Category breakdown
            f.write("Logs per Category:\n")
            for cat, count in sorted(_logs_per_category.items()):
                f.write(f"  {cat}: {count}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            # Log entries
            for entry in _log_buffer:
                elapsed = entry['time'] - start_time
                f.write(f"[{entry['category']} F{entry['frame']:04d} T{elapsed:.3f}s] {entry['message']}\n")

            # Footer
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"END OF LOG - {_total_logs} entries\n")
            f.write("=" * 80 + "\n")

        print(f"[DevLogger] ✓ Exported {_total_logs} logs to: {filepath}")
        return True

    except Exception as e:
        print(f"[DevLogger] ERROR exporting log: {e}")
        return False


def clear_log():
    """Clear the buffer. Call after export to prepare for next session."""
    global _log_buffer
    _log_buffer.clear()


def get_buffer_size() -> int:
    """Get current number of entries in buffer."""
    return len(_log_buffer)


def get_memory_usage_mb() -> float:
    """Estimate memory usage of buffer in MB."""
    import sys
    return sys.getsizeof(_log_buffer) / (1024 * 1024)


def get_stats() -> Dict:
    """Get current session statistics."""
    return {
        'total_logs': _total_logs,
        'buffer_size': len(_log_buffer),
        'current_frame': _current_frame,
        'categories': dict(_logs_per_category),
        'memory_mb': get_memory_usage_mb()
    }