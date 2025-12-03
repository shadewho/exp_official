"""
Developer Logger - Fast Memory Buffer Logging System

Replaces slow console prints with fast in-memory logging.
Performance: ~1μs per log call (vs ~1000μs+ for console print)
Export: Batch write to file when game stops.

Usage:
    from Exp_Game.developer.dev_logger import log_game, export_game_log, clear_log

    # During gameplay (fast - just appends to buffer)
    log_game("KCC", "APPLY pos=(10,5,3) ground=True")
    log_game("CAMERA", "Raycast hit dist=2.5m")

    # When game stops
    export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
    clear_log()  # Prepare for next session
"""

import time
from typing import List, Dict, Optional
from .dev_debug_gate import should_print_debug

# Global log buffer - accumulates all logs during gameplay
_log_buffer: List[Dict] = []

# Session tracking
_session_start_time: Optional[float] = None
_current_frame: int = 0

# Performance stats
_total_logs: int = 0
_logs_per_category: Dict[str, int] = {}

# Category name mapping (log category -> debug property name)
_CATEGORY_MAP = {
    'KCC': 'kcc_offload',
    'CAMERA': 'camera_offload',
    'DYNAMIC': 'dynamic_offload',
    'FRAME': 'frame_numbers',
    'PHYS-TIMING': 'physics_timing',
    'PHYS-CATCHUP': 'physics_catchup',
    'PHYS-PLATFORM': 'physics_platform',
    'PHYS-CONSISTENCY': 'physics_consistency',
    'PHYS-CAPSULE': 'physics_capsule',
    'PHYS-DEPENETRATION': 'physics_depenetration',
    'PHYS-GROUND': 'physics_ground',
    'PHYS-STEP': 'physics_step_up',
    'PHYS-SLOPES': 'physics_slopes',
    'PHYS-SLIDE': 'physics_slide',
    'PHYS-VERTICAL': 'physics_vertical',
    'PHYS-ENHANCED': 'physics_enhanced',
    'CULLING': 'performance',
    'INTERACTIONS': 'interactions',
    'AUDIO': 'audio',
    'ANIMATIONS': 'animations',
}


def start_session():
    """Call when game starts - resets session tracking."""
    global _session_start_time, _current_frame, _total_logs, _logs_per_category, _log_buffer
    _session_start_time = time.perf_counter()
    _current_frame = 0
    _total_logs = 0
    _logs_per_category.clear()
    _log_buffer.clear()  # Clear buffer from previous session

    # Log session start marker
    print("[DevLogger] Session started - fast logging active")


def increment_frame():
    """Call once per frame to track frame numbers."""
    global _current_frame
    _current_frame += 1


def log_game(category: str, message: str):
    """
    Fast in-memory logging with frequency gating. Zero I/O during gameplay.

    Args:
        category: Log category (e.g., "KCC", "CAMERA", "CULLING")
        message: The log message

    Performance: ~1μs (just list append + dict creation) + frequency check
    """
    global _log_buffer, _total_logs, _logs_per_category

    # Map category to debug property name and check frequency gate
    debug_property = _CATEGORY_MAP.get(category)
    if debug_property:
        # Check if this category should log based on master Hz setting
        if not should_print_debug(debug_property):
            return  # Skip logging if frequency gate blocks it
    # If category not in map, log anyway (for backwards compatibility)

    # Fast: just append to list (no I/O, no formatting, no console)
    _log_buffer.append({
        'frame': _current_frame,
        'time': time.perf_counter(),
        'category': category,
        'message': message
    })

    # Track stats
    _total_logs += 1
    _logs_per_category[category] = _logs_per_category.get(category, 0) + 1


def log_worker_messages(worker_logs: list):
    """
    Log messages from worker processes with frequency gating.

    Worker processes can't share the main buffer, so they collect logs
    during computation and return them in the engine result. The main
    thread then calls this function to add them to the buffer.

    Args:
        worker_logs: List of (category, message) tuples from worker
    """
    global _log_buffer, _total_logs, _logs_per_category

    for category, message in worker_logs:
        # Apply frequency gating (same as log_game)
        debug_property = _CATEGORY_MAP.get(category)
        if debug_property:
            if not should_print_debug(debug_property):
                continue  # Skip logging if frequency gate blocks it
        # If category not in map, log anyway (for backwards compatibility)

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

    Args:
        filepath: Absolute path to output file

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
            f.write(f"GAME DIAGNOSTICS LOG\n")
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

        print(f"[DevLogger] ✓ Exported {_total_logs} logs ({len(_log_buffer)} entries) to: {filepath}")
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
