# Exp_Game/systems/exp_health.py
"""
Health System - Track health for any object in the scene.

Architecture:
    1. ENABLE_HEALTH reaction fires → enable_health() adds object to cache
    2. Runtime: modify_health() / set_health() change values (future damage reactions)
    3. Game Reset: reset_all_health() restores start values
    4. Game End: clear_all_health() empties cache

The cache uses object names (strings) as keys for worker-serializable data.
Future damage calculations can be offloaded to worker processes.
"""

from ..developer.dev_logger import log_game

# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CACHE
# ══════════════════════════════════════════════════════════════════════════════
# Structure: {object_name: {"current": float, "start": float, "min": float, "max": float}}

_health_cache: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# ENABLE / DISABLE
# ══════════════════════════════════════════════════════════════════════════════

def enable_health(obj_name: str, start: float, min_val: float, max_val: float):
    """
    Enable health tracking on an object.
    Called by ENABLE_HEALTH reaction.

    Args:
        obj_name: Name of the object (for serializable lookup)
        start: Initial health value (also used on reset)
        min_val: Minimum health (usually 0)
        max_val: Maximum health
    """
    _health_cache[obj_name] = {
        "current": start,
        "start": start,
        "min": min_val,
        "max": max_val,
    }
    log_game("HEALTH", f"ENABLE obj={obj_name} start={start} range=[{min_val}, {max_val}]")


def disable_health(obj_name: str) -> bool:
    """
    Remove health tracking from an object.
    Returns True if object was being tracked.
    """
    if obj_name in _health_cache:
        del _health_cache[obj_name]
        log_game("HEALTH", f"DISABLE obj={obj_name}")
        return True
    return False


def is_health_enabled(obj_name: str) -> bool:
    """Check if an object has health tracking enabled."""
    return obj_name in _health_cache


# ══════════════════════════════════════════════════════════════════════════════
# GET / SET / MODIFY
# ══════════════════════════════════════════════════════════════════════════════

def get_health(obj_name: str) -> dict | None:
    """
    Get health data for an object.
    Returns dict with current, start, min, max - or None if not enabled.
    """
    return _health_cache.get(obj_name)


def get_current_health(obj_name: str) -> float | None:
    """Get just the current health value. Returns None if not enabled."""
    data = _health_cache.get(obj_name)
    return data["current"] if data else None


def set_health(obj_name: str, value: float) -> float | None:
    """
    Set health to a specific value (clamped to min/max).
    Returns the new health value, or None if object not tracked.
    """
    data = _health_cache.get(obj_name)
    if not data:
        return None

    old = data["current"]
    data["current"] = max(data["min"], min(data["max"], value))
    log_game("HEALTH", f"SET obj={obj_name} {old:.1f} -> {data['current']:.1f}")
    return data["current"]


def modify_health(obj_name: str, delta: float) -> float | None:
    """
    Add or subtract from health (clamped to min/max).
    Use positive delta for healing, negative for damage.
    Returns the new health value, or None if object not tracked.
    """
    data = _health_cache.get(obj_name)
    if not data:
        return None

    old = data["current"]
    data["current"] = max(data["min"], min(data["max"], old + delta))
    log_game("HEALTH", f"MODIFY obj={obj_name} {old:.1f} -> {data['current']:.1f} (delta={delta:+.1f})")
    return data["current"]


# ══════════════════════════════════════════════════════════════════════════════
# RESET / CLEAR (Lifecycle)
# ══════════════════════════════════════════════════════════════════════════════

def reset_all_health():
    """
    Reset all health values to their start values.
    Called on game reset (R key). Objects remain tracked.
    """
    count = len(_health_cache)
    for data in _health_cache.values():
        data["current"] = data["start"]
    log_game("HEALTH", f"RESET {count} objects restored to start values")


def clear_all_health():
    """
    Clear the entire health cache.
    Called on game start (invoke) and game end (cancel).
    """
    count = len(_health_cache)
    _health_cache.clear()
    log_game("HEALTH", f"CLEAR cache emptied (was tracking {count} objects)")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY / STATS
# ══════════════════════════════════════════════════════════════════════════════

def get_health_stats() -> dict:
    """Get summary statistics for debugging."""
    return {
        "count": len(_health_cache),
        "objects": list(_health_cache.keys()),
    }


def get_all_health() -> dict:
    """Get a copy of the entire health cache (for debugging/serialization)."""
    return dict(_health_cache)


def serialize_for_worker() -> dict:
    """
    Serialize health data for worker process.
    Future: Used when offloading damage calculations to worker.
    """
    return {
        name: {
            "current": data["current"],
            "start": data["start"],
            "min": data["min"],
            "max": data["max"],
        }
        for name, data in _health_cache.items()
    }
