# Exp_Game/engine/worker/interactions/__init__.py
"""
Interaction trigger evaluation - runs in worker process (NO bpy).

Handles:
- PROXIMITY distance checks
- COLLISION AABB overlap
- Tracker condition tree evaluation (for EXTERNAL triggers)
"""

from .triggers import handle_interaction_check_batch
from .trackers import (
    handle_cache_trackers,
    handle_evaluate_trackers,
    reset_tracker_state,
)

__all__ = [
    'handle_interaction_check_batch',
    'handle_cache_trackers',
    'handle_evaluate_trackers',
    'reset_tracker_state',
]
