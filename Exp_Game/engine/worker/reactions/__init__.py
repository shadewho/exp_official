# Exp_Game/engine/worker/reactions/__init__.py
"""
Reaction execution - runs in worker process (NO bpy).

Handles:
- Projectile physics simulation (gravity + sweep raycast)
- Hitscan instant raycasting
"""

from .projectiles import (
    handle_projectile_update_batch,
    reset_projectile_state,
)
from .hitscan import handle_hitscan_batch

__all__ = [
    'handle_projectile_update_batch',
    'reset_projectile_state',
    'handle_hitscan_batch',
]
