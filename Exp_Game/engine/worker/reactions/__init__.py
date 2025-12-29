# Exp_Game/engine/worker/reactions/__init__.py
"""
Reaction execution - runs in worker process (NO bpy).

Handles:
- Projectile physics simulation (gravity + sweep raycast)
- Hitscan instant raycasting
- Transform interpolation (lerp/slerp)
- Tracking movement (sweep/slide/gravity for object movement)
"""

from .projectiles import (
    handle_projectile_update_batch,
    reset_projectile_state,
)
from .hitscan import handle_hitscan_batch
from .transforms import handle_transform_batch
from .tracking import handle_tracking_batch
from .ragdoll import handle_ragdoll_update_batch

__all__ = [
    'handle_projectile_update_batch',
    'reset_projectile_state',
    'handle_hitscan_batch',
    'handle_transform_batch',
    'handle_tracking_batch',
    'handle_ragdoll_update_batch',
]
