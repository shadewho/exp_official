# Exp_Game/animations/layer_manager.py
"""
Animation Layer Manager - Per-Object Priority System

Every object with actions gets its own layer manager.
Channels determine which animation source controls each bone.

Priority (highest wins):
- PHYSICS (2): Ragdoll, rigid body, cloth sim
- OVERRIDE (1): Triggered actions, reactions
- BASE (0): Default animation (locomotion, idle, custom action)

Usage:
    # Get or create manager for any object
    mgr = get_layer_manager_for(obj)

    # Activate a channel
    mgr.activate_channel(AnimChannel.PHYSICS, influence=1.0)

    # Deactivate with fade
    mgr.deactivate_channel(AnimChannel.PHYSICS, fade_out=0.3)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional, Any

from .bone_groups import TOTAL_BONES, INDEX_TO_BONE
from ..developer.dev_logger import log_game


# =============================================================================
# CHANNEL DEFINITIONS
# =============================================================================

class AnimChannel(IntEnum):
    """
    Animation channels in priority order.
    Higher value = higher priority = wins bone ownership.
    """
    BASE = 0       # Default animation (locomotion, custom action)
    OVERRIDE = 1   # Triggered animations (reactions, one-shots)
    PHYSICS = 2    # Physics simulation (ragdoll, rigid body, cloth)
    # Future: IK = 3, PROCEDURAL = 4


# =============================================================================
# CHANNEL STATE
# =============================================================================

@dataclass
class ChannelState:
    """State for a single animation channel."""
    channel: AnimChannel
    active: bool = False
    influence: float = 0.0
    target_influence: float = 0.0
    fade_rate: float = 5.0
    fading: bool = False


# =============================================================================
# LAYER MANAGER (Per-Object)
# =============================================================================

class LayerManager:
    """
    Animation layer manager for a single object.

    Tracks which channel has control and handles smooth transitions.
    """

    def __init__(self, obj_name: str):
        self.obj_name = obj_name

        # Channel states
        self._channels: Dict[AnimChannel, ChannelState] = {
            ch: ChannelState(channel=ch)
            for ch in AnimChannel
        }

        # BASE is always active by default
        self._channels[AnimChannel.BASE].active = True
        self._channels[AnimChannel.BASE].influence = 1.0
        self._channels[AnimChannel.BASE].target_influence = 1.0

        # Stored action for restore on game end
        self._stored_action: Any = None

        _log(f"LayerManager created for {obj_name}")

    # =========================================================================
    # CHANNEL CONTROL
    # =========================================================================

    def activate_channel(
        self,
        channel: AnimChannel,
        influence: float = 1.0,
        fade_in: float = 0.0,
    ) -> None:
        """
        Activate a channel.

        Args:
            channel: Which channel to activate
            influence: Target influence (0.0-1.0)
            fade_in: Seconds to fade in (0 = instant)
        """
        state = self._channels[channel]
        state.active = True
        state.target_influence = influence

        if fade_in > 0.0:
            state.fade_rate = 1.0 / fade_in
            state.fading = True
        else:
            state.influence = influence
            state.fading = False

        _log(f"{self.obj_name}: {channel.name} activated, influence={influence}, fade={fade_in}")

    def deactivate_channel(
        self,
        channel: AnimChannel,
        fade_out: float = 0.0,
    ) -> None:
        """
        Deactivate a channel.

        Args:
            channel: Which channel to deactivate
            fade_out: Seconds to fade out (0 = instant)
        """
        state = self._channels[channel]

        if fade_out > 0.0:
            state.target_influence = 0.0
            state.fade_rate = 1.0 / fade_out
            state.fading = True
        else:
            state.active = False
            state.influence = 0.0
            state.target_influence = 0.0
            state.fading = False

        _log(f"{self.obj_name}: {channel.name} deactivating, fade={fade_out}")

    def is_channel_active(self, channel: AnimChannel) -> bool:
        """Check if channel is active."""
        return self._channels[channel].active

    def get_channel_influence(self, channel: AnimChannel) -> float:
        """Get current influence of a channel."""
        return self._channels[channel].influence

    def get_active_channel(self) -> AnimChannel:
        """Get the highest priority active channel."""
        for ch in reversed(AnimChannel):  # Highest priority first
            if self._channels[ch].active and self._channels[ch].influence > 0.01:
                return ch
        return AnimChannel.BASE

    # =========================================================================
    # UPDATE
    # =========================================================================

    def update(self, dt: float) -> None:
        """Update channel influence fading. Call once per frame."""
        for channel, state in self._channels.items():
            if not state.fading:
                continue

            # Move toward target
            diff = state.target_influence - state.influence
            max_change = state.fade_rate * dt

            if abs(diff) <= max_change:
                state.influence = state.target_influence
                state.fading = False

                # Deactivate if faded to 0
                if state.target_influence < 0.01:
                    state.active = False
                    _log(f"{self.obj_name}: {channel.name} fade complete, deactivated")
            else:
                state.influence += max_change if diff > 0 else -max_change

    # =========================================================================
    # ACTION MANAGEMENT
    # =========================================================================

    def disable_native_action(self, obj) -> None:
        """
        Clear Blender's native action to prevent C-level evaluation.
        Store for restore on game end.
        """
        if obj and obj.animation_data:
            self._stored_action = obj.animation_data.action
            obj.animation_data.action = None
            _log(f"{self.obj_name}: Disabled native action")

    def restore_native_action(self, obj) -> None:
        """Restore the native action on game end."""
        if obj and obj.animation_data and self._stored_action:
            obj.animation_data.action = self._stored_action
            _log(f"{self.obj_name}: Restored native action")
        self._stored_action = None

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def clear(self) -> None:
        """Reset to default state."""
        for ch, state in self._channels.items():
            state.active = (ch == AnimChannel.BASE)
            state.influence = 1.0 if ch == AnimChannel.BASE else 0.0
            state.target_influence = state.influence
            state.fading = False
        self._stored_action = None


# =============================================================================
# OBJECT REGISTRY
# =============================================================================

# Registry of layer managers per object
_managers: Dict[str, LayerManager] = {}


def get_layer_manager_for(obj) -> Optional[LayerManager]:
    """
    Get or create a layer manager for an object.

    Args:
        obj: Blender object (armature or mesh with actions)

    Returns:
        LayerManager for this object, or None if invalid
    """
    if not obj:
        return None

    try:
        obj_name = obj.name
    except (ReferenceError, AttributeError):
        return None

    if obj_name not in _managers:
        _managers[obj_name] = LayerManager(obj_name)

    return _managers[obj_name]


def get_layer_manager(obj_name: str = None) -> Optional[LayerManager]:
    """
    Get layer manager by object name.

    Args:
        obj_name: Object name. If None, returns character's manager.

    Returns:
        LayerManager or None
    """
    if obj_name is None:
        # Legacy: return character manager if exists
        try:
            import bpy
            armature = getattr(bpy.context.scene, "target_armature", None)
            if armature:
                return _managers.get(armature.name)
        except:
            pass
        return None

    return _managers.get(obj_name)


def get_all_managers() -> Dict[str, LayerManager]:
    """Get all registered layer managers."""
    return _managers.copy()


def remove_layer_manager(obj_name: str) -> None:
    """Remove a layer manager from registry."""
    if obj_name in _managers:
        del _managers[obj_name]
        _log(f"Removed manager for {obj_name}")


# =============================================================================
# LIFECYCLE
# =============================================================================

def init_layer_managers() -> int:
    """
    Initialize layer managers for all animated objects.

    Returns:
        Number of managers created
    """
    import bpy

    count = 0
    scene = bpy.context.scene

    # Character armature
    armature = getattr(scene, "target_armature", None)
    if armature and armature.type == 'ARMATURE':
        mgr = get_layer_manager_for(armature)
        if mgr:
            mgr.disable_native_action(armature)
            count += 1

    # All objects with animation_data
    for obj in bpy.data.objects:
        if obj == armature:
            continue  # Already handled
        if obj.animation_data and obj.animation_data.action:
            mgr = get_layer_manager_for(obj)
            if mgr:
                mgr.disable_native_action(obj)
                count += 1

    _log(f"Initialized {count} layer managers")
    return count


def shutdown_layer_managers() -> None:
    """Shutdown all layer managers and restore native actions."""
    import bpy

    for obj_name, mgr in list(_managers.items()):
        # Find object by name
        obj = bpy.data.objects.get(obj_name)
        if obj:
            mgr.restore_native_action(obj)
        mgr.clear()

    _managers.clear()
    _log("All layer managers shutdown")


def update_all_managers(dt: float) -> None:
    """Update all layer managers. Call once per frame."""
    for mgr in _managers.values():
        mgr.update(dt)


# Legacy compatibility
def init_layer_manager() -> LayerManager:
    """Legacy: Initialize character's layer manager."""
    import bpy
    armature = getattr(bpy.context.scene, "target_armature", None)
    if armature:
        return get_layer_manager_for(armature)
    return None


def shutdown_layer_manager() -> None:
    """Legacy: Shutdown character's layer manager."""
    shutdown_layer_managers()


# =============================================================================
# LOGGING
# =============================================================================

def _log(msg: str) -> None:
    """Log if debug enabled."""
    try:
        import bpy
        scene = bpy.context.scene
        if scene and getattr(scene, "dev_debug_animations", False):
            log_game("LAYER_MGR", msg)
    except:
        pass
