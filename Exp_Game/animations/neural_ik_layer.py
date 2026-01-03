# Exp_Game/animations/neural_ik_layer.py
"""
Neural IK Layer Integration

Bridges the neural IK worker output to the animation blend system.
Converts 23-bone quaternion output to 54-bone blend pose format.

Architecture:
- Worker computes IK (23 bones × 4 quaternions)
- This module converts to blend system format (54 bones × 10 floats)
- A persistent override layer uses pose_provider callback
- Blend system merges with locomotion seamlessly
"""

import numpy as np
from typing import Optional, Dict

from .bone_groups import BONE_INDEX, TOTAL_BONES
from .blend_system import AnimationLayer, LayerType, get_blend_system
from ..developer.dev_logger import log_game

# =============================================================================
# NEURAL IK BONE ORDER (from config.py - MUST match training)
# =============================================================================

# These are the 23 bones the neural network controls, in output order
NEURAL_IK_BONES = [
    # Core (4)
    "Hips", "Spine", "Spine1", "Spine2",
    # Head/Neck (3)
    "NeckLower", "NeckUpper", "Head",
    # Left Arm (4)
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    # Right Arm (4)
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    # Left Leg (4)
    "LeftThigh", "LeftShin", "LeftFoot", "LeftToeBase",
    # Right Leg (4)
    "RightThigh", "RightShin", "RightFoot", "RightToeBase",
]

# Pre-compute mapping: neural IK index -> blend system index
_NEURAL_TO_BLEND: Dict[int, int] = {}
for ik_idx, bone_name in enumerate(NEURAL_IK_BONES):
    blend_idx = BONE_INDEX.get(bone_name, -1)
    if blend_idx >= 0:
        _NEURAL_TO_BLEND[ik_idx] = blend_idx

# Pre-compute blend mask for neural IK bones only (not fingers, not Root)
_NEURAL_IK_MASK = np.zeros(TOTAL_BONES, dtype=np.float32)
for bone_name in NEURAL_IK_BONES:
    blend_idx = BONE_INDEX.get(bone_name, -1)
    if blend_idx >= 0:
        _NEURAL_IK_MASK[blend_idx] = 1.0

# =============================================================================
# CACHED POSE STATE
# =============================================================================

# Current neural IK pose in blend system format
# Shape: (54, 10) - quat(4) + loc(3) + scale(3)
_cached_pose: Optional[np.ndarray] = None

# Layer active flag
_layer_active: bool = False

# Layer influence (0-1)
_layer_weight: float = 1.0


def _create_identity_pose() -> np.ndarray:
    """Create identity pose: quat=(1,0,0,0), loc=(0,0,0), scale=(1,1,1)."""
    pose = np.zeros((TOTAL_BONES, 10), dtype=np.float32)
    pose[:, 0] = 1.0      # quat w = 1
    pose[:, 7:10] = 1.0   # scale = 1
    return pose


# =============================================================================
# POSE CONVERSION
# =============================================================================

def convert_neural_ik_result(bone_rotations: np.ndarray) -> np.ndarray:
    """
    Convert neural IK output to blend system pose format.

    Args:
        bone_rotations: Shape (23, 4) - quaternions (w, x, y, z) for each bone

    Returns:
        Shape (54, 10) - full pose array for blend system
    """
    pose = _create_identity_pose()

    for ik_idx, blend_idx in _NEURAL_TO_BLEND.items():
        if ik_idx < len(bone_rotations):
            # Copy quaternion (w, x, y, z)
            pose[blend_idx, 0:4] = bone_rotations[ik_idx]

    return pose


def update_cached_pose(bone_rotations: np.ndarray) -> None:
    """
    Update the cached IK pose from worker result.

    Call this when NEURAL_IK_SOLVE result arrives.

    Args:
        bone_rotations: Shape (23, 4) - quaternions from worker
    """
    global _cached_pose
    _cached_pose = convert_neural_ik_result(bone_rotations)


def clear_cached_pose() -> None:
    """Clear the cached pose (call on game end)."""
    global _cached_pose
    _cached_pose = None


# =============================================================================
# POSE PROVIDER (callback for blend system)
# =============================================================================

def neural_ik_pose_provider() -> Optional[np.ndarray]:
    """
    Pose provider callback for the blend system.

    Returns the cached neural IK pose, or None if no pose cached.
    The blend system will use identity for None.
    """
    return _cached_pose


# =============================================================================
# LAYER MANAGEMENT
# =============================================================================

_ik_layer: Optional[AnimationLayer] = None


def create_neural_ik_layer(weight: float = 1.0) -> Optional[AnimationLayer]:
    """
    Create the neural IK override layer.

    Call once at game start after blend system is initialized.

    Args:
        weight: Initial blend weight (0-1)

    Returns:
        The created layer, or None if blend system not available
    """
    global _ik_layer, _layer_active, _layer_weight

    blend_system = get_blend_system()
    if not blend_system:
        log_game("NEURAL_IK", "LAYER_FAIL blend_system not initialized")
        return None

    _layer_weight = weight

    _ik_layer = AnimationLayer(
        name="neural_ik",
        layer_type=LayerType.OVERRIDE,
        pose_provider=neural_ik_pose_provider,
        mask_weights=_NEURAL_IK_MASK.copy(),
        weight=weight,
        target_weight=weight,
        looping=True,
        priority=100,
        fade_in=0.0,
        fade_out=0.0,
    )

    # Add to blend system's override layers
    blend_system._override_layers.append(_ik_layer)
    _layer_active = True

    # Log with mask info
    mask_bone_count = int(np.sum(_NEURAL_IK_MASK > 0))
    log_game("NEURAL_IK", f"LAYER_CREATED weight={weight} mask_bones={mask_bone_count}")
    return _ik_layer


def set_layer_weight(weight: float) -> None:
    """
    Set the neural IK layer weight.

    Args:
        weight: Blend weight (0-1). 0 = disabled, 1 = full IK
    """
    global _layer_weight

    if _ik_layer:
        _ik_layer.weight = weight
        _ik_layer.target_weight = weight
        _layer_weight = weight


def get_layer_weight() -> float:
    """Get current layer weight."""
    return _layer_weight


def is_layer_active() -> bool:
    """Check if IK layer is active."""
    return _layer_active and _ik_layer is not None


def destroy_neural_ik_layer() -> None:
    """
    Remove the neural IK layer.

    Call on game end.
    """
    global _ik_layer, _layer_active

    if _ik_layer:
        blend_system = get_blend_system()
        if blend_system and _ik_layer in blend_system._override_layers:
            blend_system._override_layers.remove(_ik_layer)
            log_game("NEURAL_IK", "LAYER_DESTROYED")

    _ik_layer = None
    _layer_active = False
    clear_cached_pose()


# =============================================================================
# CONVENIENCE: APPLY RESULT (combines update + ensures layer exists)
# =============================================================================

def apply_neural_ik_result(bone_rotations: np.ndarray, weight: float = 1.0) -> bool:
    """
    Apply neural IK result to the animation system.

    Convenience function that:
    1. Updates the cached pose
    2. Creates layer if needed
    3. Sets the weight

    Args:
        bone_rotations: Shape (23, 4) - quaternions from worker
        weight: Blend weight (0-1)

    Returns:
        True if successful
    """
    # Update cached pose
    update_cached_pose(bone_rotations)

    # Ensure layer exists
    if not is_layer_active():
        layer = create_neural_ik_layer(weight)
        if not layer:
            return False
    else:
        set_layer_weight(weight)

    return True
