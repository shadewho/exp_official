# Exp_Game/animations/bone_groups.py
"""
Bone Groups and Blend Masks for the Exploratory Standard Rig.

Defines body part groupings for:
- Additive animation blending (upper body reach while walking)
- IK masking (apply IK to specific chains)
- Partial body animations (play animation on subset of bones)

Based on rig.md - 53 bones total.
"""

import numpy as np
from typing import Dict, List, Set, Tuple

from ..developer.dev_logger import log_game

# =============================================================================
# BONE INDEX MAP (Alphabetical - matches rig.md)
# =============================================================================

BONE_INDEX: Dict[str, int] = {
    "Head": 0,
    "Hips": 1,
    "LeftArm": 2,
    "LeftFoot": 3,
    "LeftForeArm": 4,
    "LeftHand": 5,
    "LeftHandIndex1": 6,
    "LeftHandIndex2": 7,
    "LeftHandIndex3": 8,
    "LeftHandMiddle1": 9,
    "LeftHandMiddle2": 10,
    "LeftHandMiddle3": 11,
    "LeftHandPinky1": 12,
    "LeftHandPinky2": 13,
    "LeftHandPinky3": 14,
    "LeftHandRing1": 15,
    "LeftHandRing2": 16,
    "LeftHandRing3": 17,
    "LeftHandThumb1": 18,
    "LeftHandThumb2": 19,
    "LeftHandThumb3": 20,
    "LeftShin": 21,
    "LeftShoulder": 22,
    "LeftThigh": 23,
    "LeftToeBase": 24,
    "NeckLower": 25,
    "NeckUpper": 26,
    "RightArm": 27,
    "RightFoot": 28,
    "RightForeArm": 29,
    "RightHand": 30,
    "RightHandIndex1": 31,
    "RightHandIndex2": 32,
    "RightHandIndex3": 33,
    "RightHandMiddle1": 34,
    "RightHandMiddle2": 35,
    "RightHandMiddle3": 36,
    "RightHandPinky1": 37,
    "RightHandPinky2": 38,
    "RightHandPinky3": 39,
    "RightHandRing1": 40,
    "RightHandRing2": 41,
    "RightHandRing3": 42,
    "RightHandThumb1": 43,
    "RightHandThumb2": 44,
    "RightHandThumb3": 45,
    "RightShin": 46,
    "RightShoulder": 47,
    "RightThigh": 48,
    "RightToeBase": 49,
    "Spine": 50,
    "Spine1": 51,
    "Spine2": 52,
}

# Reverse lookup
INDEX_TO_BONE: Dict[int, str] = {v: k for k, v in BONE_INDEX.items()}

TOTAL_BONES = 53


# =============================================================================
# BONE GROUP DEFINITIONS
# =============================================================================

# Finger bones (for convenience)
LEFT_FINGERS = [
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
]

RIGHT_FINGERS = [
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
    "RightHandRing1", "RightHandRing2", "RightHandRing3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
]

ALL_FINGERS = LEFT_FINGERS + RIGHT_FINGERS


# =============================================================================
# BONE GROUPS (Named sets of bones)
# =============================================================================

BONE_GROUPS: Dict[str, List[str]] = {
    # Full body
    "ALL": list(BONE_INDEX.keys()),

    # Major regions
    "UPPER_BODY": [
        "Spine", "Spine1", "Spine2",
        "NeckLower", "NeckUpper", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    ] + ALL_FINGERS,

    "LOWER_BODY": [
        "Hips",
        "LeftThigh", "LeftShin", "LeftFoot", "LeftToeBase",
        "RightThigh", "RightShin", "RightFoot", "RightToeBase",
    ],

    # Spine and head
    "SPINE": ["Spine", "Spine1", "Spine2"],
    "SPINE_FULL": ["Hips", "Spine", "Spine1", "Spine2"],
    "NECK": ["NeckLower", "NeckUpper"],
    "HEAD": ["Head"],
    "HEAD_NECK": ["NeckLower", "NeckUpper", "Head"],
    "SPINE_HEAD": ["Spine", "Spine1", "Spine2", "NeckLower", "NeckUpper", "Head"],

    # Arms (IK chains)
    "ARM_L": ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"] + LEFT_FINGERS,
    "ARM_R": ["RightShoulder", "RightArm", "RightForeArm", "RightHand"] + RIGHT_FINGERS,
    "ARMS": [
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    ] + ALL_FINGERS,

    # Arms without fingers (for simpler IK)
    "ARM_L_NO_FINGERS": ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"],
    "ARM_R_NO_FINGERS": ["RightShoulder", "RightArm", "RightForeArm", "RightHand"],

    # Just the IK chain (no shoulder)
    "ARM_L_IK": ["LeftArm", "LeftForeArm", "LeftHand"],
    "ARM_R_IK": ["RightArm", "RightForeArm", "RightHand"],

    # Legs (IK chains)
    "LEG_L": ["LeftThigh", "LeftShin", "LeftFoot", "LeftToeBase"],
    "LEG_R": ["RightThigh", "RightShin", "RightFoot", "RightToeBase"],
    "LEGS": [
        "LeftThigh", "LeftShin", "LeftFoot", "LeftToeBase",
        "RightThigh", "RightShin", "RightFoot", "RightToeBase",
    ],

    # Just the IK chain (no toe)
    "LEG_L_IK": ["LeftThigh", "LeftShin", "LeftFoot"],
    "LEG_R_IK": ["RightThigh", "RightShin", "RightFoot"],

    # Hands and fingers
    "HAND_L": ["LeftHand"] + LEFT_FINGERS,
    "HAND_R": ["RightHand"] + RIGHT_FINGERS,
    "FINGERS_L": LEFT_FINGERS,
    "FINGERS_R": RIGHT_FINGERS,
    "FINGERS": ALL_FINGERS,

    # Feet
    "FOOT_L": ["LeftFoot", "LeftToeBase"],
    "FOOT_R": ["RightFoot", "RightToeBase"],

    # Root only
    "ROOT": ["Hips"],
}


# =============================================================================
# BLEND MASK GENERATION
# =============================================================================

def get_bone_indices(group_name: str) -> List[int]:
    """
    Get bone indices for a named group.

    Args:
        group_name: Name of the bone group (e.g., "UPPER_BODY", "ARM_R")

    Returns:
        List of bone indices
    """
    if group_name not in BONE_GROUPS:
        raise ValueError(f"Unknown bone group: {group_name}")

    return [BONE_INDEX[bone] for bone in BONE_GROUPS[group_name]]


def create_blend_mask(group_name: str, weight: float = 1.0) -> np.ndarray:
    """
    Create a blend mask array for a bone group.

    Args:
        group_name: Name of the bone group
        weight: Weight to apply to bones in the group (0.0 - 1.0)

    Returns:
        numpy array of shape (53,) with weights per bone
    """
    mask = np.zeros(TOTAL_BONES, dtype=np.float32)
    indices = get_bone_indices(group_name)
    mask[indices] = weight
    return mask


def create_combined_mask(groups_weights: Dict[str, float]) -> np.ndarray:
    """
    Create a blend mask combining multiple groups with different weights.

    Args:
        groups_weights: Dict mapping group names to weights
                        e.g., {"UPPER_BODY": 1.0, "LEGS": 0.5}

    Returns:
        numpy array of shape (53,) with combined weights (max per bone)
    """
    mask = np.zeros(TOTAL_BONES, dtype=np.float32)

    for group_name, weight in groups_weights.items():
        indices = get_bone_indices(group_name)
        # Use max so overlapping groups take the higher weight
        for idx in indices:
            mask[idx] = max(mask[idx], weight)

    return mask


def create_gradient_mask(
    group_name: str,
    start_weight: float = 0.0,
    end_weight: float = 1.0
) -> np.ndarray:
    """
    Create a gradient blend mask (useful for spine falloff).

    Args:
        group_name: Name of the bone group (bones are weighted in list order)
        start_weight: Weight for first bone in group
        end_weight: Weight for last bone in group

    Returns:
        numpy array of shape (53,) with gradient weights
    """
    mask = np.zeros(TOTAL_BONES, dtype=np.float32)
    bones = BONE_GROUPS.get(group_name, [])

    if len(bones) <= 1:
        if bones:
            mask[BONE_INDEX[bones[0]]] = end_weight
        return mask

    for i, bone in enumerate(bones):
        t = i / (len(bones) - 1)  # 0.0 to 1.0
        weight = start_weight + t * (end_weight - start_weight)
        mask[BONE_INDEX[bone]] = weight

    return mask


# =============================================================================
# PRESET MASKS (Common use cases)
# =============================================================================

class BlendMasks:
    """Pre-computed blend masks for common use cases."""

    # Full body
    FULL_BODY = create_blend_mask("ALL", 1.0)

    # Upper/lower split
    UPPER_BODY = create_blend_mask("UPPER_BODY", 1.0)
    LOWER_BODY = create_blend_mask("LOWER_BODY", 1.0)

    # Individual arms (for IK)
    ARM_LEFT = create_blend_mask("ARM_L", 1.0)
    ARM_RIGHT = create_blend_mask("ARM_R", 1.0)
    ARM_LEFT_IK = create_blend_mask("ARM_L_IK", 1.0)
    ARM_RIGHT_IK = create_blend_mask("ARM_R_IK", 1.0)

    # Individual legs (for IK)
    LEG_LEFT = create_blend_mask("LEG_L", 1.0)
    LEG_RIGHT = create_blend_mask("LEG_R", 1.0)
    LEG_LEFT_IK = create_blend_mask("LEG_L_IK", 1.0)
    LEG_RIGHT_IK = create_blend_mask("LEG_R_IK", 1.0)

    # Head/look-at
    HEAD_NECK = create_blend_mask("HEAD_NECK", 1.0)
    HEAD_ONLY = create_blend_mask("HEAD", 1.0)

    # Spine (with gradient for natural falloff)
    SPINE_GRADIENT = create_gradient_mask("SPINE_HEAD", 0.2, 1.0)

    # Upper body without arms (for spine lean without affecting arm animation)
    TORSO_ONLY = create_combined_mask({
        "SPINE": 1.0,
        "HEAD_NECK": 1.0,
    })


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def apply_mask_to_pose(
    pose: np.ndarray,
    mask: np.ndarray,
    base_pose: np.ndarray = None
) -> np.ndarray:
    """
    Apply a blend mask to interpolate between base pose and target pose.

    Args:
        pose: Target pose array (num_bones, 10) - quat + loc + scale
        mask: Blend mask (num_bones,) with weights 0.0 - 1.0
        base_pose: Base pose to blend from. If None, uses identity.

    Returns:
        Blended pose array
    """
    if base_pose is None:
        # Identity pose: quat=(1,0,0,0), loc=(0,0,0), scale=(1,1,1)
        base_pose = np.zeros_like(pose)
        base_pose[:, 0] = 1.0  # quat w
        base_pose[:, 7:10] = 1.0  # scale

    result = np.copy(base_pose)

    for bone_idx in range(TOTAL_BONES):
        weight = mask[bone_idx]
        if weight > 0.001:
            # Lerp location and scale
            result[bone_idx, 4:10] = (
                base_pose[bone_idx, 4:10] * (1 - weight) +
                pose[bone_idx, 4:10] * weight
            )

            # Slerp quaternion (simplified - proper slerp would be better)
            # For small angles, lerp + normalize is acceptable
            q_base = base_pose[bone_idx, 0:4]
            q_pose = pose[bone_idx, 0:4]

            # Ensure shortest path
            if np.dot(q_base, q_pose) < 0:
                q_pose = -q_pose

            q_blend = q_base * (1 - weight) + q_pose * weight
            q_blend = q_blend / (np.linalg.norm(q_blend) + 1e-10)
            result[bone_idx, 0:4] = q_blend

    return result


def get_group_for_chain(chain_name: str) -> str:
    """
    Get the bone group name for an IK chain.

    Args:
        chain_name: IK chain name (e.g., "arm_L", "leg_R")

    Returns:
        Bone group name (e.g., "ARM_L_IK", "LEG_R_IK")
    """
    chain_to_group = {
        "arm_L": "ARM_L_IK",
        "arm_R": "ARM_R_IK",
        "leg_L": "LEG_L_IK",
        "leg_R": "LEG_R_IK",
    }
    return chain_to_group.get(chain_name, "ALL")


def log_group_info(group_name: str) -> None:
    """Log information about a bone group (for debugging)."""
    if group_name not in BONE_GROUPS:
        log_game("ANIMATIONS", f"BONE_GROUP unknown={group_name}")
        return

    bones = BONE_GROUPS[group_name]
    log_game("ANIMATIONS", f"BONE_GROUP name={group_name} count={len(bones)}")
