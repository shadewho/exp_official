# Exp_Game/engine/animations/ik_chains.py
"""
IK Chain Definitions and Constants.

This module contains all chain definitions for two-bone IK solving.
Worker-safe (NO bpy imports).

USAGE:
    from .ik_chains import LEG_IK, ARM_IK, IK_TOLERANCES
"""

from typing import Dict

# =============================================================================
# TOLERANCE CONSTANTS
# =============================================================================

class IK_TOLERANCES:
    """Tolerance constants for IK solving."""

    # Position error thresholds (meters)
    FOOT_ERROR = 0.01           # 1cm - feet must be very accurate
    HAND_ERROR = 0.03           # 3cm - hands can have more tolerance
    HIPS_ERROR = 0.005          # 0.5cm - hips position should be precise

    # Anatomical constraints
    KNEE_FORWARD_MIN = 0.01     # Knee must be at least 1cm forward of hip-ankle line
    ELBOW_SIDE_TOLERANCE = 0.02 # Elbow can be up to 2cm past body center
    ELBOW_FORWARD_MAX = 0.08    # Elbow can be up to 8cm in front of shoulder

    # Cross-body detection
    CROSS_BODY_THRESHOLD = 0.05 # 5cm past center = cross-body reach

    # IK solver geometry
    MIN_REACH_FACTOR = 0.001    # Minimum reach to attempt IK solve
    NEAR_FULL_EXTENSION = 0.98  # 98% of max reach = near full extension


# =============================================================================
# LEG IK CHAINS - Bone names and lengths from rig.md
# =============================================================================

LEG_IK: Dict[str, dict] = {
    "leg_L": {
        "root": "LeftThigh",      # Thigh
        "mid": "LeftShin",        # Shin
        "tip": "LeftFoot",        # Foot
        "len_upper": 0.448,       # Actual rig measurement
        "len_lower": 0.497,       # Actual rig measurement
        "reach": 0.945,           # Actual reach (upper + lower)
    },
    "leg_R": {
        "root": "RightThigh",
        "mid": "RightShin",
        "tip": "RightFoot",
        "len_upper": 0.448,
        "len_lower": 0.497,
        "reach": 0.945,
    },
}


# =============================================================================
# ARM IK CHAINS - Bone names and lengths from rig.md
# =============================================================================

ARM_IK: Dict[str, dict] = {
    "arm_L": {
        "root": "LeftArm",        # Upper arm
        "mid": "LeftForeArm",     # Forearm
        "tip": "LeftHand",        # Hand
        "len_upper": 0.2782,      # Actual rig measurement
        "len_lower": 0.2863,      # Actual rig measurement
        "reach": 0.5645,          # Actual reach (upper + lower)
    },
    "arm_R": {
        "root": "RightArm",
        "mid": "RightForeArm",
        "tip": "RightHand",
        "len_upper": 0.2782,
        "len_lower": 0.2863,
        "reach": 0.5645,
    },
}


# =============================================================================
# SPINE CHAIN
# =============================================================================

SPINE_CHAIN = ["Hips", "Spine", "Spine1", "Spine2", "Neck"]

# Body dimensions for IK calculations
BODY_DIMENSIONS = {
    "hip_width": 0.10,            # Distance from center to hip joint
    "shoulder_width": 0.15,       # Distance from center to shoulder
    "spine_to_shoulder": 0.50,    # Vertical distance hips to shoulders
    "shoulder_to_head": 0.10,     # Vertical distance shoulders to head base
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_chain(chain_name: str) -> dict:
    """
    Get chain definition by name.

    Args:
        chain_name: "leg_L", "leg_R", "arm_L", "arm_R"

    Returns:
        Chain definition dict or None
    """
    if chain_name.startswith("leg"):
        return LEG_IK.get(chain_name)
    elif chain_name.startswith("arm"):
        return ARM_IK.get(chain_name)
    return None


def get_chain_type(chain_name: str) -> str:
    """
    Get chain type from name.

    Args:
        chain_name: "leg_L", "leg_R", "arm_L", "arm_R"

    Returns:
        "leg" or "arm"
    """
    if chain_name.startswith("leg"):
        return "leg"
    elif chain_name.startswith("arm"):
        return "arm"
    return "unknown"


def get_chain_side(chain_name: str) -> str:
    """
    Get chain side from name.

    Args:
        chain_name: "leg_L", "leg_R", "arm_L", "arm_R"

    Returns:
        "L" or "R"
    """
    if chain_name.endswith("_L") or chain_name.endswith("_l"):
        return "L"
    return "R"


def get_max_reach(chain_name: str) -> float:
    """
    Get maximum reach for a chain.

    Args:
        chain_name: "leg_L", "leg_R", "arm_L", "arm_R"

    Returns:
        Max reach in meters
    """
    chain = get_chain(chain_name)
    if chain:
        return chain.get("reach", 0.0)
    return 0.0
