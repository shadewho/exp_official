# engine/animations/default_limits.py
"""
Default Joint Rotation Limits for the Exploratory Standard Rig.

These limits define anatomically valid rotation ranges for each bone.
Used by the engine during pose blending to prevent unnatural poses.

Format: {bone_name: {"X": [min, max], "Y": [min, max], "Z": [min, max]}}
All values in degrees.

Mirrored bones have appropriate sign flips for symmetric movement.
"""

# =============================================================================
# DEFAULT JOINT LIMITS
# =============================================================================

DEFAULT_JOINT_LIMITS = {
    # Spine & Torso
    "Spine": {"X": [-45, 45], "Y": [-45, 45], "Z": [-45, 45]},
    "Spine1": {"X": [-30, 30], "Y": [-30, 30], "Z": [-30, 30]},
    "Spine2": {"X": [-30, 30], "Y": [-30, 30], "Z": [-30, 30]},

    # Neck & Head
    "NeckLower": {"X": [-45, 45], "Y": [-45, 45], "Z": [-30, 30]},
    "NeckUpper": {"X": [-30, 30], "Y": [-30, 30], "Z": [-30, 30]},
    "Head": {"X": [-45, 45], "Y": [-60, 60], "Z": [-30, 30]},

    # Left Arm
    "LeftArm": {"X": [-90, 90], "Y": [-120, 120], "Z": [-80, 140]},
    "LeftForeArm": {"X": [0, 0], "Y": [-170, 60], "Z": [0, 90]},
    "LeftHand": {"X": [-90, 90], "Y": [0, 0], "Z": [-60, 60]},

    # Right Arm (Mirrored)
    "RightArm": {"X": [-90, 90], "Y": [-120, 120], "Z": [-140, 80]},
    "RightForeArm": {"X": [0, 0], "Y": [-60, 170], "Z": [-90, 0]},
    "RightHand": {"X": [-90, 90], "Y": [0, 0], "Z": [-60, 60]},

    # Left Leg
    "LeftThigh": {"X": [-90, 120], "Y": [-40, 40], "Z": [-20, 80]},
    "LeftShin": {"X": [-150, 10], "Y": [-10, 10], "Z": [0, 0]},
    "LeftFoot": {"X": [-80, 45], "Y": [-30, 30], "Z": [-40, 40]},
    "LeftToeBase": {"X": [-40, 40], "Y": [0, 0], "Z": [0, 0]},

    # Right Leg (Mirrored)
    "RightThigh": {"X": [-90, 120], "Y": [-40, 40], "Z": [-80, 20]},
    "RightShin": {"X": [-150, 10], "Y": [-10, 10], "Z": [0, 0]},
    "RightFoot": {"X": [-80, 45], "Y": [-30, 30], "Z": [-40, 40]},
    "RightToeBase": {"X": [-40, 40], "Y": [0, 0], "Z": [0, 0]},

    # Left Hand - Thumb
    "LeftHandThumb1": {"X": [-30, 30], "Y": [0, 0], "Z": [-50, 40]},
    "LeftHandThumb2": {"X": [0, 0], "Y": [0, 0], "Z": [-60, 0]},
    "LeftHandThumb3": {"X": [0, 0], "Y": [0, 0], "Z": [-60, 0]},

    # Left Hand - Index
    "LeftHandIndex1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "LeftHandIndex2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "LeftHandIndex3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},

    # Left Hand - Middle
    "LeftHandMiddle1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "LeftHandMiddle2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "LeftHandMiddle3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},

    # Left Hand - Ring
    "LeftHandRing1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "LeftHandRing2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "LeftHandRing3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},

    # Left Hand - Pinky
    "LeftHandPinky1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "LeftHandPinky2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "LeftHandPinky3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},

    # Right Hand - Thumb (Mirrored Z)
    "RightHandThumb1": {"X": [-30, 30], "Y": [0, 0], "Z": [-40, 50]},
    "RightHandThumb2": {"X": [0, 0], "Y": [0, 0], "Z": [0, 60]},
    "RightHandThumb3": {"X": [0, 0], "Y": [0, 0], "Z": [0, 60]},

    # Right Hand - Index
    "RightHandIndex1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "RightHandIndex2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "RightHandIndex3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},

    # Right Hand - Middle
    "RightHandMiddle1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "RightHandMiddle2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "RightHandMiddle3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},

    # Right Hand - Ring
    "RightHandRing1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "RightHandRing2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "RightHandRing3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},

    # Right Hand - Pinky
    "RightHandPinky1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "RightHandPinky2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
    "RightHandPinky3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
}


def get_default_limits():
    """Get a copy of the default joint limits."""
    return DEFAULT_JOINT_LIMITS.copy()


def get_bone_limit(bone_name: str) -> dict:
    """Get limits for a specific bone, or None if not defined."""
    return DEFAULT_JOINT_LIMITS.get(bone_name)
