# Exp_Game/animations/neural_network/config.py
"""
Neural Network IK Configuration - Environment-Aware Version

Rig-specific data from rig.md - bones, limits, positions.
This is the SINGLE SOURCE OF TRUTH for the neural network.

Key changes from v1:
- Root-relative inputs (not world space)
- Environment context (ground, contacts)
- Axis-angle outputs (not Euler - avoids gimbal lock)
- FK chain data for forward kinematics loss
"""

import numpy as np

# =============================================================================
# CONTROLLED BONES (23 total)
# =============================================================================
# These are the bones the network controls.
# Order matters - parent must come before child for FK computation.

CONTROLLED_BONES = [
    # Core (4) - in hierarchy order
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

NUM_BONES = len(CONTROLLED_BONES)  # 23
BONE_TO_INDEX = {name: i for i, name in enumerate(CONTROLLED_BONES)}
INDEX_TO_BONE = {i: name for i, name in enumerate(CONTROLLED_BONES)}

# Parent bone indices for FK computation (-1 = root/no parent in our list)
# This defines the kinematic chain
BONE_PARENTS = {
    "Hips": -1,  # Root
    "Spine": BONE_TO_INDEX["Hips"],
    "Spine1": BONE_TO_INDEX["Spine"],
    "Spine2": BONE_TO_INDEX["Spine1"],
    "NeckLower": BONE_TO_INDEX["Spine2"],
    "NeckUpper": BONE_TO_INDEX["NeckLower"],
    "Head": BONE_TO_INDEX["NeckUpper"],
    "LeftShoulder": BONE_TO_INDEX["Spine2"],
    "LeftArm": BONE_TO_INDEX["LeftShoulder"],
    "LeftForeArm": BONE_TO_INDEX["LeftArm"],
    "LeftHand": BONE_TO_INDEX["LeftForeArm"],
    "RightShoulder": BONE_TO_INDEX["Spine2"],
    "RightArm": BONE_TO_INDEX["RightShoulder"],
    "RightForeArm": BONE_TO_INDEX["RightArm"],
    "RightHand": BONE_TO_INDEX["RightForeArm"],
    "LeftThigh": BONE_TO_INDEX["Hips"],
    "LeftShin": BONE_TO_INDEX["LeftThigh"],
    "LeftFoot": BONE_TO_INDEX["LeftShin"],
    "LeftToeBase": BONE_TO_INDEX["LeftFoot"],
    "RightThigh": BONE_TO_INDEX["Hips"],
    "RightShin": BONE_TO_INDEX["RightThigh"],
    "RightFoot": BONE_TO_INDEX["RightShin"],
    "RightToeBase": BONE_TO_INDEX["RightFoot"],
}

# As array for vectorized FK
PARENT_INDICES = np.array([BONE_PARENTS[b] for b in CONTROLLED_BONES], dtype=np.int32)

# =============================================================================
# NETWORK INPUT DIMENSIONS (Environment-Aware)
# =============================================================================

# Input breakdown:
# 1. Root-relative effector targets (5 effectors, Hips is root reference)
#    - LeftHand:  pos(3) + rot(3) = 6
#    - RightHand: pos(3) + rot(3) = 6
#    - LeftFoot:  pos(3) + rot(3) = 6
#    - RightFoot: pos(3) + rot(3) = 6
#    - Head:      pos(3) + rot(3) = 6
#    Subtotal: 30
#
# 2. Root orientation (Hips reference frame)
#    - Forward direction (3)
#    - Up direction (3)
#    Subtotal: 6
#
# 3. Ground/Contact context (per foot)
#    - Left:  height_offset(1) + normal(3) + grounded(1) + desired_contact(1) = 6
#    - Right: height_offset(1) + normal(3) + grounded(1) + desired_contact(1) = 6
#    Subtotal: 12
#
# 4. Motion state
#    - Phase (0-1): 1
#    - Task type: 1
#    Subtotal: 2
#
# TOTAL INPUT: 50

INPUT_EFFECTOR_SIZE = 30   # 5 effectors × 6
INPUT_ROOT_SIZE = 6        # forward + up
INPUT_GROUND_SIZE = 12     # 2 feet × 6
INPUT_MOTION_SIZE = 2      # phase + task
INPUT_SIZE = INPUT_EFFECTOR_SIZE + INPUT_ROOT_SIZE + INPUT_GROUND_SIZE + INPUT_MOTION_SIZE  # 50

# Input slice indices for easy extraction
INPUT_SLICES = {
    'effectors': (0, INPUT_EFFECTOR_SIZE),
    'root': (INPUT_EFFECTOR_SIZE, INPUT_EFFECTOR_SIZE + INPUT_ROOT_SIZE),
    'ground': (INPUT_EFFECTOR_SIZE + INPUT_ROOT_SIZE,
               INPUT_EFFECTOR_SIZE + INPUT_ROOT_SIZE + INPUT_GROUND_SIZE),
    'motion': (INPUT_EFFECTOR_SIZE + INPUT_ROOT_SIZE + INPUT_GROUND_SIZE, INPUT_SIZE),
}

# =============================================================================
# NETWORK OUTPUT DIMENSIONS
# =============================================================================

# Output: Axis-angle rotations for all controlled bones
# Axis-angle avoids Euler gimbal lock while keeping 3 values per bone
# 23 bones × 3 (axis-angle) = 69
OUTPUT_SIZE = NUM_BONES * 3  # 69

# =============================================================================
# END EFFECTORS (targets we want to reach)
# =============================================================================
# Note: Hips is NOT an effector - it's the root reference frame

END_EFFECTORS = [
    "LeftHand",
    "RightHand",
    "LeftFoot",
    "RightFoot",
    "Head",
]

CONTACT_EFFECTORS = ["LeftFoot", "RightFoot"]  # Effectors that can be grounded

# =============================================================================
# JOINT LIMITS (degrees)
# =============================================================================
# Anatomical rotation limits per bone.
# Format: (x_min, x_max, y_min, y_max, z_min, z_max)
# Used for soft penalty during training AND hard clamp after prediction.

JOINT_LIMITS_DEG = {
    # Core
    "Hips":         (-20, 20, -30, 30, -15, 15),
    "Spine":        (-30, 45, -20, 20, -20, 20),
    "Spine1":       (-30, 45, -15, 15, -15, 15),
    "Spine2":       (-20, 30, -15, 15, -15, 15),
    # Neck/Head
    "NeckLower":    (-30, 40, -45, 45, -30, 30),
    "NeckUpper":    (-30, 40, -45, 45, -30, 30),
    "Head":         (-40, 60, -70, 70, -30, 30),
    # Left Arm
    "LeftShoulder": (-15, 15, -10, 30, -20, 20),
    "LeftArm":      (-180, 60, -90, 90, -90, 90),
    "LeftForeArm":  (0, 145, -5, 5, -90, 90),
    "LeftHand":     (-70, 70, -20, 20, -40, 40),
    # Right Arm (mirrored)
    "RightShoulder": (-15, 15, -30, 10, -20, 20),
    "RightArm":      (-180, 60, -90, 90, -90, 90),
    "RightForeArm":  (0, 145, -5, 5, -90, 90),
    "RightHand":     (-70, 70, -20, 20, -40, 40),
    # Left Leg
    "LeftThigh":    (-30, 120, -45, 45, -60, 60),
    "LeftShin":     (0, 150, -5, 5, -5, 5),
    "LeftFoot":     (-45, 45, -30, 30, -25, 25),
    "LeftToeBase":  (-30, 60, -5, 5, -5, 5),
    # Right Leg (mirrored)
    "RightThigh":   (-30, 120, -45, 45, -60, 60),
    "RightShin":    (0, 150, -5, 5, -5, 5),
    "RightFoot":    (-45, 45, -30, 30, -25, 25),
    "RightToeBase": (-30, 60, -5, 5, -5, 5),
}

# Convert to radians for network use
JOINT_LIMITS_RAD = {
    bone: tuple(np.radians(v) for v in limits)
    for bone, limits in JOINT_LIMITS_DEG.items()
}

# As numpy array for fast vectorized operations
# Shape: (23, 6) - one row per bone, 6 limits per bone
LIMITS_ARRAY = np.array([
    JOINT_LIMITS_RAD[bone] for bone in CONTROLLED_BONES
], dtype=np.float32)

# Min/max arrays for clamping (23, 3) each
LIMITS_MIN = LIMITS_ARRAY[:, [0, 2, 4]]  # x_min, y_min, z_min
LIMITS_MAX = LIMITS_ARRAY[:, [1, 3, 5]]  # x_max, y_max, z_max

# =============================================================================
# REST POSE DATA (from rig.md)
# =============================================================================
# Bone positions and lengths in rest pose (T-pose)

REST_POSITIONS = {
    "Hips":       (0.0, 0.056, 1.001),
    "Spine":      (0.0, 0.056, 1.088),
    "Spine1":     (0.0, 0.035, 1.194),
    "Spine2":     (0.0, 0.021, 1.310),
    "NeckLower":  (0.0, 0.025, 1.456),
    "NeckUpper":  (0.0, 0.035, 1.536),
    "Head":       (0.0, 0.047, 1.701),
    "LeftShoulder":  (-0.069, 0.020, 1.431),
    "LeftArm":       (-0.185, 0.020, 1.431),
    "LeftForeArm":   (-0.450, 0.042, 1.432),
    "LeftHand":      (-0.700, 0.044, 1.557),
    "RightShoulder": (0.069, 0.021, 1.431),
    "RightArm":      (0.185, 0.021, 1.431),
    "RightForeArm":  (0.450, 0.043, 1.432),
    "RightHand":     (0.700, 0.045, 1.557),
    "LeftThigh":     (-0.110, 0.000, 0.945),
    "LeftShin":      (-0.110, -0.008, 0.527),
    "LeftFoot":      (-0.110, -0.016, 0.098),
    "LeftToeBase":   (-0.110, 0.093, 0.016),
    "RightThigh":    (0.110, 0.003, 0.945),
    "RightShin":     (0.110, -0.003, 0.527),
    "RightFoot":     (0.110, -0.010, 0.098),
    "RightToeBase":  (0.110, 0.099, 0.016),
}

# Bone lengths (approximate, for FK)
BONE_LENGTHS = {}
for bone in CONTROLLED_BONES:
    parent_name = None
    for name, idx in BONE_PARENTS.items():
        if name == bone and idx >= 0:
            parent_name = INDEX_TO_BONE[idx]
            break
    if parent_name and parent_name in REST_POSITIONS and bone in REST_POSITIONS:
        p1 = np.array(REST_POSITIONS[parent_name])
        p2 = np.array(REST_POSITIONS[bone])
        BONE_LENGTHS[bone] = float(np.linalg.norm(p2 - p1))
    else:
        BONE_LENGTHS[bone] = 0.1  # Default for root

# As array
LENGTHS_ARRAY = np.array([BONE_LENGTHS[b] for b in CONTROLLED_BONES], dtype=np.float32)

# Rest pose as array (23, 3)
REST_POSITIONS_ARRAY = np.array([REST_POSITIONS[b] for b in CONTROLLED_BONES], dtype=np.float32)

# =============================================================================
# IK CHAINS (for FK computation and loss calculation)
# =============================================================================

IK_CHAINS = {
    "arm_L": {
        "bones": ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"],
        "tip": "LeftHand",
        "effector_idx": END_EFFECTORS.index("LeftHand"),
    },
    "arm_R": {
        "bones": ["RightShoulder", "RightArm", "RightForeArm", "RightHand"],
        "tip": "RightHand",
        "effector_idx": END_EFFECTORS.index("RightHand"),
    },
    "leg_L": {
        "bones": ["LeftThigh", "LeftShin", "LeftFoot", "LeftToeBase"],
        "tip": "LeftFoot",
        "effector_idx": END_EFFECTORS.index("LeftFoot"),
    },
    "leg_R": {
        "bones": ["RightThigh", "RightShin", "RightFoot", "RightToeBase"],
        "tip": "RightFoot",
        "effector_idx": END_EFFECTORS.index("RightFoot"),
    },
    "spine": {
        "bones": ["Hips", "Spine", "Spine1", "Spine2"],
        "tip": "Spine2",
        "effector_idx": -1,  # Not a target effector
    },
    "head": {
        "bones": ["NeckLower", "NeckUpper", "Head"],
        "tip": "Head",
        "effector_idx": END_EFFECTORS.index("Head"),
    },
}

# =============================================================================
# TASK TYPES (for task-aware learning)
# =============================================================================

TASK_TYPES = {
    "idle": 0,
    "locomotion": 1,
    "reach": 2,
    "grab": 3,
    "crouch": 4,
    "jump": 5,
}

# =============================================================================
# NORMALIZATION (for comparable input magnitudes)
# =============================================================================

# Position normalization: divide by this to get ~[-1, 1] range
POSITION_SCALE = 1.0  # 1 meter = 1 unit (positions already in meters)

# Rotation normalization: rotations in radians, typical range ~[-pi, pi]
ROTATION_SCALE = np.pi  # Divide by pi to get ~[-1, 1]

# Height normalization
HEIGHT_SCALE = 2.0  # Character height ~2m

# Output constraints
MAX_AXIS_ANGLE = 2.0  # Maximum rotation magnitude (radians) - ~115 degrees

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

LEARNING_RATE = 0.001
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 96
BATCH_SIZE = 32
EPOCHS_DEFAULT = 100
DROPOUT_RATE = 0.1  # Regularization

# Training/test split
TRAIN_SPLIT = 0.8  # 80% training, 20% test

# Loss weights
FK_LOSS_WEIGHT = 1.0        # Primary: reach the targets
POSE_LOSS_WEIGHT = 0.3      # Secondary: match training poses
CONTACT_LOSS_WEIGHT = 0.5   # Feet stay planted when grounded
LIMIT_PENALTY_WEIGHT = 0.1  # Soft joint limit penalty
SMOOTHNESS_WEIGHT = 0.05    # Temporal consistency (if training sequences)
SLIP_PENALTY_WEIGHT = 0.2   # Lateral foot slip penalty

# =============================================================================
# PATHS
# =============================================================================

import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(_THIS_DIR, "weights")
BEST_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "best.npy")
DATA_DIR = os.path.join(_THIS_DIR, "data")
