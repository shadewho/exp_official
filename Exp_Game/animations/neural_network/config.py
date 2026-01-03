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
import os

# =============================================================================
# PATHS - HARDCODED to Desktop (AppData version gets reinstalled)
# =============================================================================
# NOTE: Defined early because _load_or_compute_rest_data() needs DATA_DIR

DATA_DIR = r"C:\Users\spenc\Desktop\Exploratory\addons\Exploratory\Exp_Game\animations\neural_network\training_data"
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
BEST_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "best.npy")

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
    # Core (Hips intentionally unclamped)
    "Hips":         (-1e6, 1e6, -1e6, 1e6, -1e6, 1e6),
    "Spine":        (-45, 45, -45, 45, -45, 45),
    "Spine1":       (-30, 30, -30, 30, -30, 30),
    "Spine2":       (-30, 30, -30, 30, -30, 30),
    # Neck/Head
    "NeckLower":    (-45, 45, -45, 45, -30, 30),
    "NeckUpper":    (-30, 30, -30, 30, -30, 30),
    "Head":         (-45, 45, -60, 60, -30, 30),
    # Left Arm
    "LeftShoulder": (-30, 30, -30, 30, -30, 30),
    "LeftArm":      (-90, 90, -120, 120, -80, 140),
    "LeftForeArm":  (0, 0, -170, 60, 0, 90),
    "LeftHand":     (-90, 90, 0, 0, -60, 60),
    # Right Arm (mirrored)
    "RightShoulder": (-30, 30, -30, 30, -30, 30),
    "RightArm":      (-90, 90, -120, 120, -140, 80),
    "RightForeArm":  (0, 0, -60, 170, -90, 0),
    "RightHand":     (-90, 90, 0, 0, -60, 60),
    # Left Leg
    "LeftThigh":    (-90, 120, -40, 40, -20, 80),
    "LeftShin":     (-150, 10, -20, 20, -15, 15),
    "LeftFoot":     (-80, 45, -30, 30, -40, 40),
    "LeftToeBase":  (-40, 40, 0, 0, 0, 0),
    # Right Leg (mirrored)
    "RightThigh":   (-90, 120, -40, 40, -80, 20),
    "RightShin":    (-150, 10, -20, 20, -15, 15),
    "RightFoot":    (-80, 45, -30, 30, -40, 40),
    "RightToeBase": (-40, 40, 0, 0, 0, 0),
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
# REST POSE DATA (from actual rig export)
# =============================================================================
# Bone HEAD positions in rest pose (T-pose)
# Updated from rig JSON export 2024-12

REST_POSITIONS = {
    "Hips":          (-0.0, 0.046052, 1.014617),
    "Spine":         (-0.0, 0.055555, 1.158783),
    "Spine1":        (-0.0, 0.077256, 1.306403),
    "Spine2":        (-0.0, 0.072666, 1.474343),
    "NeckLower":     (-0.0, 0.047334, 1.62263),
    "NeckUpper":     (-0.0, 0.054687, 1.660835),
    "Head":          (-0.0, 0.060049, 1.702313),
    "LeftShoulder":  (-0.044947, 0.047168, 1.605088),
    "LeftArm":       (-0.13598, 0.046837, 1.570376),
    "LeftForeArm":   (-0.41385, 0.043713, 1.557294),
    "LeftHand":      (-0.700165, 0.044187, 1.556918),
    "RightShoulder": (0.044947, 0.047499, 1.605089),
    "RightArm":      (0.13598, 0.04783, 1.570375),
    "RightForeArm":  (0.41385, 0.043693, 1.557292),
    "RightHand":     (0.700165, 0.044595, 1.556919),
    "LeftThigh":     (-0.096973, 0.046025, 1.018454),
    "LeftShin":      (-0.105687, 0.053738, 0.571063),
    "LeftFoot":      (-0.110153, -0.025574, 0.080267),
    "LeftToeBase":   (-0.111254, 0.125178, 0.020779),
    "RightThigh":    (0.096973, 0.046025, 1.018454),
    "RightShin":     (0.105687, 0.053738, 0.571063),
    "RightFoot":     (0.110153, -0.025574, 0.080267),
    "RightToeBase":  (0.111254, 0.125178, 0.020779),
}

# Bone lengths from rig export
BONE_LENGTHS_FROM_RIG = {
    "Hips":          0.144478,
    "Spine":         0.149207,
    "Spine1":        0.168002,
    "Spine2":        0.150445,
    "NeckLower":     0.038906,
    "NeckUpper":     0.041823,
    "Head":          0.270918,
    "LeftShoulder":  0.097427,
    "LeftArm":       0.278195,
    "LeftForeArm":   0.286315,
    "LeftHand":      0.071973,
    "RightShoulder": 0.097427,
    "RightArm":      0.278209,
    "RightForeArm":  0.286317,
    "RightHand":     0.071865,
    "LeftThigh":     0.447542,
    "LeftShin":      0.497182,
    "LeftFoot":      0.162069,
    "LeftToeBase":   0.071482,
    "RightThigh":    0.447542,
    "RightShin":     0.497182,
    "RightFoot":     0.162069,
    "RightToeBase":  0.071482,
}

# Use accurate bone lengths from rig export
BONE_LENGTHS = BONE_LENGTHS_FROM_RIG.copy()

# As array
LENGTHS_ARRAY = np.array([BONE_LENGTHS[b] for b in CONTROLLED_BONES], dtype=np.float32)

# Rest pose as array (23, 3)
REST_POSITIONS_ARRAY = np.array([REST_POSITIONS[b] for b in CONTROLLED_BONES], dtype=np.float32)

# =============================================================================
# REST ORIENTATIONS & LOCAL OFFSETS
# =============================================================================
# Extracted from Blender armature bone matrices (bone.bone.matrix_local).
# Each 3x3 matrix = [X_axis, Y_axis, Z_axis] as columns. Y points along bone.

REST_ORIENTATIONS = np.array([
    # Hips
    [[-1.000000, +0.000000, -0.000000],
     [-0.000000, +0.065772, +0.997835],
     [+0.000000, +0.997835, -0.065772]],
    # Spine
    [[-1.000000, +0.000000, -0.000000],
     [-0.000000, +0.145444, +0.989366],
     [+0.000000, +0.989366, -0.145444]],
    # Spine1
    [[-1.000000, +0.000000, -0.000000],
     [-0.000000, -0.027317, +0.999627],
     [-0.000000, +0.999627, +0.027317]],
    # Spine2
    [[-1.000000, +0.000000, -0.000000],
     [-0.000000, -0.172248, +0.985054],
     [-0.000000, +0.985054, +0.172248]],
    # NeckLower
    [[-1.000000, +0.000000, -0.000000],
     [-0.000000, +0.189003, +0.981977],
     [+0.000000, +0.981977, -0.189003]],
    # NeckUpper
    [[-1.000000, +0.000000, -0.000000],
     [-0.000000, +0.128213, +0.991747],
     [+0.000000, +0.991747, -0.128213]],
    # Head
    [[-1.000000, +0.000000, -0.000000],
     [-0.000000, -0.060457, +0.998171],
     [-0.000000, +0.998171, +0.060457]],
    # LeftShoulder
    [[+0.011245, -0.934369, +0.356130],
     [-0.999737, -0.003396, +0.022657],
     [-0.019961, -0.356292, -0.934162]],
    # LeftArm
    [[+0.012166, -0.998830, +0.046793],
     [-0.999734, -0.011233, +0.020158],
     [-0.019609, -0.047026, -0.998701]],
    # LeftForeArm
    [[-0.001631, -0.999998, +0.001343],
     [-0.999793, +0.001657, +0.020255],
     [-0.020257, -0.001310, -0.999794]],
    # LeftHand
    [[+0.003149, -0.999645, -0.026438],
     [-0.992720, -0.006308, +0.120282],
     [-0.120406, +0.025866, -0.992388]],
    # RightShoulder
    [[+0.003965, +0.934367, -0.356289],
     [+0.999793, +0.003396, +0.020033],
     [+0.019928, -0.356295, -0.934161]],
    # RightArm
    [[+0.015614, +0.998783, -0.046783],
     [+0.999758, -0.014871, +0.016200],
     [+0.015484, -0.047025, -0.998774]],
    # RightForeArm
    [[-0.003132, +0.999994, -0.001357],
     [+0.999861, +0.003153, +0.016360],
     [+0.016364, -0.001305, -0.999865]],
    # RightHand
    [[+0.003467, +0.999772, +0.021056],
     [+0.990924, -0.006264, +0.134274],
     [+0.134376, +0.020400, -0.990720]],
    # LeftThigh
    [[+0.999811, -0.019470, +0.000739],
     [-0.000398, +0.017234, +0.999864],
     [-0.019480, -0.999662, +0.017224]],
    # LeftShin
    [[+0.999880, -0.008982, -0.012668],
     [+0.011079, -0.159523, +0.987144],
     [-0.010887, -0.987154, -0.159401]],
    # LeftFoot
    [[+0.999413, -0.006793, +0.033615],
     [-0.006018, +0.930184, +0.367078],
     [-0.033761, -0.367059, +0.929585]],
    # LeftToeBase
    [[+0.997450, -0.039139, -0.059690],
     [+0.042547, +0.997489, +0.056827],
     [+0.057315, -0.059220, +0.996598]],
    # RightThigh
    [[+0.999811, +0.019470, -0.000739],
     [+0.000398, +0.017234, +0.999864],
     [+0.019480, -0.999662, +0.017224]],
    # RightShin
    [[+0.999880, +0.008982, +0.012668],
     [-0.011079, -0.159523, +0.987145],
     [+0.010887, -0.987154, -0.159401]],
    # RightFoot
    [[+0.999413, +0.006793, -0.033615],
     [+0.006018, +0.930184, +0.367078],
     [+0.033761, -0.367059, +0.929585]],
    # RightToeBase
    [[+0.997450, +0.039139, +0.059690],
     [-0.042547, +0.997489, +0.056827],
     [-0.057315, -0.059220, +0.996598]],
], dtype=np.float32)


def _compute_local_offsets():
    """Compute local offsets in parent's coordinate frame."""
    local_offsets = np.zeros((NUM_BONES, 3), dtype=np.float32)
    for i in range(NUM_BONES):
        parent_idx = PARENT_INDICES[i]
        if parent_idx < 0:
            local_offsets[i] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            world_offset = REST_POSITIONS_ARRAY[i] - REST_POSITIONS_ARRAY[parent_idx]
            parent_rest_rot = REST_ORIENTATIONS[parent_idx]
            local_offsets[i] = parent_rest_rot.T @ world_offset
    return local_offsets


LOCAL_OFFSETS = _compute_local_offsets()  # (23, 3)

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
HIDDEN_SIZE_1 = 256
HIDDEN_SIZE_2 = 256
HIDDEN_SIZE_3 = 128
BATCH_SIZE = 16  # Smaller batch = faster FK gradient computation
EPOCHS_DEFAULT = 300  # With early stopping, can set higher
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
