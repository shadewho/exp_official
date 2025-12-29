# Exp_Game/animations/neural_ik.py
"""
███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗         ██╗██╗  ██╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║         ██║██║ ██╔╝
██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║         ██║█████╔╝
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║         ██║██╔═██╗
██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗    ██║██║  ██╗
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝╚═╝  ╚═╝

=============================================================================
FULL BODY NEURAL IK - Tailored to Exploratory Standard Rig
=============================================================================

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! THIS IS FULL BODY IK - NOT INDIVIDUAL LIMBS                            !!
!! THIS IS SPECIFIC TO THE EXPLORATORY RIG (see rig.md)                   !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

RIG SPECIFICATIONS (from rig.md):
=================================
- Total Bones: 54
- Naming: Mixamo-style (LeftArm, RightArm, etc.)
- Rest Pose: T-Pose
- Total Height: ~1.97m
- Hips Height: 1.001m
- Floor Level: Z = 0

SOFT JOINT LIMITS:
==================
The network is FREE to output any rotation, but violations add penalty to error.
This lets the network LEARN natural movement while still exploring.

    total_error = position_error + (limit_violation_penalty * PENALTY_WEIGHT)

BONE AXIS ORIENTATIONS (Critical for IK):
=========================================
- Spine/Head: X=LEFT, Y=UP, Z=FORWARD
- Left Arm: X=BACK, Y=LEFT, Z=DOWN (elbow bends around Z)
- Right Arm: X=FORWARD, Y=RIGHT, Z=DOWN (elbow bends around Z)
- Legs: X=RIGHT, Y=DOWN, Z=FORWARD (knee bends around X)
"""

import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# Data directory - HARDCODED to Desktop version
DATA_DIR = Path(r"C:\Users\spenc\Desktop\Exploratory\addons\Exploratory\Exp_Game\animations\neural_ik_data")


# =============================================================================
# EXPLORATORY RIG DEFINITION (from rig.md)
# =============================================================================

# Bones controlled by neural IK (excludes fingers and Root)
# These are the bones that affect full-body pose
CONTROLLED_BONES = [
    # Core (4)
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    # Neck/Head (3)
    "NeckLower",
    "NeckUpper",
    "Head",
    # Left Arm (4)
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    # Right Arm (4)
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    # Left Leg (4)
    "LeftThigh",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    # Right Leg (4)
    "RightThigh",
    "RightShin",
    "RightFoot",
    "RightToeBase",
]

NUM_BONES = len(CONTROLLED_BONES)  # 23 bones
OUTPUT_SIZE = NUM_BONES * 3        # 69 values (euler XYZ per bone)
INPUT_SIZE = 28                    # Full body target

# Bone name to index mapping
BONE_INDEX = {name: i for i, name in enumerate(CONTROLLED_BONES)}


# =============================================================================
# SOFT JOINT LIMITS (from rig.md) - Degrees
# =============================================================================
# Format: "BoneName": [(min_x, max_x), (min_y, max_y), (min_z, max_z)]

JOINT_LIMITS_DEG = {
    # Hips has NO limits (root of skeleton)
    "Hips": None,

    # Spine
    "Spine": [(-45, 45), (-45, 45), (-45, 45)],
    "Spine1": [(-30, 30), (-30, 30), (-30, 30)],
    "Spine2": [(-30, 30), (-30, 30), (-30, 30)],

    # Neck/Head
    "NeckLower": [(-45, 45), (-45, 45), (-30, 30)],
    "NeckUpper": [(-30, 30), (-30, 30), (-30, 30)],
    "Head": [(-45, 45), (-60, 60), (-30, 30)],

    # Left Arm
    "LeftShoulder": [(-30, 30), (-30, 30), (-30, 30)],  # Clavicle - limited
    "LeftArm": [(-90, 90), (-120, 120), (-80, 140)],
    "LeftForeArm": [(0, 0), (-170, 60), (0, 90)],  # X locked, elbow bend on Y
    "LeftHand": [(-90, 90), (0, 0), (-60, 60)],

    # Right Arm
    "RightShoulder": [(-30, 30), (-30, 30), (-30, 30)],
    "RightArm": [(-90, 90), (-120, 120), (-140, 80)],
    "RightForeArm": [(0, 0), (-60, 170), (-90, 0)],
    "RightHand": [(-90, 90), (0, 0), (-60, 60)],

    # Left Leg
    "LeftThigh": [(-90, 120), (-40, 40), (-20, 80)],
    "LeftShin": [(-150, 10), (-20, 20), (-15, 15)],  # Knee - mostly X rotation
    "LeftFoot": [(-80, 45), (-30, 30), (-40, 40)],
    "LeftToeBase": [(-40, 40), (0, 0), (0, 0)],

    # Right Leg
    "RightThigh": [(-90, 120), (-40, 40), (-80, 20)],
    "RightShin": [(-150, 10), (-20, 20), (-15, 15)],
    "RightFoot": [(-80, 45), (-30, 30), (-40, 40)],
    "RightToeBase": [(-40, 40), (0, 0), (0, 0)],
}

# Convert to radians for internal use
JOINT_LIMITS_RAD = {}
for bone, limits in JOINT_LIMITS_DEG.items():
    if limits is None:
        JOINT_LIMITS_RAD[bone] = None
    else:
        JOINT_LIMITS_RAD[bone] = [
            (np.radians(lo), np.radians(hi)) for (lo, hi) in limits
        ]


# =============================================================================
# IK CHAIN DEFINITIONS (from rig.md)
# =============================================================================

IK_CHAINS = {
    "left_arm": {
        "root": "LeftArm",
        "mid": "LeftForeArm",
        "tip": "LeftHand",
        "len_upper": 0.2782,
        "len_lower": 0.2863,
        "reach": 0.5645,
    },
    "right_arm": {
        "root": "RightArm",
        "mid": "RightForeArm",
        "tip": "RightHand",
        "len_upper": 0.2782,
        "len_lower": 0.2863,
        "reach": 0.5645,
    },
    "left_leg": {
        "root": "LeftThigh",
        "mid": "LeftShin",
        "tip": "LeftFoot",
        "len_upper": 0.4947,
        "len_lower": 0.4784,
        "reach": 0.9731,
    },
    "right_leg": {
        "root": "RightThigh",
        "mid": "RightShin",
        "tip": "RightFoot",
        "len_upper": 0.4947,
        "len_lower": 0.4775,
        "reach": 0.9722,
    },
}

# Rest pose positions (from rig.md)
REST_POSITIONS = {
    "Hips": np.array([0, 0.056, 1.001], dtype=np.float32),
    "LeftHand": np.array([-0.700, 0.044, 1.557], dtype=np.float32),
    "RightHand": np.array([0.700, 0.045, 1.557], dtype=np.float32),
    "LeftFoot": np.array([-0.110, -0.016, 0.098], dtype=np.float32),
    "RightFoot": np.array([0.110, -0.010, 0.098], dtype=np.float32),
    "Head": np.array([0, 0.047, 1.701], dtype=np.float32),
}


# =============================================================================
# FULL BODY POSE TARGET
# =============================================================================

@dataclass
class FullBodyTarget:
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! FULL BODY POSE GOAL - All positions relative to Root bone     !!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    # Target positions (meters, relative to Root)
    left_hand: np.ndarray = field(default_factory=lambda: REST_POSITIONS["LeftHand"].copy())
    right_hand: np.ndarray = field(default_factory=lambda: REST_POSITIONS["RightHand"].copy())
    left_foot: np.ndarray = field(default_factory=lambda: REST_POSITIONS["LeftFoot"].copy())
    right_foot: np.ndarray = field(default_factory=lambda: REST_POSITIONS["RightFoot"].copy())
    look_at: np.ndarray = field(default_factory=lambda: np.array([0, 2, 1.7], dtype=np.float32))

    # Hips drop (meters) - for crouching
    hips_drop: float = 0.0

    # Target weights (0 = ignore, 1 = must reach)
    left_hand_weight: float = 0.0
    right_hand_weight: float = 0.0
    left_foot_weight: float = 1.0   # Feet usually grounded
    right_foot_weight: float = 1.0
    look_at_weight: float = 0.0

    # Character orientation (world space)
    forward: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0], dtype=np.float32))
    up: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1], dtype=np.float32))

    def to_input_vector(self) -> np.ndarray:
        """Convert to neural network input (28 floats)."""
        return np.concatenate([
            self.left_hand,      # 3
            self.right_hand,     # 3
            self.left_foot,      # 3
            self.right_foot,     # 3
            self.look_at,        # 3
            [self.hips_drop],    # 1
            [self.left_hand_weight, self.right_hand_weight,
             self.left_foot_weight, self.right_foot_weight,
             self.look_at_weight],  # 5
            self.forward,        # 3
            self.up,             # 3
        ]).astype(np.float32)    # Total: 28


# =============================================================================
# SOFT LIMIT PENALTY
# =============================================================================

# How much to penalize limit violations (higher = stricter limits)
LIMIT_PENALTY_WEIGHT = 0.5


def compute_limit_penalty(rotations_rad: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute soft limit penalty for bone rotations.

    Args:
        rotations_rad: Shape (NUM_BONES * 3,) - euler XYZ per bone in radians

    Returns:
        total_penalty: Sum of all violations (used in error)
        gradient: Shape (NUM_BONES * 3,) - direction to reduce violations
    """
    penalty = 0.0
    gradient = np.zeros_like(rotations_rad)

    for i, bone_name in enumerate(CONTROLLED_BONES):
        limits = JOINT_LIMITS_RAD.get(bone_name)
        if limits is None:
            continue  # No limits for this bone (e.g., Hips)

        for axis in range(3):
            idx = i * 3 + axis
            val = rotations_rad[idx]
            lo, hi = limits[axis]

            # Check if locked axis (min == max == 0)
            if lo == 0 and hi == 0:
                # Locked - any rotation is a violation
                violation = abs(val)
                if violation > 0.01:  # Small tolerance
                    penalty += violation * 2.0  # Higher penalty for locked axes
                    gradient[idx] = -np.sign(val)
            elif val < lo:
                violation = lo - val
                penalty += violation
                gradient[idx] = 1.0  # Push toward valid range
            elif val > hi:
                violation = val - hi
                penalty += violation
                gradient[idx] = -1.0  # Push toward valid range

    return penalty * LIMIT_PENALTY_WEIGHT, gradient * LIMIT_PENALTY_WEIGHT


# =============================================================================
# NEURAL NETWORK
# =============================================================================

class FullBodyIKNetwork:
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! NEURAL NETWORK FOR FULL BODY IK                                !!
    !! Tailored to Exploratory Standard Rig (54 bones, 23 controlled) !!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Architecture:
        Input (28) → Hidden1 (64, tanh) → Hidden2 (48, tanh) → Output (69)

    Input: Full body pose goal (positions, weights, orientation)
    Output: Euler rotations (XYZ) for 23 controlled bones
    """

    def __init__(self):
        self.input_size = INPUT_SIZE      # 28
        self.hidden1_size = 64
        self.hidden2_size = 48
        self.output_size = OUTPUT_SIZE    # 69 (23 bones × 3)

        self._init_weights()
        self.learning_rate = 0.01
        self.train_count = 0

    def _init_weights(self):
        """Xavier initialization."""
        s1 = np.sqrt(2.0 / (self.input_size + self.hidden1_size))
        self.W1 = np.random.randn(self.input_size, self.hidden1_size).astype(np.float32) * s1
        self.b1 = np.zeros(self.hidden1_size, dtype=np.float32)

        s2 = np.sqrt(2.0 / (self.hidden1_size + self.hidden2_size))
        self.W2 = np.random.randn(self.hidden1_size, self.hidden2_size).astype(np.float32) * s2
        self.b2 = np.zeros(self.hidden2_size, dtype=np.float32)

        s3 = np.sqrt(2.0 / (self.hidden2_size + self.output_size))
        self.W3 = np.random.randn(self.hidden2_size, self.output_size).astype(np.float32) * s3
        self.b3 = np.zeros(self.output_size, dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Forward pass. Returns (output, cache)."""
        z1 = x @ self.W1 + self.b1
        h1 = np.tanh(z1)

        z2 = h1 @ self.W2 + self.b2
        h2 = np.tanh(z2)

        output = h2 @ self.W3 + self.b3

        return output, {'x': x, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}

    def backward(self, cache: Dict, error_grad: np.ndarray):
        """Backpropagation."""
        x, h1, h2 = cache['x'], cache['h1'], cache['h2']

        # Output layer
        dW3 = np.outer(h2, error_grad)
        db3 = error_grad
        dh2 = error_grad @ self.W3.T

        # Hidden 2
        dz2 = dh2 * (1 - h2 ** 2)
        dW2 = np.outer(h1, dz2)
        db2 = dz2
        dh1 = dz2 @ self.W2.T

        # Hidden 1
        dz1 = dh1 * (1 - h1 ** 2)
        dW1 = np.outer(x, dz1)
        db1 = dz1

        # Update
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def predict(self, target: FullBodyTarget) -> Dict[str, np.ndarray]:
        """
        Predict bone rotations for full body pose.

        Returns: Dict mapping bone name → euler XYZ (radians)
        """
        x = target.to_input_vector()
        output, _ = self.forward(x)

        bone_rotations = {}
        for i, bone_name in enumerate(CONTROLLED_BONES):
            bone_rotations[bone_name] = output[i*3 : i*3+3]

        return bone_rotations

    def train_step(self, target: FullBodyTarget, position_error_grad: np.ndarray):
        """
        Single training step with soft limit penalty.

        Args:
            target: The pose goal
            position_error_grad: Gradient from position error (shape 69,)
        """
        x = target.to_input_vector()
        output, cache = self.forward(x)

        # Compute limit penalty and its gradient
        limit_penalty, limit_grad = compute_limit_penalty(output)

        # Combined gradient
        total_grad = position_error_grad + limit_grad

        # Backprop
        self.backward(cache, total_grad)
        self.train_count += 1

        return limit_penalty

    def save(self, path: Optional[Path] = None):
        if path is None:
            path = DATA_DIR / "weights.npz"
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            train_count=self.train_count,
        )

    def load(self, path: Optional[Path] = None) -> bool:
        if path is None:
            path = DATA_DIR / "weights.npz"
        if not path.exists():
            return False
        try:
            data = np.load(path)
            self.W1, self.b1 = data['W1'], data['b1']
            self.W2, self.b2 = data['W2'], data['b2']
            self.W3, self.b3 = data['W3'], data['b3']
            self.train_count = int(data.get('train_count', 0))
            return True
        except:
            return False


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_network: Optional[FullBodyIKNetwork] = None


def get_network() -> FullBodyIKNetwork:
    global _network
    if _network is None:
        _network = FullBodyIKNetwork()
        _network.load()
    return _network


def reset_network():
    global _network
    _network = FullBodyIKNetwork()


# =============================================================================
# UTILITY: Euler to Quaternion
# =============================================================================

def euler_to_quat(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert euler XYZ (radians) to quaternion [w, x, y, z]."""
    cx = np.cos(euler_xyz[0] / 2)
    sx = np.sin(euler_xyz[0] / 2)
    cy = np.cos(euler_xyz[1] / 2)
    sy = np.sin(euler_xyz[1] / 2)
    cz = np.cos(euler_xyz[2] / 2)
    sz = np.sin(euler_xyz[2] / 2)

    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz

    return np.array([w, x, y, z], dtype=np.float32)


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to euler XYZ (radians)."""
    w, x, y, z = quat

    # Roll (X)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1, 1)
    pitch = np.arcsin(sinp)

    # Yaw (Z)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)
