# Exp_Game/animations/neural_network/context.py
"""
Context Extraction for Neural IK

Gathers environment information from Blender scene:
- Root-relative effector positions
- Ground/contact information
- Motion state

This module uses bpy - for data extraction only, not runtime.
"""

import bpy
import numpy as np
from mathutils import Vector, Matrix, Euler
from typing import Dict, Optional, Tuple

from .config import (
    END_EFFECTORS,
    CONTACT_EFFECTORS,
    INPUT_SIZE,
    INPUT_SLICES,
    TASK_TYPES,
    POSITION_SCALE,
    ROTATION_SCALE,
    HEIGHT_SCALE,
)


class ContextExtractor:
    """
    Extracts environment context for neural IK input.

    Converts Blender state into the 50-dimensional input vector:
    - Root-relative effector positions and rotations
    - Ground heights and normals
    - Contact flags
    - Motion phase and task type
    """

    def __init__(self, armature: bpy.types.Object):
        """
        Initialize context extractor.

        Args:
            armature: The target armature object
        """
        if armature is None or armature.type != 'ARMATURE':
            raise ValueError("Must provide valid armature")

        self.armature = armature
        self.ground_cache = {}  # Cache ground info per frame

    def extract(
        self,
        target_positions: Dict[str, Vector] = None,
        target_rotations: Dict[str, Euler] = None,
        ground_height: float = 0.0,
        ground_normal: Vector = None,
        contact_flags: Dict[str, bool] = None,
        motion_phase: float = 0.0,
        task_type: str = "idle",
    ) -> np.ndarray:
        """
        Extract full context vector for current armature state.

        Args:
            target_positions: Override positions for effectors (or use current)
            target_rotations: Override rotations for effectors (or use current)
            ground_height: Ground height at character position
            ground_normal: Ground surface normal (default up)
            contact_flags: Which feet should be grounded
            motion_phase: Animation phase 0-1
            task_type: Task name from TASK_TYPES

        Returns:
            Shape (50,) context vector ready for network input
        """
        pose_bones = self.armature.pose.bones
        arm_matrix = self.armature.matrix_world

        # Get root (Hips) transform
        hips_bone = pose_bones.get("Hips")
        if hips_bone is None:
            raise ValueError("Armature missing Hips bone")

        root_world_pos = arm_matrix @ hips_bone.head
        root_world_matrix = arm_matrix @ hips_bone.matrix

        # Root axes for relative transform
        root_forward = Vector(root_world_matrix.col[1][:3]).normalized()  # Y forward
        root_up = Vector(root_world_matrix.col[2][:3]).normalized()  # Z up
        root_right = root_forward.cross(root_up).normalized()

        # Build rotation matrix for world-to-root transform
        root_rot_matrix = Matrix([
            root_right,
            root_forward,
            root_up,
        ]).transposed()

        # =====================================================================
        # 1. ROOT-RELATIVE EFFECTOR TARGETS (30 values)
        # =====================================================================
        effector_data = []

        for effector_name in END_EFFECTORS:
            bone = pose_bones.get(effector_name)
            if bone is None:
                # Use zero if missing
                effector_data.extend([0.0] * 6)
                continue

            # Get world position/rotation
            if target_positions and effector_name in target_positions:
                world_pos = target_positions[effector_name]
            else:
                world_pos = arm_matrix @ bone.head

            if target_rotations and effector_name in target_rotations:
                world_rot = target_rotations[effector_name]
            else:
                bone_world_matrix = arm_matrix @ bone.matrix
                world_rot = bone_world_matrix.to_euler('XYZ')

            # Convert to root-relative
            relative_pos = root_rot_matrix @ (world_pos - root_world_pos)
            relative_rot = world_rot  # Keep world rotation for now

            effector_data.extend([
                relative_pos.x, relative_pos.y, relative_pos.z,
                relative_rot.x, relative_rot.y, relative_rot.z,
            ])

        # =====================================================================
        # 2. ROOT ORIENTATION (6 values)
        # =====================================================================
        root_data = [
            root_forward.x, root_forward.y, root_forward.z,
            root_up.x, root_up.y, root_up.z,
        ]

        # =====================================================================
        # 3. GROUND/CONTACT CONTEXT (12 values - 6 per foot)
        # =====================================================================
        if ground_normal is None:
            ground_normal = Vector((0, 0, 1))

        if contact_flags is None:
            contact_flags = {"LeftFoot": True, "RightFoot": True}

        ground_data = []
        for foot_name in CONTACT_EFFECTORS:
            foot_bone = pose_bones.get(foot_name)
            if foot_bone:
                foot_world_pos = arm_matrix @ foot_bone.head
                height_offset = foot_world_pos.z - ground_height
            else:
                height_offset = 0.0

            is_grounded = 1.0 if contact_flags.get(foot_name, True) else 0.0
            desired_contact = 1.0  # Always want contact when grounded flag set

            ground_data.extend([
                height_offset,
                ground_normal.x, ground_normal.y, ground_normal.z,
                is_grounded,
                desired_contact,
            ])

        # =====================================================================
        # 4. MOTION STATE (2 values)
        # =====================================================================
        task_value = float(TASK_TYPES.get(task_type, 0))
        motion_data = [motion_phase, task_value]

        # =====================================================================
        # COMBINE ALL
        # =====================================================================
        full_input = effector_data + root_data + ground_data + motion_data

        if len(full_input) != INPUT_SIZE:
            raise ValueError(f"Context size mismatch: got {len(full_input)}, expected {INPUT_SIZE}")

        return np.array(full_input, dtype=np.float32)

    def extract_from_current_pose(
        self,
        ground_height: float = 0.0,
        motion_phase: float = 0.0,
        task_type: str = "idle",
    ) -> np.ndarray:
        """
        Extract context from current armature pose.

        Convenience method that auto-detects ground contact.

        Args:
            ground_height: Ground Z height
            motion_phase: Animation phase 0-1
            task_type: Task name

        Returns:
            Shape (50,) context vector
        """
        pose_bones = self.armature.pose.bones
        arm_matrix = self.armature.matrix_world

        # Auto-detect contact (foot near ground)
        contact_flags = {}
        for foot_name in CONTACT_EFFECTORS:
            foot_bone = pose_bones.get(foot_name)
            if foot_bone:
                foot_z = (arm_matrix @ foot_bone.head).z
                contact_flags[foot_name] = (foot_z - ground_height) < 0.1
            else:
                contact_flags[foot_name] = True

        return self.extract(
            ground_height=ground_height,
            contact_flags=contact_flags,
            motion_phase=motion_phase,
            task_type=task_type,
        )


def extract_ground_targets(
    armature: bpy.types.Object,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract target effector positions and ground context from current pose.

    Returns:
        effector_positions: Shape (5, 3) - world positions of effectors
        ground_info: Shape (2, 6) - ground context per foot
    """
    pose_bones = armature.pose.bones
    arm_matrix = armature.matrix_world

    # Effector positions
    positions = []
    for effector_name in END_EFFECTORS:
        bone = pose_bones.get(effector_name)
        if bone:
            world_pos = arm_matrix @ bone.head
            positions.append([world_pos.x, world_pos.y, world_pos.z])
        else:
            positions.append([0.0, 0.0, 0.0])

    # Ground info (simple: assume flat ground at Z=0)
    ground_info = []
    for foot_name in CONTACT_EFFECTORS:
        foot_bone = pose_bones.get(foot_name)
        if foot_bone:
            foot_z = (arm_matrix @ foot_bone.head).z
            height_offset = foot_z  # Relative to Z=0
            is_grounded = 1.0 if foot_z < 0.1 else 0.0
        else:
            height_offset = 0.0
            is_grounded = 1.0

        ground_info.append([
            height_offset,
            0.0, 0.0, 1.0,  # Normal = up
            is_grounded,
            1.0,  # desired_contact
        ])

    return np.array(positions, dtype=np.float32), np.array(ground_info, dtype=np.float32)


def build_input_from_targets(
    effector_positions: np.ndarray,
    effector_rotations: np.ndarray,
    root_position: np.ndarray,
    root_forward: np.ndarray,
    root_up: np.ndarray,
    ground_info: np.ndarray,
    motion_phase: float = 0.0,
    task_type: float = 0.0,
) -> np.ndarray:
    """
    Build network input from components (no bpy needed).

    This is for runtime use in workers.

    Args:
        effector_positions: Shape (5, 3) - target positions (root-relative)
        effector_rotations: Shape (5, 3) - target rotations
        root_position: Shape (3,) - root world position (for reference)
        root_forward: Shape (3,) - root forward direction
        root_up: Shape (3,) - root up direction
        ground_info: Shape (2, 6) - ground context per foot
        motion_phase: 0-1 animation phase
        task_type: Task type as float

    Returns:
        Shape (50,) input vector
    """
    # Flatten effector data
    effector_data = np.concatenate([
        effector_positions.flatten(),
        effector_rotations.flatten(),
    ])[:30]  # 5 effectors × 6 = 30

    # Actually interleave pos/rot per effector
    effector_data = []
    for i in range(5):
        effector_data.extend(effector_positions[i])
        effector_data.extend(effector_rotations[i])

    # Root orientation
    root_data = np.concatenate([root_forward, root_up])

    # Ground data
    ground_data = ground_info.flatten()

    # Motion state
    motion_data = np.array([motion_phase, task_type])

    # Combine
    full_input = np.concatenate([
        np.array(effector_data),
        root_data,
        ground_data,
        motion_data,
    ])

    return full_input.astype(np.float32)


def augment_input(
    input_vector: np.ndarray,
    noise_scale: float = 0.01,
    rotation_noise: float = 0.02,
) -> np.ndarray:
    """
    Apply data augmentation to input vector.

    Args:
        input_vector: Shape (50,) or (batch, 50)
        noise_scale: Position noise magnitude
        rotation_noise: Rotation noise magnitude

    Returns:
        Augmented input vector
    """
    augmented = input_vector.copy()

    # Add noise to effector positions (first 30 values, alternating pos/rot)
    effector_start, effector_end = INPUT_SLICES['effectors']

    for i in range(5):  # 5 effectors
        base = effector_start + i * 6

        # Position noise (indices 0,1,2)
        if augmented.ndim == 1:
            augmented[base:base+3] += np.random.randn(3) * noise_scale
            augmented[base+3:base+6] += np.random.randn(3) * rotation_noise
        else:
            augmented[:, base:base+3] += np.random.randn(augmented.shape[0], 3) * noise_scale
            augmented[:, base+3:base+6] += np.random.randn(augmented.shape[0], 3) * rotation_noise

    return augmented


# =============================================================================
# INPUT NORMALIZATION
# =============================================================================
# Scale inputs to comparable magnitudes (~[-1, 1]) for stable training.

def normalize_input(input_vector: np.ndarray) -> np.ndarray:
    """
    Normalize input vector for network consumption.

    Scales positions and rotations to comparable magnitudes.
    This is CRITICAL for stable gradient flow during training.

    Args:
        input_vector: Shape (50,) or (batch, 50) - raw input

    Returns:
        Normalized input, same shape
    """
    normalized = input_vector.copy()
    single = normalized.ndim == 1
    if single:
        normalized = normalized.reshape(1, -1)

    # Effector data (30 values): alternating pos/rot per effector
    for i in range(5):
        base = i * 6
        # Position: divide by POSITION_SCALE (1m = 1 unit)
        normalized[:, base:base+3] /= POSITION_SCALE
        # Rotation: divide by pi to get ~[-1, 1]
        normalized[:, base+3:base+6] /= ROTATION_SCALE

    # Root orientation (6 values at indices 30-35): already unit vectors, skip

    # Ground context (12 values at indices 36-47)
    ground_start = INPUT_SLICES['ground'][0]
    for foot in range(2):
        foot_base = ground_start + foot * 6
        # Height offset: divide by height scale
        normalized[:, foot_base] /= HEIGHT_SCALE
        # Normal (3 values): already unit, skip
        # Contact flags (2 values): already 0/1, skip

    # Motion state (2 values at indices 48-49): phase is 0-1, task is small int, skip

    if single:
        normalized = normalized[0]

    return normalized


def denormalize_input(normalized_vector: np.ndarray) -> np.ndarray:
    """
    Reverse normalization to get original input values.

    Args:
        normalized_vector: Shape (50,) or (batch, 50) - normalized input

    Returns:
        Denormalized input, same shape
    """
    denormalized = normalized_vector.copy()
    single = denormalized.ndim == 1
    if single:
        denormalized = denormalized.reshape(1, -1)

    # Reverse effector normalization
    for i in range(5):
        base = i * 6
        denormalized[:, base:base+3] *= POSITION_SCALE
        denormalized[:, base+3:base+6] *= ROTATION_SCALE

    # Reverse ground height normalization
    ground_start = INPUT_SLICES['ground'][0]
    for foot in range(2):
        foot_base = ground_start + foot * 6
        denormalized[:, foot_base] *= HEIGHT_SCALE

    if single:
        denormalized = denormalized[0]

    return denormalized
