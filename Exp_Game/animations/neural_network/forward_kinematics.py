# Exp_Game/animations/neural_network/forward_kinematics.py
"""
Forward Kinematics Computation (Pure NumPy)

Computes bone world positions from local rotations.
Used for FK loss during training - measures if predicted rotations
actually reach the target positions.

NO BPY IMPORTS - runs in engine workers.
"""

import numpy as np
from typing import Tuple, Optional

from .config import (
    NUM_BONES,
    PARENT_INDICES,
    LENGTHS_ARRAY,
    REST_POSITIONS_ARRAY,
    BONE_TO_INDEX,
    END_EFFECTORS,
    CONTROLLED_BONES,
)


def axis_angle_to_matrix(axis_angles: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle rotations to rotation matrices.

    Args:
        axis_angles: Shape (N, 3) or (3,) - axis-angle vectors
                     The axis is the normalized direction, angle is the magnitude.

    Returns:
        Shape (N, 3, 3) or (3, 3) - rotation matrices
    """
    single = axis_angles.ndim == 1
    if single:
        axis_angles = axis_angles.reshape(1, -1)

    # Angle is the magnitude of the vector
    angles = np.linalg.norm(axis_angles, axis=1, keepdims=True)

    # Avoid division by zero
    safe_angles = np.where(angles > 1e-8, angles, 1e-8)

    # Normalized axis
    axes = axis_angles / safe_angles

    # Rodrigues' rotation formula
    # R = I + sin(θ)K + (1-cos(θ))K²
    # where K is the skew-symmetric matrix of axis

    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    angles = angles.squeeze(-1)

    c = np.cos(angles)
    s = np.sin(angles)
    t = 1 - c

    # Build rotation matrices
    n = len(axis_angles)
    R = np.zeros((n, 3, 3), dtype=np.float32)

    R[:, 0, 0] = t * x * x + c
    R[:, 0, 1] = t * x * y - s * z
    R[:, 0, 2] = t * x * z + s * y
    R[:, 1, 0] = t * x * y + s * z
    R[:, 1, 1] = t * y * y + c
    R[:, 1, 2] = t * y * z - s * x
    R[:, 2, 0] = t * x * z - s * y
    R[:, 2, 1] = t * y * z + s * x
    R[:, 2, 2] = t * z * z + c

    # For zero angles, return identity
    zero_mask = angles < 1e-8
    R[zero_mask] = np.eye(3, dtype=np.float32)

    if single:
        return R[0]
    return R


def euler_to_matrix(eulers: np.ndarray, order: str = 'XYZ') -> np.ndarray:
    """
    Convert Euler angles to rotation matrices.

    Args:
        eulers: Shape (N, 3) or (3,) - Euler angles in radians
        order: Rotation order (default XYZ)

    Returns:
        Shape (N, 3, 3) or (3, 3) - rotation matrices
    """
    single = eulers.ndim == 1
    if single:
        eulers = eulers.reshape(1, -1)

    n = len(eulers)

    # Compute sin/cos for each axis
    cx, sx = np.cos(eulers[:, 0]), np.sin(eulers[:, 0])
    cy, sy = np.cos(eulers[:, 1]), np.sin(eulers[:, 1])
    cz, sz = np.cos(eulers[:, 2]), np.sin(eulers[:, 2])

    # XYZ rotation order: Rz @ Ry @ Rx
    R = np.zeros((n, 3, 3), dtype=np.float32)

    R[:, 0, 0] = cy * cz
    R[:, 0, 1] = -cy * sz
    R[:, 0, 2] = sy
    R[:, 1, 0] = cx * sz + cz * sx * sy
    R[:, 1, 1] = cx * cz - sx * sy * sz
    R[:, 1, 2] = -cy * sx
    R[:, 2, 0] = sx * sz - cx * cz * sy
    R[:, 2, 1] = cz * sx + cx * sy * sz
    R[:, 2, 2] = cx * cy

    if single:
        return R[0]
    return R


def forward_kinematics(
    rotations: np.ndarray,
    root_position: np.ndarray = None,
    root_rotation: np.ndarray = None,
    use_axis_angle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute world positions and rotations for all bones.

    This is the core FK function. Given local rotations for each bone,
    compute where each bone ends up in world space.

    Args:
        rotations: Shape (23, 3) - local rotations per bone
                   Axis-angle if use_axis_angle=True, else Euler
        root_position: Shape (3,) - world position of Hips (default origin)
        root_rotation: Shape (3, 3) - world rotation of root (default identity)
        use_axis_angle: If True, interpret rotations as axis-angle

    Returns:
        positions: Shape (23, 3) - world positions of each bone
        world_rotations: Shape (23, 3, 3) - world rotation matrices
    """
    if root_position is None:
        root_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Default hips height

    if root_rotation is None:
        root_rotation = np.eye(3, dtype=np.float32)

    # Convert local rotations to matrices
    if use_axis_angle:
        local_rots = axis_angle_to_matrix(rotations)  # (23, 3, 3)
    else:
        local_rots = euler_to_matrix(rotations)  # (23, 3, 3)

    # Output arrays
    positions = np.zeros((NUM_BONES, 3), dtype=np.float32)
    world_rots = np.zeros((NUM_BONES, 3, 3), dtype=np.float32)

    # Process bones in order (parents before children)
    for i in range(NUM_BONES):
        parent_idx = PARENT_INDICES[i]

        if parent_idx < 0:
            # Root bone (Hips)
            world_rots[i] = root_rotation @ local_rots[i]
            positions[i] = root_position
        else:
            # Child bone - inherit parent transform
            parent_rot = world_rots[parent_idx]
            parent_pos = positions[parent_idx]

            # World rotation = parent rotation @ local rotation
            world_rots[i] = parent_rot @ local_rots[i]

            # Position = parent position + rotated bone offset
            # Use rest pose offset, rotated by parent
            rest_offset = REST_POSITIONS_ARRAY[i] - REST_POSITIONS_ARRAY[parent_idx]
            positions[i] = parent_pos + parent_rot @ rest_offset

    return positions, world_rots


def forward_kinematics_batch(
    rotations: np.ndarray,
    root_positions: np.ndarray = None,
    root_rotations: np.ndarray = None,
    use_axis_angle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch version of forward_kinematics.

    Args:
        rotations: Shape (batch, 23, 3) - local rotations
        root_positions: Shape (batch, 3) - root positions
        root_rotations: Shape (batch, 3, 3) - root rotation matrices

    Returns:
        positions: Shape (batch, 23, 3) - world positions
        world_rotations: Shape (batch, 23, 3, 3) - world rotation matrices
    """
    batch_size = rotations.shape[0]

    if root_positions is None:
        root_positions = np.tile([0.0, 0.0, 1.0], (batch_size, 1)).astype(np.float32)

    if root_rotations is None:
        root_rotations = np.tile(np.eye(3), (batch_size, 1, 1)).astype(np.float32)

    # Process each sample
    all_positions = np.zeros((batch_size, NUM_BONES, 3), dtype=np.float32)
    all_world_rots = np.zeros((batch_size, NUM_BONES, 3, 3), dtype=np.float32)

    for b in range(batch_size):
        pos, rots = forward_kinematics(
            rotations[b],
            root_positions[b],
            root_rotations[b],
            use_axis_angle=use_axis_angle,
        )
        all_positions[b] = pos
        all_world_rots[b] = rots

    return all_positions, all_world_rots


def get_effector_positions(
    all_positions: np.ndarray,
) -> np.ndarray:
    """
    Extract end-effector positions from full bone positions.

    Args:
        all_positions: Shape (23, 3) or (batch, 23, 3)

    Returns:
        Shape (5, 3) or (batch, 5, 3) - positions of end effectors
    """
    effector_indices = [BONE_TO_INDEX[e] for e in END_EFFECTORS]

    if all_positions.ndim == 2:
        return all_positions[effector_indices]
    else:
        return all_positions[:, effector_indices]


def compute_fk_loss(
    predicted_rotations: np.ndarray,
    target_effector_positions: np.ndarray,
    root_positions: np.ndarray = None,
    root_rotations: np.ndarray = None,
) -> Tuple[float, np.ndarray]:
    """
    Compute FK loss: how far are predicted effectors from targets?

    This is the PRIMARY training loss - it measures if the network's
    rotations actually reach the desired target positions.

    Args:
        predicted_rotations: Shape (batch, 69) - flattened bone rotations
        target_effector_positions: Shape (batch, 5, 3) or (batch, 15) - target positions
        root_positions: Shape (batch, 3) - root world positions
        root_rotations: Shape (batch, 3, 3) - root world rotations

    Returns:
        loss: Scalar mean squared error
        effector_errors: Shape (batch, 5) - per-effector distances
    """
    batch_size = predicted_rotations.shape[0]

    # Reshape rotations to (batch, 23, 3)
    rotations = predicted_rotations.reshape(batch_size, NUM_BONES, 3)

    # Reshape targets if needed
    if target_effector_positions.ndim == 2 and target_effector_positions.shape[1] == 15:
        target_effector_positions = target_effector_positions.reshape(batch_size, 5, 3)

    # Run FK
    positions, _ = forward_kinematics_batch(rotations, root_positions, root_rotations)

    # Extract effector positions
    effector_pos = get_effector_positions(positions)  # (batch, 5, 3)

    # Compute per-effector errors
    diff = effector_pos - target_effector_positions
    effector_errors = np.linalg.norm(diff, axis=-1)  # (batch, 5)

    # Mean squared error
    loss = float(np.mean(effector_errors ** 2))

    return loss, effector_errors


def compute_contact_loss(
    predicted_rotations: np.ndarray,
    ground_heights: np.ndarray,
    contact_flags: np.ndarray,
    root_positions: np.ndarray = None,
    ground_normals: np.ndarray = None,
    prev_foot_positions: np.ndarray = None,
    slip_weight: float = 0.2,
) -> Tuple[float, np.ndarray]:
    """
    Compute contact loss: feet should be on ground when grounded flag is set.

    Includes:
    - Height error: feet at ground height when grounded
    - Normal alignment: foot orientation aligned with ground normal
    - Slip penalty: feet shouldn't slide laterally when grounded

    Args:
        predicted_rotations: Shape (batch, 69) - flattened bone rotations
        ground_heights: Shape (batch, 2) - ground height at each foot
        contact_flags: Shape (batch, 2) - 1 if foot should be grounded
        root_positions: Shape (batch, 3) - root world positions
        ground_normals: Shape (batch, 2, 3) - ground normal per foot (optional)
        prev_foot_positions: Shape (batch, 2, 3) - previous foot positions for slip (optional)
        slip_weight: Weight for slip penalty term

    Returns:
        loss: Scalar contact violation
        foot_errors: Shape (batch, 2) - per-foot height errors
    """
    batch_size = predicted_rotations.shape[0]

    # Reshape rotations
    rotations = predicted_rotations.reshape(batch_size, NUM_BONES, 3)

    # Run FK
    positions, world_rots = forward_kinematics_batch(rotations, root_positions)

    # Get foot positions and rotations
    left_foot_idx = BONE_TO_INDEX["LeftFoot"]
    right_foot_idx = BONE_TO_INDEX["RightFoot"]

    foot_positions = np.stack([
        positions[:, left_foot_idx],  # (batch, 3)
        positions[:, right_foot_idx],
    ], axis=1)  # (batch, 2, 3)

    foot_heights = foot_positions[:, :, 2]  # (batch, 2) - Z is up

    # ==========================================================================
    # 1. HEIGHT ERROR (primary contact constraint)
    # ==========================================================================
    height_error = (foot_heights - ground_heights) * contact_flags
    height_loss = float(np.mean(height_error ** 2))

    # ==========================================================================
    # 2. NORMAL ALIGNMENT (optional - foot should align with ground)
    # ==========================================================================
    normal_loss = 0.0
    if ground_normals is not None:
        # Get foot up direction from rotation matrix (Z column)
        foot_rots = np.stack([
            world_rots[:, left_foot_idx],
            world_rots[:, right_foot_idx],
        ], axis=1)  # (batch, 2, 3, 3)

        foot_up = foot_rots[:, :, :, 2]  # (batch, 2, 3) - Z column = up

        # Dot product between foot up and ground normal (should be ~1)
        alignment = np.sum(foot_up * ground_normals, axis=-1)  # (batch, 2)
        normal_error = (1.0 - alignment) * contact_flags  # Only when grounded
        normal_loss = float(np.mean(normal_error ** 2)) * 0.1  # Lower weight

    # ==========================================================================
    # 3. SLIP PENALTY (feet shouldn't slide when grounded)
    # ==========================================================================
    slip_loss = 0.0
    if prev_foot_positions is not None:
        # Lateral movement (XY only, not height)
        foot_delta = foot_positions - prev_foot_positions  # (batch, 2, 3)
        lateral_delta = foot_delta[:, :, :2]  # (batch, 2, 2) - XY only

        # Slip = lateral movement when contact flag is set
        slip_magnitude = np.linalg.norm(lateral_delta, axis=-1)  # (batch, 2)
        slip_error = slip_magnitude * contact_flags
        slip_loss = float(np.mean(slip_error ** 2)) * slip_weight

    # Combined loss
    total_loss = height_loss + normal_loss + slip_loss

    return total_loss, np.abs(height_error)


def clamp_rotations(
    rotations: np.ndarray,
    limits_min: np.ndarray,
    limits_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clamp rotations to joint limits.

    Args:
        rotations: Shape (batch, 23, 3) or (23, 3) - bone rotations
        limits_min: Shape (23, 3) - minimum angles
        limits_max: Shape (23, 3) - maximum angles

    Returns:
        clamped: Same shape as rotations, values clamped
        violations: How much each joint was clamped
    """
    clamped = np.clip(rotations, limits_min, limits_max)
    violations = np.abs(rotations - clamped)
    return clamped, violations


# =============================================================================
# Gradient computation for FK loss (for training)
# =============================================================================

def compute_fk_loss_gradient(
    predicted_rotations: np.ndarray,
    target_effector_positions: np.ndarray,
    root_positions: np.ndarray = None,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """
    Compute gradient of FK loss with respect to rotations.
    Uses numerical differentiation (finite differences).

    Args:
        predicted_rotations: Shape (batch, 69) - flattened bone rotations
        target_effector_positions: Shape (batch, 15) - flattened target positions
        root_positions: Shape (batch, 3)
        epsilon: Step size for finite differences

    Returns:
        gradient: Shape (batch, 69) - gradient of loss w.r.t. rotations
    """
    batch_size = predicted_rotations.shape[0]
    n_params = predicted_rotations.shape[1]

    gradient = np.zeros_like(predicted_rotations)

    # Numerical gradient via central differences
    for i in range(n_params):
        # Forward step
        rotations_plus = predicted_rotations.copy()
        rotations_plus[:, i] += epsilon
        loss_plus, _ = compute_fk_loss(rotations_plus, target_effector_positions, root_positions)

        # Backward step
        rotations_minus = predicted_rotations.copy()
        rotations_minus[:, i] -= epsilon
        loss_minus, _ = compute_fk_loss(rotations_minus, target_effector_positions, root_positions)

        # Central difference
        gradient[:, i] = (loss_plus - loss_minus) / (2 * epsilon)

    return gradient
