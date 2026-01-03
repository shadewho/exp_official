# Exp_Game/animations/neural_network/forward_kinematics.py
"""
Forward Kinematics Computation (Pure NumPy)

Computes bone world positions from local rotations.
Used for FK loss during training - measures if predicted rotations
actually reach the target positions.

NO BPY IMPORTS - runs in engine workers and standalone trainer.
"""

import numpy as np
from typing import Tuple, Optional

# Support both package import (Blender) and direct import (standalone)
try:
    from .config import (
        NUM_BONES,
        PARENT_INDICES,
        LENGTHS_ARRAY,
        REST_POSITIONS_ARRAY,
        REST_ORIENTATIONS,
        LOCAL_OFFSETS,
        BONE_TO_INDEX,
        END_EFFECTORS,
        CONTROLLED_BONES,
    )
except ImportError:
    from config import (
        NUM_BONES,
        PARENT_INDICES,
        LENGTHS_ARRAY,
        REST_POSITIONS_ARRAY,
        REST_ORIENTATIONS,
        LOCAL_OFFSETS,
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

    IMPORTANT: Local rotations from Blender are in bone-local space.
    Each bone has a rest orientation that defines its local coordinate frame.
    The FK must account for these rest orientations to match Blender's transforms.

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

    # Convert local rotations to matrices (always axis-angle)
    if not use_axis_angle:
        raise ValueError("Only axis-angle rotations are supported")
    local_rots = axis_angle_to_matrix(rotations)  # (23, 3, 3)

    # Output arrays
    positions = np.zeros((NUM_BONES, 3), dtype=np.float32)
    world_rots = np.zeros((NUM_BONES, 3, 3), dtype=np.float32)

    # Process bones in order (parents before children)
    for i in range(NUM_BONES):
        parent_idx = PARENT_INDICES[i]

        if parent_idx < 0:
            # Root bone (Hips)
            # World rotation = input root rotation @ rest orientation @ local rotation
            # The rest orientation defines the bone's local frame
            # The local rotation rotates within that frame
            world_rots[i] = root_rotation @ REST_ORIENTATIONS[i] @ local_rots[i]
            positions[i] = root_position
        else:
            # Child bone
            parent_world_rot = world_rots[parent_idx]
            parent_pos = positions[parent_idx]

            # Position: parent position + parent's world rotation applied to local offset
            # LOCAL_OFFSETS[i] is the offset from parent to child in parent's local frame
            positions[i] = parent_pos + parent_world_rot @ LOCAL_OFFSETS[i]

            # World rotation: parent world @ relative rest orientation @ local rotation
            # relative_rest = how this bone's rest frame relates to parent's rest frame
            # relative_rest = parent_rest.T @ child_rest (transform from parent frame to child frame)
            parent_rest = REST_ORIENTATIONS[parent_idx]
            child_rest = REST_ORIENTATIONS[i]
            relative_rest = parent_rest.T @ child_rest

            world_rots[i] = parent_world_rot @ relative_rest @ local_rots[i]

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


def get_effector_rotations(
    all_rotations: np.ndarray,
) -> np.ndarray:
    """
    Extract end-effector rotation matrices from full bone rotations.

    Args:
        all_rotations: Shape (23, 3, 3) or (batch, 23, 3, 3)

    Returns:
        Shape (5, 3, 3) or (batch, 5, 3, 3) - rotation matrices of end effectors
    """
    effector_indices = [BONE_TO_INDEX[e] for e in END_EFFECTORS]

    if all_rotations.ndim == 3:
        return all_rotations[effector_indices]
    else:
        return all_rotations[:, effector_indices]


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrices to quaternions (w, x, y, z).

    Uses Shepperd's method for numerical stability.

    Args:
        R: Shape (3, 3) or (batch, 3, 3) or (batch, N, 3, 3)

    Returns:
        Quaternions with same leading dimensions, final dim = 4 (w, x, y, z)
    """
    original_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    n = R.shape[0]

    # Shepperd's method - choose largest diagonal to avoid numerical issues
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = np.zeros((n, 4), dtype=np.float32)

    # Case 1: trace > 0
    mask1 = trace > 0
    if np.any(mask1):
        s = np.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
        q[mask1, 0] = 0.25 * s  # w
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s  # x
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s  # y
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s  # z

    # Case 2: R[0,0] is largest diagonal
    mask2 = ~mask1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if np.any(mask2):
        s = np.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    # Case 3: R[1,1] is largest diagonal
    mask3 = ~mask1 & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
    if np.any(mask3):
        s = np.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    # Case 4: R[2,2] is largest diagonal
    mask4 = ~mask1 & ~mask2 & ~mask3
    if np.any(mask4):
        s = np.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        q[mask4, 3] = 0.25 * s

    # Normalize
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)

    return q.reshape(original_shape + (4,))


def quaternion_geodesic_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Compute geodesic distance between quaternions on SO(3).

    This is the proper rotation distance - the angle of the shortest rotation
    between two orientations. Ranges from 0 to pi.

    Args:
        q1, q2: Shape (..., 4) quaternions (w, x, y, z)

    Returns:
        Shape (...) geodesic angles in radians
    """
    # Dot product gives cos(angle/2) for unit quaternions
    dot = np.sum(q1 * q2, axis=-1)

    # Handle double-cover: q and -q represent the same rotation
    dot = np.abs(dot)

    # Clamp for numerical stability
    dot = np.clip(dot, -1.0, 1.0)

    # Geodesic distance = 2 * arccos(|dot|)
    return 2.0 * np.arccos(dot)


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (XYZ order) to quaternions.

    Args:
        euler: Shape (..., 3) Euler angles in radians

    Returns:
        Shape (..., 4) quaternions (w, x, y, z)
    """
    original_shape = euler.shape[:-1]
    euler = euler.reshape(-1, 3)

    cx = np.cos(euler[:, 0] / 2)
    sx = np.sin(euler[:, 0] / 2)
    cy = np.cos(euler[:, 1] / 2)
    sy = np.sin(euler[:, 1] / 2)
    cz = np.cos(euler[:, 2] / 2)
    sz = np.sin(euler[:, 2] / 2)

    # XYZ order: q = qz * qy * qx
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz

    q = np.stack([w, x, y, z], axis=-1)
    return q.reshape(original_shape + (4,))


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


def compute_fk_loss_with_orientation(
    predicted_rotations: np.ndarray,
    target_effector_positions: np.ndarray,
    target_effector_rotations: np.ndarray,
    root_positions: np.ndarray = None,
    root_rotations: np.ndarray = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute FK loss with both position AND orientation matching.

    Uses quaternion geodesic distance for orientation - proper SO(3) metric
    that avoids gimbal lock issues with Euler angles.

    Args:
        predicted_rotations: Shape (batch, 69) - flattened bone rotations
        target_effector_positions: Shape (batch, 5, 3) or (batch, 15) - target positions
        target_effector_rotations: Shape (batch, 5, 3) or (batch, 15) - target Euler rotations
        root_positions: Shape (batch, 3) - root world positions
        root_rotations: Shape (batch, 3, 3) - root world rotations

    Returns:
        position_loss: Scalar mean squared position error (meters²)
        orientation_loss: Scalar mean squared geodesic error (radians²)
        position_errors: Shape (batch, 5) - per-effector position distances
        orientation_errors: Shape (batch, 5) - per-effector geodesic angles
    """
    batch_size = predicted_rotations.shape[0]

    # Reshape rotations to (batch, 23, 3)
    rotations = predicted_rotations.reshape(batch_size, NUM_BONES, 3)

    # Reshape targets if needed
    if target_effector_positions.ndim == 2 and target_effector_positions.shape[1] == 15:
        target_effector_positions = target_effector_positions.reshape(batch_size, 5, 3)
    if target_effector_rotations.ndim == 2 and target_effector_rotations.shape[1] == 15:
        target_effector_rotations = target_effector_rotations.reshape(batch_size, 5, 3)

    # Run FK
    positions, world_rots = forward_kinematics_batch(rotations, root_positions, root_rotations)

    # Extract effector positions and rotations
    effector_pos = get_effector_positions(positions)  # (batch, 5, 3)
    effector_rots = get_effector_rotations(world_rots)  # (batch, 5, 3, 3)

    # Position error (MSE)
    pos_diff = effector_pos - target_effector_positions
    position_errors = np.linalg.norm(pos_diff, axis=-1)  # (batch, 5)
    position_loss = float(np.mean(position_errors ** 2))

    # Orientation error - quaternion geodesic distance (proper SO(3) metric)
    # Convert predicted rotation matrices to quaternions
    predicted_quats = rotation_matrix_to_quaternion(effector_rots)  # (batch, 5, 4)

    # Convert target Euler to quaternions
    target_quats = euler_to_quaternion(target_effector_rotations)  # (batch, 5, 4)

    # Geodesic distance on SO(3)
    orientation_errors = quaternion_geodesic_distance(predicted_quats, target_quats)  # (batch, 5)
    orientation_loss = float(np.mean(orientation_errors ** 2))

    return position_loss, orientation_loss, position_errors, orientation_errors


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
