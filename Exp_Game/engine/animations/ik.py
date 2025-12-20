# Exp_Game/engine/animations/ik.py
"""
Two-Bone IK Solver - NUMPY VECTORIZED, worker-safe, no bpy.

Analytical IK solver using law of cosines.
Used for leg/arm IK in animation system.

Architecture:
  - Runs in worker threads (no bpy dependency)
  - Returns local-space quaternions to apply to bones
  - Uses rig.md chain definitions

Usage:
  upper_quat, lower_quat = solve_two_bone_ik(
      root_pos=hip_world_pos,
      target_pos=foot_target_pos,
      pole_pos=knee_hint_pos,
      upper_len=0.495,  # thigh
      lower_len=0.478,  # shin
  )
"""

import numpy as np
from typing import Tuple, Optional

# =============================================================================
# CONSTANTS
# =============================================================================

# Rig chain definitions (from rig.md)
LEG_IK = {
    "leg_L": {
        "root": "LeftThigh",
        "mid": "LeftShin",
        "tip": "LeftFoot",
        "len_upper": 0.4947,
        "len_lower": 0.4784,
        "reach": 0.9731,
        "pole_forward": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    },
    "leg_R": {
        "root": "RightThigh",
        "mid": "RightShin",
        "tip": "RightFoot",
        "len_upper": 0.4947,
        "len_lower": 0.4775,
        "reach": 0.9722,
        "pole_forward": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    },
}

ARM_IK = {
    "arm_L": {
        "root": "LeftArm",
        "mid": "LeftForeArm",
        "tip": "LeftHand",
        "len_upper": 0.2782,
        "len_lower": 0.2863,
        "reach": 0.5645,
        "pole_back": np.array([0.0, -1.0, 0.0], dtype=np.float32),
    },
    "arm_R": {
        "root": "RightArm",
        "mid": "RightForeArm",
        "tip": "RightHand",
        "len_upper": 0.2782,
        "len_lower": 0.2863,
        "reach": 0.5645,
        "pole_back": np.array([0.0, -1.0, 0.0], dtype=np.float32),
    },
}

# =============================================================================
# VECTOR MATH UTILITIES
# =============================================================================


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector(s). Handles zero-length vectors safely."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-10)
    return v / norm


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product."""
    return np.cross(a, b)


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product."""
    return float(np.dot(a, b))


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create quaternion from axis-angle rotation.

    Args:
        axis: Normalized rotation axis (3,)
        angle: Rotation angle in radians

    Returns:
        Quaternion [w, x, y, z]
    """
    half_angle = angle * 0.5
    s = np.sin(half_angle)
    c = np.cos(half_angle)
    return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions: q1 * q2.

    Args:
        q1, q2: Quaternions [w, x, y, z]

    Returns:
        Result quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector by quaternion.

    Args:
        q: Quaternion [w, x, y, z]
        v: Vector (3,)

    Returns:
        Rotated vector (3,)
    """
    w, x, y, z = q

    # q * v * q^-1 (for unit quaternion, q^-1 = q conjugate)
    # Optimized formula
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """
    Create quaternion that rotates v_from to v_to.

    Args:
        v_from: Source direction (normalized)
        v_to: Target direction (normalized)

    Returns:
        Quaternion [w, x, y, z]
    """
    v_from = normalize(v_from)
    v_to = normalize(v_to)

    d = dot(v_from, v_to)

    # Vectors are nearly identical
    if d > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Vectors are opposite
    if d < -0.9999:
        # Find an orthogonal axis
        axis = cross(np.array([1.0, 0.0, 0.0]), v_from)
        if np.linalg.norm(axis) < 0.001:
            axis = cross(np.array([0.0, 1.0, 0.0]), v_from)
        axis = normalize(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float32)

    # General case
    axis = cross(v_from, v_to)
    s = np.sqrt((1.0 + d) * 2.0)
    inv_s = 1.0 / s

    return np.array([
        s * 0.5,
        axis[0] * inv_s,
        axis[1] * inv_s,
        axis[2] * inv_s,
    ], dtype=np.float32)


# =============================================================================
# TWO-BONE IK SOLVER
# =============================================================================


def solve_two_bone_ik(
    root_pos: np.ndarray,
    target_pos: np.ndarray,
    pole_pos: np.ndarray,
    upper_len: float,
    lower_len: float,
    initial_upper_dir: Optional[np.ndarray] = None,
    initial_lower_dir: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytical two-bone IK solver using law of cosines.

    Solves for the rotation of two bones (e.g., thigh + shin) to reach
    a target position, with a pole vector controlling the bend direction.

    Args:
        root_pos: World position of the root joint (e.g., hip) (3,)
        target_pos: World position of the target (e.g., foot) (3,)
        pole_pos: World position of the pole target (e.g., knee hint) (3,)
        upper_len: Length of upper bone (e.g., thigh)
        lower_len: Length of lower bone (e.g., shin)
        initial_upper_dir: Rest pose direction of upper bone (default: -Z)
        initial_lower_dir: Rest pose direction of lower bone (default: -Z)

    Returns:
        Tuple of:
        - upper_quat: Quaternion for upper bone (local space) [w, x, y, z]
        - lower_quat: Quaternion for lower bone (local space) [w, x, y, z]
        - mid_pos: World position of the middle joint (e.g., knee)
    """
    root_pos = np.asarray(root_pos, dtype=np.float32)
    target_pos = np.asarray(target_pos, dtype=np.float32)
    pole_pos = np.asarray(pole_pos, dtype=np.float32)

    # Default initial directions (legs point down in rest pose)
    if initial_upper_dir is None:
        initial_upper_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    if initial_lower_dir is None:
        initial_lower_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # Vector from root to target
    root_to_target = target_pos - root_pos
    target_dist = float(np.linalg.norm(root_to_target))

    # Handle edge cases
    max_reach = upper_len + lower_len - 0.001  # Tiny margin to avoid singularity
    min_reach = abs(upper_len - lower_len) + 0.001

    # Clamp target distance to valid range
    target_dist = max(min_reach, min(target_dist, max_reach))

    # Direction to target (normalized)
    if target_dist > 0.001:
        target_dir = root_to_target / target_dist
    else:
        target_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # ===========================================
    # LAW OF COSINES: Solve for joint angles
    # ===========================================
    #
    # Given: a = upper_len, b = lower_len, c = target_dist
    #
    # Upper bone angle (at root):
    #   cos(A) = (a^2 + c^2 - b^2) / (2 * a * c)
    #
    # Lower bone angle (at mid joint):
    #   cos(B) = (a^2 + b^2 - c^2) / (2 * a * b)

    a = upper_len
    b = lower_len
    c = target_dist

    # Angle at root (between upper bone and line to target)
    cos_angle_root = (a * a + c * c - b * b) / (2.0 * a * c)
    cos_angle_root = np.clip(cos_angle_root, -1.0, 1.0)
    angle_root = np.arccos(cos_angle_root)

    # Angle at mid joint (between upper and lower bones)
    cos_angle_mid = (a * a + b * b - c * c) / (2.0 * a * b)
    cos_angle_mid = np.clip(cos_angle_mid, -1.0, 1.0)
    angle_mid = np.arccos(cos_angle_mid)

    # ===========================================
    # POLE VECTOR: Determine bend direction
    # ===========================================

    # Project pole position onto the plane perpendicular to root-to-target
    pole_vec = pole_pos - root_pos
    pole_vec = pole_vec - target_dir * dot(pole_vec, target_dir)

    if np.linalg.norm(pole_vec) > 0.001:
        pole_vec = normalize(pole_vec)
    else:
        # Fallback: use forward direction for knee bend
        pole_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        pole_vec = pole_vec - target_dir * dot(pole_vec, target_dir)
        if np.linalg.norm(pole_vec) > 0.001:
            pole_vec = normalize(pole_vec)
        else:
            pole_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Rotation axis for bending (perpendicular to both target_dir and pole_vec)
    bend_axis = normalize(cross(target_dir, pole_vec))

    # ===========================================
    # COMPUTE BONE DIRECTIONS
    # ===========================================

    # Upper bone direction: rotate target_dir toward pole by angle_root
    upper_dir = quat_rotate_vector(quat_from_axis_angle(bend_axis, angle_root), target_dir)

    # Middle joint position
    mid_pos = root_pos + upper_dir * upper_len

    # Lower bone direction: points from mid to target
    lower_dir = normalize(target_pos - mid_pos)

    # ===========================================
    # COMPUTE LOCAL QUATERNIONS
    # ===========================================

    # Upper bone rotation: from rest pose to computed direction
    upper_quat = quat_from_two_vectors(initial_upper_dir, upper_dir)

    # Lower bone rotation: computed relative to upper bone
    # First, transform lower rest direction by upper rotation
    lower_rest_world = quat_rotate_vector(upper_quat, initial_lower_dir)
    # Then, find rotation from that to actual lower direction
    lower_quat = quat_from_two_vectors(lower_rest_world, lower_dir)

    return upper_quat, lower_quat, mid_pos


def solve_leg_ik(
    hip_pos: np.ndarray,
    foot_target: np.ndarray,
    knee_pole: np.ndarray,
    side: str = "L",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve IK for a leg using rig.md definitions.

    Args:
        hip_pos: World position of the hip/thigh joint
        foot_target: World position where foot should reach
        knee_pole: World position of knee hint (in front of knee)
        side: "L" for left, "R" for right

    Returns:
        Tuple of (thigh_quat, shin_quat, knee_pos)
    """
    chain = LEG_IK["leg_L"] if side == "L" else LEG_IK["leg_R"]

    return solve_two_bone_ik(
        root_pos=hip_pos,
        target_pos=foot_target,
        pole_pos=knee_pole,
        upper_len=chain["len_upper"],
        lower_len=chain["len_lower"],
        # Legs point down in rest pose
        initial_upper_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        initial_lower_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )


def solve_arm_ik(
    shoulder_pos: np.ndarray,
    hand_target: np.ndarray,
    elbow_pole: np.ndarray,
    side: str = "L",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve IK for an arm using rig.md definitions.

    Args:
        shoulder_pos: World position of the shoulder/upper arm joint
        hand_target: World position where hand should reach
        elbow_pole: World position of elbow hint (behind elbow)
        side: "L" for left, "R" for right

    Returns:
        Tuple of (upper_arm_quat, forearm_quat, elbow_pos)
    """
    chain = ARM_IK["arm_L"] if side == "L" else ARM_IK["arm_R"]

    # Arms point outward in T-pose
    arm_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float32) if side == "L" else np.array([1.0, 0.0, 0.0], dtype=np.float32)

    return solve_two_bone_ik(
        root_pos=shoulder_pos,
        target_pos=hand_target,
        pole_pos=elbow_pole,
        upper_len=chain["len_upper"],
        lower_len=chain["len_lower"],
        initial_upper_dir=arm_dir,
        initial_lower_dir=arm_dir,
    )


# =============================================================================
# FOOT GROUNDING
# =============================================================================


def compute_foot_ground_target(
    foot_rest_pos: np.ndarray,
    ground_z: float,
    foot_rest_z: float = 0.098,
) -> np.ndarray:
    """
    Compute foot target position for grounding IK.

    Args:
        foot_rest_pos: Current foot position from animation
        ground_z: Ground height at this position (from raycast)
        foot_rest_z: Foot height in rest pose (ankle to floor)

    Returns:
        Target foot position for IK
    """
    target = foot_rest_pos.copy()
    target[2] = ground_z + foot_rest_z
    return target


def compute_knee_pole_position(
    hip_pos: np.ndarray,
    foot_pos: np.ndarray,
    forward: np.ndarray = np.array([0.0, 1.0, 0.0]),
    offset: float = 0.5,
) -> np.ndarray:
    """
    Compute a default knee pole position in front of the leg.

    Args:
        hip_pos: Hip joint position
        foot_pos: Foot position (or target)
        forward: Character's forward direction
        offset: Distance in front of the knee

    Returns:
        Pole target position
    """
    mid = (hip_pos + foot_pos) * 0.5
    return mid + forward * offset


def compute_elbow_pole_position(
    shoulder_pos: np.ndarray,
    hand_pos: np.ndarray,
    backward: np.ndarray = np.array([0.0, -1.0, 0.0]),
    offset: float = 0.3,
) -> np.ndarray:
    """
    Compute a default elbow pole position behind the arm.

    Args:
        shoulder_pos: Shoulder joint position
        hand_pos: Hand position (or target)
        backward: Direction behind the character
        offset: Distance behind the elbow

    Returns:
        Pole target position
    """
    mid = (shoulder_pos + hand_pos) * 0.5
    return mid + backward * offset


# =============================================================================
# POSE MODIFICATION
# =============================================================================


def apply_ik_to_pose(
    pose: np.ndarray,
    bone_indices: dict,
    chain: str,
    upper_quat: np.ndarray,
    lower_quat: np.ndarray,
) -> np.ndarray:
    """
    Apply IK result to a pose array.

    Modifies the quaternion portion of the specified bones' transforms.

    Args:
        pose: (num_bones, 10) numpy array of transforms
        bone_indices: Dict mapping bone names to indices (from rig.md)
        chain: Chain name (e.g., "leg_L", "arm_R")
        upper_quat: Quaternion for upper bone [w, x, y, z]
        lower_quat: Quaternion for lower bone [w, x, y, z]

    Returns:
        Modified pose array (copy)
    """
    result = pose.copy()

    # Get chain definition
    if chain.startswith("leg"):
        chain_def = LEG_IK[chain]
    elif chain.startswith("arm"):
        chain_def = ARM_IK[chain]
    else:
        return result  # Unknown chain

    # Get bone indices
    upper_name = chain_def["root"]
    lower_name = chain_def["mid"]

    upper_idx = bone_indices.get(upper_name)
    lower_idx = bone_indices.get(lower_name)

    if upper_idx is None or lower_idx is None:
        return result  # Bones not found

    # Apply quaternions (first 4 floats of each bone's transform)
    result[upper_idx, 0:4] = upper_quat
    result[lower_idx, 0:4] = lower_quat

    return result
