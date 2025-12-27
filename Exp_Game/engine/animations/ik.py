# Exp_Game/engine/animations/ik.py
"""
Two-Bone IK Solver - NUMPY VECTORIZED, worker-safe, no bpy.

Analytical IK solver using law of cosines.
Used for leg/arm IK in animation system.

Architecture:
  - Runs in worker threads (no bpy dependency)
  - Returns local-space quaternions to apply to bones
  - Uses rig.md chain definitions

IMPORTANT - Root Bone Reference Frame:
  All IK target positions should be relative to the Root bone (world anchor).
  The rig hierarchy is: Root → Hips → (Spine/Legs)

  - Root: At origin (0,0,0), defines character's world position
  - Hips: Can translate/rotate for crouch/lean, children move with it
  - Leg IK: Targets are Root-relative, so feet stay planted when Hips moves

  Workflow for crouch:
    1. Move Hips down (relative to Root)
    2. Feet targets stay at ground level (Root-relative)
    3. Solve leg IK → legs bend to reach stationary foot targets
    4. Result: character crouches, feet don't move

Usage:
  upper_quat, lower_quat = solve_two_bone_ik(
      root_pos=hip_world_pos,
      target_pos=foot_target_pos,  # Relative to Root bone
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
        "len_upper": 0.448,   # Actual rig measurement
        "len_lower": 0.497,   # Actual rig measurement
        "reach": 0.945,       # Actual reach (upper + lower)
        "pole_forward": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    },
    "leg_R": {
        "root": "RightThigh",
        "mid": "RightShin",
        "tip": "RightFoot",
        "len_upper": 0.448,   # Actual rig measurement
        "len_lower": 0.497,   # Actual rig measurement
        "reach": 0.945,       # Actual reach (upper + lower)
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
    char_forward: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve IK for a leg - returns DIRECTIONS, not quaternions.

    The solver computes WHERE the joints should be. The main thread
    then uses Blender's actual bone data to compute rotations.

    Args:
        hip_pos: World position of the hip/thigh joint
        foot_target: World position where foot should reach
        knee_pole: World position of knee hint (in front of knee)
        side: "L" for left, "R" for right
        char_forward: Character's forward direction (CRITICAL for high kicks)

    Returns:
        Tuple of (thigh_dir, shin_dir, knee_pos)
        - thigh_dir: Direction vector for thigh (hip to knee)
        - shin_dir: Direction vector for shin (knee to foot)
        - knee_pos: World position of knee
    """
    chain = LEG_IK["leg_L"] if side == "L" else LEG_IK["leg_R"]
    upper_len = chain["len_upper"]
    lower_len = chain["len_lower"]

    hip_pos = np.asarray(hip_pos, dtype=np.float32)
    foot_target = np.asarray(foot_target, dtype=np.float32)
    knee_pole = np.asarray(knee_pole, dtype=np.float32)

    # Direction from hip to target
    reach_vec = foot_target - hip_pos
    reach_dist = float(np.linalg.norm(reach_vec))

    # Clamp to reachable distance
    max_reach = upper_len + lower_len - 0.001
    min_reach = abs(upper_len - lower_len) + 0.001
    clamped_dist = max(min_reach, min(reach_dist, max_reach))

    if reach_dist > 0.001:
        reach_dir = reach_vec / reach_dist
    else:
        # Fallback: straight down
        reach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # === LAW OF COSINES ===
    a, b, c = upper_len, lower_len, clamped_dist

    # Angle at hip (between thigh and reach line)
    cos_hip = (a*a + c*c - b*b) / (2*a*c)
    cos_hip = np.clip(cos_hip, -1.0, 1.0)
    hip_angle = np.arccos(cos_hip)

    # === COMPUTE KNEE POSITION ===
    # For legs, the knee must ALWAYS be in front of the body (+Y in character space)
    # regardless of where the foot target is. This is anatomically correct.
    #
    # The old approach (projecting pole perpendicular to reach) fails when
    # reaching UP and forward - the projection becomes tiny or points DOWN.
    #
    # New approach: Use char_forward directly to define the bend plane.
    # The bend axis is perpendicular to the plane containing (reach_dir, char_forward)

    if char_forward is None:
        char_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    char_forward = np.asarray(char_forward, dtype=np.float32)

    # Bend axis: perpendicular to the plane formed by reach and forward
    # This ensures knee always goes "forward" relative to character, not reach line
    bend_axis = normalize(cross(reach_dir, char_forward))

    # Handle edge case: reach is exactly aligned with char_forward
    if np.linalg.norm(bend_axis) < 0.001:
        # Fallback: use character's right vector as bend axis
        # (knee would go forward, which is correct)
        char_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        bend_axis = normalize(cross(char_forward, char_up))  # This gives char_right

    # Thigh direction: rotate reach_dir TOWARD pole by hip_angle
    # POSITIVE angle for legs (knee goes forward)
    thigh_dir = quat_rotate_vector(quat_from_axis_angle(bend_axis, hip_angle), reach_dir)

    # Knee position
    knee_pos = hip_pos + thigh_dir * upper_len

    # Shin direction: from knee to foot (clamped)
    clamped_foot = hip_pos + reach_dir * clamped_dist
    shin_dir = normalize(clamped_foot - knee_pos)

    return thigh_dir, shin_dir, knee_pos


def solve_arm_ik(
    shoulder_pos: np.ndarray,
    hand_target: np.ndarray,
    elbow_pole: np.ndarray,
    side: str = "L",
    char_forward: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve IK for an arm - returns POSITIONS only.

    The solver computes WHERE the joints should be. The main thread
    then uses Blender's actual bone data to compute rotations.

    Args:
        shoulder_pos: World position of the shoulder/upper arm joint
        hand_target: World position where hand should reach
        elbow_pole: World position of elbow hint (behind elbow)
        side: "L" for left, "R" for right
        char_forward: Character's forward direction (for elbow constraint)

    Returns:
        Tuple of (upper_dir, lower_dir, elbow_pos)
        - upper_dir: Direction vector for upper arm (shoulder to elbow)
        - lower_dir: Direction vector for forearm (elbow to hand)
        - elbow_pos: World position of elbow
    """
    chain = ARM_IK["arm_L"] if side == "L" else ARM_IK["arm_R"]
    upper_len = chain["len_upper"]
    lower_len = chain["len_lower"]

    shoulder_pos = np.asarray(shoulder_pos, dtype=np.float32)
    hand_target = np.asarray(hand_target, dtype=np.float32)
    elbow_pole = np.asarray(elbow_pole, dtype=np.float32)

    # Direction from shoulder to target
    reach_vec = hand_target - shoulder_pos
    reach_dist = float(np.linalg.norm(reach_vec))

    # Clamp to reachable distance
    max_reach = upper_len + lower_len - 0.001
    min_reach = abs(upper_len - lower_len) + 0.001
    clamped_dist = max(min_reach, min(reach_dist, max_reach))

    if reach_dist > 0.001:
        reach_dir = reach_vec / reach_dist
    else:
        reach_dir = np.array([-1.0, 0.0, 0.0] if side == "L" else [1.0, 0.0, 0.0], dtype=np.float32)

    # === LAW OF COSINES ===
    a, b, c = upper_len, lower_len, clamped_dist

    # Angle at shoulder (between upper arm and reach line)
    cos_shoulder = (a*a + c*c - b*b) / (2*a*c)
    cos_shoulder = np.clip(cos_shoulder, -1.0, 1.0)
    shoulder_angle = np.arccos(cos_shoulder)

    # === COMPUTE ELBOW POSITION ===
    # Pole vector defines where the elbow should go (BEHIND for arms)
    pole_vec = elbow_pole - shoulder_pos
    pole_vec = pole_vec - reach_dir * dot(pole_vec, reach_dir)  # Project perpendicular to reach

    if np.linalg.norm(pole_vec) > 0.001:
        pole_vec = normalize(pole_vec)
    else:
        # Fallback: elbow behind (-Y in character space)
        pole_vec = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        pole_vec = pole_vec - reach_dir * dot(pole_vec, reach_dir)
        if np.linalg.norm(pole_vec) > 0.001:
            pole_vec = normalize(pole_vec)
        else:
            pole_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # Bend axis (perpendicular to arm plane)
    # This axis is perpendicular to both the reach direction and pole direction
    bend_axis = normalize(cross(pole_vec, reach_dir))

    # Upper arm direction: rotate reach_dir TOWARD pole by shoulder_angle
    # NEGATIVE angle because right-hand rule: cross(pole, reach) points such that
    # negative rotation moves reach_dir toward pole_vec (elbow goes backward)
    upper_dir = quat_rotate_vector(quat_from_axis_angle(bend_axis, -shoulder_angle), reach_dir)

    # Elbow position
    elbow_pos = shoulder_pos + upper_dir * upper_len

    # =========================================================================
    # CROSS-BODY CONSTRAINT: Elbow must stay BEHIND shoulder in character space
    # =========================================================================
    # When reaching across the body, the "backward relative to arm line" direction
    # can actually be FORWARD in character space. This is anatomically impossible.
    # Enforce: elbow_forward_offset <= 0 (elbow behind or at shoulder level)

    if char_forward is not None:
        char_forward = np.asarray(char_forward, dtype=np.float32)
        char_forward = normalize(char_forward)

        # How far forward is elbow relative to shoulder?
        shoulder_to_elbow = elbow_pos - shoulder_pos
        forward_offset = float(np.dot(shoulder_to_elbow, char_forward))

        if forward_offset > 0.01:  # Elbow is in front of shoulder
            # Need to move elbow backward by rotating upper_dir around reach_dir
            # The elbow must stay on the "elbow sphere" (distance = upper_len from shoulder)

            # SAVE ORIGINAL - all rotations should be from this base!
            original_upper_dir = upper_dir.copy()

            # Strategy: Sample many angles around reach_dir, find the one that
            # puts elbow most backward while still being geometrically valid
            best_upper_dir = upper_dir
            best_elbow = elbow_pos
            best_forward = forward_offset

            # Sample rotation angles from 0 to 2*PI around reach_dir
            # More samples = better chance of finding optimal elbow position
            num_samples = 24  # Every 15 degrees
            for i in range(num_samples):
                test_angle = (2.0 * np.pi * i) / num_samples
                test_quat = quat_from_axis_angle(reach_dir, test_angle)
                test_upper_dir = quat_rotate_vector(test_quat, original_upper_dir)
                test_elbow = shoulder_pos + test_upper_dir * upper_len
                test_forward = float(np.dot(test_elbow - shoulder_pos, char_forward))

                if test_forward < best_forward:
                    best_forward = test_forward
                    best_upper_dir = test_upper_dir
                    best_elbow = test_elbow

            # Use the best position found (most backward)
            if best_forward < forward_offset - 0.001:
                upper_dir = best_upper_dir
                elbow_pos = best_elbow

    # Hand position (clamped to reach)
    hand_pos = shoulder_pos + reach_dir * clamped_dist

    # Lower arm direction (from elbow to hand)
    lower_dir = normalize(hand_pos - elbow_pos)

    # Return DIRECTIONS and POSITION - rotations computed in main thread
    return upper_dir.astype(np.float32), lower_dir.astype(np.float32), elbow_pos


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

    NOTE: All positions should be in Root-relative coordinates (character space).
    The Root bone is the world anchor at the character's feet.

    Args:
        foot_rest_pos: Current foot position from animation (Root-relative)
        ground_z: Ground height at this position (Root-relative, usually 0)
        foot_rest_z: Foot height in rest pose (ankle to floor)

    Returns:
        Target foot position for IK (Root-relative)
    """
    target = foot_rest_pos.copy()
    target[2] = ground_z + foot_rest_z
    return target


def compute_knee_pole_position(
    hip_pos: np.ndarray,
    foot_pos: np.ndarray,
    char_forward: np.ndarray = None,
    char_right: np.ndarray = None,
    side: str = "L",
    offset: float = 0.5,
) -> np.ndarray:
    """
    Compute anatomically-correct knee pole position.

    Human knees ALWAYS bend forward relative to the pelvis - this is anatomical,
    not dependent on where the foot is. Even when kicking backward, the knee
    bends forward.

    NOTE: All positions should be in Root-relative coordinates (character space).

    Args:
        hip_pos: Hip joint position (Root-relative)
        foot_pos: Foot position or target (Root-relative)
        char_forward: Character's forward direction (from pelvis/armature)
        char_right: Character's right direction
        side: "L" for left leg, "R" for right leg
        offset: Distance in front of the knee

    Returns:
        Pole target position (Root-relative)
    """
    hip_pos = np.asarray(hip_pos, dtype=np.float32)
    foot_pos = np.asarray(foot_pos, dtype=np.float32)

    # Fallback to world Y if no character orientation provided
    if char_forward is None:
        char_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if char_right is None:
        char_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    char_forward = np.asarray(char_forward, dtype=np.float32)
    char_right = np.asarray(char_right, dtype=np.float32)

    # SIMPLE APPROACH: Knee always bends forward + slightly outward
    # This is anatomically correct - knees don't bend sideways or backward

    # Slight outward bias (anatomical valgus - knees track over toes)
    outward_bias = 0.1
    if side == "L":
        outward = -char_right  # Left knee goes slightly left
    else:
        outward = char_right   # Right knee goes slightly right

    # Pole direction: forward + slight outward
    pole_dir = normalize(char_forward + outward * outward_bias)

    # Position pole in front of the knee area
    # Use hip position + forward offset (not midpoint, as that moves with target)
    # But we do want some vertical offset based on target height
    mid = (hip_pos + foot_pos) * 0.5

    # Pole is ALWAYS in front of character, at roughly knee height
    pole_pos = np.array([
        mid[0] + pole_dir[0] * offset,
        mid[1] + pole_dir[1] * offset,
        mid[2]  # Keep Z at midpoint height
    ], dtype=np.float32)

    return pole_pos


def compute_elbow_pole_position(
    shoulder_pos: np.ndarray,
    hand_pos: np.ndarray,
    char_forward: np.ndarray = None,
    char_up: np.ndarray = None,
    side: str = "L",
    offset: float = 0.3,
) -> np.ndarray:
    """
    Compute anatomically-correct elbow pole position.

    Human elbows bend AWAY from the reach direction:
    - Reaching forward → elbow goes back
    - Reaching up → elbow goes down/back
    - Reaching sideways → elbow goes back
    - Natural resting: elbow points back and slightly down

    NOTE: All positions should be in Root-relative coordinates (character space).

    Args:
        shoulder_pos: Shoulder joint position (Root-relative)
        hand_pos: Hand position or target (Root-relative)
        char_forward: Character's forward direction
        char_up: Character's up direction (usually world Z)
        side: "L" for left arm, "R" for right arm
        offset: Distance from elbow midpoint

    Returns:
        Pole target position (Root-relative)
    """
    shoulder_pos = np.asarray(shoulder_pos, dtype=np.float32)
    hand_pos = np.asarray(hand_pos, dtype=np.float32)

    # Fallbacks
    if char_forward is None:
        char_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if char_up is None:
        char_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    char_forward = np.asarray(char_forward, dtype=np.float32)
    char_up = np.asarray(char_up, dtype=np.float32)
    char_back = -char_forward
    char_down = -char_up

    # Reach direction (shoulder to hand)
    reach_vec = hand_pos - shoulder_pos
    reach_length = np.linalg.norm(reach_vec)

    if reach_length < 0.001:
        # Arm collapsed - use default back/down position
        mid = (shoulder_pos + hand_pos) * 0.5
        return mid + (char_back * 0.7 + char_down * 0.3) * offset

    reach_dir = reach_vec / reach_length

    # =========================================================================
    # RIG-AWARE ELBOW POLE COMPUTATION (from rig.md)
    # =========================================================================
    # Elbows ALWAYS bend BACKWARD in character space (pole_back: (0, -1, 0))
    # This is the anatomical standard - humans can't bend elbows forward.
    #
    # Primary direction: BACKWARD (char_back = -char_forward)
    # Secondary bias: OUTWARD (left arm → left, right arm → right)
    # =========================================================================

    char_right = np.cross(char_forward, char_up)
    char_right = normalize(char_right)

    # Start with strong BACKWARD direction - this is non-negotiable for human arms
    pole_dir = char_back * 1.0  # Primary: BACKWARD

    # Add outward bias - elbows also go slightly outward from body center
    if side == "L":
        pole_dir = pole_dir + (-char_right) * 0.5  # Left elbow goes LEFT
    else:
        pole_dir = pole_dir + char_right * 0.5     # Right elbow goes RIGHT

    # Slight downward bias for natural arm hanging position
    pole_dir = pole_dir + (-char_up) * 0.2

    pole_dir = normalize(pole_dir)

    # Ensure pole is on correct side for each arm (prevents elbow flip through body)
    if side == "L":
        # Left elbow must NOT be on right side of body
        if np.dot(pole_dir, char_right) > 0.1:
            # Pole is on wrong side, force it back/left
            pole_dir = pole_dir - char_right * (np.dot(pole_dir, char_right) + 0.2)
            pole_dir = normalize(pole_dir)
    else:
        # Right elbow must NOT be on left side of body
        if np.dot(pole_dir, char_right) < -0.1:
            # Pole is on wrong side, force it back/right
            pole_dir = pole_dir - char_right * (np.dot(pole_dir, char_right) - 0.2)
            pole_dir = normalize(pole_dir)

    # Position: midpoint of arm + offset in pole direction
    mid = (shoulder_pos + hand_pos) * 0.5
    return mid + pole_dir * offset


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
