# Exp_Game/engine/animations/ik_solver.py
"""
IK Solver - Core two-bone IK solving algorithms.

Worker-safe (NO bpy imports). Pure numpy math.

This is the SINGLE source of truth for IK solving. All IK computation
should flow through these functions.

SOLVE FUNCTIONS:
    solve_two_bone_ik() - Generic two-bone IK with pole vector
    solve_leg_ik()      - Leg-specific with knee-forward constraint
    solve_arm_ik()      - Arm-specific with elbow side preference

POLE FUNCTIONS:
    compute_knee_pole_position()  - Knee pole for natural leg bend

DIAGNOSTICS:
    get_last_solve_diagnostics() - Get detailed debug info from last solve
    clear_diagnostics()          - Clear diagnostic buffer
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

from .ik_math import (
    normalize,
    safe_normalize,
    quat_from_axis_angle,
    quat_from_two_vectors,
    quat_rotate_vector,
    quat_multiply,
    cross,
)
from .ik_chains import LEG_IK, ARM_IK, IK_TOLERANCES, get_chain


# =============================================================================
# DIAGNOSTIC SYSTEM - Captures solver decisions for debugging
# =============================================================================

_last_diagnostics: Dict[str, Any] = {}

def get_last_solve_diagnostics() -> Dict[str, Any]:
    """Get diagnostic data from the last IK solve. Returns copy."""
    return _last_diagnostics.copy()

def clear_diagnostics():
    """Clear diagnostic buffer."""
    global _last_diagnostics
    _last_diagnostics = {}

def _record_diagnostic(key: str, value: Any):
    """Record a diagnostic value."""
    _last_diagnostics[key] = value

def _fmt_vec(v) -> str:
    """Format vector for diagnostics."""
    if v is None:
        return "None"
    return f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})"


# =============================================================================
# CORE TWO-BONE IK SOLVER
# =============================================================================

def solve_two_bone_ik(
    root_pos: np.ndarray,
    target_pos: np.ndarray,
    pole_pos: np.ndarray,
    upper_len: float,
    lower_len: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve two-bone IK using law of cosines.

    The bones form a kinematic chain: root -> mid -> tip.
    Returns quaternions for the upper and lower bones, plus the mid position.

    Args:
        root_pos: World position of chain root (hip/shoulder)
        target_pos: World position of target (foot/hand)
        pole_pos: World position of pole target (controls bend direction)
        upper_len: Length of upper bone (thigh/upper arm)
        lower_len: Length of lower bone (shin/forearm)

    Returns:
        Tuple of:
            - upper_quat: Quaternion for upper bone [w, x, y, z]
            - lower_quat: Quaternion for lower bone [w, x, y, z]
            - mid_pos: World position of mid joint (knee/elbow)
    """
    root_pos = np.asarray(root_pos, dtype=np.float32)
    target_pos = np.asarray(target_pos, dtype=np.float32)
    pole_pos = np.asarray(pole_pos, dtype=np.float32)

    # Vector from root to target
    reach_vec = target_pos - root_pos
    reach_dist = float(np.linalg.norm(reach_vec))

    # Handle degenerate case (target at root)
    if reach_dist < 0.001:
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        mid_pos = root_pos + np.array([0, 0, -upper_len], dtype=np.float32)
        return identity, identity, mid_pos

    reach_dir = reach_vec / reach_dist

    # Clamp reach to valid range
    max_reach = upper_len + lower_len - 0.001
    min_reach = abs(upper_len - lower_len) + 0.001
    clamped_dist = max(min_reach, min(reach_dist, max_reach))

    # Law of cosines for angle at root (hip/shoulder)
    a, b, c = upper_len, lower_len, clamped_dist
    cos_root = (a * a + c * c - b * b) / (2 * a * c)
    cos_root = np.clip(cos_root, -1.0, 1.0)
    root_angle = np.arccos(cos_root)

    # Law of cosines for angle at mid (knee/elbow)
    cos_mid = (a * a + b * b - c * c) / (2 * a * b)
    cos_mid = np.clip(cos_mid, -1.0, 1.0)
    mid_angle = np.pi - np.arccos(cos_mid)

    # Build coordinate frame for the IK plane
    # X = perpendicular to reach_dir (toward pole)
    # Y = reach_dir
    # Z = perpendicular to both

    pole_vec = pole_pos - root_pos
    pole_on_reach = pole_vec - reach_dir * np.dot(pole_vec, reach_dir)
    pole_norm = float(np.linalg.norm(pole_on_reach))

    if pole_norm > 0.001:
        plane_x = pole_on_reach / pole_norm
    else:
        # Pole is on reach line - use arbitrary perpendicular
        up = np.array([0, 0, 1], dtype=np.float32)
        plane_x = normalize(cross(reach_dir, up))
        if np.linalg.norm(plane_x) < 0.001:
            right = np.array([1, 0, 0], dtype=np.float32)
            plane_x = normalize(cross(reach_dir, right))

    plane_z = normalize(cross(reach_dir, plane_x))

    # Upper bone direction: rotate reach_dir toward pole by root_angle
    upper_dir = reach_dir * np.cos(root_angle) + plane_x * np.sin(root_angle)
    upper_dir = normalize(upper_dir)

    # Mid joint position
    mid_pos = root_pos + upper_dir * upper_len

    # Lower bone direction (from mid to target)
    lower_dir = target_pos - mid_pos
    lower_dir = safe_normalize(lower_dir, reach_dir)

    # Convert directions to quaternions (assuming rest pose is Y-forward)
    rest_dir = np.array([0, 1, 0], dtype=np.float32)
    upper_quat = quat_from_two_vectors(rest_dir, upper_dir)
    lower_quat = quat_from_two_vectors(rest_dir, lower_dir)

    return upper_quat, lower_quat, mid_pos


# =============================================================================
# LEG IK SOLVER
# =============================================================================

def solve_leg_ik(
    hip_pos: np.ndarray,
    foot_target: np.ndarray,
    knee_pole: np.ndarray,
    side: str,
    char_forward: np.ndarray = None,
    char_up: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve leg IK with STRICT anatomical constraints.

    CRITICAL: Human knees can ONLY bend ONE WAY - forward. The shin can
    only rotate toward the BACK of the thigh. This is non-negotiable.

    This solver DIRECTLY COMPUTES the correct knee position instead of
    sampling and hoping to pick the right one.

    Args:
        hip_pos: World position of hip joint
        foot_target: World position of foot target
        knee_pole: Pole position for knee direction
        side: "L" or "R"
        char_forward: Character forward direction
        char_up: Character up direction

    Returns:
        Tuple of:
            - thigh_dir: Direction vector for thigh bone
            - shin_dir: Direction vector for shin bone
            - knee_pos: World position of knee joint
    """
    hip_pos = np.asarray(hip_pos, dtype=np.float32)
    foot_target = np.asarray(foot_target, dtype=np.float32)
    knee_pole = np.asarray(knee_pole, dtype=np.float32)

    if char_forward is None:
        char_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if char_up is None:
        char_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    char_forward = np.asarray(char_forward, dtype=np.float32)
    char_up = np.asarray(char_up, dtype=np.float32)
    char_right = cross(char_forward, char_up)
    char_down = -char_up

    # Get chain lengths
    chain = LEG_IK[f"leg_{side}"]
    upper_len = chain["len_upper"]
    lower_len = chain["len_lower"]

    # Vector from hip to foot
    reach_vec = foot_target - hip_pos
    reach_dist = float(np.linalg.norm(reach_vec))

    if reach_dist < 0.001:
        thigh_dir = char_down.copy()
        knee_pos = hip_pos + thigh_dir * upper_len
        shin_dir = char_down.copy()
        return thigh_dir, shin_dir, knee_pos

    reach_dir = reach_vec / reach_dist

    # Clamp to valid range
    max_reach = upper_len + lower_len - 0.001
    min_reach = abs(upper_len - lower_len) + 0.001
    clamped_dist = max(min_reach, min(reach_dist, max_reach))

    # Law of cosines for hip angle (angle between thigh and reach direction)
    a, b, c = upper_len, lower_len, clamped_dist
    cos_hip = (a * a + c * c - b * b) / (2 * a * c)
    cos_hip = np.clip(cos_hip, -1.0, 1.0)
    hip_angle = np.arccos(cos_hip)

    # =========================================================================
    # DETERMINE KICK TYPE
    # =========================================================================
    hip_z = float(np.dot(hip_pos, char_up))
    target_z = float(np.dot(foot_target, char_up))
    target_y = float(np.dot(foot_target - hip_pos, char_forward))
    target_x = float(np.dot(foot_target - hip_pos, char_right))

    height_above_hip = target_z - hip_z

    # ANY foot above hip level needs special handling - use low threshold
    is_elevated_kick = height_above_hip > 0.02  # 2cm above hip = elevated
    is_kick_forward = target_y > 0.05
    is_kick_backward = target_y < -0.05
    is_kick_sideways = abs(target_x) > 0.2

    # =========================================================================
    # DIRECT KNEE POSITION CALCULATION
    # =========================================================================
    # The knee must lie on a circle around the hip-foot axis.
    # We DIRECTLY pick the anatomically correct point on this circle.
    #
    # ANATOMY RULE: Knee ALWAYS bends FORWARD relative to character.
    # For ANY kick type, the knee must be in FRONT of the hip-foot line
    # in the character's forward direction.
    # =========================================================================

    # Build perpendicular vectors to reach direction
    # We want perp1 to be primarily in the forward direction
    forward_component = char_forward - reach_dir * np.dot(char_forward, reach_dir)
    forward_norm = float(np.linalg.norm(forward_component))

    if forward_norm > 0.001:
        # Use forward direction projected onto the plane perpendicular to reach
        perp1 = forward_component / forward_norm
    else:
        # Reach is parallel to forward - use down direction
        down_component = char_down - reach_dir * np.dot(char_down, reach_dir)
        down_norm = float(np.linalg.norm(down_component))
        if down_norm > 0.001:
            perp1 = down_component / down_norm
        else:
            perp1 = char_right.copy()

    perp2 = cross(reach_dir, perp1)
    perp2_len = float(np.linalg.norm(perp2))
    if perp2_len > 0.001:
        perp2 = perp2 / perp2_len

    # =========================================================================
    # FIND THE CORRECT KNEE POSITION
    # =========================================================================
    # For ANY elevated kick, we want the knee to be:
    # 1. FORWARD in character space (so shin bends back toward thigh, not away)
    # 2. BELOW the foot for front kicks (so shin goes UP)
    # 3. FORWARD of hip for back kicks (so shin goes BACK to foot)
    #
    # The knee circle is parameterized by angle around reach_dir.
    # knee_offset_from_reach = sin(hip_angle) * upper_len
    # The knee is at: hip + reach_dir * cos(hip_angle) * upper_len + perp * sin(hip_angle) * upper_len
    # =========================================================================

    knee_offset_dist = np.sin(hip_angle) * upper_len
    knee_along_reach = np.cos(hip_angle) * upper_len

    # The knee position for a given angle theta on the circle:
    # knee_pos = hip + reach_dir * knee_along_reach + (perp1 * cos(theta) + perp2 * sin(theta)) * knee_offset_dist

    # Since perp1 is aligned with forward direction (projected),
    # theta = 0 gives maximum forward knee position
    # theta = pi gives maximum backward knee position

    if is_elevated_kick:
        # For elevated kicks, we WANT knee forward
        # theta = 0 puts knee in the perp1 direction (forward)
        best_theta = 0.0

        # But we should verify this is actually forward
        test_knee = hip_pos + reach_dir * knee_along_reach + perp1 * knee_offset_dist
        knee_fwd = float(np.dot(test_knee - hip_pos, char_forward))

        if knee_fwd < 0.02:
            # perp1 didn't give us forward knee - try opposite
            best_theta = np.pi
            test_knee = hip_pos + reach_dir * knee_along_reach - perp1 * knee_offset_dist
            knee_fwd = float(np.dot(test_knee - hip_pos, char_forward))

            if knee_fwd < 0.02:
                # Neither worked - sample to find best forward position
                best_theta = 0.0
                best_fwd = -1000.0
                for i in range(32):
                    theta = (2.0 * np.pi * i) / 32
                    offset = perp1 * np.cos(theta) + perp2 * np.sin(theta)
                    test_knee = hip_pos + reach_dir * knee_along_reach + offset * knee_offset_dist
                    test_fwd = float(np.dot(test_knee - hip_pos, char_forward))
                    if test_fwd > best_fwd:
                        best_fwd = test_fwd
                        best_theta = theta

        offset_dir = perp1 * np.cos(best_theta) + perp2 * np.sin(best_theta)
        knee_pos = hip_pos + reach_dir * knee_along_reach + offset_dir * knee_offset_dist

    else:
        # Normal stance - use pole vector to guide knee
        pole_vec = knee_pole - hip_pos
        pole_projected = pole_vec - reach_dir * np.dot(pole_vec, reach_dir)
        pole_norm = float(np.linalg.norm(pole_projected))

        if pole_norm > 0.001:
            offset_dir = pole_projected / pole_norm
        else:
            offset_dir = perp1

        knee_pos = hip_pos + reach_dir * knee_along_reach + offset_dir * knee_offset_dist

        # Still ensure knee is forward for normal stance
        knee_fwd = float(np.dot(knee_pos - hip_pos, char_forward))
        if knee_fwd < 0.01:
            # Force forward
            knee_pos = knee_pos + char_forward * (0.05 - knee_fwd)

    # =========================================================================
    # FINAL HARD ENFORCEMENT
    # =========================================================================
    # After all calculations, VERIFY the knee is forward.
    # If not, FORCE it forward. This is non-negotiable.
    # =========================================================================

    hip_to_knee = knee_pos - hip_pos
    knee_forward_final = float(np.dot(hip_to_knee, char_forward))

    # Minimum forward offset based on kick type
    if is_elevated_kick:
        min_forward = 0.08  # 8cm for elevated kicks
    else:
        min_forward = 0.02  # 2cm for normal stance

    if knee_forward_final < min_forward:
        # FORCE knee forward - this overrides everything
        correction = min_forward - knee_forward_final + 0.03
        knee_pos = knee_pos + char_forward * correction

    # Recalculate thigh direction from corrected knee position
    thigh_dir = knee_pos - hip_pos
    thigh_len = float(np.linalg.norm(thigh_dir))
    if thigh_len > 0.001:
        thigh_dir = thigh_dir / thigh_len
    else:
        thigh_dir = char_down.copy()

    # Scale knee position to correct thigh length
    knee_pos = hip_pos + thigh_dir * upper_len

    # Shin direction (from knee to foot)
    shin_dir = foot_target - knee_pos
    shin_len = float(np.linalg.norm(shin_dir))
    if shin_len > 0.001:
        shin_dir = shin_dir / shin_len
    else:
        shin_dir = char_down.copy()

    # =========================================================================
    # RECORD DIAGNOSTICS
    # =========================================================================
    diag_key = f"leg_{side}"
    _record_diagnostic(f"{diag_key}_hip_pos", _fmt_vec(hip_pos))
    _record_diagnostic(f"{diag_key}_foot_target", _fmt_vec(foot_target))
    _record_diagnostic(f"{diag_key}_knee_pole", _fmt_vec(knee_pole))
    _record_diagnostic(f"{diag_key}_char_forward", _fmt_vec(char_forward))
    _record_diagnostic(f"{diag_key}_char_up", _fmt_vec(char_up))
    _record_diagnostic(f"{diag_key}_reach_dist", round(reach_dist, 3))
    _record_diagnostic(f"{diag_key}_height_above_hip", round(height_above_hip, 3))
    _record_diagnostic(f"{diag_key}_is_elevated_kick", is_elevated_kick)
    _record_diagnostic(f"{diag_key}_is_kick_forward", is_kick_forward)
    _record_diagnostic(f"{diag_key}_is_kick_backward", is_kick_backward)
    _record_diagnostic(f"{diag_key}_is_kick_sideways", is_kick_sideways)
    _record_diagnostic(f"{diag_key}_knee_pos_final", _fmt_vec(knee_pos))
    _record_diagnostic(f"{diag_key}_knee_forward_final", round(knee_forward_final, 3))
    _record_diagnostic(f"{diag_key}_min_forward_required", round(min_forward, 3))
    _record_diagnostic(f"{diag_key}_thigh_dir", _fmt_vec(thigh_dir))
    _record_diagnostic(f"{diag_key}_shin_dir", _fmt_vec(shin_dir))
    _record_diagnostic(f"{diag_key}_correction_applied", knee_forward_final < min_forward)

    return thigh_dir, shin_dir, knee_pos


# =============================================================================
# ARM IK SOLVER
# =============================================================================

def solve_arm_ik(
    shoulder_pos: np.ndarray,
    hand_target: np.ndarray,
    elbow_pole: np.ndarray,
    side: str,
    char_forward: np.ndarray = None,
    char_up: np.ndarray = None,
    spine_x: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve arm IK with STRICT anatomical constraints.

    CRITICAL RULES (NON-NEGOTIABLE):
    1. Elbow MUST stay on correct side (L elbow on left, R elbow on right)
    2. When reaching UP, elbow MUST be BELOW the hand (pointing back/down)
    3. Hand cannot cross too far past spine centerline

    This solver DIRECTLY COMPUTES the correct elbow position instead of
    sampling and hoping to pick the right one.

    Args:
        shoulder_pos: World position of shoulder joint
        hand_target: World position of hand target
        elbow_pole: Pole position for elbow direction hint
        side: "L" or "R"
        char_forward: Character forward direction
        char_up: Character up direction
        spine_x: X position of spine centerline (for crossover clamping)

    Returns:
        Tuple of:
            - upper_dir: Direction vector for upper arm
            - forearm_dir: Direction vector for forearm
            - elbow_pos: World position of elbow joint
    """
    shoulder_pos = np.asarray(shoulder_pos, dtype=np.float32)
    hand_target = np.asarray(hand_target, dtype=np.float32)
    elbow_pole = np.asarray(elbow_pole, dtype=np.float32)

    if char_forward is None:
        char_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if char_up is None:
        char_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    char_forward = np.asarray(char_forward, dtype=np.float32)
    char_up = np.asarray(char_up, dtype=np.float32)
    char_right = cross(char_forward, char_up)
    char_down = -char_up
    char_back = -char_forward

    # Outward direction based on side (left arm = -X, right arm = +X)
    char_outward = -char_right if side == "L" else char_right
    char_inward = -char_outward

    # Get chain lengths
    chain = ARM_IK[f"arm_{side}"]
    upper_len = chain["len_upper"]
    lower_len = chain["len_lower"]

    # =========================================================================
    # SPINE CROSSOVER CONSTRAINT
    # =========================================================================
    hand_offset = hand_target - shoulder_pos
    hand_x = float(np.dot(hand_offset, char_right))
    hand_y = float(np.dot(hand_offset, char_forward))
    hand_z = float(np.dot(hand_target, char_up))

    # Height-based crossover limit
    chest_height = 1.35
    belly_height = 1.0
    max_crossover_chest = 0.30
    max_crossover_belly = 0.05

    if hand_z >= chest_height:
        max_crossover = max_crossover_chest
    elif hand_z <= belly_height:
        max_crossover = max_crossover_belly
    else:
        t = (hand_z - belly_height) / (chest_height - belly_height)
        max_crossover = max_crossover_belly + t * (max_crossover_chest - max_crossover_belly)

    # Behind body = stricter limit
    if hand_y < -0.05:
        behind_factor = min(1.0, abs(hand_y) / 0.3)
        max_crossover = max_crossover * (1.0 - behind_factor * 0.8)

    # Clamp hand target to prevent spine crossing
    if side == "L":
        if hand_x > max_crossover:
            hand_target = hand_target - char_right * (hand_x - max_crossover)
    else:
        if hand_x < -max_crossover:
            hand_target = hand_target - char_right * (hand_x + max_crossover)

    # =========================================================================
    # BASIC GEOMETRY
    # =========================================================================
    reach_vec = hand_target - shoulder_pos
    reach_dist = float(np.linalg.norm(reach_vec))

    if reach_dist < 0.001:
        upper_dir = char_outward.copy()
        elbow_pos = shoulder_pos + upper_dir * upper_len
        forearm_dir = char_down.copy()
        return upper_dir, forearm_dir, elbow_pos

    reach_dir = reach_vec / reach_dist

    # Clamp to valid range
    max_reach = upper_len + lower_len - 0.001
    min_reach = abs(upper_len - lower_len) + 0.001
    clamped_dist = max(min_reach, min(reach_dist, max_reach))

    # Law of cosines for shoulder angle
    a, b, c = upper_len, lower_len, clamped_dist
    cos_shoulder = (a * a + c * c - b * b) / (2 * a * c)
    cos_shoulder = np.clip(cos_shoulder, -1.0, 1.0)
    shoulder_angle = np.arccos(cos_shoulder)

    elbow_offset_dist = np.sin(shoulder_angle) * upper_len
    elbow_along_reach = np.cos(shoulder_angle) * upper_len

    # =========================================================================
    # ANALYZE REACH TYPE
    # =========================================================================
    shoulder_z = float(np.dot(shoulder_pos, char_up))
    target_z = float(np.dot(hand_target, char_up))
    target_y = float(np.dot(hand_target - shoulder_pos, char_forward))
    target_x = float(np.dot(hand_target - shoulder_pos, char_right))

    height_above_shoulder = target_z - shoulder_z

    # Use very low threshold - ANY reach near or above shoulder needs care
    is_reaching_up = height_above_shoulder > 0.0
    is_reaching_behind = target_y < -0.05
    is_reaching_down = height_above_shoulder < -0.25

    # =========================================================================
    # BUILD PERPENDICULAR VECTORS FOR ELBOW CIRCLE
    # =========================================================================
    # We want perp1 to be in the BACK+DOWN direction for most reaches,
    # and perp2 to be in the OUTWARD direction.

    # For reaching UP: elbow should go BACK and DOWN
    # For reaching FORWARD: elbow should go BACK and DOWN
    # For reaching BEHIND: elbow should go OUTWARD and DOWN

    if is_reaching_behind:
        # Behind reaches: elbow should point outward
        preferred_dir = char_outward + char_down * 0.3
    else:
        # All other reaches: elbow should point back and down
        # Scale down component by how high we're reaching
        down_weight = max(0.2, 1.0 - height_above_shoulder * 2.0)
        preferred_dir = char_back * 0.7 + char_down * down_weight + char_outward * 0.3

    preferred_dir = preferred_dir / (float(np.linalg.norm(preferred_dir)) + 0.001)

    # Project preferred direction onto plane perpendicular to reach
    preferred_projected = preferred_dir - reach_dir * np.dot(preferred_dir, reach_dir)
    preferred_norm = float(np.linalg.norm(preferred_projected))

    if preferred_norm > 0.001:
        perp1 = preferred_projected / preferred_norm
    else:
        # Fallback: use outward direction
        outward_projected = char_outward - reach_dir * np.dot(char_outward, reach_dir)
        outward_norm = float(np.linalg.norm(outward_projected))
        if outward_norm > 0.001:
            perp1 = outward_projected / outward_norm
        else:
            perp1 = char_back.copy()

    perp2 = cross(reach_dir, perp1)
    perp2_norm = float(np.linalg.norm(perp2))
    if perp2_norm > 0.001:
        perp2 = perp2 / perp2_norm

    # =========================================================================
    # DIRECT ELBOW POSITION CALCULATION
    # =========================================================================
    # Start with elbow in the preferred direction (perp1)
    elbow_pos = shoulder_pos + reach_dir * elbow_along_reach + perp1 * elbow_offset_dist

    # =========================================================================
    # VERIFY AND CORRECT ELBOW POSITION
    # =========================================================================

    elbow_offset = elbow_pos - shoulder_pos
    elbow_z = float(np.dot(elbow_pos, char_up))
    elbow_y = float(np.dot(elbow_offset, char_forward))
    elbow_outward = float(np.dot(elbow_offset, char_outward))

    needs_correction = False
    best_theta = 0.0

    # CHECK 1: Elbow must be on correct side
    if elbow_outward < 0.01:
        needs_correction = True

    # CHECK 2: For reaching up, elbow must be below hand
    if is_reaching_up and elbow_z > target_z - 0.05:
        needs_correction = True

    # CHECK 3: Elbow shouldn't point forward when reaching up/forward
    if not is_reaching_behind and elbow_y > 0.05:
        needs_correction = True

    if needs_correction:
        # Sample the elbow circle to find position that satisfies ALL constraints
        best_elbow = elbow_pos
        best_score = -1e10

        for i in range(32):
            theta = (2.0 * np.pi * i) / 32
            offset_dir = perp1 * np.cos(theta) + perp2 * np.sin(theta)
            test_elbow = shoulder_pos + reach_dir * elbow_along_reach + offset_dir * elbow_offset_dist

            test_offset = test_elbow - shoulder_pos
            test_z = float(np.dot(test_elbow, char_up))
            test_y = float(np.dot(test_offset, char_forward))
            test_outward = float(np.dot(test_offset, char_outward))

            score = 0.0

            # RULE 1: Elbow on correct side (CRITICAL)
            if test_outward < 0.01:
                score -= 1000.0  # Massive penalty
            else:
                score += test_outward * 50.0  # Reward outward

            # RULE 2: For reaching up, elbow below hand
            if is_reaching_up:
                if test_z > target_z - 0.05:
                    score -= 500.0  # Big penalty for elbow above hand
                else:
                    score += (target_z - test_z) * 30.0  # Reward being below

            # RULE 3: Elbow should be back, not forward
            if not is_reaching_behind:
                if test_y > 0.05:
                    score -= 200.0
                else:
                    score += (-test_y) * 20.0  # Reward being back

            if score > best_score:
                best_score = score
                best_elbow = test_elbow

        elbow_pos = best_elbow

    # =========================================================================
    # FINAL HARD ENFORCEMENT (NON-NEGOTIABLE)
    # =========================================================================

    # ENFORCE 1: Elbow MUST be on correct side
    elbow_offset = elbow_pos - shoulder_pos
    elbow_outward_final = float(np.dot(elbow_offset, char_outward))

    min_outward = 0.03  # 3cm minimum on correct side
    if elbow_outward_final < min_outward:
        correction = (min_outward - elbow_outward_final + 0.02)
        elbow_pos = elbow_pos + char_outward * correction

    # ENFORCE 2: For reaching up, elbow MUST be below hand
    if is_reaching_up:
        elbow_z_final = float(np.dot(elbow_pos, char_up))
        max_elbow_z = target_z - 0.08  # At least 8cm below hand

        if elbow_z_final > max_elbow_z:
            # Push elbow down
            correction = elbow_z_final - max_elbow_z + 0.03
            elbow_pos = elbow_pos - char_up * correction

    # ENFORCE 3: Elbow should not be forward for non-behind reaches
    if not is_reaching_behind:
        elbow_offset = elbow_pos - shoulder_pos
        elbow_y_final = float(np.dot(elbow_offset, char_forward))
        if elbow_y_final > 0.02:
            # Push elbow back
            elbow_pos = elbow_pos - char_forward * (elbow_y_final + 0.02)

    # =========================================================================
    # COMPUTE FINAL DIRECTIONS
    # =========================================================================

    # Upper arm direction
    upper_dir = elbow_pos - shoulder_pos
    upper_len_actual = float(np.linalg.norm(upper_dir))
    if upper_len_actual > 0.001:
        upper_dir = upper_dir / upper_len_actual
    else:
        upper_dir = char_outward.copy()

    # Scale elbow to correct distance
    elbow_pos = shoulder_pos + upper_dir * upper_len

    # Forearm direction
    forearm_dir = hand_target - elbow_pos
    forearm_len = float(np.linalg.norm(forearm_dir))
    if forearm_len > 0.001:
        forearm_dir = forearm_dir / forearm_len
    else:
        forearm_dir = char_down.copy()

    # =========================================================================
    # RECORD DIAGNOSTICS
    # =========================================================================
    diag_key = f"arm_{side}"
    _record_diagnostic(f"{diag_key}_shoulder_pos", _fmt_vec(shoulder_pos))
    _record_diagnostic(f"{diag_key}_hand_target", _fmt_vec(hand_target))
    _record_diagnostic(f"{diag_key}_elbow_pole", _fmt_vec(elbow_pole))
    _record_diagnostic(f"{diag_key}_char_forward", _fmt_vec(char_forward))
    _record_diagnostic(f"{diag_key}_char_up", _fmt_vec(char_up))
    _record_diagnostic(f"{diag_key}_reach_dist", round(reach_dist, 3))
    _record_diagnostic(f"{diag_key}_height_above_shoulder", round(height_above_shoulder, 3))
    _record_diagnostic(f"{diag_key}_is_reaching_up", is_reaching_up)
    _record_diagnostic(f"{diag_key}_is_reaching_behind", is_reaching_behind)
    _record_diagnostic(f"{diag_key}_is_reaching_down", is_reaching_down)
    _record_diagnostic(f"{diag_key}_elbow_pos_final", _fmt_vec(elbow_pos))
    _record_diagnostic(f"{diag_key}_elbow_outward_final", round(elbow_outward_final, 3))
    _record_diagnostic(f"{diag_key}_upper_dir", _fmt_vec(upper_dir))
    _record_diagnostic(f"{diag_key}_forearm_dir", _fmt_vec(forearm_dir))
    # Compute elbow height relative to hand for diagnostics
    elbow_z_diag = float(np.dot(elbow_pos, char_up))
    _record_diagnostic(f"{diag_key}_elbow_z", round(elbow_z_diag, 3))
    _record_diagnostic(f"{diag_key}_target_z", round(target_z, 3))
    _record_diagnostic(f"{diag_key}_elbow_below_hand", elbow_z_diag < target_z)

    return upper_dir, forearm_dir, elbow_pos


# =============================================================================
# POLE POSITION COMPUTATION
# =============================================================================

def compute_knee_pole_position(
    hip_pos: np.ndarray,
    foot_pos: np.ndarray,
    char_forward: np.ndarray,
    char_right: np.ndarray,
    side: str,
    offset: float = 0.5,
) -> np.ndarray:
    """
    Compute pole position for knee IK.

    Places pole in front of the midpoint between hip and foot,
    ensuring the knee bends forward naturally.

    Args:
        hip_pos: Hip joint world position
        foot_pos: Foot target world position
        char_forward: Character forward direction
        char_right: Character right direction
        side: "L" or "R"
        offset: Distance of pole from leg line

    Returns:
        World position for knee pole target
    """
    hip_pos = np.asarray(hip_pos, dtype=np.float32)
    foot_pos = np.asarray(foot_pos, dtype=np.float32)
    char_forward = np.asarray(char_forward, dtype=np.float32)

    # Midpoint of leg
    mid = (hip_pos + foot_pos) * 0.5

    # Pole goes FORWARD from midpoint (knees bend forward)
    pole_dir = normalize(char_forward)

    # Small outward bias for natural stance
    if side == "L":
        pole_dir = pole_dir - char_right * 0.1
    else:
        pole_dir = pole_dir + char_right * 0.1

    pole_dir = normalize(pole_dir)

    return mid + pole_dir * offset


def compute_elbow_pole_position(
    shoulder_pos: np.ndarray,
    hand_pos: np.ndarray,
    char_forward: np.ndarray,
    char_up: np.ndarray,
    side: str,
    offset: float = 0.3,
) -> np.ndarray:
    """
    Compute pole position for elbow IK.

    NOTE: This is used primarily for visualization. The solve_arm_ik
    function does its own elbow circle sampling for better results.

    Args:
        shoulder_pos: Shoulder world position
        hand_pos: Hand target world position
        char_forward: Character forward direction
        char_up: Character up direction
        side: "L" or "R"
        offset: Distance of pole from arm line

    Returns:
        World position for elbow pole target
    """
    shoulder_pos = np.asarray(shoulder_pos, dtype=np.float32)
    hand_pos = np.asarray(hand_pos, dtype=np.float32)
    char_forward = np.asarray(char_forward, dtype=np.float32)
    char_up = np.asarray(char_up, dtype=np.float32)

    char_back = -char_forward
    char_down = -char_up
    char_right = cross(char_forward, char_up)
    char_left = -char_right

    # Midpoint of arm
    mid = (shoulder_pos + hand_pos) * 0.5

    # Default: elbow goes BACK + DOWN
    pole_dir = normalize(char_back + char_down * 0.5)

    # Add outward bias for correct side
    if side == "L":
        pole_dir = pole_dir + char_left * 0.3
    else:
        pole_dir = pole_dir + char_right * 0.3

    pole_dir = normalize(pole_dir)

    return mid + pole_dir * offset


# =============================================================================
# FOOT GROUND TARGET
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
    foot_rest_pos = np.asarray(foot_rest_pos, dtype=np.float32)
    target = foot_rest_pos.copy()
    target[2] = ground_z + foot_rest_z
    return target


# =============================================================================
# POSE APPLICATION
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
    chain_def = get_chain(chain)
    if not chain_def:
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
