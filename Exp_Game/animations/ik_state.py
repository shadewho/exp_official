# Exp_Game/animations/ik_state.py
"""
IK State Capture and Anatomical Validation.

This module provides:
1. IKFrameState - Complete state of an IK solve for logging
2. Anatomical validation - Is the pose humanly possible/correct?
3. Human-readable state description for debugging

All output goes to the diagnostics logger, not console.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from mathutils import Vector

from ..developer.dev_logger import log_game
from ..engine.animations.ik import LEG_IK, ARM_IK


@dataclass
class IKFrameState:
    """Complete state of an IK solve for a single chain."""

    # Identity
    chain: str                          # "arm_R", "arm_L", "leg_R", "leg_L"
    frame: int = 0                      # Frame number

    # Target
    target_pos: Tuple[float, float, float] = (0, 0, 0)
    target_object: str = ""             # Name of target object if any

    # Chain positions (world space)
    root_pos: Tuple[float, float, float] = (0, 0, 0)   # Shoulder/Hip
    mid_pos: Tuple[float, float, float] = (0, 0, 0)    # Elbow/Knee
    tip_pos: Tuple[float, float, float] = (0, 0, 0)    # Hand/Foot

    # Chain lengths
    upper_length: float = 0.0           # Root to mid
    lower_length: float = 0.0           # Mid to tip
    max_reach: float = 0.0              # upper + lower

    # Results
    error_distance: float = 0.0         # Distance from tip to target (meters)
    extension_pct: float = 0.0          # 0-100%, how stretched is the limb
    bend_angle: float = 0.0             # Joint angle in degrees (180 = straight)

    # Anatomical validation
    bend_direction: str = ""            # "FORWARD", "BACKWARD", "LEFT", "RIGHT", "INLINE"
    bend_correct: bool = True           # Is bend direction anatomically right?
    bend_expected: str = ""             # What direction SHOULD it bend?

    # Joint limits
    within_limits: bool = True
    limit_violations: List[str] = field(default_factory=list)

    # IK success
    reachable: bool = True              # Is target within max reach?
    reach_distance: float = 0.0         # Distance from root to target


def compute_ik_state(
    armature,
    chain: str,
    target_pos: Vector,
    target_object_name: str = "",
    frame: int = 0
) -> IKFrameState:
    """
    Compute the full IK state for a chain.

    Args:
        armature: Blender armature object
        chain: Chain name ("arm_R", "arm_L", "leg_R", "leg_L")
        target_pos: World-space target position
        target_object_name: Name of target object (for logging)
        frame: Current frame number

    Returns:
        IKFrameState with all computed values
    """
    # Get chain definition
    is_arm = chain.startswith("arm")
    chain_def = ARM_IK.get(chain) if is_arm else LEG_IK.get(chain)

    if not chain_def:
        state = IKFrameState(chain=chain, frame=frame)
        state.limit_violations = [f"Unknown chain: {chain}"]
        return state

    # Get bone positions
    pose_bones = armature.pose.bones
    root_bone = pose_bones.get(chain_def["root"])
    mid_bone = pose_bones.get(chain_def["mid"])
    tip_bone = pose_bones.get(chain_def["tip"])

    if not all([root_bone, mid_bone, tip_bone]):
        state = IKFrameState(chain=chain, frame=frame)
        state.limit_violations = ["Missing bones in chain"]
        return state

    # World positions
    arm_matrix = armature.matrix_world
    root_world = arm_matrix @ root_bone.head
    mid_world = arm_matrix @ mid_bone.head
    tip_world = arm_matrix @ tip_bone.head
    target_world = Vector(target_pos)

    # Chain lengths (actual, from current bone positions)
    upper_len = (mid_world - root_world).length
    lower_len = (tip_world - mid_world).length
    max_reach = upper_len + lower_len

    # Reach analysis
    reach_distance = (target_world - root_world).length
    reachable = reach_distance <= max_reach

    # Error (how far is tip from target)
    error_distance = (tip_world - target_world).length

    # Extension percentage (how stretched is the limb)
    current_reach = (tip_world - root_world).length
    extension_pct = (current_reach / max_reach * 100) if max_reach > 0 else 0

    # Bend angle (angle at mid joint)
    # 180 degrees = straight, 90 degrees = right angle, 0 = folded back on itself
    v1 = (root_world - mid_world).normalized()
    v2 = (tip_world - mid_world).normalized()
    dot_product = max(-1, min(1, v1.dot(v2)))  # Clamp for numerical stability
    bend_angle = np.degrees(np.arccos(dot_product))

    # Get character forward direction (for anatomical validation)
    char_forward = Vector((arm_matrix[0][1], arm_matrix[1][1], arm_matrix[2][1]))
    char_right = Vector((arm_matrix[0][0], arm_matrix[1][0], arm_matrix[2][0]))

    # Compute bend direction
    bend_direction, bend_correct, bend_expected = validate_bend_direction(
        chain, root_world, mid_world, tip_world, char_forward, char_right
    )

    # Build state
    state = IKFrameState(
        chain=chain,
        frame=frame,
        target_pos=tuple(target_world),
        target_object=target_object_name,
        root_pos=tuple(root_world),
        mid_pos=tuple(mid_world),
        tip_pos=tuple(tip_world),
        upper_length=upper_len,
        lower_length=lower_len,
        max_reach=max_reach,
        error_distance=error_distance,
        extension_pct=extension_pct,
        bend_angle=bend_angle,
        bend_direction=bend_direction,
        bend_correct=bend_correct,
        bend_expected=bend_expected,
        reachable=reachable,
        reach_distance=reach_distance,
    )

    return state


def validate_bend_direction(
    chain: str,
    root_pos: Vector,
    mid_pos: Vector,
    tip_pos: Vector,
    char_forward: Vector,
    char_right: Vector
) -> Tuple[str, bool, str]:
    """
    Determine if elbow/knee is bending the anatomically correct direction.

    For ARMS: elbow should bend BACKWARD (away from front of body)
    For LEGS: knee should bend FORWARD (toward front of body)

    Args:
        chain: Chain name to determine limb type
        root_pos: Shoulder/hip world position
        mid_pos: Elbow/knee world position
        tip_pos: Hand/foot world position
        char_forward: Character's forward direction (+Y typically)
        char_right: Character's right direction (+X typically)

    Returns:
        (direction: str, is_correct: bool, expected: str)
    """
    is_arm = chain.startswith("arm")
    is_right_side = chain.endswith("_R")

    # Vector from root to tip (the "reach line")
    reach_vec = tip_pos - root_pos
    reach_length = reach_vec.length

    if reach_length < 0.001:
        return ("INLINE", True, "BACKWARD" if is_arm else "FORWARD")

    reach_dir = reach_vec.normalized()

    # Vector from root to mid
    to_mid = mid_pos - root_pos

    # Project mid onto reach line
    projection_length = to_mid.dot(reach_dir)
    projection_point = root_pos + reach_dir * projection_length

    # Perpendicular offset (which side of the reach line is the joint?)
    offset = mid_pos - projection_point
    offset_length = offset.length

    # If joint is nearly on the reach line, limb is almost straight
    if offset_length < 0.02:  # 2cm tolerance
        return ("INLINE", True, "BACKWARD" if is_arm else "FORWARD")

    offset_dir = offset.normalized()

    # Determine direction relative to character
    front_back = offset_dir.dot(char_forward)
    left_right = offset_dir.dot(char_right)

    # Determine primary direction
    if abs(front_back) > abs(left_right):
        if front_back > 0.1:
            direction = "FORWARD"
        elif front_back < -0.1:
            direction = "BACKWARD"
        else:
            direction = "INLINE"
    else:
        if left_right > 0.1:
            direction = "RIGHT"
        elif left_right < -0.1:
            direction = "LEFT"
        else:
            direction = "INLINE"

    # Determine correctness based on limb type
    if is_arm:
        # Elbows should bend backward (and slightly outward is OK)
        expected = "BACKWARD"
        if direction == "BACKWARD":
            is_correct = True
        elif direction == "INLINE":
            is_correct = True  # Straight arm is fine
        elif direction in ("LEFT", "RIGHT"):
            # Slightly outward is acceptable for arms
            is_correct = True
        else:
            is_correct = False  # FORWARD is wrong
    else:
        # Knees should bend forward
        expected = "FORWARD"
        if direction == "FORWARD":
            is_correct = True
        elif direction == "INLINE":
            is_correct = True  # Straight leg is fine
        else:
            is_correct = False  # BACKWARD, LEFT, RIGHT are wrong for knees

    return (direction, is_correct, expected)


def log_ik_state(state: IKFrameState, include_detail: bool = True):
    """
    Log IK state to the diagnostics file in human-readable format.

    Args:
        state: The IK frame state to log
        include_detail: Whether to include verbose detail
    """
    chain = state.chain

    # Header
    log_game("IK", f"{'='*60}")
    log_game("IK", f"CHAIN: {chain} | FRAME: {state.frame}")
    log_game("IK", f"{'='*60}")

    # Target
    tp = state.target_pos
    target_label = state.target_object if state.target_object else "manual"
    if state.reachable:
        log_game("IK", f"TARGET: ({tp[0]:+.3f}, {tp[1]:+.3f}, {tp[2]:+.3f}) [{target_label}]")
    else:
        overshoot = state.reach_distance - state.max_reach
        log_game("IK", f"TARGET: ({tp[0]:+.3f}, {tp[1]:+.3f}, {tp[2]:+.3f}) [{target_label}] UNREACHABLE (+{overshoot:.2f}m)")

    # Chain positions
    rp, mp, ep = state.root_pos, state.mid_pos, state.tip_pos
    joint_name = "elbow" if chain.startswith("arm") else "knee"
    tip_name = "hand" if chain.startswith("arm") else "foot"
    root_name = "shoulder" if chain.startswith("arm") else "hip"

    log_game("IK", f"CHAIN: {root_name}({rp[0]:+.2f},{rp[1]:+.2f},{rp[2]:+.2f}) -> "
                   f"{joint_name}({mp[0]:+.2f},{mp[1]:+.2f},{mp[2]:+.2f}) -> "
                   f"{tip_name}({ep[0]:+.2f},{ep[1]:+.2f},{ep[2]:+.2f})")

    # Results
    error_cm = state.error_distance * 100
    error_status = "OK" if error_cm < 5 else "HIGH" if error_cm < 20 else "FAILED"
    log_game("IK", f"RESULT: error={error_cm:.1f}cm [{error_status}] | "
                   f"extension={state.extension_pct:.0f}% | "
                   f"{joint_name}_angle={state.bend_angle:.0f}deg")

    # Anatomical validation
    bend_ok = "OK" if state.bend_correct else "WRONG"
    bend_symbol = "+" if state.bend_correct else "X"

    log_game("IK", f"ANATOMY: {joint_name}_bends={state.bend_direction} [{bend_symbol}] "
                   f"(expected: {state.bend_expected})")

    if not state.bend_correct:
        log_game("IK", f"  ^^^ PROBLEM: {joint_name} should bend {state.bend_expected}, not {state.bend_direction}")

    # Joint limits
    if state.limit_violations:
        log_game("IK", f"LIMITS: VIOLATED - {', '.join(state.limit_violations)}")
    else:
        log_game("IK", f"LIMITS: OK")

    # Summary line
    all_ok = state.bend_correct and state.reachable and not state.limit_violations and error_cm < 10
    if all_ok:
        log_game("IK", f"STATUS: SUCCESS - pose is anatomically correct")
    else:
        problems = []
        if not state.bend_correct:
            problems.append(f"{joint_name} bends wrong way")
        if not state.reachable:
            problems.append("target unreachable")
        if state.limit_violations:
            problems.append("joint limits violated")
        if error_cm >= 10:
            problems.append(f"high error ({error_cm:.0f}cm)")
        log_game("IK", f"STATUS: ISSUES - {', '.join(problems)}")

    log_game("IK", f"{'='*60}")


def get_bone_world_position(armature, bone_name: str) -> Optional[Vector]:
    """Get world position of a bone's head."""
    pose_bone = armature.pose.bones.get(bone_name)
    if not pose_bone:
        return None
    return armature.matrix_world @ pose_bone.head


def get_chain_positions(armature, chain: str) -> Optional[Tuple[Vector, Vector, Vector]]:
    """Get world positions of root, mid, tip for a chain."""
    is_arm = chain.startswith("arm")
    chain_def = ARM_IK.get(chain) if is_arm else LEG_IK.get(chain)

    if not chain_def:
        return None

    root_pos = get_bone_world_position(armature, chain_def["root"])
    mid_pos = get_bone_world_position(armature, chain_def["mid"])
    tip_pos = get_bone_world_position(armature, chain_def["tip"])

    if not all([root_pos, mid_pos, tip_pos]):
        return None

    return (root_pos, mid_pos, tip_pos)
