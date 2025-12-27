# Exp_Game/animations/pose_validator.py
"""
Pose Validator - Automatic pose quality evaluation.

Uses the rig's joint limits and anatomical constraints to determine
if a pose is valid/invalid. This allows the IK system to self-evaluate.

Data source: rig.md joint rotation limits
"""

import numpy as np
from mathutils import Vector, Euler
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from ..developer.dev_logger import log_game


# Joint rotation limits from rig.md (degrees)
# Format: {bone_name: {"X": [min, max], "Y": [min, max], "Z": [min, max]}}
JOINT_LIMITS = {
    # Spine
    "Spine": {"X": [-45, 45], "Y": [-45, 45], "Z": [-45, 45]},
    "Spine1": {"X": [-30, 30], "Y": [-30, 30], "Z": [-30, 30]},
    "Spine2": {"X": [-30, 30], "Y": [-30, 30], "Z": [-30, 30]},

    # Neck & Head
    "NeckLower": {"X": [-45, 45], "Y": [-45, 45], "Z": [-30, 30]},
    "NeckUpper": {"X": [-30, 30], "Y": [-30, 30], "Z": [-30, 30]},
    "Head": {"X": [-45, 45], "Y": [-60, 60], "Z": [-30, 30]},

    # Left Arm
    "LeftShoulder": {"X": [-15, 15], "Y": [-15, 15], "Z": [-15, 15]},
    "LeftArm": {"X": [-90, 90], "Y": [-120, 120], "Z": [-80, 140]},
    "LeftForeArm": {"X": [0, 0], "Y": [-170, 60], "Z": [0, 90]},
    "LeftHand": {"X": [-90, 90], "Y": [0, 0], "Z": [-60, 60]},

    # Right Arm
    "RightShoulder": {"X": [-15, 15], "Y": [-15, 15], "Z": [-15, 15]},
    "RightArm": {"X": [-90, 90], "Y": [-120, 120], "Z": [-140, 80]},
    "RightForeArm": {"X": [0, 0], "Y": [-60, 170], "Z": [-90, 0]},
    "RightHand": {"X": [-90, 90], "Y": [0, 0], "Z": [-60, 60]},

    # Left Leg
    "LeftThigh": {"X": [-90, 120], "Y": [-40, 40], "Z": [-20, 80]},
    "LeftShin": {"X": [-150, 10], "Y": [-10, 10], "Z": [0, 0]},
    "LeftFoot": {"X": [-80, 45], "Y": [-30, 30], "Z": [-40, 40]},
    "LeftToeBase": {"X": [-40, 40], "Y": [0, 0], "Z": [0, 0]},

    # Right Leg
    "RightThigh": {"X": [-90, 120], "Y": [-40, 40], "Z": [-80, 20]},
    "RightShin": {"X": [-150, 10], "Y": [-10, 10], "Z": [0, 0]},
    "RightFoot": {"X": [-80, 45], "Y": [-30, 30], "Z": [-40, 40]},
    "RightToeBase": {"X": [-40, 40], "Y": [0, 0], "Z": [0, 0]},
}

# IK bend directions (pole vectors) from rig.md
# Describes which direction the mid-joint (elbow/knee) should point
IK_BEND_RULES = {
    # Legs: knee bends FORWARD (+Y in character space)
    "leg_L": {"mid_bone": "LeftShin", "bend_axis": "Y", "bend_sign": 1, "description": "knee forward"},
    "leg_R": {"mid_bone": "RightShin", "bend_axis": "Y", "bend_sign": 1, "description": "knee forward"},

    # Arms: elbow bends BACKWARD (-Y in character space)
    "arm_L": {"mid_bone": "LeftForeArm", "bend_axis": "Y", "bend_sign": -1, "description": "elbow backward"},
    "arm_R": {"mid_bone": "RightForeArm", "bend_axis": "Y", "bend_sign": -1, "description": "elbow backward"},
}


@dataclass
class BoneViolation:
    """A single joint limit violation."""
    bone_name: str
    axis: str  # "X", "Y", or "Z"
    current_angle: float  # degrees
    min_allowed: float
    max_allowed: float
    overshoot: float  # how far outside limits (degrees)


@dataclass
class PoseValidation:
    """Result of validating a pose."""
    valid: bool
    score: float  # 0.0 = terrible, 1.0 = perfect
    violations: List[BoneViolation] = field(default_factory=list)
    bend_correct: bool = True
    bend_issues: List[str] = field(default_factory=list)
    ik_error_cm: float = 0.0
    summary: str = ""


def validate_pose(
    armature,
    chain: str = None,
    target_pos: Vector = None,
    debug: bool = True
) -> PoseValidation:
    """
    Validate the current pose of an armature.

    Checks:
    1. Joint rotation limits - is each bone within allowed range?
    2. IK bend direction - is elbow/knee bending the right way?
    3. IK error - how far is tip from target?

    Args:
        armature: Blender armature object
        chain: Optional IK chain to focus on (arm_L, arm_R, leg_L, leg_R)
        target_pos: Optional IK target position for error calculation
        debug: Whether to log details

    Returns:
        PoseValidation with results
    """
    result = PoseValidation(valid=True, score=1.0)

    if debug:
        log_game("POSE_VALID", "=" * 50)
        log_game("POSE_VALID", f"Validating pose for: {armature.name}")

    # Determine which bones to check
    if chain:
        bones_to_check = get_chain_bones(chain)
    else:
        bones_to_check = list(JOINT_LIMITS.keys())

    # Check joint limits
    total_overshoot = 0.0
    for bone_name in bones_to_check:
        if bone_name not in JOINT_LIMITS:
            continue

        pose_bone = armature.pose.bones.get(bone_name)
        if not pose_bone:
            continue

        limits = JOINT_LIMITS[bone_name]
        violations = check_bone_limits(pose_bone, limits)

        for v in violations:
            result.violations.append(v)
            total_overshoot += v.overshoot
            if debug:
                log_game("POSE_VALID", f"  VIOLATION: {v.bone_name} {v.axis}={v.current_angle:.1f}° "
                                       f"(allowed: [{v.min_allowed}, {v.max_allowed}]) "
                                       f"overshoot: {v.overshoot:.1f}°")

    # Check IK bend direction
    if chain and chain in IK_BEND_RULES:
        bend_ok, bend_msg = check_bend_direction(armature, chain)
        result.bend_correct = bend_ok
        if not bend_ok:
            result.bend_issues.append(bend_msg)
            if debug:
                log_game("POSE_VALID", f"  BEND ERROR: {bend_msg}")

    # Calculate IK error if target provided
    if chain and target_pos:
        from ..engine.animations.ik import ARM_IK, LEG_IK
        chain_def = ARM_IK.get(chain) or LEG_IK.get(chain)
        if chain_def:
            tip_bone = armature.pose.bones.get(chain_def["tip"])
            if tip_bone:
                tip_world = armature.matrix_world @ tip_bone.head
                error = (tip_world - target_pos).length
                result.ik_error_cm = error * 100
                if debug:
                    log_game("POSE_VALID", f"  IK error: {result.ik_error_cm:.1f}cm")

    # Calculate overall score
    # Penalties: violations, wrong bend, high IK error
    penalty = 0.0

    # Joint violations: -0.1 per degree of overshoot
    penalty += total_overshoot * 0.01

    # Wrong bend direction: -0.3
    if not result.bend_correct:
        penalty += 0.3

    # IK error: -0.01 per cm over 5cm
    if result.ik_error_cm > 5:
        penalty += (result.ik_error_cm - 5) * 0.01

    result.score = max(0.0, 1.0 - penalty)
    result.valid = len(result.violations) == 0 and result.bend_correct and result.ik_error_cm < 10

    # Generate summary
    issues = []
    if result.violations:
        issues.append(f"{len(result.violations)} joint violations")
    if not result.bend_correct:
        issues.append("wrong bend direction")
    if result.ik_error_cm > 10:
        issues.append(f"high IK error ({result.ik_error_cm:.0f}cm)")

    if issues:
        result.summary = f"INVALID: {', '.join(issues)}"
    else:
        result.summary = f"VALID (score: {result.score:.2f})"

    if debug:
        log_game("POSE_VALID", f"  RESULT: {result.summary}")
        log_game("POSE_VALID", "=" * 50)

    return result


def check_bone_limits(pose_bone, limits: Dict) -> List[BoneViolation]:
    """Check if a bone's rotation is within limits."""
    violations = []

    # Get rotation in Euler angles (degrees)
    pose_bone.rotation_mode = 'XYZ'
    euler = pose_bone.rotation_euler
    angles = {
        "X": np.degrees(euler.x),
        "Y": np.degrees(euler.y),
        "Z": np.degrees(euler.z),
    }

    for axis in ["X", "Y", "Z"]:
        if axis not in limits:
            continue

        min_val, max_val = limits[axis]

        # Skip if no limit (both zero means any rotation allowed on that axis)
        if min_val == 0 and max_val == 0:
            continue

        current = angles[axis]

        # Normalize angle to -180 to 180
        while current > 180:
            current -= 360
        while current < -180:
            current += 360

        if current < min_val:
            overshoot = min_val - current
            violations.append(BoneViolation(
                bone_name=pose_bone.name,
                axis=axis,
                current_angle=current,
                min_allowed=min_val,
                max_allowed=max_val,
                overshoot=overshoot
            ))
        elif current > max_val:
            overshoot = current - max_val
            violations.append(BoneViolation(
                bone_name=pose_bone.name,
                axis=axis,
                current_angle=current,
                min_allowed=min_val,
                max_allowed=max_val,
                overshoot=overshoot
            ))

    return violations


def check_bend_direction(armature, chain: str) -> Tuple[bool, str]:
    """
    Check if the IK chain's mid-joint is bending the correct direction.

    Uses world-space geometry to determine bend direction.
    """
    from ..engine.animations.ik import ARM_IK, LEG_IK

    rule = IK_BEND_RULES.get(chain)
    if not rule:
        return True, ""

    chain_def = ARM_IK.get(chain) or LEG_IK.get(chain)
    if not chain_def:
        return True, ""

    pose_bones = armature.pose.bones
    root_bone = pose_bones.get(chain_def["root"])
    mid_bone = pose_bones.get(chain_def["mid"])
    tip_bone = pose_bones.get(chain_def["tip"])

    if not all([root_bone, mid_bone, tip_bone]):
        return True, ""

    arm_matrix = armature.matrix_world

    # Get world positions
    root_pos = arm_matrix @ root_bone.head
    mid_pos = arm_matrix @ mid_bone.head
    tip_pos = arm_matrix @ tip_bone.head

    # Vector from root to tip (the "reach line")
    reach_vec = tip_pos - root_pos
    reach_len = reach_vec.length

    if reach_len < 0.01:
        return True, ""  # Can't determine if arm is collapsed

    reach_dir = reach_vec.normalized()

    # Vector from root to mid
    to_mid = mid_pos - root_pos

    # Project mid onto reach line
    proj_len = to_mid.dot(reach_dir)
    proj_point = root_pos + reach_dir * proj_len

    # Offset of mid from the reach line
    offset = mid_pos - proj_point
    offset_len = offset.length

    if offset_len < 0.02:
        return True, ""  # Limb is nearly straight, can't determine bend

    offset_dir = offset.normalized()

    # Get character's forward direction
    char_forward = Vector((arm_matrix[0][1], arm_matrix[1][1], arm_matrix[2][1]))

    # Check if offset is in the correct direction
    # For legs: forward means positive dot with char_forward
    # For arms: backward means negative dot with char_forward
    dot_forward = offset_dir.dot(char_forward)

    expected_sign = rule["bend_sign"]
    description = rule["description"]

    # Check if bend matches expected
    if expected_sign > 0:
        # Should bend forward (positive dot)
        if dot_forward < -0.1:
            return False, f"{description} expected, but bending backward"
    else:
        # Should bend backward (negative dot)
        if dot_forward > 0.1:
            return False, f"{description} expected, but bending forward"

    return True, ""


def get_chain_bones(chain: str) -> List[str]:
    """Get all bones involved in an IK chain."""
    from ..engine.animations.ik import ARM_IK, LEG_IK

    chain_def = ARM_IK.get(chain) or LEG_IK.get(chain)
    if not chain_def:
        return []

    return [chain_def["root"], chain_def["mid"], chain_def["tip"]]


def log_validation_result(result: PoseValidation):
    """Log validation result in human-readable format."""
    log_game("POSE_VALID", "=" * 50)
    log_game("POSE_VALID", f"POSE VALIDATION RESULT")
    log_game("POSE_VALID", f"  Valid: {result.valid}")
    log_game("POSE_VALID", f"  Score: {result.score:.2f}")

    if result.violations:
        log_game("POSE_VALID", f"  Joint Violations ({len(result.violations)}):")
        for v in result.violations:
            log_game("POSE_VALID", f"    {v.bone_name} {v.axis}: {v.current_angle:.1f}° "
                                   f"(limit: [{v.min_allowed}, {v.max_allowed}])")

    if result.bend_issues:
        log_game("POSE_VALID", f"  Bend Issues:")
        for issue in result.bend_issues:
            log_game("POSE_VALID", f"    {issue}")

    if result.ik_error_cm > 0:
        log_game("POSE_VALID", f"  IK Error: {result.ik_error_cm:.1f}cm")

    log_game("POSE_VALID", f"  Summary: {result.summary}")
    log_game("POSE_VALID", "=" * 50)
