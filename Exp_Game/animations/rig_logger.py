# Exp_Game/animations/rig_logger.py
"""
Rig State Logger - Captures bone transforms for debugging.

Logs bone positions, rotations, and constraint violations so Claude
can understand what's happening with the rig during IK testing.

Usage:
    from .rig_logger import log_rig_state, log_bone_chain, log_ik_attempt

    # Log full rig state
    log_rig_state(armature, "POSE_APPLIED")

    # Log specific chain
    log_bone_chain(armature, "arm_L", "IK_SOLVE")

    # Log IK attempt with target
    log_ik_attempt(armature, "arm_L", target_pos, result_pos)
"""

import bpy
import math
import numpy as np
from mathutils import Vector, Quaternion, Euler

from ..developer.dev_logger import log_game

# =============================================================================
# ANATOMICAL REFERENCE DATA
# =============================================================================

# Human joint rotation limits (degrees)
# These are approximate physiological limits for a healthy adult
JOINT_LIMITS = {
    # === SPINE ===
    "Spine": {
        "flexion": (-30, 45),      # Forward/back bend
        "lateral": (-30, 30),       # Side bend
        "twist": (-30, 30),         # Rotation
    },
    "Spine1": {
        "flexion": (-20, 30),
        "lateral": (-20, 20),
        "twist": (-20, 20),
    },
    "Spine2": {
        "flexion": (-20, 30),
        "lateral": (-20, 20),
        "twist": (-20, 20),
    },

    # === NECK/HEAD ===
    "Neck": {
        "flexion": (-40, 60),       # Look up/down
        "lateral": (-40, 40),       # Tilt
        "twist": (-50, 50),         # Turn
    },
    "Head": {
        "flexion": (-30, 40),
        "lateral": (-30, 30),
        "twist": (-40, 40),
    },

    # === SHOULDERS (Clavicle + Arm) ===
    "LeftShoulder": {
        "elevation": (-5, 30),      # Shrug up
        "protraction": (-15, 30),   # Forward/back
    },
    "RightShoulder": {
        "elevation": (-5, 30),
        "protraction": (-15, 30),
    },

    # === UPPER ARM ===
    "LeftArm": {
        "flexion": (-60, 180),      # Raise forward (180 = overhead)
        "abduction": (-50, 180),    # Raise sideways
        "rotation": (-90, 90),      # Internal/external rotation
    },
    "RightArm": {
        "flexion": (-60, 180),
        "abduction": (-50, 180),
        "rotation": (-90, 90),
    },

    # === FOREARM (Elbow) ===
    "LeftForeArm": {
        "flexion": (0, 145),        # Elbow bend (0=straight, 145=max)
        "pronation": (-90, 90),     # Wrist twist
    },
    "RightForeArm": {
        "flexion": (0, 145),
        "pronation": (-90, 90),
    },

    # === HAND/WRIST ===
    "LeftHand": {
        "flexion": (-70, 80),       # Bend wrist forward/back
        "deviation": (-20, 30),     # Side to side
    },
    "RightHand": {
        "flexion": (-70, 80),
        "deviation": (-20, 30),
    },

    # === THIGH (Hip) ===
    "LeftThigh": {
        "flexion": (-20, 120),      # Lift leg forward
        "abduction": (-30, 45),     # Spread legs
        "rotation": (-45, 45),      # Rotate leg in/out
    },
    "RightThigh": {
        "flexion": (-20, 120),
        "abduction": (-30, 45),
        "rotation": (-45, 45),
    },

    # === SHIN (Knee) ===
    "LeftShin": {
        "flexion": (0, 140),        # Knee bend (0=straight)
    },
    "RightShin": {
        "flexion": (0, 140),
    },

    # === FOOT (Ankle) ===
    "LeftFoot": {
        "dorsiflexion": (-50, 30),  # Point toe up/down
        "inversion": (-20, 30),     # Tilt sole in/out
    },
    "RightFoot": {
        "dorsiflexion": (-50, 30),
        "inversion": (-20, 30),
    },
}

# Bones to skip (too many fingers, etc.)
SKIP_BONES = {
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumb4",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumb4",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4",
    "LeftToe", "RightToe",
}

# IK Chain definitions for focused logging
IK_CHAINS = {
    "arm_L": ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"],
    "arm_R": ["RightShoulder", "RightArm", "RightForeArm", "RightHand"],
    "leg_L": ["LeftThigh", "LeftShin", "LeftFoot"],
    "leg_R": ["RightThigh", "RightShin", "RightFoot"],
    "spine": ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"],
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def quat_to_euler_degrees(quat):
    """Convert quaternion to euler angles in degrees (XYZ order)."""
    euler = quat.to_euler('XYZ')
    return (
        math.degrees(euler.x),
        math.degrees(euler.y),
        math.degrees(euler.z)
    )


def get_bone_world_rotation(armature, pose_bone):
    """Get bone's rotation in world space as euler degrees."""
    world_matrix = armature.matrix_world @ pose_bone.matrix
    euler = world_matrix.to_euler('XYZ')
    return (
        math.degrees(euler.x),
        math.degrees(euler.y),
        math.degrees(euler.z)
    )


def get_bone_local_rotation(pose_bone):
    """Get bone's local rotation as euler degrees."""
    if pose_bone.rotation_mode == 'QUATERNION':
        euler = pose_bone.rotation_quaternion.to_euler('XYZ')
    else:
        euler = pose_bone.rotation_euler
    return (
        math.degrees(euler.x),
        math.degrees(euler.y),
        math.degrees(euler.z)
    )


def check_joint_limits(bone_name, rotation_degrees):
    """
    Check if rotation violates anatomical limits.

    Returns list of violations: [(axis, value, limit_min, limit_max), ...]
    """
    if bone_name not in JOINT_LIMITS:
        return []

    limits = JOINT_LIMITS[bone_name]
    violations = []

    rx, ry, rz = rotation_degrees

    # Map euler angles to joint movements (simplified)
    # This is approximate - real mapping depends on bone orientation
    angle_map = {
        "flexion": rx,
        "lateral": ry,
        "twist": rz,
        "rotation": rz,
        "abduction": ry,
        "elevation": rx,
        "protraction": ry,
        "pronation": rz,
        "deviation": ry,
        "dorsiflexion": rx,
        "inversion": ry,
    }

    for movement, (min_val, max_val) in limits.items():
        if movement in angle_map:
            angle = angle_map[movement]
            if angle < min_val:
                violations.append((movement, angle, min_val, max_val))
            elif angle > max_val:
                violations.append((movement, angle, min_val, max_val))

    return violations


# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

def log_rig_snapshot(armature, label: str = "SNAPSHOT"):
    """
    Log complete rig state - all bone positions and rotations.

    Use sparingly - this produces a LOT of output.
    """
    if not armature or armature.type != 'ARMATURE':
        log_game("RIG", f"{label} ERROR no_armature")
        return

    log_game("RIG", f"{label} START armature={armature.name}")

    arm_matrix = armature.matrix_world
    arm_pos = arm_matrix.translation
    log_game("RIG", f"ARMATURE world_pos=({arm_pos.x:.3f},{arm_pos.y:.3f},{arm_pos.z:.3f})")

    for pose_bone in armature.pose.bones:
        if pose_bone.name in SKIP_BONES:
            continue

        # World position
        world_pos = arm_matrix @ pose_bone.head

        # Local rotation
        local_rot = get_bone_local_rotation(pose_bone)

        # Check limits
        violations = check_joint_limits(pose_bone.name, local_rot)

        # Format output
        pos_str = f"({world_pos.x:.3f},{world_pos.y:.3f},{world_pos.z:.3f})"
        rot_str = f"({local_rot[0]:.1f},{local_rot[1]:.1f},{local_rot[2]:.1f})"

        if violations:
            viol_str = " VIOLATION:" + ",".join([f"{v[0]}={v[1]:.1f}[{v[2]},{v[3]}]" for v in violations])
        else:
            viol_str = ""

        log_game("RIG", f"BONE {pose_bone.name} pos={pos_str} rot={rot_str}{viol_str}")

    log_game("RIG", f"{label} END")


def log_bone_chain(armature, chain_name: str, label: str = "CHAIN"):
    """
    Log a specific IK chain's state.

    More focused than full snapshot - good for debugging specific limbs.
    """
    if chain_name not in IK_CHAINS:
        log_game("RIG", f"{label} ERROR unknown_chain={chain_name}")
        return

    bone_names = IK_CHAINS[chain_name]
    arm_matrix = armature.matrix_world

    log_game("RIG", f"{label} chain={chain_name} bones={len(bone_names)}")

    total_violations = 0

    for bone_name in bone_names:
        pose_bone = armature.pose.bones.get(bone_name)
        if not pose_bone:
            log_game("RIG", f"  {bone_name} MISSING")
            continue

        # Positions
        head_world = arm_matrix @ pose_bone.head
        tail_world = arm_matrix @ pose_bone.tail
        length = (tail_world - head_world).length

        # Rotation
        local_rot = get_bone_local_rotation(pose_bone)

        # Check limits
        violations = check_joint_limits(bone_name, local_rot)
        total_violations += len(violations)

        # Direction vector (where bone points)
        direction = (tail_world - head_world).normalized()

        log_game("RIG", f"  {bone_name}:")
        log_game("RIG", f"    head=({head_world.x:.3f},{head_world.y:.3f},{head_world.z:.3f})")
        log_game("RIG", f"    tail=({tail_world.x:.3f},{tail_world.y:.3f},{tail_world.z:.3f})")
        log_game("RIG", f"    dir=({direction.x:.3f},{direction.y:.3f},{direction.z:.3f}) len={length:.3f}m")
        log_game("RIG", f"    rot_local=({local_rot[0]:.1f},{local_rot[1]:.1f},{local_rot[2]:.1f})deg")

        if violations:
            for movement, angle, min_val, max_val in violations:
                log_game("RIG", f"    VIOLATION {movement}={angle:.1f}deg limit=[{min_val},{max_val}]")

    log_game("RIG", f"{label} END violations={total_violations}")


def log_ik_attempt(armature, chain: str, target_pos, influence: float = 1.0):
    """
    Log an IK attempt with target information.

    Call this BEFORE applying IK to capture the goal.
    """
    if chain not in IK_CHAINS:
        return

    # Get root bone position
    bone_names = IK_CHAINS[chain]
    root_bone = armature.pose.bones.get(bone_names[0])
    tip_bone = armature.pose.bones.get(bone_names[-1])

    if not root_bone or not tip_bone:
        return

    arm_matrix = armature.matrix_world
    root_world = arm_matrix @ root_bone.head
    tip_world = arm_matrix @ tip_bone.tail
    target = Vector(target_pos)

    # Calculate distances
    current_dist = (tip_world - target).length
    root_to_target = (target - root_world).length

    # Chain length (sum of bone lengths)
    chain_length = 0
    for bone_name in bone_names:
        pb = armature.pose.bones.get(bone_name)
        if pb:
            chain_length += (arm_matrix @ pb.tail - arm_matrix @ pb.head).length

    reach_pct = (root_to_target / chain_length * 100) if chain_length > 0 else 0

    log_game("RIG", f"IK_GOAL chain={chain} influence={influence:.2f}")
    log_game("RIG", f"  target=({target.x:.3f},{target.y:.3f},{target.z:.3f})")
    log_game("RIG", f"  tip_current=({tip_world.x:.3f},{tip_world.y:.3f},{tip_world.z:.3f})")
    log_game("RIG", f"  distance_to_target={current_dist:.3f}m")
    log_game("RIG", f"  root_to_target={root_to_target:.3f}m chain_length={chain_length:.3f}m reach={reach_pct:.0f}%")

    if reach_pct > 100:
        log_game("RIG", f"  WARNING target_unreachable reach={reach_pct:.0f}%")


def log_ik_result(armature, chain: str, target_pos):
    """
    Log IK result AFTER applying.

    Shows how close we got and any violations introduced.
    """
    if chain not in IK_CHAINS:
        return

    bone_names = IK_CHAINS[chain]
    tip_bone = armature.pose.bones.get(bone_names[-1])

    if not tip_bone:
        return

    arm_matrix = armature.matrix_world
    tip_world = arm_matrix @ tip_bone.tail
    target = Vector(target_pos)

    error = (tip_world - target).length

    log_game("RIG", f"IK_RESULT chain={chain}")
    log_game("RIG", f"  tip_final=({tip_world.x:.3f},{tip_world.y:.3f},{tip_world.z:.3f})")
    log_game("RIG", f"  error={error:.4f}m ({error*100:.1f}cm)")

    # Log chain state with violations
    log_bone_chain(armature, chain, "IK_CHAIN_STATE")


def log_collision_check(armature, chain: str):
    """
    Check for self-collision (arm through torso, etc.)

    Basic implementation - checks if limb endpoints cross body midline
    when they shouldn't.
    """
    if chain not in IK_CHAINS:
        return

    arm_matrix = armature.matrix_world

    # Get spine position as body center reference
    spine_bone = armature.pose.bones.get("Spine1")
    if not spine_bone:
        return

    spine_pos = arm_matrix @ spine_bone.head

    # Get character's right direction (for left/right checks)
    char_right = Vector((arm_matrix[0][0], arm_matrix[1][0], arm_matrix[2][0]))

    bone_names = IK_CHAINS[chain]

    collision_warnings = []

    for bone_name in bone_names:
        pose_bone = armature.pose.bones.get(bone_name)
        if not pose_bone:
            continue

        bone_pos = arm_matrix @ pose_bone.head

        # Vector from spine to bone
        to_bone = bone_pos - spine_pos

        # How far left/right of spine center
        lateral_offset = to_bone.dot(char_right)

        # Check for arm crossing body
        if "Left" in bone_name and lateral_offset > 0.15:  # Left bone on right side
            collision_warnings.append(f"{bone_name} crossed_midline lateral={lateral_offset:.3f}m")
        elif "Right" in bone_name and lateral_offset < -0.15:  # Right bone on left side
            collision_warnings.append(f"{bone_name} crossed_midline lateral={lateral_offset:.3f}m")

        # Check if bone is behind spine (arm through back)
        char_forward = Vector((arm_matrix[0][1], arm_matrix[1][1], arm_matrix[2][1]))
        forward_offset = to_bone.dot(char_forward)

        if "Arm" in bone_name or "Hand" in bone_name:
            if forward_offset < -0.2:  # Arm significantly behind spine
                collision_warnings.append(f"{bone_name} behind_body forward={forward_offset:.3f}m")

    if collision_warnings:
        log_game("RIG", f"COLLISION chain={chain} warnings={len(collision_warnings)}")
        for warn in collision_warnings:
            log_game("RIG", f"  {warn}")
    else:
        log_game("RIG", f"COLLISION chain={chain} clear")


# =============================================================================
# HIGH-LEVEL TEST FUNCTIONS
# =============================================================================

def start_rig_test_session(armature, test_type: str = "GENERAL"):
    """Call when starting a test - logs initial rig state."""
    log_game("RIG", f"SESSION_START type={test_type}")
    log_game("RIG", f"armature={armature.name if armature else 'None'}")

    if armature:
        # Log initial state of main chains
        for chain_name in ["arm_L", "arm_R", "leg_L", "leg_R"]:
            log_bone_chain(armature, chain_name, "INITIAL")


def end_rig_test_session(armature, test_type: str = "GENERAL"):
    """Call when ending a test - logs final state and summary."""
    log_game("RIG", f"SESSION_END type={test_type}")

    if armature:
        # Log final state
        for chain_name in ["arm_L", "arm_R", "leg_L", "leg_R"]:
            log_bone_chain(armature, chain_name, "FINAL")
            log_collision_check(armature, chain_name)

    log_game("RIG", "SESSION_COMPLETE")
