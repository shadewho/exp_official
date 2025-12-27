# Exp_Game/animations/ik_solver.py
"""
Two-Bone IK Solver with Anatomical Constraints.

This module provides:
1. solve_two_bone_ik() - Calculate joint positions to reach target
2. apply_ik_to_chain() - Apply solution to Blender armature
3. Full logging at each step for debugging

The solver respects anatomical constraints:
- Elbows bend BACKWARD (away from body front)
- Knees bend FORWARD (toward body front)

Bone axis data from rig.md:
- All mid-joints (elbows/knees) bend around their LOCAL X AXIS
- Root joints (upper arm/thigh) have more freedom
"""

import numpy as np
import math
from mathutils import Vector, Matrix, Quaternion, Euler
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from ..developer.dev_logger import log_game
from ..engine.animations.ik import LEG_IK, ARM_IK


# =============================================================================
# BONE AXIS DATA (from rig.md bone orientations)
# =============================================================================
# Format: bone_name -> (X_axis, Y_axis, Z_axis) in world space at rest
# Y-axis = bone direction (head to tail)
# X-axis = bend axis for elbows/knees

BONE_AXES = {
    # Left Arm
    "LeftArm": {
        "X": Vector((+0.01, -1.00, -0.02)),   # BACK
        "Y": Vector((-1.00, -0.01, -0.05)),   # LEFT (bone direction)
        "Z": Vector((+0.05, +0.02, -1.00)),   # DOWN
    },
    "LeftForeArm": {
        "X": Vector((-0.00, -1.00, -0.02)),   # BACK
        "Y": Vector((-1.00, +0.00, -0.00)),   # LEFT (bone direction)
        "Z": Vector((+0.00, +0.02, -1.00)),   # DOWN
    },

    # Right Arm
    "RightArm": {
        "X": Vector((+0.02, +1.00, +0.02)),   # FORWARD
        "Y": Vector((+1.00, -0.01, -0.05)),   # RIGHT (bone direction)
        "Z": Vector((-0.05, +0.02, -1.00)),   # DOWN
    },
    "RightForeArm": {
        "X": Vector((-0.00, +1.00, +0.02)),   # FORWARD
        "Y": Vector((+1.00, +0.00, -0.00)),   # RIGHT (bone direction)
        "Z": Vector((-0.00, +0.02, -1.00)),   # DOWN
    },

    # Left Leg
    "LeftThigh": {
        "X": Vector((+1.00, -0.00, -0.06)),   # RIGHT
        "Y": Vector((-0.06, -0.00, -1.00)),   # DOWN (bone direction)
        "Z": Vector((+0.00, +1.00, -0.00)),   # FORWARD
    },
    "LeftShin": {
        "X": Vector((+1.00, +0.01, -0.01)),   # RIGHT
        "Y": Vector((-0.01, -0.15, -0.99)),   # DOWN (bone direction)
        "Z": Vector((-0.01, +0.99, -0.15)),   # FORWARD
    },

    # Right Leg
    "RightThigh": {
        "X": Vector((+1.00, +0.00, +0.06)),   # RIGHT
        "Y": Vector((+0.06, -0.00, -1.00)),   # DOWN (bone direction)
        "Z": Vector((-0.00, +1.00, -0.00)),   # FORWARD
    },
    "RightShin": {
        "X": Vector((+1.00, -0.01, +0.01)),   # RIGHT
        "Y": Vector((+0.01, -0.13, -0.99)),   # DOWN (bone direction)
        "Z": Vector((+0.01, +0.99, -0.13)),   # FORWARD
    },
}


@dataclass
class IKSolution:
    """Result of an IK solve."""
    success: bool
    root_pos: Vector           # World position of root (shoulder/hip)
    mid_pos: Vector            # World position of mid (elbow/knee)
    tip_pos: Vector            # World position of tip (hand/foot)
    target_pos: Vector         # Target we're reaching for
    error: float               # Distance from tip to target (meters)
    clamped: bool              # Was target clamped to max reach?
    bend_direction: Vector     # Direction the joint bends toward


def solve_two_bone_ik(
    root_pos: Vector,
    upper_length: float,
    lower_length: float,
    target_pos: Vector,
    bend_hint: Vector,
    debug: bool = True
) -> IKSolution:
    """
    Solve two-bone IK using law of cosines.

    Args:
        root_pos: World position of root joint (shoulder/hip)
        upper_length: Length of upper bone (shoulder→elbow or hip→knee)
        lower_length: Length of lower bone (elbow→hand or knee→foot)
        target_pos: World position we want tip to reach
        bend_hint: Direction the mid joint should bend toward (for elbow/knee direction)
        debug: Whether to log debug info

    Returns:
        IKSolution with calculated positions
    """
    root = Vector(root_pos)
    target = Vector(target_pos)

    # Vector from root to target
    to_target = target - root
    target_dist = to_target.length

    max_reach = upper_length + lower_length
    min_reach = abs(upper_length - lower_length)

    # Handle unreachable targets
    clamped = False
    if target_dist > max_reach - 0.001:
        # Clamp to max reach (slightly inside to avoid numerical issues)
        target = root + to_target.normalized() * (max_reach - 0.001)
        target_dist = max_reach - 0.001
        clamped = True
        if debug:
            log_game("IK_SOLVE", f"Target clamped: was {to_target.length:.3f}m, max reach {max_reach:.3f}m")
    elif target_dist < min_reach + 0.001:
        # Target too close - extend to minimum
        target = root + to_target.normalized() * (min_reach + 0.001)
        target_dist = min_reach + 0.001
        clamped = True
        if debug:
            log_game("IK_SOLVE", f"Target extended: was {to_target.length:.3f}m, min reach {min_reach:.3f}m")

    # Recalculate after potential clamping
    to_target = target - root
    target_dist = to_target.length

    # Law of cosines to find angle at root
    # c² = a² + b² - 2ab*cos(C)
    # cos(C) = (a² + b² - c²) / (2ab)
    # where: a = upper_length, b = target_dist, c = lower_length
    # We want angle at root between upper bone and line to target

    # Using law of cosines: cos(angle_at_root) = (upper² + dist² - lower²) / (2 * upper * dist)
    cos_angle_root = (upper_length**2 + target_dist**2 - lower_length**2) / (2 * upper_length * target_dist)
    cos_angle_root = np.clip(cos_angle_root, -1.0, 1.0)
    angle_at_root = np.arccos(cos_angle_root)

    if debug:
        log_game("IK_SOLVE", f"Geometry: upper={upper_length:.3f}m, lower={lower_length:.3f}m, target_dist={target_dist:.3f}m")
        log_game("IK_SOLVE", f"Angle at root: {np.degrees(angle_at_root):.1f}deg")

    # Direction from root to target (normalized)
    reach_dir = to_target.normalized()

    # We need a plane perpendicular to reach_dir to place the elbow/knee
    # The bend_hint tells us which side of this plane to put the joint

    # Create a coordinate frame:
    # X = reach direction (toward target)
    # Y = bend direction (perpendicular to reach, toward where joint should be)
    # Z = cross product

    # Project bend_hint onto plane perpendicular to reach_dir
    bend_in_plane = bend_hint - reach_dir * bend_hint.dot(reach_dir)
    if bend_in_plane.length < 0.001:
        # bend_hint is parallel to reach - use world up as fallback
        bend_in_plane = Vector((0, 0, 1)) - reach_dir * Vector((0, 0, 1)).dot(reach_dir)
        if bend_in_plane.length < 0.001:
            bend_in_plane = Vector((0, 1, 0)) - reach_dir * Vector((0, 1, 0)).dot(reach_dir)

    bend_dir = bend_in_plane.normalized()

    # Mid joint position:
    # Start at root, go angle_at_root away from reach_dir, in bend_dir direction
    # Distance = upper_length

    # The mid position is: root + upper_length * (cos(angle)*reach_dir + sin(angle)*bend_dir)
    # Convert numpy floats to Python floats for Vector math
    cos_a = float(np.cos(angle_at_root))
    sin_a = float(np.sin(angle_at_root))
    mid_pos = root + upper_length * (cos_a * reach_dir + sin_a * bend_dir)

    # Tip position is at target (or as close as we can get)
    tip_pos = target

    # Calculate actual error
    error = (tip_pos - Vector(target_pos)).length

    if debug:
        log_game("IK_SOLVE", f"Solution: root={tuple(root)}, mid={tuple(mid_pos)}, tip={tuple(tip_pos)}")
        log_game("IK_SOLVE", f"Error: {error*100:.1f}cm, clamped={clamped}")

    return IKSolution(
        success=True,
        root_pos=root,
        mid_pos=mid_pos,
        tip_pos=tip_pos,
        target_pos=Vector(target_pos),
        error=error,
        clamped=clamped,
        bend_direction=bend_dir
    )


def get_bend_hint(chain: str, armature) -> Vector:
    """
    Get the anatomically correct bend direction for a chain.

    Args:
        chain: Chain name (arm_R, arm_L, leg_R, leg_L)
        armature: Blender armature object

    Returns:
        World-space vector indicating bend direction
    """
    # Get character's forward direction from armature matrix
    arm_matrix = armature.matrix_world

    # Character axes (assuming Y-forward, Z-up convention)
    char_forward = Vector((arm_matrix[0][1], arm_matrix[1][1], arm_matrix[2][1]))
    char_backward = -char_forward
    char_right = Vector((arm_matrix[0][0], arm_matrix[1][0], arm_matrix[2][0]))
    char_left = -char_right

    is_arm = chain.startswith("arm")
    is_right = chain.endswith("_R")

    if is_arm:
        # Elbows bend BACKWARD (and slightly outward)
        outward = char_right if is_right else char_left
        return (char_backward * 0.8 + outward * 0.2).normalized()
    else:
        # Knees bend FORWARD
        return char_forward


def apply_ik_to_chain(
    armature,
    chain: str,
    solution: IKSolution,
    debug: bool = True
) -> bool:
    """
    Apply IK solution to armature bones.

    Simple approach:
    1. Root bone (upper arm/thigh): rotate to point at solved mid position
    2. Mid bone (forearm/shin): rotate to point at target

    Uses direct world-to-local rotation calculation.

    Args:
        armature: Blender armature object
        chain: Chain name (arm_R, arm_L, leg_R, leg_L)
        solution: IK solution with target positions
        debug: Whether to log debug info

    Returns:
        True if successful
    """
    import bpy

    # Get chain definition
    is_arm = chain.startswith("arm")
    chain_def = ARM_IK.get(chain) if is_arm else LEG_IK.get(chain)

    if not chain_def:
        log_game("IK_APPLY", f"Unknown chain: {chain}")
        return False

    pose_bones = armature.pose.bones
    root_bone = pose_bones.get(chain_def["root"])
    mid_bone = pose_bones.get(chain_def["mid"])
    tip_bone = pose_bones.get(chain_def["tip"])

    if not all([root_bone, mid_bone, tip_bone]):
        log_game("IK_APPLY", f"Missing bones in chain {chain}")
        return False

    arm_matrix = armature.matrix_world
    target_pos = solution.target_pos

    if debug:
        log_game("IK_APPLY", f"=== Applying IK to chain: {chain} ===")
        log_game("IK_APPLY", f"  Bones: {chain_def['root']} -> {chain_def['mid']} -> {chain_def['tip']}")
        tp = target_pos
        log_game("IK_APPLY", f"  Target: ({tp.x:.3f}, {tp.y:.3f}, {tp.z:.3f})")
        mp = solution.mid_pos
        log_game("IK_APPLY", f"  Solved mid: ({mp.x:.3f}, {mp.y:.3f}, {mp.z:.3f})")

    # Reset rotations first so we work from rest pose
    root_bone.rotation_mode = 'QUATERNION'
    mid_bone.rotation_mode = 'QUATERNION'
    root_bone.rotation_quaternion = Quaternion()
    mid_bone.rotation_quaternion = Quaternion()
    bpy.context.view_layer.update()

    # Get current positions (should be rest pose now)
    root_head = arm_matrix @ root_bone.head
    mid_head = arm_matrix @ mid_bone.head
    tip_head = arm_matrix @ tip_bone.head

    if debug:
        log_game("IK_APPLY", f"  Rest positions:")
        log_game("IK_APPLY", f"    root: ({root_head.x:.3f}, {root_head.y:.3f}, {root_head.z:.3f})")
        log_game("IK_APPLY", f"    mid:  ({mid_head.x:.3f}, {mid_head.y:.3f}, {mid_head.z:.3f})")
        log_game("IK_APPLY", f"    tip:  ({tip_head.x:.3f}, {tip_head.y:.3f}, {tip_head.z:.3f})")

    # Get pole vectors (where elbow/knee should point)
    # Knees point FORWARD (+Y world), elbows point BACKWARD (-Y world)
    is_leg = chain.startswith("leg")
    if is_leg:
        pole = Vector((0, 1, 0))  # Knee forward
    else:
        pole = Vector((0, -1, 0))  # Elbow backward

    if debug:
        log_game("IK_APPLY", f"  Pole vector: ({pole.x}, {pole.y}, {pole.z}) [{'knee forward' if is_leg else 'elbow back'}]")

    # Step 1: Rotate root bone to point at the solved mid position
    rotate_bone_toward(armature, root_bone, solution.mid_pos, debug, "Root", pole_world=pole)
    bpy.context.view_layer.update()

    # Step 2: Rotate mid bone to point at target
    rotate_bone_toward(armature, mid_bone, target_pos, debug, "Mid", pole_world=pole)
    bpy.context.view_layer.update()

    # Final check
    final_tip = arm_matrix @ tip_bone.head
    final_error = (final_tip - target_pos).length

    if debug:
        log_game("IK_APPLY", f"  RESULT:")
        log_game("IK_APPLY", f"    tip: ({final_tip.x:.3f}, {final_tip.y:.3f}, {final_tip.z:.3f})")
        log_game("IK_APPLY", f"    error: {final_error*100:.1f}cm")

    return final_error < 0.15  # Success if within 15cm


def rotate_bone_toward(armature, pose_bone, target_world: Vector, debug: bool, label: str,
                       pole_world: Vector = None):
    """
    Rotate a bone so it points toward a world-space target.

    Uses track-to rotation with pole vector to control roll (elbow/knee direction).
    """
    import bpy
    arm_matrix = armature.matrix_world

    # =========================================================================
    # BEFORE STATE
    # =========================================================================
    head_before = arm_matrix @ pose_bone.head
    tail_before = arm_matrix @ pose_bone.tail
    dir_before = (tail_before - head_before).normalized()
    bone_length = (tail_before - head_before).length

    # Direction we want bone to point (world space)
    target_dir = (target_world - head_before)
    if target_dir.length < 0.001:
        if debug:
            log_game("IK_APPLY", f"  {label}: target too close, skipping")
        return
    target_dir = target_dir.normalized()

    if debug:
        log_game("IK_APPLY", f"  {label} BONE: {pose_bone.name}")
        log_game("IK_APPLY", f"    BEFORE: head=({head_before.x:.3f},{head_before.y:.3f},{head_before.z:.3f})")
        log_game("IK_APPLY", f"    BEFORE: tail=({tail_before.x:.3f},{tail_before.y:.3f},{tail_before.z:.3f})")
        log_game("IK_APPLY", f"    BEFORE: dir=({dir_before.x:.3f},{dir_before.y:.3f},{dir_before.z:.3f})")
        log_game("IK_APPLY", f"    WANT:   dir=({target_dir.x:.3f},{target_dir.y:.3f},{target_dir.z:.3f})")

    # Default pole to world Z if not specified
    if pole_world is None:
        pole_world = Vector((0, 0, 1))

    # =========================================================================
    # BUILD TARGET ROTATION MATRIX
    # =========================================================================
    # Y-axis = target direction (where bone points)
    # Z-axis = as close to pole as possible (perpendicular to Y)
    # X-axis = cross product

    y_axis = target_dir

    # Make Z perpendicular to Y, as close to pole as possible
    z_axis = pole_world - y_axis * pole_world.dot(y_axis)
    if z_axis.length < 0.001:
        z_axis = Vector((1, 0, 0)) - y_axis * Vector((1, 0, 0)).dot(y_axis)
    z_axis = z_axis.normalized()

    x_axis = y_axis.cross(z_axis).normalized()
    z_axis = x_axis.cross(y_axis).normalized()

    # Build world-space target matrix
    target_matrix_world = Matrix((
        (x_axis.x, y_axis.x, z_axis.x),
        (x_axis.y, y_axis.y, z_axis.y),
        (x_axis.z, y_axis.z, z_axis.z)
    )).to_4x4()

    # =========================================================================
    # CONVERT TO LOCAL SPACE - Simple approach
    # =========================================================================
    # The bone's current world matrix (at identity pose after reset)
    bone_world = arm_matrix @ pose_bone.matrix

    # The bone's current Y-axis in world space
    bone_y_world = Vector((bone_world[0][1], bone_world[1][1], bone_world[2][1])).normalized()

    # Rotation needed in WORLD space to go from current Y to target Y
    world_rotation = bone_y_world.rotation_difference(target_dir)

    # The bone's current world orientation as quaternion
    bone_world_quat = bone_world.to_quaternion()

    # Apply world rotation to get new world orientation
    new_world_quat = world_rotation @ bone_world_quat

    # Now convert new world orientation to local pose rotation
    # local_rotation = rest_world_inv @ new_world
    # where rest_world = armature @ bone.matrix_local
    bone_rest_world = arm_matrix @ pose_bone.bone.matrix_local
    bone_rest_world_quat = bone_rest_world.to_quaternion()

    # But we also need to account for parent's current rotation
    if pose_bone.parent:
        parent_world = arm_matrix @ pose_bone.parent.matrix
        parent_world_quat = parent_world.to_quaternion()
        # New local = parent_inv @ new_world @ rest_in_parent_inv...
        # Actually simpler: the pose rotation in local space
        # final_world = parent_world @ bone_rest_in_parent @ pose_rotation
        # So: pose_rotation = bone_rest_in_parent_inv @ parent_world_inv @ new_world

        parent_rest = pose_bone.parent.bone.matrix_local
        bone_rest_in_parent = parent_rest.inverted() @ pose_bone.bone.matrix_local
        bone_rest_in_parent_quat = bone_rest_in_parent.to_quaternion()

        rotation = bone_rest_in_parent_quat.inverted() @ parent_world_quat.inverted() @ new_world_quat
    else:
        # No parent - simpler
        rotation = bone_rest_world_quat.inverted() @ new_world_quat

    pose_bone.rotation_quaternion = rotation

    if debug:
        log_game("IK_APPLY", f"    bone_y_world: ({bone_y_world.x:.3f},{bone_y_world.y:.3f},{bone_y_world.z:.3f})")
        log_game("IK_APPLY", f"    world_rot needed: ({world_rotation.w:.3f},{world_rotation.x:.3f},{world_rotation.y:.3f},{world_rotation.z:.3f})")

    # =========================================================================
    # VERIFY - DID IT WORK?
    # =========================================================================
    bpy.context.view_layer.update()

    head_after = arm_matrix @ pose_bone.head
    tail_after = arm_matrix @ pose_bone.tail
    dir_after = (tail_after - head_after).normalized()

    # Check if bone now points at target
    dot_product = dir_after.dot(target_dir)
    angle_error = math.degrees(math.acos(max(-1, min(1, dot_product))))

    # Check where tail actually ended up vs where we wanted it
    expected_tail = head_after + target_dir * bone_length
    tail_error = (tail_after - expected_tail).length

    if debug:
        log_game("IK_APPLY", f"    AFTER:  head=({head_after.x:.3f},{head_after.y:.3f},{head_after.z:.3f})")
        log_game("IK_APPLY", f"    AFTER:  tail=({tail_after.x:.3f},{tail_after.y:.3f},{tail_after.z:.3f})")
        log_game("IK_APPLY", f"    AFTER:  dir=({dir_after.x:.3f},{dir_after.y:.3f},{dir_after.z:.3f})")
        log_game("IK_APPLY", f"    VERIFY: angle_error={angle_error:.1f}deg tail_error={tail_error*100:.1f}cm")

        if angle_error > 5:
            log_game("IK_APPLY", f"    !!! ROTATION FAILED - bone not pointing at target !!!")
            log_game("IK_APPLY", f"    !!! wanted dir ({target_dir.x:.3f},{target_dir.y:.3f},{target_dir.z:.3f})")
            log_game("IK_APPLY", f"    !!! got dir    ({dir_after.x:.3f},{dir_after.y:.3f},{dir_after.z:.3f})")

        # Did head move? (it shouldn't for local rotation)
        head_moved = (head_after - head_before).length
        if head_moved > 0.01:
            log_game("IK_APPLY", f"    !!! HEAD MOVED {head_moved*100:.1f}cm - unexpected !!!")


def rotation_between_vectors(v1: Vector, v2: Vector) -> Quaternion:
    """
    Compute quaternion rotation from v1 to v2.

    Uses the axis-angle approach with proper handling of edge cases.
    """
    v1 = v1.normalized()
    v2 = v2.normalized()

    dot = v1.dot(v2)

    if dot > 0.9999:
        return Quaternion()  # Identity - vectors already aligned

    if dot < -0.9999:
        # 180 degree rotation - need to find a perpendicular axis
        axis = v1.cross(Vector((1, 0, 0)))
        if axis.length < 0.001:
            axis = v1.cross(Vector((0, 1, 0)))
        axis.normalize()
        return Quaternion(axis, 3.14159265359)

    # Standard case: rotation around cross product axis
    axis = v1.cross(v2)
    axis.normalize()

    angle = float(np.arccos(np.clip(dot, -1, 1)))

    return Quaternion(axis, angle)
