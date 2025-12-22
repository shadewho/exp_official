# Exp_Game/animations/runtime_ik.py
"""
Runtime IK - Real-time IK solving during gameplay.

This module handles IK overlay on top of animations during gameplay.
For initial testing, IK solves on main thread (~50μs per solve).
Can be offloaded to workers later if needed.

Usage:
    from Exp_Game.animations.runtime_ik import apply_runtime_ik

    # In game loop, after animation is applied:
    apply_runtime_ik(armature, delta_time)
"""

import bpy
import numpy as np
from typing import Optional, Tuple, Dict
from mathutils import Vector, Quaternion

# Import worker-safe IK solver
from ..engine.animations.ik import (
    solve_two_bone_ik,
    LEG_IK, ARM_IK,
    compute_knee_pole_position,
    compute_elbow_pole_position,
)

# Import logger
from ..developer.dev_logger import log_game


# =============================================================================
# RUNTIME STATE
# =============================================================================

# Cached state for smooth IK transitions
_ik_state: Dict = {
    "last_target": None,
    "last_mid_pos": None,
    "last_influence": 0.0,
    "active": False,
}


# =============================================================================
# IK APPLICATION
# =============================================================================

def apply_runtime_ik(armature: bpy.types.Object, delta_time: float = 0.033) -> bool:
    """
    Apply runtime IK to armature based on scene properties.

    Call this AFTER animation pose is applied but BEFORE final render.

    Args:
        armature: The armature object to apply IK to
        delta_time: Frame time in seconds (for smoothing)

    Returns:
        True if IK was applied, False if skipped
    """
    scene = bpy.context.scene

    # Check if runtime IK is enabled
    if not getattr(scene, 'runtime_ik_enabled', False):
        _ik_state["active"] = False
        return False

    # Enable visualizer if IK visual debug is on
    if getattr(scene, 'dev_debug_ik_visual', False):
        from .test_panel import enable_ik_visualizer
        enable_ik_visualizer()

    if not armature or armature.type != 'ARMATURE':
        log_game("IK", "SKIP no_armature")
        return False

    # Get IK parameters from scene
    chain = getattr(scene, 'runtime_ik_chain', 'arm_R')
    influence = getattr(scene, 'runtime_ik_influence', 1.0)

    if influence < 0.001:
        log_game("IK", f"SKIP influence={influence:.3f}")
        return False

    # Get chain definition
    if chain.startswith("leg"):
        chain_def = LEG_IK.get(chain)
    elif chain.startswith("arm"):
        chain_def = ARM_IK.get(chain)
    else:
        log_game("IK", f"SKIP unknown_chain={chain}")
        return False

    if not chain_def:
        return False

    # Get bone references
    pose_bones = armature.pose.bones
    root_bone = pose_bones.get(chain_def["root"])
    mid_bone = pose_bones.get(chain_def["mid"])
    tip_bone = pose_bones.get(chain_def["tip"])

    if not all([root_bone, mid_bone, tip_bone]):
        log_game("IK", f"SKIP bones_not_found chain={chain}")
        return False

    # Get world positions from current pose
    arm_matrix = armature.matrix_world
    root_pos = np.array(arm_matrix @ root_bone.head, dtype=np.float32)
    tip_pos = np.array(arm_matrix @ tip_bone.head, dtype=np.float32)

    # Get IK target position
    target_obj = getattr(scene, 'runtime_ik_target', None)
    if target_obj is not None:
        # Use target object's world position
        target_pos = np.array(target_obj.matrix_world.translation, dtype=np.float32)
    else:
        # Fallback to computed test target
        target_pos = _compute_test_target(chain, root_pos, tip_pos, armature)

    # Compute pole position
    if chain.startswith("leg"):
        pole_pos = compute_knee_pole_position(
            root_pos, target_pos,
            forward=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            offset=0.5
        )
    else:
        pole_pos = compute_elbow_pole_position(
            root_pos, target_pos,
            backward=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            offset=0.3
        )

    # Solve IK
    upper_len = chain_def["len_upper"]
    lower_len = chain_def["len_lower"]
    max_reach = chain_def["reach"]

    # Check if target is reachable
    target_dist = float(np.linalg.norm(target_pos - root_pos))
    reachable = target_dist <= max_reach

    # Solve IK - get quaternions and mid position
    upper_quat, lower_quat, mid_pos = solve_two_bone_ik(
        root_pos=root_pos,
        target_pos=target_pos,
        pole_pos=pole_pos,
        upper_len=upper_len,
        lower_len=lower_len,
    )

    # Compute bone directions from IK solution
    # Upper bone points from root to mid
    upper_dir = mid_pos - root_pos
    upper_dir_norm = upper_dir / (np.linalg.norm(upper_dir) + 1e-10)

    # Lower bone points from mid to target
    lower_dir = target_pos - mid_pos
    lower_dir_norm = lower_dir / (np.linalg.norm(lower_dir) + 1e-10)

    # Apply to bones with influence blending
    _apply_ik_to_bone(root_bone, upper_dir_norm, influence, armature)
    _apply_ik_to_bone(mid_bone, lower_dir_norm, influence, armature)

    # Update state for visualization
    _ik_state["last_target"] = target_pos.copy()
    _ik_state["last_mid_pos"] = mid_pos.copy()
    _ik_state["last_influence"] = influence
    _ik_state["active"] = True
    _ik_state["chain"] = chain
    _ik_state["root_pos"] = root_pos.copy()
    _ik_state["pole_pos"] = pole_pos.copy()
    _ik_state["reachable"] = reachable

    # Log result
    log_game("IK",
        f"SOLVE chain={chain} target=({target_pos[0]:.2f},{target_pos[1]:.2f},{target_pos[2]:.2f}) "
        f"dist={target_dist:.3f}m reach={max_reach:.3f}m reachable={reachable} influence={influence:.2f}"
    )

    return True


def _compute_test_target(
    chain: str,
    root_pos: np.ndarray,
    current_tip: np.ndarray,
    armature: bpy.types.Object
) -> np.ndarray:
    """
    Compute a test target for IK.

    For testing purposes, we'll:
    - Legs: Try to ground the foot (use character Z as ground reference)
    - Arms: Reach forward from the shoulder (extended arm reach)

    TODO: Replace with actual game logic targets
    """
    # Get character position and forward direction
    char_z = armature.location.z
    # Get armature's forward direction (Y axis in Blender)
    arm_matrix = armature.matrix_world
    forward_world = np.array([arm_matrix[0][1], arm_matrix[1][1], arm_matrix[2][1]], dtype=np.float32)

    if chain.startswith("leg"):
        # Ground the foot - keep XY, set Z to ground
        target = current_tip.copy()
        target[2] = char_z + 0.1  # Slight offset for foot height
        return target
    else:
        # Arm: reach forward and outward from shoulder
        # Distance should be close to max reach (~0.56m for arms)
        reach_dist = 0.45  # Slightly less than max reach

        # Direction: forward + outward (left arm goes -X, right arm goes +X)
        if chain == "arm_L":
            side_offset = np.array([-0.2, 0.0, 0.0], dtype=np.float32)  # Left
        else:
            side_offset = np.array([0.2, 0.0, 0.0], dtype=np.float32)   # Right

        # Target is forward + side + slightly down from shoulder
        target = root_pos + forward_world * reach_dist * 0.8 + side_offset
        target[2] -= 0.1  # Slightly below shoulder height

        return target


def _apply_ik_to_bone(
    bone: bpy.types.PoseBone,
    target_direction: np.ndarray,
    influence: float,
    armature: bpy.types.Object
) -> None:
    """
    Apply IK rotation to a pose bone by pointing it toward a target direction.

    The IK solver gives us world-space directions. We need to convert these
    to local bone rotations that account for:
    - The armature's world transform
    - The bone's rest pose orientation
    - The parent bone's current rotation

    Args:
        bone: The pose bone to modify
        target_direction: World-space direction the bone should point (normalized)
        influence: Blend factor (0 = animation only, 1 = IK only)
        armature: The armature object (for world transform)
    """
    from mathutils import Matrix

    # Get current rotation for blending
    current = bone.rotation_quaternion.copy()

    # Get the bone's rest direction in armature space
    # Bones point along their Y axis in Blender
    bone_rest_matrix = bone.bone.matrix_local
    rest_direction = bone_rest_matrix.to_3x3() @ Vector((0, 1, 0))

    # Convert world target direction to armature space
    arm_inv = armature.matrix_world.inverted()
    target_arm_space = (arm_inv.to_3x3() @ Vector(target_direction)).normalized()

    # If bone has a parent, we need to account for parent's current pose
    if bone.parent:
        # Get parent's pose-space matrix
        parent_pose_matrix = bone.parent.matrix
        # Convert target to parent's local space
        parent_inv = parent_pose_matrix.inverted()
        target_local = (parent_inv.to_3x3() @ target_arm_space).normalized()
    else:
        target_local = target_arm_space

    # Compute rotation from rest direction to target direction
    # Both should now be in the same coordinate space
    rest_local = bone_rest_matrix.to_3x3().inverted() @ rest_direction
    if bone.parent:
        rest_local = (bone.parent.matrix.inverted().to_3x3() @ rest_direction).normalized()

    # Use rotation_difference to get the quaternion that rotates rest to target
    rest_vec = Vector(rest_local).normalized()
    target_vec = Vector(target_local).normalized()

    # Compute the rotation quaternion
    ik_rotation = rest_vec.rotation_difference(target_vec)

    # Slerp between current and IK based on influence
    if influence >= 0.999:
        bone.rotation_quaternion = ik_rotation
    else:
        blended = current.slerp(ik_rotation, influence)
        bone.rotation_quaternion = blended


# =============================================================================
# STATE ACCESS (for visualizer)
# =============================================================================

def get_ik_state() -> Dict:
    """
    Get current IK state for visualization.

    Returns:
        Dict with IK state (target, mid_pos, influence, active, etc.)
    """
    return _ik_state.copy()


def is_ik_active() -> bool:
    """Check if runtime IK is currently active."""
    return _ik_state.get("active", False)


def clear_ik_state() -> None:
    """Clear IK state (call when game stops)."""
    _ik_state.clear()
    _ik_state["active"] = False
