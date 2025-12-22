# Exp_Game/animations/runtime_ik.py
"""
Runtime IK - Real-time IK solving during gameplay.

This module handles IK overlay on top of animations during gameplay.
For initial testing, IK solves on main thread (~50μs per solve).
Can be offloaded to workers later if needed.

ARCHITECTURE:
    1. BlendSystem IK targets (PRODUCTION) - programmatic IK from code/reactions
       - blend_sys.set_ik_target("arm_R", position)
       - blend_sys.set_ik_target_object("arm_R", "TargetCube")

    2. Scene property IK (TESTING) - manual testing via Developer Tools panel
       - scene.runtime_ik_enabled, runtime_ik_chain, runtime_ik_target

Usage:
    from Exp_Game.animations.runtime_ik import apply_runtime_ik

    # In game loop, after animation is applied:
    apply_runtime_ik(armature, delta_time)
"""

import bpy
import numpy as np
from typing import Optional, Tuple, Dict, TYPE_CHECKING
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

if TYPE_CHECKING:
    from .blend_system import IKTarget


# =============================================================================
# RUNTIME STATE
# =============================================================================

# Cached state for visualization (per-chain)
_ik_state: Dict = {
    "chains": {},  # chain_name -> {last_target, last_mid_pos, ...}
    "active": False,
}


# =============================================================================
# IK APPLICATION
# =============================================================================

def apply_runtime_ik(armature: bpy.types.Object, delta_time: float = 0.033) -> bool:
    """
    Apply runtime IK to armature.

    Checks two sources for IK targets (in order):
    1. BlendSystem IK targets (production/programmatic)
    2. Scene properties (testing/development) - only if "Use BlendSystem" is OFF

    When "Use BlendSystem" toggle is ON in the UI, scene properties are routed
    through BlendSystem to test the production code path.

    Call this AFTER animation pose is applied but BEFORE final render.

    Args:
        armature: The armature object to apply IK to
        delta_time: Frame time in seconds (for smoothing)

    Returns:
        True if IK was applied, False if skipped
    """
    if not armature or armature.type != 'ARMATURE':
        return False

    scene = bpy.context.scene
    applied_any = False

    # Enable visualizer if IK visual debug is on
    if getattr(scene, 'dev_debug_ik_visual', False):
        from .test_panel import enable_ik_visualizer
        enable_ik_visualizer()

    # =========================================================================
    # SCENE PROPERTY → BLENDSYSTEM BRIDGE (for UI testing)
    # When "Use BlendSystem" is enabled, route scene props through BlendSystem
    # =========================================================================
    from .blend_system import get_blend_system

    blend_sys = get_blend_system()
    use_blend_system = getattr(scene, 'runtime_ik_use_blend_system', False)
    scene_ik_enabled = getattr(scene, 'runtime_ik_enabled', False)

    if blend_sys and use_blend_system and scene_ik_enabled:
        # Route scene properties through BlendSystem
        chain = getattr(scene, 'runtime_ik_chain', 'arm_R')
        influence = getattr(scene, 'runtime_ik_influence', 1.0)
        target_obj = getattr(scene, 'runtime_ik_target', None)

        if target_obj is not None and influence >= 0.001:
            # Set BlendSystem IK target from scene properties
            blend_sys.set_ik_target_object(chain, target_obj.name, influence)
        elif target_obj is None:
            # Clear BlendSystem target if no target object
            blend_sys.clear_ik_target(chain)

    # =========================================================================
    # SOURCE 1: BlendSystem IK targets (PRODUCTION)
    # =========================================================================
    if blend_sys:
        active_targets = blend_sys.get_active_ik_targets()

        for chain, ik_target in active_targets.items():
            # Resolve target position
            if ik_target.target_object:
                # Track object position
                obj = bpy.data.objects.get(ik_target.target_object)
                if obj:
                    target_pos = np.array(obj.matrix_world.translation, dtype=np.float32)
                else:
                    log_game("IK", f"SKIP object_not_found chain={chain} obj={ik_target.target_object}")
                    continue
            else:
                # Use stored position
                target_pos = ik_target.target_position

            # Solve and apply
            success = _solve_and_apply_chain(
                armature=armature,
                chain=chain,
                target_pos=target_pos,
                influence=ik_target.influence,
                pole_pos=ik_target.pole_position,
                source="BlendSystem"
            )

            if success:
                applied_any = True

    # =========================================================================
    # SOURCE 2: Scene properties (TESTING - direct path)
    # Only used if "Use BlendSystem" is OFF and scene testing is enabled
    # =========================================================================
    if not applied_any and scene_ik_enabled and not use_blend_system:
        chain = getattr(scene, 'runtime_ik_chain', 'arm_R')
        influence = getattr(scene, 'runtime_ik_influence', 1.0)

        if influence >= 0.001:
            # Get target position from scene
            target_obj = getattr(scene, 'runtime_ik_target', None)
            if target_obj is not None:
                target_pos = np.array(target_obj.matrix_world.translation, dtype=np.float32)
            else:
                # Compute test target
                target_pos = _compute_test_target_for_chain(chain, armature)

            if target_pos is not None:
                success = _solve_and_apply_chain(
                    armature=armature,
                    chain=chain,
                    target_pos=target_pos,
                    influence=influence,
                    pole_pos=None,
                    source="SceneProps"
                )
                if success:
                    applied_any = True

    # Update global state
    _ik_state["active"] = applied_any

    return applied_any


def _solve_and_apply_chain(
    armature: bpy.types.Object,
    chain: str,
    target_pos: np.ndarray,
    influence: float,
    pole_pos: Optional[np.ndarray] = None,
    source: str = "unknown"
) -> bool:
    """
    Solve IK for a single chain and apply to bones.

    Args:
        armature: The armature object
        chain: Chain name ("arm_L", "arm_R", "leg_L", "leg_R")
        target_pos: World-space target position
        influence: IK influence (0-1)
        pole_pos: Optional pole position (computed if None)
        source: Source identifier for logging

    Returns:
        True if IK was applied successfully
    """
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

    # Extract character orientation from armature matrix
    # In Blender: Y = forward, X = right, Z = up
    char_forward = np.array([arm_matrix[0][1], arm_matrix[1][1], arm_matrix[2][1]], dtype=np.float32)
    char_right = np.array([arm_matrix[0][0], arm_matrix[1][0], arm_matrix[2][0]], dtype=np.float32)
    char_up = np.array([arm_matrix[0][2], arm_matrix[1][2], arm_matrix[2][2]], dtype=np.float32)

    # Normalize (in case armature has scale)
    char_forward = char_forward / (np.linalg.norm(char_forward) + 1e-10)
    char_right = char_right / (np.linalg.norm(char_right) + 1e-10)
    char_up = char_up / (np.linalg.norm(char_up) + 1e-10)

    # Determine side from chain name
    side = "L" if chain.endswith("_L") or chain.endswith("_l") else "R"

    # Compute pole position if not provided (character-relative)
    if pole_pos is None:
        if chain.startswith("leg"):
            pole_pos = compute_knee_pole_position(
                root_pos, target_pos,
                char_forward=char_forward,
                char_right=char_right,
                side=side,
                offset=0.5
            )
        else:
            pole_pos = compute_elbow_pole_position(
                root_pos, target_pos,
                char_forward=char_forward,
                char_up=char_up,
                side=side,
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
    upper_dir = mid_pos - root_pos
    upper_dir_norm = upper_dir / (np.linalg.norm(upper_dir) + 1e-10)

    lower_dir = target_pos - mid_pos
    lower_dir_norm = lower_dir / (np.linalg.norm(lower_dir) + 1e-10)

    # Apply to bones with influence blending
    _apply_ik_to_bone(root_bone, upper_dir_norm, influence, armature)
    _apply_ik_to_bone(mid_bone, lower_dir_norm, influence, armature)

    # Update per-chain state for visualization
    _ik_state["chains"][chain] = {
        "last_target": target_pos.copy(),
        "last_mid_pos": mid_pos.copy(),
        "root_pos": root_pos.copy(),
        "pole_pos": pole_pos.copy(),
        "influence": influence,
        "reachable": reachable,
    }

    # Also set legacy state for visualizer compatibility
    _ik_state["last_target"] = target_pos.copy()
    _ik_state["last_mid_pos"] = mid_pos.copy()
    _ik_state["last_influence"] = influence
    _ik_state["chain"] = chain
    _ik_state["root_pos"] = root_pos.copy()
    _ik_state["pole_pos"] = pole_pos.copy()
    _ik_state["reachable"] = reachable

    # Log result
    log_game("IK",
        f"SOLVE src={source} chain={chain} target=({target_pos[0]:.2f},{target_pos[1]:.2f},{target_pos[2]:.2f}) "
        f"dist={target_dist:.3f}m reach={max_reach:.3f}m ok={reachable} inf={influence:.2f}"
    )

    return True


def _compute_test_target_for_chain(chain: str, armature: bpy.types.Object) -> Optional[np.ndarray]:
    """
    Compute a test target position for a chain.

    Used by scene property testing mode.
    """
    # Get chain definition
    if chain.startswith("leg"):
        chain_def = LEG_IK.get(chain)
    elif chain.startswith("arm"):
        chain_def = ARM_IK.get(chain)
    else:
        return None

    if not chain_def:
        return None

    pose_bones = armature.pose.bones
    root_bone = pose_bones.get(chain_def["root"])
    tip_bone = pose_bones.get(chain_def["tip"])

    if not root_bone or not tip_bone:
        return None

    arm_matrix = armature.matrix_world
    root_pos = np.array(arm_matrix @ root_bone.head, dtype=np.float32)
    tip_pos = np.array(arm_matrix @ tip_bone.head, dtype=np.float32)

    return _compute_test_target(chain, root_pos, tip_pos, armature)


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

    Computes the rotation needed to point bone Y-axis toward target,
    working in the bone's local coordinate space.

    Args:
        bone: The pose bone to modify
        target_direction: World-space direction the bone should point (normalized)
        influence: Blend factor (0 = animation only, 1 = IK only)
        armature: The armature object (for world transform)
    """
    # Get current rotation for blending (this is animation + any previous IK)
    current = bone.rotation_quaternion.copy()

    # Convert world target direction to armature space
    arm_matrix_inv = armature.matrix_world.inverted()
    target_arm = Vector((arm_matrix_inv.to_3x3() @ Vector(target_direction))).normalized()

    # Get target direction in bone's LOCAL space
    # We need to transform through the parent chain
    if bone.parent:
        # Parent's posed matrix transforms from parent-local to armature space
        # We need the inverse to go armature -> parent-local
        parent_matrix_inv = bone.parent.matrix.inverted()
        target_parent = (parent_matrix_inv.to_3x3() @ target_arm).normalized()
    else:
        target_parent = target_arm

    # Now transform from parent space to bone-local space using rest pose
    # bone.bone.matrix_local is bone rest pose in armature space
    # We need just the rotation part relative to parent
    if bone.parent:
        # Get bone's rest orientation relative to parent's rest
        parent_rest = bone.parent.bone.matrix_local
        bone_rest = bone.bone.matrix_local
        bone_rest_local = parent_rest.inverted() @ bone_rest
        rest_rot_inv = bone_rest_local.to_3x3().inverted()
    else:
        rest_rot_inv = bone.bone.matrix_local.to_3x3().inverted()

    target_local = (rest_rot_inv @ target_parent).normalized()

    # In bone-local space, Y is the bone direction
    # Find rotation from Y-axis to target direction
    bone_y = Vector((0, 1, 0))
    ik_rotation = bone_y.rotation_difference(target_local)

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
    _ik_state["chains"] = {}
    _ik_state["active"] = False
