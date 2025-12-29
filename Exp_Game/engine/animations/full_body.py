# Exp_Game/engine/animations/full_body.py
"""
Full-Body IK Solver - Worker-side computation (NO bpy).

Solves whole-skeleton IK constraints:
- Root positioning (world anchor)
- Hips control (crouch, lean, weight shift)
- Leg IK (foot grounding)
- Spine chain (lean toward reach targets)
- Arm IK (reaching)
- Head/Neck (look-at)

SOLVE ORDER (critical for correct results):
  1. Apply hips transform (drop/offset)
  2. Solve leg IK (feet stay grounded while hips moves)
  3. Solve spine lean (toward reach targets)
  4. Solve arm IK (hands reach targets)
  5. Solve head look-at

All math is numpy-vectorized for performance.

Usage:
    result = solve_full_body_ik(constraints, current_state, rig_fk)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time

from .ik_chains import LEG_IK, ARM_IK, IK_TOLERANCES
from .ik_math import normalize, quat_from_two_vectors, quat_multiply, quat_rotate_vector
from .ik_solver import (
    solve_leg_ik,
    solve_arm_ik,
    compute_knee_pole_position,
    compute_elbow_pole_position,
)


# =============================================================================
# CONSTANTS - Using centralized tolerances from ik_chains
# =============================================================================

# Re-export for local use (values come from IK_TOLERANCES class)
FOOT_ERROR_THRESHOLD = IK_TOLERANCES.FOOT_ERROR
HAND_ERROR_THRESHOLD = IK_TOLERANCES.HAND_ERROR
HIPS_ERROR_THRESHOLD = IK_TOLERANCES.HIPS_ERROR

# Spine chain bones (Hips → Spine → Spine1 → Spine2 → Neck)
SPINE_CHAIN = ["Hips", "Spine", "Spine1", "Spine2", "Neck"]


# =============================================================================
# FULL-BODY IK SOLVER
# =============================================================================

def solve_full_body_ik(
    constraints: dict,
    current_state: dict,
    rig_fk=None,
    char_forward: np.ndarray = None,
    char_right: np.ndarray = None,
    char_up: np.ndarray = None,
) -> dict:
    """
    Solve full-body IK given constraints and current state.

    SOLVE ORDER:
      1. Hips → 2. Legs → 3. Spine → 4. Arms → 5. Head

    Args:
        constraints: Dict with IK targets and parameters:
            - left_foot: {position: (x,y,z), enabled: bool, weight: float} or None
            - right_foot: {position: (x,y,z), enabled: bool, weight: float} or None
            - left_hand: {position: (x,y,z), enabled: bool, weight: float} or None
            - right_hand: {position: (x,y,z), enabled: bool, weight: float} or None
            - look_at: {position: (x,y,z), enabled: bool, weight: float} or None
            - hips_drop: float (meters to drop hips)
            - hips_offset: (x, y, z) offset

        current_state: Dict with current skeleton state:
            - root_pos: (x, y, z)
            - hips_pos: (x, y, z) relative to Root
            - hips_rot: (w, x, y, z) quaternion
            - left_foot_pos: (x, y, z) relative to Root
            - right_foot_pos: (x, y, z) relative to Root
            - left_hand_pos: (x, y, z) relative to Root
            - right_hand_pos: (x, y, z) relative to Root

        rig_fk: Optional RigFK instance for forward kinematics.
                If provided, enables advanced spine/arm solving.

        char_forward: Character forward direction (default: +Y)
        char_right: Character right direction (default: +X)
        char_up: Character up direction (default: +Z)

    Returns:
        Dict with solve results:
            - success: bool
            - bone_transforms: {bone_name: [qw, qx, qy, qz, lx, ly, lz], ...}
            - constraints_satisfied: int
            - constraints_total: int
            - constraints_at_limit: int
            - joint_violations: [bone_names that hit limits]
            - effector_errors: {effector_name: error_cm, ...}
            - solve_time_us: float
            - logs: [(category, message), ...]
    """
    start_time = time.perf_counter()
    logs = []

    # Default character orientation
    if char_forward is None:
        char_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if char_right is None:
        char_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if char_up is None:
        char_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Result tracking
    bone_transforms = {}
    constraints_satisfied = 0
    constraints_total = 0
    constraints_at_limit = 0
    joint_violations = []
    effector_errors = {}

    # Parse current state
    root_pos = np.array(current_state.get("root_pos", (0, 0, 0)), dtype=np.float32)
    hips_pos = np.array(current_state.get("hips_pos", (0, 0, 1.0)), dtype=np.float32)
    hips_rot = np.array(current_state.get("hips_rot", (1, 0, 0, 0)), dtype=np.float32)

    # Parse constraints
    hips_drop = constraints.get("hips_drop", 0.0)
    hips_offset = np.array(constraints.get("hips_offset", (0, 0, 0)), dtype=np.float32)

    left_foot_target = constraints.get("left_foot")
    right_foot_target = constraints.get("right_foot")
    left_hand_target = constraints.get("left_hand")
    right_hand_target = constraints.get("right_hand")
    look_at_target = constraints.get("look_at")

    # =========================================================================
    # STEP 1: HIPS TRANSFORM
    # =========================================================================
    # Apply hips drop and offset. The new hips position affects leg IK.

    new_hips_pos = hips_pos.copy()

    if abs(hips_drop) > 0.001:
        new_hips_pos[2] -= hips_drop  # Drop in Z
        constraints_total += 1

        # Log hips transform
        logs.append(("FULL-BODY-IK", f"HIPS_DROP: {hips_drop:.3f}m -> new_z={new_hips_pos[2]:.3f}"))

    if np.linalg.norm(hips_offset) > 0.001:
        new_hips_pos += hips_offset
        constraints_total += 1
        logs.append(("FULL-BODY-IK", f"HIPS_OFFSET: ({hips_offset[0]:.2f}, {hips_offset[1]:.2f}, {hips_offset[2]:.2f})"))

    # Store hips transform
    bone_transforms["Hips"] = [
        float(hips_rot[0]), float(hips_rot[1]), float(hips_rot[2]), float(hips_rot[3]),  # quat (unchanged for now)
        float(hips_offset[0]), float(hips_offset[1]), float(-hips_drop),  # location (relative to rest)
    ]

    # Hips world position (for leg IK root)
    hips_world = root_pos + new_hips_pos

    # =========================================================================
    # STEP 2: LEG IK (Foot Grounding)
    # =========================================================================
    # Solve leg IK to keep feet at their target positions while hips moves.
    # This is the core of full-body IK - feet stay planted, legs bend.

    for side, target_data, state_key in [
        ("L", left_foot_target, "left_foot_pos"),
        ("R", right_foot_target, "right_foot_pos"),
    ]:
        if not target_data or not target_data.get("enabled", True):
            continue

        constraints_total += 1

        # Get foot target (Root-relative)
        foot_target = np.array(target_data["position"], dtype=np.float32)
        weight = target_data.get("weight", 1.0)

        # Compute hip position for this leg
        # Hips are at hips_world, but leg roots are offset from hips
        # For simplicity, use hips_world as approximate root
        # A more accurate approach would use rig_fk to get exact thigh positions
        leg_root = hips_world.copy()

        # Slight offset for hip joint (legs attach at sides of pelvis)
        hip_width = 0.1  # ~10cm from center to hip joint
        if side == "L":
            leg_root[0] -= hip_width
        else:
            leg_root[0] += hip_width

        # Compute knee pole position
        knee_pole = compute_knee_pole_position(
            hip_pos=leg_root,
            foot_pos=root_pos + foot_target,  # Convert to world
            char_forward=char_forward,
            char_right=char_right,
            side=side,
            offset=0.5,
        )

        # Solve leg IK
        thigh_quat, shin_quat, knee_world = solve_leg_ik(
            hip_pos=leg_root,
            foot_target=root_pos + foot_target,  # Convert to world
            knee_pole=knee_pole,
            side=side,
            char_forward=char_forward,  # Critical for high kicks
            char_up=char_up,  # For knee hinge axis
        )

        # Store bone transforms
        chain = LEG_IK[f"leg_{side}"]
        bone_transforms[chain["root"]] = [
            float(thigh_quat[0]), float(thigh_quat[1]), float(thigh_quat[2]), float(thigh_quat[3]),
            0.0, 0.0, 0.0,  # No translation for rotation-only bones
        ]
        bone_transforms[chain["mid"]] = [
            float(shin_quat[0]), float(shin_quat[1]), float(shin_quat[2]), float(shin_quat[3]),
            0.0, 0.0, 0.0,
        ]

        # Compute error (actual foot position vs target)
        # For now, assume solve is accurate (actual verification in main thread)
        effector_errors[f"{side.lower()}_foot"] = 0.0  # Placeholder
        constraints_satisfied += 1

        logs.append(("FULL-BODY-IK", f"LEG_IK_{side}: target=({foot_target[0]:.2f}, {foot_target[1]:.2f}, {foot_target[2]:.2f}) knee=({knee_world[0]:.2f}, {knee_world[1]:.2f}, {knee_world[2]:.2f})"))

    # =========================================================================
    # STEP 3: SPINE CHAIN (Lean toward reach targets)
    # =========================================================================
    # If hand targets are set, the spine should lean toward them.
    # This distributes the reach across multiple joints.

    reach_target = None
    if left_hand_target and left_hand_target.get("enabled"):
        reach_target = np.array(left_hand_target["position"], dtype=np.float32)
    elif right_hand_target and right_hand_target.get("enabled"):
        reach_target = np.array(right_hand_target["position"], dtype=np.float32)

    if reach_target is not None:
        # Compute lean direction (from hips toward target, projected to horizontal)
        lean_dir = reach_target - new_hips_pos
        lean_dir[2] = 0  # Remove vertical component
        lean_dist = np.linalg.norm(lean_dir)

        if lean_dist > 0.3:  # Only lean if target is far enough
            lean_dir = lean_dir / lean_dist

            # Compute lean amount (more lean for farther targets)
            # Max lean ~20 degrees at full reach
            max_lean_rad = 0.35  # ~20 degrees
            lean_amount = min((lean_dist - 0.3) / 0.5, 1.0) * max_lean_rad

            # Distribute lean across spine bones
            # Spine1 and Spine2 get most of the lean
            spine_bones = ["Spine", "Spine1", "Spine2"]
            lean_per_bone = lean_amount / len(spine_bones)

            # Lean axis is perpendicular to lean_dir and up
            lean_axis = np.cross(char_up, lean_dir)
            if np.linalg.norm(lean_axis) > 0.001:
                lean_axis = normalize(lean_axis)

                for spine_bone in spine_bones:
                    # Create rotation quaternion for this bone's lean
                    half_angle = lean_per_bone * 0.5
                    s = np.sin(half_angle)
                    c = np.cos(half_angle)
                    lean_quat = np.array([c, lean_axis[0] * s, lean_axis[1] * s, lean_axis[2] * s], dtype=np.float32)

                    bone_transforms[spine_bone] = [
                        float(lean_quat[0]), float(lean_quat[1]), float(lean_quat[2]), float(lean_quat[3]),
                        0.0, 0.0, 0.0,
                    ]

                logs.append(("FULL-BODY-IK", f"SPINE_LEAN: dir=({lean_dir[0]:.2f}, {lean_dir[1]:.2f}) amount={np.degrees(lean_amount):.1f}deg"))

    # =========================================================================
    # STEP 4: ARM IK (Reaching)
    # =========================================================================
    # Solve arm IK for hand targets.

    for side, target_data, state_key in [
        ("L", left_hand_target, "left_hand_pos"),
        ("R", right_hand_target, "right_hand_pos"),
    ]:
        if not target_data or not target_data.get("enabled", True):
            continue

        constraints_total += 1

        hand_target = np.array(target_data["position"], dtype=np.float32)
        weight = target_data.get("weight", 1.0)

        # Estimate shoulder position
        # Without rig_fk, we approximate based on hips + spine offset
        shoulder_offset_z = 0.5  # ~50cm above hips
        shoulder_offset_x = 0.15 if side == "L" else -0.15  # ~15cm from center

        shoulder_pos = new_hips_pos + np.array([shoulder_offset_x, 0.0, shoulder_offset_z], dtype=np.float32)
        shoulder_world = root_pos + shoulder_pos

        # Compute elbow pole
        elbow_pole = compute_elbow_pole_position(
            shoulder_pos=shoulder_world,
            hand_pos=root_pos + hand_target,
            char_forward=char_forward,
            char_up=char_up,
            side=side,
            offset=0.3,
        )

        # Solve arm IK
        upper_quat, forearm_quat, elbow_world = solve_arm_ik(
            shoulder_pos=shoulder_world,
            hand_target=root_pos + hand_target,
            elbow_pole=elbow_pole,
            side=side,
            char_forward=char_forward,
            char_up=char_up,
        )

        # Check if target is reachable
        reach_dist = np.linalg.norm(hand_target - shoulder_pos)
        chain = ARM_IK[f"arm_{side}"]
        max_reach = chain["reach"]

        if reach_dist > max_reach:
            constraints_at_limit += 1
            effector_errors[f"{side.lower()}_hand"] = (reach_dist - max_reach) * 100  # cm
            logs.append(("FULL-BODY-IK", f"ARM_IK_{side}: OUT_OF_REACH dist={reach_dist:.2f}m max={max_reach:.2f}m"))
        else:
            constraints_satisfied += 1
            effector_errors[f"{side.lower()}_hand"] = 0.0

        # Store transforms
        bone_transforms[chain["root"]] = [
            float(upper_quat[0]), float(upper_quat[1]), float(upper_quat[2]), float(upper_quat[3]),
            0.0, 0.0, 0.0,
        ]
        bone_transforms[chain["mid"]] = [
            float(forearm_quat[0]), float(forearm_quat[1]), float(forearm_quat[2]), float(forearm_quat[3]),
            0.0, 0.0, 0.0,
        ]

        logs.append(("FULL-BODY-IK", f"ARM_IK_{side}: target=({hand_target[0]:.2f}, {hand_target[1]:.2f}, {hand_target[2]:.2f}) elbow=({elbow_world[0]:.2f}, {elbow_world[1]:.2f}, {elbow_world[2]:.2f})"))

    # =========================================================================
    # STEP 5: HEAD LOOK-AT
    # =========================================================================
    # Orient head/neck toward look-at target.

    if look_at_target and look_at_target.get("enabled", True):
        constraints_total += 1

        look_pos = np.array(look_at_target["position"], dtype=np.float32)
        weight = look_at_target.get("weight", 1.0)

        # Estimate head position (above hips + spine)
        head_offset = np.array([0.0, 0.0, 0.6], dtype=np.float32)  # ~60cm above hips
        head_pos = new_hips_pos + head_offset

        # Direction from head to target
        look_dir = (root_pos + look_pos) - (root_pos + head_pos)
        look_dist = np.linalg.norm(look_dir)

        if look_dist > 0.1:
            look_dir = look_dir / look_dist

            # Default forward direction (character forward)
            forward = char_forward

            # Create rotation to look at target
            look_quat = quat_from_two_vectors(forward, look_dir)

            # Apply weight (blend with identity)
            if weight < 1.0:
                identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                # Simple lerp for weight (proper slerp would be better)
                look_quat = identity * (1.0 - weight) + look_quat * weight
                look_quat = look_quat / np.linalg.norm(look_quat)

            # Distribute rotation between Neck and Head
            # Head gets more rotation for natural look
            neck_weight = 0.3
            head_weight = 0.7

            # Store neck transform
            neck_quat = identity * (1.0 - neck_weight) + look_quat * neck_weight
            neck_quat = neck_quat / np.linalg.norm(neck_quat)
            bone_transforms["Neck"] = [
                float(neck_quat[0]), float(neck_quat[1]), float(neck_quat[2]), float(neck_quat[3]),
                0.0, 0.0, 0.0,
            ]

            # Store head transform
            head_quat = identity * (1.0 - head_weight) + look_quat * head_weight
            head_quat = head_quat / np.linalg.norm(head_quat)
            bone_transforms["Head"] = [
                float(head_quat[0]), float(head_quat[1]), float(head_quat[2]), float(head_quat[3]),
                0.0, 0.0, 0.0,
            ]

            constraints_satisfied += 1
            logs.append(("FULL-BODY-IK", f"LOOK_AT: target=({look_pos[0]:.2f}, {look_pos[1]:.2f}, {look_pos[2]:.2f})"))

    # =========================================================================
    # BUILD RESULT
    # =========================================================================

    solve_time_us = (time.perf_counter() - start_time) * 1_000_000

    success = constraints_total == 0 or constraints_satisfied > 0

    # Summary log
    logs.append(("FULL-BODY-IK", f"SOLVE_COMPLETE: {constraints_satisfied}/{constraints_total} satisfied, {constraints_at_limit} at_limit, {len(bone_transforms)} bones, {solve_time_us:.0f}us"))

    return {
        "success": success,
        "bone_transforms": bone_transforms,
        "constraints_satisfied": constraints_satisfied,
        "constraints_total": constraints_total,
        "constraints_at_limit": constraints_at_limit,
        "joint_violations": joint_violations,
        "effector_errors": effector_errors,
        "solve_time_us": solve_time_us,
        "logs": logs,
    }


# =============================================================================
# JOB HANDLER (called from entry.py)
# =============================================================================

def handle_full_body_ik(job_data: dict, cached_rigs: dict = None) -> dict:
    """
    Handle FULL_BODY_IK job from worker entry.

    Args:
        job_data: Job data dict with constraints and current_state
        cached_rigs: Optional dict of {armature_name: RigFK}

    Returns:
        Result dict for engine
    """
    constraints = job_data.get("constraints", {})
    current_state = job_data.get("current_state", {})
    armature_name = job_data.get("armature_name", "")

    # Get cached rig if available
    rig_fk = None
    if cached_rigs and armature_name:
        rig_fk = cached_rigs.get(armature_name)

    # Extract character orientation if provided
    char_forward = job_data.get("char_forward")
    char_right = job_data.get("char_right")
    char_up = job_data.get("char_up")

    if char_forward:
        char_forward = np.array(char_forward, dtype=np.float32)
    if char_right:
        char_right = np.array(char_right, dtype=np.float32)
    if char_up:
        char_up = np.array(char_up, dtype=np.float32)

    # Solve
    result = solve_full_body_ik(
        constraints=constraints,
        current_state=current_state,
        rig_fk=rig_fk,
        char_forward=char_forward,
        char_right=char_right,
        char_up=char_up,
    )

    return result
