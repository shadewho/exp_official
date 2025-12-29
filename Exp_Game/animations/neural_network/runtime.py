# Exp_Game/animations/neural_network/runtime.py
"""
Runtime Integration for Neural IK

Applies trained neural network IK during gameplay:
1. Build context from current state
2. Predict rotations
3. Clamp to limits
4. Optionally run classical IK refinement
5. Apply to armature

This is the bridge between training and gameplay.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .config import (
    CONTROLLED_BONES,
    END_EFFECTORS,
    CONTACT_EFFECTORS,
    INPUT_SIZE,
    OUTPUT_SIZE,
    NUM_BONES,
    BONE_TO_INDEX,
    LIMITS_MIN,
    LIMITS_MAX,
    TASK_TYPES,
)
from .network import get_network, FullBodyIKNetwork
from .forward_kinematics import (
    forward_kinematics,
    axis_angle_to_matrix,
    clamp_rotations,
)


@dataclass
class IKResult:
    """Result from neural IK solve."""
    rotations: np.ndarray  # Shape (23, 3) - axis-angle per bone
    effector_positions: np.ndarray  # Shape (5, 3) - where effectors ended up
    effector_errors: np.ndarray  # Shape (5,) - distance to targets
    limit_violations: float  # How much was clamped
    success: bool  # Did we reach targets within threshold?


class NeuralIKSolver:
    """
    Runtime solver that uses the trained neural network.

    Usage:
        solver = NeuralIKSolver()
        result = solver.solve(
            effector_targets=...,
            root_position=...,
            root_forward=...,
            ground_heights=...,
            contact_flags=...,
        )
        # Apply result.rotations to armature
    """

    def __init__(self, network: FullBodyIKNetwork = None):
        """Initialize solver with network."""
        self.network = network or get_network()
        self.last_result: Optional[IKResult] = None

    def build_context(
        self,
        effector_targets: np.ndarray,
        effector_rotations: np.ndarray,
        root_forward: np.ndarray,
        root_up: np.ndarray,
        ground_heights: np.ndarray,
        ground_normals: np.ndarray,
        contact_flags: np.ndarray,
        motion_phase: float = 0.0,
        task_type: int = 0,
    ) -> np.ndarray:
        """
        Build the 50-dimensional context vector.

        Args:
            effector_targets: Shape (5, 3) - root-relative target positions
            effector_rotations: Shape (5, 3) - target rotations
            root_forward: Shape (3,) - root forward direction
            root_up: Shape (3,) - root up direction
            ground_heights: Shape (2,) - ground height per foot
            ground_normals: Shape (2, 3) - ground normal per foot
            contact_flags: Shape (2,) - is foot grounded
            motion_phase: 0-1 animation phase
            task_type: Task type index

        Returns:
            Shape (50,) context vector
        """
        # Interleave effector pos/rot
        effector_data = []
        for i in range(5):
            effector_data.extend(effector_targets[i])
            effector_data.extend(effector_rotations[i])

        # Root orientation
        root_data = list(root_forward) + list(root_up)

        # Ground context (6 per foot)
        ground_data = []
        for i in range(2):
            ground_data.append(ground_heights[i])
            ground_data.extend(ground_normals[i])
            ground_data.append(contact_flags[i])
            ground_data.append(1.0)  # desired_contact

        # Motion state
        motion_data = [motion_phase, float(task_type)]

        # Combine
        context = np.array(
            effector_data + root_data + ground_data + motion_data,
            dtype=np.float32
        )

        assert len(context) == INPUT_SIZE, f"Context size {len(context)} != {INPUT_SIZE}"
        return context

    def solve(
        self,
        effector_targets: np.ndarray,
        effector_rotations: np.ndarray = None,
        root_position: np.ndarray = None,
        root_forward: np.ndarray = None,
        root_up: np.ndarray = None,
        ground_heights: np.ndarray = None,
        ground_normals: np.ndarray = None,
        contact_flags: np.ndarray = None,
        motion_phase: float = 0.0,
        task_type: str = "idle",
        clamp_output: bool = True,
        error_threshold: float = 0.1,
    ) -> IKResult:
        """
        Solve IK for given targets.

        Args:
            effector_targets: Shape (5, 3) - target positions (root-relative)
            effector_rotations: Shape (5, 3) - target rotations (optional)
            root_position: Shape (3,) - root world position
            root_forward: Shape (3,) - root forward direction
            root_up: Shape (3,) - root up direction
            ground_heights: Shape (2,) - ground height per foot
            ground_normals: Shape (2, 3) - ground normal per foot
            contact_flags: Shape (2,) - is foot grounded
            motion_phase: 0-1 animation phase
            task_type: Task name
            clamp_output: Whether to clamp rotations to limits
            error_threshold: Max error for success

        Returns:
            IKResult with rotations and diagnostics
        """
        # Defaults
        if effector_rotations is None:
            effector_rotations = np.zeros((5, 3), dtype=np.float32)
        if root_position is None:
            root_position = np.array([0, 0, 1], dtype=np.float32)
        if root_forward is None:
            root_forward = np.array([0, 1, 0], dtype=np.float32)
        if root_up is None:
            root_up = np.array([0, 0, 1], dtype=np.float32)
        if ground_heights is None:
            ground_heights = np.zeros(2, dtype=np.float32)
        if ground_normals is None:
            ground_normals = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)
        if contact_flags is None:
            contact_flags = np.ones(2, dtype=np.float32)

        # Get task type index
        task_idx = TASK_TYPES.get(task_type, 0)

        # Build context
        context = self.build_context(
            effector_targets=effector_targets,
            effector_rotations=effector_rotations,
            root_forward=root_forward,
            root_up=root_up,
            ground_heights=ground_heights,
            ground_normals=ground_normals,
            contact_flags=contact_flags,
            motion_phase=motion_phase,
            task_type=task_idx,
        )

        # Predict rotations
        if clamp_output:
            rotations_flat, violation = self.network.predict_clamped(context)
        else:
            rotations_flat = self.network.predict(context)
            violation = 0.0

        # Reshape to (23, 3)
        rotations = rotations_flat.reshape(NUM_BONES, 3)

        # Run FK to see where effectors end up
        positions, _ = forward_kinematics(rotations, root_position)

        # Extract effector positions
        effector_indices = [BONE_TO_INDEX[e] for e in END_EFFECTORS]
        effector_positions = positions[effector_indices]

        # Convert targets from root-relative to world for comparison
        # (assuming targets are already in world space for now)
        effector_errors = np.linalg.norm(
            effector_positions - effector_targets,
            axis=1
        )

        # Check success
        max_error = float(np.max(effector_errors))
        success = max_error < error_threshold

        result = IKResult(
            rotations=rotations,
            effector_positions=effector_positions,
            effector_errors=effector_errors,
            limit_violations=violation,
            success=success,
        )

        self.last_result = result
        return result

    def solve_with_refinement(
        self,
        effector_targets: np.ndarray,
        max_iterations: int = 3,
        **kwargs,
    ) -> IKResult:
        """
        Solve with classical IK refinement after neural prediction.

        The network proposes an initial pose, then we refine with
        gradient descent to hit exact targets.

        Args:
            effector_targets: Shape (5, 3) - target positions
            max_iterations: Max refinement iterations
            **kwargs: Additional args passed to solve()

        Returns:
            IKResult with refined rotations
        """
        # Get initial prediction
        result = self.solve(effector_targets, **kwargs)

        if result.success:
            return result  # Already good enough

        # Refinement: simple gradient descent on FK error
        rotations = result.rotations.copy()
        learning_rate = 0.1

        for _ in range(max_iterations):
            # Compute FK
            positions, _ = forward_kinematics(rotations)

            # Effector positions
            effector_indices = [BONE_TO_INDEX[e] for e in END_EFFECTORS]
            current_pos = positions[effector_indices]

            # Error
            errors = effector_targets - current_pos
            max_error = float(np.max(np.linalg.norm(errors, axis=1)))

            if max_error < 0.05:
                break  # Good enough

            # Numerical gradient (simplified - in practice use Jacobian)
            # This is a placeholder for proper IK refinement
            epsilon = 0.01
            for i, effector_name in enumerate(END_EFFECTORS):
                chain = self._get_chain_for_effector(effector_name)
                for bone_idx in chain:
                    for axis in range(3):
                        # Perturb
                        rotations[bone_idx, axis] += epsilon
                        new_pos, _ = forward_kinematics(rotations)
                        new_effector_pos = new_pos[effector_indices[i]]

                        # Gradient
                        grad = (new_effector_pos - current_pos[i]) / epsilon

                        # Update in direction of error
                        rotations[bone_idx, axis] -= epsilon  # Undo perturbation
                        rotations[bone_idx, axis] += learning_rate * np.dot(errors[i], grad)

            # Clamp to limits
            rotations, _ = clamp_rotations(rotations, LIMITS_MIN, LIMITS_MAX)

        # Final result
        positions, _ = forward_kinematics(rotations)
        effector_positions = positions[[BONE_TO_INDEX[e] for e in END_EFFECTORS]]
        effector_errors = np.linalg.norm(effector_positions - effector_targets, axis=1)

        return IKResult(
            rotations=rotations,
            effector_positions=effector_positions,
            effector_errors=effector_errors,
            limit_violations=0.0,
            success=float(np.max(effector_errors)) < 0.1,
        )

    def _get_chain_for_effector(self, effector_name: str) -> List[int]:
        """Get bone indices in the chain leading to an effector."""
        from .config import IK_CHAINS

        for chain_name, chain_data in IK_CHAINS.items():
            if chain_data['tip'] == effector_name:
                return [BONE_TO_INDEX[b] for b in chain_data['bones']]

        return []


# =============================================================================
# Module-level solver
# =============================================================================

_solver: Optional[NeuralIKSolver] = None


def get_solver() -> NeuralIKSolver:
    """Get or create the global solver."""
    global _solver
    if _solver is None:
        _solver = NeuralIKSolver()
    return _solver


def solve_ik(
    effector_targets: np.ndarray,
    **kwargs,
) -> IKResult:
    """
    Convenience function to solve IK.

    Args:
        effector_targets: Shape (5, 3) - target positions
        **kwargs: Additional args

    Returns:
        IKResult
    """
    return get_solver().solve(effector_targets, **kwargs)


# =============================================================================
# BRIDGE TO OLD FULL_BODY_IK INTERFACE
# =============================================================================
# This wrapper converts the old FullBodyTarget format to the new context format.

def solve_from_full_body_target(
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    left_foot: np.ndarray,
    right_foot: np.ndarray,
    head_look_at: np.ndarray,
    hips_position: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    ground_height: float = 0.0,
    left_foot_grounded: bool = True,
    right_foot_grounded: bool = True,
    motion_phase: float = 0.0,
    task_type: str = "idle",
    use_refinement: bool = False,
) -> IKResult:
    """
    Solve IK using the old FullBodyTarget-style interface.

    This bridges the legacy interface with the new environment-aware neural IK.

    Args:
        left_hand: World position (3,)
        right_hand: World position (3,)
        left_foot: World position (3,)
        right_foot: World position (3,)
        head_look_at: World position to look at (3,)
        hips_position: World position of hips (3,)
        forward: Character forward direction (3,)
        up: Character up direction (3,)
        ground_height: Ground Z height
        left_foot_grounded: Whether left foot should be grounded
        right_foot_grounded: Whether right foot should be grounded
        motion_phase: Animation phase 0-1
        task_type: Task name
        use_refinement: Whether to use classical IK refinement after neural prediction

    Returns:
        IKResult with bone rotations
    """
    # Convert to numpy arrays
    left_hand = np.asarray(left_hand, dtype=np.float32)
    right_hand = np.asarray(right_hand, dtype=np.float32)
    left_foot = np.asarray(left_foot, dtype=np.float32)
    right_foot = np.asarray(right_foot, dtype=np.float32)
    head_look_at = np.asarray(head_look_at, dtype=np.float32)
    hips_position = np.asarray(hips_position, dtype=np.float32)
    forward = np.asarray(forward, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    # Build root transform for world-to-root conversion
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)

    # Convert world positions to root-relative (Hips frame)
    def to_root_relative(world_pos):
        offset = world_pos - hips_position
        return np.array([
            np.dot(offset, right),
            np.dot(offset, forward),
            np.dot(offset, up),
        ], dtype=np.float32)

    # Effector targets in root-relative space
    # Order: LeftHand, RightHand, LeftFoot, RightFoot, Head
    effector_targets = np.array([
        to_root_relative(left_hand),
        to_root_relative(right_hand),
        to_root_relative(left_foot),
        to_root_relative(right_foot),
        to_root_relative(head_look_at),
    ], dtype=np.float32)

    # Ground context
    ground_heights = np.array([
        ground_height,
        ground_height,
    ], dtype=np.float32)

    ground_normals = np.array([
        [0.0, 0.0, 1.0],  # Up
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    contact_flags = np.array([
        1.0 if left_foot_grounded else 0.0,
        1.0 if right_foot_grounded else 0.0,
    ], dtype=np.float32)

    # Solve
    solver = get_solver()

    if use_refinement:
        return solver.solve_with_refinement(
            effector_targets=effector_targets,
            root_position=hips_position,
            root_forward=forward,
            root_up=up,
            ground_heights=ground_heights,
            ground_normals=ground_normals,
            contact_flags=contact_flags,
            motion_phase=motion_phase,
            task_type=task_type,
        )
    else:
        return solver.solve(
            effector_targets=effector_targets,
            root_position=hips_position,
            root_forward=forward,
            root_up=up,
            ground_heights=ground_heights,
            ground_normals=ground_normals,
            contact_flags=contact_flags,
            motion_phase=motion_phase,
            task_type=task_type,
        )


def rotations_to_bone_dict(rotations: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convert rotation array to bone name -> rotation dict.

    Args:
        rotations: Shape (23, 3) - axis-angle per bone

    Returns:
        Dict mapping bone name to rotation (3,)
    """
    result = {}
    for i, bone_name in enumerate(CONTROLLED_BONES):
        result[bone_name] = rotations[i]
    return result
