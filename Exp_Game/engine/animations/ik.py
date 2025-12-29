# Exp_Game/engine/animations/ik.py
"""
IK Solver - Compatibility Layer.

This module re-exports from the refactored IK modules for backwards compatibility.
All new code should import directly from:
    - ik_chains.py  - Chain definitions (LEG_IK, ARM_IK)
    - ik_math.py    - Vector/quaternion utilities
    - ik_solver.py  - Core solving functions

Worker-safe (NO bpy imports).
"""

# =============================================================================
# RE-EXPORTS FROM NEW MODULES
# =============================================================================

# Chain definitions
from .ik_chains import (
    LEG_IK,
    ARM_IK,
    SPINE_CHAIN,
    BODY_DIMENSIONS,
    IK_TOLERANCES,
    get_chain,
    get_chain_type,
    get_chain_side,
    get_max_reach,
)

# Math utilities
from .ik_math import (
    normalize,
    safe_normalize,
    dot,
    cross,
    lerp,
    project_onto_plane,
    quat_identity,
    quat_normalize,
    quat_multiply,
    quat_conjugate,
    quat_inverse,
    quat_rotate_vector,
    quat_from_axis_angle,
    quat_from_two_vectors,
    quat_slerp,
    quat_to_euler,
    euler_to_quat,
)

# Core solvers
from .ik_solver import (
    solve_two_bone_ik,
    solve_leg_ik,
    solve_arm_ik,
    compute_knee_pole_position,
    compute_elbow_pole_position,
    compute_foot_ground_target,
    apply_ik_to_pose,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Chain definitions
    "LEG_IK",
    "ARM_IK",
    "SPINE_CHAIN",
    "BODY_DIMENSIONS",
    "IK_TOLERANCES",
    "get_chain",
    "get_chain_type",
    "get_chain_side",
    "get_max_reach",

    # Math utilities
    "normalize",
    "safe_normalize",
    "dot",
    "cross",
    "lerp",
    "project_onto_plane",
    "quat_identity",
    "quat_normalize",
    "quat_multiply",
    "quat_conjugate",
    "quat_inverse",
    "quat_rotate_vector",
    "quat_from_axis_angle",
    "quat_from_two_vectors",
    "quat_slerp",
    "quat_to_euler",
    "euler_to_quat",

    # Core solvers
    "solve_two_bone_ik",
    "solve_leg_ik",
    "solve_arm_ik",
    "compute_knee_pole_position",
    "compute_elbow_pole_position",
    "compute_foot_ground_target",
    "apply_ik_to_pose",
]
