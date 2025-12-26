# Exp_Game/engine/animations/__init__.py
"""
Animation Engine - Worker-safe animation computation.

This module contains NO bpy references and can be used in worker processes.
All classes and functions here are pickle-safe for multiprocessing.

NUMPY OPTIMIZATION (2025-12):
  - All animation data stored as numpy arrays
  - Vectorized blending (30-100x faster than Python loops)
  - Static bone detection (skip bones that don't animate)
  - Processes ALL bones in single vectorized operations

Core Components:
- BakedAnimation: Pre-baked animation data (numpy arrays)
- AnimationCache: Storage and lookup for baked animations
- baker: Functions to bake Blender Actions to BakedAnimation
- blend: Interpolation and blending math (vectorized numpy)

Transform Format:
  10-float array: [quat_w, quat_x, quat_y, quat_z, loc_x, loc_y, loc_z, scale_x, scale_y, scale_z]

Array Shapes:
  - Single pose: (num_bones, 10)
  - Animation frames: (num_frames, num_bones, 10)

Main Thread Components (in Exp_Game/animations/):
- AnimationController: Main thread orchestrator (uses bpy)
- apply: Utility functions for applying transforms via bpy
"""

from .data import BakedAnimation, Transform, IDENTITY_TRANSFORM
from .cache import AnimationCache
from .blend import (
    # Vectorized numpy functions
    normalize_quaternions,
    slerp_vectorized,
    blend_transforms,
    sample_bone_animation,
    blend_bone_poses,
    sample_object_animation,
    blend_object_transforms,
    IDENTITY,
)
from .ik import (
    # Two-bone IK solver
    solve_two_bone_ik,
    solve_leg_ik,
    solve_arm_ik,
    apply_ik_to_pose,
    # Pole/target helpers
    compute_foot_ground_target,
    compute_knee_pole_position,
    compute_elbow_pole_position,
    # Chain definitions
    LEG_IK,
    ARM_IK,
)
from .joint_limits import (
    # Joint limit enforcement
    clamp_rotation,
    apply_limits_to_pose,
    is_within_limits,
    get_limit_info,
    # Worker cache
    set_worker_limits,
    get_worker_limits,
    has_worker_limits,
    # File I/O
    save_limits_to_file,
    load_limits_from_file,
)
from .default_limits import (
    # Default rig joint limits
    DEFAULT_JOINT_LIMITS,
    get_default_limits,
    get_bone_limit,
)

__all__ = [
    # Data
    "BakedAnimation",
    "Transform",
    "IDENTITY_TRANSFORM",

    # Cache
    "AnimationCache",

    # Numpy blending (vectorized, high performance)
    "normalize_quaternions",
    "slerp_vectorized",
    "blend_transforms",
    "sample_bone_animation",
    "blend_bone_poses",
    "sample_object_animation",
    "blend_object_transforms",
    "IDENTITY",

    # IK solver
    "solve_two_bone_ik",
    "solve_leg_ik",
    "solve_arm_ik",
    "apply_ik_to_pose",
    "compute_foot_ground_target",
    "compute_knee_pole_position",
    "compute_elbow_pole_position",
    "LEG_IK",
    "ARM_IK",

    # Joint limits
    "clamp_rotation",
    "apply_limits_to_pose",
    "is_within_limits",
    "get_limit_info",
    "set_worker_limits",
    "get_worker_limits",
    "has_worker_limits",
    "save_limits_to_file",
    "load_limits_from_file",
    "DEFAULT_JOINT_LIMITS",
    "get_default_limits",
    "get_bone_limit",
]


def register():
    """No operators to register in worker-safe module."""
    pass


def unregister():
    """No operators to unregister in worker-safe module."""
    pass
