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
    # Legacy tuple interface (backwards compatibility)
    lerp,
    slerp,
    blend_transform,
    interpolate_transform,
    sample_animation,
    blend_bone_poses,
    blend_object_transforms,
    # NEW: Numpy vectorized functions (high performance)
    slerp_vectorized,
    lerp_vectorized,
    blend_transforms_vectorized,
    interpolate_frames,
    sample_animation_numpy,
    blend_poses_numpy,
    sample_and_blend_batch,
    normalize_quaternions,
    IDENTITY,
)

__all__ = [
    # Data
    "BakedAnimation",
    "Transform",
    "IDENTITY_TRANSFORM",

    # Cache
    "AnimationCache",

    # Legacy blending (tuple-based, backwards compatible)
    "lerp",
    "slerp",
    "blend_transform",
    "interpolate_transform",
    "sample_animation",
    "blend_bone_poses",
    "blend_object_transforms",

    # Numpy blending (high performance)
    "slerp_vectorized",
    "lerp_vectorized",
    "blend_transforms_vectorized",
    "interpolate_frames",
    "sample_animation_numpy",
    "blend_poses_numpy",
    "sample_and_blend_batch",
    "normalize_quaternions",
    "IDENTITY",
]


def register():
    """No operators to register in worker-safe module."""
    pass


def unregister():
    """No operators to unregister in worker-safe module."""
    pass
