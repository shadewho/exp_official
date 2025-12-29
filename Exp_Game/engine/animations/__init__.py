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

NOTE: Joint limits removed - not needed for rigid animations.
NOTE: IK code removed - using neural network approach instead.
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
]


def register():
    """No operators to register in worker-safe module."""
    pass


def unregister():
    """No operators to unregister in worker-safe module."""
    pass
