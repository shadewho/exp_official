# Exp_Game/engine/animations/__init__.py
"""
Animation Engine - Worker-safe animation computation.

This module contains NO bpy references and can be used in worker processes.
All classes and functions here are pickle-safe for multiprocessing.

Core Components:
- BakedAnimation: Pre-baked animation data (bones + object transforms)
- AnimationCache: Storage and lookup for baked animations
- baker: Functions to bake Blender Actions to BakedAnimation
- blend: Interpolation and blending math (slerp, lerp)

Transform Format:
  10-float tuple: (quat_w, quat_x, quat_y, quat_z, loc_x, loc_y, loc_z, scale_x, scale_y, scale_z)

Main Thread Components (in Exp_Game/animations/):
- AnimationController: Main thread orchestrator (uses bpy)
- apply: Utility functions for applying transforms via bpy
"""

from .data import BakedAnimation, Transform
from .cache import AnimationCache
from .blend import (
    lerp,
    slerp,
    blend_transform,
    interpolate_transform,
    sample_animation,
    blend_bone_poses,
    blend_object_transforms,
)

__all__ = [
    # Data
    "BakedAnimation",
    "Transform",

    # Cache
    "AnimationCache",

    # Blending
    "lerp",
    "slerp",
    "blend_transform",
    "interpolate_transform",
    "sample_animation",
    "blend_bone_poses",
    "blend_object_transforms",
]


def register():
    """No operators to register in worker-safe module."""
    pass


def unregister():
    """No operators to unregister in worker-safe module."""
    pass
