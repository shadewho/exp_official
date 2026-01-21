# Exp_Game/animations/controller.py
"""
AnimationController - Main thread animation orchestrator.

MAIN THREAD ARCHITECTURE (2025-01 optimization):
- All animation work happens on main thread (no IPC overhead)
- Sampling and blending use vectorized numpy (fast)
- Direct apply to Blender pose bones

Usage:
    controller = AnimationController()
    controller.play("Player", "Walk")

    # Each frame:
    controller.update_state(delta_time)      # Update times/fades
    controller.compute_and_apply_local()     # Sample, blend, apply (all local)
"""

import bpy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Import worker-safe modules from engine (also work on main thread)
from ..engine.animations.data import BakedAnimation
from ..engine.animations.cache import AnimationCache
from ..engine.animations.blend import (
    sample_bone_animation,
    blend_bone_poses,
    sample_object_animation,
    blend_object_transforms,
)
from .bone_groups import BONE_INDEX, INDEX_TO_BONE, TOTAL_BONES


import numpy as np


def _has_significant_delta(prev: tuple, new: tuple, eps: float = 1e-5) -> bool:
    """Return True if any component differs more than eps."""
    if prev is None or new is None:
        return True
    if len(prev) != len(new):
        return True
    for a, b in zip(prev, new):
        if abs(a - b) > eps:
            return True
    return False


# Pre-computed identity pose for standard rig (TOTAL_BONES x 10)
# quat=(1,0,0,0), loc=(0,0,0), scale=(1,1,1)
_IDENTITY_POSE = np.zeros((TOTAL_BONES, 10), dtype=np.float32)
_IDENTITY_POSE[:, 0] = 1.0   # quat w = 1
_IDENTITY_POSE[:, 7:10] = 1.0  # scale = 1


@dataclass
class PlayingAnimation:
    """
    A single animation instance playing on an object.

    Tracks playback time, weight, and fade state.
    """
    animation_name: str
    time: float = 0.0
    weight: float = 1.0
    speed: float = 1.0
    looping: bool = True

    # Fade state
    fade_in: float = 0.0      # Seconds to fade in (0 = instant)
    fade_out: float = 0.0     # Seconds to fade out (0 = instant)
    fade_progress: float = 1.0  # Current fade (0 = invisible, 1 = full weight)
    fading_out: bool = False

    # Lifetime
    play_count: int = 0       # Times looped (for non-looping, stops at 1)
    finished: bool = False

    # PERFORMANCE: Cache effective_weight to avoid recomputing weight * fade_progress
    # Updated once per frame in update_state(), used in get_active_weights() and compute_and_apply_local()
    effective_weight: float = 1.0


@dataclass
class ObjectAnimState:
    """
    Animation state for a single object.

    Tracks all playing animations and handles blending.
    """
    object_name: str
    playing: List[PlayingAnimation] = field(default_factory=list)

    def get_active_weights(self) -> List[Tuple[str, float]]:
        """Get list of (anim_name, effective_weight) for active animations."""
        result = []
        for p in self.playing:
            if not p.finished:
                # PERFORMANCE: Use cached effective_weight instead of recomputing
                if p.effective_weight > 0.001:
                    result.append((p.animation_name, p.effective_weight))
        return result


class AnimationController:
    """
    Main animation controller for the game.

    Manages animation playback for all objects, handles blending,
    and applies final poses via bpy.

    Architecture:
    - Uses AnimationCache from engine/animations for storage
    - Calls blend.py functions for sampling/blending (worker-safe math)
    - Only this module touches bpy (pose.bones, object transforms)
    """

    def __init__(self):
        # Animation storage (worker-safe)
        self.cache = AnimationCache()

        # Per-object animation state
        self._states: Dict[str, ObjectAnimState] = {}

        # Global time scale (1.0 = normal, 0.5 = half speed)
        self.time_scale: float = 1.0

        # Last applied transforms to skip redundant bpy writes
        self._last_bone_transforms: Dict[str, Dict[str, tuple]] = {}
        self._last_object_transforms: Dict[str, tuple] = {}

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def add_animation(self, animation: BakedAnimation) -> None:
        """Add a baked animation to the cache."""
        self.cache.add(animation)

    def add_animations(self, animations: List[BakedAnimation]) -> int:
        """Add multiple animations. Returns count added."""
        return self.cache.add_many(animations)

    def get_animation(self, name: str) -> Optional[BakedAnimation]:
        """Get animation by name."""
        return self.cache.get(name)

    def has_animation(self, name: str) -> bool:
        """Check if animation exists."""
        return self.cache.has(name)

    # =========================================================================
    # PLAYBACK CONTROL
    # =========================================================================

    def play(
        self,
        object_name: str,
        animation_name: str,
        weight: float = 1.0,
        speed: float = 1.0,
        looping: bool = True,
        fade_in: float = 0.0,
        fade_out: float = 0.0,
        replace: bool = False
    ) -> bool:
        """
        Start playing an animation on an object.

        Args:
            object_name: Name of the Blender object
            animation_name: Name of the animation to play
            weight: Blend weight (0-1)
            speed: Playback speed multiplier
            looping: Whether to loop
            fade_in: Fade in duration in seconds
            fade_out: Fade out duration when stopping
            replace: If True, stop all other animations on this object

        Returns:
            True if animation started, False if animation not found
        """
        if not self.cache.has(animation_name):
            return False

        # Get or create object state
        if object_name not in self._states:
            self._states[object_name] = ObjectAnimState(object_name=object_name)

        state = self._states[object_name]

        # Replace mode: crossfade - fade out current animations over same duration as fade_in
        if replace:
            for p in state.playing:
                if not p.finished and not p.fading_out:
                    # Use fade_in duration for fade_out to create smooth crossfade
                    p.fade_out = fade_in
                    p.fading_out = True

        # Check if already playing this animation
        for p in state.playing:
            if p.animation_name == animation_name and not p.finished:
                # Reset to start
                p.time = 0.0
                p.play_count = 0
                p.finished = False
                p.fading_out = False
                p.fade_progress = 0.0 if fade_in > 0 else 1.0
                return True

        # Create new playing animation
        playing = PlayingAnimation(
            animation_name=animation_name,
            weight=weight,
            speed=speed,
            looping=looping,
            fade_in=fade_in,
            fade_out=fade_out,
            fade_progress=0.0 if fade_in > 0 else 1.0
        )
        state.playing.append(playing)

        # Bind in cache
        self.cache.bind(object_name, animation_name)

        return True

    def stop(
        self,
        object_name: str,
        animation_name: Optional[str] = None,
        fade_out: Optional[float] = None
    ) -> None:
        """
        Stop animation(s) on an object.

        Args:
            object_name: Name of the object
            animation_name: Specific animation to stop, or None for all
            fade_out: Override fade out duration
        """
        if object_name not in self._states:
            return

        state = self._states[object_name]

        for p in state.playing:
            if animation_name is None or p.animation_name == animation_name:
                if fade_out is not None:
                    p.fade_out = fade_out

                if p.fade_out > 0:
                    p.fading_out = True
                else:
                    p.finished = True

    def is_playing(self, object_name: str, animation_name: Optional[str] = None) -> bool:
        """Check if object has any (or specific) animation playing."""
        if object_name not in self._states:
            return False

        state = self._states[object_name]

        for p in state.playing:
            if not p.finished:
                if animation_name is None or p.animation_name == animation_name:
                    return True

        return False

    def _cleanup(self) -> None:
        """Remove finished animations from states."""
        for state in self._states.values():
            state.playing = [p for p in state.playing if not p.finished]

    # =========================================================================
    # STATE QUERIES
    # =========================================================================

    def get_playing(self, object_name: str) -> List[str]:
        """Get names of all animations playing on an object."""
        if object_name not in self._states:
            return []

        return [
            p.animation_name
            for p in self._states[object_name].playing
            if not p.finished
        ]

    def get_state(self, object_name: str) -> Optional[ObjectAnimState]:
        """Get full animation state for an object."""
        return self._states.get(object_name)

    def clear(self) -> None:
        """Clear all animation states (not the cache)."""
        self._states.clear()
        self._last_bone_transforms.clear()
        self._last_object_transforms.clear()
        # Clear pose_bone lookup caches
        for attr in list(vars(self).keys()):
            if attr.startswith('_pb_cache_'):
                delattr(self, attr)

    def clear_all(self) -> None:
        """Clear both states and cache."""
        self._states.clear()
        self.cache.clear()
        self._last_bone_transforms.clear()
        self._last_object_transforms.clear()
        # Clear pose_bone lookup caches
        for attr in list(vars(self).keys()):
            if attr.startswith('_pb_cache_'):
                delattr(self, attr)

    # =========================================================================
    # FRAME UPDATE & COMPUTE METHODS
    # =========================================================================

    def update_state(self, delta_time: float) -> None:
        """
        Update playback state only (times, fades). No pose computation.
        Call this before compute_and_apply_local().

        Args:
            delta_time: Time since last frame in seconds
        """
        dt = delta_time * self.time_scale

        for object_name, state in self._states.items():
            for p in state.playing:
                if p.finished:
                    continue

                anim = self.cache.get(p.animation_name)
                if anim is None:
                    p.finished = True
                    continue

                # Update time
                p.time += dt * p.speed

                # Handle looping
                if p.time >= anim.duration:
                    if p.looping:
                        p.time = p.time % anim.duration
                        p.play_count += 1
                    else:
                        p.time = anim.duration
                        p.play_count = 1
                        p.finished = True

                # Update fade
                if p.fading_out:
                    if p.fade_out > 0:
                        p.fade_progress -= dt / p.fade_out
                        if p.fade_progress <= 0:
                            p.fade_progress = 0
                            p.finished = True
                    else:
                        p.finished = True
                elif p.fade_progress < 1.0 and p.fade_in > 0:
                    p.fade_progress += dt / p.fade_in
                    if p.fade_progress > 1.0:
                        p.fade_progress = 1.0

                # PERFORMANCE: Cache effective_weight once per frame
                # Saves recomputing weight * fade_progress in get_active_weights() and compute_and_apply_local()
                p.effective_weight = p.weight * p.fade_progress

        # Cleanup finished animations
        self._cleanup()

    def compute_and_apply_local(self) -> int:
        """
        Compute and apply all animation poses locally on main thread.
        Call after update_state(). No worker involvement - pure local compute.

        OPTIMIZED: Uses cached bone mappings and numpy arrays throughout.
        - No dict/tuple conversions in hot path
        - Cached standard bone index mappings for fast remapping
        - Direct numpy array application

        Returns:
            Total number of transforms applied
        """
        total_applied = 0

        for object_name, state in self._states.items():
            # Collect active animations for this object
            # Store poses in STANDARD bone order (TOTAL_BONES, 10)
            std_poses = []
            bone_weights = []
            object_transforms_list = []
            object_weights = []

            for p in state.playing:
                if p.finished or p.effective_weight < 0.001:
                    continue

                # Get the BakedAnimation from cache
                anim = self.cache.get(p.animation_name)
                if anim is None:
                    continue

                # Sample bone animation
                if anim.has_bones:
                    # Ensure standard bone mapping is computed (once per animation)
                    if not anim._std_mapping_ready:
                        anim.compute_std_bone_mapping(BONE_INDEX, TOTAL_BONES)

                    # Build anim_data dict for sampling functions
                    anim_data = {
                        "bone_transforms": anim.bone_transforms,
                        "object_transforms": anim.object_transforms,
                        "duration": anim.duration,
                        "fps": anim.fps,
                        "animated_mask": anim.animated_mask,
                    }

                    pose, _ = sample_bone_animation(anim_data, p.time, p.looping)
                    if pose.size > 0:
                        # FAST: Remap to standard bone order using cached mapping
                        std_pose = anim.remap_to_standard(pose, _IDENTITY_POSE)
                        std_poses.append(std_pose)
                        bone_weights.append(p.effective_weight)

                # Sample object animation
                if anim.has_object:
                    anim_data = {
                        "bone_transforms": anim.bone_transforms,
                        "object_transforms": anim.object_transforms,
                        "duration": anim.duration,
                        "fps": anim.fps,
                        "animated_mask": anim.animated_mask,
                    }
                    obj_transform = sample_object_animation(anim_data, p.time, p.looping)
                    if obj_transform is not None:
                        object_transforms_list.append(obj_transform)
                        object_weights.append(p.effective_weight)

            # Blend bone poses if we have any (all in standard order now)
            final_pose = None
            if std_poses:
                final_pose = blend_bone_poses(std_poses, bone_weights)

            # Blend object transforms if we have any
            final_obj_transform = None
            if object_transforms_list:
                final_obj_transform = blend_object_transforms(object_transforms_list, object_weights)

            # FAST: Apply using numpy arrays directly (no dict/tuple conversion)
            if final_pose is not None or final_obj_transform is not None:
                count = self.apply_pose_fast(object_name, final_pose, final_obj_transform)
                total_applied += count

        return total_applied

    def get_compute_job_data(self) -> Dict[str, dict]:
        """
        Get job data for worker-based animation compute.

        NOTE: Main game uses compute_and_apply_local() instead. This method
        is kept for the developer test panel (animations/test_panel.py) which
        optionally uses workers for testing.

        Returns:
            Dict[object_name, job_data] where job_data is:
            {
                "object_name": str,
                "playing": [
                    {"anim_name": str, "time": float, "weight": float, "looping": bool},
                    ...
                ]
            }
        """
        result = {}

        for object_name, state in self._states.items():
            playing_data = []

            for p in state.playing:
                if p.finished:
                    continue

                # PERFORMANCE: Use cached effective_weight instead of recomputing
                if p.effective_weight < 0.001:
                    continue

                playing_data.append({
                    "anim_name": p.animation_name,
                    "time": p.time,
                    "weight": p.effective_weight,
                    "looping": p.looping
                })

            if playing_data:
                result[object_name] = {
                    "object_name": object_name,
                    "playing": playing_data
                }

        return result

    def apply_worker_result(
        self,
        object_name: str,
        bone_transforms: Dict[str, tuple],
        object_transform: tuple = None
    ) -> int:
        """
        Apply computed transforms to Blender objects.
        Used by compute_and_apply_local() internally.

        Supports both:
        - Armatures: applies bone_transforms to pose bones
        - Objects: applies object_transform to object location/rotation/scale

        Args:
            object_name: Name of the Blender object
            bone_transforms: Dict[bone_name, Transform] from worker result (for armatures)
            object_transform: (10-float tuple) for object-level transforms (mesh, empty, etc.)

        Returns:
            Number of transforms applied (bones + 1 if object transform)
        """
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            return 0

        count = 0
        bone_cache = self._last_bone_transforms.setdefault(object_name, {})

        # Apply bone transforms (for armatures)
        if obj.type == 'ARMATURE' and bone_transforms:
            pose_bones = obj.pose.bones

            for bone_name, transform in bone_transforms.items():
                pose_bone = pose_bones.get(bone_name)
                if pose_bone is None:
                    continue

                prev_transform = bone_cache.get(bone_name)
                if prev_transform is not None and not _has_significant_delta(prev_transform, transform):
                    # Skip redundant write
                    continue

                # Transform: (qw, qx, qy, qz, lx, ly, lz, sx, sy, sz)
                qw, qx, qy, qz = transform[0:4]
                lx, ly, lz = transform[4:7]
                sx, sy, sz = transform[7:10]

                pose_bone.rotation_quaternion = (qw, qx, qy, qz)
                pose_bone.location = (lx, ly, lz)
                pose_bone.scale = (sx, sy, sz)
                bone_cache[bone_name] = transform
                count += 1

        # Apply object-level transform (for mesh, empty, etc. OR armature root motion)
        if object_transform is not None:
            prev_object = self._last_object_transforms.get(object_name)
            if prev_object is None or _has_significant_delta(prev_object, object_transform):
                # Transform: (qw, qx, qy, qz, lx, ly, lz, sx, sy, sz)
                qw, qx, qy, qz = object_transform[0:4]
                lx, ly, lz = object_transform[4:7]
                sx, sy, sz = object_transform[7:10]

                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = (qw, qx, qy, qz)
                obj.location = (lx, ly, lz)
                obj.scale = (sx, sy, sz)
                self._last_object_transforms[object_name] = object_transform
                count += 1

        return count

    def apply_pose_fast(
        self,
        object_name: str,
        pose: np.ndarray,
        object_transform: np.ndarray = None
    ) -> int:
        """
        FAST: Apply pose directly from numpy array using indexed access.
        Avoids dict/tuple conversions. Uses standard BONE_INDEX order.

        Args:
            object_name: Name of the Blender armature
            pose: (TOTAL_BONES, 10) numpy array in standard BONE_INDEX order, or None
            object_transform: Optional (10,) numpy array for object transform

        Returns:
            Number of bones updated
        """
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            return 0

        count = 0

        # Apply bone transforms (for armatures)
        if obj.type == 'ARMATURE' and pose is not None:
            pose_bones = obj.pose.bones
            bone_cache = self._last_bone_transforms.setdefault(object_name, {})

            # Get or build pose_bone lookup cache for this armature
            pb_cache_key = f"_pb_cache_{object_name}"
            if not hasattr(self, pb_cache_key):
                # Build pose_bone lookup once per armature
                pb_lookup = {}
                for bone_idx in range(TOTAL_BONES):
                    bone_name = INDEX_TO_BONE.get(bone_idx)
                    if bone_name:
                        pb = pose_bones.get(bone_name)
                        if pb:
                            pb_lookup[bone_idx] = pb
                setattr(self, pb_cache_key, pb_lookup)

            pb_lookup = getattr(self, pb_cache_key)

            # Apply bones using cached pose_bone lookup
            for bone_idx, pose_bone in pb_lookup.items():
                # Get transform from numpy array
                transform = pose[bone_idx]

                # Skip if unchanged (vectorized comparison)
                bone_name = INDEX_TO_BONE[bone_idx]
                prev = bone_cache.get(bone_name)
                if prev is not None:
                    if np.allclose(prev, transform, atol=1e-5):
                        continue

                # Apply transform (unavoidable individual bpy writes)
                pose_bone.rotation_quaternion = (
                    float(transform[0]), float(transform[1]),
                    float(transform[2]), float(transform[3])
                )
                pose_bone.location = (
                    float(transform[4]), float(transform[5]), float(transform[6])
                )
                pose_bone.scale = (
                    float(transform[7]), float(transform[8]), float(transform[9])
                )

                # Cache as numpy array
                bone_cache[bone_name] = transform.copy()
                count += 1

        # Apply object transform if provided
        if object_transform is not None:
            prev_obj = self._last_object_transforms.get(object_name)
            if prev_obj is None or not np.allclose(prev_obj, object_transform, atol=1e-5):
                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = (
                    float(object_transform[0]), float(object_transform[1]),
                    float(object_transform[2]), float(object_transform[3])
                )
                obj.location = (
                    float(object_transform[4]), float(object_transform[5]),
                    float(object_transform[6])
                )
                obj.scale = (
                    float(object_transform[7]), float(object_transform[8]),
                    float(object_transform[9])
                )
                self._last_object_transforms[object_name] = object_transform.copy()
                count += 1

        return count

    def get_cache_data_for_workers(self) -> dict:
        """
        Get animation cache data formatted for CACHE_ANIMATIONS job.

        NOTE: Main game uses compute_and_apply_local() instead - no worker caching.
        This method is kept for the developer test panel (animations/test_panel.py)
        which optionally uses workers for testing.

        Returns:
            Dict ready to send as CACHE_ANIMATIONS job data:
            {"animations": {anim_name: {bone_transforms, bone_names, duration, fps, ...}, ...}}
        """
        animations_dict = {}

        for name, anim in self.cache._animations.items():
            # Use to_dict() which properly serializes numpy arrays
            animations_dict[name] = anim.to_dict()

        return {"animations": animations_dict}

    def has_active_animations(self) -> bool:
        """Check if any object has active (non-finished) animations."""
        for state in self._states.values():
            for p in state.playing:
                if not p.finished:
                    return True
        return False
