# Exp_Game/animations/controller.py
"""
AnimationController - Main thread animation orchestrator.

WORKER-ONLY ARCHITECTURE:
- Main thread: Manages playback state (times, weights, fades)
- Worker: Computes blended poses (sampling + blending math)
- Main thread: Applies final pose via bpy

Usage:
    controller = AnimationController()
    controller.play("Player", "Walk")

    # Each frame:
    controller.update_state(delta_time)          # Update times/fades
    job_data = controller.get_compute_job_data() # Get data for worker
    # ... submit job to engine ...
    controller.apply_worker_result(result)       # Apply computed pose
"""

import bpy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Import worker-safe modules from engine
from ..engine.animations.data import BakedAnimation, Transform
from ..engine.animations.cache import AnimationCache


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
                effective = p.weight * p.fade_progress
                if effective > 0.001:
                    result.append((p.animation_name, effective))
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

        # Replace mode: fade out all current animations
        if replace:
            for p in state.playing:
                if not p.finished and not p.fading_out:
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

    def clear_all(self) -> None:
        """Clear both states and cache."""
        self._states.clear()
        self.cache.clear()

    # =========================================================================
    # WORKER-OFFLOADED METHODS
    # =========================================================================

    def update_state(self, delta_time: float) -> None:
        """
        Update playback state only (times, fades). No pose computation.
        Call this before get_compute_job_data() in the worker flow.

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

        # Cleanup finished animations
        self._cleanup()

    def get_compute_job_data(self) -> Dict[str, dict]:
        """
        Get job data for all objects with active animations.
        Call after update_state() to get data for ANIMATION_COMPUTE jobs.

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

                effective_weight = p.weight * p.fade_progress
                if effective_weight < 0.001:
                    continue

                playing_data.append({
                    "anim_name": p.animation_name,
                    "time": p.time,
                    "weight": effective_weight,
                    "looping": p.looping
                })

            if playing_data:
                result[object_name] = {
                    "object_name": object_name,
                    "playing": playing_data
                }

        return result

    def apply_worker_result(self, object_name: str, bone_transforms: Dict[str, tuple]) -> int:
        """
        Apply bone transforms computed by worker.
        Call after receiving ANIMATION_COMPUTE result.

        Args:
            object_name: Name of the Blender object
            bone_transforms: Dict[bone_name, Transform] from worker result

        Returns:
            Number of bones updated
        """
        if not bone_transforms:
            return 0

        obj = bpy.data.objects.get(object_name)
        if obj is None or obj.type != 'ARMATURE':
            return 0

        pose_bones = obj.pose.bones
        count = 0

        for bone_name, transform in bone_transforms.items():
            pose_bone = pose_bones.get(bone_name)
            if pose_bone is None:
                continue

            # Transform: (qw, qx, qy, qz, lx, ly, lz, sx, sy, sz)
            qw, qx, qy, qz = transform[0:4]
            lx, ly, lz = transform[4:7]
            sx, sy, sz = transform[7:10]

            pose_bone.rotation_quaternion = (qw, qx, qy, qz)
            pose_bone.location = (lx, ly, lz)
            pose_bone.scale = (sx, sy, sz)
            count += 1

        return count

    def get_cache_data_for_workers(self) -> dict:
        """
        Get animation cache data formatted for CACHE_ANIMATIONS job.
        Call at game start to send animations to workers.

        Returns:
            Dict ready to send as CACHE_ANIMATIONS job data:
            {"animations": {anim_name: {bones: {...}, duration, fps}, ...}}
        """
        animations_dict = {}

        for name, anim in self.cache._animations.items():
            # Convert BakedAnimation to plain dict for worker
            animations_dict[name] = {
                "name": anim.name,
                "duration": anim.duration,
                "fps": anim.fps,
                "frame_count": anim.frame_count,
                "bones": anim.bones,  # Already Dict[str, List[Transform]]
                "looping": anim.looping
            }

        return {"animations": animations_dict}

    def has_active_animations(self) -> bool:
        """Check if any object has active (non-finished) animations."""
        for state in self._states.values():
            for p in state.playing:
                if not p.finished:
                    return True
        return False
