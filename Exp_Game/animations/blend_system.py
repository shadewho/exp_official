# Exp_Game/animations/blend_system.py
"""
Animation Blend System - Layered animation with masks.

ARCHITECTURE:
- Layers stack on top of base locomotion
- Each layer has a mask (which bones it affects) and weight
- Designed to integrate with Exp_Nodes reaction system

LAYER TYPES:
- BASE: Full body locomotion (walk, run, idle) - only one active
- ADDITIVE: Adds to base pose (lean, reach, breathe) - can stack
- OVERRIDE: Replaces bones completely (reactions, hit) - can stack

INTEGRATION:
- Called from reaction executors via BlendSystemAPI
- Nodes create ReactionDefinitions that trigger layer changes
- State persists on the modal operator

Usage:
    # From a reaction executor:
    blend_system = get_blend_system()
    blend_system.play_additive("reach_forward", mask="UPPER_BODY", weight=1.0, duration=0.5)

    # From game loop (each frame):
    final_pose = blend_system.evaluate(delta_time)
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from .bone_groups import (
    BONE_INDEX, INDEX_TO_BONE, TOTAL_BONES, BlendMasks,
    create_blend_mask, get_bone_indices
)
from ..developer.dev_logger import log_game

# Import vectorized blend functions for performance
from ..engine.animations.blend import (
    normalize_quaternions,
    slerp_vectorized,
    blend_transforms,
    IDENTITY,
)

# =============================================================================
# PERFORMANCE CONSTANTS (pre-computed, avoid allocations in loops)
# =============================================================================

# Identity quaternion [w, x, y, z] - used for additive blending
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Pre-computed identity pose (53 bones x 10 floats)
# quat=(1,0,0,0), loc=(0,0,0), scale=(1,1,1)
_IDENTITY_POSE = np.zeros((TOTAL_BONES, 10), dtype=np.float32)
_IDENTITY_POSE[:, 0] = 1.0   # quat w = 1
_IDENTITY_POSE[:, 7:10] = 1.0  # scale = 1

# Minimum weight threshold for processing
WEIGHT_THRESHOLD = 0.001


# =============================================================================
# LAYER TYPES
# =============================================================================

class LayerType(Enum):
    BASE = "base"           # Full body, one active at a time
    ADDITIVE = "additive"   # Adds rotation/position to base
    OVERRIDE = "override"   # Replaces bones completely


# =============================================================================
# ANIMATION LAYER
# =============================================================================

@dataclass
class AnimationLayer:
    """
    A single animation layer in the blend stack.
    """
    name: str
    layer_type: LayerType

    # Animation source (name in cache or callable)
    animation_name: Optional[str] = None
    pose_provider: Optional[Callable[[], np.ndarray]] = None

    # Blend settings
    mask_name: str = "ALL"              # Bone group name
    mask_weights: np.ndarray = None     # Computed mask array
    weight: float = 1.0                 # Current blend weight
    target_weight: float = 1.0          # Target weight (for fading)

    # Timing
    time: float = 0.0                   # Current playback time
    speed: float = 1.0                  # Playback speed
    duration: float = -1.0              # Total duration (-1 = from animation)
    looping: bool = False               # Loop animation
    cached_anim_duration: float = 1.0   # Cached animation duration (set once)

    # Fade
    fade_in: float = 0.0                # Fade in duration
    fade_out: float = 0.0               # Fade out duration
    fade_time: float = 0.0              # Current fade progress
    fading_out: bool = False            # Currently fading out

    # State
    active: bool = True                 # Layer is active
    finished: bool = False              # Layer completed (remove it)
    priority: int = 0                   # Higher = evaluated later (wins conflicts)

    # Timestamps
    start_time: float = 0.0             # When layer started

    # Pre-computed mask indices for vectorized operations
    mask_indices: np.ndarray = None     # Indices where mask > threshold

    def __post_init__(self):
        if self.mask_weights is None:
            self.mask_weights = create_blend_mask(self.mask_name, 1.0)
        # Pre-compute indices for vectorized mask application
        if self.mask_indices is None:
            self.mask_indices = np.where(self.mask_weights > WEIGHT_THRESHOLD)[0]


# =============================================================================
# BLEND SYSTEM
# =============================================================================

class BlendSystem:
    """
    Main animation blend system.

    Manages layers and evaluates blend stack.
    """

    def __init__(self, animation_cache=None):
        """
        Initialize blend system.

        Args:
            animation_cache: AnimationCache instance for looking up animations
        """
        self.cache = animation_cache

        # Layer stacks
        self._base_layer: Optional[AnimationLayer] = None
        self._additive_layers: List[AnimationLayer] = []
        self._override_layers: List[AnimationLayer] = []

        # Reference pose (T-pose for additive calculations)
        self._reference_pose: Optional[np.ndarray] = None

        # Last evaluated pose (for debugging/visualization)
        self._last_pose: Optional[np.ndarray] = None

        # Callbacks for when layers finish
        self._on_layer_finished: List[Callable[[str], None]] = []

        # Layer counter for unique naming (faster than perf_counter)
        self._layer_counter: int = 0

    # =========================================================================
    # LAYER MANAGEMENT
    # =========================================================================

    def set_base_layer(
        self,
        animation_name: str,
        speed: float = 1.0,
        fade_time: float = 0.15,
        looping: bool = True
    ) -> bool:
        """
        Set the base locomotion layer.

        Args:
            animation_name: Name of animation in cache
            speed: Playback speed
            fade_time: Crossfade duration
            looping: Whether to loop

        Returns:
            True if successful
        """
        # Fade out old base if exists
        if self._base_layer and not self._base_layer.finished:
            self._base_layer.fading_out = True
            self._base_layer.fade_out = fade_time
            # Move to override temporarily for crossfade
            self._override_layers.append(self._base_layer)

        # Create new base layer
        self._base_layer = AnimationLayer(
            name=f"base_{animation_name}",
            layer_type=LayerType.BASE,
            animation_name=animation_name,
            mask_name="ALL",
            weight=0.0 if fade_time > 0 else 1.0,
            target_weight=1.0,
            speed=speed,
            looping=looping,
            fade_in=fade_time,
            start_time=time.perf_counter()
        )

        log_game("ANIMATIONS", f"BASE_LAYER set={animation_name} speed={speed:.2f} fade={fade_time:.2f}s")
        return True

    def play_additive(
        self,
        animation_name: str,
        mask: str = "UPPER_BODY",
        weight: float = 1.0,
        speed: float = 1.0,
        duration: float = -1.0,
        fade_in: float = 0.1,
        fade_out: float = 0.1,
        looping: bool = False,
        priority: int = 0
    ) -> str:
        """
        Play an additive animation layer.

        Args:
            animation_name: Name of animation in cache
            mask: Bone group name for masking
            weight: Blend weight
            speed: Playback speed
            duration: How long to play (-1 = full animation)
            fade_in: Fade in time
            fade_out: Fade out time
            looping: Whether to loop
            priority: Layer priority (higher = later evaluation)

        Returns:
            Layer name (for stopping later)
        """
        # Use counter for fast unique naming
        self._layer_counter += 1
        layer_name = f"additive_{animation_name}_{self._layer_counter}"

        # Cache animation duration once (avoid repeated lookups)
        anim_duration = self._get_animation_duration(animation_name)

        layer = AnimationLayer(
            name=layer_name,
            layer_type=LayerType.ADDITIVE,
            animation_name=animation_name,
            mask_name=mask,
            weight=0.0 if fade_in > 0 else weight,
            target_weight=weight,
            speed=speed,
            duration=duration,
            looping=looping,
            cached_anim_duration=anim_duration,
            fade_in=fade_in,
            fade_out=fade_out,
            priority=priority,
            start_time=time.perf_counter()
        )

        self._additive_layers.append(layer)
        self._sort_layers()

        log_game("ANIMATIONS", f"ADDITIVE_PLAY name={animation_name} mask={mask} weight={weight:.2f}")
        return layer_name

    def play_override(
        self,
        animation_name: str,
        mask: str = "ALL",
        weight: float = 1.0,
        speed: float = 1.0,
        duration: float = -1.0,
        fade_in: float = 0.15,
        fade_out: float = 0.15,
        looping: bool = False,
        priority: int = 0
    ) -> str:
        """
        Play an override animation layer (replaces bones completely).

        Use for reactions like trips, hits, etc.
        """
        # Use counter for fast unique naming
        self._layer_counter += 1
        layer_name = f"override_{animation_name}_{self._layer_counter}"

        # Cache animation duration once (avoid repeated lookups)
        anim_duration = self._get_animation_duration(animation_name)

        layer = AnimationLayer(
            name=layer_name,
            layer_type=LayerType.OVERRIDE,
            animation_name=animation_name,
            mask_name=mask,
            weight=0.0 if fade_in > 0 else weight,
            target_weight=weight,
            speed=speed,
            duration=duration,
            looping=looping,
            cached_anim_duration=anim_duration,
            fade_in=fade_in,
            fade_out=fade_out,
            priority=priority,
            start_time=time.perf_counter()
        )

        self._override_layers.append(layer)
        self._sort_layers()

        log_game("ANIMATIONS", f"OVERRIDE_PLAY name={animation_name} mask={mask} weight={weight:.2f}")
        return layer_name

    def stop_layer(self, layer_name: str, fade_out: float = 0.15) -> bool:
        """
        Stop a layer by name with optional fade out.
        """
        for layer_list in [self._additive_layers, self._override_layers]:
            for layer in layer_list:
                if layer.name == layer_name and not layer.finished:
                    layer.fading_out = True
                    layer.fade_out = fade_out
                    log_game("ANIMATIONS", f"LAYER_STOP name={layer_name} fade={fade_out:.2f}s")
                    return True
        return False

    def stop_all_additive(self, fade_out: float = 0.15) -> int:
        """Stop all additive layers."""
        count = 0
        for layer in self._additive_layers:
            if not layer.finished and not layer.fading_out:
                layer.fading_out = True
                layer.fade_out = fade_out
                count += 1
        return count

    def stop_all_override(self, fade_out: float = 0.15) -> int:
        """Stop all override layers."""
        count = 0
        for layer in self._override_layers:
            if not layer.finished and not layer.fading_out:
                layer.fading_out = True
                layer.fade_out = fade_out
                count += 1
        return count

    # =========================================================================
    # EVALUATION
    # =========================================================================

    def update(self, delta_time: float) -> None:
        """
        Update layer timings (call each frame before evaluate).

        Args:
            delta_time: Time since last frame
        """
        current_time = time.perf_counter()

        # Update base layer
        if self._base_layer:
            self._update_layer(self._base_layer, delta_time)

        # Update additive layers
        for layer in self._additive_layers:
            self._update_layer(layer, delta_time)

        # Update override layers
        for layer in self._override_layers:
            self._update_layer(layer, delta_time)

        # Remove finished layers
        self._cleanup_finished_layers()

    def _update_layer(self, layer: AnimationLayer, delta_time: float) -> None:
        """Update a single layer's timing and fade."""
        if layer.finished:
            return

        # Update playback time
        layer.time += delta_time * layer.speed

        # Use cached animation duration (set once at layer creation)
        anim_duration = layer.cached_anim_duration

        # For looping animations, wrap time at animation duration
        if layer.looping and anim_duration > 0:
            layer.time = layer.time % anim_duration

        # Check layer duration (separate from animation duration for looping)
        # layer.duration controls how long the LAYER plays before fading out
        if layer.duration > 0 and not layer.fading_out:
            # Calculate total elapsed time since layer started
            elapsed = time.perf_counter() - layer.start_time
            if elapsed >= layer.duration:
                layer.fading_out = True
                log_game("ANIMATIONS", f"LAYER_DURATION_END name={layer.animation_name} elapsed={elapsed:.2f}s")
        elif not layer.looping and layer.time >= anim_duration and not layer.fading_out:
            # Non-looping layer reached end of animation
            layer.time = anim_duration
            layer.fading_out = True

        # Update fade
        if layer.fading_out:
            if layer.fade_out > 0:
                layer.weight -= delta_time / layer.fade_out
                if layer.weight <= 0:
                    layer.weight = 0
                    layer.finished = True
            else:
                layer.finished = True
        elif layer.weight < layer.target_weight and layer.fade_in > 0:
            layer.weight += delta_time / layer.fade_in
            if layer.weight >= layer.target_weight:
                layer.weight = layer.target_weight

    def evaluate(self, base_pose: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Evaluate the blend stack and return final pose.

        PERFORMANCE: No per-frame logging - use layer events for debugging.

        Args:
            base_pose: Optional pose to use as base when no BASE layer exists.
                       If None and no BASE layer, uses identity pose.

        Returns:
            Pose array (num_bones, 10) or None if nothing to evaluate
        """
        # Fast check: anything to evaluate?
        has_base = self._base_layer is not None
        has_additive = any(l.weight > WEIGHT_THRESHOLD and not l.finished for l in self._additive_layers)
        has_override = any(l.weight > WEIGHT_THRESHOLD and not l.finished for l in self._override_layers)

        if not has_base and not has_additive and not has_override:
            return None

        # 1. Get base pose
        if has_base:
            pose = self._sample_layer(self._base_layer)
            if pose is None:
                pose = base_pose if base_pose is not None else self._create_identity_pose()
        elif base_pose is not None:
            # Use provided base pose (e.g., current armature locomotion)
            pose = base_pose.copy()
        else:
            # No base layer and no provided base - use identity
            pose = self._create_identity_pose()

        # 2. Apply additive layers (vectorized)
        for layer in self._additive_layers:
            if layer.weight > WEIGHT_THRESHOLD and not layer.finished:
                additive_pose = self._sample_layer(layer)
                if additive_pose is not None:
                    pose = self._apply_additive(pose, additive_pose, layer)

        # 3. Apply override layers (vectorized)
        for layer in self._override_layers:
            if layer.weight > WEIGHT_THRESHOLD and not layer.finished:
                override_pose = self._sample_layer(layer)
                if override_pose is not None:
                    pose = self._apply_override(pose, override_pose, layer)

        # Store final pose
        self._last_pose = pose

        return pose

    def _create_identity_pose(self) -> np.ndarray:
        """Create identity pose (T-pose): quat=(1,0,0,0), loc=(0,0,0), scale=(1,1,1)."""
        # Return a copy of pre-computed constant (faster than allocating + setting)
        return _IDENTITY_POSE.copy()

    def _sample_layer(self, layer: AnimationLayer) -> Optional[np.ndarray]:
        """
        Sample animation pose at current layer time.

        PERFORMANCE: No per-frame logging, optimized bone remapping.
        """
        if layer.pose_provider:
            return layer.pose_provider()

        if layer.animation_name and self.cache:
            anim = self.cache.get(layer.animation_name)
            if anim:
                # Sample from BakedAnimation (uses numpy internally)
                raw_pose = anim.sample(layer.time)
                if raw_pose is None or raw_pose.size == 0:
                    return None

                # Remap from BakedAnimation's bone order to standard BONE_INDEX order
                # This loop is unavoidable without caching bone index mappings per-animation
                result = _IDENTITY_POSE.copy()  # Use pre-computed constant
                for i, bone_name in enumerate(anim.bone_names):
                    std_idx = BONE_INDEX.get(bone_name, -1)
                    if std_idx >= 0 and i < len(raw_pose):
                        result[std_idx] = raw_pose[i]

                return result

        return None

    def _apply_additive(
        self,
        base_pose: np.ndarray,
        additive_pose: np.ndarray,
        layer: AnimationLayer
    ) -> np.ndarray:
        """
        Apply additive layer to base pose using VECTORIZED numpy operations.

        ~10-20x faster than per-bone Python loop.
        """
        # Use pre-computed mask indices (only process masked bones)
        indices = layer.mask_indices
        if len(indices) == 0:
            return base_pose

        result = base_pose.copy()
        weight = layer.weight

        # Get per-bone weights for masked bones only
        bone_weights = layer.mask_weights[indices] * weight  # (n_masked,)
        bone_weights = bone_weights[:, np.newaxis]  # (n_masked, 1) for broadcasting

        # Extract quaternions for masked bones
        base_quats = result[indices, 0:4]  # (n_masked, 4)
        add_quats = additive_pose[indices, 0:4]  # (n_masked, 4)

        # Additive: base + (add - identity) * weight
        # Using pre-computed IDENTITY_QUAT constant (no allocation in loop)
        quat_diff = add_quats - IDENTITY_QUAT  # (n_masked, 4)
        blended_quats = base_quats + quat_diff * bone_weights

        # Vectorized normalization
        blended_quats = normalize_quaternions(blended_quats)
        result[indices, 0:4] = blended_quats

        # Additive location offset (vectorized)
        result[indices, 4:7] += additive_pose[indices, 4:7] * bone_weights

        return result

    def _apply_override(
        self,
        base_pose: np.ndarray,
        override_pose: np.ndarray,
        layer: AnimationLayer
    ) -> np.ndarray:
        """
        Apply override layer using VECTORIZED blend_transforms.

        Uses slerp_vectorized for quaternions, lerp for loc/scale.
        ~10-20x faster than per-bone Python loop.
        """
        # Use pre-computed mask indices (only process masked bones)
        indices = layer.mask_indices
        if len(indices) == 0:
            return base_pose

        result = base_pose.copy()
        weight = layer.weight

        # For uniform weight across all masked bones, use blend_transforms directly
        # blend_transforms handles: slerp(quat) + lerp(loc) + lerp(scale)
        base_masked = base_pose[indices]  # (n_masked, 10)
        override_masked = override_pose[indices]  # (n_masked, 10)

        # Vectorized blend of all masked bones at once
        blended = blend_transforms(base_masked, override_masked, weight)
        result[indices] = blended

        return result

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _get_animation_duration(self, name: str) -> float:
        """Get duration of an animation from cache."""
        if name and self.cache:
            anim = self.cache.get(name)
            if anim:
                return anim.duration
        return 1.0

    def _sort_layers(self) -> None:
        """Sort layers by priority."""
        self._additive_layers.sort(key=lambda l: l.priority)
        self._override_layers.sort(key=lambda l: l.priority)

    def _cleanup_finished_layers(self) -> None:
        """Remove finished layers and trigger callbacks. Single-pass filter."""
        # Fast path: skip if no layers
        if not self._additive_layers and not self._override_layers:
            return

        # Single-pass filter (avoids creating intermediate list)
        finished_names = []

        # Additive - filter in place
        new_additive = []
        for layer in self._additive_layers:
            if layer.finished:
                finished_names.append(layer.name)
            else:
                new_additive.append(layer)
        self._additive_layers = new_additive

        # Override - filter in place
        new_override = []
        for layer in self._override_layers:
            if layer.finished:
                finished_names.append(layer.name)
            else:
                new_override.append(layer)
        self._override_layers = new_override

        # Log layer completions (important events, not per-frame noise)
        for name in finished_names:
            log_game("ANIMATIONS", f"LAYER_FINISHED name={name}")
            for callback in self._on_layer_finished:
                callback(name)

    def get_layer_count(self) -> Tuple[int, int, int]:
        """Get count of (base, additive, override) layers."""
        base = 1 if self._base_layer else 0
        return (base, len(self._additive_layers), len(self._override_layers))

    def clear_all(self) -> None:
        """Clear all layers."""
        self._base_layer = None
        self._additive_layers.clear()
        self._override_layers.clear()
        self._last_pose = None

    def _sample_armature_pose(self, armature) -> np.ndarray:
        """Sample current pose from armature to use as base."""
        pose = self._create_identity_pose()
        pose_bones = armature.pose.bones

        for bone_idx in range(TOTAL_BONES):
            bone_name = INDEX_TO_BONE.get(bone_idx)
            if not bone_name:
                continue

            pose_bone = pose_bones.get(bone_name)
            if not pose_bone:
                continue

            # Sample current transform
            q = pose_bone.rotation_quaternion
            pose[bone_idx, 0:4] = [q.w, q.x, q.y, q.z]
            pose[bone_idx, 4:7] = list(pose_bone.location)
            pose[bone_idx, 7:10] = list(pose_bone.scale)

        return pose

    def apply_to_armature(self, armature) -> int:
        """
        Evaluate blend stack and apply result to armature.

        Args:
            armature: bpy.types.Object (must be ARMATURE type)

        Returns:
            Number of bones updated, or 0 if nothing to apply
        """
        if not armature or armature.type != 'ARMATURE':
            return 0

        # Sample current armature pose to use as base (preserves locomotion)
        current_pose = self._sample_armature_pose(armature)

        # Evaluate to get final pose, using current pose as base
        pose = self.evaluate(base_pose=current_pose)
        if pose is None:
            return 0

        # Apply to armature bones
        pose_bones = armature.pose.bones
        count = 0

        for bone_idx in range(TOTAL_BONES):
            bone_name = INDEX_TO_BONE.get(bone_idx)
            if not bone_name:
                continue

            pose_bone = pose_bones.get(bone_name)
            if not pose_bone:
                continue

            # Extract transform: (qw, qx, qy, qz, lx, ly, lz, sx, sy, sz)
            transform = pose[bone_idx]

            pose_bone.rotation_quaternion = (
                float(transform[0]),
                float(transform[1]),
                float(transform[2]),
                float(transform[3])
            )
            pose_bone.location = (
                float(transform[4]),
                float(transform[5]),
                float(transform[6])
            )
            pose_bone.scale = (
                float(transform[7]),
                float(transform[8]),
                float(transform[9])
            )
            count += 1

        # No per-frame logging - performance critical path

        return count


# =============================================================================
# GLOBAL ACCESS (for reaction system integration)
# =============================================================================

_blend_system_instance: Optional[BlendSystem] = None


def get_blend_system() -> Optional[BlendSystem]:
    """
    Get the global blend system instance.

    Returns None if not initialized (game not running).
    """
    return _blend_system_instance


def init_blend_system(animation_cache=None) -> BlendSystem:
    """
    Initialize the global blend system.

    Call at game start.
    """
    global _blend_system_instance
    _blend_system_instance = BlendSystem(animation_cache)
    cache_count = animation_cache.count if animation_cache else 0
    cache_names = animation_cache.names if animation_cache else []
    log_game("ANIMATIONS", f"BLEND_INIT cache={animation_cache is not None} count={cache_count} names={cache_names}")
    return _blend_system_instance


def shutdown_blend_system() -> None:
    """
    Shutdown the global blend system.

    Call at game end.
    """
    global _blend_system_instance
    if _blend_system_instance:
        _blend_system_instance.clear_all()
        _blend_system_instance = None
        log_game("ANIMATIONS", "BLEND_SHUTDOWN")
