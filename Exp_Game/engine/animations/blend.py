# Exp_Game/engine/animations/blend.py
"""
Animation blending math - NUMPY VECTORIZED, worker-safe, no bpy.

SINGLE SOURCE OF TRUTH for all animation math.
Used by both main thread and worker processes.

NUMPY OPTIMIZATION:
  - All operations work on entire arrays at once
  - Vectorized quaternion slerp processes ALL bones in one call
  - No Python loops for blending - pure numpy broadcasting
  - 30-100x faster than per-bone Python loops

Transform format (10 floats per bone/object):
  [quat_w, quat_x, quat_y, quat_z, loc_x, loc_y, loc_z, scale_x, scale_y, scale_z]

Array shapes:
  - Single transform: (10,)
  - Single pose: (num_bones, 10)
  - Animation frames: (num_frames, num_bones, 10) for bones
  - Animation frames: (num_frames, 10) for objects
"""

import numpy as np
from typing import Tuple, List, Dict, Optional

# Identity transform
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)


# =============================================================================
# VECTORIZED QUATERNION OPERATIONS
# =============================================================================

def normalize_quaternions(quats: np.ndarray) -> np.ndarray:
    """
    Normalize quaternions (in-place safe).

    Args:
        quats: (..., 4) array of quaternions [w, x, y, z]

    Returns:
        Normalized quaternions, same shape
    """
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    return quats / norms


def slerp_vectorized(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Vectorized spherical linear interpolation for multiple quaternions.

    Args:
        q1: (..., 4) array of quaternions [w, x, y, z]
        q2: (..., 4) array of quaternions [w, x, y, z]
        t: Interpolation factor (0 = q1, 1 = q2)

    Returns:
        Interpolated quaternions, same shape as input
    """
    # Ensure float32 for consistency
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32).copy()  # Copy because we may negate

    # Dot product for each quaternion pair
    dot = np.sum(q1 * q2, axis=-1)

    # Take shorter path: negate q2 where dot < 0
    neg_mask = dot < 0
    if np.any(neg_mask):
        neg_mask_expanded = neg_mask[..., np.newaxis]
        q2 = np.where(neg_mask_expanded, -q2, q2)
        dot = np.abs(dot)

    # Clamp dot to valid range
    dot = np.clip(dot, -1.0, 1.0)

    # For nearly identical quaternions, use linear interpolation
    linear_mask = dot > 0.9995

    # Calculate slerp coefficients
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    # Avoid division by zero for very small angles
    safe_sin = np.maximum(np.abs(sin_theta_0), 1e-10)

    s0 = np.cos(theta) - dot * sin_theta / safe_sin
    s1 = sin_theta / safe_sin

    # Expand coefficients for broadcasting with quaternion components
    s0 = s0[..., np.newaxis]
    s1 = s1[..., np.newaxis]

    # Compute slerp result
    result = s0 * q1 + s1 * q2

    # For linear cases, use simple lerp + normalize
    if np.any(linear_mask):
        linear_mask_expanded = linear_mask[..., np.newaxis]
        lerp_result = q1 + t * (q2 - q1)
        lerp_result = normalize_quaternions(lerp_result)
        result = np.where(linear_mask_expanded, lerp_result, result)

    return result


# =============================================================================
# VECTORIZED TRANSFORM OPERATIONS
# =============================================================================

def blend_transforms(t1: np.ndarray, t2: np.ndarray, weight: float) -> np.ndarray:
    """
    Blend two transform arrays. Processes ALL bones/objects at once.

    Args:
        t1: (..., 10) transforms (weight=0 returns this)
        t2: (..., 10) transforms (weight=1 returns this)
        weight: Blend factor (0-1)

    Returns:
        Blended transforms, same shape
    """
    t1 = np.asarray(t1, dtype=np.float32)
    t2 = np.asarray(t2, dtype=np.float32)

    # Split into components
    q1, loc1, scale1 = t1[..., 0:4], t1[..., 4:7], t1[..., 7:10]
    q2, loc2, scale2 = t2[..., 0:4], t2[..., 4:7], t2[..., 7:10]

    # Slerp quaternions, lerp location and scale
    q_blend = slerp_vectorized(q1, q2, weight)
    loc_blend = loc1 + (loc2 - loc1) * weight
    scale_blend = scale1 + (scale2 - scale1) * weight

    # Concatenate back together
    return np.concatenate([q_blend, loc_blend, scale_blend], axis=-1)


# =============================================================================
# BONE ANIMATION SAMPLING & BLENDING
# =============================================================================

def sample_bone_animation(
    anim_data: dict,
    anim_time: float,
    loop: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Sample bone animation at a specific time using numpy.
    Returns pose for ALL bones at once.

    Args:
        anim_data: Cached animation dict with numpy arrays
        anim_time: Current time in seconds
        loop: Whether to loop

    Returns:
        tuple: (pose_array, stats_dict)
        - pose_array: (num_bones, 10) numpy array
        - stats_dict: {"total": int, "animated": int, "static": int, "skipped": bool}
    """
    bone_transforms = anim_data.get("bone_transforms")
    if bone_transforms is None or not isinstance(bone_transforms, np.ndarray) or bone_transforms.size == 0:
        return np.empty((0, 10), dtype=np.float32), {"total": 0, "animated": 0, "static": 0, "skipped": False}

    duration = anim_data.get("duration", 0.0)
    fps = anim_data.get("fps", 30.0)
    animated_mask = anim_data.get("animated_mask")

    num_frames = bone_transforms.shape[0]
    num_bones = bone_transforms.shape[1] if len(bone_transforms.shape) > 1 else 0

    # Count animated vs static bones
    if animated_mask is not None and isinstance(animated_mask, np.ndarray):
        num_animated = int(np.sum(animated_mask))
        num_static = num_bones - num_animated
    else:
        num_animated = num_bones
        num_static = 0
        animated_mask = None

    stats = {"total": num_bones, "animated": num_animated, "static": num_static, "skipped": False}

    if duration <= 0 or num_frames <= 1:
        return bone_transforms[0].copy(), stats

    # Handle looping
    if loop:
        anim_time = anim_time % duration
    else:
        anim_time = max(0.0, min(anim_time, duration))

    frame_float = anim_time * fps
    frame_low = int(frame_float)
    frame_high = frame_low + 1
    t = frame_float - frame_low

    # Clamp indices
    frame_low = min(frame_low, num_frames - 1)
    frame_high = min(frame_high, num_frames - 1)

    # No interpolation needed
    if frame_low == frame_high or t < 0.001:
        return bone_transforms[frame_low].copy(), stats

    # STATIC BONE OPTIMIZATION
    if animated_mask is not None and num_static > 0:
        pose = bone_transforms[0].copy()
        animated_indices = np.where(animated_mask)[0]

        if len(animated_indices) > 0:
            pose[animated_indices] = blend_transforms(
                bone_transforms[frame_low, animated_indices],
                bone_transforms[frame_high, animated_indices],
                t
            )

        stats["skipped"] = True
        return pose, stats

    # No mask or all bones animated - interpolate everything
    return blend_transforms(bone_transforms[frame_low], bone_transforms[frame_high], t), stats


def blend_bone_poses(poses: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """
    Blend multiple bone poses by weight using numpy.

    Args:
        poses: List of (num_bones, 10) numpy arrays
        weights: List of weights

    Returns:
        Blended pose (num_bones, 10)
    """
    if not poses:
        return np.empty((0, 10), dtype=np.float32)

    if len(poses) == 1:
        return poses[0].copy()

    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        return poses[0].copy()

    # Iterative blending (needed for correct quaternion slerp accumulation)
    result = poses[0].copy()
    accumulated_weight = weights[0] / total_weight

    for i in range(1, len(poses)):
        w = weights[i] / total_weight
        if accumulated_weight + w > 0:
            blend_t = w / (accumulated_weight + w)
            result = blend_transforms(result, poses[i], blend_t)
            accumulated_weight += w

    return result


# =============================================================================
# OBJECT ANIMATION SAMPLING & BLENDING
# =============================================================================

def sample_object_animation(
    anim_data: dict,
    anim_time: float,
    loop: bool = True
) -> Optional[np.ndarray]:
    """
    Sample object-level animation at a specific time.
    Returns a single 10-float transform for the object itself.

    Args:
        anim_data: Cached animation dict with numpy arrays
        anim_time: Current time in seconds
        loop: Whether to loop

    Returns:
        (10,) numpy array or None if no object transforms
    """
    object_transforms = anim_data.get("object_transforms")
    if object_transforms is None or not isinstance(object_transforms, np.ndarray) or object_transforms.size == 0:
        return None

    duration = anim_data.get("duration", 0.0)
    fps = anim_data.get("fps", 30.0)
    num_frames = object_transforms.shape[0]

    if duration <= 0 or num_frames <= 1:
        return object_transforms[0].copy()

    # Handle looping
    if loop:
        anim_time = anim_time % duration
    else:
        anim_time = max(0.0, min(anim_time, duration))

    frame_float = anim_time * fps
    frame_low = int(frame_float)
    frame_high = frame_low + 1
    t = frame_float - frame_low

    # Clamp indices
    frame_low = min(frame_low, num_frames - 1)
    frame_high = min(frame_high, num_frames - 1)

    # No interpolation needed
    if frame_low == frame_high or t < 0.001:
        return object_transforms[frame_low].copy()

    # Interpolate using blend_transforms (treats single transform as 1-element pose)
    low_pose = object_transforms[frame_low].reshape(1, 10)
    high_pose = object_transforms[frame_high].reshape(1, 10)
    blended = blend_transforms(low_pose, high_pose, t)
    return blended[0]


def blend_object_transforms(transforms: List[np.ndarray], weights: List[float]) -> Optional[np.ndarray]:
    """
    Blend multiple object transforms by weight.

    Args:
        transforms: List of (10,) numpy arrays
        weights: List of weights

    Returns:
        Blended transform (10,) or None if empty
    """
    if not transforms:
        return None

    if len(transforms) == 1:
        return transforms[0].copy()

    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        return transforms[0].copy()

    # Iterative blending
    result = transforms[0].reshape(1, 10)
    accumulated_weight = weights[0] / total_weight

    for i in range(1, len(transforms)):
        w = weights[i] / total_weight
        if accumulated_weight + w > 0:
            blend_t = w / (accumulated_weight + w)
            other = transforms[i].reshape(1, 10)
            result = blend_transforms(result, other, blend_t)
            accumulated_weight += w

    return result[0]