# Exp_Game/engine/animations/blend.py
"""
Animation blending math - worker-safe, no bpy.

Functions for interpolating and blending transforms.
Works with the 10-float transform format:
  (quat_w, quat_x, quat_y, quat_z, loc_x, loc_y, loc_z, scale_x, scale_y, scale_z)
"""

import math
from typing import Tuple, List, Dict, Optional
from .data import Transform, BakedAnimation


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between two values."""
    return a + (b - a) * t


def slerp(q1: Tuple[float, float, float, float],
          q2: Tuple[float, float, float, float],
          t: float) -> Tuple[float, float, float, float]:
    """
    Spherical linear interpolation between two quaternions.
    Quaternions are (w, x, y, z) format.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    # Dot product
    dot = w1*w2 + x1*x2 + y1*y2 + z1*z2

    # Take shorter path
    if dot < 0.0:
        w2, x2, y2, z2 = -w2, -x2, -y2, -z2
        dot = -dot

    # Very close - use linear interpolation
    if dot > 0.9995:
        w = w1 + t * (w2 - w1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        z = z1 + t * (z2 - z1)
        # Normalize
        length = (w*w + x*x + y*y + z*z) ** 0.5
        if length > 0:
            w, x, y, z = w/length, x/length, y/length, z/length
        return (w, x, y, z)

    # Standard slerp
    theta_0 = math.acos(min(1.0, max(-1.0, dot)))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    if abs(sin_theta_0) < 1e-10:
        return (w1, x1, y1, z1)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (
        s0 * w1 + s1 * w2,
        s0 * x1 + s1 * x2,
        s0 * y1 + s1 * y2,
        s0 * z1 + s1 * z2,
    )


def blend_transform(t1: Transform, t2: Transform, weight: float) -> Transform:
    """
    Blend two transforms. weight=0 returns t1, weight=1 returns t2.
    Uses slerp for quaternion, lerp for location/scale.
    """
    q = slerp(t1[0:4], t2[0:4], weight)
    l = (lerp(t1[4], t2[4], weight),
         lerp(t1[5], t2[5], weight),
         lerp(t1[6], t2[6], weight))
    s = (lerp(t1[7], t2[7], weight),
         lerp(t1[8], t2[8], weight),
         lerp(t1[9], t2[9], weight))
    return q + l + s


def interpolate_transform(frames: List[Transform], frame_float: float) -> Transform:
    """
    Interpolate between frames at a fractional frame index.
    """
    if not frames:
        return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    frame_low = int(frame_float)
    frame_high = frame_low + 1
    t = frame_float - frame_low

    # Clamp indices
    max_idx = len(frames) - 1
    frame_low = min(frame_low, max_idx)
    frame_high = min(frame_high, max_idx)

    if frame_low == frame_high or t < 0.001:
        return frames[frame_low]

    return blend_transform(frames[frame_low], frames[frame_high], t)


def sample_animation(anim: BakedAnimation, time: float, loop: bool = True) -> Tuple[Dict[str, Transform], Optional[Transform]]:
    """
    Sample animation at a specific time.

    Returns:
        (bone_transforms, object_transform)
        - bone_transforms: Dict[bone_name, Transform] or empty dict
        - object_transform: Transform or None
    """
    if anim.duration <= 0:
        bones = {name: frames[0] for name, frames in anim.bones.items() if frames}
        obj = anim.object_transforms[0] if anim.object_transforms else None
        return bones, obj

    # Handle looping
    if loop:
        time = time % anim.duration
    else:
        time = max(0.0, min(time, anim.duration))

    frame_float = time * anim.fps

    # Sample bones
    bones = {}
    for bone_name, frames in anim.bones.items():
        if frames:
            bones[bone_name] = interpolate_transform(frames, frame_float)

    # Sample object
    obj = None
    if anim.object_transforms:
        obj = interpolate_transform(anim.object_transforms, frame_float)

    return bones, obj


def blend_bone_poses(poses: List[Tuple[Dict[str, Transform], float]]) -> Dict[str, Transform]:
    """
    Blend multiple bone poses by weight.

    Args:
        poses: List of (pose_dict, weight) tuples

    Returns:
        Blended pose dict
    """
    if not poses:
        return {}

    if len(poses) == 1:
        return poses[0][0]

    # Normalize weights
    total_weight = sum(w for _, w in poses)
    if total_weight <= 0:
        return poses[0][0]

    # Collect all bone names
    all_bones = set()
    for pose, _ in poses:
        all_bones.update(pose.keys())

    result = {}
    for bone_name in all_bones:
        bone_data = [(pose[bone_name], w / total_weight)
                     for pose, w in poses if bone_name in pose]

        if not bone_data:
            continue

        if len(bone_data) == 1:
            result[bone_name] = bone_data[0][0]
            continue

        # Iterative blending
        blended = bone_data[0][0]
        acc_weight = bone_data[0][1]

        for transform, weight in bone_data[1:]:
            if acc_weight + weight > 0:
                blend_t = weight / (acc_weight + weight)
                blended = blend_transform(blended, transform, blend_t)
                acc_weight += weight

        result[bone_name] = blended

    return result


def blend_object_transforms(transforms: List[Tuple[Transform, float]]) -> Optional[Transform]:
    """
    Blend multiple object transforms by weight.

    Args:
        transforms: List of (transform, weight) tuples

    Returns:
        Blended transform or None
    """
    if not transforms:
        return None

    if len(transforms) == 1:
        return transforms[0][0]

    total_weight = sum(w for _, w in transforms)
    if total_weight <= 0:
        return transforms[0][0]

    # Iterative blending
    blended = transforms[0][0]
    acc_weight = transforms[0][1] / total_weight

    for transform, weight in transforms[1:]:
        w = weight / total_weight
        if acc_weight + w > 0:
            blend_t = w / (acc_weight + w)
            blended = blend_transform(blended, transform, blend_t)
            acc_weight += w

    return blended
