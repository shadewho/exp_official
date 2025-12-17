# Exp_Game/animations/apply.py
"""
Apply utilities - Standalone functions for applying transforms via bpy.

These are lower-level utilities that can be used independently of AnimationController.
Useful for one-shot applications or custom animation logic.
"""

import bpy
from typing import Dict, Optional
from ..engine.animations.data import Transform, BakedAnimation
from ..engine.animations.blend import sample_animation, interpolate_transform


def apply_transform_to_object(obj, transform: Transform) -> None:
    """
    Apply a transform tuple to any Blender object.

    Args:
        obj: bpy.types.Object (any type)
        transform: 10-float tuple (qw, qx, qy, qz, lx, ly, lz, sx, sy, sz)
    """
    qw, qx, qy, qz = transform[0:4]
    lx, ly, lz = transform[4:7]
    sx, sy, sz = transform[7:10]

    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = (qw, qx, qy, qz)
    obj.location = (lx, ly, lz)
    obj.scale = (sx, sy, sz)


def apply_transform_to_bone(pose_bone, transform: Transform) -> None:
    """
    Apply a transform tuple to a pose bone.

    Args:
        pose_bone: bpy.types.PoseBone
        transform: 10-float tuple (qw, qx, qy, qz, lx, ly, lz, sx, sy, sz)
    """
    qw, qx, qy, qz = transform[0:4]
    lx, ly, lz = transform[4:7]
    sx, sy, sz = transform[7:10]

    pose_bone.rotation_quaternion = (qw, qx, qy, qz)
    pose_bone.location = (lx, ly, lz)
    pose_bone.scale = (sx, sy, sz)


def apply_bone_pose(armature, bone_transforms: Dict[str, Transform]) -> int:
    """
    Apply transforms to multiple bones on an armature.

    Args:
        armature: bpy.types.Object (must be ARMATURE type)
        bone_transforms: Dict mapping bone_name to Transform

    Returns:
        Number of bones successfully updated
    """
    if armature.type != 'ARMATURE':
        return 0

    pose_bones = armature.pose.bones
    count = 0

    for bone_name, transform in bone_transforms.items():
        pose_bone = pose_bones.get(bone_name)
        if pose_bone is not None:
            apply_transform_to_bone(pose_bone, transform)
            count += 1

    return count


def apply_animation_frame(
    obj,
    animation: BakedAnimation,
    frame: int,
    apply_bones: bool = True,
    apply_object: bool = True
) -> None:
    """
    Apply a specific frame from a baked animation.

    Args:
        obj: Target object (armature or any object)
        animation: BakedAnimation to sample from
        frame: Frame index to apply (clamped to valid range)
        apply_bones: Apply bone transforms (if armature)
        apply_object: Apply object transforms (if present)
    """
    frame = max(0, min(frame, animation.frame_count - 1))

    # Apply bone transforms
    if apply_bones and animation.bones and obj.type == 'ARMATURE':
        pose_bones = obj.pose.bones
        for bone_name, frames in animation.bones.items():
            if frame < len(frames):
                pose_bone = pose_bones.get(bone_name)
                if pose_bone:
                    apply_transform_to_bone(pose_bone, frames[frame])

    # Apply object transforms
    if apply_object and animation.object_transforms:
        if frame < len(animation.object_transforms):
            apply_transform_to_object(obj, animation.object_transforms[frame])


def apply_animation_time(
    obj,
    animation: BakedAnimation,
    time: float,
    loop: bool = True,
    apply_bones: bool = True,
    apply_object: bool = True
) -> None:
    """
    Apply animation at a specific time with interpolation.

    Args:
        obj: Target object (armature or any object)
        animation: BakedAnimation to sample from
        time: Time in seconds
        loop: Whether to loop time within duration
        apply_bones: Apply bone transforms (if armature)
        apply_object: Apply object transforms (if present)
    """
    # Sample the animation (uses worker-safe blend.py)
    bones, obj_transform = sample_animation(animation, time, loop)

    # Apply bone transforms
    if apply_bones and bones and obj.type == 'ARMATURE':
        apply_bone_pose(obj, bones)

    # Apply object transforms
    if apply_object and obj_transform:
        apply_transform_to_object(obj, obj_transform)


def apply_animation_normalized(
    obj,
    animation: BakedAnimation,
    t: float,
    apply_bones: bool = True,
    apply_object: bool = True
) -> None:
    """
    Apply animation at normalized time (0.0 = start, 1.0 = end).

    Args:
        obj: Target object
        animation: BakedAnimation to sample from
        t: Normalized time (0.0 to 1.0)
        apply_bones: Apply bone transforms
        apply_object: Apply object transforms
    """
    t = max(0.0, min(1.0, t))
    time = t * animation.duration
    apply_animation_time(obj, animation, time, loop=False,
                        apply_bones=apply_bones, apply_object=apply_object)


def reset_pose(armature) -> int:
    """
    Reset all pose bones to identity (rest pose).

    Args:
        armature: bpy.types.Object (must be ARMATURE type)

    Returns:
        Number of bones reset
    """
    if armature.type != 'ARMATURE':
        return 0

    identity = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    count = 0

    for pose_bone in armature.pose.bones:
        apply_transform_to_bone(pose_bone, identity)
        count += 1

    return count


def get_object_transform(obj) -> Transform:
    """
    Get current transform of an object as a Transform tuple.

    Args:
        obj: bpy.types.Object

    Returns:
        10-float Transform tuple
    """
    obj.rotation_mode = 'QUATERNION'
    q = obj.rotation_quaternion
    l = obj.location
    s = obj.scale

    return (q.w, q.x, q.y, q.z, l.x, l.y, l.z, s.x, s.y, s.z)


def get_bone_transform(pose_bone) -> Transform:
    """
    Get current transform of a pose bone as a Transform tuple.

    Args:
        pose_bone: bpy.types.PoseBone

    Returns:
        10-float Transform tuple
    """
    q = pose_bone.rotation_quaternion
    l = pose_bone.location
    s = pose_bone.scale

    return (q.w, q.x, q.y, q.z, l.x, l.y, l.z, s.x, s.y, s.z)
