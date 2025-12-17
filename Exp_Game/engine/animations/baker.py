# Exp_Game/engine/animations/baker.py
"""
Animation Baker - Unified Action to BakedAnimation conversion.

Bakes Blender Actions to worker-safe BakedAnimation data.
Supports both armature (bone) and object-level animations.

Blender 5.0+ only - uses layered action API.
"""

from typing import Dict, List, Tuple, Optional, Set
from .data import BakedAnimation, Transform

DEFAULT_FPS = 30.0


def bake_action(
    action,           # bpy.types.Action
    target_object,    # bpy.types.Object (any type)
    fps: float = DEFAULT_FPS,
    bone_filter: Optional[Set[str]] = None
) -> BakedAnimation:
    """
    Bake a Blender Action to a BakedAnimation.

    Works with any object type:
    - Armatures: bakes bone transforms
    - Other objects: bakes object-level transforms (loc/rot/scale)
    - Can have both if action contains both types of FCurves

    Args:
        action: Blender Action to bake
        target_object: Target object (armature or any object)
        fps: Frames per second (default 30)
        bone_filter: Optional bone name filter (armatures only)

    Returns:
        BakedAnimation with bones and/or object_transforms
    """
    if action is None:
        raise ValueError("Cannot bake None action")

    frame_start, frame_end = action.frame_range
    frame_count = int(frame_end - frame_start) + 1
    duration = frame_count / fps

    all_fcurves = _get_all_fcurves(action)
    bone_fcurves, object_fcurves = _categorize_fcurves(all_fcurves)

    # Bake bone transforms (if armature and has bone FCurves)
    bones_data = {}
    if target_object.type == 'ARMATURE' and bone_fcurves:
        bones_data = _bake_bones(
            target_object, bone_fcurves, frame_start, frame_count, bone_filter
        )

    # Bake object transforms (if has object FCurves)
    object_transforms = []
    if object_fcurves:
        object_transforms = _bake_object(object_fcurves, frame_start, frame_count)

    return BakedAnimation(
        name=action.name,
        duration=duration,
        fps=fps,
        bones=bones_data,
        object_transforms=object_transforms,
        looping=True
    )


def _get_all_fcurves(action) -> List:
    """Get all FCurves from action using Blender 5.0 layered API."""
    all_fcurves = []
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                all_fcurves.extend(channelbag.fcurves)
    return all_fcurves


def _categorize_fcurves(fcurves) -> Tuple[Dict, Dict]:
    """
    Categorize FCurves into bone and object types.

    Returns:
        (bone_fcurves, object_fcurves)
        - bone_fcurves: {bone_name: {property: {index: fcurve}}}
        - object_fcurves: {property: {index: fcurve}}
    """
    bone_map = {}
    object_map = {}

    for fcurve in fcurves:
        data_path = fcurve.data_path

        # Bone FCurve: pose.bones["BoneName"].property
        if data_path.startswith('pose.bones["'):
            try:
                start = data_path.find('["') + 2
                end = data_path.find('"]', start)
                bone_name = data_path[start:end]
                prop_start = data_path.rfind('.') + 1
                prop_name = data_path[prop_start:]

                if bone_name not in bone_map:
                    bone_map[bone_name] = {}
                if prop_name not in bone_map[bone_name]:
                    bone_map[bone_name][prop_name] = {}
                bone_map[bone_name][prop_name][fcurve.array_index] = fcurve
            except (ValueError, IndexError):
                continue

        # Object FCurve: location, rotation_euler, rotation_quaternion, scale
        elif data_path in ('location', 'rotation_euler', 'rotation_quaternion', 'scale'):
            if data_path not in object_map:
                object_map[data_path] = {}
            object_map[data_path][fcurve.array_index] = fcurve

    return bone_map, object_map


def _bake_bones(
    armature,
    bone_fcurves: Dict,
    frame_start: float,
    frame_count: int,
    bone_filter: Optional[Set[str]]
) -> Dict[str, List[Transform]]:
    """Bake bone transforms from FCurves."""
    bones_data = {}
    pose_bones = armature.pose.bones

    for bone_name, fcurves in bone_fcurves.items():
        if bone_filter is not None and bone_name not in bone_filter:
            continue

        pose_bone = pose_bones.get(bone_name)
        if pose_bone is None:
            continue

        frames = []
        for frame_idx in range(frame_count):
            frame_num = frame_start + frame_idx
            transform = _sample_transform(fcurves, frame_num)
            frames.append(transform)

        if frames:
            bones_data[bone_name] = frames

    return bones_data


def _bake_object(
    object_fcurves: Dict,
    frame_start: float,
    frame_count: int
) -> List[Transform]:
    """Bake object-level transforms from FCurves."""
    frames = []
    for frame_idx in range(frame_count):
        frame_num = frame_start + frame_idx
        transform = _sample_transform(object_fcurves, frame_num)
        frames.append(transform)
    return frames


def _sample_transform(fcurves: Dict[str, Dict[int, any]], frame: float) -> Transform:
    """
    Sample transform at a specific frame.

    Returns:
        (quat_w, quat_x, quat_y, quat_z, loc_x, loc_y, loc_z, scale_x, scale_y, scale_z)
    """
    quat = [1.0, 0.0, 0.0, 0.0]
    loc = [0.0, 0.0, 0.0]
    scale = [1.0, 1.0, 1.0]

    # Quaternion rotation
    if "rotation_quaternion" in fcurves:
        for i in range(4):
            if i in fcurves["rotation_quaternion"]:
                quat[i] = fcurves["rotation_quaternion"][i].evaluate(frame)

    # Euler rotation (convert to quaternion)
    elif "rotation_euler" in fcurves:
        import mathutils
        euler = [0.0, 0.0, 0.0]
        for i in range(3):
            if i in fcurves["rotation_euler"]:
                euler[i] = fcurves["rotation_euler"][i].evaluate(frame)
        euler_obj = mathutils.Euler(euler, 'XYZ')
        quat_obj = euler_obj.to_quaternion()
        quat = [quat_obj.w, quat_obj.x, quat_obj.y, quat_obj.z]

    # Location
    if "location" in fcurves:
        for i in range(3):
            if i in fcurves["location"]:
                loc[i] = fcurves["location"][i].evaluate(frame)

    # Scale
    if "scale" in fcurves:
        for i in range(3):
            if i in fcurves["scale"]:
                scale[i] = fcurves["scale"][i].evaluate(frame)

    return tuple(quat + loc + scale)
