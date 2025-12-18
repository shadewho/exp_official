# Exp_Game/engine/animations/baker.py
"""
Animation Baker - Unified Action to BakedAnimation conversion.

Bakes Blender Actions to worker-safe BakedAnimation data with numpy arrays.
Supports both armature (bone) and object-level animations.

NUMPY OPTIMIZATION:
  - Outputs contiguous numpy arrays for maximum performance
  - Detects static bones at bake time (skip at runtime)
  - Shape: (num_frames, num_bones, 10) for bone transforms

Blender 5.0+ only - uses layered action API.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from .data import BakedAnimation

DEFAULT_FPS = 30.0

# Threshold for detecting static bones (max variance across all frames)
STATIC_BONE_THRESHOLD = 1e-6


def bake_action(
    action,           # bpy.types.Action
    target_object,    # bpy.types.Object (any type)
    fps: float = DEFAULT_FPS,
    bone_filter: Optional[Set[str]] = None
) -> BakedAnimation:
    """
    Bake a Blender Action to a BakedAnimation with numpy arrays.

    Works with any object type:
    - Armatures: bakes bone transforms to (num_frames, num_bones, 10) array
    - Other objects: bakes object-level transforms to (num_frames, 10) array
    - Can have both if action contains both types of FCurves

    Args:
        action: Blender Action to bake
        target_object: Target object (armature or any object)
        fps: Frames per second (default 30)
        bone_filter: Optional bone name filter (armatures only)

    Returns:
        BakedAnimation with numpy arrays and static bone detection
    """
    if action is None:
        raise ValueError("Cannot bake None action")

    frame_start, frame_end = action.frame_range
    frame_count = int(frame_end - frame_start) + 1
    duration = frame_count / fps

    all_fcurves = _get_all_fcurves(action)
    bone_fcurves, object_fcurves = _categorize_fcurves(all_fcurves)

    # Bake bone transforms (if armature and has bone FCurves)
    bone_names = []
    bone_transforms = None
    animated_mask = None

    if target_object.type == 'ARMATURE' and bone_fcurves:
        bone_names, bone_transforms, animated_mask = _bake_bones_numpy(
            target_object, bone_fcurves, frame_start, frame_count, bone_filter
        )

    # Bake object transforms (if has object FCurves)
    object_transforms = None
    if object_fcurves:
        object_transforms = _bake_object_numpy(object_fcurves, frame_start, frame_count)

    return BakedAnimation(
        name=action.name,
        duration=duration,
        fps=fps,
        bone_names=bone_names,
        bone_transforms=bone_transforms,
        animated_mask=animated_mask,
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


def _bake_bones_numpy(
    armature,
    bone_fcurves: Dict,
    frame_start: float,
    frame_count: int,
    bone_filter: Optional[Set[str]]
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Bake bone transforms to numpy arrays.

    Returns:
        (bone_names, bone_transforms, animated_mask)
        - bone_names: List[str] of bone names in order
        - bone_transforms: np.ndarray shape (num_frames, num_bones, 10)
        - animated_mask: np.ndarray bool shape (num_bones,) - True if bone animates
    """
    pose_bones = armature.pose.bones

    # Determine which bones to bake (filtered or all with FCurves)
    bones_to_bake = []
    for bone_name in bone_fcurves.keys():
        if bone_filter is not None and bone_name not in bone_filter:
            continue
        if bone_name in pose_bones:
            bones_to_bake.append(bone_name)

    if not bones_to_bake:
        return [], np.empty((0, 0, 10), dtype=np.float32), np.empty(0, dtype=bool)

    # Sort for consistent ordering
    bones_to_bake.sort()
    num_bones = len(bones_to_bake)

    # Pre-allocate numpy array: (frames, bones, 10)
    transforms = np.zeros((frame_count, num_bones, 10), dtype=np.float32)

    # Set identity defaults (quat w=1, scale=1)
    transforms[:, :, 0] = 1.0  # quat_w
    transforms[:, :, 7] = 1.0  # scale_x
    transforms[:, :, 8] = 1.0  # scale_y
    transforms[:, :, 9] = 1.0  # scale_z

    # Bake each bone
    for bone_idx, bone_name in enumerate(bones_to_bake):
        fcurves = bone_fcurves[bone_name]

        for frame_idx in range(frame_count):
            frame_num = frame_start + frame_idx
            transform = _sample_transform(fcurves, frame_num)
            transforms[frame_idx, bone_idx, :] = transform

    # Detect static bones (all frames identical within threshold)
    animated_mask = _detect_animated_bones(transforms)

    return bones_to_bake, transforms, animated_mask


def _bake_object_numpy(
    object_fcurves: Dict,
    frame_start: float,
    frame_count: int
) -> np.ndarray:
    """
    Bake object-level transforms to numpy array.

    Returns:
        np.ndarray shape (num_frames, 10)
    """
    transforms = np.zeros((frame_count, 10), dtype=np.float32)

    # Set identity defaults
    transforms[:, 0] = 1.0  # quat_w
    transforms[:, 7] = 1.0  # scale_x
    transforms[:, 8] = 1.0  # scale_y
    transforms[:, 9] = 1.0  # scale_z

    for frame_idx in range(frame_count):
        frame_num = frame_start + frame_idx
        transform = _sample_transform(object_fcurves, frame_num)
        transforms[frame_idx, :] = transform

    return transforms


def _sample_transform(fcurves: Dict[str, Dict[int, any]], frame: float) -> np.ndarray:
    """
    Sample transform at a specific frame.

    Returns:
        np.ndarray shape (10,): [quat_w, quat_x, quat_y, quat_z, loc_x, loc_y, loc_z, scale_x, scale_y, scale_z]
    """
    result = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # Quaternion rotation
    if "rotation_quaternion" in fcurves:
        for i in range(4):
            if i in fcurves["rotation_quaternion"]:
                result[i] = fcurves["rotation_quaternion"][i].evaluate(frame)

    # Euler rotation (convert to quaternion)
    elif "rotation_euler" in fcurves:
        import mathutils
        euler = [0.0, 0.0, 0.0]
        for i in range(3):
            if i in fcurves["rotation_euler"]:
                euler[i] = fcurves["rotation_euler"][i].evaluate(frame)
        euler_obj = mathutils.Euler(euler, 'XYZ')
        quat_obj = euler_obj.to_quaternion()
        result[0:4] = [quat_obj.w, quat_obj.x, quat_obj.y, quat_obj.z]

    # Location
    if "location" in fcurves:
        for i in range(3):
            if i in fcurves["location"]:
                result[4 + i] = fcurves["location"][i].evaluate(frame)

    # Scale
    if "scale" in fcurves:
        for i in range(3):
            if i in fcurves["scale"]:
                result[7 + i] = fcurves["scale"][i].evaluate(frame)

    return result


def _detect_animated_bones(transforms: np.ndarray) -> np.ndarray:
    """
    Detect which bones actually animate (have varying transforms).

    Args:
        transforms: np.ndarray shape (num_frames, num_bones, 10)

    Returns:
        np.ndarray bool shape (num_bones,) - True if bone has movement
    """
    if transforms.shape[0] <= 1:
        # Single frame - nothing animates
        return np.zeros(transforms.shape[1], dtype=bool)

    # Compute variance across frames for each bone
    # A bone is static if ALL its transform components have near-zero variance
    # Shape: (num_bones, 10)
    variance = np.var(transforms, axis=0)

    # Sum variance across all 10 transform components per bone
    # Shape: (num_bones,)
    total_variance = np.sum(variance, axis=1)

    # Bone is animated if total variance exceeds threshold
    animated = total_variance > STATIC_BONE_THRESHOLD

    return animated
