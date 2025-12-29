# Exp_Game/engine/animations/baker.py
"""
Animation Baker - Unified Action to BakedAnimation conversion.

Bakes Blender Actions to worker-safe BakedAnimation data with numpy arrays.
Supports both armature (bone) and object-level animations.

NO ARMATURE DEPENDENCY: Baking extracts data directly from FCurves.
The action defines what bones/properties to animate - we just bake that data.
At runtime, apply code handles missing bones gracefully.

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
    fps: float = DEFAULT_FPS,
    bone_filter: Optional[Set[str]] = None
) -> BakedAnimation:
    """
    Bake a Blender Action to a BakedAnimation with numpy arrays.

    NO ARMATURE REQUIRED - extracts all data from FCurves directly.

    Bakes:
    - Bone transforms from pose.bones["..."] FCurves → (num_frames, num_bones, 10) array
    - Object transforms from location/rotation/scale FCurves → (num_frames, 10) array

    Args:
        action: Blender Action to bake
        fps: Source frames per second of the action (will be clamped to 30 for output)
        bone_filter: Optional set of bone names to include (None = all)

    Returns:
        BakedAnimation with numpy arrays and static bone detection
    """
    if action is None:
        raise ValueError("Cannot bake None action")

    # Clamp to fixed runtime rate (30Hz)
    target_fps = DEFAULT_FPS

    # Source fps (from scene/action); fall back to DEFAULT_FPS if invalid
    source_fps = fps if fps and fps > 0 else DEFAULT_FPS

    frame_start, frame_end = action.frame_range
    raw_frame_count = int(frame_end - frame_start) + 1

    # Duration based on source fps
    duration = (raw_frame_count - 1) / source_fps if raw_frame_count > 1 else 0.0

    # Number of samples at target fps (include last frame)
    sample_frames = max(1, int(round(duration * target_fps)) + 1)

    all_fcurves = _get_all_fcurves(action)
    bone_fcurves, object_fcurves = _categorize_fcurves(all_fcurves)

    # Bake bone transforms (if has bone FCurves)
    bone_names = []
    bone_transforms = None
    animated_mask = None

    if bone_fcurves:
        bone_names, bone_transforms, animated_mask = _bake_bones_numpy(
            bone_fcurves,
            frame_start,
            sample_frames,
            source_fps,
            target_fps,
            bone_filter
        )

    # Bake object transforms (if has object FCurves)
    object_transforms = None
    if object_fcurves:
        object_transforms = _bake_object_numpy(
            object_fcurves,
            frame_start,
            sample_frames,
            source_fps,
            target_fps
        )

    return BakedAnimation(
        name=action.name,
        duration=duration,
        fps=target_fps,
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
    bone_fcurves: Dict,
    frame_start: float,
    sample_frames: int,
    source_fps: float,
    target_fps: float,
    bone_filter: Optional[Set[str]]
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Bake bone transforms to numpy arrays.

    NO ARMATURE VALIDATION - bakes all bones from FCurves directly.
    At runtime, apply code will skip bones that don't exist on target.

    Returns:
        (bone_names, bone_transforms, animated_mask)
        - bone_names: List[str] of bone names in order
        - bone_transforms: np.ndarray shape (sample_frames, num_bones, 10)
        - animated_mask: np.ndarray bool shape (num_bones,) - True if bone animates
    """
    # Get all bone names from FCurves (no armature validation)
    bones_to_bake = []
    for bone_name in bone_fcurves.keys():
        if bone_filter is not None and bone_name not in bone_filter:
            continue
        bones_to_bake.append(bone_name)

    if not bones_to_bake:
        return [], np.empty((0, 0, 10), dtype=np.float32), np.empty(0, dtype=bool)

    # Sort for consistent ordering
    bones_to_bake.sort()
    num_bones = len(bones_to_bake)

    # Pre-allocate numpy array: (sample_frames, bones, 10)
    transforms = np.zeros((sample_frames, num_bones, 10), dtype=np.float32)

    # Set identity defaults (quat w=1, scale=1)
    transforms[:, :, 0] = 1.0  # quat_w
    transforms[:, :, 7] = 1.0  # scale_x
    transforms[:, :, 8] = 1.0  # scale_y
    transforms[:, :, 9] = 1.0  # scale_z

    # Bake each bone
    for bone_idx, bone_name in enumerate(bones_to_bake):
        fcurves = bone_fcurves[bone_name]

        for sample_idx in range(sample_frames):
            # Map target sample time to source frame domain
            t_seconds = sample_idx / target_fps
            frame_num = frame_start + t_seconds * source_fps
            transform = _sample_transform(fcurves, frame_num)
            transforms[sample_idx, bone_idx, :] = transform

    # Detect static bones (all frames identical within threshold)
    animated_mask = _detect_animated_bones(transforms)

    return bones_to_bake, transforms, animated_mask


def _bake_object_numpy(
    object_fcurves: Dict,
    frame_start: float,
    sample_frames: int,
    source_fps: float,
    target_fps: float
) -> np.ndarray:
    """
    Bake object-level transforms to numpy array.

    Returns:
        np.ndarray shape (sample_frames, 10)
    """
    transforms = np.zeros((sample_frames, 10), dtype=np.float32)

    # Set identity defaults
    transforms[:, 0] = 1.0  # quat_w
    transforms[:, 7] = 1.0  # scale_x
    transforms[:, 8] = 1.0  # scale_y
    transforms[:, 9] = 1.0  # scale_z

    for sample_idx in range(sample_frames):
        # Map target sample time to source frame domain
        t_seconds = sample_idx / target_fps
        frame_num = frame_start + t_seconds * source_fps
        transform = _sample_transform(object_fcurves, frame_num)
        transforms[sample_idx, :] = transform

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
