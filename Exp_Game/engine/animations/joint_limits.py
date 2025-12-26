# engine/animations/joint_limits.py
"""
Joint Limits - Worker-safe joint constraint system.

NO BPY IMPORTS - this runs in engine workers.

Stores rotation limits for each bone and provides clamping functions
that are called during POSE_BLEND_COMPUTE to ensure anatomically valid poses.

Data Flow:
1. User records limits using Joint Limit Recorder (main thread)
2. Limits saved to JSON file with rig
3. On game start, limits loaded and sent to worker via CACHE_JOINT_LIMITS
4. Worker uses limits during pose blending
"""

import numpy as np
import math
import json
import os
from typing import Dict, Optional, Tuple, Any, List


# =============================================================================
# WORKER-SIDE CACHE
# =============================================================================

# Cached joint limits in worker process
# Format: {bone_name: {"X": [min, max], "Y": [min, max], "Z": [min, max]}}
_worker_joint_limits: Dict[str, Dict[str, List[float]]] = {}


def set_worker_limits(limits: Dict[str, Dict[str, List[float]]]) -> None:
    """
    Cache joint limits in worker (called via CACHE_JOINT_LIMITS job).

    Args:
        limits: Dict of bone_name -> {"X": [min, max], "Y": [min, max], "Z": [min, max]}
    """
    global _worker_joint_limits
    _worker_joint_limits = limits.copy()


def get_worker_limits() -> Dict[str, Dict[str, List[float]]]:
    """Get cached joint limits in worker."""
    return _worker_joint_limits


def has_worker_limits() -> bool:
    """Check if worker has joint limits cached."""
    return len(_worker_joint_limits) > 0


# =============================================================================
# QUATERNION <-> EULER CONVERSION (Numpy-based, worker-safe)
# =============================================================================

def quaternion_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to euler angles [x, y, z] in degrees.
    Uses XYZ rotation order (matches Blender default).
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # Roll (X-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)  # Clamp for numerical stability
    pitch = math.asin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([math.degrees(roll), math.degrees(pitch), math.degrees(yaw)], dtype=np.float32)


def euler_xyz_to_quaternion(euler_deg: np.ndarray) -> np.ndarray:
    """
    Convert euler angles [x, y, z] in degrees to quaternion [w, x, y, z].
    Uses XYZ rotation order.
    """
    roll = math.radians(euler_deg[0])   # X
    pitch = math.radians(euler_deg[1])  # Y
    yaw = math.radians(euler_deg[2])    # Z

    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # Normalize
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm

    return np.array([w, x, y, z], dtype=np.float32)


# =============================================================================
# LIMIT CLAMPING (Worker-safe)
# =============================================================================

def clamp_rotation(
    bone_name: str,
    quaternion: np.ndarray,
    limits: Optional[Dict[str, Dict[str, List[float]]]] = None
) -> Tuple[np.ndarray, bool, Dict[str, float]]:
    """
    Clamp a bone's rotation to its joint limits.

    Args:
        bone_name: Name of the bone
        quaternion: Rotation as [w, x, y, z]
        limits: Joint limits dict (uses worker cache if None)

    Returns:
        Tuple of (clamped_quaternion, was_clamped, violations)
        violations: {axis: degrees_over_limit}
    """
    if limits is None:
        limits = _worker_joint_limits

    bone_limits = limits.get(bone_name)
    if not bone_limits:
        return quaternion, False, {}

    # Convert to euler
    euler = quaternion_to_euler_xyz(quaternion)
    clamped_euler = euler.copy()
    was_clamped = False
    violations = {}

    # Clamp each axis
    for i, axis in enumerate(['X', 'Y', 'Z']):
        axis_limits = bone_limits.get(axis)
        if axis_limits and len(axis_limits) == 2:
            min_deg, max_deg = axis_limits[0], axis_limits[1]
            original = euler[i]

            if original < min_deg:
                violations[axis] = min_deg - original
                clamped_euler[i] = min_deg
                was_clamped = True
            elif original > max_deg:
                violations[axis] = original - max_deg
                clamped_euler[i] = max_deg
                was_clamped = True

    # Convert back to quaternion
    if was_clamped:
        return euler_xyz_to_quaternion(clamped_euler), True, violations

    return quaternion, False, violations


def apply_limits_to_pose(
    bone_transforms: Dict[str, Any],
    limits: Optional[Dict[str, Dict[str, List[float]]]] = None
) -> Tuple[Dict[str, Any], int, List[str]]:
    """
    Apply joint limits to an entire pose (all bones).

    Args:
        bone_transforms: Dict of bone_name -> [qw, qx, qy, qz, lx, ly, lz, sx, sy, sz]
        limits: Joint limits dict (uses worker cache if None)

    Returns:
        Tuple of (clamped_transforms, num_clamped, clamped_bone_names)
    """
    if limits is None:
        limits = _worker_joint_limits

    clamped_transforms = {}
    num_clamped = 0
    clamped_bones = []

    for bone_name, transform in bone_transforms.items():
        # Extract quaternion
        quat = np.array([transform[0], transform[1], transform[2], transform[3]], dtype=np.float32)

        # Clamp
        clamped_quat, was_clamped, _ = clamp_rotation(bone_name, quat, limits)

        if was_clamped:
            num_clamped += 1
            clamped_bones.append(bone_name)

        # Rebuild transform
        clamped_transforms[bone_name] = [
            float(clamped_quat[0]), float(clamped_quat[1]),
            float(clamped_quat[2]), float(clamped_quat[3]),
            transform[4], transform[5], transform[6],  # Location unchanged
            transform[7], transform[8], transform[9],  # Scale unchanged
        ]

    return clamped_transforms, num_clamped, clamped_bones


# =============================================================================
# FILE I/O (For saving/loading limits with rig)
# =============================================================================

def save_limits_to_file(limits: Dict[str, Dict[str, List[float]]], filepath: str) -> bool:
    """Save joint limits to JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(limits, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving joint limits: {e}")
        return False


def load_limits_from_file(filepath: str) -> Optional[Dict[str, Dict[str, List[float]]]]:
    """Load joint limits from JSON file."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading joint limits: {e}")
        return None


# =============================================================================
# VALIDATION
# =============================================================================

def is_within_limits(
    bone_name: str,
    quaternion: np.ndarray,
    limits: Optional[Dict[str, Dict[str, List[float]]]] = None
) -> Tuple[bool, Dict[str, float]]:
    """
    Check if a rotation is within limits without clamping.

    Returns:
        Tuple of (is_valid, violations_dict)
    """
    _, was_clamped, violations = clamp_rotation(bone_name, quaternion, limits)
    return not was_clamped, violations


def get_limit_info(bone_name: str, limits: Optional[Dict[str, Dict[str, List[float]]]] = None) -> Optional[Dict[str, List[float]]]:
    """Get limit info for a specific bone."""
    if limits is None:
        limits = _worker_joint_limits
    return limits.get(bone_name)
