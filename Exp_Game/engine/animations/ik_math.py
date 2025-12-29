# Exp_Game/engine/animations/ik_math.py
"""
IK Math Utilities - Vector and Quaternion operations.

Pure numpy math for IK solving. Worker-safe (NO bpy imports).

All quaternions are in [w, x, y, z] format.
All vectors are numpy arrays with dtype=float32.
"""

import numpy as np
from typing import Tuple


# =============================================================================
# VECTOR OPERATIONS
# =============================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        v: Input vector

    Returns:
        Normalized vector (or zero vector if input is zero)
    """
    length = np.linalg.norm(v)
    if length < 1e-10:
        return np.zeros_like(v)
    return v / length


def safe_normalize(v: np.ndarray, fallback: np.ndarray = None) -> np.ndarray:
    """
    Normalize a vector with fallback for zero-length vectors.

    Args:
        v: Input vector
        fallback: Vector to return if input is zero-length

    Returns:
        Normalized vector or fallback
    """
    length = np.linalg.norm(v)
    if length < 1e-10:
        if fallback is not None:
            return fallback.copy()
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Default up
    return v / length


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two vectors."""
    return float(np.dot(a, b))


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two vectors."""
    return np.cross(a, b)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two vectors."""
    return a * (1.0 - t) + b * t


def project_onto_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project a vector onto a plane defined by its normal."""
    normal = normalize(normal)
    return v - normal * np.dot(v, normal)


# =============================================================================
# QUATERNION OPERATIONS
# =============================================================================

def quat_identity() -> np.ndarray:
    """Return identity quaternion [w, x, y, z]."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length."""
    length = np.linalg.norm(q)
    if length < 1e-10:
        return quat_identity()
    return q / length


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.

    Args:
        q1, q2: Quaternions in [w, x, y, z] format

    Returns:
        Product quaternion q1 * q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Return conjugate of quaternion (inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Return inverse of quaternion."""
    conj = quat_conjugate(q)
    norm_sq = np.dot(q, q)
    if norm_sq < 1e-10:
        return quat_identity()
    return conj / norm_sq


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate a vector by a quaternion.

    Args:
        q: Quaternion [w, x, y, z]
        v: Vector [x, y, z]

    Returns:
        Rotated vector
    """
    # Convert vector to quaternion form
    v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)

    # q * v * q^-1
    q_conj = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)

    return np.array([result[1], result[2], result[3]], dtype=np.float32)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create quaternion from axis-angle representation.

    Args:
        axis: Rotation axis (will be normalized)
        angle: Rotation angle in radians

    Returns:
        Quaternion [w, x, y, z]
    """
    axis = normalize(axis)
    half_angle = angle * 0.5
    s = np.sin(half_angle)
    c = np.cos(half_angle)

    return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def quat_from_two_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Create quaternion that rotates v1 to align with v2.

    Args:
        v1: Source direction (will be normalized)
        v2: Target direction (will be normalized)

    Returns:
        Rotation quaternion
    """
    v1 = normalize(v1)
    v2 = normalize(v2)

    d = np.dot(v1, v2)

    # Vectors are parallel (same direction)
    if d > 0.999999:
        return quat_identity()

    # Vectors are anti-parallel (opposite direction)
    if d < -0.999999:
        # Find perpendicular axis
        axis = np.cross(np.array([1, 0, 0], dtype=np.float32), v1)
        if np.linalg.norm(axis) < 0.001:
            axis = np.cross(np.array([0, 1, 0], dtype=np.float32), v1)
        axis = normalize(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float32)

    # General case
    axis = np.cross(v1, v2)
    w = 1.0 + d
    q = np.array([w, axis[0], axis[1], axis[2]], dtype=np.float32)
    return quat_normalize(q)


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between quaternions.

    Args:
        q1: Start quaternion
        q2: End quaternion
        t: Interpolation factor (0-1)

    Returns:
        Interpolated quaternion
    """
    # Ensure shortest path
    d = np.dot(q1, q2)
    if d < 0:
        q2 = -q2
        d = -d

    # If quaternions are very close, use linear interpolation
    if d > 0.9995:
        result = q1 + t * (q2 - q1)
        return quat_normalize(result)

    # Slerp
    theta = np.arccos(np.clip(d, -1, 1))
    sin_theta = np.sin(theta)

    s1 = np.sin((1 - t) * theta) / sin_theta
    s2 = np.sin(t * theta) / sin_theta

    return quat_normalize(s1 * q1 + s2 * q2)


def quat_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (XYZ order).

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        (x, y, z) Euler angles in radians
    """
    w, x, y, z = q

    # Roll (X)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (Z)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def euler_to_quat(x: float, y: float, z: float) -> np.ndarray:
    """
    Convert Euler angles to quaternion (XYZ order).

    Args:
        x, y, z: Euler angles in radians

    Returns:
        Quaternion [w, x, y, z]
    """
    cx, sx = np.cos(x * 0.5), np.sin(x * 0.5)
    cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
    cz, sz = np.cos(z * 0.5), np.sin(z * 0.5)

    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ], dtype=np.float32)
