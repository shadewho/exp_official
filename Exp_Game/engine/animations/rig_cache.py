# Exp_Game/engine/animations/rig_cache.py
"""
Rig Data Cache - Extract and cache armature data for worker-based FK/IK.

This module handles caching armature rest pose data that workers need to:
1. Compute forward kinematics (bone world positions from quaternions)
2. Apply IK using the delta-based approach (point_bone_at_target)

MAIN THREAD: Extracts data from bpy armature, sends to worker once at game start.
WORKER: Uses cached data to compute FK/IK without any bpy access.

Data format (pickle-safe, no bpy objects):
{
    "armature_name": str,
    "bone_count": int,
    "bones": {
        "BoneName": {
            "rest_local": [16 floats],    # 4x4 matrix, column-major
            "rest_world": [16 floats],    # rest pose world matrix
            "parent": str or None,
            "length": float,
            "children": [str, ...],
        },
        ...
    },
    "root_bones": [str, ...],  # Bones with no parent
    "bone_order": [str, ...],  # Topologically sorted (parents before children)
}
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


# =============================================================================
# MAIN THREAD: Extract rig data from Blender armature
# =============================================================================

def extract_rig_data(armature_obj) -> dict:
    """
    Extract rig data from a Blender armature for worker-based FK/IK.

    MAIN THREAD ONLY - uses bpy.

    Args:
        armature_obj: Blender armature object (bpy.types.Object with armature data)

    Returns:
        Dict with all rig data needed for worker-side FK/IK (pickle-safe)
    """
    if armature_obj is None or armature_obj.type != 'ARMATURE':
        return {}

    armature = armature_obj.data
    bones_data = {}
    root_bones = []

    # Extract data from each bone
    for bone in armature.bones:
        # Rest pose local matrix (relative to parent, or to armature if no parent)
        rest_local = list(bone.matrix_local.transposed())  # Column-major for numpy
        rest_local_flat = [x for row in rest_local for x in row]

        # Rest pose world matrix (relative to armature)
        # For rest pose, bone.matrix_local IS the world-relative matrix
        # (world = armature @ matrix_local)
        rest_world_flat = rest_local_flat  # In rest pose, same as local

        # Parent info
        parent_name = bone.parent.name if bone.parent else None
        if parent_name is None:
            root_bones.append(bone.name)

        # Bone length
        length = bone.length

        # Children
        children = [child.name for child in bone.children]

        bones_data[bone.name] = {
            "rest_local": rest_local_flat,
            "rest_world": rest_world_flat,
            "parent": parent_name,
            "length": length,
            "children": children,
        }

    # Topological sort: parents before children
    bone_order = _topological_sort(bones_data, root_bones)

    return {
        "armature_name": armature_obj.name,
        "bone_count": len(bones_data),
        "bones": bones_data,
        "root_bones": root_bones,
        "bone_order": bone_order,
    }


def _topological_sort(bones_data: dict, root_bones: list) -> list:
    """
    Sort bones so parents come before children.
    Uses BFS from root bones.
    """
    order = []
    queue = list(root_bones)
    visited = set()

    while queue:
        bone_name = queue.pop(0)
        if bone_name in visited:
            continue
        visited.add(bone_name)
        order.append(bone_name)

        # Add children to queue
        bone_data = bones_data.get(bone_name, {})
        for child in bone_data.get("children", []):
            if child not in visited:
                queue.append(child)

    return order


# =============================================================================
# WORKER SIDE: Forward Kinematics
# =============================================================================

class RigFK:
    """
    Worker-side rig for forward kinematics computation.

    No bpy access - uses only cached numpy matrices.
    """

    __slots__ = (
        'bone_count', 'bone_order', 'root_bones',
        '_rest_local', '_rest_world', '_lengths', '_parent_indices',
        '_bone_to_idx', '_idx_to_bone', '_ready'
    )

    def __init__(self):
        self.bone_count = 0
        self.bone_order = []
        self.root_bones = []
        self._rest_local = {}    # {bone_name: 4x4 numpy matrix}
        self._rest_world = {}    # {bone_name: 4x4 numpy matrix}
        self._lengths = {}       # {bone_name: float}
        self._parent_indices = {}  # {bone_name: parent_name or None}
        self._bone_to_idx = {}
        self._idx_to_bone = {}
        self._ready = False

    def load(self, rig_data: dict) -> bool:
        """
        Load rig data from cache (received from main thread).

        Args:
            rig_data: Dict from extract_rig_data()

        Returns:
            True if loaded successfully
        """
        if not rig_data or "bones" not in rig_data:
            return False

        bones = rig_data["bones"]
        self.bone_count = rig_data.get("bone_count", len(bones))
        self.bone_order = rig_data.get("bone_order", list(bones.keys()))
        self.root_bones = rig_data.get("root_bones", [])

        # Build lookup tables
        for idx, bone_name in enumerate(self.bone_order):
            self._bone_to_idx[bone_name] = idx
            self._idx_to_bone[idx] = bone_name

        # Convert matrices to numpy
        for bone_name, bone_data in bones.items():
            # Rest local matrix (4x4)
            rest_local_flat = bone_data.get("rest_local", [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])
            self._rest_local[bone_name] = np.array(rest_local_flat, dtype=np.float32).reshape(4, 4).T

            # Rest world matrix
            rest_world_flat = bone_data.get("rest_world", rest_local_flat)
            self._rest_world[bone_name] = np.array(rest_world_flat, dtype=np.float32).reshape(4, 4).T

            # Length
            self._lengths[bone_name] = bone_data.get("length", 0.1)

            # Parent
            self._parent_indices[bone_name] = bone_data.get("parent")

        self._ready = True
        return True

    def is_ready(self) -> bool:
        return self._ready

    def compute_world_matrices(
        self,
        armature_world: np.ndarray,
        bone_quats: Dict[str, Tuple],
        bone_locs: Optional[Dict[str, Tuple]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute world matrices for all bones given pose quaternions.

        Forward kinematics: traverse from root to tips, accumulating transforms.

        Args:
            armature_world: 4x4 armature world matrix
            bone_quats: {bone_name: (qw, qx, qy, qz)}
            bone_locs: {bone_name: (lx, ly, lz)} or None for rest locations

        Returns:
            {bone_name: 4x4 world matrix}
        """
        if not self._ready:
            return {}

        world_matrices = {}

        for bone_name in self.bone_order:
            # Get pose transform
            quat = bone_quats.get(bone_name, (1, 0, 0, 0))
            loc = bone_locs.get(bone_name, (0, 0, 0)) if bone_locs else (0, 0, 0)

            # Build pose matrix (rotation + location)
            pose_matrix = _quat_loc_to_matrix(quat, loc)

            # Rest local matrix
            rest_local = self._rest_local.get(bone_name, np.eye(4, dtype=np.float32))

            # Get parent world matrix
            parent_name = self._parent_indices.get(bone_name)
            if parent_name and parent_name in world_matrices:
                parent_world = world_matrices[parent_name]
            else:
                # Root bone: parent is armature
                parent_world = armature_world

            # Bone world = parent_world @ rest_local @ pose
            # Note: rest_local already includes position relative to parent
            bone_world = parent_world @ rest_local @ pose_matrix

            world_matrices[bone_name] = bone_world

        return world_matrices

    def get_bone_head_tail(
        self,
        bone_name: str,
        world_matrices: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bone head and tail positions in world space.

        Args:
            bone_name: Bone to query
            world_matrices: Output from compute_world_matrices

        Returns:
            (head_pos, tail_pos) as 3D vectors
        """
        if bone_name not in world_matrices:
            return np.zeros(3), np.zeros(3)

        world_mat = world_matrices[bone_name]

        # Head is at the bone's origin (translation of matrix)
        head = world_mat[:3, 3].copy()

        # Tail is head + (bone Y-axis * length)
        # In Blender, bones point along their local Y-axis
        length = self._lengths.get(bone_name, 0.1)
        bone_y_axis = world_mat[:3, 1]  # Y column of rotation part
        tail = head + bone_y_axis * length

        return head, tail

    def get_bone_direction(
        self,
        bone_name: str,
        world_matrices: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Get normalized direction bone is pointing (head to tail).
        """
        world_mat = world_matrices.get(bone_name)
        if world_mat is None:
            return np.array([0, 1, 0], dtype=np.float32)

        # Bone Y-axis is the pointing direction
        return world_mat[:3, 1].copy()

    def get_rest_local(self, bone_name: str) -> np.ndarray:
        """Get bone's rest local matrix."""
        return self._rest_local.get(bone_name, np.eye(4, dtype=np.float32))

    def get_parent(self, bone_name: str) -> Optional[str]:
        """Get parent bone name, or None if root."""
        return self._parent_indices.get(bone_name)

    def get_length(self, bone_name: str) -> float:
        """Get bone length."""
        return self._lengths.get(bone_name, 0.1)


# =============================================================================
# WORKER SIDE: Delta-based IK (point_bone_at_target)
# =============================================================================

def compute_point_at_target_quat(
    bone_head: np.ndarray,
    bone_tail: np.ndarray,
    bone_world_rot: np.ndarray,
    target_world: np.ndarray,
    rest_local_rot: np.ndarray,
    parent_world_rot: Optional[np.ndarray]
) -> np.ndarray:
    """
    Compute local quaternion to point bone at target.

    Worker-safe version of point_bone_at_target() from test_panel.py.

    Args:
        bone_head: Bone head world position (3,)
        bone_tail: Bone tail world position (3,)
        bone_world_rot: Bone's current world rotation quaternion [w,x,y,z]
        target_world: Target world position (3,)
        rest_local_rot: Bone's rest local rotation quaternion [w,x,y,z]
        parent_world_rot: Parent's world rotation [w,x,y,z], or None if root

    Returns:
        Local quaternion [w,x,y,z] to apply to bone
    """
    # Current direction (where bone is pointing)
    current_dir = bone_tail - bone_head
    current_len = np.linalg.norm(current_dir)
    if current_len < 0.001:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    current_dir = current_dir / current_len

    # Desired direction (where we want it to point)
    desired_dir = target_world - bone_head
    desired_len = np.linalg.norm(desired_dir)
    if desired_len < 0.001:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    desired_dir = desired_dir / desired_len

    # World-space rotation delta from current to desired
    world_delta = _quat_from_two_vectors(current_dir, desired_dir)

    # Apply delta to current world rotation
    new_world_rot = _quat_multiply(world_delta, bone_world_rot)

    # Convert to local space
    # local = rest_inv @ parent_world_inv @ new_world
    rest_inv = _quat_conjugate(rest_local_rot)

    if parent_world_rot is not None:
        parent_inv = _quat_conjugate(parent_world_rot)
        local_rot = _quat_multiply(rest_inv, _quat_multiply(parent_inv, new_world_rot))
    else:
        # No parent - just undo rest orientation
        local_rot = _quat_multiply(rest_inv, new_world_rot)

    return local_rot


# =============================================================================
# NUMPY QUATERNION MATH (worker-safe)
# =============================================================================

def _quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Create quaternion that rotates v_from to v_to. Returns [w,x,y,z]."""
    v_from = v_from / (np.linalg.norm(v_from) + 1e-10)
    v_to = v_to / (np.linalg.norm(v_to) + 1e-10)

    d = np.dot(v_from, v_to)

    if d > 0.9999:
        return np.array([1, 0, 0, 0], dtype=np.float32)

    if d < -0.9999:
        # Opposite vectors - find orthogonal axis
        axis = np.cross(np.array([1, 0, 0]), v_from)
        if np.linalg.norm(axis) < 0.001:
            axis = np.cross(np.array([0, 1, 0]), v_from)
        axis = axis / np.linalg.norm(axis)
        return np.array([0, axis[0], axis[1], axis[2]], dtype=np.float32)

    axis = np.cross(v_from, v_to)
    s = np.sqrt((1 + d) * 2)
    inv_s = 1 / s

    return np.array([s * 0.5, axis[0] * inv_s, axis[1] * inv_s, axis[2] * inv_s], dtype=np.float32)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply quaternions q1 * q2. Format [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (inverse for unit quaternions). [w,x,y,z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def _quat_loc_to_matrix(quat: Tuple, loc: Tuple) -> np.ndarray:
    """Convert quaternion + location to 4x4 matrix."""
    w, x, y, z = quat

    # Rotation part
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    m = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy), loc[0]],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx), loc[1]],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy), loc[2]],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    return m


def _matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Extract quaternion [w,x,y,z] from 4x4 or 3x3 matrix."""
    # Ensure we're working with the rotation part
    r = m[:3, :3] if m.shape[0] == 4 else m

    trace = r[0, 0] + r[1, 1] + r[2, 2]

    if trace > 0:
        s = np.sqrt(trace + 1) * 2
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z], dtype=np.float32)
