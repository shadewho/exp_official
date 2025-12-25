# engine/animations/rig_calibration.py
"""
Rig Calibration - Worker-safe bone orientation analysis.

This module contains the analysis math for rig calibration.
NO BPY IMPORTS - this runs in worker processes.

The calibration data is used by:
- IK solver (to know correct bend axes and limits)
- Pose validation (to check anatomical constraints)
- Constrained interpolation (to enforce limits during blending)

Data Flow:
1. Main thread extracts raw bone data (matrices, positions) using bpy
2. This module analyzes the data (numpy-based math)
3. Results are cached and sent to worker for IK solving
"""

import numpy as np
import math
from typing import Dict, List, Optional, Any, Tuple


# =============================================================================
# ANATOMICAL JOINT DEFINITIONS
# =============================================================================

# Expected joint characteristics for a humanoid rig
ANATOMICAL_JOINTS: Dict[str, Dict[str, Any]] = {
    # Legs - knees bend on local X axis (flexion/extension)
    "LeftThigh": {"type": "ball", "primary_bend": "X", "expected_child": "LeftShin"},
    "LeftShin": {"type": "hinge", "primary_bend": "X", "expected_child": "LeftFoot", "limits": (0, 150)},
    "LeftFoot": {"type": "hinge", "primary_bend": "X", "expected_child": "LeftToeBase"},
    "RightThigh": {"type": "ball", "primary_bend": "X", "expected_child": "RightShin"},
    "RightShin": {"type": "hinge", "primary_bend": "X", "expected_child": "RightFoot", "limits": (0, 150)},
    "RightFoot": {"type": "hinge", "primary_bend": "X", "expected_child": "RightToeBase"},

    # Arms - elbows bend on local X axis
    "LeftArm": {"type": "ball", "primary_bend": "X", "expected_child": "LeftForeArm"},
    "LeftForeArm": {"type": "hinge", "primary_bend": "X", "expected_child": "LeftHand", "limits": (0, 145)},
    "LeftHand": {"type": "ball", "primary_bend": "X", "expected_child": None},
    "RightArm": {"type": "ball", "primary_bend": "X", "expected_child": "RightForeArm"},
    "RightForeArm": {"type": "hinge", "primary_bend": "X", "expected_child": "RightHand", "limits": (0, 145)},
    "RightHand": {"type": "ball", "primary_bend": "X", "expected_child": None},

    # Shoulders
    "LeftShoulder": {"type": "hinge", "primary_bend": "Z", "expected_child": "LeftArm"},
    "RightShoulder": {"type": "hinge", "primary_bend": "Z", "expected_child": "RightArm"},

    # Spine
    "Spine": {"type": "limited_ball", "primary_bend": "X", "expected_child": "Spine1"},
    "Spine1": {"type": "limited_ball", "primary_bend": "X", "expected_child": "Spine2"},
    "Spine2": {"type": "limited_ball", "primary_bend": "X", "expected_child": "NeckLower"},

    # Neck/Head
    "NeckLower": {"type": "limited_ball", "primary_bend": "X", "expected_child": "NeckUpper"},
    "NeckUpper": {"type": "limited_ball", "primary_bend": "X", "expected_child": "Head"},
    "Head": {"type": "limited_ball", "primary_bend": "X", "expected_child": None},

    # Root
    "Hips": {"type": "root", "primary_bend": None, "expected_child": "Spine"},
}

# IK chain definitions
IK_CHAINS: Dict[str, List[str]] = {
    "arm_L": ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"],
    "arm_R": ["RightShoulder", "RightArm", "RightForeArm", "RightHand"],
    "leg_L": ["LeftThigh", "LeftShin", "LeftFoot"],
    "leg_R": ["RightThigh", "RightShin", "RightFoot"],
}


# =============================================================================
# CALIBRATION DATA STRUCTURE
# =============================================================================

class BoneCalibration:
    """
    Calibration data for a single bone.
    Worker-safe - no bpy references.
    """
    __slots__ = (
        'name', 'length', 'roll', 'roll_deg',
        'local_x', 'local_y', 'local_z',
        'bone_axis', 'up_axis', 'side_axis',
        'head', 'tail',
        'parent', 'children', 'parent_angle_deg',
        'joint_type', 'expected_bend_axis', 'expected_child',
        'child_matches', 'anatomical_limits', 'detected_bend_axis'
    )

    def __init__(self):
        self.name: str = ""
        self.length: float = 0.0
        self.roll: float = 0.0
        self.roll_deg: float = 0.0
        self.local_x: np.ndarray = np.zeros(3, dtype=np.float32)
        self.local_y: np.ndarray = np.zeros(3, dtype=np.float32)
        self.local_z: np.ndarray = np.zeros(3, dtype=np.float32)
        self.bone_axis: str = "+Y"
        self.up_axis: str = "+Z"
        self.side_axis: str = "+X"
        self.head: np.ndarray = np.zeros(3, dtype=np.float32)
        self.tail: np.ndarray = np.zeros(3, dtype=np.float32)
        self.parent: Optional[str] = None
        self.children: List[str] = []
        self.parent_angle_deg: float = 0.0
        self.joint_type: str = "unknown"
        self.expected_bend_axis: Optional[str] = None
        self.expected_child: Optional[str] = None
        self.child_matches: bool = True
        self.anatomical_limits: Optional[Tuple[float, float]] = None
        self.detected_bend_axis: str = "X"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'length': self.length,
            'roll': self.roll,
            'roll_deg': self.roll_deg,
            'local_x': tuple(self.local_x),
            'local_y': tuple(self.local_y),
            'local_z': tuple(self.local_z),
            'bone_axis': self.bone_axis,
            'up_axis': self.up_axis,
            'side_axis': self.side_axis,
            'head': tuple(self.head),
            'tail': tuple(self.tail),
            'parent': self.parent,
            'children': self.children,
            'parent_angle_deg': self.parent_angle_deg,
            'joint_type': self.joint_type,
            'expected_bend_axis': self.expected_bend_axis,
            'expected_child': self.expected_child,
            'child_matches': self.child_matches,
            'anatomical_limits': self.anatomical_limits,
            'detected_bend_axis': self.detected_bend_axis,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'BoneCalibration':
        """Create from dictionary."""
        bc = BoneCalibration()
        bc.name = data['name']
        bc.length = data['length']
        bc.roll = data['roll']
        bc.roll_deg = data['roll_deg']
        bc.local_x = np.array(data['local_x'], dtype=np.float32)
        bc.local_y = np.array(data['local_y'], dtype=np.float32)
        bc.local_z = np.array(data['local_z'], dtype=np.float32)
        bc.bone_axis = data['bone_axis']
        bc.up_axis = data['up_axis']
        bc.side_axis = data['side_axis']
        bc.head = np.array(data['head'], dtype=np.float32)
        bc.tail = np.array(data['tail'], dtype=np.float32)
        bc.parent = data['parent']
        bc.children = data['children']
        bc.parent_angle_deg = data['parent_angle_deg']
        bc.joint_type = data['joint_type']
        bc.expected_bend_axis = data['expected_bend_axis']
        bc.expected_child = data['expected_child']
        bc.child_matches = data['child_matches']
        bc.anatomical_limits = data['anatomical_limits']
        bc.detected_bend_axis = data['detected_bend_axis']
        return bc


# =============================================================================
# ANALYSIS FUNCTIONS (Worker-Safe)
# =============================================================================

def compute_bone_roll(bone_y: np.ndarray, bone_z: np.ndarray) -> float:
    """
    Compute bone roll from its Y and Z axes.

    Roll is the rotation around the bone's Y axis.
    We measure the angle of bone_z relative to a reference "up".

    Args:
        bone_y: Bone direction (normalized)
        bone_z: Bone up direction (normalized)

    Returns:
        Roll angle in radians
    """
    # Reference up is world Z projected perpendicular to bone Y
    world_z = np.array([0, 0, 1], dtype=np.float32)

    # Project world Z onto plane perpendicular to bone Y
    ref_up = world_z - bone_y * np.dot(world_z, bone_y)
    ref_len = np.linalg.norm(ref_up)

    if ref_len < 0.001:
        # Bone pointing straight up/down, use world Y
        world_y = np.array([0, 1, 0], dtype=np.float32)
        ref_up = world_y - bone_y * np.dot(world_y, bone_y)
        ref_len = np.linalg.norm(ref_up)

    if ref_len < 0.001:
        return 0.0

    ref_up = ref_up / ref_len

    # Compute signed angle between ref_up and bone_z around bone_y
    cos_angle = np.clip(np.dot(ref_up, bone_z), -1.0, 1.0)
    cross = np.cross(ref_up, bone_z)
    sin_angle = np.dot(cross, bone_y)

    return math.atan2(sin_angle, cos_angle)


def classify_axis(axis: np.ndarray) -> str:
    """
    Classify which world axis a vector most closely aligns with.

    Args:
        axis: Normalized direction vector

    Returns:
        String like "+Y", "-Z", etc.
    """
    abs_vals = np.abs(axis)
    max_idx = np.argmax(abs_vals)

    labels = ['X', 'Y', 'Z']
    sign = '+' if axis[max_idx] > 0 else '-'
    return f"{sign}{labels[max_idx]}"


def detect_bend_axis(bone_name: str) -> str:
    """
    Detect the primary bend axis for a bone based on its name.

    For human joints:
    - Elbows/knees bend perpendicular to bone direction (local X)
    - Spine bends forward/back (local X) and side-to-side (local Z)

    Args:
        bone_name: Name of the bone

    Returns:
        "X", "Y", or "Z" indicating the local axis for primary bending
    """
    name_lower = bone_name.lower()

    if any(part in name_lower for part in ["arm", "forearm", "thigh", "shin", "leg"]):
        return "X"  # Limb bones bend on X
    elif any(part in name_lower for part in ["spine", "neck"]):
        return "X"  # Spine bends on X (forward/back)
    elif "head" in name_lower:
        return "X"  # Head nods on X
    else:
        return "X"  # Default


def analyze_bone_data(
    bone_name: str,
    matrix_local: np.ndarray,  # 4x4 matrix
    length: float,
    head_local: np.ndarray,
    tail_local: np.ndarray,
    parent_name: Optional[str],
    parent_matrix: Optional[np.ndarray],
    children: List[str]
) -> BoneCalibration:
    """
    Analyze raw bone data and produce calibration.

    This is the core analysis function - worker-safe, no bpy.

    Args:
        bone_name: Name of the bone
        matrix_local: 4x4 bone matrix in armature space
        length: Bone length
        head_local: Head position in armature space
        tail_local: Tail position in armature space
        parent_name: Name of parent bone (or None)
        parent_matrix: Parent's 4x4 matrix (or None)
        children: List of child bone names

    Returns:
        BoneCalibration with all analysis results
    """
    calib = BoneCalibration()
    calib.name = bone_name
    calib.length = length
    calib.children = children
    calib.parent = parent_name

    # Extract local axes from matrix (columns of rotation part)
    # In Blender bone space: Y points along bone, Z is up, X is side
    calib.local_x = np.array([matrix_local[0, 0], matrix_local[1, 0], matrix_local[2, 0]], dtype=np.float32)
    calib.local_y = np.array([matrix_local[0, 1], matrix_local[1, 1], matrix_local[2, 1]], dtype=np.float32)
    calib.local_z = np.array([matrix_local[0, 2], matrix_local[1, 2], matrix_local[2, 2]], dtype=np.float32)

    # Normalize
    calib.local_x = calib.local_x / (np.linalg.norm(calib.local_x) + 1e-10)
    calib.local_y = calib.local_y / (np.linalg.norm(calib.local_y) + 1e-10)
    calib.local_z = calib.local_z / (np.linalg.norm(calib.local_z) + 1e-10)

    # Compute roll
    calib.roll = compute_bone_roll(calib.local_y, calib.local_z)
    calib.roll_deg = math.degrees(calib.roll)

    # Classify axes
    calib.bone_axis = classify_axis(calib.local_y)
    calib.up_axis = classify_axis(calib.local_z)
    calib.side_axis = classify_axis(calib.local_x)

    # Store positions
    calib.head = np.array(head_local, dtype=np.float32)
    calib.tail = np.array(tail_local, dtype=np.float32)

    # Compute angle to parent
    if parent_matrix is not None:
        parent_y = np.array([parent_matrix[0, 1], parent_matrix[1, 1], parent_matrix[2, 1]], dtype=np.float32)
        parent_y = parent_y / (np.linalg.norm(parent_y) + 1e-10)
        dot = np.clip(np.dot(calib.local_y, parent_y), -1.0, 1.0)
        calib.parent_angle_deg = math.degrees(math.acos(dot))
    else:
        calib.parent_angle_deg = 0.0

    # Anatomical info
    if bone_name in ANATOMICAL_JOINTS:
        anat = ANATOMICAL_JOINTS[bone_name]
        calib.joint_type = anat["type"]
        calib.expected_bend_axis = anat["primary_bend"]
        calib.expected_child = anat["expected_child"]

        if anat["expected_child"]:
            calib.child_matches = anat["expected_child"] in children
        else:
            calib.child_matches = True

        if "limits" in anat:
            calib.anatomical_limits = anat["limits"]
    else:
        calib.joint_type = "unknown"
        calib.expected_bend_axis = None
        calib.child_matches = True

    # Detect actual bend axis
    calib.detected_bend_axis = detect_bend_axis(bone_name)

    return calib


def analyze_rig_data(
    bone_data: List[Dict[str, Any]]
) -> Dict[str, BoneCalibration]:
    """
    Analyze all bones in a rig.

    Args:
        bone_data: List of dicts with raw bone data extracted from bpy

    Returns:
        Dict mapping bone names to BoneCalibration
    """
    # First pass: create calibrations without parent angle
    calibrations: Dict[str, BoneCalibration] = {}

    # Build parent matrix lookup
    matrices: Dict[str, np.ndarray] = {}
    for bd in bone_data:
        matrices[bd['name']] = np.array(bd['matrix_local'], dtype=np.float32)

    # Analyze each bone
    for bd in bone_data:
        parent_matrix = None
        if bd['parent']:
            parent_matrix = matrices.get(bd['parent'])

        calib = analyze_bone_data(
            bone_name=bd['name'],
            matrix_local=np.array(bd['matrix_local'], dtype=np.float32),
            length=bd['length'],
            head_local=bd['head_local'],
            tail_local=bd['tail_local'],
            parent_name=bd['parent'],
            parent_matrix=parent_matrix,
            children=bd['children']
        )
        calibrations[bd['name']] = calib

    return calibrations


def get_chain_calibration(
    calibrations: Dict[str, BoneCalibration],
    chain: str
) -> List[BoneCalibration]:
    """
    Get calibration for an IK chain.

    Args:
        calibrations: Full rig calibration
        chain: Chain name ("arm_L", "arm_R", "leg_L", "leg_R")

    Returns:
        List of BoneCalibration for bones in the chain
    """
    bones = IK_CHAINS.get(chain, [])
    return [calibrations[b] for b in bones if b in calibrations]


def generate_report(calibrations: Dict[str, BoneCalibration]) -> str:
    """
    Generate a human-readable calibration report.

    Args:
        calibrations: Dict of bone name to BoneCalibration

    Returns:
        Multi-line string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("RIG CALIBRATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total bones analyzed: {len(calibrations)}")
    lines.append("")

    # IK Chain Analysis
    lines.append("-" * 40)
    lines.append("IK CHAINS")
    lines.append("-" * 40)

    for chain_name in ["arm_L", "arm_R", "leg_L", "leg_R"]:
        chain_data = get_chain_calibration(calibrations, chain_name)
        if chain_data:
            lines.append(f"\n{chain_name.upper()}:")
            for calib in chain_data:
                lines.append(f"  {calib.name}:")
                lines.append(f"    bone_axis: {calib.bone_axis}")
                lines.append(f"    bend_axis: {calib.detected_bend_axis}")
                lines.append(f"    roll: {calib.roll_deg:.1f} deg")
                lines.append(f"    length: {calib.length:.4f}m")
                if calib.anatomical_limits:
                    lim = calib.anatomical_limits
                    lines.append(f"    limits: {lim[0]}deg - {lim[1]}deg")

    lines.append("")
    lines.append("-" * 40)
    lines.append("POTENTIAL ISSUES")
    lines.append("-" * 40)

    issues = []
    for name, calib in calibrations.items():
        if not calib.child_matches:
            issues.append(f"  {name}: Expected child '{calib.expected_child}' not found")

        roll = abs(calib.roll_deg)
        if roll > 45 and roll < 135:
            issues.append(f"  {name}: Unusual roll ({calib.roll_deg:.1f} deg)")

    if issues:
        lines.extend(issues)
    else:
        lines.append("  No issues detected")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# =============================================================================
# CALIBRATION CACHE (Worker-side)
# =============================================================================

# Global cache for worker to access during IK solving
_worker_calibration: Dict[str, BoneCalibration] = {}


def set_worker_calibration(calibrations: Dict[str, Dict[str, Any]]) -> None:
    """
    Set calibration data in worker (called via CACHE_RIG_CALIBRATION job).

    Args:
        calibrations: Dict of bone name to calibration dict (serialized)
    """
    global _worker_calibration
    _worker_calibration.clear()

    for name, data in calibrations.items():
        _worker_calibration[name] = BoneCalibration.from_dict(data)


def get_worker_calibration(bone_name: str = None) -> Any:
    """
    Get calibration data in worker.

    Args:
        bone_name: Specific bone name, or None for all

    Returns:
        BoneCalibration or full dict
    """
    if bone_name:
        return _worker_calibration.get(bone_name)
    return _worker_calibration


def is_worker_calibrated() -> bool:
    """Check if worker has calibration data."""
    return len(_worker_calibration) > 0
