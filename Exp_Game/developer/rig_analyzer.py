# Exp_Game/developer/rig_analyzer.py
"""
Rig Analyzer - UI and data extraction for rig calibration.

This module handles:
- Blender UI operators for rig analysis
- Data extraction from armature (requires bpy)
- Calls engine module for actual analysis (worker-safe)

The analysis math is in engine/animations/rig_calibration.py so it can be:
- Reused by IK solver in worker
- Used for pose validation
- Used for constrained interpolation

Data Flow:
1. User clicks "Analyze Rig" button
2. This module extracts raw bone data (matrices, positions) via bpy
3. Engine module analyzes the data (numpy-based, worker-safe)
4. Results cached here for UI display and sent to worker for IK
"""

import bpy
from typing import Dict, List, Any, Optional

from .dev_logger import log_game

# Import worker-safe analysis from engine
from ..engine.animations.rig_calibration import (
    analyze_rig_data,
    get_chain_calibration,
    generate_report,
    BoneCalibration,
    IK_CHAINS,
)


# =============================================================================
# LOCAL CALIBRATION CACHE (Main Thread)
# =============================================================================

_rig_calibration: Dict[str, BoneCalibration] = {}


# =============================================================================
# DATA EXTRACTION (Main Thread - requires bpy)
# =============================================================================

def extract_bone_data(armature: bpy.types.Object) -> List[Dict[str, Any]]:
    """
    Extract raw bone data from armature.

    This is the ONLY function that needs bpy access.
    Converts Blender Bone data into plain dicts/arrays for analysis.

    Args:
        armature: The armature object

    Returns:
        List of bone data dicts (serializable, worker-safe)
    """
    arm_data = armature.data
    bone_data = []

    for bone in arm_data.bones:
        # Extract matrix as nested list (4x4)
        matrix = bone.matrix_local
        matrix_list = [
            [matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]],
            [matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]],
            [matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]],
            [matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]],
        ]

        bd = {
            'name': bone.name,
            'length': bone.length,
            'matrix_local': matrix_list,
            'head_local': list(bone.head_local),
            'tail_local': list(bone.tail_local),
            'parent': bone.parent.name if bone.parent else None,
            'children': [c.name for c in bone.children],
        }
        bone_data.append(bd)

    return bone_data


# =============================================================================
# PUBLIC API
# =============================================================================

def analyze_rig(armature: bpy.types.Object) -> Dict[str, BoneCalibration]:
    """
    Analyze an armature and calibrate bone orientations.

    1. Extracts raw data (main thread, bpy)
    2. Analyzes data (engine module, worker-safe)
    3. Caches results

    Args:
        armature: The armature object to analyze

    Returns:
        Dict mapping bone names to BoneCalibration
    """
    global _rig_calibration

    if not armature or armature.type != 'ARMATURE':
        log_game("RIG-ANALYZE", "ERROR: Not an armature object")
        return {}

    log_game("RIG-ANALYZE", f"Analyzing rig: {armature.name}")

    # Step 1: Extract raw data (requires bpy)
    bone_data = extract_bone_data(armature)
    log_game("RIG-ANALYZE", f"Extracted {len(bone_data)} bones")

    # Step 2: Analyze data (worker-safe engine module)
    _rig_calibration = analyze_rig_data(bone_data)
    log_game("RIG-ANALYZE", f"Analysis complete: {len(_rig_calibration)} bones calibrated")

    return _rig_calibration


def get_calibration(bone_name: str = None) -> Any:
    """
    Get calibration data for a bone or all bones.

    Args:
        bone_name: Name of bone, or None for all bones

    Returns:
        BoneCalibration for bone, or full calibration dict
    """
    if bone_name:
        return _rig_calibration.get(bone_name)
    return _rig_calibration.copy()


def get_calibration_for_chain(chain: str) -> List[BoneCalibration]:
    """
    Get calibration for an IK chain.

    Args:
        chain: Chain name ("arm_L", "arm_R", "leg_L", "leg_R")

    Returns:
        List of BoneCalibration for bones in the chain
    """
    return get_chain_calibration(_rig_calibration, chain)


def is_calibrated() -> bool:
    """Check if rig has been calibrated."""
    return len(_rig_calibration) > 0


def get_calibration_report() -> str:
    """Generate calibration report."""
    if not _rig_calibration:
        return "No rig calibration data. Run analyze_rig() first."
    return generate_report(_rig_calibration)


def get_serializable_calibration() -> Dict[str, Dict[str, Any]]:
    """
    Get calibration as serializable dicts (for sending to worker).

    Returns:
        Dict of bone name to calibration dict
    """
    return {name: calib.to_dict() for name, calib in _rig_calibration.items()}


# =============================================================================
# BLENDER OPERATORS
# =============================================================================

class RIG_OT_analyze(bpy.types.Operator):
    """Analyze the target armature's bone orientations"""
    bl_idname = "rig.analyze"
    bl_label = "Analyze Rig"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        armature = getattr(context.scene, 'target_armature', None)
        return armature is not None and armature.type == 'ARMATURE'

    def execute(self, context):
        armature = context.scene.target_armature

        # Analyze
        analyze_rig(armature)

        # Generate and store report
        report = get_calibration_report()
        context.scene.rig_analysis_report = report

        self.report({'INFO'}, f"Analyzed {len(_rig_calibration)} bones")

        return {'FINISHED'}


class RIG_OT_copy_report(bpy.types.Operator):
    """Copy rig analysis report to clipboard"""
    bl_idname = "rig.copy_report"
    bl_label = "Copy Report"

    def execute(self, context):
        report = getattr(context.scene, 'rig_analysis_report', '')
        if report:
            context.window_manager.clipboard = report
            self.report({'INFO'}, "Report copied to clipboard")
        else:
            self.report({'WARNING'}, "No report to copy - run analysis first")
        return {'FINISHED'}


# =============================================================================
# REGISTRATION
# =============================================================================

def register():
    bpy.utils.register_class(RIG_OT_analyze)
    bpy.utils.register_class(RIG_OT_copy_report)


def unregister():
    bpy.utils.unregister_class(RIG_OT_copy_report)
    bpy.utils.unregister_class(RIG_OT_analyze)
