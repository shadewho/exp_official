# Exp_Game/animations/rig_probe.py
"""
Rig Probe - Diagnostic tool for verifying rig data.

ONE JOB: Dump actual bone data from Blender, compare to expectations, report mismatches.

This is the foundation for debugging animation/IK issues. If rig assumptions
are wrong, everything built on top will be wrong.

Output goes to: C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt

Usage:
    1. Select an armature
    2. Run "Probe Rig" operator from Developer Tools panel
    3. Check diagnostics_latest.txt for full report
"""

import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Vector

from ..developer.dev_logger import log_game, start_session, export_game_log, clear_log
from ..engine.animations.ik_chains import LEG_IK, ARM_IK


# Expected bone positions from rig.md (key bones only)
EXPECTED_POSITIONS = {
    "Hips": (0.0, 0.056, 1.001),
    "Head": (0.0, 0.047, 1.701),
    "LeftHand": (-0.700, 0.044, 1.557),
    "RightHand": (0.700, 0.045, 1.557),
    "LeftFoot": (-0.110, -0.016, 0.098),
    "RightFoot": (0.110, -0.010, 0.098),
    "LeftArm": (-0.136, 0.047, 1.570),
    "RightArm": (0.136, 0.048, 1.570),
    "LeftThigh": (-0.075, 0.054, 1.065),
    "RightThigh": (0.075, 0.054, 1.065),
}

# Expected bone lengths from rig.md
EXPECTED_LENGTHS = {
    "Hips": 0.136,
    "Spine": 0.145,
    "Spine1": 0.191,
    "Spine2": 0.149,
    "Head": 0.272,
    "LeftArm": 0.278,
    "LeftForeArm": 0.286,
    "RightArm": 0.278,
    "RightForeArm": 0.286,
    "LeftThigh": 0.495,
    "LeftShin": 0.478,
    "RightThigh": 0.495,
    "RightShin": 0.478,
}


def probe_rig(armature) -> dict:
    """
    Probe an armature and return diagnostic data.

    Args:
        armature: Blender armature object

    Returns:
        Dict with probe results
    """
    results = {
        "armature_name": armature.name,
        "bone_count": len(armature.data.bones),
        "bones": {},
        "ik_chains": {},
        "position_errors": [],  # Raw position differences
        "length_errors": [],    # Bone length mismatches (real issues)
        "reach_errors": [],     # IK reach mismatches (real issues)
        "warnings": [],
        "world_offset": None,   # Detected consistent offset
    }

    arm_matrix = armature.matrix_world

    # Probe each bone
    for bone in armature.data.bones:
        head_local = Vector(bone.head_local)
        tail_local = Vector(bone.tail_local)
        head_world = arm_matrix @ head_local
        tail_world = arm_matrix @ tail_local

        direction = tail_local - head_local
        length = direction.length
        rest_dir = direction.normalized() if length > 0.0001 else Vector((0, 0, 1))

        bone_data = {
            "head_local": tuple(head_local),
            "tail_local": tuple(tail_local),
            "head_world": tuple(head_world),
            "tail_world": tuple(tail_world),
            "length": length,
            "rest_direction": tuple(rest_dir),
            "parent": bone.parent.name if bone.parent else None,
        }
        results["bones"][bone.name] = bone_data

        # Track position differences (we'll analyze for offset later)
        if bone.name in EXPECTED_POSITIONS:
            expected = EXPECTED_POSITIONS[bone.name]
            actual = tuple(head_world)
            diff = (actual[0] - expected[0], actual[1] - expected[1], actual[2] - expected[2])
            results["position_errors"].append({
                "bone": bone.name,
                "expected": expected,
                "actual": actual,
                "diff": diff,
            })

        # Check bone lengths (these are real structural issues)
        if bone.name in EXPECTED_LENGTHS:
            expected_len = EXPECTED_LENGTHS[bone.name]
            if abs(length - expected_len) > 0.01:
                results["length_errors"].append({
                    "bone": bone.name,
                    "expected": expected_len,
                    "actual": length,
                    "error": abs(length - expected_len),
                })

    # Detect consistent world offset
    if results["position_errors"]:
        diffs = [e["diff"] for e in results["position_errors"]]
        avg_x = sum(d[0] for d in diffs) / len(diffs)
        avg_y = sum(d[1] for d in diffs) / len(diffs)
        avg_z = sum(d[2] for d in diffs) / len(diffs)

        # Check if all bones have similar offset (consistent world position shift)
        variance = sum(
            (d[0] - avg_x)**2 + (d[1] - avg_y)**2 + (d[2] - avg_z)**2
            for d in diffs
        ) / len(diffs)

        if variance < 0.001:  # Low variance = consistent offset
            offset_mag = (avg_x**2 + avg_y**2 + avg_z**2) ** 0.5
            results["world_offset"] = {
                "x": avg_x,
                "y": avg_y,
                "z": avg_z,
                "magnitude": offset_mag,
            }

    # Probe IK chains
    all_chains = {**LEG_IK, **ARM_IK}
    for chain_name, chain_def in all_chains.items():
        root_name = chain_def["root"]
        mid_name = chain_def["mid"]
        tip_name = chain_def["tip"]

        root_bone = results["bones"].get(root_name)
        mid_bone = results["bones"].get(mid_name)
        tip_bone = results["bones"].get(tip_name)

        if not all([root_bone, mid_bone, tip_bone]):
            results["warnings"].append(f"IK chain {chain_name}: missing bones")
            continue

        upper_len = root_bone["length"]
        lower_len = mid_bone["length"]
        actual_reach = upper_len + lower_len
        expected_reach = chain_def["reach"]

        chain_data = {
            "root": root_name,
            "mid": mid_name,
            "tip": tip_name,
            "upper_length": upper_len,
            "lower_length": lower_len,
            "actual_reach": actual_reach,
            "expected_reach": expected_reach,
            "reach_error": abs(actual_reach - expected_reach),
        }
        results["ik_chains"][chain_name] = chain_data

        if abs(actual_reach - expected_reach) > 0.01:
            results["reach_errors"].append({
                "chain": chain_name,
                "expected": expected_reach,
                "actual": actual_reach,
                "error": abs(actual_reach - expected_reach),
            })

    return results


def log_probe_report(results: dict):
    """Log probe results using the dev_logger system."""
    log_game("RIG-PROBE", "=" * 60)
    log_game("RIG-PROBE", f"Armature: {results['armature_name']} | Bones: {results['bone_count']}")
    log_game("RIG-PROBE", "=" * 60)

    # World offset detection (this is NOT an error)
    offset = results.get("world_offset")
    if offset:
        log_game("RIG-PROBE", f"WORLD_OFFSET detected: ({offset['x']:+.3f}, {offset['y']:+.3f}, {offset['z']:+.3f}) = {offset['magnitude']:.3f}m")
        log_game("RIG-PROBE", "  (This is normal - character world position differs from reference)")
    else:
        log_game("RIG-PROBE", "WORLD_OFFSET: none detected")

    # IK Chains (the important stuff)
    log_game("RIG-PROBE", "-" * 60)
    log_game("RIG-PROBE", "IK CHAINS:")
    for chain_name, chain_data in results["ik_chains"].items():
        status = "OK" if chain_data["reach_error"] < 0.01 else "MISMATCH"
        log_game("RIG-PROBE", f"  {chain_name}: upper={chain_data['upper_length']:.3f}m lower={chain_data['lower_length']:.3f}m reach={chain_data['actual_reach']:.3f}m [{status}]")

    # Bone rest directions (critical for IK)
    log_game("RIG-PROBE", "-" * 60)
    log_game("RIG-PROBE", "BONE REST DIRECTIONS:")
    ik_bones = ["LeftArm", "LeftForeArm", "RightArm", "RightForeArm",
                "LeftThigh", "LeftShin", "RightThigh", "RightShin"]
    for bone_name in ik_bones:
        bone_data = results["bones"].get(bone_name)
        if bone_data:
            d = bone_data["rest_direction"]
            if abs(d[0]) > 0.9:
                primary = "LEFT" if d[0] < 0 else "RIGHT"
            elif abs(d[1]) > 0.9:
                primary = "FORWARD" if d[1] > 0 else "BACK"
            elif abs(d[2]) > 0.9:
                primary = "UP" if d[2] > 0 else "DOWN"
            else:
                primary = "MIXED"
            log_game("RIG-PROBE", f"  {bone_name:16} ({d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f}) -> {primary}")

    # Real issues (bone lengths, IK reach)
    has_issues = results["length_errors"] or results["reach_errors"]

    if has_issues:
        log_game("RIG-PROBE", "-" * 60)
        log_game("RIG-PROBE", "STRUCTURAL ISSUES (FIX THESE!):")
        for e in results["length_errors"]:
            log_game("RIG-PROBE", f"  LENGTH {e['bone']}: expected {e['expected']:.3f}m, got {e['actual']:.3f}m")
        for e in results["reach_errors"]:
            log_game("RIG-PROBE", f"  IK_REACH {e['chain']}: expected {e['expected']:.3f}m, got {e['actual']:.3f}m")
    else:
        log_game("RIG-PROBE", "-" * 60)
        log_game("RIG-PROBE", "RESULT: RIG STRUCTURE OK - IK chains and bone lengths verified")

    log_game("RIG-PROBE", "=" * 60)


def describe_direction(vec) -> str:
    """Describe a vector direction in human terms."""
    x, y, z = vec[0], vec[1], vec[2]

    # Find dominant axis
    ax = abs(x)
    ay = abs(y)
    az = abs(z)

    parts = []

    # Primary direction (largest component)
    if ax > 0.5:
        parts.append("RIGHT" if x > 0 else "LEFT")
    if ay > 0.5:
        parts.append("FORWARD" if y > 0 else "BACK")
    if az > 0.5:
        parts.append("UP" if z > 0 else "DOWN")

    if not parts:
        # Mixed direction, show all significant components
        if ax > 0.3:
            parts.append("right" if x > 0 else "left")
        if ay > 0.3:
            parts.append("fwd" if y > 0 else "back")
        if az > 0.3:
            parts.append("up" if z > 0 else "down")

    return "+".join(parts) if parts else "MIXED"


def dump_bone_orientations(armature) -> str:
    """
    Extract local axis orientations for all bones.
    Returns markdown-formatted text for rig.md.
    """
    lines = []
    lines.append("## Bone Local Axis Orientations")
    lines.append("")
    lines.append("Each bone's local coordinate system in rest pose (T-pose).")
    lines.append("- **Y-axis**: Typically points along the bone (head → tail)")
    lines.append("- **X-axis**: Perpendicular, often the 'twist' axis")
    lines.append("- **Z-axis**: Perpendicular, often the 'bend' axis")
    lines.append("")
    lines.append("| Bone | +X Points | +Y Points | +Z Points | X Vector | Y Vector | Z Vector |")
    lines.append("|------|-----------|-----------|-----------|----------|----------|----------|")

    # Sort bones for consistent output
    bone_names = sorted([b.name for b in armature.data.bones])

    for bone_name in bone_names:
        bone = armature.data.bones[bone_name]

        # Get bone's rest matrix (local to armature space)
        m = bone.matrix_local

        # Extract local axes (columns of rotation matrix)
        local_x = (m[0][0], m[1][0], m[2][0])
        local_y = (m[0][1], m[1][1], m[2][1])
        local_z = (m[0][2], m[1][2], m[2][2])

        x_desc = describe_direction(local_x)
        y_desc = describe_direction(local_y)
        z_desc = describe_direction(local_z)

        # Format vectors
        x_vec = f"({local_x[0]:+.2f}, {local_x[1]:+.2f}, {local_x[2]:+.2f})"
        y_vec = f"({local_y[0]:+.2f}, {local_y[1]:+.2f}, {local_y[2]:+.2f})"
        z_vec = f"({local_z[0]:+.2f}, {local_z[1]:+.2f}, {local_z[2]:+.2f})"

        lines.append(f"| `{bone_name}` | {x_desc} | {y_desc} | {z_desc} | {x_vec} | {y_vec} | {z_vec} |")

    lines.append("")
    lines.append("### Key Bones for IK")
    lines.append("")

    # Highlight IK-relevant bones
    ik_bones = [
        ("RightArm", "Upper arm - shoulder to elbow"),
        ("RightForeArm", "Forearm - elbow to wrist"),
        ("RightHand", "Hand/wrist"),
        ("LeftArm", "Upper arm - shoulder to elbow"),
        ("LeftForeArm", "Forearm - elbow to wrist"),
        ("LeftHand", "Hand/wrist"),
        ("RightThigh", "Upper leg - hip to knee"),
        ("RightShin", "Lower leg - knee to ankle"),
        ("RightFoot", "Foot/ankle"),
        ("LeftThigh", "Upper leg - hip to knee"),
        ("LeftShin", "Lower leg - knee to ankle"),
        ("LeftFoot", "Foot/ankle"),
    ]

    lines.append("| Bone | Description | Y-Axis (Bone Direction) | Bend Axis |")
    lines.append("|------|-------------|------------------------|-----------|")

    for bone_name, desc in ik_bones:
        bone = armature.data.bones.get(bone_name)
        if bone:
            m = bone.matrix_local
            local_y = (m[0][1], m[1][1], m[2][1])
            local_z = (m[0][2], m[1][2], m[2][2])
            y_desc = describe_direction(local_y)
            z_desc = describe_direction(local_z)
            lines.append(f"| `{bone_name}` | {desc} | {y_desc} | Z: {z_desc} |")

    return "\n".join(lines)


class ANIM2_OT_DumpOrientations(Operator):
    """Dump all bone orientations to file for rig.md documentation"""
    bl_idname = "anim2.dump_orientations"
    bl_label = "Dump Bone Orientations"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        armature = getattr(context.scene, 'target_armature', None)
        if armature and armature.type == 'ARMATURE':
            return True
        obj = context.active_object
        return obj and obj.type == 'ARMATURE'

    def execute(self, context):
        # Get armature
        armature = getattr(context.scene, 'target_armature', None)
        if not armature or armature.type != 'ARMATURE':
            armature = context.active_object

        if not armature or armature.type != 'ARMATURE':
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        # Generate markdown
        markdown = dump_bone_orientations(armature)

        # Write to file
        output_path = "C:/Users/spenc/Desktop/engine_output_files/bone_orientations.md"
        with open(output_path, 'w') as f:
            f.write(markdown)

        self.report({'INFO'}, f"Bone orientations written to {output_path}")
        return {'FINISHED'}


class ANIM2_OT_ProbeRig(Operator):
    """Probe the selected armature and report rig data to diagnostics file"""
    bl_idname = "anim2.probe_rig"
    bl_label = "Probe Rig"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        armature = getattr(context.scene, 'target_armature', None)
        if armature and armature.type == 'ARMATURE':
            return True
        obj = context.active_object
        return obj and obj.type == 'ARMATURE'

    def execute(self, context):
        # Get armature
        armature = getattr(context.scene, 'target_armature', None)
        if not armature or armature.type != 'ARMATURE':
            armature = context.active_object

        if not armature or armature.type != 'ARMATURE':
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        # Start fresh log session
        start_session()

        # Run probe
        results = probe_rig(armature)

        # Log report
        log_probe_report(results)

        # Export to diagnostics file
        export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
        clear_log()

        # Check for real issues
        has_issues = results["length_errors"] or results["reach_errors"]
        offset = results.get("world_offset")

        if has_issues:
            issue_count = len(results["length_errors"]) + len(results["reach_errors"])
            self.report({'WARNING'}, f"Found {issue_count} structural issues - check diagnostics_latest.txt")
        elif offset:
            self.report({'INFO'}, f"Rig OK (world offset {offset['magnitude']:.2f}m detected - normal)")
        else:
            self.report({'INFO'}, f"Rig OK: {results['bone_count']} bones verified")

        return {'FINISHED'}


# Registration
def register():
    bpy.utils.register_class(ANIM2_OT_ProbeRig)
    bpy.utils.register_class(ANIM2_OT_DumpOrientations)


def unregister():
    bpy.utils.unregister_class(ANIM2_OT_DumpOrientations)
    bpy.utils.unregister_class(ANIM2_OT_ProbeRig)
