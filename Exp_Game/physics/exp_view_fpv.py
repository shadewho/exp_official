# Exp_Game/physics/exp_view_fpv.py
# First-person: camera at FPV bone, bone rotation from camera.

import math
import bpy
from mathutils import Matrix, Vector


def update_first_person_camera(context, op):
    """
    First person camera update (called every frame from game loop).
      • Camera POSITION = capsule height (always stable)
      • Camera ROTATION = mouse pitch / yaw
      • FPV bone matrix_basis overridden to match camera
      • rv3d.view_distance = 0 (camera AT the bone)
    Falls back to capsule top if no FPV bone is set.
    """
    scene = context.scene
    if not op or not getattr(op, "target_object", None):
        return

    # ── Camera rotation from mouse pitch / yaw ────────────────────
    pitch = float(op.pitch)
    yaw   = float(op.yaw)
    cx = math.cos(pitch); sx = math.sin(pitch)
    sy = math.sin(yaw);   cy = math.cos(yaw)
    direction = Vector((cx * sy, -cx * cy, sx))
    if direction.length > 1e-9:
        direction.normalize()
    q_cam = direction.to_track_quat('Z', 'Y')

    # ── Resolve FPV bone ──────────────────────────────────────────
    arm = getattr(scene, "target_armature", None)
    bone_name = (getattr(scene, "fpv_view_bone", "") or "").strip()
    pb = None

    if arm and arm.type == 'ARMATURE' and bone_name:
        pb = arm.pose.bones.get(bone_name)

    # ── Camera position (capsule height — immune to stale bone matrices) ─
    cp = scene.char_physics
    cap_h = float(getattr(cp, "height", 2.0))
    cam_pos = op.target_object.location + Vector((0.0, 0.0, cap_h))

    # ── Apply to viewport ─────────────────────────────────────────
    rv3d = getattr(op, "_view3d_rv3d", None)
    if rv3d is None:
        if hasattr(op, "_maybe_rebind_view3d"):
            try:
                op._maybe_rebind_view3d(context)
            except Exception:
                pass
            rv3d = getattr(op, "_view3d_rv3d", None)
        if rv3d is None:
            return

    try:
        rv3d.view_location = cam_pos
        rv3d.view_rotation = q_cam
        rv3d.view_distance = 0.0
    except Exception:
        pass

    # ── Solve FPV bone rotation via matrix_basis ───────────────────
    if pb:
        _solve_bone_rotation(pb, arm, direction, scene)


def _solve_bone_rotation(pb, arm, direction, scene):
    """
    Override ONLY the FPV bone's rotation via matrix_basis.
    Position is left to the parent chain / animation.
    """
    dir_bone = direction.copy()
    if getattr(scene, "fpv_invert_pitch", False):
        dir_bone.z = -dir_bone.z

    q_target = dir_bone.to_track_quat('Z', 'Y')

    # Desired world-space matrix (rotation only, no translation)
    M_world = q_target.to_matrix().to_4x4()

    # World → armature pose space (rotation only)
    M_pose = arm.matrix_world.inverted() @ M_world

    # Factor out parent chain + rest pose
    M_parent = pb.parent.matrix.copy() if pb.parent else Matrix.Identity(4)
    M_rest = pb.bone.matrix_local.copy()

    M_full = (M_parent @ M_rest).inverted() @ M_pose

    # Rotation-only matrix_basis — bone position stays animation-driven
    pb.matrix_basis = M_full.to_quaternion().to_matrix().to_4x4()
    pb.scale = (1.0, 1.0, 1.0)


class EXP_OT_TestFPV(bpy.types.Operator):
    """Move the viewport to the FPV position for a quick preview"""
    bl_idname = "exploratory.test_fpv"
    bl_label = "Test FPV"

    def execute(self, context):
        scene = context.scene
        arm = getattr(scene, "target_armature", None)
        if not arm:
            self.report({'WARNING'}, "No Target Armature set")
            return {'CANCELLED'}

        cp = scene.char_physics
        cap_h = float(getattr(cp, "height", 2.0))
        cam_pos = arm.location + Vector((0.0, 0.0, cap_h))

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        rv3d = space.region_3d
                        rv3d.view_location = cam_pos
                        rv3d.view_distance = 0.0
                        rv3d.view_perspective = 'PERSP'
                        return {'FINISHED'}

        self.report({'WARNING'}, "No 3D viewport found")
        return {'CANCELLED'}


def reset_fpv_rot_scale(scene: bpy.types.Scene) -> None:
    """Reset the FPV pose bone's matrix_basis to identity."""
    arm = getattr(scene, "target_armature", None)
    if not (arm and isinstance(arm, bpy.types.Object) and arm.type == 'ARMATURE'):
        return

    name = (getattr(scene, "fpv_view_bone", "") or "").strip()
    if not name:
        return

    pb = arm.pose.bones.get(name)
    if not pb:
        return

    pb.matrix_basis = Matrix.Identity(4)

    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
