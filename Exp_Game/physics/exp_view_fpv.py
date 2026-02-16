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
    """Preview FPV for 3 seconds then revert the viewport"""
    bl_idname = "exploratory.test_fpv"
    bl_label = "Test FPV"

    _timer = None
    _saved = None   # (space, rv3d, old_location, old_rotation, old_distance, old_perspective, old_lens)
    _deadline = 0.0

    def invoke(self, context, event):
        import time

        scene = context.scene
        arm = getattr(scene, "target_armature", None)
        if not arm:
            self.report({'WARNING'}, "No Target Armature set")
            return {'CANCELLED'}

        # Find a VIEW_3D space + region_3d
        space = None
        rv3d = None
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                rv3d = space.region_3d
                break
        if rv3d is None:
            self.report({'WARNING'}, "No 3D viewport found")
            return {'CANCELLED'}

        # ── Save current viewport state ──
        self._saved = (
            space,
            rv3d,
            rv3d.view_location.copy(),
            rv3d.view_rotation.copy(),
            rv3d.view_distance,
            rv3d.view_perspective,
            space.lens,
        )

        # ── Build FPV view (same method as exp_spawn.py) ──
        cp = scene.char_physics
        cap_h = float(getattr(cp, "height", 2.0)) if cp else 2.0
        anchor = arm.location + Vector((0.0, 0.0, cap_h))

        arm_yaw = arm.matrix_world.to_euler('XYZ').z
        pitch = 0.0
        yaw = arm_yaw

        cx = math.cos(pitch); sx = math.sin(pitch)
        sy = math.sin(yaw);   cy = math.cos(yaw)
        cam_dir = Vector((cx * sy, -cx * cy, sx))
        if cam_dir.length > 1e-9:
            cam_dir.normalize()
        cam_quat = cam_dir.to_track_quat('Z', 'Y')

        lens_mm = float(getattr(scene, "viewport_lens_mm", 55.0))

        # ── Apply ──
        rv3d.view_perspective = 'PERSP'
        space.lens = lens_mm
        rv3d.view_location = anchor
        rv3d.view_rotation = cam_quat
        rv3d.view_distance = 0.0006

        # ── Start 3-second timer ──
        self._deadline = time.time() + 3.0
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        import time
        if event.type == 'TIMER' and time.time() >= self._deadline:
            self._restore(context)
            return {'FINISHED'}
        return {'PASS_THROUGH'}

    def _restore(self, context):
        # Revert viewport to saved state
        if self._saved:
            space, rv3d, loc, rot, dist, persp, lens = self._saved
            rv3d.view_location = loc
            rv3d.view_rotation = rot
            rv3d.view_distance = dist
            rv3d.view_perspective = persp
            space.lens = lens
            self._saved = None

        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def cancel(self, context):
        self._restore(context)


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
