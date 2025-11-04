# Exp_Game/physics/exp_view_fpv.py
# First-person bone driver (strict, no magic names).
# Replaces: update_first_person_for_operator()

import bpy
from mathutils import Matrix, Euler
def _get_arm(scene):
    arm = getattr(scene, "target_armature", None)
    return arm if isinstance(arm, bpy.types.Object) and arm.type == 'ARMATURE' else None

def _get_fpv_pose_bone(scene, arm):
    name = (getattr(scene, "fpv_view_bone", "") or "").strip()
    return arm.pose.bones.get(name) if name else None

def update_first_person_for_operator(context, op, anchor, direction):
    """
    FIRST-person hard lock (merged):
      • Location: take the *viewport* anchor (rv3d.view_location) → kills sway/boost.
      • Rotation: take the stable camera direction passed in (pitch/yaw → 'direction')
                  rather than rv3d.view_rotation → avoids micro jitters.
      • Do NOT rotate the armature object here.
      • Solve the FPV pose bone parent-aware.
    """


    scene = context.scene
    arm = _get_arm(scene)
    if not arm:
        return

    pb = _get_fpv_pose_bone(scene, arm)
    if not pb:
        return

    # --- Location: prefer current viewport anchor (locks to what the user sees)
    rv3d = getattr(op, "_view3d_rv3d", None)
    if rv3d is None and hasattr(op, "_maybe_rebind_view3d"):
        try:
            op._maybe_rebind_view3d(context)
        except Exception:
            pass
        rv3d = getattr(op, "_view3d_rv3d", None)

    anchor_world = rv3d.view_location.copy() if rv3d else anchor

    # --- Rotation: build from the passed-in direction (stable, jitter-free)
    dir_fpv = direction.copy()
    if bool(getattr(scene, "fpv_invert_pitch", False)):
        dir_fpv.z = -dir_fpv.z
    q = dir_fpv.normalized().to_track_quat('Z', 'Y')

    # --- Apply as a world-space solve for the pose bone (parent-aware)
    M_world  = Matrix.Translation(anchor_world) @ q.to_matrix().to_4x4()
    M_pose   = arm.matrix_world.inverted() @ M_world
    M_parent = pb.parent.matrix.copy() if pb.parent else Matrix.Identity(4)
    M_rest   = pb.bone.matrix_local.copy()

    pb.matrix_basis = (M_parent @ M_rest).inverted() @ M_pose
    pb.scale = (1.0, 1.0, 1.0)


def reset_fpv_rot_scale(scene: bpy.types.Scene) -> None:
    """
    Reset ONLY rotation + scale of the FPV pose bone to identity.
    (Do NOT touch location.)
    """
    arm = getattr(scene, "target_armature", None)
    if not (arm and isinstance(arm, bpy.types.Object) and arm.type == 'ARMATURE'):
        return

    name = (getattr(scene, "fpv_view_bone", "") or "").strip()
    if not name:
        return

    pb = arm.pose.bones.get(name)
    if not pb:
        return

    # Rotation → identity (respect current rotation_mode)
    mode = pb.rotation_mode
    if mode == 'QUATERNION':
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    elif mode == 'AXIS_ANGLE':
        pb.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
    else:
        pb.rotation_euler = Euler((0.0, 0.0, 0.0), mode or 'XYZ')

    # Scale → 1
    pb.scale = (1.0, 1.0, 1.0)

    # Don’t touch pb.location. Ensure updates are seen.
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass