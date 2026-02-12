# Exploratory/Exp_Game/physics/exp_locked_view.py
import math
from mathutils import Vector
from ..modal.exp_view_helpers import _resolved_move_keys, smooth_rotate_towards_camera


def run_locked_view(modal_op, context, steps: int) -> None:
    """
    LOCKED view movement:
      axis OFF  → full WASD relative to the locked camera yaw
      axis X/Y  → A/D only, constrained to a world axis
    """
    scene = context.scene
    if not modal_op.target_object or not modal_op.physics_controller or steps <= 0:
        return

    pc    = modal_op.physics_controller
    prefs = getattr(modal_op, "_prefs", None) or context.preferences.addons["Exploratory"].preferences
    dt    = float(getattr(modal_op, "physics_dt", 1.0 / 30.0))
    static_bvh      = getattr(modal_op, "bvh_tree", None)
    dyn_objects_map  = getattr(modal_op, "dynamic_objects_map", None)
    v_lin_map        = getattr(modal_op, "platform_linear_velocity_map", {}) or {}

    axis = getattr(scene, "view_locked_move_axis", "OFF")
    locked_yaw = float(getattr(scene, "view_locked_yaw", 0.0))
    flip = bool(getattr(scene, "view_locked_flip_axis", True))

    # ── Resolve keys & movement yaw ──────────────────────────────────
    if axis == "OFF":
        keys = set(_resolved_move_keys(modal_op))
        yaw_for_move = locked_yaw
    else:
        # Axis-constrained: drop W/S, override yaw to world axis
        keys = set(_resolved_move_keys(modal_op))
        keys.discard(modal_op.pref_forward_key)
        keys.discard(modal_op.pref_backward_key)
        yaw_for_move = 0.0 if axis == "X" else (math.pi * 0.5)
        # Flip reverses both movement direction AND facing
        if flip:
            yaw_for_move += math.pi

    # ── Physics steps ────────────────────────────────────────────────
    for _ in range(int(steps)):
        pc.try_consume_jump()
        pc.step(
            dt=dt,
            prefs=prefs,
            keys_pressed=keys,
            camera_yaw=yaw_for_move,
            static_bvh=static_bvh,
            dynamic_map=dyn_objects_map,
            platform_linear_velocity_map=v_lin_map,
            engine=getattr(modal_op, 'engine', None),
            context=context,
            physics_frame=getattr(modal_op, '_physics_frame', 0),
        )
        modal_op.z_velocity        = pc.vel.z
        modal_op.is_grounded       = pc.on_ground
        modal_op.grounded_platform = pc.ground_obj

    # ── Character rotation: always face movement direction ───────────
    smooth_rotate_towards_camera(modal_op, override_yaw=yaw_for_move)


def update_locked_camera(context, op):
    """
    Position the camera at locked yaw / pitch / distance from the character.
    No obstruction raycasting — just a direct viewport write.
    """
    scene = context.scene
    if not op or not getattr(op, "target_object", None):
        return

    # Anchor at capsule top (head height)
    cp    = scene.char_physics
    cap_h = float(getattr(cp, "height", 2.0))
    anchor = op.target_object.location + Vector((0.0, 0.0, cap_h))

    # Direction from locked pitch / yaw
    pitch = float(getattr(scene, "view_locked_pitch", 0.0))
    yaw   = float(getattr(scene, "view_locked_yaw", 0.0))
    cx = math.cos(pitch); sx = math.sin(pitch)
    sy = math.sin(yaw);   cy = math.cos(yaw)
    direction = Vector((cx * sy, -cx * cy, sx))
    if direction.length > 1e-9:
        direction.normalize()

    distance = float(getattr(scene, "view_locked_distance", 6.0))

    # Get rv3d
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
        rv3d.view_location = anchor
        rv3d.view_rotation = direction.to_track_quat('Z', 'Y')
        rv3d.view_distance = distance
    except Exception:
        pass
