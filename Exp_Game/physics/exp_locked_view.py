# Exploratory/Exp_Game/physics/exp_locked_view.py
import math
from ..modal.exp_view_helpers import _resolved_move_keys

def run_locked_view(modal_op, context, steps: int) -> None:
    """
    LOCKED view + Axis:
      • Constrain movement to a true world axis by overriding camera_yaw.
      • Face the travel axis; optionally flip 180° via scene.view_locked_flip_axis.
    """
    scene = context.scene
    if not modal_op.target_object or not modal_op.physics_controller or steps <= 0:
        return

    axis = getattr(scene, "view_locked_move_axis", "OFF")
    if axis not in ("X", "Y"):
        return

    pc = modal_op.physics_controller
    prefs = getattr(modal_op, "_prefs", None) or context.preferences.addons["Exploratory"].preferences
    dt = float(getattr(modal_op, "physics_dt", 1.0 / 30.0))
    static_bvh = getattr(modal_op, "bvh_tree", None)
    dyn_objects_map = getattr(modal_op, "dynamic_objects_map", None)
    v_lin_map  = getattr(modal_op, "platform_linear_velocity_map", {}) or {}
    v_ang_map  = getattr(modal_op, "platform_ang_velocity_map", {}) or {}

    # 1) Keep only A/D; drop W/S
    keys = set(_resolved_move_keys(modal_op))
    keys.discard(modal_op.pref_forward_key)
    keys.discard(modal_op.pref_backward_key)

    # 2) Map A/D onto world axis via yaw override:
    yaw_for_move = 0.0 if axis == "X" else (math.pi * 0.5)

    # 3) Physics at fixed rate
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
            platform_ang_velocity_map=v_ang_map,
        )
        modal_op.z_velocity        = pc.vel.z
        modal_op.is_grounded       = pc.on_ground
        modal_op.grounded_platform = pc.ground_obj

    # 4) Face the axis (optionally flip 180°)
    desired = None
    if axis == "X":
        if modal_op.pref_right_key in keys:
            desired =  math.pi * 0.5   # +X
        elif modal_op.pref_left_key in keys:
            desired = -math.pi * 0.5   # -X
    else:
        if modal_op.pref_right_key in keys:
            desired = math.pi          # +Y
        elif modal_op.pref_left_key in keys:
            desired = 0.0              # -Y

    if desired is not None:
        if bool(getattr(scene, "view_locked_flip_axis", False)):
            desired += math.pi
        modal_op.target_object.rotation_euler.z = desired

