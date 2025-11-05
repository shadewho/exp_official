import bpy
import mathutils
from .exp_bvh_local import LocalBVH

def update_dynamic_meshes(modal_op):
    """
    Distance-gated dynamic proxies with minimal depsgraph traffic:
      • Distance gate BEFORE evaluated_get().
      • ACTIVE movers: LocalBVH cache (Blender) + XR dynamic init (once) + XR xforms each frame.
      • Outputs include dynamic_bvh_map and per-mover velocities as before.
      • NEW (M2.verify): counts active movers + gate toggles; updates DevHUD.
    """

    scene = bpy.context.scene

    # --- Caches / state (create once) ---
    if not hasattr(modal_op, "cached_local_bvhs"):
        modal_op.cached_local_bvhs = {}
    if not hasattr(modal_op, "_dyn_active_state"):
        modal_op._dyn_active_state = {}
    if not hasattr(modal_op, "platform_prev_positions"):
        modal_op.platform_prev_positions = {}
    if not hasattr(modal_op, "platform_prev_matrices"):
        modal_op.platform_prev_matrices = {}
    if not hasattr(modal_op, "_cached_dyn_radius"):
        modal_op._cached_dyn_radius = {}

    if not hasattr(modal_op, "_xr_geom_gate_switches"):
        modal_op._xr_geom_gate_switches = 0

    try:
        from ..xr_systems.xr_ports import geom as _geom_port
        _geom_port._ensure_dyn_maps(modal_op)
    except Exception:
        _geom_port = None

    # --- Outputs (reuse dicts to avoid churn) ---
    if not hasattr(modal_op, "dynamic_bvh_map"):
        modal_op.dynamic_bvh_map = {}
    else:
        modal_op.dynamic_bvh_map.clear()

    if not hasattr(modal_op, "platform_motion_map"):
        modal_op.platform_motion_map = {}
    else:
        modal_op.platform_motion_map.clear()

    if not hasattr(modal_op, "platform_linear_velocity_map"):
        modal_op.platform_linear_velocity_map = {}
    else:
        modal_op.platform_linear_velocity_map.clear()

    if not hasattr(modal_op, "platform_ang_velocity_map"):
        modal_op.platform_ang_velocity_map = {}
    else:
        modal_op.platform_ang_velocity_map.clear()

    if not hasattr(modal_op, "platform_delta_quat_map"):
        modal_op.platform_delta_quat_map = {}
    else:
        modal_op.platform_delta_quat_map.clear()

    if not hasattr(modal_op, "platform_delta_map"):
        modal_op.platform_delta_map = {}
    else:
        modal_op.platform_delta_map.clear()

    # Safe dt for velocity calc — prefer fixed physics dt (30 Hz)
    frame_dt = getattr(modal_op, "physics_dt", None)
    if frame_dt is None or frame_dt <= 0.0:
        frame_dt = getattr(modal_op, "delta_time", 0.0)
    if frame_dt is None or frame_dt <= 1e-8:
        frame_dt = 1e-8

    # Player location (once)
    player_loc = None
    if getattr(modal_op, "target_object", None):
        player_loc = modal_op.target_object.matrix_world.translation

    xr_batch = []
    active_count = 0
    gate_toggles_this_frame = 0

    for pm in scene.proxy_meshes:
        dyn_obj = pm.mesh_object
        if not dyn_obj or dyn_obj.type != 'MESH' or not pm.is_moving:
            continue

        # -------- 1) Distance gate BEFORE depsgraph/evaluated_get ----------
        cur_M_quick = dyn_obj.matrix_world
        cur_pos_quick = cur_M_quick.translation

        active = True
        prev_active = modal_op._dyn_active_state.get(dyn_obj)
        if pm.register_distance > 0.0 and player_loc is not None:
            base = float(pm.register_distance)
            margin = base * 0.10
            threshold = (base + margin) if (prev_active is True) else max(0.0, base - margin)
            d2 = (cur_pos_quick - player_loc).length_squared
            active = (d2 <= (threshold * threshold))

        if prev_active is None or prev_active != active:
            modal_op._dyn_active_state[dyn_obj] = active
            gate_toggles_this_frame += 1

        if not active:
            # Maintain previous pose cheaply
            modal_op.platform_prev_positions[dyn_obj] = cur_pos_quick.copy()
            modal_op.platform_prev_matrices[dyn_obj] = cur_M_quick.copy()
            continue

        active_count += 1

        # -------- 2) ACTIVE path ----------
        cur_M = cur_M_quick.copy()

        # Blender LocalBVH cache for gameplay (unchanged authority)
        lbvh = modal_op.cached_local_bvhs.get(dyn_obj)
        if lbvh is None:
            lbvh = LocalBVH(dyn_obj)
            modal_op.cached_local_bvhs[dyn_obj] = lbvh
        lbvh.update_xform(cur_M)

        # XR dynamic init (LOCAL triangles once)
        if _geom_port and (dyn_obj not in modal_op._xr_dyn_inited):
            try:
                did = _geom_port.ensure_dyn_id(modal_op, dyn_obj)
                _geom_port.init_dynamic_single(modal_op, dyn_obj, did)
            except Exception:
                pass

        # Accumulate xform (row-major 4x4)
        if _geom_port:
            M = cur_M
            M16 = [float(M[0][0]), float(M[0][1]), float(M[0][2]), float(M[0][3]),
                   float(M[1][0]), float(M[1][1]), float(M[1][2]), float(M[1][3]),
                   float(M[2][0]), float(M[2][1]), float(M[2][2]), float(M[2][3]),
                   float(M[3][0]), float(M[3][1]), float(M[3][2]), float(M[3][3])]
            try:
                did = modal_op._xr_dyn_id_map[dyn_obj]
                xr_batch.append((did, M16))
            except Exception:
                try:
                    _geom_port.ensure_dyn_id(modal_op, dyn_obj)
                except Exception:
                    pass

        # --- your existing velocity & delta calc (unchanged) ---
        if dyn_obj not in modal_op._cached_dyn_radius:
            bbox_world = [dyn_obj.matrix_world @ mathutils.Vector(c) for c in dyn_obj.bound_box]
            center_world = sum(bbox_world, mathutils.Vector()) / 8.0
            rad_center = max((p - center_world).length for p in bbox_world)
            origin_world = dyn_obj.matrix_world.translation
            center_offset = (center_world - origin_world).length
            modal_op._cached_dyn_radius[dyn_obj] = rad_center + center_offset

        rad = modal_op._cached_dyn_radius.get(dyn_obj, 0.0)
        modal_op.dynamic_bvh_map[dyn_obj] = (lbvh, rad)

        prev_pos = modal_op.platform_prev_positions.get(dyn_obj)
        cur_pos  = cur_M.translation.copy()
        if prev_pos is not None:
            disp = cur_pos - prev_pos
            modal_op.platform_motion_map[dyn_obj] = disp
            modal_op.platform_linear_velocity_map[dyn_obj] = disp / frame_dt
        modal_op.platform_prev_positions[dyn_obj] = cur_pos

        prev_M = modal_op.platform_prev_matrices.get(dyn_obj, cur_M.copy())
        delta_M = cur_M @ prev_M.inverted()
        R = delta_M.to_3x3(); R.normalize()
        dq = R.to_quaternion()
        modal_op.platform_delta_quat_map[dyn_obj] = dq
        modal_op.platform_delta_map[dyn_obj] = delta_M
        modal_op.platform_prev_matrices[dyn_obj] = cur_M.copy()

        try:
            axis, angle = dq.to_axis_angle()
        except Exception:
            axis, angle = mathutils.Vector((0.0, 0.0, 1.0)), 0.0
        omega = axis * (angle / frame_dt) if angle > 1.0e-9 else mathutils.Vector((0.0, 0.0, 0.0))
        modal_op.platform_ang_velocity_map[dyn_obj] = omega

    # --- Send XR xforms batch (non-blocking) ---
    if xr_batch and _geom_port:
        try:
            _geom_port.update_xforms_batch(modal_op, xr_batch)
        except Exception:
            pass

    # --- NEW: push per-frame counters to HUD ---
    try:
        from ..Developers.exp_dev_interface import devhud_set
        devhud_set("XR.geom.active", int(active_count), volatile=True)
        if gate_toggles_this_frame > 0:
            modal_op._xr_geom_gate_switches += int(gate_toggles_this_frame)
        devhud_set("XR.geom.gate_switches", int(getattr(modal_op, "_xr_geom_gate_switches", 0) or 0), volatile=True)
    except Exception:
        pass


