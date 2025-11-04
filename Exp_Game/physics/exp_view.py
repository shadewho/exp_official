# Exploratory/Exp_Game/physics/exp_view.py
import math
import time
from mathutils import Vector
from .exp_raycastutils import raycast_closest_any

# ===========================
# Tunables (kept small / sane)
# ===========================
_MIN_CAM_ABS             = 0.0006          # 0.6 mm absolute floor
_MIN_CAM_RADIUS_FACTOR   = 0.04            # 4% of capsule radius
_NEARCLIP_TO_RADIUS_K    = 0.60            # camera radius ~60% of near clip
_R_CAM_FLOOR             = 0.008           # 8 mm minimum camera thickness
_EXTRA_PULL_METERS       = 0.25            # post-hit inward safety pull
_EXTRA_PULL_R_K          = 2.0             # + K * r_cam inward
_LOS_STEPS               = 1               # binary-search steps for clear LoS
_PUSHOUT_ITERS           = 1               # tiny nearest-point pushout
_LOS_EPS                 = 1.0e-3

# -------------------------
# Lightweight temporal filters
# -------------------------
class _CamSmoother:
    __slots__ = ("last_allowed", "last_t")
    OUTWARD_RATE = 10.0  # outward growth clamp (m/s)
    def __init__(self):
        self.last_allowed = None
        self.last_t = time.perf_counter()
    def filter(self, target: float) -> float:
        now = time.perf_counter()
        dt  = max(1e-6, now - self.last_t)
        if self.last_allowed is None:
            self.last_allowed = target
        elif target >= self.last_allowed:
            self.last_allowed = min(target, self.last_allowed + self.OUTWARD_RATE * dt)
        else:
            self.last_allowed = target  # immediate pull-in
        self.last_t = now
        return self.last_allowed

class _CamLatch:
    __slots__ = ("latched_obj", "latched_until", "latched_allowed")
    RELEASE_PAD_MIN = 0.06
    RELEASE_PAD_K   = 1.6
    HOLD_TIME       = 0.14
    def __init__(self):
        self.latched_obj    = None
        self.latched_until  = 0.0
        self.latched_allowed= None
    def filter(self, hit_obj_token, allowed_now: float, r_cam: float) -> float:
        now = time.perf_counter()
        if hit_obj_token is not None:
            self.latched_obj     = hit_obj_token
            self.latched_allowed = allowed_now
            self.latched_until   = now + self.HOLD_TIME
            return allowed_now
        if self.latched_obj is not None and self.latched_allowed is not None:
            need_pad = max(self.RELEASE_PAD_MIN, self.RELEASE_PAD_K * r_cam)
            if (now < self.latched_until) and (allowed_now < (self.latched_allowed + need_pad)):
                return self.latched_allowed
            self.latched_obj = None
            self.latched_allowed = None
            self.latched_until = 0.0
        return allowed_now

# Per-operator caches (we key by id(op))
_SMOOTHERS = {}
_LATCHES   = {}
_VIEWS     = {}  # id(op) -> dict with inputs/outputs and rv3d & debounce caches

def _view_state_for(op_key):
    v = _VIEWS.get(op_key)
    if v is None:
        v = {
            # last inputs used for early-out
            "last_params": None,
            "last_time": 0.0,
            # cached outputs for apply
            "anchor": None,
            "direction": None,
            "allowed": 3.0,
            # last-applied rv3d channels to skip redundant writes
            "rv3d_loc": None, "rv3d_rot": None, "rv3d_dst": None,
            # micro eps + heartbeat (pose unchanged → skip solve)
            "eps_pitch": 0.0015, "eps_yaw": 0.0015,
            "eps_anchor": 0.008, "eps_dist": 0.010,
            "heartbeat": 0.16,
        }
        _VIEWS[op_key] = v
    return v

# -------------------------
# Helpers (occlusion + pushout)
# -------------------------
_STATIC_TOKEN = "__STATIC__"

def _multi_ray_min_hit(static_bvh, dynamic_bvh_map, origin, direction, max_dist, r_cam):
    if direction.length <= 1e-9 or max_dist <= 1e-9:
        return (None, None)
    dnorm = direction.normalized()
    best = (None, None)
    if static_bvh:
        hit = static_bvh.ray_cast(origin, dnorm, max_dist)
        if hit and hit[0] is not None:
            best = (hit[3], _STATIC_TOKEN)
    if dynamic_bvh_map:
        pf_pad = 0.05 + r_cam
        for obj, (bvh_like, approx_rad) in dynamic_bvh_map.items():
            center = obj.matrix_world.translation
            oc = center - origin
            t = oc.dot(dnorm)
            if t < -approx_rad or t > max_dist + approx_rad:
                continue
            closest = (oc if t < 0.0 else (oc - dnorm * (max_dist if t > max_dist else t)))
            if closest.length_squared <= (approx_rad + pf_pad) * (approx_rad + pf_pad):
                h = bvh_like.ray_cast(origin, dnorm, max_dist)
                if h and h[0] is not None:
                    d = h[3]
                    if best[0] is None or d < best[0]:
                        best = (d, obj)
    return best if best[0] is not None else (None, None)


def _los_blocked(static_bvh, dynamic_bvh_map, a: Vector, b: Vector):
    d = b - a
    dist = d.length
    if dist <= 1e-9:
        return False
    dnorm = d / dist
    if static_bvh:
        h = static_bvh.ray_cast(a, dnorm, max(0.0, dist - _LOS_EPS))
        if h and h[0] is not None:
            return True
    if dynamic_bvh_map:
        for _obj, (bvh_like, _r) in dynamic_bvh_map.items():
            h = bvh_like.ray_cast(a, dnorm, max(0.0, dist - _LOS_EPS))
            if h and h[0] is not None:
                return True
    return False



def _binary_search_clear_los(static_bvh, dynamic_bvh_map, anchor, direction, low, high, steps):
    lo, hi = low, high
    for _ in range(max(1, int(steps))):
        mid = 0.5 * (lo + hi)
        cam = anchor + direction * mid
        if _los_blocked(static_bvh, dynamic_bvh_map, anchor, cam):
            hi = mid
        else:
            lo = mid
    return lo


def _camera_sphere_pushout_any(static_bvh, dynamic_bvh_map, pos, radius, max_iters=_PUSHOUT_ITERS):
    if radius <= 1.0e-6:
        return pos
    def push_once_bvh(bvh_like, p):
        res = bvh_like.find_nearest(p)
        if not res or res[0] is None or res[1] is None:
            return p, False
        hit_co, hit_n, _idx, dist = res
        n = hit_n if (p - hit_co).dot(hit_n) >= 0.0 else -hit_n
        if dist < radius:
            return p + n * ((radius - dist) + 1.0e-4), True
        return p, False
    p = pos
    moved = True
    it = 0
    while moved and it < max(1, int(max_iters)):
        moved = False
        if static_bvh:
            p, m = push_once_bvh(static_bvh, p); moved = moved or m
        if dynamic_bvh_map:
            for _obj, (bvh_like, _r) in dynamic_bvh_map.items():
                p, m = push_once_bvh(bvh_like, p); moved = moved or m
        it += 1
    return p

# -------------------------
# Public, single entrypoint
# -------------------------
def view_queue_third_job(context, op):
    """
    THIRD-person obstruction solve moved to XR with exact BVHs for BOTH static and dynamic.
    No local rays, no proxies, no static/dynamic split. XR computes candidate+allowed.
    """
    if not op or not getattr(op, "target_object", None):
        return
    mode = getattr(context.scene, "view_mode", "THIRD")
    if mode in ("FIRST", "LOCKED"):
        return
    if not getattr(context.scene, "view_obstruction_enabled", True):
        return

    # Lazily sync exact static + dynamic meshes (triangles) to XR
    try:
        if getattr(op, "_xr", None):
            if not getattr(op, "_view_static_synced_exact", False):
                from ..xr_systems.xr_ports.view import sync_view_static_meshes_exact
                sync_view_static_meshes_exact(context, op)
                op._view_static_synced_exact = True
            from ..xr_systems.xr_ports.view import sync_view_dynamic_meshes_exact
            sync_view_dynamic_meshes_exact(op)
    except Exception:
        pass

    # Direction & anchor (unchanged)
    pitch = float(op.pitch); yaw = float(op.yaw)
    cx = math.cos(pitch); sx = math.sin(pitch)
    sy = math.sin(yaw);   cy = math.cos(yaw)
    direction = Vector((cx * sy, -cx * cy, sx))
    if direction.length > 1.0e-9:
        direction.normalize()

    cp     = context.scene.char_physics
    cap_h  = float(getattr(cp, "height", 2.0))
    cap_r  = float(getattr(cp, "radius", 0.30))
    anchor = op.target_object.location + Vector((0.0, 0.0, cap_h))

    clip_start = float(getattr(op, "_clip_start_cached", 0.1) or 0.1)
    if clip_start <= 0.0: clip_start = 0.1
    r_cam      = max(_R_CAM_FLOOR, clip_start * _NEARCLIP_TO_RADIUS_K)
    desired_max= max(0.0, context.scene.orbit_distance + context.scene.zoom_factor)
    min_cam    = max(_MIN_CAM_ABS, cap_r * _MIN_CAM_RADIUS_FACTOR)

    # Per-frame dynamic transforms (in tandem with statics)
    dyn_objects = []
    dyn_map = getattr(op, "dynamic_bvh_map", {}) or {}
    if dyn_map:
        for obj in dyn_map.keys():
            try:
                M = obj.matrix_world
                # row-major 4x4 flatten
                Mm = [M[0][0],M[0][1],M[0][2],M[0][3],
                      M[1][0],M[1][1],M[1][2],M[1][3],
                      M[2][0],M[2][1],M[2][2],M[2][3],
                      0.0,     0.0,     0.0,     1.0]
                dyn_objects.append({"id": obj.name, "M": Mm})
            except Exception:
                continue

    # Enqueue ONE XR job that computes candidate + allowed using exact BVHs for both static+dynamic
    try:
        from ..xr_systems.xr_ports.view import queue_view_third_full_exact
        queue_view_third_full_exact(
            op=op,
            op_key=id(op),
            anchor_xyz=(anchor.x, anchor.y, anchor.z),
            dir_xyz=(direction.x, direction.y, direction.z),
            min_cam=float(min_cam),
            desired_max=float(desired_max),
            r_cam=float(r_cam),
            dyn_objects=dyn_objects,
        )
    except Exception:
        # We do not compute locally if XR errors; no fallback geometry.
        pass


def apply_view_from_xr(context, op):
    """
    Apply camera using the latest XR answer (if any).
    FIRST/LOCKED are applied locally exactly as before.
    THIRD uses XR ONLY; if XR hasn't replied yet, clamp to min_cam (safe).
    """
    if not op or not getattr(op, "target_object", None):
        return

    op_key = id(op)
    view   = _view_state_for(op_key)

    mode = getattr(context.scene, "view_mode", "THIRD")

    # Direction & anchor
    if mode == 'LOCKED':
        lp = float(getattr(context.scene, "view_locked_pitch", math.radians(60.0)))
        ly = float(getattr(context.scene, "view_locked_yaw", 0.0))
        cx = math.cos(lp); sx = math.sin(lp)
        sy = math.sin(ly); cy = math.cos(ly)
        direction = Vector((cx * sy, -cx * cy, sx))
        if direction.length > 1.0e-9:
            direction.normalize()
        cap_h  = float(getattr(context.scene.char_physics, "height", 2.0))
        anchor = op.target_object.location + Vector((0.0, 0.0, cap_h))
        final_allowed = max(_MIN_CAM_ABS, float(getattr(context.scene, "view_locked_distance", 6.0)))
        view["anchor"] = anchor; view["direction"] = direction; view["allowed"] = float(final_allowed)
        op._cam_allowed_last = float(final_allowed)
        _apply_to_viewport(context, op, view)
        return

    # FIRST-person
    if mode == 'FIRST':
        cap_h  = float(getattr(context.scene.char_physics, "height", 2.0))
        anchor = op.target_object.location + Vector((0.0, 0.0, cap_h))
        pitch = float(op.pitch); yaw = float(op.yaw)
        cx = math.cos(pitch); sx = math.sin(pitch)
        sy = math.sin(yaw);   cy = math.cos(yaw)
        direction = Vector((cx * sy, -cx * cy, sx))
        if direction.length > 1.0e-9:
            direction.normalize()
        view["anchor"] = anchor; view["direction"] = direction; view["allowed"] = 0.0
        op._cam_allowed_last = 0.0
        try:
            from .exp_view_fpv import update_first_person_for_operator as _fpv_update
            _fpv_update(context, op, anchor, direction)
        except Exception:
            pass
        _apply_to_viewport(context, op, view)
        return

    # THIRD-person
    # Compute anchor/dir; final distance comes ONLY from XR.
    pitch = float(op.pitch); yaw = float(op.yaw)
    cx = math.cos(pitch); sx = math.sin(pitch)
    sy = math.sin(yaw);   cy = math.cos(yaw)
    direction = Vector((cx * sy, -cx * cy, sx))
    if direction.length > 1.0e-9:
        direction.normalize()

    cp     = context.scene.char_physics
    cap_h  = float(getattr(cp, "height", 2.0))
    cap_r  = float(getattr(cp, "radius", 0.30))
    anchor = op.target_object.location + Vector((0.0, 0.0, cap_h))

    clip_start = float(getattr(op, "_clip_start_cached", 0.1) or 0.1)
    if clip_start <= 0.0:
        clip_start = 0.1
    r_cam = max(_R_CAM_FLOOR, clip_start * _NEARCLIP_TO_RADIUS_K)
    min_cam = max(_MIN_CAM_ABS, cap_r * _MIN_CAM_RADIUS_FACTOR)

    xr_allowed = getattr(op, "_xr_view_allowed", None)
    final_allowed = float(xr_allowed) if isinstance(xr_allowed, (int, float)) else float(min_cam)

    view["anchor"]    = anchor
    view["direction"] = direction
    view["allowed"]   = final_allowed
    op._cam_allowed_last = final_allowed

    _apply_to_viewport(context, op, view)

    

def update_camera_for_operator(context, op):
    """
    The ONE function exp_loop should call each frame (no dynamic-map copying).

    Change: final camera distance is decided by XR ONLY (no fallback).
    Blender still computes raw geometry candidate (identical to your original logic),
    then we enqueue an XR job for latch/smoothing and consume ONLY XR output.
    """
    if not op or not getattr(op, "target_object", None):
        return

    from ..xr_systems.xr_ports.view import queue_view_third

    op_key = id(op)
    view   = _view_state_for(op_key)

    # --- 1) Direction from operator pitch/yaw (unchanged) ---
    pitch = float(op.pitch)
    yaw   = float(op.yaw)
    cx = math.cos(pitch); sx = math.sin(pitch)
    sy = math.sin(yaw);   cy = math.cos(yaw)
    direction = Vector((cx * sy, -cx * cy, sx))
    if direction.length > 1e-9:
        direction.normalize()

    # --- 2) Anchor at capsule top (unchanged) ---
    cp     = context.scene.char_physics
    cap_h  = float(getattr(cp, "height", 2.0))
    cap_r  = float(getattr(cp, "radius", 0.30))
    anchor = op.target_object.location + Vector((0.0, 0.0, cap_h))

    # --- 3) Camera “thickness” from near clip (unchanged) ---
    clip_start = float(getattr(op, "_clip_start_cached", 0.1) or 0.1)
    if clip_start <= 0.0:
        clip_start = 0.1
    r_cam = max(_R_CAM_FLOOR, clip_start * _NEARCLIP_TO_RADIUS_K)

    # --- 4) Desired boom + minimum (unchanged) ---
    desired_max = max(0.0, context.scene.orbit_distance + context.scene.zoom_factor)
    min_cam     = max(_MIN_CAM_ABS, cap_r * _MIN_CAM_RADIUS_FACTOR)

    # === LOCKED mode (unchanged, stays local) ===
    if getattr(context.scene, "view_mode", "THIRD") == 'LOCKED':
        lp = float(getattr(context.scene, "view_locked_pitch", math.radians(60.0)))
        ly = float(getattr(context.scene, "view_locked_yaw", 0.0))
        cx = math.cos(lp); sx = math.sin(lp)
        sy = math.sin(ly); cy = math.cos(ly)
        direction = Vector((cx * sy, -cx * cy, sx))
        if direction.length > 1.0e-9:
            direction.normalize()
        final_allowed = max(_MIN_CAM_ABS, float(getattr(context.scene, "view_locked_distance", 6.0)))
        view["anchor"] = anchor; view["direction"] = direction; view["allowed"] = float(final_allowed)
        op._cam_allowed_last = float(final_allowed)
        _apply_to_viewport(context, op, view)
        return

    # === FIRST PERSON (unchanged behavior) ===
    if getattr(context.scene, "view_mode", "THIRD") == 'FIRST':
        view["anchor"] = anchor; view["direction"] = direction; view["allowed"] = 0.0
        op._cam_allowed_last = 0.0
        try:
            from .exp_view_fpv import update_first_person_for_operator as _fpv_update
            _fpv_update(context, op, anchor, direction)
        except Exception:
            pass
        _apply_to_viewport(context, op, view)
        return

    # === HARD SWITCH: obstruction OFF (unchanged fast path) ===
    if not getattr(context.scene, "view_obstruction_enabled", True):
        final_allowed = max(min_cam, desired_max)
        view["anchor"] = anchor; view["direction"] = direction; view["allowed"] = float(final_allowed)
        op._cam_allowed_last = float(final_allowed)
        _apply_to_viewport(context, op, view)
        return

    # --- 5) Early-out reuse (unchanged) ---
    params = (pitch, yaw, anchor.x, anchor.y, anchor.z, desired_max, r_cam)
    now = time.perf_counter()
    lp = view["last_params"]
    if lp is not None:
        lp0, lp1, lx, ly, lz, ldm, lrc = lp
        same_pose = (
            abs(pitch - lp0) < view["eps_pitch"] and
            abs(yaw   - lp1) < view["eps_yaw"]   and
            math.hypot(math.hypot(anchor.x - lx, anchor.y - ly), anchor.z - lz) < view["eps_anchor"] and
            abs(desired_max - ldm) < view["eps_dist"] and
            abs(r_cam - lrc) < 1e-6
        )
        if same_pose and (now - view["last_time"]) < view["heartbeat"]:
            # even on heartbeat reuse we still APPLY XR's last answer
            xr_allowed = getattr(op, "_xr_view_allowed", None)
            if isinstance(xr_allowed, (int, float)):
                view["anchor"] = anchor; view["direction"] = direction; view["allowed"] = float(xr_allowed)
                op._cam_allowed_last = float(xr_allowed)
                _apply_to_viewport(context, op, view)
                return

    view["last_params"] = params
    view["last_time"]   = now

    # --- 6) Obstruction solve (IDENTICAL geometry, local) ---
    static_bvh       = getattr(op, "bvh_tree", None)
    dynamic_bvh_map  = getattr(op, "dynamic_bvh_map", None)

    hit_dist, hit_token = _multi_ray_min_hit(static_bvh, dynamic_bvh_map, anchor, direction, desired_max, r_cam)
    if hit_dist is not None:
        base_allowed = max(min_cam, min(desired_max, hit_dist - r_cam))
        allowed      = max(min_cam, base_allowed - (_EXTRA_PULL_METERS + _EXTRA_PULL_R_K * r_cam))
    else:
        allowed = desired_max

    # 7) Ensure clear LoS (binary search inward if needed)
    candidate = anchor + direction * allowed
    if _los_blocked(static_bvh, dynamic_bvh_map, anchor, candidate):
        allowed = _binary_search_clear_los(static_bvh, dynamic_bvh_map, anchor, direction,
                                           low=min_cam, high=allowed, steps=_LOS_STEPS)
        candidate = anchor + direction * allowed

    # 8) Tiny nearest-point pushout (unchanged)
    candidate = _camera_sphere_pushout_any(static_bvh, dynamic_bvh_map, candidate, r_cam, max_iters=_PUSHOUT_ITERS)
    allowed_after_push = (candidate - anchor).length

    # --- XR ENQUEUE: from here on, XR is authoritative ---
    token_str = None
    if hit_token == _STATIC_TOKEN:
        token_str = "__STATIC__"
    elif hit_token is not None:
        try:
            token_str = getattr(hit_token, "name", "DYNAMIC")
        except Exception:
            token_str = "DYNAMIC"

    # HUD: show the local candidate (for comparison) without applying it
    try:
        from ..Developers.exp_dev_interface import devhud_set, devhud_series_push
        devhud_set("VIEW.candidate", f"{allowed_after_push:.3f} m", volatile=True)
        devhud_series_push("view_candidate", float(allowed_after_push))
    except Exception:
        pass

    # Optional debug assert channel: compare XR last vs candidate
    if bool(getattr(context.scene, "view_debug_assert_dyn", False)):
        xr_last = getattr(op, "_xr_view_allowed", None)
        if isinstance(xr_last, (int, float)):
            try:
                from ..Developers.exp_dev_interface import devhud_set
                devhud_set("VIEW.assert_delta", f"{abs(xr_last - allowed_after_push):.3f} m", volatile=True)
            except Exception:
                pass

    # Queue XR job (no stall). XR will return the ONLY value we apply.
    queue_view_third(
        op=op,
        op_key=op_key,
        anchor_xyz=(anchor.x, anchor.y, anchor.z),
        dir_xyz=(direction.x, direction.y, direction.z),
        min_cam=min_cam,
        desired_max=desired_max,
        r_cam=r_cam,
        candidate_allowed=allowed_after_push,
        hit_token=token_str
    )

    # Apply using XR's last known answer; ABSOLUTELY NO local fallback
    xr_allowed = getattr(op, "_xr_view_allowed", None)
    final_allowed = float(xr_allowed) if isinstance(xr_allowed, (int, float)) else float(min_cam)

    view["anchor"]    = anchor
    view["direction"] = direction
    view["allowed"]   = final_allowed
    op._cam_allowed_last = final_allowed

    _apply_to_viewport(context, op, view)



def _apply_to_viewport(context, op, view):
    anchor    = view.get("anchor");    direction = view.get("direction")
    desired_d = view.get("allowed", 3.0)
    if anchor is None or direction is None:
        return

    # --- sanitize inputs (avoid 0/NaN distance & zero-length dir) ---
    try:
        desired_d = float(desired_d)
    except Exception:
        desired_d = _MIN_CAM_ABS
    if desired_d != desired_d:  # NaN check
        desired_d = _MIN_CAM_ABS
    desired_d = max(desired_d, _MIN_CAM_ABS)

    dir_vec = direction if direction.length > 1.0e-12 else Vector((0.0, -1.0, 0.0))

    # ensure a bound rv3d exists on the operator
    rv3d = getattr(op, "_view3d_rv3d", None)
    if rv3d is None:
        ok = False
        try:
            if hasattr(op, "_maybe_rebind_view3d"):
                ok = bool(op._maybe_rebind_view3d(context))
        except Exception:
            ok = False
        rv3d = getattr(op, "_view3d_rv3d", None) if ok else None
        if rv3d is None:
            return

    POS_EPS  = 1e-4
    ANG_EPS  = 1e-4
    DIST_EPS = 1e-4

    last_loc = view["rv3d_loc"]
    last_rot = view["rv3d_rot"]
    last_dst = view["rv3d_dst"]

    target_rot = dir_vec.to_track_quat('Z', 'Y')

    need_loc = (last_loc is None) or ((anchor - last_loc).length > POS_EPS)
    if last_rot is None:
        need_rot = True
    else:
        dq = last_rot.rotation_difference(target_rot)
        need_rot = abs(dq.angle) > ANG_EPS
    need_dst = (last_dst is None) or (abs(desired_d - last_dst) > DIST_EPS)

    try:
        if need_loc:
            rv3d.view_location = anchor
            view["rv3d_loc"] = anchor.copy()
        if need_rot:
            rv3d.view_rotation = target_rot
            view["rv3d_rot"] = target_rot.copy()
        if need_dst:
            rv3d.view_distance = desired_d
            view["rv3d_dst"] = float(desired_d)
    except Exception:
        # single retry after rebind (layout/fullscreen swaps)
        if hasattr(op, "_maybe_rebind_view3d") and op._maybe_rebind_view3d(context):
            try:
                rv3d = op._view3d_rv3d
                if need_loc:
                    rv3d.view_location = anchor
                    view["rv3d_loc"] = anchor.copy()
                if need_rot:
                    rv3d.view_rotation = target_rot
                    view["rv3d_rot"] = target_rot.copy()
                if need_dst:
                    rv3d.view_distance = desired_d
                    view["rv3d_dst"] = float(desired_d)
            except Exception:
                pass


# Convenience util (kept for any callers)
def shortest_angle_diff(current, target):
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return diff
