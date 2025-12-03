# Exploratory/Exp_Game/physics/exp_view.py
#
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  NO MAIN THREAD FALLBACK - PURE ENGINE OFFLOAD                               ║
# ║                                                                              ║
# ║  Camera raycasting is 100% offloaded to the multiprocessing engine.          ║
# ║  DO NOT add main-thread raycast fallbacks, hybrid modes, or legacy code.     ║
# ║  If no engine result is available, HOLD the current camera position.         ║
# ║  Work through issues with pure engine - no safety nets.                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
import math
import time
from mathutils import Vector
from .exp_raycastutils import raycast_closest_any
from ..developer.dev_debug_gate import should_print_debug

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
    OUTWARD_RATE = 15.0  # outward growth clamp (m/s) - snappier camera feel
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

def _smooth_for(op_key):
    s = _SMOOTHERS.get(op_key)
    if s is None:
        s = _CamSmoother()
        _SMOOTHERS[op_key] = s
    return s

def _latch_for(op_key):
    l = _LATCHES.get(op_key)
    if l is None:
        l = _CamLatch()
        _LATCHES[op_key] = l
    return l

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
            # Timing stats for debug output
            "last_static_calc_time_us": 0.0,
            "last_dynamic_calc_time_us": 0.0,
            # Same-frame offload state (PURE ENGINE)
            "pending_job_id": None,
            "pending_job_submit_time": 0.0,  # Time when pending job was submitted
            "cached_worker_result": None,  # (hit, hit_distance, source) fresh this frame
            "last_submit_params": None,    # For early-out when pose unchanged
            # Debug stats (PURE ENGINE - no fallback stats needed)
            "stats_submit": 0,
            "stats_skip_pending": 0,
            "stats_skip_unchanged": 0,
            "stats_fresh": 0,
            "stats_hold": 0,
            "stats_last_report": 0.0,
        }
        _VIEWS[op_key] = v
    return v


def cache_camera_worker_result(op_key, job_id, hit, hit_distance, hit_source="STATIC", calc_time_us=0.0):
    """
    Called by game loop when camera occlusion result arrives.
    Cache every result - we're submitting frequently for fresh data.
    """
    view = _view_state_for(op_key)

    # Cache every result that comes back (we submit every frame for freshness)
    result = (hit, hit_distance, hit_source)
    view["cached_worker_result"] = result
    view["last_static_calc_time_us"] = calc_time_us

    # Clear pending_job_id if this is the job we're waiting for
    pending = view.get("pending_job_id")
    if pending is not None and job_id == pending:
        view["pending_job_id"] = None
        view["pending_job_submit_time"] = 0.0

# -------------------------
# Helpers (occlusion + pushout)
# -------------------------
_STATIC_TOKEN = "__STATIC__"

def _multi_ray_min_hit(static_bvh, dynamic_bvh_map, origin, direction, max_dist, r_cam,
                       debug_timing=False):
    """
    Center ray only: treat camera like a point (then subtract r_cam).
    Returns (nearest_hit_distance, hit_obj_token, static_time_us, dynamic_time_us) or (None, None, 0, 0).
    dynamic_bvh_map: dict {obj: (bvh_like, approx_radius)}

    NOTE: Camera occlusion uses main thread for both static and dynamic.
    1-frame latency causes jumpiness because camera direction changes unpredictably with mouse input.
    """
    if direction.length <= 1e-9 or max_dist <= 1e-9:
        return (None, None, 0.0, 0.0)
    dnorm = direction.normalized()

    best = (None, None)
    static_time_us = 0.0
    dynamic_time_us = 0.0

    # Static geometry (main thread - camera needs instant feedback)
    if static_bvh:
        t0 = time.perf_counter() if debug_timing else 0
        hit = static_bvh.ray_cast(origin, dnorm, max_dist)
        if debug_timing:
            static_time_us = (time.perf_counter() - t0) * 1_000_000
        if hit and hit[0] is not None:
            best = (hit[3], _STATIC_TOKEN)

    # Dynamic movers (always main thread - need Blender's BVHTree)
    if dynamic_bvh_map:
        t0 = time.perf_counter() if debug_timing else 0
        # small prefilter radius for movers
        pf_pad = 0.05 + r_cam
        for obj, (bvh_like, approx_rad) in dynamic_bvh_map.items():
            try:
                # quick sphere segment test (center ray)
                center = obj.matrix_world.translation
                oc = center - origin
                t = oc.dot(dnorm)
                if t < -approx_rad or t > max_dist + approx_rad:
                    pass
                else:
                    # distance from segment to center
                    if t < 0.0:
                        closest = oc
                    elif t > max_dist:
                        closest = oc - dnorm * max_dist
                    else:
                        closest = oc - dnorm * t
                    if closest.length_squared <= (approx_rad + pf_pad) * (approx_rad + pf_pad):
                        h = bvh_like.ray_cast(origin, dnorm, max_dist)
                        if h and h[0] is not None:
                            d = h[3]
                            if best[0] is None or d < best[0]:
                                best = (d, obj)
            except Exception:
                # any bad obj/bvh should not tank the frame
                continue
        if debug_timing:
            dynamic_time_us = (time.perf_counter() - t0) * 1_000_000

    if best[0] is not None:
        return (best[0], best[1], static_time_us, dynamic_time_us)
    return (None, None, static_time_us, dynamic_time_us)


def _los_blocked(static_bvh, dynamic_bvh_map, a: Vector, b: Vector):
    """
    True if anything blocks the line segment a→b.
    dynamic_bvh_map: dict {obj: (bvh_like, approx_radius)}
    """
    d = b - a
    dist = d.length
    if dist <= 1e-9:
        return False
    dnorm = d / dist

    # Static
    if static_bvh:
        h = static_bvh.ray_cast(a, dnorm, max(0.0, dist - _LOS_EPS))
        if h and h[0] is not None:
            return True

    # Dynamic
    if dynamic_bvh_map:
        for _obj, (bvh_like, _r) in dynamic_bvh_map.items():
            try:
                h = bvh_like.ray_cast(a, dnorm, max(0.0, dist - _LOS_EPS))
                if h and h[0] is not None:
                    return True
            except Exception:
                continue

    return False


def _binary_search_clear_los(static_bvh, dynamic_bvh_map, anchor, direction, low, high, steps):
    """
    Find the nearest distance along 'direction' from 'anchor' that has clear LoS.
    dynamic_bvh_map: dict {obj: (bvh_like, approx_radius)}
    """
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
    """Tiny nearest-point pushout vs static + all active dynamic BVHs."""
    if radius <= 1.0e-6:
        return pos

    def push_once_bvh(bvh_like, p):
        try:
            res = bvh_like.find_nearest(p)
        except Exception:
            return p, False
        if not res or res[0] is None or res[1] is None:
            return p, False
        hit_co, hit_n, _idx, dist = res
        n = hit_n
        if (p - hit_co).dot(n) < 0.0:
            n = -n
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
def update_camera_for_operator(context, op):
    """
    The ONE function exp_loop should call each frame (no dynamic-map copying).
    Supports engine offloading for static geometry raycast (1-frame latency pattern).
    """
    if not op or not getattr(op, "target_object", None):
        import bpy
        if getattr(bpy.context.scene, "dev_debug_camera_offload", False):
            print("[Camera UPDATE SKIP] No operator or target object")
        return

    op_key = id(op)
    view   = _view_state_for(op_key)

    # --- 1) Direction from operator pitch/yaw
    pitch = float(op.pitch)
    yaw   = float(op.yaw)
    cx = math.cos(pitch); sx = math.sin(pitch)
    sy = math.sin(yaw);   cy = math.cos(yaw)
    direction = Vector((cx * sy, -cx * cy, sx))
    if direction.length > 1e-9:
        direction.normalize()

    # --- 2) Anchor at capsule top
    cp     = context.scene.char_physics
    cap_h  = float(getattr(cp, "height", 2.0))
    cap_r  = float(getattr(cp, "radius", 0.30))
    anchor = op.target_object.location + Vector((0.0, 0.0, cap_h))

    # --- 3) Camera "thickness" from near clip
    clip_start = float(getattr(op, "_clip_start_cached", 0.1) or 0.1)
    if clip_start <= 0.0:
        clip_start = 0.1
    r_cam = max(_R_CAM_FLOOR, clip_start * _NEARCLIP_TO_RADIUS_K)

    # --- 4) Desired boom + minimum
    desired_max = max(0.0, context.scene.orbit_distance + context.scene.zoom_factor)
    min_cam     = max(_MIN_CAM_ABS, cap_r * _MIN_CAM_RADIUS_FACTOR)

    # --- 5) Early-out (pose unchanged) to skip the heavy solve
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
            # reuse last allowed; only refresh anchor/dir for apply
            view["anchor"]    = anchor
            view["direction"] = direction
            _apply_to_viewport(context, op, view)
            return

    view["last_params"] = params
    view["last_time"]   = now

    # --- 6) Obstruction solve - ALL IN ENGINE (no main thread raycasting)
    # Still need BVH refs for LoS checks and pushout (these are quick operations)
    static_bvh = getattr(op, "bvh_tree", None)
    dynamic_bvh_map = getattr(op, "dynamic_bvh_map", None)

    # Check for fresh result this frame (must match current pending job)
    cached_result = view.get("cached_worker_result")
    hit_dist = None
    hit_token = None
    result_status = "NONE"

    # Check debug flag once
    import bpy
    camera_debug_enabled = getattr(bpy.context.scene, "dev_debug_camera_offload", False)

    if cached_result is not None:
        # PURE ENGINE: Use the result from the worker for primary raycast
        worker_hit, worker_dist, worker_source = cached_result[:3]

        if worker_hit and worker_dist is not None:
            hit_dist = worker_dist
            hit_token = _STATIC_TOKEN if worker_source == "STATIC" else "__DYNAMIC__"

        view["cached_worker_result"] = None  # Consume
        view["stats_fresh"] = view.get("stats_fresh", 0) + 1
        result_status = "FRESH"
    else:
        # No result for current direction - HOLD distance but update position/direction
        # CRITICAL: Must update anchor/direction so camera follows character even during HOLD
        #
        # WITH EXPLICIT SYNC: HOLD should be EXTREMELY RARE (<1% of frames)
        # HOLD only occurs if:
        # 1. Explicit polling timed out (worker took >3ms - VERY unusual)
        # 2. Engine is severely overloaded (job queue saturated)
        # 3. System is under extreme stress (CPU throttling, context switches)
        #
        # If HOLD is frequent (>5% of frames), this indicates a SERIOUS problem:
        # - Engine workers are failing to complete jobs in 3ms
        # - System performance is degraded
        # - Engine health should be investigated immediately
        #
        # HOLD preserves last known safe distance to prevent camera jumping, but can cause
        # clipping if character moves toward obstacles while in HOLD state.
        view["anchor"] = anchor
        view["direction"] = direction
        view["stats_hold"] = view.get("stats_hold", 0) + 1
        # Keep existing "allowed" distance (don't change zoom level)
        if camera_debug_enabled:
            print(f"[Camera HOLD] ⚠️ No fresh result - keeping allowed={view.get('allowed', 0):.3f}m (SHOULD BE RARE!)")
        _apply_to_viewport(context, op, view)
        return

    # Calculate base allowed distance from primary raycast
    if hit_dist is not None:
        base_allowed = max(min_cam, min(desired_max, hit_dist - r_cam))
        allowed = max(min_cam, base_allowed - (_EXTRA_PULL_METERS + _EXTRA_PULL_R_K * r_cam))
    else:
        allowed = desired_max

    # --- LoS check (main thread - Blender BVH is fast)
    candidate = anchor + direction * allowed
    if _los_blocked(static_bvh, dynamic_bvh_map, anchor, candidate):
        allowed = _binary_search_clear_los(static_bvh, dynamic_bvh_map, anchor, direction,
                                           low=min_cam, high=allowed, steps=_LOS_STEPS)
        candidate = anchor + direction * allowed

    # --- Pushout (main thread - Blender BVH is fast)
    candidate = _camera_sphere_pushout_any(static_bvh, dynamic_bvh_map, candidate, r_cam, max_iters=_PUSHOUT_ITERS)
    allowed_after_push = (candidate - anchor).length

    # --- 9) Latch + smoothing
    latched = _latch_for(op_key).filter(hit_token, max(min_cam, min(allowed_after_push, desired_max)), r_cam)
    final_allowed = _smooth_for(op_key).filter(latched)

    # Cache for apply and for external readbacks if needed
    view["anchor"]    = anchor
    view["direction"] = direction
    view["allowed"]   = float(final_allowed)
    op._cam_allowed_last = float(final_allowed)

    # --- 10) Debug output (with Hz frequency gating)
    if should_print_debug("camera_offload"):
        if hit_dist is not None:
            src = "STATIC" if hit_token == _STATIC_TOKEN else "DYNAMIC"
            result_info = f"HIT({src})={hit_dist:.3f}m"
        else:
            result_info = "MISS"

        if camera_debug_enabled:
            final_info = f"allowed={final_allowed:.3f}m"
            print(f"[Camera] {result_status}: {result_info} → {final_info}")

    # Print stats summary every 2 seconds (always, regardless of Hz gate)
    now = time.perf_counter()
    if camera_debug_enabled and now - view.get("stats_last_report", 0.0) >= 2.0:
        view["stats_last_report"] = now
        total_submit = view.get("stats_submit", 0)
        skip_pending = view.get("stats_skip_pending", 0)
        skip_unchanged = view.get("stats_skip_unchanged", 0)
        fresh = view.get("stats_fresh", 0)
        hold = view.get("stats_hold", 0)
        timeout = view.get("stats_timeout", 0)
        total_frames = fresh + hold

        if total_frames > 0:
            fresh_pct = 100.0 * fresh / total_frames
            hold_pct = 100.0 * hold / total_frames
            timeout_pct = 100.0 * timeout / total_submit if total_submit > 0 else 0.0

            print(f"[Camera STATS] Submit: {total_submit} | Timeout: {timeout} ({timeout_pct:.1f}%) | SkipPending: {skip_pending} | SkipUnchanged: {skip_unchanged}")
            print(f"[Camera STATS] FRESH: {fresh} ({fresh_pct:.1f}%) | HOLD: {hold} ({hold_pct:.1f}%)")

            # Warning if HOLD or timeout is too high
            if hold_pct > 5.0:
                print(f"[Camera WARNING] ⚠️ HOLD rate {hold_pct:.1f}% is HIGH - engine may be struggling!")
            if timeout_pct > 5.0:
                print(f"[Camera WARNING] ⚠️ Timeout rate {timeout_pct:.1f}% is HIGH - workers taking >3ms!")

    # --- 11) Apply to ONE cached rv3d, skipping redundant writes
    _apply_to_viewport(context, op, view)


def _apply_to_viewport(context, op, view):
    anchor    = view.get("anchor");    direction = view.get("direction")
    desired_d = view.get("allowed", 3.0)
    if anchor is None or direction is None:
        return

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

    target_rot = direction.to_track_quat('Z', 'Y')

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


# -------------------------
# Same-frame engine offloading
# -------------------------
# Submit job EARLY in frame, poll LATE - workers complete in ~100-200µs,
# physics takes ~1-5ms, so result is ready by camera update.

def submit_camera_occlusion_early(op, context):
    """
    Submit camera occlusion job at START of frame (before physics).
    Called from game loop BEFORE physics integration.
    Includes both static (cached grid) and dynamic (serialized triangles) geometry.

    Returns:
        job_id if submitted successfully, None otherwise
    """
    if not op or not getattr(op, "target_object", None):
        import bpy
        if getattr(bpy.context.scene, "dev_debug_camera_offload", False):
            print("[Camera SKIP] No operator or target object")
        return None

    engine = getattr(op, "engine", None)
    if not engine:
        import bpy
        if getattr(bpy.context.scene, "dev_debug_camera_offload", False):
            print("[Camera SKIP] No engine available")
        return None

    op_key = id(op)
    view = _view_state_for(op_key)

    # Check debug flag once at start
    import bpy
    debug_camera = getattr(bpy.context.scene, "dev_debug_camera_offload", False)

    # NOTE: Old timeout check removed - no longer needed with explicit polling
    # Explicit polling in game loop handles timeouts properly (3ms timeout with active wait)
    # If a job times out during polling, it's handled gracefully there

    # Calculate current camera params
    pitch = float(op.pitch)
    yaw = float(op.yaw)
    cx = math.cos(pitch); sx = math.sin(pitch)
    sy = math.sin(yaw); cy = math.cos(yaw)
    direction = Vector((cx * sy, -cx * cy, sx))
    if direction.length > 1e-9:
        direction.normalize()

    cp = context.scene.char_physics
    cap_h = float(getattr(cp, "height", 2.0))
    cap_r = float(getattr(cp, "radius", 0.30))
    anchor = op.target_object.location + Vector((0.0, 0.0, cap_h))

    desired_max = max(0.0, context.scene.orbit_distance + context.scene.zoom_factor)

    # Early-out: DISABLED - was causing timing-dependent behavior
    # TODO: Re-enable with proper dynamic mesh tracking after debug investigation
    # last_submit = view.get("last_submit_params")
    # if last_submit is not None:
    #     lp, ly, lax, lay, laz, ldm = last_submit
    #     pose_unchanged = (
    #         abs(pitch - lp) < 0.0015 and
    #         abs(yaw - ly) < 0.0015 and
    #         abs(anchor.x - lax) < 0.008 and
    #         abs(anchor.y - lay) < 0.008 and
    #         abs(anchor.z - laz) < 0.008 and
    #         abs(desired_max - ldm) < 0.010
    #     )
    #     if pose_unchanged:
    #         view["stats_skip_unchanged"] = view.get("stats_skip_unchanged", 0) + 1
    #         if debug_camera:
    #             print(f"[Camera SKIP] pose unchanged (Δpitch={abs(pitch-lp):.4f} Δyaw={abs(yaw-ly):.4f})")
    #         return

    # Gather dynamic mesh triangles (world space)
    dynamic_bvh_map = getattr(op, "dynamic_bvh_map", None)
    dynamic_triangles = []
    dynamic_obj_count = 0

    if dynamic_bvh_map:
        for obj, (bvh_like, approx_rad) in dynamic_bvh_map.items():
            dynamic_obj_count += 1
            try:
                # Quick sphere check - skip if object too far
                center = obj.matrix_world.translation
                to_obj = center - anchor
                along_ray = to_obj.dot(direction)
                if along_ray < -approx_rad or along_ray > desired_max + approx_rad:
                    continue

                # Get mesh and transform to world space
                mesh = obj.data
                matrix = obj.matrix_world
                verts = [matrix @ v.co for v in mesh.vertices]

                for poly in mesh.polygons:
                    if len(poly.vertices) >= 3:
                        # Triangulate faces
                        v0 = verts[poly.vertices[0]]
                        for i in range(1, len(poly.vertices) - 1):
                            v1 = verts[poly.vertices[i]]
                            v2 = verts[poly.vertices[i + 1]]
                            dynamic_triangles.append((
                                (v0.x, v0.y, v0.z),
                                (v1.x, v1.y, v1.z),
                                (v2.x, v2.y, v2.z)
                            ))
            except Exception:
                continue

    # Submit job with static and dynamic data (LoS+Pushout done on main thread)
    job_data = {
        "ray_origin": (anchor.x, anchor.y, anchor.z),
        "ray_direction": (direction.x, direction.y, direction.z),
        "max_distance": float(desired_max),
        "dynamic_triangles": dynamic_triangles,
    }
    job_id = engine.submit_job("CAMERA_OCCLUSION_FULL", job_data)
    view["pending_job_id"] = job_id
    view["pending_job_submit_time"] = time.perf_counter()
    view["stats_submit"] = view.get("stats_submit", 0) + 1

    if debug_camera:
        print(f"[Camera SUBMIT] job={job_id} origin=({anchor.x:.2f},{anchor.y:.2f},{anchor.z:.2f}) dir=({direction.x:.2f},{direction.y:.2f},{direction.z:.2f}) max={desired_max:.2f}m dynamic_objs={dynamic_obj_count} dynamic_tris={len(dynamic_triangles)}")

    return job_id  # Return job_id for explicit polling


def poll_camera_result_with_timeout(op, context, job_id, timeout=0.003):
    """
    Explicitly poll for camera occlusion result with timeout (same-frame pattern).
    Similar to KCC physics polling - waits for specific job to complete.

    This ensures camera gets fresh data every frame without relying on timing assumptions.

    Args:
        op: Operator with engine reference
        context: Blender context
        job_id: Specific camera job ID to wait for
        timeout: Max time to wait in seconds (default 3ms)

    Returns:
        True if result received and cached, False if timeout
    """
    if job_id is None:
        return False

    engine = getattr(op, "engine", None)
    if not engine:
        return False

    op_key = id(op)
    view = _view_state_for(op_key)

    # Check debug flag
    import bpy
    debug_camera = getattr(bpy.context.scene, "dev_debug_camera_offload", False)

    # CRITICAL: Small guaranteed delay before polling starts
    # This gives the worker time to:
    # 1. Pick up the job from the queue (queue operations take ~10-20µs)
    # 2. Start processing (worker thread needs CPU time slice)
    # 3. Get a few microseconds of execution before we start hammering with polls
    #
    # Without this, when debug is OFF, the polling loop runs SO FAST that it can:
    # - Start before worker even grabs the job from queue
    # - Starve worker of CPU time by constantly polling
    # - Miss the result window due to race conditions
    #
    # When debug was ON, print statements accidentally provided this delay.
    # Now it's explicit and deterministic: 150µs is enough for worker to start,
    # but small enough to maintain low latency (<0.5% of 33ms frame).
    time.sleep(0.00015)  # 150µs pre-poll delay

    poll_start = time.perf_counter()
    result_found = False
    poll_count = 0

    if debug_camera:
        print(f"[Camera POLL START] Waiting for job={job_id} (timeout={timeout*1000:.1f}ms) [+150µs pre-delay]")

    while True:
        elapsed = time.perf_counter() - poll_start
        if elapsed >= timeout:
            if debug_camera:
                print(f"[Camera POLL TIMEOUT] job={job_id} not ready after {elapsed*1000:.1f}ms ({poll_count} polls)")
            break

        results = engine.poll_results(max_results=20)
        for result in results:
            if result.job_id == job_id and result.job_type == "CAMERA_OCCLUSION_FULL":
                # Found our camera job!
                if result.success:
                    # Cache result for update_camera_for_operator to use
                    hit = result.result.get("hit", False)
                    hit_distance = result.result.get("hit_distance", None)
                    hit_source = result.result.get("hit_source", "STATIC")
                    calc_time_us = result.result.get("calc_time_us", 0.0)

                    cache_camera_worker_result(op_key, job_id, hit, hit_distance, hit_source, calc_time_us)

                    poll_time_us = elapsed * 1_000_000

                    if debug_camera:
                        status = f"HIT({hit_source}) dist={hit_distance:.3f}m" if hit else "MISS"
                        print(f"[Camera POLL SUCCESS] job={job_id} {status} | "
                              f"worker={calc_time_us:.0f}µs poll={poll_time_us:.0f}µs ({poll_count} polls)")

                    result_found = True
                else:
                    # Job failed
                    if debug_camera:
                        print(f"[Camera POLL FAILED] job={job_id} error: {result.error}")
                break
            else:
                # Cache non-camera results for other systems
                # (this is important - don't drop other results!)
                pass

        if result_found:
            break

        poll_count += 1

        # Adaptive sleep: Always sleep a tiny bit to give workers CPU time
        # This prevents the polling loop from starving worker threads
        # 30µs is small enough for low latency but gives workers breathing room
        if poll_count >= 2:  # Start sleeping after first poll
            time.sleep(0.00003)  # 30µs - balance between responsiveness and CPU sharing

    # Update stats
    if result_found:
        view["stats_fresh"] = view.get("stats_fresh", 0) + 1
    else:
        view["stats_timeout"] = view.get("stats_timeout", 0) + 1

    return result_found


