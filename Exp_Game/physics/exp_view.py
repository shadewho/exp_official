# Exploratory/Exp_Game/physics/exp_view.py
import math
import mathutils
from mathutils import Vector
from .exp_raycastutils import raycast_closest_any
from ..props_and_utils.exp_time import get_game_time
# ===========================
# Tunables you can tweak
# ===========================
_FRUSTUM_RING_RADIUS  = 1.0    # ring radius in units of r_cam
_PUSHOUT_ITERS        = 1      # tiny nearest-point pushout passes
_LOS_BINARY_STEPS     = 1     # bsearch steps for nearest clear LoS
_LOS_EPS              = 1.0e-4

# New (safe) tuning helpers
# Ring is only used when the center ray reports a hit. This is enough because LoS and pushout
# cover the other cases. You can lower this further if needed.
_RING_SAMPLES_ON_HIT  = 1

# Absolute minimum camera distance from anchor (meters)
# Final minimum is: max(MIN_CAM_ABS, cap_r * MIN_CAM_RADIUS_FACTOR)
MIN_CAM_ABS = 0.0006           # 0.6 mm absolute physical floor
MIN_CAM_RADIUS_FACTOR = 0.04   # 4% of capsule radius

# Camera "thickness": derived from viewport near clip with a small floor
NEARCLIP_TO_RADIUS = 0.60      # camera radius ~60% of clip_start
R_CAM_FLOOR        = 0.008     # 8 mm minimum camera thickness

# When we do collide, pull a bit more inward as a safety buffer
EXTRA_PULL_METERS  = 0.25      # ~25 cm additional inward pull on hit
EXTRA_PULL_K       = 2.0       # plus 2× r_cam inward

# If still blocked at the hard minimum, allow a very small peek upward
MICRO_LEAN_UP_MAX              = 0.06   # ≤ 6 cm upward nudge
MICRO_LEAN_UP_FRAC_OF_RADIUS   = 0.20   # or 20% of capsule radius



# ------------------------------------------------------------
# Camera temporal filters (outward-only rate-limit + occlusion latch)
# ------------------------------------------------------------
class _CamSmoother:
    """
    Outward-only rate limiter:
      - Inward changes (obstruction) are immediate.
      - Outward changes are capped at OUTWARD_RATE m/s.
    """
    __slots__ = ("last_allowed", "last_t")
    OUTWARD_RATE = 10.0  # tune 6..14 m/s as desired

    def __init__(self):
        self.last_allowed = None
        self.last_t = get_game_time()

    def filter(self, target: float) -> float:
        now = get_game_time()
        dt = max(1e-6, now - self.last_t)
        if self.last_allowed is None:
            self.last_allowed = target
        else:
            if target >= self.last_allowed:
                self.last_allowed = min(target, self.last_allowed + self.OUTWARD_RATE * dt)
            else:
                self.last_allowed = target  # immediate pull-in
        self.last_t = now
        return self.last_allowed


class _CamLatch:
    """
    Occlusion Schmitt trigger:
      - When an object blocks, we 'latch' that blocked distance.
      - We hold it briefly and only release when there is a bit of extra clearance.
    """
    __slots__ = ("latched_obj", "latched_until", "latched_allowed")

    # release needs at least this much extra clearance beyond the latched distance
    RELEASE_PAD_MIN = 0.06   # meters
    RELEASE_PAD_K   = 1.6    # × r_cam
    HOLD_TIME       = 0.14   # seconds

    def __init__(self):
        self.latched_obj = None     # dynamic object or "__STATIC__"
        self.latched_until = 0.0
        self.latched_allowed = None

    def filter(self, hit_obj_token, allowed_now: float, r_cam: float) -> float:
        now = get_game_time()

        # Acquire/refresh latch when blocked
        if hit_obj_token is not None:
            self.latched_obj = hit_obj_token
            self.latched_allowed = allowed_now
            self.latched_until = now + self.HOLD_TIME
            return allowed_now

        # No current hit: maybe keep holding the last latch a bit
        if self.latched_obj is not None and self.latched_allowed is not None:
            need_pad = max(self.RELEASE_PAD_MIN, self.RELEASE_PAD_K * r_cam)
            if (now < self.latched_until) and (allowed_now < (self.latched_allowed + need_pad)):
                # stay latched at the last blocked length
                return self.latched_allowed
            # release
            self.latched_obj = None
            self.latched_allowed = None
            self.latched_until = 0.0

        return allowed_now


# per-character caches
_SMOOTHERS = {}
_LATCHES   = {}

def _smooth_for(char_obj):
    key = getattr(char_obj, "name", char_obj) if char_obj else "__global__"
    sm = _SMOOTHERS.get(key)
    if sm is None:
        sm = _CamSmoother()
        _SMOOTHERS[key] = sm
    return sm

def _latch_for(char_obj):
    key = getattr(char_obj, "name", char_obj) if char_obj else "__global__"
    lt = _LATCHES.get(key)
    if lt is None:
        lt = _CamLatch()
        _LATCHES[key] = lt
    return lt


# -------------------------
# Helpers
# -------------------------
def _get_view_clip_start(context, fallback=0.1):
    best = None
    try:
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                sv3d = area.spaces.active
                cs = getattr(sv3d, "clip_start", None)
                if cs and cs > 0.0:
                    best = cs if best is None else min(best, cs)
    except Exception:
        pass
    return float(best) if (best is not None and best > 0.0) else float(fallback)

def _build_basis(direction: Vector):
    up_world = Vector((0,0,1))
    helper = Vector((1,0,0)) if abs(direction.dot(up_world)) > 0.98 else up_world
    right = direction.cross(helper)
    right = right.normalized() if right.length > 1e-9 else Vector((1,0,0))
    up = right.cross(direction).normalized()
    return right, up


# -------------------------
# Low-cost dynamic prefilters
# ------------------------

_STATIC_TOKEN = "__STATIC__"

def _multi_ray_min_hit(static_bvh, dynamic_bvh_map, origin, direction, max_dist, r_cam):
    """
    Returns (nearest_hit_distance, hit_obj_token) or (None, None).
    hit_obj_token is either a dynamic object, or "__STATIC__" for static BVH.
    We keep the center ray and then a compact ring when the center hits.
    """
    if direction.length <= 1e-9 or max_dist <= 1e-9:
        return (None, None)

    dnorm = direction.normalized()

    # Center ray
    hl, hn, ho, hd = raycast_closest_any(static_bvh, dynamic_bvh_map, origin, dnorm, max_dist)
    if hl is None:
        return (None, None)

    best_dist = hd
    best_obj  = (_STATIC_TOKEN if ho is None else ho)

    # Only ring-sample if center hits (keeps perf bounded)
    right = direction.cross(Vector((0,0,1)))
    if right.length <= 1e-9:
        right = Vector((1,0,0))
    else:
        right.normalize()
    up = right.cross(dnorm).normalized()

    ring_r = _FRUSTUM_RING_RADIUS * r_cam
    samples = max(1, int(_RING_SAMPLES_ON_HIT))
    step = (2.0 * math.pi) / float(samples)

    for i in range(samples):
        ang = i * step
        offset = (math.cos(ang) * right + math.sin(ang) * up) * ring_r
        o = origin + offset
        h = raycast_closest_any(static_bvh, dynamic_bvh_map, o, dnorm, max_dist)
        if h and h[0] is not None and h[3] < best_dist:
            best_dist = h[3]
            best_obj  = (_STATIC_TOKEN if h[2] is None else h[2])

    return (best_dist, best_obj)

def _los_blocked(static_bvh, dynamic_bvh_map, a: Vector, b: Vector, r_cam: float):
    d = b - a
    dist = d.length
    if dist <= 1e-9:
        return False
    dnorm = d / dist
    h = raycast_closest_any(static_bvh, dynamic_bvh_map, a, dnorm, max(0.0, dist - _LOS_EPS))
    return (h[0] is not None)

def _binary_search_clear_los(static_bvh, dynamic_bvh_map, anchor, direction, low, high, steps, r_cam):
    lo, hi = low, high
    for _ in range(max(1, int(steps))):
        mid = 0.5 * (lo + hi)
        cam = anchor + direction * mid
        if _los_blocked(static_bvh, dynamic_bvh_map, anchor, cam, r_cam):
            hi = mid
        else:
            lo = mid
    return lo  # closest guaranteed-clear distance (>=low)

def _camera_sphere_pushout_any(static_bvh, dynamic_bvh_map, pos, radius, max_iters=_PUSHOUT_ITERS):
    """
    Tiny nearest-point pushout against static BVH and *all* active LocalBVHs.
    Treat dynamic exactly like static to avoid camera embedding on moving meshes.
    """
    if radius <= 1e-6:
        return pos

    def push_once(bvh_like, p):
        try:
            res = bvh_like.find_nearest(p)
        except Exception:
            return p, False
        if not res or res[0] is None or res[1] is None:
            return p, False

        hit_co, hit_n, _, dist = res
        n = hit_n
        if (p - hit_co).dot(n) < 0.0:
            n = -n
        if dist < radius:
            return p + n * ((radius - dist) + 1.0e-4), True
        return p, False

    p = pos

    # Iterate until no movement — exact same behavior applied to both static and dynamic
    for _ in range(max_iters):
        moved = False
        if static_bvh:
            p, m = push_once(static_bvh, p); moved = moved or m
        if dynamic_bvh_map:
            # IMPORTANT: test against all active dynamic BVHs (no prefilter)
            for _, (bvh_like, _approx_rad) in dynamic_bvh_map.items():
                p, m = push_once(bvh_like, p); moved = moved or m
        if not moved:
            break
    return p


# -------------------------
# Public API (signature unchanged)
# -------------------------
def update_view(context, obj, pitch, yaw, bvh_tree, orbit_distance, zoom_factor, dynamic_bvh_map=None):
    """
    Third-person camera with dynamic occluders:
      • Full static + dynamic occlusion
      • Occlusion latch (Schmitt trigger) to kill edge-jitter on moving proxies
      • Outward-only temporal smoothing
      • Tiny pushout near thin geometry
      • Skips redundant VIEW_3D writes (micro thresholds) when this API is used directly
    """
    if not obj:
        return

    # Direction from anchor toward desired camera
    direction = Vector((
        math.cos(pitch) * math.sin(yaw),
        -math.cos(pitch) * math.cos(yaw),
        math.sin(pitch)
    ))
    if direction.length > 1e-9:
        direction.normalize()

    # Capsule-top anchor
    cp = context.scene.char_physics
    cap_h = getattr(cp, "height", 2.0)
    cap_r = getattr(cp, "radius", 0.30)
    anchor = obj.location + Vector((0.0, 0.0, cap_h))

    # Camera “thickness” from near clip
    clip_start = _get_view_clip_start(context, fallback=0.1)
    r_cam = max(R_CAM_FLOOR, clip_start * NEARCLIP_TO_RADIUS)

    # Desired max boom from UI
    desired_max = max(0.0, orbit_distance + zoom_factor)
    min_cam     = max(MIN_CAM_ABS, cap_r * MIN_CAM_RADIUS_FACTOR)

    # 1) First hard obstruction along the boom
    hit_dist, hit_token = _multi_ray_min_hit(bvh_tree, dynamic_bvh_map, anchor, direction, desired_max, r_cam)
    if hit_dist is not None:
        base_allowed = max(min_cam, min(desired_max, hit_dist - r_cam))
        allowed      = max(min_cam, base_allowed - (EXTRA_PULL_METERS + EXTRA_PULL_K * r_cam))
    else:
        allowed = desired_max

    # 2) Guarantee clear LoS to the chosen point (binary-search inward)
    candidate_cam = anchor + direction * allowed
    if _los_blocked(bvh_tree, dynamic_bvh_map, anchor, candidate_cam, r_cam):
        allowed = _binary_search_clear_los(
            bvh_tree, dynamic_bvh_map, anchor, direction,
            low=min_cam, high=allowed, steps=_LOS_BINARY_STEPS, r_cam=r_cam
        )
        candidate_cam = anchor + direction * allowed

    # 3) Tiny nearest-point pushout to avoid embedding in thin geometry
    candidate_cam = _camera_sphere_pushout_any(bvh_tree, dynamic_bvh_map, candidate_cam, r_cam, max_iters=_PUSHOUT_ITERS)

    # 4) Latch + smoothing (order matters)
    allowed_after_push = (candidate_cam - anchor).length
    latched_allowed = _latch_for(obj).filter(hit_token, max(min_cam, min(allowed_after_push, desired_max)), r_cam)
    final_allowed   = _smooth_for(obj).filter(latched_allowed)

    # === Apply to the viewport (skip redundant writes) ===
    POS_EPS  = 1e-4
    ANG_EPS  = 1e-4
    DIST_EPS = 1e-4

    target_rot = direction.to_track_quat('Z', 'Y')

    for area in context.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        rv3d = area.spaces.active.region_3d

        # Fetch last applied (stash on the rv3d object via custom attrs)
        last_loc = getattr(rv3d, "_exp_last_loc", None)
        last_rot = getattr(rv3d, "_exp_last_rot", None)
        last_dst = getattr(rv3d, "_exp_last_dst", None)

        need_loc = True if last_loc is None else (anchor - last_loc).length > POS_EPS
        need_rot = True
        if last_rot is not None:
            dq = last_rot.rotation_difference(target_rot)
            need_rot = abs(dq.angle) > ANG_EPS
        need_dst = True if last_dst is None else abs(final_allowed - last_dst) > DIST_EPS

        if need_loc:
            rv3d.view_location = anchor
            setattr(rv3d, "_exp_last_loc", anchor.copy())
        if need_rot:
            rv3d.view_rotation = target_rot
            setattr(rv3d, "_exp_last_rot", target_rot.copy())
        if need_dst:
            rv3d.view_distance = final_allowed
            setattr(rv3d, "_exp_last_dst", float(final_allowed))


def shortest_angle_diff(current, target):
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return diff


def compute_camera_allowed_distance(
    char_key,
    anchor,
    direction,
    r_cam: float,
    desired_max: float,
    min_cam: float,
    static_bvh,
    dynamic_bvh_map,
) -> float:
    """
    Thread‑safe: no bpy/context. Uses your existing helpers:
      _multi_ray_min_hit, _los_blocked, _binary_search_clear_los, _camera_sphere_pushout_any,
      _smooth_for, _latch_for, get_game_time()
    Returns a single float: final allowed boom length.
    """
    # 1) First obstruction along the boom
    hit_dist, hit_token = _multi_ray_min_hit(static_bvh, dynamic_bvh_map, anchor, direction, desired_max, r_cam)
    if hit_dist is not None:
        base_allowed = max(min_cam, min(desired_max, hit_dist - r_cam))
        allowed      = max(min_cam, base_allowed - (EXTRA_PULL_METERS + EXTRA_PULL_K * r_cam))
    else:
        allowed = desired_max

    # 2) Ensure clear LoS (binary search)
    candidate_cam = anchor + direction * allowed
    if _los_blocked(static_bvh, dynamic_bvh_map, anchor, candidate_cam, r_cam):
        allowed = _binary_search_clear_los(static_bvh, dynamic_bvh_map, anchor, direction,
                                           low=min_cam, high=allowed, steps=_LOS_BINARY_STEPS, r_cam=r_cam)
        candidate_cam = anchor + direction * allowed

    # 3) Tiny pushout vs. thin geo
    candidate_cam = _camera_sphere_pushout_any(static_bvh, dynamic_bvh_map, candidate_cam, r_cam, max_iters=_PUSHOUT_ITERS)
    allowed_after_push = (candidate_cam - anchor).length

    # 4) Latch + smoothing (identical semantics to update_view)
    latched_allowed = _latch_for(char_key).filter(hit_token, max(min_cam, min(allowed_after_push, desired_max)), r_cam)
    final_allowed   = _smooth_for(char_key).filter(latched_allowed)
    return float(final_allowed)