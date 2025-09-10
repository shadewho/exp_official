# Exploratory/Exp_Game/physics/exp_view.py
import math
import mathutils
from mathutils import Vector
from .exp_raycastutils import raycast_closest_any

# ===========================
# Tunables you can tweak (no addon properties touched)
# ===========================
_FRUSTUM_RING_SAMPLES = 16     # legacy max ring rays (kept for compatibility)
_FRUSTUM_RING_RADIUS  = 1.0    # ring radius in units of r_cam
_PUSHOUT_ITERS        = 3      # tiny nearest-point pushout passes
_LOS_BINARY_STEPS     = 10     # bsearch steps for nearest clear LoS
_LOS_EPS              = 1.0e-4

# New (safe) tuning helpers
# Ring is only used when the center ray reports a hit. This is enough because LoS and pushout
# cover the other cases. You can lower this further if needed.
_RING_SAMPLES_ON_HIT  = 8

# Prefilter margins:
#   - Rays: allow a little slack so we don't accidentally cull near‑misses.
#   - Pushout: consider only dynamics within (camera radius + margin).
_DYNAMIC_PREFILTER_MARGIN = 0.40   # meters
_PUSHOUT_NEAR_MARGIN      = 0.60   # meters

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
# -------------------------
def _prefilter_dynamic_by_ray(dynamic_bvh_map, origin: Vector, dnorm: Vector, max_dist: float, r_cam: float, margin: float):
    """
    Return a subset dict {obj: (bvh_like, rad)} of dynamic_bvh_map that the ray could plausibly hit.
    Uses a cheap cylinder test against each object's origin (world translation) with radius≈(obj_rad + r_cam + margin).
    """
    if not dynamic_bvh_map:
        return None
    subset = {}
    md = max(0.0, float(max_dist) + margin)
    for obj, (bvh_like, approx_rad) in dynamic_bvh_map.items():
        # Approximate center from object origin (cheap and ok for a coarse gate)
        c = obj.matrix_world.translation
        oc = c - origin
        t = oc.dot(dnorm)
        if t < -margin or t > md:
            continue
        # perpendicular distance from ray to center
        perp = (oc - dnorm * t).length
        if perp <= (approx_rad + r_cam + margin):
            subset[obj] = (bvh_like, approx_rad)
    return subset if subset else None

def _prefilter_dynamic_by_point(dynamic_bvh_map, point: Vector, reach: float):
    """
    Return a subset dict of dynamics within (approx_rad + reach) of 'point'.
    Used to localize nearest-point pushout.
    """
    if not dynamic_bvh_map:
        return None
    subset = {}
    r = float(reach)
    for obj, (bvh_like, approx_rad) in dynamic_bvh_map.items():
        if (obj.matrix_world.translation - point).length <= (approx_rad + r):
            subset[obj] = (bvh_like, approx_rad)
    return subset if subset else None


def _raycast_closest_any_prefiltered(static_bvh, dynamic_bvh_map, origin: Vector, dnorm: Vector, max_dist: float, r_cam: float, margin: float):
    """
    Same output contract as BVH.ray_cast wrapper you use elsewhere:
    Returns: (hit_loc, hit_norm, hit_obj, hit_dist) or (None, None, None, None)
    But only checks a small subset of dynamic BVHs.
    """
    best = (None, None, None, 1e9)

    # Static: unchanged
    if static_bvh:
        hit = static_bvh.ray_cast(origin, dnorm, max_dist)
        if hit and hit[0] is not None and hit[3] < best[3]:
            best = (hit[0], hit[1], None, hit[3])

    # Dynamics: prefiltered subset only
    sub = _prefilter_dynamic_by_ray(dynamic_bvh_map, origin, dnorm, max_dist, r_cam, margin)
    if sub:
        for obj, (bvh_like, _) in sub.items():
            h = bvh_like.ray_cast(origin, dnorm, max_dist)
            if h and h[0] is not None and h[3] < best[3]:
                best = (h[0], h[1], obj, h[3])

    return best if best[0] is not None else (None, None, None, None)


def _multi_ray_min_hit(static_bvh, dynamic_bvh_map, origin, direction, max_dist, r_cam):
    """
    Returns nearest hit distance along 'direction' using:
      1) a center ray,
      2) only if center hits, a ring of offset rays (same accuracy for static & dynamic).
    None => no hit within max_dist.
    """
    if direction.length <= 1e-9 or max_dist <= 1e-9:
        return None

    dnorm = direction.normalized()

    # Center ray — use unified full check (static + all dynamic)
    hl, hn, ho, hd = raycast_closest_any(static_bvh, dynamic_bvh_map, origin, dnorm, max_dist)
    if hl is None:
        return None

    best = hd

    # If center hits, sample a ring to emulate camera "thickness" near edges
    right, up = _build_basis(dnorm)
    ring_r = _FRUSTUM_RING_RADIUS * r_cam
    samples = max(1, int(_RING_SAMPLES_ON_HIT))
    step = (2.0 * math.pi) / float(samples)

    for i in range(samples):
        ang = i * step
        offset = (math.cos(ang) * right + math.sin(ang) * up) * ring_r
        o = origin + offset
        h = raycast_closest_any(static_bvh, dynamic_bvh_map, o, dnorm, max_dist)
        if h and h[0] is not None and h[3] < best:
            best = h[3]

    return best

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
    Ultra-close third-person camera that:
      • Uses capsule-top as anchor
      • Center ray + small adaptive ring near obstacles (fast)
      • Binary-searches to nearest clear LoS (prefiltered dynamics)
      • Adds a small inward safety buffer on hits
      • Pushes out of thin geometry slightly (nearby dynamics only)
      • Positions the viewport correctly (pivot=anchor, distance=allowed)
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

    # Capsule-aware anchor (top of the capsule)
    cp = context.scene.char_physics
    cap_h = getattr(cp, "height", 1.80)
    cap_r = getattr(cp, "radius", 0.22)
    anchor = obj.location + Vector((0.0, 0.0, cap_h))

    # Desired boom length from UI inputs
    desired_max = max(0.0, orbit_distance + zoom_factor)

    # Camera "thickness" from near clip => r_cam
    clip_start = _get_view_clip_start(context, fallback=0.1)
    r_cam = max(R_CAM_FLOOR, clip_start * NEARCLIP_TO_RADIUS)

    # Absolute minimum approach distance
    min_cam = max(MIN_CAM_ABS, cap_r * MIN_CAM_RADIUS_FACTOR)

    # 1) Find earliest obstruction along the boom
    hit_dist = _multi_ray_min_hit(bvh_tree, dynamic_bvh_map, anchor, direction, desired_max, r_cam)
    if hit_dist is not None:
        # distance to surface minus camera radius, plus our extra inward buffer
        base_allowed = max(min_cam, min(desired_max, hit_dist - r_cam))
        allowed = max(
            min_cam,
            base_allowed - (EXTRA_PULL_METERS + EXTRA_PULL_K * r_cam)
        )
    else:
        allowed = desired_max

    # 2) Ensure clear LoS from anchor to the chosen point (prefiltered dynamics)
    candidate_cam = anchor + direction * allowed
    if _los_blocked(bvh_tree, dynamic_bvh_map, anchor, candidate_cam, r_cam):
        allowed = _binary_search_clear_los(
            bvh_tree, dynamic_bvh_map, anchor, direction,
            low=min_cam, high=allowed, steps=_LOS_BINARY_STEPS, r_cam=r_cam
        )
        candidate_cam = anchor + direction * allowed

    # 3) If STILL blocked at the hard minimum, micro-lean upward a hair to peek
    if _los_blocked(bvh_tree, dynamic_bvh_map, anchor, candidate_cam, r_cam) and allowed <= (min_cam + 1e-6):
        up_nudge = min(MICRO_LEAN_UP_MAX, cap_r * MICRO_LEAN_UP_FRAC_OF_RADIUS)
        candidate_cam = candidate_cam + Vector((0, 0, up_nudge))

    # 4) Tiny nearest-point pushout to avoid embedding in thin geometry (near dynamics only)
    candidate_cam = _camera_sphere_pushout_any(bvh_tree, dynamic_bvh_map, candidate_cam, r_cam, max_iters=_PUSHOUT_ITERS)

    # === Crucial: position the viewport correctly ===
    # In Blender, the eye is:
    #     eye = view_location - (rotated -Z) * view_distance
    # We want the *pivot/target* at the anchor, and the eye to be at candidate_cam.
    # With rotation such that rotated Z == direction, rotated -Z == -direction,
    # setting view_distance = allowed yields eye = anchor + direction*allowed.
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            rv3d = area.spaces.active.region_3d
            rv3d.view_location = anchor
            rv3d.view_rotation = direction.to_track_quat('Z', 'Y')
            rv3d.view_distance = allowed


def shortest_angle_diff(current, target):
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return diff
