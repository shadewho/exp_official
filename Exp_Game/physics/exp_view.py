# Exploratory/Exp_Game/physics/exp_view.py
import math
import mathutils
from mathutils import Vector
from .exp_raycastutils import raycast_closest_any

# ===========================
# Tunables you can tweak (no addon properties touched)
# ===========================
_FRUSTUM_RING_SAMPLES = 16     # how many rays around the center ray
_FRUSTUM_RING_RADIUS  = 1.0    # ring radius in units of r_cam
_PUSHOUT_ITERS        = 3      # tiny nearest-point pushout passes
_LOS_BINARY_STEPS     = 10     # bsearch steps for nearest clear LoS
_LOS_EPS              = 1.0e-4

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

def _multi_ray_min_hit(static_bvh, dynamic_bvh_map, origin, direction, max_dist, r_cam):
    """
    Returns nearest hit distance along 'direction' using a center ray plus a ring of offset rays.
    None => no hit within max_dist.
    """
    best = None
    # Center
    hl, hn, ho, hd = raycast_closest_any(static_bvh, dynamic_bvh_map, origin, direction, max_dist)
    if hl is not None:
        best = hd

    # Ring
    right, up = _build_basis(direction)
    ring_r = _FRUSTUM_RING_RADIUS * r_cam
    step = (2.0 * math.pi) / float(max(1, _FRUSTUM_RING_SAMPLES))
    for i in range(_FRUSTUM_RING_SAMPLES):
        ang = i * step
        offset = (math.cos(ang) * right + math.sin(ang) * up) * ring_r
        o = origin + offset
        hl, hn, ho, hd = raycast_closest_any(static_bvh, dynamic_bvh_map, o, direction, max_dist)
        if hl is not None:
            best = hd if (best is None or hd < best) else best
    return best  # None => clear

def _los_blocked(static_bvh, dynamic_bvh_map, a: Vector, b: Vector):
    d = b - a
    dist = d.length
    if dist <= 1e-9:
        return False
    hl, hn, ho, hd = raycast_closest_any(static_bvh, dynamic_bvh_map, a, d.normalized(), dist - _LOS_EPS)
    return (hl is not None)

def _binary_search_clear_los(static_bvh, dynamic_bvh_map, anchor, direction, low, high, steps):
    lo, hi = low, high
    for _ in range(max(1, int(steps))):
        mid = 0.5 * (lo + hi)
        cam = anchor + direction * mid
        if _los_blocked(static_bvh, dynamic_bvh_map, anchor, cam):
            hi = mid
        else:
            lo = mid
    return lo  # closest guaranteed-clear distance (>=low)

def _camera_sphere_pushout_any(static_bvh, dynamic_bvh_map, pos, radius, max_iters=_PUSHOUT_ITERS):
    """
    Tiny nearest-point pushout against static BVH and any LocalBVHs.
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
    for _ in range(max_iters):
        moved = False
        if static_bvh:
            p, m = push_once(static_bvh, p); moved = moved or m
        if dynamic_bvh_map:
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
      • Spherecasts boom (center + ring) to find first obstacle
      • Binary-searches to nearest clear LoS
      • Adds a small inward safety buffer on hits
      • Pushes out of thin geometry slightly
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

    # 2) Ensure clear LoS from anchor to the chosen point
    candidate_cam = anchor + direction * allowed
    if _los_blocked(bvh_tree, dynamic_bvh_map, anchor, candidate_cam):
        allowed = _binary_search_clear_los(
            bvh_tree, dynamic_bvh_map, anchor, direction,
            low=min_cam, high=allowed, steps=_LOS_BINARY_STEPS
        )
        candidate_cam = anchor + direction * allowed

    # 3) If STILL blocked at the hard minimum, micro-lean upward a hair to peek
    if _los_blocked(bvh_tree, dynamic_bvh_map, anchor, candidate_cam) and allowed <= (min_cam + 1e-6):
        up_nudge = min(MICRO_LEAN_UP_MAX, cap_r * MICRO_LEAN_UP_FRAC_OF_RADIUS)
        candidate_cam = candidate_cam + Vector((0, 0, up_nudge))

    # 4) Tiny nearest-point pushout to avoid embedding in thin geometry
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
