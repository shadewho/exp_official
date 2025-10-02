# Exploratory/physics/exp_physics.py
from mathutils import Vector

def capsule_collision_resolve(
    bvh_like,
    pos: Vector,                     # ← NEW: current capsule base position (world)
    radius=0.2,
    heights=(0.3, 1.0, 1.9),
    max_iterations=2,
    push_strength=0.5,
    max_push_per_iter=None,
    average_contacts=True,
    # Walkable-floor classification and skipping (unchanged semantics)
    floor_cos_limit=None,            # pass cos(radians(slope_limit_deg)) or None
    ignore_floor_contacts=False,     # if True, skip walkable-floor contacts
    # Capsule-based step band filtering (relative height from base)
    ignore_contacts_below_height=None,     # e.g., radius + step_height
    ignore_below_only_if_floor_like=True,
):
    """
    Softer capsule pushout with capsule-based step-band filtering.
    • Pure-function style: does NOT touch bpy or object; returns corrected pos.
    • Treat faces above the slope limit as walls: horizontal-only correction.
    """
    if not bvh_like or pos is None:
        return pos

    class _BVHWrapper:
        def __init__(self, bvh):
            self._bvh = bvh
        def ray_cast(self, o, d, dist):
            return self._bvh.ray_cast(o, d, dist)
        def find_nearest(self, p, distance=float("inf")):
            res = self._bvh.find_nearest(p, distance)
            if res is None or res[0] is None:
                return (None, None, -1, 0.0)
            co, n, idx, _ = res
            return (co, n.normalized(), idx, (p - co).length)

    # Normalize interface so both BVHTree and LocalBVH work the same
    bvh = bvh_like if hasattr(bvh_like, "find_nearest") else _BVHWrapper(bvh_like)

    up  = Vector((0, 0, 1))
    # Only used to avoid zero-length math; not a slope threshold tweak.
    tiny = 1.0e-9

    if max_push_per_iter is None:
        max_push_per_iter = max(0.10, float(radius) * 0.35)

    out_pos = pos.copy()

    for _ in range(max(1, int(max_iterations))):
        base = out_pos.copy()
        corr = Vector((0.0, 0.0, 0.0))
        any_penetration = False

        for h in heights:
            c = base.copy(); c.z += float(h)
            co, n, _, dist = bvh.find_nearest(c)
            if co is None or n is None:
                continue

            n = n.normalized()
            # Ensure outward-facing normal relative to the capsule sample
            if (c - co).dot(n) < 0.0:
                n = -n

            # Walkable classification (exact)
            is_floor_like = False
            if floor_cos_limit is not None:
                is_floor_like = (n.dot(up) >= float(floor_cos_limit))

            # Step-band filtering (unchanged semantics)
            if ignore_contacts_below_height is not None:
                if (co.z - base.z) <= float(ignore_contacts_below_height):
                    if ignore_below_only_if_floor_like and is_floor_like:
                        continue

            # Global walkable-floor skip (unchanged semantics)
            if ignore_floor_contacts and is_floor_like:
                continue

            pen = float(radius) - float(dist)
            if pen <= tiny:
                continue

            # Treat too-steep faces like walls: horizontal-only push
            n_for_push = n
            if (floor_cos_limit is not None) and (not is_floor_like):
                n_for_push = Vector((n.x, n.y, 0.0))
                if n_for_push.length <= tiny:
                    continue
                n_for_push.normalize()

            any_penetration = True
            if average_contacts:
                corr += n_for_push * pen
            else:
                step = min(pen * float(push_strength), float(max_push_per_iter))
                out_pos += n_for_push * step

        if not any_penetration:
            break

        if average_contacts:
            L = corr.length
            if L > tiny:
                step_len = min(L * float(push_strength), float(max_push_per_iter))
                out_pos += (corr / L) * step_len
            else:
                break

    return out_pos


def remove_steep_slope_component(move_dir, slope_normal, max_slope_dot=0.7):
    """
    XY-only uphill clamp for steep slopes.
    - If the slope is above the limit, remove only the uphill component from the XY motion.
    - Returns a vector with z = 0.0.
    """
    up = Vector((0, 0, 1))
    n = slope_normal
    if n.length <= 1.0e-12:
        return Vector((move_dir.x, move_dir.y, 0.0))
    n = n.normalized()

    # If walkable (angle <= limit), leave motion unchanged (exact comparison)
    if n.dot(up) >= float(max_slope_dot):
        return Vector((move_dir.x, move_dir.y, 0.0))

    # In-plane uphill direction and its XY
    uphill = up - n * up.dot(n)     # lies in the plane, points uphill
    g_xy = Vector((uphill.x, uphill.y))
    if g_xy.length <= 1.0e-12:
        return Vector((move_dir.x, move_dir.y, 0.0))
    g_xy.normalize()

    m_xy = Vector((move_dir.x, move_dir.y))
    comp = m_xy.dot(g_xy)
    if comp > 0.0:
        m_xy -= g_xy * comp  # remove only the uphill component

    return Vector((m_xy.x, m_xy.y, 0.0))


