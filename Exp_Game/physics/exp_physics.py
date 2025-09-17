# Exploratory/physics/exp_physics.py
from mathutils import Vector

def capsule_collision_resolve(
    bvh_like,
    obj,
    radius=0.2,
    heights=(0.3, 1.0, 1.9),
    max_iterations=6,
    push_strength=0.5,
    max_push_per_iter=None,
    average_contacts=True,
    # Walkable-floor classification and skipping
    floor_cos_limit=None,          # pass cos(radians(slope_limit_deg)) or None
    ignore_floor_contacts=False,   # if True, skip walkable-floor contacts
    # Capsule-based step band filtering (relative height from base)
    ignore_contacts_below_height=None,     # e.g., radius + step_height
    ignore_below_only_if_floor_like=True,  # NEW: only skip if walkable floor
):
    """
    Softer capsule pushout with capsule-based step-band filtering.
      • Samples nearest at several axial heights.
      • Skips walkable floors when requested to avoid downhill creep.
      • Skips contacts inside the step band ONLY if they are walkable floors.
        Steep/wall contacts are NEVER skipped, preventing bleed-through.
    Accepts either a BVHTree or a LocalBVH (must provide .ray_cast and .find_nearest).
    """
    if not bvh_like or not obj:
        return

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

    bvh = bvh_like if hasattr(bvh_like, "find_nearest") else _BVHWrapper(bvh_like)

    up  = Vector((0, 0, 1))
    eps = 1.0e-4

    if max_push_per_iter is None:
        max_push_per_iter = max(0.10, float(radius) * 0.35)

    for _ in range(max(1, int(max_iterations))):
        base = obj.location.copy()

        corr = Vector((0.0, 0.0, 0.0))
        any_penetration = False

        for h in heights:
            c = base.copy(); c.z += float(h)
            co, n, _, dist = bvh.find_nearest(c)
            if co is None or n is None:
                continue

            n = n.normalized()
            if (c - co).dot(n) < 0.0:
                n = -n

            # Floor-like classification once
            is_floor_like = False
            if floor_cos_limit is not None:
                try:
                    is_floor_like = (n.dot(up) >= float(floor_cos_limit))
                except Exception:
                    is_floor_like = False

            # Capsule-based step band filtering:
            if ignore_contacts_below_height is not None:
                if (co.z - base.z) <= (float(ignore_contacts_below_height) + eps):
                    # Skip only if it's a walkable floor and the caller requested floor-only.
                    if ignore_below_only_if_floor_like and is_floor_like:
                        continue
                    # If not floor-like (i.e., steep), DO NOT skip. Steep must always block.

            # Optional global walkable-floor skip (e.g., while grounded)
            if ignore_floor_contacts and is_floor_like:
                continue

            pen = float(radius) - float(dist)
            if pen <= eps:
                continue

            any_penetration = True
            if average_contacts:
                corr += n * pen
            else:
                step = min(pen * float(push_strength), float(max_push_per_iter))
                obj.location += n * (step + eps)

        if not any_penetration:
            break

        if average_contacts:
            L = corr.length
            if L > eps:
                step_len = min(L * float(push_strength), float(max_push_per_iter))
                obj.location += (corr / L) * step_len
            else:
                break

def remove_steep_slope_component(move_dir, slope_normal, max_slope_dot=0.7):
    """
    When the slope is too steep (dot(up, n) < max_slope_dot), remove ONLY the
    uphill component *along the plane* from move_dir. This blocks climbing but
    still allows cross-slope movement.

    move_dir: Vector (XYZ)
    slope_normal: contact normal (unit or not; normalized here)
    max_slope_dot: cos(max_walkable_angle)
    """
    up = Vector((0, 0, 1))
    n = slope_normal.normalized()

    # Steep if floor-likeness is below the limit (strict, no epsilons)
    if n.dot(up) >= max_slope_dot:
        return move_dir

    # "Uphill along the plane" = the projection of world up onto the plane
    uphill = up - n * up.dot(n)  # lies in the plane, points uphill
    if uphill.length <= 1.0e-9:
        return move_dir
    uphill.normalize()

    # Project intended motion into the plane, then zero its uphill component
    m_plane = move_dir - n * move_dir.dot(n)
    uphill_amt = m_plane.dot(uphill)
    if uphill_amt > 0.0:
        m_plane -= uphill * uphill_amt

        # Keep any component that was orthogonal to the plane (into air/ground)
        m_normal = n * move_dir.dot(n)
        return m_plane + m_normal

    return move_dir
