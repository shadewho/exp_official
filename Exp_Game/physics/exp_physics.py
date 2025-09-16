# exp_physics.py
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
    If slope is too steep => dot( slope_normal, (0,0,1) ) < max_slope_dot => remove uphill portion.

    e.g., max_slope_dot=0.6 => ~53°, 0.7 => ~45°, 0.8 => ~36°, etc.
    """
    up = Vector((0,0,1))
    slope_dot = slope_normal.dot(up)
    if slope_dot < max_slope_dot:
        # remove uphill portion
        uphill_dot = move_dir.normalized().dot(slope_normal)
        if uphill_dot > 0:
            remove_amt = uphill_dot * move_dir.length
            remove_vec = slope_normal * remove_amt
            move_dir -= remove_vec
    return move_dir
