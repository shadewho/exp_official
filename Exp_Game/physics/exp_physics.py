# exp_physics.py
import bpy
import math
import mathutils
from mathutils import Vector
from .exp_raycastutils import BVHTree, create_bvh_tree
from ..interactions.exp_interactions import approximate_bounding_sphere_radius

def apply_gravity(z_velocity, gravity, dt):
    """
    Applies gravity to z_velocity over time dt.
    """
    z_velocity += gravity * dt
    return z_velocity


def is_colliding_overhead(bvh_tree, obj, overhead_offset=2.0, cast_height=0.2, max_distance=0.5):
    """
    Checks if there's geometry above the character's head within max_distance.
    If so => True.
    """
    if not bvh_tree or not obj:
        return False

    origin = obj.location.copy()
    origin.z += (overhead_offset + cast_height)
    direction = Vector((0,0,1))

    hit = bvh_tree.ray_cast(origin, direction, max_distance)
    return (hit[0] is not None)

def capsule_collision_resolve(
    bvh_like,
    obj,
    radius=0.2,
    heights=(0.3, 1.0, 1.9),
    max_iterations=6,
    push_strength=1.0
):
    """
    Robust capsule pushout:
      • Uses BVH.find_nearest at several vertical sample heights. This
        works even if we START INSIDE thin geometry (ray-casts may fail).
      • Pushes away along the face normal oriented AWAY from the center.
      • Optionally does a shallow ring ray test as a final polish.

    Accepts either a BVHTree or a LocalBVH (has .ray_cast and .find_nearest).
    """
    if not bvh_like or not obj:
        return

    # For plain BVHTree, expose a uniform API like LocalBVH
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

    if hasattr(bvh_like, "find_nearest"):
        bvh = bvh_like  # LocalBVH already provides the API
    else:
        bvh = _BVHWrapper(bvh_like)

    eps = 1.0e-4
    up  = Vector((0, 0, 1))

    for _ in range(max_iterations):
        any_fix = False
        base = obj.location.copy()

        # --- Nearest-point pushout at multiple heights ---
        for h in heights:
            center = base.copy(); center.z += h
            co, n, _, dist = bvh.find_nearest(center)
            if co is None or n is None:
                continue

            # Distance from center to surface
            if dist < (radius - eps):
                # Ensure we push *away* from surface, regardless of triangle winding
                dir_to_center = (center - co)
                if dir_to_center.dot(n) < 0.0:
                    n = -n
                penetration = (radius - dist) + eps
                obj.location += n * (penetration * push_strength)
                any_fix = True

        if not any_fix:
            break


def raycast_down(bvh_tree, obj, dist=3.0):
    """
    Cast a ray from about 1 meter above the feet, downward by dist => returns (loc, norm).
    If no hit => (None, None).
    """
    if not bvh_tree or not obj:
        return None, None

    origin = obj.location + Vector((0,0,1.0))
    direction = Vector((0,0,-1))

    result = bvh_tree.ray_cast(origin, direction, dist)
    if result[0] is not None:
        return result[0], result[1]
    return None, None


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


def is_tall_obstacle(bvh_tree, point, height_check=1.0):
    """
    If there's geometry above 'point' up to 'height_check' => it's a tall obstacle/wall.
    """
    if not bvh_tree:
        return False

    start = point.copy()
    end = point.copy()
    end.z += height_check

    direction = (end - start).normalized()
    length = (end - start).length

    hit = bvh_tree.ray_cast(start, direction, length)
    return (hit[0] is not None)


def sweep_down_segment(bvh_tree, obj, old_z, new_z):
    """
    We do a segment from old_z down to new_z. If there's ground in that segment => return the z of that ground.
    Otherwise => None => keep falling.
    """
    if new_z >= old_z:
        return None  # not going down

    origin = obj.location.copy()
    origin.z = old_z + 0.01
    segment_dist = (old_z - new_z) + 0.05

    direction = Vector((0,0,-1))
    hit = bvh_tree.ray_cast(origin, direction, segment_dist)
    if hit[0] is not None:
        hit_loc, hit_normal, face_idx, dist = hit
        if hit_loc.z >= new_z:
            return hit_loc.z
    return None