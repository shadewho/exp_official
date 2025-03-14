# exp_physics.py

import math
import mathutils
from mathutils import Vector
from .exp_raycastutils import BVHTree


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
    bvh_tree,
    obj,
    radius=0.2,
    heights=(0.3, 1.0, 1.9),
    max_iterations=3,
    push_strength=0.2
):
    """
    A shape-based collision approach. We cast multiple radial rays
    around the character at different heights, pushing out if inside geometry.

    - radius: how wide the character is
    - heights: which "layers" (above feet) do we check
    - max_iterations: how many times we attempt to push out
    - push_strength: fraction of the push we apply each iteration (0.2 => gentle, 1.0 => strong)

    This ensures we never remain inside geometry if we overlap.
    """
    if not bvh_tree or not obj:
        return

    # Ensure bvh_tree is the actual BVH tree (if it's returned as a tuple, extract the first element)
    if isinstance(bvh_tree, tuple):
        bvh_tree = bvh_tree[0]

    for _ in range(max_iterations):
        any_fix = False
        loc = obj.location.copy()

        # We'll do 8 rays around the circle (like a capsule's cross-section)
        n_rays = 8
        step_angle = (2.0 * math.pi) / float(n_rays)

        for h in heights:
            center = loc.copy()
            center.z += h

            for i in range(n_rays):
                angle = i * step_angle
                direction = Vector((math.cos(angle), math.sin(angle), 0.0))
                start = center
                end = center + (direction * radius)

                hit = bvh_tree.ray_cast(start, (end - start))
                if hit[0] is not None:
                    # we are overlapping geometry => push out
                    hit_loc, hit_normal, face_idx, dist = hit
                    if dist < radius:
                        pen_depth = (radius - dist) + 0.0001
                        push_vec = hit_normal * pen_depth * push_strength
                        obj.location += push_vec
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