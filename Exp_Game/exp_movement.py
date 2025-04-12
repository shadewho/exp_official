# File: exp_movement.py

import math
import mathutils
from mathutils import Vector

# We import relevant helpers from exp_physics:
from .exp_physics import (
    apply_gravity,
    is_colliding_overhead,
    capsule_collision_resolve,
    raycast_down,
    remove_steep_slope_component,
    sweep_down_segment,
)

from .exp_view import update_view
from ..exp_preferences import ExploratoryAddonPreferences  # to read user prefs


def calculate_movement_direction(keys_pressed, obj):
    """
    DO NOT CHANGE ANYTHING INSIDE THIS FUNCTION.
    It's needed for forward direction reference.

    This function expects 'W','A','S','D' in keys_pressed to compute movement direction.
    """
    forward = obj.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    move_dir = Vector((0, 0, 0))

    if 'W' in keys_pressed:
        move_dir += forward
    if 'S' in keys_pressed:
        move_dir += forward
    if 'D' in keys_pressed:
        move_dir += forward
    if 'A' in keys_pressed:
        move_dir += forward

    if move_dir.length > 1e-5:
        move_dir.normalize()

    return move_dir


def find_best_floor_below(target_object, static_bvh, dynamic_bvh_map, max_distance=3.0):
    best_loc  = None
    best_norm = None
    best_obj  = None


    # 1) Static check
    if static_bvh:
        # If static_bvh is a tuple, extract the BVH tree.
        bvh = static_bvh[0] if isinstance(static_bvh, tuple) else static_bvh
        loc_s, norm_s = raycast_down(bvh, target_object, dist=max_distance)
        if loc_s:
            best_loc  = loc_s
            best_norm = norm_s
            best_obj  = None  # Indicates it's the static floor

    # 2) Dynamic checks
    if dynamic_bvh_map:
        for dyn_obj, dyn_bvh in dynamic_bvh_map.items():
            if dyn_bvh:
                bvh_dyn = dyn_bvh[0] if isinstance(dyn_bvh, tuple) else dyn_bvh
                loc_d, norm_d = raycast_down(bvh_dyn, target_object, dist=max_distance)
                if loc_d:
                    if best_loc is None:
                        best_loc  = loc_d
                        best_norm = norm_d
                        best_obj  = dyn_obj
                    else:
                        current_z = target_object.location.z
                        dist_best = abs(best_loc.z - current_z)
                        dist_dyn  = abs(loc_d.z - current_z)
                        if dist_dyn < dist_best:
                            best_loc  = loc_d
                            best_norm = norm_d
                            best_obj  = dyn_obj

    return (best_loc, best_norm, best_obj)


# ------------------------------------------------------------
# #1: New move_character(...) function 
# ------------------------------------------------------------
def move_character(
    op,                 # <-- New argument: reference to the modal operator
    target_object,
    keys_pressed,
    bvh_tree,
    delta_time,
    speed,
    gravity,
    z_velocity,
    jump_timer,
    is_jumping,
    is_grounded,
    jump_duration,
    sensitivity,
    pitch,
    yaw,
    context,
    dynamic_bvh_map=None,
    platform_motion_map=None
):
    """
    A robust approach ensuring the character does NOT pass through
    static or dynamic geometry, including vertical & horizontal collisions.
    Also identifies which dynamic object is 'best_obj' (the floor),
    so we can ride along with it later.

    We'll do multiple sub-steps for stable collision. Each sub-step:
        1) If not grounded, apply gravity.
        2) Move down with sweep.
        3) Move horizontally, remove steep slope component.
        4) Resolve collisions (capsule) vs static & dynamic
        5) Snap to floor if close enough
        6) Return (z_velocity, is_grounded)

    In the final lines, we store 'best_obj' in op.grounded_platform if grounded.

    NOTE: The 'op' parameter is your modal operator, so we can do:
        op.grounded_platform = best_obj
    at the end of this function.
    """

    if not target_object:
        return z_velocity, is_grounded

    # If no static BVH and no dynamic BVHs, skip collision
    if not bvh_tree and not dynamic_bvh_map:
        return z_velocity, is_grounded

    # Build a set of standard WASD keys from the user-chosen keys in preferences
    prefs = context.preferences.addons["Exploratory"].preferences
    forward_key  = prefs.key_forward
    backward_key = prefs.key_backward
    left_key     = prefs.key_left
    right_key    = prefs.key_right
    run_key      = prefs.key_run

    # Remap to 'W','A','S','D','LEFT_SHIFT' if pressed
    remapped_keys = set()
    if forward_key  in keys_pressed: remapped_keys.add('W')
    if backward_key in keys_pressed: remapped_keys.add('S')
    if left_key     in keys_pressed: remapped_keys.add('A')
    if right_key    in keys_pressed: remapped_keys.add('D')
    if run_key      in keys_pressed: remapped_keys.add('LEFT_SHIFT')

    # Calculate base direction
    from .exp_movement import calculate_movement_direction
    base_dir = calculate_movement_direction(remapped_keys, target_object)

    # 2x speed if SHIFT is held
    run_speed = speed * (2.0 if 'LEFT_SHIFT' in remapped_keys else 1.0)
    base_dir *= run_speed

    # We'll do multiple sub-steps
    num_substeps = 4
    dt_sub = delta_time / float(num_substeps)

    STEP_OFFSET = 0.3
    STEEP_DOT   = 0.3

    best_obj = None  # We'll store the dynamic floor reference if found

    # Sub-step loop
    for _ in range(num_substeps):
        # A) Gravity if not grounded
        if not is_grounded or is_jumping:
            z_velocity = apply_gravity(z_velocity, gravity, dt_sub)
            # Check overhead collisions if we are going up
            if z_velocity > 0.0 and bvh_tree:
                if is_colliding_overhead(bvh_tree, target_object):
                    z_velocity = 0

        # B) Move vertically (downwards sweep for static)
        old_z = target_object.location.z
        new_z = old_z + (z_velocity * dt_sub)

        if z_velocity < 0.0 and bvh_tree:
            floor_z = sweep_down_segment(bvh_tree, target_object, old_z, new_z)
            if floor_z is not None:
                # Hit static ground
                target_object.location.z = floor_z
                z_velocity = 0
                is_grounded = True
            else:
                # Keep falling
                target_object.location.z = new_z
                is_grounded = False
        else:
            # Either moving up or no static bvh => just set new z
            target_object.location.z = new_z

        # C) Horizontal move, remove steep slope if we have static
        move_dir = base_dir.copy()
        if bvh_tree:
            loc_s, norm_s = raycast_down(bvh_tree, target_object)
            if loc_s and norm_s:
                up_dot = norm_s.dot(Vector((0, 0, 1)))
                if up_dot > 0.5:
                    remove_steep_slope_component(move_dir, norm_s, STEEP_DOT)

        target_object.location += move_dir * dt_sub

        # D) capsule_collision_resolve => static
        if bvh_tree:
            capsule_collision_resolve(bvh_tree, target_object, radius=0.2, max_iterations=2)

        # E) capsule_collision_resolve => dynamic (except the floor object)
        loc_below, norm_below, obj_below = find_best_floor_below(
            target_object, bvh_tree, dynamic_bvh_map, max_distance=3.0
        )

        # Collide with all dynamic objects except the one that might be under feet
        if dynamic_bvh_map:
            for dyn_obj, dyn_bvh in dynamic_bvh_map.items():
                capsule_collision_resolve(dyn_bvh, target_object, radius=0.2, max_iterations=2)

        # Now snap + carry if we do have a floor under feet
        if loc_below:
            foot_dist = abs(target_object.location.z - loc_below.z)
            if foot_dist < STEP_OFFSET and z_velocity <= 0:
                target_object.location.z = loc_below.z
                z_velocity = 0
                is_grounded = True
                is_jumping = False

                if obj_below is not None:
                    best_obj = obj_below  # We'll assign later
            else:
                is_grounded = False
        else:
            is_grounded = False

    update_view(
        context,
        target_object,
        pitch,
        yaw,
        bvh_tree,
        context.scene.orbit_distance,
        context.scene.zoom_factor
    )

    # ---------------------------
    # FINAL: Store the best_obj
    # ---------------------------
    if is_grounded and best_obj is not None:
        op.grounded_platform = best_obj
    else:
        op.grounded_platform = None

    return z_velocity, is_grounded