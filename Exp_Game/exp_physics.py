# exp_physics.py
import bpy
import math
import mathutils
from mathutils import Vector
from .exp_raycastutils import BVHTree, create_bvh_tree
from .exp_interactions import approximate_bounding_sphere_radius


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


def update_dynamic_meshes(modal_op):
    """
    Updates dynamic mesh state for the modal operator by:
      - Updating or rebuilding the BVH for each dynamic (moving) mesh.
      - Updating the dictionary that stores the lateral motion of each dynamic mesh.
      - Calculating the delta matrices to account for vertical platform motion.
      - Applying the dynamic platform delta if the character is grounded.
    
    This function is designed to move all the dynamic mesh updating code from the modal operator into exp_physics.
    
    It assumes that the modal operator (modal_op) has the following attributes already set:
      - moving_meshes: the list of dynamic mesh objects.
      - cached_dynamic_bvhs: a dictionary for caching BVH trees (if not present, it is created).
      - platform_prev_positions: a dictionary storing the previous translation for each dynamic mesh.
      - platform_prev_matrices: a dictionary storing the previous full matrix for each dynamic mesh.
      - platform_motion_map: a dictionary to store lateral motion vectors.
      - platform_delta_map: a dictionary to store the computed delta matrices.
      - apply_platform_delta_if_grounded(): a method on modal_op to apply the delta to the target object.
    
    Adjust MATRIX_THRESHOLD (and any other thresholds) as needed.
    """
    if not hasattr(modal_op, "cached_dynamic_bvhs"):
        modal_op.cached_dynamic_bvhs = {}

    modal_op.dynamic_bvh_map = {}
    MATRIX_THRESHOLD = 1e-3

    # Where is the player?
    player_loc = None
    if modal_op.target_object:
        player_loc = modal_op.target_object.matrix_world.translation

    scene = bpy.context.scene

    for dyn_obj in modal_op.moving_meshes:
        if not dyn_obj or dyn_obj.type != 'MESH':
            continue

        # Find the corresponding ProxyMeshEntry (to read register_distance)
        proxy_entry = None
        for pm in scene.proxy_meshes:
            if pm.mesh_object == dyn_obj:
                proxy_entry = pm
                break
        if not proxy_entry:
            # If no proxy entry, skip
            continue

        #-----------------------------------------
        # 1) Check if we "actively" update it
        #-----------------------------------------
        register_dist = proxy_entry.register_distance
        # By default, always process if dist=0 or we have no player
        actively_update = True
        if register_dist > 0 and player_loc is not None:
            distance = (dyn_obj.matrix_world.translation - player_loc).length
            if distance > register_dist:
                actively_update = False

        #-----------------------------------------
        # 2) Build or reuse BVH so it's always solid
        #-----------------------------------------
        current_matrix = dyn_obj.matrix_world.copy()
        cached = modal_op.cached_dynamic_bvhs.get(dyn_obj)

        if cached:
            last_matrix, cached_bvh, dyn_radius, was_active = cached

            # If we are not actively updating now but we were active before,
            # or vice versa, we might want to do a one-time transform update:
            if actively_update and not was_active:
                # The object just came back into range => do a fresh BVH build
                dyn_bvh = create_bvh_tree([dyn_obj])
                if dyn_bvh:
                    dyn_radius = approximate_bounding_sphere_radius(dyn_obj)
                    modal_op.cached_dynamic_bvhs[dyn_obj] = (current_matrix, dyn_bvh, dyn_radius, actively_update)
                    modal_op.dynamic_bvh_map[dyn_obj] = (dyn_bvh, dyn_radius)
                continue
            elif (current_matrix.to_translation() - last_matrix.to_translation()).length < MATRIX_THRESHOLD:
                # The object hasn't moved much => reuse the BVH
                modal_op.dynamic_bvh_map[dyn_obj] = (cached_bvh, dyn_radius)
                # Update the 'was_active' state
                modal_op.cached_dynamic_bvhs[dyn_obj] = (last_matrix, cached_bvh, dyn_radius, actively_update)
                continue

        # If no cache or we changed significantly => rebuild if actively updating
        if actively_update:
            dyn_bvh = create_bvh_tree([dyn_obj])
            if dyn_bvh:
                dyn_radius = approximate_bounding_sphere_radius(dyn_obj)
                modal_op.dynamic_bvh_map[dyn_obj] = (dyn_bvh, dyn_radius)
                modal_op.cached_dynamic_bvhs[dyn_obj] = (current_matrix, dyn_bvh, dyn_radius, actively_update)
        else:
            # If we are out of range & no cache at all, build it once
            if not cached:
                dyn_bvh = create_bvh_tree([dyn_obj])
                if dyn_bvh:
                    dyn_radius = approximate_bounding_sphere_radius(dyn_obj)
                    # CHANGED "dynamic_bvhs" → "cached_dynamic_bvhs"
                    modal_op.cached_dynamic_bvhs[dyn_obj] = (current_matrix, dyn_bvh, dyn_radius, actively_update)
                    modal_op.dynamic_bvh_map[dyn_obj] = (dyn_bvh, dyn_radius)
            else:
                # Reuse previous
                last_matrix, cached_bvh, dyn_radius, was_active = cached
                modal_op.dynamic_bvh_map[dyn_obj] = (cached_bvh, dyn_radius)
                modal_op.cached_dynamic_bvhs[dyn_obj] = (last_matrix, cached_bvh, dyn_radius, actively_update)

    #-------------------------------------------------
    # 3) Update platform motion only if "actively_update"
    #-------------------------------------------------
    for dyn_obj in modal_op.moving_meshes:
        proxy_entry = next(
            (pm for pm in scene.proxy_meshes if pm.mesh_object == dyn_obj),
            None
        )
        if not proxy_entry:
            continue
        register_dist = proxy_entry.register_distance

        actively_update = True
        if register_dist > 0 and player_loc is not None:
            dist = (dyn_obj.matrix_world.translation - player_loc).length
            if dist > register_dist:
                actively_update = False

        old_pos = modal_op.platform_prev_positions.get(dyn_obj, None)
        new_pos = dyn_obj.matrix_world.translation.copy()

        if old_pos is not None and actively_update:
            # If it's out of range => skip the motion computations
            motion_vec = new_pos - old_pos
            modal_op.platform_motion_map[dyn_obj] = motion_vec

        modal_op.platform_prev_positions[dyn_obj] = new_pos

    #-------------------------------------------------
    # 4) Compute "delta" transforms
    #-------------------------------------------------
    modal_op.platform_delta_map = {}
    for dyn_obj in modal_op.moving_meshes:
        old_mat = modal_op.platform_prev_matrices[dyn_obj]
        new_mat = dyn_obj.matrix_world.copy()

        # Check if we actively update:
        proxy_entry = next(
            (pm for pm in scene.proxy_meshes if pm.mesh_object == dyn_obj),
            None
        )
        if not proxy_entry:
            continue

        register_dist = proxy_entry.register_distance
        actively_update = True
        if register_dist > 0 and player_loc is not None:
            dist = (new_mat.translation - player_loc).length
            if dist > register_dist:
                actively_update = False

        if actively_update:
            delta_mat = new_mat @ old_mat.inverted()
        else:
            # freeze the delta to identity if out of range
            delta_mat = mathutils.Matrix.Identity(4)

        modal_op.platform_delta_map[dyn_obj] = delta_mat
        modal_op.platform_prev_matrices[dyn_obj] = new_mat

    #-------------------------------------------------
    # 5) Apply Delta If Grounded
    #-------------------------------------------------
    modal_op.apply_platform_delta_if_grounded()