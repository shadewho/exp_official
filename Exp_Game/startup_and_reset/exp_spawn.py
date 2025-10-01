# File: exp_spawn.py
########################################
import bpy
import math
from mathutils import Vector, Euler
from ..physics.exp_raycastutils import create_bvh_tree

def _raycast_obj_along_world_z(obj: bpy.types.Object, origin_xy: Vector, world_dir: Vector):
    """
    Ray-cast against `obj.data` along a WORLD +Z or -Z direction, passing through (origin_xy.x, origin_xy.y).
    Returns world-space hit location or None.
    """
    # Build world-space ray origin far enough outside the object’s AABB to avoid grazing
    world_bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    z_min = min(v.z for v in world_bb)
    z_max = max(v.z for v in world_bb)
    height = max(0.01, z_max - z_min)

    if world_dir.z < 0.0:
        start_world = Vector((origin_xy.x, origin_xy.y, z_max + height * 0.5))
    else:
        start_world = Vector((origin_xy.x, origin_xy.y, z_min - height * 0.5))

    # Transform the ray into the object’s local space for Mesh.ray_cast
    inv = obj.matrix_world.inverted()
    start_local = inv @ start_world
    # Transform direction via the inverse transpose for correctness, then normalize
    dir_local = (inv.to_3x3() @ world_dir).normalized()

    mesh = obj.data
    if not hasattr(mesh, "ray_cast"):
        return None

    hit, loc_local, normal_local, face_idx = mesh.ray_cast(start_local, dir_local)
    if not hit:
        return None

    # Return world-space hit location
    return obj.matrix_world @ loc_local


def spawn_user():
    """Spawn the user character in the scene based on the current settings."""
    scene = bpy.context.scene
    arm = scene.target_armature

    if not arm:
        print("Error: No target armature set in scene.target_armature.")
        return

    # Capsule-aware vertical clearance above the contacted surface
    char_cfg = getattr(scene, "char_physics", None)
    capsule_clear = max(getattr(char_cfg, "radius", 0.25), 0.25)

    spawn_obj = scene.spawn_object
    if spawn_obj:
        if scene.spawn_use_nearest_z_surface and spawn_obj.type == 'MESH':
            # Cast along global ±Z through the spawn object’s XY origin
            origin_xy = Vector((spawn_obj.location.x, spawn_obj.location.y, 0.0))

            hit_down = _raycast_obj_along_world_z(spawn_obj, origin_xy, Vector((0, 0, -1)))
            hit_up   = _raycast_obj_along_world_z(spawn_obj, origin_xy, Vector((0, 0,  1)))

            candidates = [h for h in (hit_down, hit_up) if h is not None]
            if candidates:
                # choose the Z closest to the object’s world origin (stable for large meshes)
                z0 = spawn_obj.location.z
                chosen = min(candidates, key=lambda P: abs(P.z - z0))
                arm.location = Vector((origin_xy.x, origin_xy.y, chosen.z + capsule_clear))
            else:
                # Fallbacks:
                # 1) if no triangles under the XY column, place the character just above the mesh AABB.
                world_bb = [spawn_obj.matrix_world @ Vector(c) for c in spawn_obj.bound_box]
                z_max = max(v.z for v in world_bb)
                arm.location = Vector((origin_xy.x, origin_xy.y, z_max + capsule_clear))

            # Keep only yaw from the spawn object
            arm.rotation_euler = Euler((0.0, 0.0, spawn_obj.rotation_euler.z), 'XYZ')

        else:
            # direct placement (non-surface mode)
            arm.location = spawn_obj.location.copy()
            arm.location.z += capsule_clear  # still keep the capsule out of minor overlaps
            arm.rotation_euler = Euler((0.0, 0.0, spawn_obj.rotation_euler.z), 'XYZ')
    else:
        print("No spawn_object set!")

    # --- camera setup (unchanged) ---
    obj_loc = arm.location
    obj_rot = arm.matrix_world.to_quaternion()
    yaw = obj_rot.to_euler('XYZ').z
    pitch_rad = math.radians(scene.pitch_angle)
    cam_dir = Vector((
        math.cos(pitch_rad) * math.sin(yaw),
        -math.cos(pitch_rad) * math.cos(yaw),
        math.sin(pitch_rad)
    ))
    view_pos = obj_loc + cam_dir * scene.orbit_distance

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            r3d = area.spaces.active.region_3d
            r3d.view_location = view_pos
            r3d.view_rotation = cam_dir.to_track_quat('Z', 'Y')
            r3d.view_distance = scene.orbit_distance + scene.zoom_factor

#-----------------------------------------------------------
#remove character to maintain a clean scene character state
#-----------------------------------------------------------
class EXPLORATORY_OT_RemoveCharacter(bpy.types.Operator):
    bl_idname = "exploratory.remove_character"
    bl_label  = "Remove Character"

    def execute(self, context):
        scene = context.scene

        # ← early-out if locked
        if scene.character_spawn_lock:
            self.report({'INFO'}, "Character spawn is locked; skipping removal.")
            return {'CANCELLED'}

        arm = scene.target_armature
        if not arm:
            return {'CANCELLED'}

        # ------------------------------------------------------------------
        # 1) Collect objects to remove and gather their materials & images
        # ------------------------------------------------------------------
        to_remove = [arm] + list(arm.children_recursive)

        # Materials directly assigned to meshes under the armature
        mats_to_consider = set()
        # Images referenced by those materials (walk node groups, too)
        images_to_consider = set()

        def _gather_images_from_node_tree(nt, seen_nt):
            if not nt or nt in seen_nt:
                return
            seen_nt.add(nt)
            for node in nt.nodes:
                if node.type == 'TEX_IMAGE' and getattr(node, "image", None):
                    images_to_consider.add(node.image)
                elif node.type == 'GROUP' and getattr(node, "node_tree", None):
                    _gather_images_from_node_tree(node.node_tree, seen_nt)

        for obj in to_remove:
            if obj.type == 'MESH' and obj.data:
                # materials in slots
                for mat in obj.data.materials:
                    if mat and mat.library is None:  # only local data
                        mats_to_consider.add(mat)
                        if getattr(mat, "node_tree", None):
                            _gather_images_from_node_tree(mat.node_tree, set())

        # ------------------------------------------------------------------
        # 2) Unlink & remove objects (all scenes), then clear the pointer
        # ------------------------------------------------------------------
        for obj in to_remove:
            # Unlink from every scene collection that contains it
            for sc in bpy.data.scenes:
                try:
                    if obj.name in sc.collection.objects:
                        sc.collection.objects.unlink(obj)
                except Exception:
                    pass
            # Remove the object datablock
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass

        scene.target_armature = None

        # ------------------------------------------------------------------
        # 3) Purge orphaned geometry/armatures created by the removal
        # ------------------------------------------------------------------
        for mesh in list(bpy.data.meshes):
            try:
                if mesh.library is None and mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
            except Exception:
                pass

        for arm_data in list(bpy.data.armatures):
            try:
                if arm_data.library is None and arm_data.users == 0:
                    bpy.data.armatures.remove(arm_data)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 4) Safely purge materials & images that belonged to the character
        #     • Only remove if:
        #         - local (not linked from a library)
        #         - has zero users after the object removals
        #         - not protected by a fake user
        # ------------------------------------------------------------------
        for mat in list(mats_to_consider):
            try:
                if mat and mat.library is None and mat.users == 0 and not mat.use_fake_user:
                    bpy.data.materials.remove(mat)
            except Exception:
                pass

        for img in list(images_to_consider):
            try:
                if img and img.library is None and img.users == 0 and not img.use_fake_user:
                    bpy.data.images.remove(img)
            except Exception:
                pass

        return {'FINISHED'}