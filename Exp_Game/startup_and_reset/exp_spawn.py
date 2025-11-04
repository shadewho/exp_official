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
    """Spawn the user character and APPLY the current View/Camera UI settings."""
    scene = bpy.context.scene
    arm = scene.target_armature

    if not arm:
        print("Error: No target armature set in scene.target_armature.")
        return

    # Capsule-aware vertical clearance above the contacted surface
    char_cfg = getattr(scene, "char_physics", None)
    capsule_clear = max(getattr(char_cfg, "radius", 0.25), 0.25)

    # --------------------------------------------
    # Position character at spawn (unchanged logic)
    # --------------------------------------------
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
            arm.location.z += capsule_clear
            arm.rotation_euler = Euler((0.0, 0.0, spawn_obj.rotation_euler.z), 'XYZ')
    else:
        print("No spawn_object set!")

    # --------------------------------------------
    # APPLY VIEW / CAMERA UI SETTINGS AT SPAWN
    # --------------------------------------------

    # Anchor is the character "head" (capsule top) used by the camera solver.
    cap_h = float(getattr(scene.char_physics, "height", 2.0)) if getattr(scene, "char_physics", None) else 2.0
    anchor = arm.location.copy()
    anchor.z += cap_h

    # Get character yaw for THIRD/FIRST default orientation
    arm_yaw = arm.matrix_world.to_euler('XYZ').z

    # Helper: build direction from pitch/yaw (radians)
    def dir_from_pitch_yaw(pitch_rad: float, yaw_rad: float) -> Vector:
        cx = math.cos(pitch_rad); sx = math.sin(pitch_rad)
        sy = math.sin(yaw_rad);   cy = math.cos(yaw_rad)
        d = Vector((cx * sy, -cx * cy, sx))
        if d.length > 1.0e-9:
            d.normalize()
        return d

    # Resolve the desired camera direction + distance per View Mode
    vm = getattr(scene, "view_mode", 'THIRD')
    proj = getattr(scene, "view_projection", 'PERSP')

    if vm == 'LOCKED':
        pitch = float(getattr(scene, "view_locked_pitch", math.radians(60.0)))
        yaw   = float(getattr(scene, "view_locked_yaw", 0.0))
        dist  = float(getattr(scene, "view_locked_distance", 6.0))
        cam_dir = dir_from_pitch_yaw(pitch, yaw)

    elif vm == 'FIRST':
        # FIRST: head-height, tiny distance, oriented by current character yaw + UI pitch
        pitch = math.radians(float(getattr(scene, "pitch_angle", 15.0)))
        yaw   = arm_yaw
        # Very small non-zero distance to keep Blender happy
        dist  = 0.0006
        cam_dir = dir_from_pitch_yaw(pitch, yaw)

    else:  # 'THIRD'
        pitch = math.radians(float(getattr(scene, "pitch_angle", 15.0)))
        yaw   = arm_yaw
        # Match your UI distance logic
        dist  = float(getattr(scene, "orbit_distance", 2.0)) + float(getattr(scene, "zoom_factor", 4.0))
        cam_dir = dir_from_pitch_yaw(pitch, yaw)

    cam_quat = cam_dir.to_track_quat('Z', 'Y')

    # Stamp to ALL VIEW_3D regions in all open windows
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                space = area.spaces.active
                r3d = space.region_3d

                # Projection (Perspective / Orthographic)
                try:
                    r3d.view_perspective = proj  # 'PERSP' or 'ORTHO'
                except Exception:
                    pass
                
                # Lens (mm) from scene
                try:
                    space.lens = float(getattr(scene, "viewport_lens_mm", 55.0))
                except Exception:
                    pass

                # Location, rotation, distance
                try:
                    r3d.view_location = anchor
                except Exception:
                    pass
                try:
                    r3d.view_rotation = cam_quat
                except Exception:
                    pass
                try:
                    r3d.view_distance = max(0.0006, float(dist))
                except Exception:
                    pass
    except Exception as e:
        print(f"[WARN] stamping view settings failed: {e}")

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