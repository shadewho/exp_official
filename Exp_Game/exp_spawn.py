# File: exp_spawn.py
########################################
import bpy
import math
from mathutils import Vector, Euler
from .exp_raycastutils import create_bvh_tree

def spawn_user():
    """Spawn the user character in the scene based on the current settings."""
    scene = bpy.context.scene
    arm = scene.target_armature
    margin = 0.25  # safety margin above the surface

    if not arm:
        print("Error: No target armature set in scene.target_armature.")
        return

    spawn_obj = scene.spawn_object
    if spawn_obj:
        if scene.spawn_use_nearest_z_surface and spawn_obj.type == 'MESH':
            bvh = create_bvh_tree([spawn_obj])
            if bvh:
                # Compute world‐space bounding box Z extents
                world_bb = [spawn_obj.matrix_world @ Vector(corner) for corner in spawn_obj.bound_box]
                z_min = min(v.z for v in world_bb)
                z_max = max(v.z for v in world_bb)
                origin = spawn_obj.location.copy()

                # Raycast down from just above top
                origin_down = Vector((origin.x, origin.y, z_max + 0.001))
                hit_down = bvh.ray_cast(origin_down, Vector((0, 0, -1)))
                loc_down = hit_down[0] if hit_down and hit_down[0] else None

                # Raycast up from just below bottom
                origin_up = Vector((origin.x, origin.y, z_min - 0.001))
                hit_up = bvh.ray_cast(origin_up, Vector((0, 0, 1)))
                loc_up = hit_up[0] if hit_up and hit_up[0] else None

                # Choose the hit closest in Z to the origin
                candidates = [loc for loc in (loc_down, loc_up) if loc]
                if candidates:
                    chosen = min(candidates, key=lambda loc: abs(loc.z - origin.z))
                    arm.location = Vector((origin.x, origin.y, chosen.z + margin))
                    # Always upright: zero X/Y rotation, keep only Z (yaw)
                    arm.rotation_euler = Euler((0.0, 0.0, spawn_obj.rotation_euler.z), 'XYZ')
                    print(f"[spawn_user] surface Z={chosen.z:.3f}, placed at Z={arm.location.z:.3f}")
                else:
                    arm.location = origin
                    arm.rotation_euler = Euler((0.0, 0.0, spawn_obj.rotation_euler.z), 'XYZ')
                    print("[spawn_user] no surface hit; using origin")
            else:
                arm.location = spawn_obj.location.copy()
                arm.rotation_euler = Euler((0.0, 0.0, spawn_obj.rotation_euler.z), 'XYZ')
                print("[spawn_user] BVH failed; using origin")
        else:
            arm.location = spawn_obj.location.copy()
            arm.rotation_euler = Euler((0.0, 0.0, spawn_obj.rotation_euler.z), 'XYZ')
            print("[spawn_user] flag off or non-mesh; using origin")
    else:
        print("No spawn_object set; character remains in place.")

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

        # only remove the target armature + its children
        to_remove = [arm] + list(arm.children_recursive)
        for obj in to_remove:
            # unlink from every scene
            for sc in bpy.data.scenes:
                if obj.name in sc.collection.objects:
                    sc.collection.objects.unlink(obj)
            # remove the object
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass

        # clear the scene pointer
        scene.target_armature = None

        # purge any unused mesh datablocks
        for mesh in list(bpy.data.meshes):
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)

        # purge any unused armature datablocks
        for arm_data in list(bpy.data.armatures):
            if arm_data.users == 0:
                bpy.data.armatures.remove(arm_data)

        return {'FINISHED'}
