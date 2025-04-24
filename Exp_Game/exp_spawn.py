# File: exp_spawn.py
########################################
import bpy
import math
from mathutils import Vector, Matrix, Quaternion

def spawn_user():
    """
    1) If there's an object named 'exp_spawn', move our target object (e.g. 'exp_character')
       to that location (and optionally rotation).
    2) Then set up the camera to orbit around 'target armature' using distance/pitch from the scene.
    """
    scene = bpy.context.scene
    arm = scene.target_armature
    distance = scene.orbit_distance
    view_distance = scene.zoom_factor
    pitch = scene.pitch_angle

    if not arm:
        print("Error: No target armature set in scene.target_armature.")
        return

    # Instead of: spawn_location_obj = bpy.data.objects.get("exp_spawn")
    # We read the pointer property:
    spawn_location_obj = scene.spawn_object

    if spawn_location_obj:
        print("Moving character to spawn_object location:", spawn_location_obj.name)
        arm.location = spawn_location_obj.location
        arm.rotation_euler = spawn_location_obj.rotation_euler
    else:
        print("No spawn_object set; using the character's current location.")

    # 3) camera logic:

    # Compute world‑space “behind + pitched” direction
    obj_location = arm.location
    obj_rotation = arm.matrix_world.to_quaternion()
    yaw = obj_rotation.to_euler('XYZ').z
    pitch_rad = math.radians(pitch)
    cam_dir = Vector((
        math.cos(pitch_rad) * math.sin(yaw),
        -math.cos(pitch_rad) * math.cos(yaw),
        math.sin(pitch_rad)
    ))
    view_position = obj_location + cam_dir * distance

    # Apply to every VIEW_3D in the current screen:
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            r3d = area.spaces.active.region_3d
            r3d.view_location = view_position
            r3d.view_rotation = cam_dir.to_track_quat('Z', 'Y')
            # combine orbit_distance + zoom_factor
            r3d.view_distance = distance + view_distance

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
