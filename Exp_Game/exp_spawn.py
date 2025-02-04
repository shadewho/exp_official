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

    # 3) We still do your camera logic:

    #   - Get the character's new location (after possibly moving to exp_spawn)
    obj_location = arm.location
    obj_rotation = arm.matrix_world.to_quaternion()

    #   - The local Y axis in object space
    direction = Vector((0, 1, 0))
    #   - Rotate direction by the object's world rotation => world direction
    direction = obj_rotation @ direction

    #   - Calculate how far behind the object we place the camera
    view_position = obj_location + direction * distance

    #   - Set the 3D Viewport location
    space_data = bpy.context.space_data
    space_data.region_3d.view_location = view_position

    #   - Now apply pitch + object rotation as yaw
    pitch_rad = math.radians(pitch)
    pitch_quaternion = Quaternion((1, 0, 0), pitch_rad)
    yaw_quaternion = obj_rotation
    final_quaternion = yaw_quaternion @ pitch_quaternion

    space_data.region_3d.view_rotation = final_quaternion

    #   - Finally, set the 'zoom' factor
    space_data.region_3d.view_distance = view_distance
