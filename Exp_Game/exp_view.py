import mathutils
import math
import bpy
from .exp_raycastutils import raycast_to_ground

def update_view(context, obj, pitch, yaw, bvh_tree, orbit_distance, zoom_factor):
    # Calculate the direction vector
    direction = mathutils.Vector((
        math.cos(pitch) * math.sin(yaw),  # Flipped the direction to look down positive Y axis
        -math.cos(pitch) * math.cos(yaw),
        math.sin(pitch)
    ))

    # Set the desired view location to orbit around the object at head height
    head_height = 2.0  # 2 meters above the object's origin
    scan_point = obj.location + mathutils.Vector((0, 0, head_height ))  # Set scan point to head height
    view_location = scan_point + direction * orbit_distance  # Start with the default orbit distance

    # Extend the scan distance to account for the view distance property
    extended_scan_distance = orbit_distance + zoom_factor  # Extend the raycast distance

    # Perform obstruction checking
    if bvh_tree:
        # Calculate the direction vector from the head to the extended view location
        direction_to_view = (view_location - scan_point).normalized()

        # Perform the raycast from the scan point towards the extended view location
        extended_view_location = scan_point + direction_to_view * extended_scan_distance
        hit_location = raycast_to_ground(bvh_tree, scan_point, direction_to_view)

        # If the ray hits something, adjust the view location
        if hit_location:
            distance_to_hit = (scan_point - hit_location).length
            # If the hit location is closer than the desired extended distance, adjust to avoid obstruction
            if distance_to_hit < extended_scan_distance:
                buffer_distance = -(bpy.context.scene.zoom_factor) -0.5  ############ .5 additional buffer distance
                adjusted_distance = distance_to_hit + buffer_distance
                view_location = scan_point + direction_to_view * adjusted_distance

    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'WINDOW':
                    region_3d = area.spaces.active.region_3d
                    region_3d.view_location = view_location
                    region_3d.view_rotation = direction.to_track_quat('Z', 'Y').to_matrix().to_3x3().to_quaternion()

