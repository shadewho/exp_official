import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def create_bvh_tree(ground_objs):
    """
    Creates a BVHTree. 
    - If 'ground_objs' is a single mesh object, return a BVH for that object.
    - If 'ground_objs' is a list (or iterable) of mesh objects,
      merge them in world space and return a single BVH for all.
    """

    # If user gives just a single object, wrap it in a list
    if isinstance(ground_objs, bpy.types.Object):
        ground_objs = [ground_objs]

    # We'll build one big BMesh that includes all objects
    combined_bm = bmesh.new()
    
    for obj in ground_objs:
        if not obj or obj.type != 'MESH':
            continue

        # 1) Make a temporary BMesh from the object's data
        temp_bm = bmesh.new()
        temp_bm.from_mesh(obj.data)
        
        # 2) Transform into world space
        temp_bm.transform(obj.matrix_world)

        # 3) Convert that temp_bm to a temporary Mesh
        temp_mesh = bpy.data.meshes.new("temp_merge")
        temp_bm.to_mesh(temp_mesh)
        temp_bm.free()

        # 4) Merge the resulting mesh into the combined_bm
        combined_bm.from_mesh(temp_mesh)

        # Clean up the temporary Mesh datablock
        bpy.data.meshes.remove(temp_mesh)

    # If everything was empty or invalid, skip
    if len(combined_bm.faces) == 0:
        combined_bm.free()
        return None

    # Finally, build a BVH from the combined BMesh
    bvh_tree = BVHTree.FromBMesh(combined_bm)
    combined_bm.free()
    return bvh_tree


def raycast_to_ground(bvh_tree, origin, direction=Vector((0, 0, -2.0))):
    """Performs a raycast to detect ground from an origin point."""
    if not bvh_tree:
        return None

    hit_location, normal, face_index, distance = bvh_tree.ray_cast(origin, direction)
    if hit_location:
        return hit_location
    return None

