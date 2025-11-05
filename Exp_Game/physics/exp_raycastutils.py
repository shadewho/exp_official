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

def raycast_closest_any(static_bvh, dynamic_bvh_map, origin, direction, max_distance):
    """
    Unified ray that returns the closest hit across static BVH and dynamic LocalBVHs.
    Returns: (hit_loc, hit_normal, hit_obj, hit_dist) or (None,None,None,None)
    """
    if not isinstance(direction, Vector) or direction.length <= 1e-9 or max_distance <= 1e-9:
        return (None, None, None, None)

    dnorm = direction.normalized()
    best = (None, None, None, 1e9)

    if static_bvh:
        hit = static_bvh.ray_cast(origin, dnorm, max_distance)
        if hit and hit[0] is not None and hit[3] < best[3]:
            best = (hit[0], hit[1], None, hit[3])

    if dynamic_bvh_map:
        for obj, (bvh_like, _) in dynamic_bvh_map.items():
            h = bvh_like.ray_cast(origin, dnorm, max_distance)
            if h and h[0] is not None and h[3] < best[3]:
                best = (h[0], h[1], obj, h[3])

    return best if best[0] is not None else (None, None, None, None)

def collect_world_triangles(objs, max_tris: int | None = None) -> list[float]:
    """
    Return a flat [x,y,z,...] list of world-space triangle vertices for the given mesh objects.
    Triangulates faces via BMesh. Intended to run once at game start for XR static init.
    """
    tris = []
    if not objs:
        return tris

    deps = bpy.context.evaluated_depsgraph_get()

    for obj in objs:
        try:
            if not obj or obj.type != 'MESH':
                continue

            eval_obj = obj.evaluated_get(deps)
            mesh = eval_obj.to_mesh(preserve_all_data_layers=False, depsgraph=deps)
            if mesh is None:
                continue

            bm = bmesh.new()
            bm.from_mesh(mesh)
            # Triangulate in local space, then transform to world
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            bm.transform(obj.matrix_world)

            bm.verts.ensure_lookup_table()
            for f in bm.faces:
                if len(f.verts) != 3:
                    continue
                v0, v1, v2 = f.verts[0].co, f.verts[1].co, f.verts[2].co
                tris.extend((float(v0.x), float(v0.y), float(v0.z),
                             float(v1.x), float(v1.y), float(v1.z),
                             float(v2.x), float(v2.y), float(v2.z)))
                if max_tris is not None and (len(tris) // 9) >= int(max_tris):
                    bm.free()
                    eval_obj.to_mesh_clear()
                    return tris

            bm.free()
            eval_obj.to_mesh_clear()
        except Exception:
            # Best-effort; skip bad objects
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass
            continue

    return tris

def collect_local_triangles(obj, max_tris: int | None = None) -> list[float]:
    """
    Return a flat [x,y,z,...] list of LOCAL-SPACE triangle vertices for a single mesh object.
    Used to create XR LocalBVH inputs (we send xforms each frame; no rebuilds).
    """
    import bmesh
    if not obj or getattr(obj, "type", None) != 'MESH' or obj.data is None:
        return []
    tris = []
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)  # LOCAL space
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.verts.ensure_lookup_table()
        for f in bm.faces:
            if len(f.verts) != 3:
                continue
            v0, v1, v2 = f.verts[0].co, f.verts[1].co, f.verts[2].co
            tris.extend((float(v0.x), float(v0.y), float(v0.z),
                         float(v1.x), float(v1.y), float(v1.z),
                         float(v2.x), float(v2.y), float(v2.z)))
            if max_tris is not None and (len(tris) // 9) >= int(max_tris):
                break
    finally:
        bm.free()
    return tris

