# Exp_Game/physics/exp_bvh_local.py
import bmesh
import mathutils
from mathutils import Vector
from mathutils.bvhtree import BVHTree

class LocalBVH:
    """
    BVH built in LOCAL space of an object. Rays are transformed
    into local space; hits are transformed back to world space.
    This means rigid motion needs NO BVH rebuilds.
    """
    __slots__ = ("obj", "bvh")

    def __init__(self, obj):
        self.obj = obj
        bm = bmesh.new()
        bm.from_mesh(obj.data)           # local coordinates
        self.bvh = BVHTree.FromBMesh(bm)
        bm.free()

    def ray_cast(self, origin_world, direction_world, distance=None):
        """
        Matches BVHTree.ray_cast signature:
        Returns: (hit_loc_world, hit_norm_world, face_index, dist_world)
                 or (None,None,-1,0.0)
        """
        M = self.obj.matrix_world
        Minv = M.inverted()
        Minv3 = Minv.to_3x3()
        MinvT3 = Minv3.transposed()

        # Normalize direction and compute effective distance in local space
        if distance is None:
            dlen = direction_world.length
            if dlen <= 1e-12:
                return (None, None, -1, 0.0)
            direction_world = direction_world / dlen
            distance = dlen

        origin_local = Minv @ origin_world
        dir_local = Minv3 @ direction_world

        # Non-uniform scale: the "distance" mapping is not 1:1.
        step_local = Minv3 @ (direction_world * distance)
        dist_local = step_local.length
        if dist_local <= 1e-12:
            return (None, None, -1, 0.0)

        hit = self.bvh.ray_cast(origin_local, dir_local.normalized(), dist_local)
        if hit[0] is None:
            return (None, None, -1, 0.0)

        hit_loc_local, hit_norm_local, face_index, dist_local_res = hit
        hit_loc_world = M @ hit_loc_local
        hit_norm_world = (MinvT3 @ hit_norm_local).normalized()

        # Convert resolved local travel back to world distance
        vec_local = dir_local.normalized() * dist_local_res
        vec_world = M.to_3x3() @ vec_local
        dist_world = vec_world.length

        return (hit_loc_world, hit_norm_world, face_index, dist_world)

    def find_nearest(self, point_world, distance=float("inf")):
        """
        World-space wrapper for BVHTree.find_nearest.
        Returns (co_world, normal_world, face_index, world_distance)
        or (None, None, -1, 0.0)
        """
        M = self.obj.matrix_world
        Minv = M.inverted()
        Minv3 = Minv.to_3x3()
        MinvT3 = Minv3.transposed()

        res = self.bvh.find_nearest(Minv @ point_world, distance)
        if res is None or res[0] is None:
            return (None, None, -1, 0.0)

        co_l, n_l, idx, _ = res
        co_w = M @ co_l
        n_w = (MinvT3 @ n_l).normalized()
        dist_w = (point_world - co_w).length
        return (co_w, n_w, idx, dist_w)
