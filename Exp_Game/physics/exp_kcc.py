#Exploratory/Exp_Game/physics/exp_kcc.py

#steps and bumps great
#slopes need work
##
##
import math
import mathutils
from mathutils import Vector
from .exp_physics import capsule_collision_resolve, remove_steep_slope_component

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class KCCConfig:
    def __init__(self, scene_cfg):
        self.radius          = getattr(scene_cfg, "radius", 0.22)
        self.height          = getattr(scene_cfg, "height", 1.8)
        self.slope_limit_deg = getattr(scene_cfg, "slope_limit_deg", 50.0)
        self.step_height     = getattr(scene_cfg, "step_height", 0.4)
        self.snap_down       = getattr(scene_cfg, "snap_down", 0.5)
        self.gravity         = getattr(scene_cfg, "gravity", -9.81)
        self.max_walk        = getattr(scene_cfg, "max_speed_walk", 2.5)
        self.max_run         = getattr(scene_cfg, "max_speed_run", 5.5)
        self.accel_ground    = getattr(scene_cfg, "accel_ground", 20.0)
        self.accel_air       = getattr(scene_cfg, "accel_air", 5.0)
        self.coyote_time     = getattr(scene_cfg, "coyote_time", 0.08)
        self.jump_buffer     = getattr(scene_cfg, "jump_buffer", 0.12)
        self.jump_speed      = getattr(scene_cfg, "jump_speed", 7.0)
        self.steep_slide_gain = getattr(scene_cfg, "steep_slide_gain", 18.0)
        self.steep_min_speed  = getattr(scene_cfg, "steep_min_speed", 2.5)

class KinematicCharacterController:
    """
    Simple, robust capsule controller:
      • Horizontal pre-sweep to clamp into walls
      • Clean step-up: lift (measured), short forward march (swept), drop
      • Ceiling sweep blocks jump-through ceilings
      • Grounding decoupled from snap
    """

    def __init__(self, obj, scene_cfg):
        self.obj = obj
        self.cfg = KCCConfig(scene_cfg)
        self.vel = Vector((0.0, 0.0, 0.0))
        self.on_ground = False
        self.ground_norm = Vector((0,0,1))
        self.ground_obj = None
        self.on_walkable = True
        self._coyote = 0.0
        self._jump_buf = 0.0
        self._ray_phase = False
        
    
    # --------------------
    # Input & helpers
    # --------------------

    def _input_vector(self, keys_pressed, prefs, camera_yaw):
        fwd_key  = prefs.key_forward
        back_key = prefs.key_backward
        left_key = prefs.key_left
        right_key= prefs.key_right
        run_key  = prefs.key_run

        x = 0.0; y = 0.0
        if fwd_key in keys_pressed:   y += 1.0
        if back_key in keys_pressed:  y -= 1.0
        if right_key in keys_pressed: x += 1.0
        if left_key in keys_pressed:  x -= 1.0

        # Camera‑plane intent (no slope steering)
        v = Vector((x, y, 0.0))
        if v.length > 1e-6:
            v.normalize()

        Rz = mathutils.Matrix.Rotation(camera_yaw, 4, 'Z')
        world3 = Rz @ Vector((v.x, v.y, 0.0))

        # Use XY only for acceleration; keep heading strictly camera‑plane.
        xy = Vector((world3.x, world3.y))
        if xy.length > 1e-6:
            xy.normalize()

        # If standing on a too‑steep slope, remove only the uphill component
        # (prevents "walking up walls" while avoiding terrain‑induced steering).
        if self.on_ground and self.ground_norm is not None and not self._slope_ok(self.ground_norm):
            cos_limit = math.cos(math.radians(self.cfg.slope_limit_deg))
            v3 = Vector((xy.x, xy.y, 0.0))
            v3 = remove_steep_slope_component(v3, self.ground_norm, max_slope_dot=cos_limit)
            xy = Vector((v3.x, v3.y))

        return xy, (run_key in keys_pressed)

    def _hit_in_step_band(self, hit_z: float, origin_z: float) -> bool:
        """
        True if a contact point is within the capsule-based step band:
        [foot_z, foot_z + step_height]
        foot_z is the bottom-sphere center height = origin_z + radius.
        """
        r = float(self.cfg.radius)
        band_top = origin_z + r + float(self.cfg.step_height)
        return hit_z <= (band_top + 0.005)
    
    def _accelerate(self, cur_xy, wish_dir_xy, target_speed, accel, dt):
        wish = Vector((wish_dir_xy[0], wish_dir_xy[1]))
        desired = wish * target_speed
        t = clamp(accel * dt, 0.0, 1.0)
        return cur_xy.lerp(desired, t)

    def _slope_ok(self, n: Vector):
        up = Vector((0,0,1))
        ang = math.degrees(up.angle(n))
        return ang <= self.cfg.slope_limit_deg

    @staticmethod
    def _ray_hits_sphere_segment(origin: Vector, dir_norm: Vector, max_dist: float,
                                center: Vector, radius: float) -> bool:
        """
        Fast prefilter: does the segment [origin, origin + dir_norm*max_dist]
        pass within 'radius' of 'center'? 'dir_norm' must be normalized.
        """
        oc = center - origin
        t = oc.dot(dir_norm)
        if t < -radius or t > max_dist + radius:
            return False
        if t < 0.0:     # before the segment start
            closest = oc
        elif t > max_dist:  # beyond the segment end
            closest = oc - dir_norm * max_dist
        else:
            closest = oc - dir_norm * t
        return closest.length_squared <= (radius * radius)

    @staticmethod
    def _capsule_may_touch_sphere(cap_origin: Vector, cap_height: float, cap_radius: float,
                                sphere_center: Vector, sphere_radius: float, pad: float = 0.25) -> bool:
        """
        Quick reject for capsule-vs-sphere pushout. Approximates capsule by a
        vertical cylinder+caps and tests simple XY ring + Z range overlap.
        """
        dx = sphere_center.x - cap_origin.x
        dy = sphere_center.y - cap_origin.y
        ring = cap_radius + sphere_radius + pad
        if (dx*dx + dy*dy) > ring * ring:
            return False
        z0 = cap_origin.z - pad
        z1 = cap_origin.z + cap_height + pad
        return (sphere_center.z + sphere_radius) >= z0 and (sphere_center.z - sphere_radius) <= z1

    def _raycast_any(self, static_bvh, dynamic_map, origin: Vector, direction: Vector, distance: float):
        """
        Returns (hit_loc, hit_normal, hit_obj, hit_dist) or (None,None,None,None)
        """
        if distance <= 1e-9 or direction.length <= 1e-9:
            return (None, None, None, None)

        dnorm = direction.normalized()
        best = (None, None, None, 1e9)

        if static_bvh:
            hit = static_bvh.ray_cast(origin, dnorm, distance)
            if hit and hit[0] is not None and hit[3] < best[3]:
                best = (hit[0], hit[1], None, hit[3])

        # Prefilter dynamic by ray/segment vs bounding sphere
        if dynamic_map:
            pf_pad = float(self.cfg.radius) + 0.05  # small Minkowski pad
            for obj, (bvh_like, rad) in dynamic_map.items():
                center = obj.matrix_world.translation
                if not self._ray_hits_sphere_segment(origin, dnorm, distance, center, float(rad) + pf_pad):
                    continue    
                hit = bvh_like.ray_cast(origin, dnorm, distance)
                if hit and hit[0] is not None and hit[3] < best[3]:
                    best = (hit[0], hit[1], obj, hit[3])

        return best if best[0] is not None else (None, None, None, None)

    def _raycast_down_any(self, static_bvh, dynamic_map, origin: Vector, max_dist: float):
        """
        Returns (loc, norm, obj, dist) or (None,None,None,None)
        """
        best = (None, None, None, 1e9)
        d = Vector((0,0,-1))
        start = origin + Vector((0,0,1.0))

        if static_bvh:
            hit = static_bvh.ray_cast(start, d, max_dist + 1.0)
            if hit and hit[0] is not None:
                dist = (origin - hit[0]).length
                best = (hit[0], hit[1], None, dist)

        if dynamic_map:
            pf_pad = float(self.cfg.radius) + 0.05
            for obj, (bvh_like, rad) in dynamic_map.items():
                c = obj.matrix_world.translation
                # Quick XY ring check around the ray line
                dx = c.x - start.x; dy = c.y - start.y
                ring = (float(rad) + pf_pad); ring2 = ring * ring
                if (dx*dx + dy*dy) > ring2:
                    continue
                # Approximate Z overlap with the downward segment
                seg_top = start.z
                seg_bot = start.z - (max_dist + 1.0)
                if (c.z + float(rad) + pf_pad) < seg_bot or (c.z - float(rad) - pf_pad) > seg_top:
                    continue

                hit = bvh_like.ray_cast(start, d, max_dist + 1.0)
                if hit and hit[0] is not None:
                    dist = (origin - hit[0]).length
                    if dist < best[3]:
                        best = (hit[0], hit[1], obj, dist)

        return best if best[0] is not None else (None, None, None, None)



    def _sweep_limit_3d(self, static_bvh, dynamic_map, origin: Vector, disp: Vector):
        """
        Ray-based sweep along disp with capsule-aware distances.
        Alternates between LOW-only and MID-only samples per physics step.
        Returns (allowed_len_along_disp, hit_any, wall_normal).
        """
        move_len = disp.length
        if move_len <= 1.0e-6:
            return (0.0, False, None)

        dnorm = disp / move_len
        r = float(self.cfg.radius)

        # Capsule-aware sweep range (Minkowski sum)
        test_dist = move_len + r + 0.05
        skin = 0.02

        min_allowed = move_len
        hit_any     = False
        best_norm   = None

        EARLY_OUT_FRACTION = 0.15
        early_stop_len = move_len * EARLY_OUT_FRACTION

        # ← only one height per step (alternating LOW ↔ MID)
        for h in self._alt_sweep_heights():
            o = origin.copy(); o.z += h
            loc, n, _, dist = self._raycast_any(static_bvh, dynamic_map, o, dnorm, test_dist)
            if loc is None or n is None:
                continue

            n = n.normalized()
            hit_any = True

            allowed = max(0.0, dist - r - skin)
            if allowed < min_allowed:
                min_allowed = allowed
                best_norm   = n
                if min_allowed <= early_stop_len:
                    break

        return (min_allowed, hit_any, best_norm)


    

    def _try_step_up(self, static_bvh, dynamic_map, pos: Vector,
                    forward: Vector, move_len: float,
                    low_hit_normal: Vector, low_hit_dist: float):
        """
        Cheaper step-up that reuses the low forward hit you just computed.
        Only two extra rays max:
        • 1 headroom up-ray
        • 1 raised low forward ray
        """
        if not self.on_ground or move_len <= 1.0e-6:
            return (False, pos)

        r      = float(self.cfg.radius)
        max_up = max(0.0, float(self.cfg.step_height))
        if max_up <= 1.0e-6:
            return (False, pos)

        up = Vector((0, 0, 1))
        floor_cos = math.cos(math.radians(self.cfg.slope_limit_deg))

        # We only step if the low forward contact is a steep riser (not walkable)
        if low_hit_normal is None or low_hit_normal.dot(up) >= floor_cos:
            return (False, pos)

        # 1) Headroom check at true top
        top_start = pos + Vector((0, 0, float(self.cfg.height)))
        locUp, _, _, _ = self._raycast_any(static_bvh, dynamic_map, top_start, up, max_up)
        if locUp is not None:
            return (False, pos)  # no headroom

        # 2) Raise locally
        raised_pos = pos.copy(); raised_pos.z += max_up

        # 3) Single raised low forward ray (capsule-aware stop)
        ray_len = move_len + r
        o_low_raised = raised_pos + Vector((0, 0, r))
        h2, n2, _, d2 = self._raycast_any(static_bvh, dynamic_map, o_low_raised, forward, ray_len)
        if h2 is None:
            raised_pos += forward * move_len
        else:
            allow2 = max(0.0, d2 - r)
            if allow2 > 0.0:
                raised_pos += forward * allow2

        # 4) Single shared downward pass will snap us after caller integrates Z.
        # We still do a safety "drop budget" check here to avoid stepping onto voids.
        drop_max = max_up + max(0.0, float(self.cfg.snap_down))
        locD, nD, gobjD, _ = self._raycast_down_any(static_bvh, dynamic_map, raised_pos, drop_max)
        if locD is None or nD is None or nD.dot(up) < floor_cos:
            return (False, pos)

        new_pos = raised_pos.copy()
        new_pos.z = locD.z
        self.vel.z          = 0.0
        self.on_ground      = True
        self.ground_norm    = nD
        self.ground_obj     = gobjD
        self._coyote        = self.cfg.coyote_time
        return (True, new_pos)



    
    def _alt_sweep_heights(self):
        """
        Alternating horizontal sweep samples:
        • Phase False: LOW only  -> (radius)
        • Phase True:  MID only  -> (0.5 * height)
        """
        r = float(self.cfg.radius)
        if not self._ray_phase:
            return (r,)
        else:
            mid = 0.5 * float(self.cfg.height)
            return (mid,)


    def step(
        self,
        dt,
        prefs,
        keys_pressed,
        camera_yaw,
        static_bvh,
        dynamic_map,
        platform_linear_velocity_map=None,
        platform_ang_velocity_map=None,
    ):
        """
        SIMPLE MODE — further trimmed:
        • Reuse low forward hit in step-up (no multi-height sweep inside).
        • Exactly ONE downward ray per step (shared by vertical integrate + snap).
        • Keep aggressive steep-slope slide + jump rules.
        • Still: low forward + (optional) slide try + (optional) ceiling ray.
        """
        pos = self.obj.location.copy()
        rot = self.obj.rotation_euler.copy()

        # 1) Input / speed
        wish_dir_xy, is_running = self._input_vector(keys_pressed, prefs, camera_yaw)
        target_speed = self.cfg.max_run if is_running else self.cfg.max_walk

        # 2) Timers
        self._coyote   = max(0.0, self._coyote - dt)
        self._jump_buf = max(0.0, self._jump_buf - dt)

        # 3) Horizontal accel
        cur_xy = Vector((self.vel.x, self.vel.y))
        accel  = self.cfg.accel_ground if self.on_ground else self.cfg.accel_air
        new_xy = self._accelerate(cur_xy, wish_dir_xy, target_speed, accel, dt)
        self.vel.x, self.vel.y = new_xy.x, new_xy.y

        up = Vector((0, 0, 1))
        floor_cos = math.cos(math.radians(self.cfg.slope_limit_deg))

        # 4) Vertical + steep behavior
        if not self.on_ground:
            self.vel.z += self.cfg.gravity * dt
        else:
            if self.on_walkable:
                self.vel.z = max(self.vel.z, 0.0)
            else:
                # aggressive downhill on steep
                n = self.ground_norm if self.ground_norm is not None else up
                uphill = up - n * up.dot(n)
                downhill = Vector((-uphill.x, -uphill.y, -uphill.z))
                if downhill.length > 0.0:
                    downhill.normalize()
                    slide_acc = downhill * (abs(self.cfg.gravity) * float(self.cfg.steep_slide_gain))
                    self.vel.x += slide_acc.x * dt
                    self.vel.y += slide_acc.y * dt
                    self.vel.z = min(0.0, self.vel.z + slide_acc.z * dt)

                    d_xy = Vector((downhill.x, downhill.y, 0.0))
                    if d_xy.length > 0.0:
                        d_xy.normalize()
                        v_xy = Vector((self.vel.x, self.vel.y, 0.0))
                        along = v_xy.dot(d_xy)
                        if along < float(self.cfg.steep_min_speed):
                            v_xy = v_xy - d_xy * along + d_xy * float(self.cfg.steep_min_speed)
                            self.vel.x, self.vel.y = v_xy.x, v_xy.y

        # 5) Moving platform carry + yaw follow
        carry = Vector((0, 0, 0))
        if self.on_ground and self.ground_obj:
            v_lin = Vector((0, 0, 0))
            if platform_linear_velocity_map and self.ground_obj in platform_linear_velocity_map:
                v_lin = platform_linear_velocity_map[self.ground_obj]
            v_rot = Vector((0, 0, 0))
            if platform_ang_velocity_map and self.ground_obj in platform_ang_velocity_map:
                omega = platform_ang_velocity_map[self.ground_obj]
                r_vec = (pos - self.ground_obj.matrix_world.translation)
                v_rot = omega.cross(r_vec)
                rot.z += omega.z * dt
            carry = v_lin + v_rot

        # 6) Ceiling ray (only if going up)
        if self.vel.z > 0.0:
            top = pos + Vector((0, 0, float(self.cfg.height)))
            up_dist = self.vel.z * dt
            loc_up, _, _, _ = self._raycast_any(static_bvh, dynamic_map, top, up, up_dist)
            if loc_up is not None:
                pos.z = loc_up.z - float(self.cfg.height)
                self.vel.z = 0.0

        # 7) Horizontal move: low forward ray (and one slide try)
        r = float(self.cfg.radius)
        hvel = Vector((self.vel.x + carry.x, self.vel.y + carry.y, 0.0))

        if self.on_ground and self.ground_norm is not None and not self.on_walkable:
            hv3 = Vector((hvel.x, hvel.y, 0.0))
            hv3 = remove_steep_slope_component(hv3, self.ground_norm, max_slope_dot=floor_cos)
            hvel = Vector((hv3.x, hv3.y, 0.0))

        move_len = hvel.length * dt
        forward  = hvel.normalized() if move_len > 0.0 else Vector((0, 0, 0))

        # Precompute low forward hit ONCE; reuse for step-up decision
        low_hit_normal = None
        low_hit_dist   = 0.0

        if forward.length > 0.0:
            o_low = pos + Vector((0, 0, r))
            ray_len = move_len + r
            hitL, nL, _, dL = self._raycast_any(static_bvh, dynamic_map, o_low, forward, ray_len)

            if self.on_ground and self.cfg.step_height > 0.0 and hitL is not None:
                low_hit_normal = nL.normalized()
                low_hit_dist   = dL

            did_step = False
            if self.on_ground and self.cfg.step_height > 0.0 and low_hit_normal is not None:
                did_step, pos = self._try_step_up(
                    static_bvh, dynamic_map, pos, forward, move_len,
                    low_hit_normal, low_hit_dist
                )

            if (not did_step):
                moved = 0.0
                if hitL is None:
                    pos += forward * move_len
                    moved = move_len
                else:
                    allowed = max(0.0, low_hit_dist - r)
                    if allowed > 0.0:
                        pos += forward * allowed
                        moved = allowed

                    hit_n = low_hit_normal if low_hit_normal is not None else nL.normalized()
                    vn = hvel.dot(hit_n)
                    if vn > 0.0:
                        hvel -= hit_n * vn
                        self.vel.x, self.vel.y = hvel.x - carry.x, hvel.y - carry.y

                    remaining = max(0.0, move_len - moved)
                    # Only do a slide ray if there’s meaningful distance left
                    if remaining > (0.15 * r):
                        slide_dir = forward - hit_n * forward.dot(hit_n)
                        if hit_n.dot(up) < floor_cos:
                            slide_dir = Vector((slide_dir.x, slide_dir.y, 0.0))
                        if slide_dir.length > 0.0:
                            slide_dir.normalize()
                            o2 = pos + Vector((0, 0, r))
                            h2, n2, _, d2 = self._raycast_any(static_bvh, dynamic_map, o2, slide_dir, remaining + r)
                            if h2 is None:
                                pos += slide_dir * remaining
                            else:
                                allow2 = max(0.0, d2 - r)
                                if allow2 > 0.0:
                                    pos += slide_dir * allow2

        # 8) Single DOWNWARD ray shared by both vertical penetration and snap
        dz = self.vel.z * dt
        down_max = self.cfg.snap_down if dz >= 0.0 else max(self.cfg.snap_down, -dz)
        locD, nD, gobjD, _ = self._raycast_down_any(static_bvh, dynamic_map, pos, down_max)

        if dz < 0.0:
            if locD is not None and nD is not None and (pos.z + dz) <= locD.z:
                pos.z = locD.z
                self.vel.z = 0.0
                self.on_ground   = True
                self.on_walkable = (nD.dot(up) >= floor_cos)
                self.ground_norm = nD
                self.ground_obj  = gobjD
                self._coyote     = self.cfg.coyote_time
            else:
                pos.z += dz
        else:
            pos.z += dz

        # Snap/grounding using the SAME hit if it’s within snap window
        was_grounded = self.on_ground
        if locD is not None and nD is not None and (abs(locD.z - pos.z) <= float(self.cfg.snap_down)) and self.vel.z <= 0.0:
            self.on_ground   = True
            self.on_walkable = (nD.dot(up) >= floor_cos)
            self.ground_norm = nD
            self.ground_obj  = gobjD
            pos.z            = locD.z
            self.vel.z       = 0.0
            self._coyote     = self.cfg.coyote_time
        else:
            self.on_ground   = False
            self.on_walkable = False
            self.ground_obj  = None
            if was_grounded:
                self._coyote = self.cfg.coyote_time

        # Write-back
        self.obj.location = pos
        if abs(rot.z - self.obj.rotation_euler.z) > 0.0:
            self.obj.rotation_euler = rot


    def request_jump(self):
        self._jump_buf = self.cfg.jump_buffer

    def try_consume_jump(self):
        """
        Only allow jump if:
        • grounded on WALKABLE ground, or
        • within coyote time (which is granted from walkable support)
        Prevents repeated jumping up non-walkable steep surfaces.
        """
        can_jump_from_ground = (self.on_ground and self.on_walkable)
        if self._jump_buf > 0.0 and (can_jump_from_ground or self._coyote > 0.0):
            self._jump_buf = 0.0
            self.vel.z = self.cfg.jump_speed
            self.on_ground = False
            self.ground_obj = None
            return True
        return False