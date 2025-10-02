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

    def _raycast_any(self, static_bvh, dynamic_map, origin: Vector, direction: Vector, distance: float,
                     platform_motion_map=None,
                     platform_ang_velocity_map=None,
                     dt: float = 0.0):
        """
        Returns (hit_loc, hit_normal, hit_obj, hit_dist) or (None,None,None,None)

        Anti-tunneling prefilter for dynamics:
        pad = char_radius + approx_rad + |disp| + |ω|*approx_rad*dt
        (The last term accounts for rotational sweep at the rim.)
        """
        if distance <= 1e-9 or direction.length <= 1e-9:
            return (None, None, None, None)

        dnorm = direction.normalized()
        best = (None, None, None, 1e9)

        # STATIC
        if static_bvh:
            hit = static_bvh.ray_cast(origin, dnorm, distance)
            if hit and hit[0] is not None and hit[3] < best[3]:
                best = (hit[0], hit[1], None, hit[3])

        # DYNAMIC with speculative pad
        if dynamic_map:
            char_r = float(self.cfg.radius)
            for obj, (bvh_like, approx_rad) in dynamic_map.items():
                disp_len = 0.0
                if platform_motion_map and obj in platform_motion_map:
                    disp_len = platform_motion_map[obj].length

                ang_sweep = 0.0
                if platform_ang_velocity_map and obj in platform_ang_velocity_map and dt > 0.0:
                    ang_sweep = platform_ang_velocity_map[obj].length * float(approx_rad) * float(dt)

                pf_pad = char_r + float(approx_rad) + disp_len + ang_sweep
                center = obj.matrix_world.translation
                if not self._ray_hits_sphere_segment(origin, dnorm, distance, center, pf_pad):
                    continue

                hit = bvh_like.ray_cast(origin, dnorm, distance)
                if hit and hit[0] is not None and hit[3] < best[3]:
                    best = (hit[0], hit[1], obj, hit[3])

        return best if best[0] is not None else (None, None, None, None)



    def _raycast_down_any(self, static_bvh, dynamic_map, origin: Vector, max_dist: float,
                          platform_motion_map=None,
                          platform_ang_velocity_map=None,
                          dt: float = 0.0):
        """
        Returns (loc, norm, obj, dist) or (None,None,None,None)
        Uses the same angular/linear sweep pad in its quick prefilter.
        """
        best = (None, None, None, 1.0e9)
        d = Vector((0,0,-1))
        start = origin + Vector((0,0,1.0))

        if static_bvh:
            hit = static_bvh.ray_cast(start, d, max_dist + 1.0)
            if hit and hit[0] is not None:
                dist = (origin - hit[0]).length
                best = (hit[0], hit[1], None, dist)

        if dynamic_map:
            char_r = float(self.cfg.radius)
            for obj, (bvh_like, approx_rad) in dynamic_map.items():
                # combined pad
                disp_len = platform_motion_map.get(obj, Vector((0,0,0))).length if platform_motion_map else 0.0
                ang_sweep = 0.0
                if platform_ang_velocity_map and obj in platform_ang_velocity_map and dt > 0.0:
                    ang_sweep = platform_ang_velocity_map[obj].length * float(approx_rad) * float(dt)
                ring = char_r + float(approx_rad) + disp_len + ang_sweep

                c = obj.matrix_world.translation
                dx = c.x - start.x; dy = c.y - start.y
                if (dx*dx + dy*dy) > (ring * ring):
                    continue

                # crude Z overlap window with same pad
                seg_top = start.z
                seg_bot = start.z - (max_dist + 1.0)
                if (c.z + float(approx_rad) + ring) < seg_bot or (c.z - float(approx_rad) - ring) > seg_top:
                    continue

                hit = bvh_like.ray_cast(start, d, max_dist + 1.0)
                if hit and hit[0] is not None:
                    dist = (origin - hit[0]).length
                    if dist < best[3]:
                        best = (hit[0], hit[1], obj, dist)

        return best if best[0] is not None else (None, None, None, None)


    def _try_step_up(self, static_bvh, dynamic_map, pos: Vector,
                     forward: Vector, move_len: float,
                     low_hit_normal: Vector, low_hit_dist: float,
                     platform_motion_map=None,
                     platform_ang_velocity_map=None,
                     dt: float = 0.0):
        """
        Cheaper step-up that reuses the low forward hit you just computed.
        Now rotation-safe: all rays use the same linear+angular speculative pads.
        """
        if not self.on_ground or move_len <= 1.0e-6:
            return (False, pos)

        r      = float(self.cfg.radius)
        max_up = max(0.0, float(self.cfg.step_height))
        if max_up <= 1.0e-6:
            return (False, pos)

        up = Vector((0, 0, 1))
        floor_cos = math.cos(math.radians(self.cfg.slope_limit_deg))

        # Only step if the low forward contact is a steep riser (not walkable)
        if low_hit_normal is None or low_hit_normal.dot(up) >= floor_cos:
            return (False, pos)

        # 1) Headroom at true top
        top_start = pos + Vector((0, 0, float(self.cfg.height)))
        locUp, _, _, _ = self._raycast_any(
            static_bvh, dynamic_map, top_start, up, max_up,
            platform_motion_map=platform_motion_map,
            platform_ang_velocity_map=platform_ang_velocity_map,
            dt=dt
        )
        if locUp is not None:
            return (False, pos)  # no headroom

        # 2) Raise locally
        raised_pos = pos.copy(); raised_pos.z += max_up

        # 3) Single raised low forward ray (capsule-aware stop)
        ray_len = move_len + r
        o_low_raised = raised_pos + Vector((0, 0, r))
        h2, n2, _, d2 = self._raycast_any(
            static_bvh, dynamic_map, o_low_raised, forward, ray_len,
            platform_motion_map=platform_motion_map,
            platform_ang_velocity_map=platform_ang_velocity_map,
            dt=dt
        )
        if h2 is None:
            raised_pos += forward * move_len
        else:
            allow2 = max(0.0, d2 - r)
            if allow2 > 0.0:
                raised_pos += forward * allow2

        # 4) Safety drop check using same pads
        drop_max = max_up + max(0.0, float(self.cfg.snap_down))
        locD, nD, gobjD, _ = self._raycast_down_any(
            static_bvh, dynamic_map, raised_pos, drop_max,
            platform_motion_map=platform_motion_map,
            platform_ang_velocity_map=platform_ang_velocity_map,
            dt=dt
        )
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
        platform_motion_map=None,   # per-object frame displacement (Vector)
    ):
        """
        SIMPLE MODE — trimmed + anti-ghosting (unchanged aim)

        Fixes in this version:
        • Treat platform carry as full 3D (XY + Z) and include it in vertical integration.
        • Expand snap window by the platform’s upward displacement for this step.
        • Permit snap while riding a rising platform even if self.vel.z > 0.
        """
        pos = self.obj.location.copy()
        rot = self.obj.rotation_euler.copy()

        # 0) Cheap proxy push (kept)
        if dynamic_map:
            cap_r   = float(self.cfg.radius)
            cap_h   = float(self.cfg.height)
            cap_org = pos.copy()
            for obj, (_bvh_like, approx_rad) in dynamic_map.items():
                v_lin = Vector((0, 0, 0))
                if platform_linear_velocity_map and obj in platform_linear_velocity_map:
                    v_lin = platform_linear_velocity_map[obj]

                omega = Vector((0, 0, 0))
                if platform_ang_velocity_map and obj in platform_ang_velocity_map:
                    omega = platform_ang_velocity_map[obj]

                if v_lin.length <= 1.0e-9 and omega.length <= 1.0e-9:
                    continue

                center = obj.matrix_world.translation
                r_vec  = (pos - center)
                v_rot  = omega.cross(r_vec)

                swept = v_lin.length * dt + omega.length * float(approx_rad) * dt
                push_r = float(approx_rad) + swept
                if self._capsule_may_touch_sphere(cap_org, cap_h, cap_r, center, push_r, pad=0.0):
                    if not (self.on_ground and self.ground_obj is obj):
                        v_proxy = v_lin + v_rot
                        if v_proxy.length > 0.0:
                            pos += v_proxy * dt

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

        # 4) Vertical + steep behavior (unchanged)
        if not self.on_ground:
            self.vel.z += self.cfg.gravity * dt
        else:
            if self.on_walkable:
                self.vel.z = max(self.vel.z, 0.0)
            else:
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

        # 5) Moving platform carry (NOW full 3D)
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
            carry = v_lin + v_rot  # X, Y, and Z

        # 6) Ceiling ray (only if going up) — pads preserved
        if self.vel.z > 0.0:
            top = pos + Vector((0, 0, float(self.cfg.height)))
            up_dist = self.vel.z * dt
            loc_up, _, _, _ = self._raycast_any(
                static_bvh, dynamic_map, top, up, up_dist,
                platform_motion_map=platform_motion_map,
                platform_ang_velocity_map=platform_ang_velocity_map,
                dt=dt
            )
            if loc_up is not None:
                pos.z = loc_up.z - float(self.cfg.height)
                self.vel.z = 0.0

        # 7) Horizontal move with single low ray (+ slide try) — pads preserved
        r = float(self.cfg.radius)
        hvel = Vector((self.vel.x + carry.x, self.vel.y + carry.y, 0.0))

        if self.on_ground and self.ground_norm is not None and not self._slope_ok(self.ground_norm):
            hv3 = Vector((hvel.x, hvel.y, 0.0))
            hv3 = remove_steep_slope_component(hv3, self.ground_norm, max_slope_dot=floor_cos)
            hvel = Vector((hv3.x, hv3.y, 0.0))

        move_len = hvel.length * dt
        forward  = hvel.normalized() if move_len > 0.0 else Vector((0, 0, 0))

        low_hit_normal = None
        low_hit_dist   = 0.0

        if forward.length > 0.0:
            o_low = pos + Vector((0, 0, r))
            ray_len = move_len + r
            hitL, nL, _, dL = self._raycast_any(
                static_bvh, dynamic_map, o_low, forward, ray_len,
                platform_motion_map=platform_motion_map,
                platform_ang_velocity_map=platform_ang_velocity_map,
                dt=dt
            )

            if self.on_ground and self.cfg.step_height > 0.0 and hitL is not None:
                low_hit_normal = nL.normalized()
                low_hit_dist   = dL

            did_step = False
            if self.on_ground and self.cfg.step_height > 0.0 and low_hit_normal is not None:
                did_step, pos = self._try_step_up(
                    static_bvh, dynamic_map, pos, forward, move_len,
                    low_hit_normal, low_hit_dist,
                    platform_motion_map=platform_motion_map,
                    platform_ang_velocity_map=platform_ang_velocity_map,
                    dt=dt
                )

            if (not did_step):
                moved = 0.0
                if hitL is None:
                    pos += forward * move_len
                    moved = move_len
                else:
                    allowed = max(0.0, dL - r)
                    if allowed > 0.0:
                        pos += forward * allowed
                        moved = allowed

                    hit_n = low_hit_normal if low_hit_normal is not None else nL.normalized()
                    vn = hvel.dot(hit_n)
                    if vn > 0.0:
                        hvel -= hit_n * vn
                        self.vel.x, self.vel.y = hvel.x - carry.x, hvel.y - carry.y

                    remaining = max(0.0, move_len - moved)
                    if remaining > (0.15 * r):
                        slide_dir = forward - hit_n * forward.dot(hit_n)
                        if hit_n.dot(up) < floor_cos:
                            slide_dir = Vector((slide_dir.x, slide_dir.y, 0.0))
                        if slide_dir.length > 0.0:
                            slide_dir.normalize()
                            o2 = pos + Vector((0, 0, r))
                            h2, n2, _, d2 = self._raycast_any(
                                static_bvh, dynamic_map, o2, slide_dir, remaining + r,
                                platform_motion_map=platform_motion_map,
                                platform_ang_velocity_map=platform_ang_velocity_map,
                                dt=dt
                            )
                            if h2 is None:
                                pos += slide_dir * remaining
                            else:
                                allow2 = max(0.0, d2 - r)
                                if allow2 > 0.0:
                                    pos += slide_dir * allow2

        # 8) Downward grounding/snap — **include vertical carry**
        dz = (self.vel.z + carry.z) * dt

        # If we move up by dz this tick, allow extra snap budget equal to dz.
        if dz >= 0.0:
            down_max = float(self.cfg.snap_down) + dz
        else:
            down_max = max(float(self.cfg.snap_down), -dz)

        locD, nD, gobjD, _ = self._raycast_down_any(
            static_bvh, dynamic_map, pos, down_max,
            platform_motion_map=platform_motion_map,
            platform_ang_velocity_map=platform_ang_velocity_map,
            dt=dt
        )

        # Integrate vertical with carry
        pos.z += dz

        was_grounded = self.on_ground

        # Snap window expands by the amount we moved upward this step.
        snap_window = float(self.cfg.snap_down) + max(0.0, dz)

        if locD is not None and nD is not None and abs(locD.z - pos.z) <= snap_window:
            # Allow snap if: (a) not moving up relative to support, or (b) we’re on/near a rising platform.
            allow_snap = (self.vel.z <= 0.0) or (carry.z > 0.0) or (gobjD is not None and gobjD == self.ground_obj)
            if allow_snap:
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