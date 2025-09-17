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

        if dynamic_map:
            for obj, (bvh_like, _) in dynamic_map.items():
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
            hit = static_bvh.ray_cast(start, d, max_dist+1.0)
            if hit and hit[0] is not None:
                dist = (origin - hit[0]).length
                best = (hit[0], hit[1], None, dist)

        if dynamic_map:
            for obj, (bvh_like, _) in dynamic_map.items():
                hit = bvh_like.ray_cast(start, d, max_dist+1.0)
                if hit and hit[0] is not None:
                    dist = (origin - hit[0]).length
                    if dist < best[3]:
                        best = (hit[0], hit[1], obj, dist)

        return best if best[0] is not None else (None, None, None, None)

    def _sweep_limit_3d(self, static_bvh, dynamic_map, disp: Vector):
        """
        Ray-based sweep along disp with capsule-aware distances.
        • Cast at heights (r, H*0.5, H-r).
        • Floors (≤ slope_limit) inside the step band are ignored to allow stepping.
        • Steep surfaces (> slope_limit) are NEVER ignored, even inside the band.
        Returns (allowed_len_along_disp, hit_any, wall_normal).
        """
        move_len = disp.length
        if move_len <= 1.0e-6:
            return (0.0, False, None)

        dnorm  = disp / move_len
        origin = self.obj.location.copy()

        r = float(self.cfg.radius)
        H = float(self.cfg.height)
        heights = (r, 0.5 * H, H - r)

        # Capsule-aware sweep range (Minkowski sum)
        test_dist = move_len + r + 0.05   # cast far enough (Minkowski)
        skin = 0.02                       # general separation (not a slope fudge)

        up = Vector((0, 0, 1))
        floor_like_dot = math.cos(math.radians(self.cfg.slope_limit_deg))

        min_allowed = move_len
        hit_any     = False
        best_norm   = None

        for h in heights:
            o = origin.copy(); o.z += h
            loc, n, _, dist = self._raycast_any(static_bvh, dynamic_map, o, dnorm, test_dist)
            if loc is None or n is None:
                continue

            n = n.normalized()
            # NOTE: no step-band skipping of “floor-like” contacts here

            hit_any = True
            # CRITICAL: subtract the capsule radius so the center never advances into the face
            allowed = max(0.0, dist - r - skin)
            if allowed < min_allowed:
                min_allowed = allowed
                best_norm   = n

        return (min_allowed, hit_any, best_norm)
    

    def _try_step_up(self, static_bvh, dynamic_map, forward: Vector, move_len: float) -> bool:
        """
        Explicit step-up:
          1) Detect a steep riser directly ahead (low sample).
          2) Ensure headroom for raising by up to step_height.
          3) Temporarily raise, sweep forward, then drop onto a walkable top.
        Returns True if we performed a step and updated position/grounding.
        """
        if not self.on_ground or move_len <= 1.0e-6:
            return False

        r      = float(self.cfg.radius)
        H      = float(self.cfg.height)
        max_up = max(0.0, float(self.cfg.step_height))
        if max_up <= 1.0e-6:
            return False

        origin = self.obj.location.copy()
        up     = Vector((0, 0, 1))
        floor_cos = math.cos(math.radians(self.cfg.slope_limit_deg))

        # 1) Low forward ray (near foot) to detect a vertical riser
        low_h = min(r, H - r)   # sample near the base (not a slope fudge)
        o_low = origin.copy(); o_low.z += low_h
        hitL, nL, _, distL = self._raycast_any(
            static_bvh, dynamic_map, o_low, forward, move_len + r + 0.05
        )
        if hitL is None:
            return False
        
        # Only step if what we hit is a steep barrier (not walkable)
        if nL.dot(up) >= floor_cos:
            return False

        # 2) Headroom: from current top, must be able to raise by max_up
        top_start = origin + Vector((0, 0, H))
        locUp, _, _, _ = self._raycast_any(static_bvh, dynamic_map, top_start, up, max_up + 0.01)
        if locUp is not None:
            return False  # no headroom to raise

        # 3) Raise temporarily
        raise_h = max_up
        self.obj.location.z += raise_h

        # 4) Re-run forward sweep from the raised pose
        allow2, hit2, norm2 = self._sweep_limit_3d(static_bvh, dynamic_map, forward * move_len)
        if allow2 > 0.0:
            self.obj.location += forward * allow2

        # 5) Drop down to ground (≤ raise_h + snap_down) and validate walkable top
        drop_max = raise_h + max(0.0, float(self.cfg.snap_down))
        locD, nD, gobjD, _ = self._raycast_down_any(static_bvh, dynamic_map, self.obj.location, drop_max)
        if locD is None or nD is None or nD.dot(up) < floor_cos:
            # Revert if no valid walkable top
            self.obj.location.z -= raise_h
            return False

        # Land on the step top
        self.obj.location.z = locD.z
        self.vel.z          = 0.0
        self.on_ground      = True
        self.ground_norm    = nD
        self.ground_obj     = gobjD
        self._coyote        = self.cfg.coyote_time
        return True

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
        # 1) Input and target speed
        wish_dir_xy, is_running = self._input_vector(keys_pressed, prefs, camera_yaw)
        target_speed = self.cfg.max_run if is_running else self.cfg.max_walk

        # 2) Timers (coyote & jump buffer)
        self._coyote   = max(0.0, self._coyote - dt)
        self._jump_buf = max(0.0, self._jump_buf - dt)

        # 3) Horizontal acceleration (strictly camera-plane)
        cur_xy = Vector((self.vel.x, self.vel.y))
        accel  = self.cfg.accel_ground if self.on_ground else self.cfg.accel_air
        new_xy = self._accelerate(cur_xy, wish_dir_xy, target_speed, accel, dt)
        self.vel.x, self.vel.y = new_xy.x, new_xy.y

        up = Vector((0, 0, 1))
        floor_cos = math.cos(math.radians(self.cfg.slope_limit_deg))

        # 4) Gravity and steep-slope sliding
        if not self.on_ground:
            # Airborne → normal gravity
            self.vel.z += self.cfg.gravity * dt
        else:
            if self.on_walkable:
                # On walkable ground: never accumulate downward z
                self.vel.z = max(self.vel.z, 0.0)
            else:
                # On too-steep ground: apply gravity's *tangential* component to slide
                g = Vector((0, 0, self.cfg.gravity))
                n = self.ground_norm if self.ground_norm is not None else up
                g_tan = g - n * g.dot(n)  # gravity along the plane

                # --- Stronger slide: simple multiplier
                g_tan *= 12.0

                self.vel.x += g_tan.x * dt
                self.vel.y += g_tan.y * dt
                self.vel.z = min(0.0, self.vel.z + g_tan.z * dt)


        # 5) Moving platform carry velocity (v + ω×r), and yaw follow
        carry = Vector((0, 0, 0))
        if self.on_ground and self.ground_obj:
            v_lin = Vector((0, 0, 0))
            if platform_linear_velocity_map and self.ground_obj in platform_linear_velocity_map:
                v_lin = platform_linear_velocity_map[self.ground_obj]

            v_rot = Vector((0, 0, 0))
            if platform_ang_velocity_map and self.ground_obj in platform_ang_velocity_map:
                omega = platform_ang_velocity_map[self.ground_obj]  # rad/s
                r = (self.obj.matrix_world.translation - self.ground_obj.matrix_world.translation)
                v_rot = omega.cross(r)

                # rotate yaw with platform spin (z-only)
                yaw_delta = omega.z * dt
                eul = self.obj.rotation_euler
                eul.z += yaw_delta
                self.obj.rotation_euler = eul

            carry = v_lin + v_rot

        # 6) Ceiling sweep (block upward motion before integration)
        if self.vel.z > 0.0:
            top = self.obj.location + Vector((0, 0, self.cfg.height))
            up_dist = self.vel.z * dt + 0.02
            loc_up, _, _, _ = self._raycast_any(static_bvh, dynamic_map, top, Vector((0, 0, 1)), up_dist)
            if loc_up is not None:
                new_top_z = loc_up.z - 0.001
                self.obj.location.z = new_top_z - self.cfg.height
                self.vel.z = 0.0

        # 7) Horizontal sweep (single sweep + simple slide). Clamp uphill on steep.
        hvel = Vector((self.vel.x + carry.x, self.vel.y + carry.y, 0.0))

        if self.on_ground and self.ground_norm is not None and not self.on_walkable:
            # Remove uphill component along the plane so you can’t climb
            hv3 = Vector((hvel.x, hvel.y, 0.0))
            cos_limit = math.cos(math.radians(self.cfg.slope_limit_deg))
            hv3 = remove_steep_slope_component(hv3, self.ground_norm, max_slope_dot=cos_limit)
            hvel = Vector((hv3.x, hv3.y, 0.0))

        move_len = hvel.length * dt
        forward  = hvel.normalized() if move_len > 1.0e-8 else Vector((0, 0, 0))

        allowed_len, hit_any, hit_norm = self._sweep_limit_3d(
            static_bvh, dynamic_map, forward * max(0.0, move_len)
        )

        did_step = False
        if self.on_ground and forward.length > 1.0e-6 and self.cfg.step_height > 0.0:
            did_step = self._try_step_up(static_bvh, dynamic_map, forward, move_len)

        # 8) Integrate horizontal if we didn't step (with ONE simple slide try)
        if not did_step and forward.length > 1.0e-6:
            moved = 0.0
            if allowed_len > 0.0:
                self.obj.location += forward * allowed_len
                moved = allowed_len

            if hit_any and hit_norm is not None:
                vn = Vector((self.vel.x + carry.x, self.vel.y + carry.y, 0.0)).dot(hit_norm)
                if vn > 0.0:
                    hvel -= hit_norm * vn
                    self.vel.x, self.vel.y = hvel.x - carry.x, hvel.y - carry.y

                rem = max(0.0, move_len - moved)
                # Tangent slide direction
                slide_dir = forward - hit_norm * forward.dot(hit_norm)

                # -- if the surface is above the slope limit, slide only in XY (treat as a wall)
                up = Vector((0, 0, 1))
                floor_cos = math.cos(math.radians(self.cfg.slope_limit_deg))
                if hit_norm.dot(up) < floor_cos:
                    slide_dir = Vector((slide_dir.x, slide_dir.y, 0.0))

                if slide_dir.length > 1.0e-6 and rem > 0.0:
                    slide_dir.normalize()
                    allow2, _, _ = self._sweep_limit_3d(static_bvh, dynamic_map, slide_dir * rem)
                    if allow2 > 0.0:
                        self.obj.location += slide_dir * allow2

        # 9) Vertical integration with tiny downward CCD (ceilings already handled)
        dz = self.vel.z * dt
        if dz < 0.0:
            loc, norm, gobj, _ = self._raycast_down_any(
                static_bvh, dynamic_map, self.obj.location, max(-dz, 0.0) + 0.05
            )
            if loc is not None and norm is not None:
                will_penetrate = (self.obj.location.z + dz) <= loc.z
                if will_penetrate:
                    # Land on the contact. Treat as grounded even if too steep.
                    self.obj.location.z = loc.z
                    self.vel.z = 0.0
                    self.on_ground   = True
                    self.on_walkable = (norm.dot(up) >= floor_cos)
                    self.ground_norm = norm
                    self.ground_obj  = gobj
                    self._coyote     = self.cfg.coyote_time
                else:
                    self.obj.location.z += dz
            else:
                self.obj.location.z += dz
        else:
            self.obj.location.z += dz

        # 10) Robust capsule pushout (static + dynamic) — unchanged behavior
        r = float(self.cfg.radius)
        H = float(self.cfg.height)
        mid = max(r, min(0.5 * H, H - r))
        band_rel_h = r + float(self.cfg.step_height)

        if self.on_ground and forward.length > 1.0e-6:
            low_sample = min(band_rel_h + 0.01, max(0.0, H - r - 0.05))
            mid_sample = max(mid, low_sample + 0.05)
            push_heights = (low_sample, mid_sample, H - r)
        else:
            push_heights = (r, mid, H - r)

        ignore_floor = (self.on_ground and self.vel.z <= 0.05)

        if static_bvh:
            capsule_collision_resolve(
                static_bvh, self.obj,
                radius=self.cfg.radius,
                heights=push_heights,
                max_iterations=3,
                push_strength=0.6,
                floor_cos_limit=floor_cos,
                ignore_floor_contacts=ignore_floor,          # only skips walkable floors
                ignore_contacts_below_height=band_rel_h,
                ignore_below_only_if_floor_like=True,
            )

        if dynamic_map:
            for bvh_like, _ in dynamic_map.values():
                capsule_collision_resolve(
                    bvh_like, self.obj,
                    radius=self.cfg.radius,
                    heights=push_heights,
                    max_iterations=3,
                    push_strength=0.6,
                    floor_cos_limit=floor_cos,
                    ignore_floor_contacts=ignore_floor,
                    ignore_contacts_below_height=band_rel_h,
                    ignore_below_only_if_floor_like=True,
                )

        # 11) Grounding pass (snap-down). “Supported” = grounded; walkable is a flag.
        was_grounded = self.on_ground
        loc, norm, gobj, _ = self._raycast_down_any(static_bvh, dynamic_map, self.obj.location, self.cfg.snap_down)
        has_contact = (loc is not None and norm is not None)

        if has_contact and self.vel.z <= 0.05:
            self.on_ground   = True
            self.on_walkable = (norm.dot(up) >= floor_cos)
            self.ground_norm = norm
            self.ground_obj  = gobj
            self.obj.location.z = loc.z
            self.vel.z = 0.0
            self._coyote = self.cfg.coyote_time
        else:
            self.on_ground   = False
            self.on_walkable = False
            self.ground_obj  = None
            if was_grounded:
                self._coyote = self.cfg.coyote_time




    def request_jump(self):
        self._jump_buf = self.cfg.jump_buffer

    def try_consume_jump(self):
        if self._jump_buf > 0.0 and (self.on_ground or self._coyote > 0.0):
            self._jump_buf = 0.0
            self.vel.z = self.cfg.jump_speed
            self.on_ground = False
            self.ground_obj = None
            return True
        return False