#Exploratory/Exp_Game/physics/exp_kcc.py
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
        test_dist = move_len + r + 0.05
        skin = 0.02  # small constant along-sweep slack

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
            is_floor_like = (n.dot(up) >= floor_like_dot)

            # Only ignore within the step band if the contact is walkable floor.
            if self.on_ground and is_floor_like and self._hit_in_step_band(loc.z, origin.z):
                continue

            hit_any = True
            allowed = max(0.0, dist - skin)
            if allowed < min_allowed:
                min_allowed = allowed
                best_norm   = n

        return (min_allowed, hit_any, best_norm)


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

        # 3) Horizontal acceleration (strictly camera‑plane)
        cur_xy = Vector((self.vel.x, self.vel.y))
        accel  = self.cfg.accel_ground if self.on_ground else self.cfg.accel_air
        new_xy = self._accelerate(cur_xy, wish_dir_xy, target_speed, accel, dt)
        self.vel.x, self.vel.y = new_xy.x, new_xy.y

        # 4) Gravity
        if not self.on_ground:
            self.vel.z += self.cfg.gravity * dt
        else:
            self.vel.z = max(self.vel.z, 0.0)

        # 5) Moving platform carry velocity (v + ω×r), and yaw follow
        carry = Vector((0,0,0))
        if self.on_ground and self.ground_obj:
            v_lin = Vector((0,0,0))
            if platform_linear_velocity_map and self.ground_obj in platform_linear_velocity_map:
                v_lin = platform_linear_velocity_map[self.ground_obj]

            v_rot = Vector((0,0,0))
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
            top = self.obj.location + Vector((0,0, self.cfg.height))
            up_dist = self.vel.z * dt + 0.02
            loc_up, _, _, _ = self._raycast_any(static_bvh, dynamic_map, top, Vector((0,0,1)), up_dist)
            if loc_up is not None:
                new_top_z = loc_up.z - 0.001
                self.obj.location.z = new_top_z - self.cfg.height
                self.vel.z = 0.0

        # 7) Forward sweep & optional step‑up (no slope steering)
        hvel = Vector((self.vel.x + carry.x, self.vel.y + carry.y, 0.0))

        # If we are on a too‑steep slope, remove only the uphill component; otherwise leave heading untouched.
        if self.on_ground and self.ground_norm is not None and not self._slope_ok(self.ground_norm):
            hv3 = Vector((hvel.x, hvel.y, 0.0))
            cos_limit = math.cos(math.radians(self.cfg.slope_limit_deg))
            hv3 = remove_steep_slope_component(hv3, self.ground_norm, max_slope_dot=cos_limit)
            hvel = Vector((hv3.x, hv3.y, 0.0))

        move_len = hvel.length * dt
        forward  = hvel.normalized() if move_len > 1.0e-8 else Vector((0,0,0))

        # Unified clamp: one sweep that includes Z (via disp vector)
        allowed_len, hit_any, hit_norm = self._sweep_limit_3d(
            static_bvh, dynamic_map, forward * max(0.0, move_len)
        )

        # Always attempt step‑up when moving on ground
        did_step = False

        # 8) Integrate horizontal if we didn't step (with ONE simple slide try)
        if not did_step and forward.length > 1.0e-6:
            moved = 0.0

            # Move up to the hit
            if allowed_len > 0.0:
                self.obj.location += forward * allowed_len
                moved = allowed_len
                

        # 9) Vertical integration with tiny downward CCD (ceilings already handled)
        dz = self.vel.z * dt
        if dz < 0.0:
            up = Vector((0,0,1))
            floor_like_dot = math.cos(math.radians(self.cfg.slope_limit_deg))
            # Cast down far enough to cover this frame's fall distance
            loc, norm, gobj, _ = self._raycast_down_any(
                static_bvh, dynamic_map, self.obj.location, max(-dz, 0.0) + 0.05
            )
            if loc is not None and norm is not None and norm.dot(up) >= floor_like_dot:
                # If applying dz would put us below the floor, clamp and land
                if (self.obj.location.z + dz) <= loc.z:
                    self.obj.location.z = loc.z
                    self.vel.z = 0.0
                    self.on_ground = True
                    self.ground_norm = norm
                    self.ground_obj = gobj
                    self._coyote = self.cfg.coyote_time
                else:
                    self.obj.location.z += dz
            else:
                self.obj.location.z += dz
        else:
            self.obj.location.z += dz

        # 10) Robust capsule pushout (static + dynamic)
        r = float(self.cfg.radius)
        H = float(self.cfg.height)
        mid = max(r, min(0.5 * H, H - r))

        # Capsule-based step band height from base (relative, not world-z):
        #   band = radius + step_height
        band_rel_h = r + float(self.cfg.step_height)

        # If grounded and moving, sample low near the band to clear risers,
        # but NEVER skip steep contacts (handled inside resolver).
        if self.on_ground and forward.length > 1.0e-6:
            low_sample = min(band_rel_h + 0.01, max(0.0, H - r - 0.05))
            mid_sample = max(mid, low_sample + 0.05)
            push_heights = (low_sample, mid_sample, H - r)
        else:
            push_heights = (r, mid, H - r)

        floor_cos   = math.cos(math.radians(self.cfg.slope_limit_deg))
        ignore_floor = (self.on_ground and self.vel.z <= 0.05)

        if static_bvh:
            capsule_collision_resolve(
                static_bvh, self.obj,
                radius=self.cfg.radius,
                heights=push_heights,
                max_iterations=3,
                push_strength=0.6,
                floor_cos_limit=floor_cos,
                ignore_floor_contacts=ignore_floor,          # skip only walkable floors
                ignore_contacts_below_height=band_rel_h,     # band is capsule-based
                ignore_below_only_if_floor_like=True,        # steep contacts never skipped
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

        # 11) Grounding (do not mark grounded while ascending)
        was_grounded = self.on_ground
        loc, norm, gobj, _ = self._raycast_down_any(static_bvh, dynamic_map, self.obj.location, self.cfg.snap_down)
        supported = (loc is not None and norm is not None and self._slope_ok(norm))

        if supported and self.vel.z <= 0.05:
            self.on_ground   = True
            self.ground_norm = norm
            self.ground_obj  = gobj
            self.obj.location.z = loc.z
            self.vel.z = 0.0
            self._coyote = self.cfg.coyote_time
        else:
            self.on_ground  = False
            self.ground_obj = None
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