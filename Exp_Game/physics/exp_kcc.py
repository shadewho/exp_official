# Exp_Game/physics/exp_kcc.py
import math
import mathutils
from mathutils import Vector
from .exp_physics import capsule_collision_resolve

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
        self.step_forward_min = getattr(scene_cfg, "step_forward_min", 0.20)

class KinematicCharacterController:
    """
    Simple, robust capsule controller:
      • Horizontal pre-sweep to clamp into walls
      • Proper step-up: lift -> move -> drop with free-space checks
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
    def _project_on_plane(self, v3, n):
        """Project 3D vector v3 onto the plane with normal n."""
        if n.length <= 1e-9:
            return v3
        n = n.normalized()
        return v3 - n * v3.dot(n)
    # --------------------
    # Input & helpers
    # --------------------
    def _input_vector(self, keys_pressed, prefs, camera_yaw):
        from mathutils import Vector
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

        v = Vector((x, y, 0.0))
        if v.length > 1e-6:
            v.normalize()

        # Camera-space to world-space (horizontal)
        Rz = mathutils.Matrix.Rotation(camera_yaw, 4, 'Z')
        world3 = Rz @ Vector((v.x, v.y, 0.0))

        # While grounded on a walkable slope, align to the tangent plane,
        # but DO NOT normalize in 3D (that steals magnitude into Z).
        if self.on_ground and self.ground_norm is not None and self._slope_ok(self.ground_norm):
            world3 = self._project_on_plane(world3, self.ground_norm)
            xy = Vector((world3.x, world3.y))
            if xy.length > 1e-6:
                xy.normalize()  # keep full horizontal intent (unit in XY)
                return xy, (run_key in keys_pressed)
            # fallback if XY collapsed (rare)
            return Vector((0.0, 0.0)), (run_key in keys_pressed)

        # Flat ground / air: original behavior
        return world3.xy, (run_key in keys_pressed)

    def _accelerate(self, cur_xy, wish_dir_xy, target_speed, accel, dt):
        wish = Vector((wish_dir_xy[0], wish_dir_xy[1]))
        desired = wish * target_speed
        t = clamp(accel * dt, 0.0, 1.0)  # critically-damped lerp
        return cur_xy.lerp(desired, t)

    def _slope_ok(self, n: Vector):
        up = Vector((0,0,1))
        # angle between normal and up should be <= slope limit
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
        Swept-sphere idea along the displacement (includes Z).
        Returns (allowed_len_along_disp, hit_any, wall_normal).
        Floor-like hits are ignored so we don't fight grounding.
        """
        move_len = disp.length
        if move_len <= 1e-6:
            return (0.0, False, None)

        dnorm  = disp / move_len
        origin = self.obj.location.copy()

        # LOWER foot sample so stairs/low ledges are actually "seen"
        # 3–8 cm, clamped by capsule radius
        foot_h = max(0.03, min(0.08, 0.5 * self.cfg.radius))
        heights = (foot_h, self.cfg.height * 0.5, self.cfg.height - 0.1)

        skin      = max(0.03, self.cfg.radius * 0.95)
        test_dist = move_len + self.cfg.radius + 0.05
        up        = Vector((0, 0, 1))

        min_allowed = move_len
        hit_any     = False
        best_norm   = None

        for h in heights:
            o = origin.copy(); o.z += h
            loc, n, _, dist = self._raycast_any(static_bvh, dynamic_map, o, dnorm, test_dist)
            if loc is None or n is None:
                continue

            # ignore floor-like surfaces
            floor_like_dot = math.cos(math.radians(self.cfg.slope_limit_deg))
            if n.dot(up) >= floor_like_dot:
                continue

            hit_any = True
            allowed = max(0.0, dist - skin)
            if allowed < min_allowed:
                min_allowed = allowed
                best_norm   = n.normalized()

        return (min_allowed, hit_any, best_norm)


    def _forward_sweep_limit(self, static_bvh, dynamic_map, forward_xy: Vector, move_len: float):
        """
        Forward pre-sweep using three vertical samples. We classify 'low bump' vs 'wall'
        by comparing the low/mid hit distances. If only the low sample hits (or hits
        much earlier than mid), we flag it as a curb rather than a full wall so the
        step solver can do its job.
        """
        self._last_forward_low_bump = False

        if move_len <= 1e-6 or forward_xy.length <= 1e-6:
            return (0.0, False, None)

        disp = Vector((forward_xy.x, forward_xy.y, 0.0)) * move_len
        move_len = disp.length
        dnorm = disp / move_len

        origin = self.obj.location.copy()
        up = Vector((0, 0, 1))
        floor_like_dot = math.cos(math.radians(self.cfg.slope_limit_deg))

        # Height samples: low (curb sniff), mid, near-top
        foot_h = max(0.03, min(0.08, 0.5 * self.cfg.radius))
        heights = (foot_h, self.cfg.height * 0.5, self.cfg.height - 0.1)

        # Sphere-sweep length (ray + radius reserve)
        test_dist = move_len + self.cfg.radius + 0.05

        # We allow the sweep to approach a touch closer than a strict radius.
        # The capsule pushout that runs after integration still guarantees no penetration.
        # This little slack keeps tiny lips from behaving like full walls.
        contact_slack = 0.02  # ~2cm
        contact_subtract = max(0.0, self.cfg.radius - contact_slack)

        min_allowed = move_len
        hit_any = False
        best_norm = None

        # For curb classification
        low_hit = False
        mid_hit = False
        low_dist = None
        mid_dist = None

        for idx, h in enumerate(heights):
            o = origin.copy(); o.z += h
            loc, n, _, dist = self._raycast_any(static_bvh, dynamic_map, o, dnorm, test_dist)
            if loc is None or n is None:
                continue

            # Ignore floor-like hits (i.e., we're not trying to block on walkable slopes)
            if n.dot(up) >= floor_like_dot:
                continue

            hit_any = True
            allowed_here = max(0.0, dist - contact_subtract)
            if allowed_here < min_allowed:
                min_allowed = allowed_here
                best_norm = n.normalized()

            if idx == 0:      # low sample
                low_hit = True
                low_dist = dist
            elif idx == 1:    # mid sample
                mid_hit = True
                mid_dist = dist

        # Classify curb vs wall: if the low sample hits but the mid sample is
        # either absent or much farther away, treat the obstacle as a small bump.
        if hit_any and low_hit:
            curb_gap = self.cfg.radius * 0.75  # how much farther mid must be to count as a curb
            is_curb = (not mid_hit) or (mid_dist is None) or ((mid_dist - (low_dist or 0.0)) > curb_gap)
            if is_curb:
                self._last_forward_low_bump = True
                # Ensure we never clamp motion to zero on a curb; give the step solver room to act.
                min_allowed = max(min_allowed, min(0.10, self.cfg.radius * 0.35))

        return (min_allowed, hit_any, best_norm)


    def _try_step_up(self, dt, static_bvh, dynamic_map, forward: Vector, desired_len: float):
        """
        Classic step offset (distance-aware version):
        1) detect low obstacle at foot height, with free space near step height (mid)
        2) ensure overhead clearance (no low ceiling)
        3) lift by step_height
        4) move horizontally (guarantee a tiny push across the lip)
        5) drop down (snap)
        """
        if desired_len <= 1e-6 or forward.length <= 1e-6:
            return False
        if self.cfg.step_height <= 1e-6:
            return False
        if not self.on_ground:
            return False

        origin = self.obj.location.copy()

        # Samples
        foot_h = max(0.03, min(0.08, 0.5 * self.cfg.radius))
        mid_h  = min(self.cfg.height * 0.5, foot_h + self.cfg.step_height + 0.05)

        # Robust look-ahead independent of speed
        min_probe = max(self.cfg.radius * 1.5, 0.25)
        test_dist = max(min_probe, desired_len + self.cfg.radius + 0.02)

        # Low hit?
        foot_o = origin.copy(); foot_o.z += foot_h
        loc_low, _, _, dist_low = self._raycast_any(
            static_bvh, dynamic_map, foot_o, Vector((forward.x, forward.y, 0.0)), test_dist
        )
        if loc_low is None:
            return False  # nothing to step onto

        # Mid must be clear "near the low hit". A hit far away shouldn't cancel the step.
        mid_o = origin.copy(); mid_o.z += mid_h
        loc_mid, _, _, dist_mid = self._raycast_any(
            static_bvh, dynamic_map, mid_o, Vector((forward.x, forward.y, 0.0)), test_dist
        )
        if loc_mid is not None and dist_mid <= (dist_low + self.cfg.radius * 0.75):
            # Mid-height is blocked right where the low is → real wall
            return False

        # Overhead clearance for the lift
        cap_top = origin.copy(); cap_top.z += self.cfg.height
        up_clear = self._raycast_any(static_bvh, dynamic_map, cap_top, Vector((0, 0, 1)), self.cfg.step_height + 0.02)
        if up_clear[0] is not None:
            return False

        # 3) lift
        saved = origin.copy()
        self.obj.location = saved + Vector((0, 0, self.cfg.step_height + 0.02))

        # 4) move horizontally after lifting
        disp_after_lift = Vector((forward.x, forward.y, 0.0)) * desired_len
        if disp_after_lift.length > 1e-6:
            dnorm = disp_after_lift.normalized()
        else:
            dnorm = Vector((forward.x, forward.y, 0.0)).normalized() if forward.length > 1e-6 else Vector((0, 0, 0))

        # We still check for walls after the lift…
        allowed_after_lift, _, _ = self._sweep_limit_3d(static_bvh, dynamic_map, disp_after_lift)

        # …but we guarantee a minimal lip‑crossing push to clear the riser.
        required_clear = self.cfg.radius + 0.03  # 3 cm safety over geometric radius
        min_push = max(required_clear, self.cfg.step_forward_min, desired_len)
        push_len = max(allowed_after_lift, min_push)
        if push_len > 1e-6 and dnorm.length > 1e-6:
            self.obj.location += dnorm * push_len

        # 5) drop down to ground
        loc, norm, gobj, _ = self._raycast_down_any(static_bvh, dynamic_map, self.obj.location, self.cfg.step_height + self.cfg.snap_down)
        if loc and norm and self._slope_ok(norm):
            self.obj.location.z = loc.z
            self.on_ground = True
            self.ground_obj = gobj
            self.ground_norm = norm
            self._coyote = self.cfg.coyote_time
            self.vel.z = min(self.vel.z, 0.0)
            return True

        # Failed → revert
        self.obj.location = saved
        return False
    def _micro_bump_step(self, static_bvh, dynamic_map, forward: Vector, move_len: float):
        """Last-resort curb hop for tiny lips. Very small lift, nudge, then drop."""
        if forward.length <= 1e-6:
            return False

        origin = self.obj.location.copy()
        # Respect capsule height; keep this tiny.
        lift = min(self.cfg.step_height * 0.5, 0.06)

        # Overhead clearance
        cap_top = origin + Vector((0, 0, self.cfg.height))
        up_clear = self._raycast_any(static_bvh, dynamic_map, cap_top, Vector((0, 0, 1)), lift + 0.02)
        if up_clear[0] is not None:
            return False

        self.obj.location = origin + Vector((0, 0, lift))

        dnorm = forward.normalized()
        min_push = max(self.cfg.radius + 0.03, self.cfg.step_forward_min, move_len)
        self.obj.location += dnorm * min_push

        loc, norm, gobj, _ = self._raycast_down_any(static_bvh, dynamic_map, self.obj.location, lift + self.cfg.snap_down)
        if loc and norm and self._slope_ok(norm):
            self.obj.location.z = loc.z
            self.on_ground = True
            self.ground_obj = gobj
            self.ground_norm = norm
            self._coyote = self.cfg.coyote_time
            self.vel.z = min(self.vel.z, 0.0)
            return True

        # Revert if we didn’t find stable ground
        self.obj.location = origin
        return False
    # --------------------
    # Public stepping API
    # --------------------
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

        # 3) Horizontal acceleration
        cur_xy = Vector((self.vel.x, self.vel.y))
        accel  = self.cfg.accel_ground if self.on_ground else self.cfg.accel_air
        new_xy = self._accelerate(cur_xy, wish_dir_xy, target_speed, accel, dt)
        self.vel.x, self.vel.y = new_xy.x, new_xy.y

        # 4) Gravity
        if not self.on_ground:
            self.vel.z += self.cfg.gravity * dt
        else:
            # On walkable ground, kill downward velocity so gravity can't bleed into downhill slide
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
            loc_up, _, _, dist_up = self._raycast_any(static_bvh, dynamic_map, top, Vector((0,0,1)), up_dist)
            if loc_up is not None:
                new_top_z = loc_up.z - 0.001
                self.obj.location.z = new_top_z - self.cfg.height
                self.vel.z = 0.0

        # 7) Forward sweep & optional step-up  (use tangent velocity on ground)
        hvel = Vector((self.vel.x + carry.x, self.vel.y + carry.y, 0.0))

        if self.on_ground and self.ground_norm is not None and hvel.length > 1e-8:
            # Project direction to slope tangent but keep the original horizontal speed.
            speed_xy = Vector((hvel.x, hvel.y)).length
            hv3 = self._project_on_plane(Vector((hvel.x, hvel.y, 0.0)), self.ground_norm)
            hv_xy = Vector((hv3.x, hv3.y))
            if hv_xy.length > 1e-6:
                hv_xy.normalize()
                hv_xy *= speed_xy
                hvel = Vector((hv_xy.x, hv_xy.y, 0.0))

        move_len = hvel.length * dt
        forward  = hvel.normalized() if move_len > 1e-8 else Vector((0,0,0))

        allowed_len, hit_wall, wall_norm = self._forward_sweep_limit(static_bvh, dynamic_map, forward, move_len) if move_len > 0 else (0.0, False, None)

        did_step = False
        if hit_wall and self.on_ground and forward.length > 1e-6:
            did_step = self._try_step_up(dt, static_bvh, dynamic_map, forward, move_len)
            if (not did_step) and self._last_forward_low_bump:
                # Tiny curb hop as a last resort
                did_step = self._micro_bump_step(static_bvh, dynamic_map, forward, move_len)

        # 8) Integrate position (horizontal then vertical)
        if not did_step:
            if allowed_len > 0.0 and forward.length > 1e-6:
                self.obj.location += forward * allowed_len
            self.obj.location.z += self.vel.z * dt

        # 9) Robust capsule pushout (static + dynamic)
        #    IMPORTANT: sample at [radius, mid_clamped, height - radius] (not foot_h),
        #    otherwise the solver thinks we penetrate flat ground constantly.
        r = float(self.cfg.radius)
        H = float(self.cfg.height)
        mid = max(r, min(0.5 * H, H - r))
        push_heights = (r, mid, H - r)

        # Treat walkable floors specially to prevent slope creep.
        floor_cos = math.cos(math.radians(self.cfg.slope_limit_deg))
        ignore_floor = (self.on_ground and self.vel.z <= 0.05)

        if static_bvh:
            capsule_collision_resolve(
                static_bvh, self.obj,
                radius=self.cfg.radius,
                heights=push_heights,
                max_iterations=4,
                push_strength=0.9,
                floor_cos_limit=floor_cos,
                ignore_floor_contacts=ignore_floor,
            )

        if dynamic_map:
            for bvh_like, _ in dynamic_map.values():
                capsule_collision_resolve(
                    bvh_like, self.obj,
                    radius=self.cfg.radius,
                    heights=push_heights,
                    max_iterations=3,
                    push_strength=0.9,
                    floor_cos_limit=floor_cos,
                    ignore_floor_contacts=ignore_floor,
                )

        # 10) Grounding (do not mark grounded while ascending)
        was_grounded = self.on_ground

        loc, norm, gobj, dist = self._raycast_down_any(static_bvh, dynamic_map, self.obj.location, self.cfg.snap_down)
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
