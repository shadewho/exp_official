#Exploratory/Exp_Game/physics/exp_kcc.py
#removed
#march over lip. never want to add forward velocity.

#still poor uphill slopes or any steep angles. i know large capsules solve the collision issue
#if capsule goes up it solves steep issue but then steps dont work. they need to work together

"""
-changes to this file are fragile and must be tested thoroughly. 
-test steep slopes, mesh collisions at many angles, and steps. 

-many of the physics functions are tied closely with the stepping logic,
test small steps large steps, sprinting and running etc.

-steps are handled by boosting the character upward while maintaining
forward velocity

-steps are not detected by the capsule radius, but rather a single point tied to the capsule

-steps are prone to capsule conflict, make sure that a wide capsule does not block the
registration of steps in any of the slope logic or other traverse code. ie dont let a wall 
collisions or steep slope interfere with steps under the step height value.

"""
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

    # --- Step tunables (no UI knobs; chosen to be sane and universal) ---
    _STEP_LIFT_OVERSHOOT  = 0.01   # tiny extra headroom to avoid grazing undersides
    _STEP_MARCH_ITERS     = 1      # a few micro-sweeps to crest the lip cleanly
    _EPS                  = 1.0e-6

    # --- Radius-independent step probes (meters) ---
    _STEP_PROBE_OFFSETS_M = (0.0, +0.12, -0.12)   # center, shoulders in meters (not % of radius)
    _STEP_PROBE_FORWARD_M = 0.35                  # how far ahead to look for the tread
    _STEP_FOOT_H_M        = 0.10                  # foot sample height for riser feel
    _STEP_CLEAR_RADIUS_M  = 0.10                  # effective "clearance sphere" for lip math   

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
    def _project_on_plane(self, v3, n):
        if n.length <= 1e-9:
            return v3
        n = n.normalized()
        return v3 - n * v3.dot(n)
    
    def _dynamic_lip_clear(self, static_bvh, dynamic_map, origin: Vector, forward_xy: Vector) -> float:
        """
        VERTICAL-ONLY VERSION:
        Compute additional *vertical* lift (meters) required so a swept sphere placed
        at the current XY won't intersect the nearby riser plane. We DO NOT push
        forward here—caller should only use this to increase lift.

        Returns: extra_lift_z >= 0.0 (meters)
        """
        EPS = 1.0e-6
        up = Vector((0, 0, 1))

        # Reuse same forward sample directions to *find* the riser plane,
        # but we won't advance along forward.
        f = Vector((forward_xy.x, forward_xy.y, 0.0))
        if f.length <= EPS:
            return 0.0
        f.normalize()

        R = float(self._STEP_CLEAR_RADIUS_M)
        SAFETY = max(0.010, min(0.60 * R, 0.025))

        # Ignore floors (we only care about the vertical-ish riser face)
        floor_dot = math.cos(math.radians(self.cfg.slope_limit_deg))

        # Heights to probe in front of the capsule
        foot_h = float(self._STEP_FOOT_H_M)
        heights = (foot_h, self.cfg.height * 0.5)

        best_extra_z = 0.0

        for h in heights:
            o = origin.copy(); o.z += float(h)
            # Just far enough to hit the riser near the nose
            max_dist = max(0.15, 2.5 * R)
            hit_loc, hit_n, _, t = self._raycast_any(static_bvh, dynamic_map, o, f, max_dist)
            if hit_loc is None or hit_n is None:
                continue

            # Skip floor-like contacts; we're clearing a riser
            if hit_n.dot(up) >= floor_dot:
                continue

            # Normal distance from our sample point to the plane ≈ t * (f · n)
            fn = max(f.dot(hit_n), 0.0)
            if fn <= EPS:
                # Forward ~parallel to the plane; vertical lift won't help plane clearance
                # (we will rely on the general step lift and subsequent ticks).
                continue

            d0 = t * fn                                      # current normal separation
            need_n = (R + SAFETY) - d0                       # missing gap along plane normal
            if need_n <= 0.0:
                continue

            # Lifting by Δz changes normal separation by (up · n) * Δz.
            # Solve (up·n)*Δz >= need_n  →  Δz >= need_n / max(up·n, EPS)
            upn = max(hit_n.dot(up), EPS)
            extra_z = need_n / upn

            if extra_z > best_extra_z:
                best_extra_z = extra_z

        return max(0.0, best_extra_z)
    

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

    def _hit_below_step_window(self, hit_z: float, origin_z: float) -> bool:
        """
        True if a contact point is within the vertical band the step logic owns:
        [feet + _STEP_FOOT_H_M, feet + _STEP_FOOT_H_M + step_height]
        We treat wall-like hits in this band as NON-blocking during sweeps.
        """
        band_top = float(self._STEP_FOOT_H_M) + float(self.cfg.step_height)
        return (hit_z - origin_z) <= (band_top + 0.005)  # 5mm slack
    
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
        Swept-sphere idea along the displacement (includes Z).
        Returns (allowed_len_along_disp, hit_any, wall_normal).
        Floor-like hits are ignored so we don't fight grounding.
        """
        move_len = disp.length
        if move_len <= 1e-6:
            return (0.0, False, None)

        dnorm  = disp / move_len
        origin = self.obj.location.copy()

        # height samples: foot/mid/top of capsule
        foot_h = float(self._STEP_FOOT_H_M)
        heights = (foot_h, self.cfg.height * 0.5, self.cfg.height - 0.1)

        skin = max(0.01, min(0.03, self.cfg.radius * 0.15))
        test_dist = move_len + self.cfg.radius + 0.05
        up        = Vector((0, 0, 1))

        min_allowed = move_len
        hit_any     = False
        best_norm   = None

        floor_like_dot = math.cos(math.radians(self.cfg.slope_limit_deg))

        for h in heights:
            o = origin.copy(); o.z += h
            loc, n, _, dist = self._raycast_any(static_bvh, dynamic_map, o, dnorm, test_dist)
            if loc is None or n is None:
                continue

            # Ignore floors (as before)
            if n.dot(up) >= floor_like_dot:
                continue

            # NEW: if the wall contact is within the step window, let step logic handle it
            if self.on_ground and self._hit_below_step_window(loc.z, origin.z):
                continue

            hit_any = True
            allowed = max(0.0, dist - skin)
            if allowed < min_allowed:
                min_allowed = allowed
                best_norm   = n.normalized()

        return (min_allowed, hit_any, best_norm)

    def _estimate_step_height_ahead(self, static_bvh, dynamic_map, origin: Vector,
                                    forward_xy: Vector, sample_forward: float, max_height: float):
        """
        Radius-independent: probe center and ± fixed meter offsets, not ±k*radius.
        Returns (rise, top_norm, top_loc) or (None,None,None).
        """
        base_loc, base_norm, _, _ = self._raycast_down_any(
            static_bvh, dynamic_map, origin, max(1.0, self.cfg.snap_down + max_height + 0.5)
        )
        if base_loc is None:
            return (None, None, None)

        f = Vector((forward_xy.x, forward_xy.y, 0.0))
        if f.length <= 1e-6 or sample_forward <= 1e-6:
            return (0.0, base_norm, origin)
        f.normalize()

        right = Vector((f.y, -f.x, 0.0))  # XY 90° CW

        lateral_offsets = tuple(self._STEP_PROBE_OFFSETS_M)
        ahead = max(sample_forward, float(self._STEP_PROBE_FORWARD_M))

        best_rise = 1e9
        best = (None, None, None)

        for off in lateral_offsets:
            probe_xy = origin + f * ahead + right * off
            top_loc, top_norm, _, _ = self._raycast_down_any(
                static_bvh, dynamic_map, probe_xy, max_height + 1.0
            )
            if top_loc is None or top_norm is None:
                continue
            h = top_loc.z - base_loc.z
            if h > 0.005 and h <= (self.cfg.step_height + 1.0e-3) and self._slope_ok(top_norm):
                if h < best_rise:
                    best_rise = h
                    best = (h, top_norm, top_loc)

        return best

    def _steep_guard_block(self, static_bvh, dynamic_map, origin: Vector, forward_xy: Vector) -> Vector:
        """
        Conservative pre-sweep guard that prevents tunneling into steep faces
        inside the step window *only when no real step is possible*.

        Returns: projected XY direction (same length or shorter).
        """
        EPS = 1.0e-6
        if forward_xy.length <= EPS:
            return forward_xy

        f = Vector((forward_xy.x, forward_xy.y, 0.0)).normalized()
        up = Vector((0,0,1))
        floor_dot = math.cos(math.radians(self.cfg.slope_limit_deg))

        # 1) Is a true step actually possible? (walkable top + genuine riser)
        sample_forward = float(self._STEP_PROBE_FORWARD_M)
        max_h = self.cfg.step_height + 0.05
        step_res = self._estimate_step_height_ahead(static_bvh, dynamic_map, origin, f, sample_forward, max_h)
        has_walkable_top = bool(step_res and step_res[0] is not None and step_res[1] is not None and self._slope_ok(step_res[1]))
        has_riser = self._dynamic_lip_clear(static_bvh, dynamic_map, origin, f) > 1.0e-4
        step_possible = (has_walkable_top and has_riser)

        if step_possible:
            # Do not interfere with step logic.
            return forward_xy

        # 2) Probe a short distance ahead at foot height to detect near-vertical face
        foot_h = float(self._STEP_FOOT_H_M)
        o = origin.copy(); o.z += foot_h

        # Short reach ~ capsule radius; enough to catch faces in the step window
        reach = max(0.10, self.cfg.radius * 1.25)
        loc, n, _, dist = self._raycast_any(static_bvh, dynamic_map, o, f, reach)
        if loc is None or n is None:
            return forward_xy

        # Ignore floor-like contacts; we only care about steep faces
        if n.dot(up) >= floor_dot:
            return forward_xy

        # If the contact is actually within the step window heights, treat as "wall"
        if self._hit_below_step_window(loc.z, origin.z):
            # Remove the into-wall component so we can't advance into the face this tick.
            v = Vector((forward_xy.x, forward_xy.y, 0.0))
            into = v.dot(n)
            if into > 0.0:
                v = v - n * into
                if v.length > EPS:
                    v.normalize()
                    return v
                else:
                    return Vector((0.0, 0.0, 0.0))

        return forward_xy
    
    def _try_step_up(self, dt, static_bvh, dynamic_map, forward: Vector, desired_len: float):
        """
        Step-up with RISER-GATED behavior and forward carry:
        1) Require real horizontal intent and walkable top within step_height
        2) Require a genuine riser (non-floor-like face) in front
        3) Lift vertically (headroom-checked) + small extra from _dynamic_lip_clear
        4) Drop onto tread and micro-march forward to carry momentum
        """
        EPS = self._EPS
        if desired_len <= EPS or forward.length <= EPS:
            return False
        if self.cfg.step_height <= EPS:
            return False
        if not self.on_ground:
            return False

        # Ignore vanishingly small intent this tick (prevents “free hops”)
        if desired_len < 0.015:
            return False

        origin = self.obj.location.copy()

        # Measure rise ahead (what the top would be if we succeeded)
        sample_forward = float(self._STEP_PROBE_FORWARD_M)
        max_h = self.cfg.step_height + 0.05
        res = self._estimate_step_height_ahead(
            static_bvh, dynamic_map, origin, forward, sample_forward, max_h
        )
        if not res or res[0] is None or res[1] is None:
            return False

        h, top_norm, top_loc = res
        if h <= 0.005 or h > (self.cfg.step_height + 1.0e-3):
            return False
        if not self._slope_ok(top_norm):
            return False

        # ── NEW, IMPORTANT: only treat as a step if there is a genuine riser ahead ──
        # (_dynamic_lip_clear returns >0 only when a non-floor-like face is detected in front.)
        extra_probe = self._dynamic_lip_clear(static_bvh, dynamic_map, origin, forward)
        if extra_probe <= 1.0e-4:
            # Continuous slope/ramp — let normal sweep handle walking
            return False

        # Headroom check for the baseline lift
        lift = min(self.cfg.step_height, max(0.0, h)) + self._STEP_LIFT_OVERSHOOT
        cap_top = origin + Vector((0, 0, self.cfg.height))
        up_hit = self._raycast_any(static_bvh, dynamic_map, cap_top, Vector((0, 0, 1)), lift + 0.01)
        if up_hit[0] is not None:
            return False  # no ceiling room for lift

        saved = origin.copy()

        # 3a) baseline vertical lift
        self.obj.location = saved + Vector((0, 0, lift))

        # 3b) add just-enough extra vertical clearance for the riser if available
        extra_lift = extra_probe
        if extra_lift > EPS:
            # Re-check headroom for the extra lift
            extra_hit = self._raycast_any(static_bvh, dynamic_map, cap_top + Vector((0, 0, lift)),
                                        Vector((0, 0, 1)), extra_lift + 0.01)
            if extra_hit[0] is None:
                self.obj.location.z += extra_lift  # safe to add vertical clearance

        # 4) Drop onto walkable top; if fail, revert
        loc, norm, gobj, _ = self._raycast_down_any(
            static_bvh, dynamic_map, self.obj.location, lift + self.cfg.snap_down
        )
        if loc and norm and self._slope_ok(norm):
            self.obj.location.z = loc.z
            self.on_ground = True
            self.ground_obj = gobj
            self.ground_norm = norm
            self._coyote = self.cfg.coyote_time
            self.vel.z = min(self.vel.z, 0.0)
            return True

        # Failed → revert to original state
        self.obj.location = saved
        return False


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
        if self.on_ground and forward.length > 1.0e-6:
            did_step = self._try_step_up(dt, static_bvh, dynamic_map, forward, move_len)

        # 8) Integrate horizontal if we didn't step (with ONE simple slide try)
        if not did_step and forward.length > 1.0e-6:
            moved = 0.0

            # Move up to the hit
            if allowed_len > 0.0:
                self.obj.location += forward * allowed_len
                moved = allowed_len

            # If we hit something, try a single slide along the contact plane
            if hit_any and hit_norm is not None:
                remaining = max(0.0, move_len - moved)
                if remaining > 1.0e-6:
                    slide_vec = forward - hit_norm * forward.dot(hit_norm)
                    if slide_vec.length > 1.0e-6:
                        slide_dir = slide_vec.normalized()
                        allowed2, _, _ = self._sweep_limit_3d(static_bvh, dynamic_map, slide_dir * remaining)
                        if allowed2 > 0.0:
                            self.obj.location += slide_dir * allowed2

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

        # 10) Robust capsule pushout (static + dynamic), ignoring floors if grounded
        r = float(self.cfg.radius)
        H = float(self.cfg.height)
        mid = max(r, min(0.5 * H, H - r))

        if self.on_ground and forward.length > 1.0e-6:
            step_band_top = float(self._STEP_FOOT_H_M) + float(self.cfg.step_height)
            low_sample = max(r, step_band_top + 0.01)          # a hair above the band
            mid_sample = max(mid, low_sample + 0.05)           # keep an upper sample
            push_heights = (low_sample, mid_sample, H - r)
        else:
            push_heights = (r, mid, H - r)

        floor_cos = math.cos(math.radians(self.cfg.slope_limit_deg))
        ignore_floor = (self.on_ground and self.vel.z <= 0.05)

        if static_bvh:
            capsule_collision_resolve(
                static_bvh, self.obj,
                radius=self.cfg.radius,
                heights=push_heights,
                max_iterations=3,
                push_strength=0.6,
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
                    push_strength=0.6,
                    floor_cos_limit=floor_cos,
                    ignore_floor_contacts=ignore_floor,
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