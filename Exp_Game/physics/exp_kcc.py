# Exploratory/Exp_Game/physics/exp_kcc.py
import math
import mathutils
from mathutils import Vector

# ---- Small helpers ---------------------------------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

_UP = Vector((0.0, 0.0, 1.0))

def remove_steep_slope_component(move_dir: Vector, slope_normal: Vector, max_slope_dot: float = 0.7) -> Vector:
    """
    XY-only uphill clamp for steep slopes.
    Returns a vector with z = 0.0. No behavior change vs. original.
    """
    n = slope_normal
    if n.length <= 1.0e-12:
        return Vector((move_dir.x, move_dir.y, 0.0))
    n = n.normalized()

    # Walkable => unchanged
    if n.dot(_UP) >= float(max_slope_dot):
        return Vector((move_dir.x, move_dir.y, 0.0))

    # In-plane uphill direction and its XY
    uphill = _UP - n * _UP.dot(n)
    g_xy = Vector((uphill.x, uphill.y))
    if g_xy.length <= 1.0e-12:
        return Vector((move_dir.x, move_dir.y, 0.0))
    g_xy.normalize()

    m_xy = Vector((move_dir.x, move_dir.y))
    comp = m_xy.dot(g_xy)
    if comp > 0.0:
        m_xy -= g_xy * comp  # remove only the uphill component

    return Vector((m_xy.x, m_xy.y, 0.0))

# ---- Config ----------------------------------------------------------------

class KCCConfig:
    def __init__(self, scene_cfg):
        self.radius           = getattr(scene_cfg, "radius", 0.22)
        self.height           = getattr(scene_cfg, "height", 1.8)
        self.slope_limit_deg  = getattr(scene_cfg, "slope_limit_deg", 50.0)
        self.step_height      = getattr(scene_cfg, "step_height", 0.4)
        self.snap_down        = getattr(scene_cfg, "snap_down", 0.5)
        self.gravity          = getattr(scene_cfg, "gravity", -9.81)
        self.max_walk         = getattr(scene_cfg, "max_speed_walk", 2.5)
        self.max_run          = getattr(scene_cfg, "max_speed_run", 5.5)
        self.accel_ground     = getattr(scene_cfg, "accel_ground", 20.0)
        self.accel_air        = getattr(scene_cfg, "accel_air", 5.0)
        self.coyote_time      = getattr(scene_cfg, "coyote_time", 0.08)
        self.jump_buffer      = getattr(scene_cfg, "jump_buffer", 0.12)
        self.jump_speed       = getattr(scene_cfg, "jump_speed", 7.0)
        self.steep_slide_gain = getattr(scene_cfg, "steep_slide_gain", 18.0)
        self.steep_min_speed  = getattr(scene_cfg, "steep_min_speed", 2.5)

# ---- Controller -------------------------------------------------------------

class KinematicCharacterController:
    """
    Capsule controller:
      • Horizontal pre-sweep (height bands), optional slide
      • Clean step-up: lift -> short forward -> drop
      • Ceiling ray blocks jump-through ceilings
      • Single downward ray shared by penetration & snap
    """

    def __init__(self, obj, scene_cfg):
        self.obj = obj
        self.cfg = KCCConfig(scene_cfg)

        self.vel          = Vector((0.0, 0.0, 0.0))
        self.on_ground    = False
        self.on_walkable  = True
        self.ground_norm  = _UP.copy()
        self.ground_obj   = None
        self._coyote      = 0.0
        self._jump_buf    = 0.0

        # Cached constants to avoid recomputation every step
        self._up = _UP
        self._floor_cos = math.cos(math.radians(self.cfg.slope_limit_deg))
        self._refresh_bands()

    # If you allow live changes to radius/height/step_height, call this again.
    def _refresh_bands(self):
        r = float(self.cfg.radius)
        h = float(self.cfg.height)
        z_low   = r
        z_above = min(h - r, r + float(self.cfg.step_height) + r)  # clearly above the step band
        z_waist = clamp(h * 0.5, r, h - r)
        z_chest = clamp(h * (2.0 / 3.0), r, h - r)
        z_head  = h - r
        # Unique & sorted
        self._bands = tuple(sorted({float(clamp(z, r, h - r)) for z in (z_low, z_above, z_waist, z_chest, z_head)}))
        self._z_low = z_low

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

        # Camera-plane intent (no terrain-induced yaw)
        v_len2 = x*x + y*y
        if v_len2 > 1.0e-12:
            inv_len = 1.0 / math.sqrt(v_len2)
            vx = x * inv_len
            vy = y * inv_len
        else:
            vx = vy = 0.0

        # Rotate by camera yaw about Z
        Rz = mathutils.Matrix.Rotation(camera_yaw, 4, 'Z')
        world3 = Rz @ Vector((vx, vy, 0.0))
        xy_len2 = world3.x * world3.x + world3.y * world3.y
        if xy_len2 > 1.0e-12:
            inv_xy = 1.0 / math.sqrt(xy_len2)
            xy = Vector((world3.x * inv_xy, world3.y * inv_xy))
        else:
            xy = Vector((0.0, 0.0))

        # Steep-slope uphill removal when standing on non-walkable
        if self.on_ground and self.ground_norm is not None and not self._slope_ok(self.ground_norm):
            v3 = Vector((xy.x, xy.y, 0.0))
            v3 = remove_steep_slope_component(v3, self.ground_norm, max_slope_dot=self._floor_cos)
            xy = Vector((v3.x, v3.y))

        return xy, (run_key in keys_pressed)

    def _accelerate(self, cur_xy: Vector, wish_dir_xy: Vector, target_speed: float, accel: float, dt: float) -> Vector:
        desired = Vector((wish_dir_xy.x * target_speed, wish_dir_xy.y * target_speed))
        t = clamp(accel * dt, 0.0, 1.0)
        return cur_xy.lerp(desired, t)

    def _slope_ok(self, n: Vector) -> bool:
        # Exact comparison to your configured limit (no eps tweaks)
        return math.degrees(_UP.angle(n)) <= self.cfg.slope_limit_deg

    # ---- Fast geometric prefilters / rays ----------------------------------

    @staticmethod
    def _ray_hits_sphere_segment(origin: Vector, dir_norm: Vector, max_dist: float,
                                 center: Vector, radius: float) -> bool:
        oc = center - origin
        t = oc.dot(dir_norm)
        if t < -radius or t > max_dist + radius:
            return False
        if t < 0.0:
            closest = oc
        elif t > max_dist:
            closest = oc - dir_norm * max_dist
        else:
            closest = oc - dir_norm * t
        return closest.length_squared <= (radius * radius)

    def _raycast_any(self, static_bvh, dynamic_map, origin: Vector, direction: Vector, distance: float):
        """Generic ray (normalizes direction). Returns (loc, norm, obj, dist) or (None, None, None, None)."""
        if distance <= 1.0e-9 or direction.length <= 1.0e-9:
            return (None, None, None, None)
        return self._raycast_any_norm(static_bvh, dynamic_map, origin, direction.normalized(), distance)

    def _raycast_any_norm(self, static_bvh, dynamic_map, origin: Vector, dnorm: Vector, distance: float):
        """Normalized-direction fast path. Returns (loc, norm, obj, dist) or (None, None, None, None)."""
        if distance <= 1.0e-9:
            return (None, None, None, None)

        best = (None, None, None, 1e9)

        if static_bvh:
            hit = static_bvh.ray_cast(origin, dnorm, distance)
            if hit and hit[0] is not None and hit[3] < best[3]:
                best = (hit[0], hit[1], None, hit[3])

        if dynamic_map:
            pf_pad = float(self.cfg.radius) + 0.05
            for obj, (bvh_like, rad) in dynamic_map.items():
                center = obj.matrix_world.translation
                if not self._ray_hits_sphere_segment(origin, dnorm, distance, center, float(rad) + pf_pad):
                    continue
                hit = bvh_like.ray_cast(origin, dnorm, distance)
                if hit and hit[0] is not None and hit[3] < best[3]:
                    best = (hit[0], hit[1], obj, hit[3])

        return best if best[0] is not None else (None, None, None, None)

    def _raycast_down_any(self, static_bvh, dynamic_map, origin: Vector, max_dist: float):
        """Single downward probe used for both penetration resolve and snap."""
        best = (None, None, None, 1e9)
        d = Vector((0.0, 0.0, -1.0))
        start = origin + Vector((0.0, 0.0, 1.0))  # guard above head

        if static_bvh:
            hit = static_bvh.ray_cast(start, d, max_dist + 1.0)
            if hit and hit[0] is not None:
                dist = (origin - hit[0]).length
                best = (hit[0], hit[1], None, dist)

        if dynamic_map:
            pf_pad = float(self.cfg.radius) + 0.05
            ring2_extra = (pf_pad * pf_pad)
            seg_top = start.z
            seg_bot = start.z - (max_dist + 1.0)

            for obj, (bvh_like, rad) in dynamic_map.items():
                c = obj.matrix_world.translation
                dx = c.x - start.x
                dy = c.y - start.y
                ring = float(rad) + pf_pad
                if (dx*dx + dy*dy) > (ring * ring + ring2_extra):
                    continue
                if (c.z + float(rad) + pf_pad) < seg_bot or (c.z - float(rad) - pf_pad) > seg_top:
                    continue

                hit = bvh_like.ray_cast(start, d, max_dist + 1.0)
                if hit and hit[0] is not None:
                    dist = (origin - hit[0]).length
                    if dist < best[3]:
                        best = (hit[0], hit[1], obj, dist)

        return best if best[0] is not None else (None, None, None, None)

    # ---- Step-up ------------------------------------------------------------

    def _try_step_up(self, static_bvh, dynamic_map, pos: Vector,
                     forward_norm: Vector, move_len: float,
                     low_hit_normal: Vector, low_hit_dist: float):
        if not self.on_ground or move_len <= 1.0e-6:
            return (False, pos)

        r      = float(self.cfg.radius)
        max_up = max(0.0, float(self.cfg.step_height))
        if max_up <= 1.0e-6:
            return (False, pos)

        up = self._up
        floor_cos = self._floor_cos

        # Low band must be a steep riser (non-walkable)
        if low_hit_normal is None or low_hit_normal.dot(up) >= floor_cos:
            return (False, pos)

        # 1) Headroom
        top_start = pos + Vector((0.0, 0.0, float(self.cfg.height)))
        locUp, _, _, _ = self._raycast_any_norm(static_bvh, dynamic_map, top_start, up, max_up)
        if locUp is not None:
            return (False, pos)

        # 2) Raise
        raised_pos = pos.copy()
        raised_pos.z += max_up

        # 3) Raised low forward ray
        ray_len = move_len + r
        o_low_raised = raised_pos + Vector((0.0, 0.0, r))
        h2, n2, _, d2 = self._raycast_any_norm(static_bvh, dynamic_map, o_low_raised, forward_norm, ray_len)

        advanced = 0.0
        if h2 is None:
            raised_pos += forward_norm * move_len
            advanced = move_len
        else:
            allow2 = max(0.0, d2 - r)
            if allow2 > 0.0:
                raised_pos += forward_norm * allow2
            advanced = allow2

        if advanced <= 1.0e-5:
            return (False, pos)

        # 4) Drop
        drop_max = max_up + max(0.0, float(self.cfg.snap_down))
        locD, nD, gobjD, _ = self._raycast_down_any(static_bvh, dynamic_map, raised_pos, drop_max)
        if locD is None or nD is None or nD.dot(up) < floor_cos:
            return (False, pos)

        new_pos = raised_pos.copy()
        new_pos.z = locD.z
        self.vel.z       = 0.0
        self.on_ground   = True
        self.ground_norm = nD
        self.ground_obj  = gobjD
        self._coyote     = self.cfg.coyote_time
        return (True, new_pos)

    # ---- Main step ----------------------------------------------------------

    def step(
        self,
        dt: float,
        prefs,
        keys_pressed,
        camera_yaw: float,
        static_bvh,
        dynamic_map,
        platform_linear_velocity_map=None,
        platform_ang_velocity_map=None,
    ):
        pos = self.obj.location.copy()
        rot = self.obj.rotation_euler.copy()

        cfg        = self.cfg
        up         = self._up
        floor_cos  = self._floor_cos
        bands      = self._bands
        z_low      = self._z_low
        r          = float(cfg.radius)
        h          = float(cfg.height)

        # 1) Input / speed
        wish_dir_xy, is_running = self._input_vector(keys_pressed, prefs, camera_yaw)
        target_speed = cfg.max_run if is_running else cfg.max_walk

        # 2) Timers
        self._coyote   = max(0.0, self._coyote - dt)
        self._jump_buf = max(0.0, self._jump_buf - dt)

        # 3) Horizontal accel
        cur_xy = Vector((self.vel.x, self.vel.y))
        accel  = cfg.accel_ground if self.on_ground else cfg.accel_air
        new_xy = self._accelerate(cur_xy, wish_dir_xy, target_speed, accel, dt)
        self.vel.x, self.vel.y = new_xy.x, new_xy.y

        # 4) Vertical + steep behavior
        if not self.on_ground:
            self.vel.z += cfg.gravity * dt
        else:
            if self.on_walkable:
                self.vel.z = max(self.vel.z, 0.0)
            else:
                n = self.ground_norm if self.ground_norm is not None else up
                uphill   = up - n * up.dot(n)
                downhill = Vector((-uphill.x, -uphill.y, -uphill.z))
                if downhill.length > 0.0:
                    downhill.normalize()
                    slide_acc = downhill * (abs(cfg.gravity) * float(cfg.steep_slide_gain))
                    self.vel.x += slide_acc.x * dt
                    self.vel.y += slide_acc.y * dt
                    self.vel.z  = min(0.0, self.vel.z + slide_acc.z * dt)

                    d_xy = Vector((downhill.x, downhill.y, 0.0))
                    if d_xy.length > 0.0:
                        d_xy.normalize()
                        v_xy  = Vector((self.vel.x, self.vel.y, 0.0))
                        along = v_xy.dot(d_xy)
                        if along < float(cfg.steep_min_speed):
                            v_xy = v_xy - d_xy * along + d_xy * float(cfg.steep_min_speed)
                            self.vel.x, self.vel.y = v_xy.x, v_xy.y

        # 5) Moving platform carry + yaw follow
        carry = Vector((0.0, 0.0, 0.0))
        if self.on_ground and self.ground_obj:
            v_lin = platform_linear_velocity_map.get(self.ground_obj, Vector((0.0, 0.0, 0.0))) if platform_linear_velocity_map else Vector((0.0, 0.0, 0.0))
            v_rot = Vector((0.0, 0.0, 0.0))
            if platform_ang_velocity_map and self.ground_obj in platform_ang_velocity_map:
                omega = platform_ang_velocity_map[self.ground_obj]
                r_vec = (pos - self.ground_obj.matrix_world.translation)
                v_rot = omega.cross(r_vec)
                rot.z += omega.z * dt
            carry = v_lin + v_rot

        # 6) Ceiling (only if going up)
        if self.vel.z > 0.0:
            top     = pos + Vector((0.0, 0.0, h))
            up_dist = self.vel.z * dt
            loc_up, _, _, _ = self._raycast_any_norm(static_bvh, dynamic_map, top, up, up_dist)
            if loc_up is not None:
                pos.z     = loc_up.z - h
                self.vel.z = 0.0

        # 7) Horizontal move (bands + optional slide)
        hvel = Vector((self.vel.x + carry.x, self.vel.y + carry.y, 0.0))

        # Clamp uphill component on too-steep when grounded (unchanged)
        if self.on_ground and self.ground_norm is not None and not self.on_walkable:
            hv3 = remove_steep_slope_component(Vector((hvel.x, hvel.y, 0.0)), self.ground_norm, max_slope_dot=floor_cos)
            hvel = Vector((hv3.x, hv3.y, 0.0))

        # Stationary early-out for forward sweeps: keeps snap/grounding later
        dz = self.vel.z * dt
        hvel_len2 = hvel.x*hvel.x + hvel.y*hvel.y
        if self.on_ground and abs(self.vel.z) < 1.0e-6 and hvel_len2 < 1.0e-10:
            # Skip forward sweeps & ceiling when fully idle; vertical block still runs.
            pass
        else:
            move_total = (math.sqrt(hvel_len2) * dt) if hvel_len2 > 0.0 else 0.0
            if move_total <= 1.0e-9:
                forward = Vector((0.0, 0.0, 0.0))
                fwd_norm = forward
            else:
                forward  = hvel / (move_total / dt)   # normalized * dt cancels below
                fwd_norm = forward.normalized()

            max_step = max(r, 0.9 * r)
            if hvel_len2 <= (max_step / max(dt, 1e-12))**2:
                sub_steps = 1
            else:
                sub_steps = min(3, int(math.ceil((math.sqrt(hvel_len2) * dt) / max_step)))
            step_len = move_total / float(sub_steps) if sub_steps > 0 else 0.0

            for _sub in range(sub_steps):
                if step_len <= 1.0e-9 or fwd_norm.length <= 1.0e-9:
                    break

                ray_len = step_len + r

                # Cast along all bands and choose the nearest hit
                best_d = None
                best_n = None
                best_z = None
                hit_low_for_step = (None, None, None, None)

                for z in bands:
                    o = pos + Vector((0.0, 0.0, z))
                    hL, nL, oL, dL = self._raycast_any_norm(static_bvh, dynamic_map, o, fwd_norm, ray_len)

                    if z == z_low:
                        hit_low_for_step = (hL, nL, oL, dL)

                    if hL is None:
                        continue
                    nN = nL.normalized()
                    if best_d is None or dL < best_d:
                        best_d, best_n, best_z = dL, nN, z

                # Step-up driven strictly by low band
                did_step = False
                if self.on_ground and cfg.step_height > 0.0:
                    hL, nL, _oL, dL = hit_low_for_step
                    if hL is not None:
                        did_step, pos = self._try_step_up(
                            static_bvh, dynamic_map, pos, fwd_norm, step_len,
                            nL.normalized(), dL
                        )

                if did_step:
                    continue

                # No step: clamp advance by nearest band hit
                if best_d is None:
                    pos += fwd_norm * step_len
                else:
                    allowed = max(0.0, best_d - r)
                    moved = 0.0
                    if allowed > 0.0:
                        pos += fwd_norm * allowed
                        moved = allowed

                    # Cheap slide: project velocity off blocking normal
                    vn = hvel.dot(best_n)
                    if vn > 0.0:
                        hvel -= best_n * vn
                        self.vel.x, self.vel.y = hvel.x - carry.x, hvel.y - carry.y

                    remaining = max(0.0, step_len - moved)
                    if remaining > (0.15 * r):
                        # Slide tangent to normal; XY-only on too-steep
                        slide_dir = fwd_norm - best_n * fwd_norm.dot(best_n)
                        if best_n.dot(_UP) < floor_cos:
                            slide_dir = Vector((slide_dir.x, slide_dir.y, 0.0))
                        if slide_dir.length > 1.0e-12:
                            slide_dir.normalize()
                            slide_z = r if best_z is None else float(clamp(best_z, r, h - r))
                            o2 = pos + Vector((0.0, 0.0, slide_z))
                            h2, n2, _o2, d2 = self._raycast_any_norm(static_bvh, dynamic_map, o2, slide_dir, remaining + r)
                            if h2 is None:
                                pos += slide_dir * remaining
                            else:
                                allow2 = max(0.0, d2 - r)
                                if allow2 > 0.0:
                                    pos += slide_dir * allow2

        # 8) Single DOWNWARD ray shared by both vertical penetration and snap
        #    (We intentionally reuse the same probe result for grounding.)
        down_max = cfg.snap_down if dz >= 0.0 else max(cfg.snap_down, -dz)
        locD, nD, gobjD, _ = self._raycast_down_any(static_bvh, dynamic_map, pos, down_max)

        if dz < 0.0:
            if locD is not None and nD is not None and (pos.z + dz) <= locD.z:
                pos.z = locD.z
                self.vel.z = 0.0
                self.on_ground   = True
                self.on_walkable = (nD.dot(up) >= floor_cos)
                self.ground_norm = nD
                self.ground_obj  = gobjD
                self._coyote     = cfg.coyote_time
            else:
                pos.z += dz
        else:
            pos.z += dz

        # Snap/grounding using the same hit if within snap window
        was_grounded = self.on_ground
        if locD is not None and nD is not None and (abs(locD.z - pos.z) <= float(cfg.snap_down)) and self.vel.z <= 0.0:
            self.on_ground   = True
            self.on_walkable = (nD.dot(up) >= floor_cos)
            self.ground_norm = nD
            self.ground_obj  = gobjD
            pos.z            = locD.z
            self.vel.z       = 0.0
            self._coyote     = cfg.coyote_time
        else:
            self.on_ground   = False
            self.on_walkable = False
            self.ground_obj  = None
            if was_grounded:
                self._coyote = cfg.coyote_time

        # Write-back
        self.obj.location = pos
        if abs(rot.z - self.obj.rotation_euler.z) > 0.0:
            self.obj.rotation_euler = rot

    # ---- Jumping ------------------------------------------------------------

    def request_jump(self):
        self._jump_buf = self.cfg.jump_buffer

    def try_consume_jump(self):
        """
        Only allow jump if:
        • grounded on WALKABLE ground, or
        • within coyote time (granted from walkable support)
        """
        can_jump_from_ground = (self.on_ground and self.on_walkable)
        if self._jump_buf > 0.0 and (can_jump_from_ground or self._coyote > 0.0):
            self._jump_buf = 0.0
            self.vel.z     = self.cfg.jump_speed
            self.on_ground = False
            self.ground_obj = None
            return True
        return False
