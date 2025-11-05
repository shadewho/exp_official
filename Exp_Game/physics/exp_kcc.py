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

    # --------------------
    # Input & helpers
    # --------------------
    def _accelerate(self, cur_xy: Vector, wish_dir_xy: Vector, target_speed: float, accel: float, dt: float) -> Vector:
        """
        XR-only horizontal acceleration blend (no local fallback).
        Queues kcc.accel_xy.v1 and uses only the last XR result.
        If XR hasn't replied yet this step, we keep cur_xy unchanged.
        """
        try:
            from ..xr_systems.xr_ports.kcc import queue_accel_xy
            queue_accel_xy(
                self,
                float(cur_xy.x), float(cur_xy.y),
                float(wish_dir_xy.x), float(wish_dir_xy.y),
                float(target_speed), float(accel), float(dt)
            )
        except Exception:
            # No fallback by design.
            pass

        xy = getattr(self, "_xr_accel_xy", None)
        if isinstance(xy, (tuple, list)) and len(xy) == 2:
            return Vector((float(xy[0]), float(xy[1])))

        # No XR answer yet → keep current velocity (no local compute).
        return Vector((cur_xy.x, cur_xy.y))


    def _input_vector(self, keys_pressed, prefs, camera_yaw):
        """
        XR-driven input:
          • Enqueue XR job to compute wish XY (intent + yaw + steep-slope clamp).
          • Return ONLY the last XR-provided XY (no local compute / no fallback).
            If XR hasn't replied yet, returns (0,0).
        """
        # Intent from keys (edge cases: both pressed => cancel on that axis)
        dx = 0.0; dy = 0.0
        if prefs.key_forward  in keys_pressed: dy += 1.0
        if prefs.key_backward in keys_pressed: dy -= 1.0
        if prefs.key_right    in keys_pressed: dx += 1.0
        if prefs.key_left     in keys_pressed: dx -= 1.0

        # Enqueue XR compute for NEXT frame (non-blocking)
        try:
            from ..xr_systems.xr_ports.kcc import queue_move_xy
            g = self.ground_norm if self.ground_norm is not None else self._up
            queue_move_xy(
                self,
                float(dx), float(dy), float(camera_yaw),
                bool(self.on_ground),
                (float(g.x), float(g.y), float(g.z)),
                float(self._floor_cos),
            )
        except Exception:
            # We DO NOT compute locally if XR errors; we simply have no update.
            pass

        # Use ONLY the XR result (no legacy path). If not yet available, zero.
        try:
            from mathutils import Vector
        except Exception:
            # Should never happen inside Blender; defensive.
            class _V(tuple):
                @property
                def x(self): return self[0]
                @property
                def y(self): return self[1]
            Vector = lambda xy: _V((float(xy[0]), float(xy[1])))

        xy_tup = getattr(self, "_xr_wish_xy", None)
        if not (isinstance(xy_tup, (tuple, list)) and len(xy_tup) == 2):
            xy_vec = Vector((0.0, 0.0))
        else:
            xy_vec = Vector((float(xy_tup[0]), float(xy_tup[1])))
            # Keep normalized-ish
            l2 = xy_vec.x * xy_vec.x + xy_vec.y * xy_vec.y
            if l2 > 1.0 + 1.0e-6:
                try:
                    xy_vec.normalize()
                except Exception:
                    pass

        return xy_vec, (prefs.key_run in keys_pressed)

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

    def _forward_sweep_min3(self, static_bvh, dynamic_map, pos: Vector,
                            fwd_norm: Vector, step_len: float):
        """
        Minimal forward sweep with one cheap slide attempt:
        • Cast 3 rays: feet (z=r), mid (z=h*0.5), head (z=h-r)
        • Move up to (nearest_hit - r)
        • Project XY velocity off blocking normal
        • Slide only if approach angle to the wall normal is in [20°, 70°]
            (not near-parallel to the wall and not head-on), and then advance
            once along the tangent at half speed.
        Returns: (new_pos, hit_normal or None)
        """
        r = float(self.cfg.radius)
        h = float(self.cfg.height)
        floor_cos = self._floor_cos
        ray_len = step_len + r

        # ---- primary forward rays (feet, mid, head) ----
        best_d = None
        best_n = None
        for z in (r, clamp(h * 0.5, r, h - r), h - r):
            o = pos + Vector((0.0, 0.0, z))
            hit_loc, hit_n, _obj, d = self._raycast_any_norm(static_bvh, dynamic_map, o, fwd_norm, ray_len)
            if hit_loc is None:
                continue
            if best_d is None or d < best_d:
                best_d = d
                best_n = hit_n.normalized()

        # No block: full advance
        if best_d is None:
            return pos + fwd_norm * step_len, None

        # Clamp to contact
        allowed = max(0.0, best_d - r)
        moved_pos = pos
        if allowed > 1.0e-9:
            moved_pos = pos + fwd_norm * allowed

        # Remove normal component from XY velocity (retain tangent momentum)
        if best_n is not None:
            hvel = Vector((self.vel.x, self.vel.y, 0.0))
            vn = hvel.dot(best_n)
            if vn > 0.0:
                hvel -= best_n * vn
                self.vel.x, self.vel.y = hvel.x, hvel.y

        # ---- decide if we should slide at all ----
        remaining = max(0.0, step_len - allowed)
        if remaining <= (0.15 * r):
            return moved_pos, best_n

        # Approach angle (to wall normal) gate: slide only if 20°..70°
        #   θ = arccos(|fwd·n|) in degrees  (0° = head-on, 90° = parallel to wall)
        dot = abs(fwd_norm.dot(best_n))
        # clamp for numeric stability
        dot = max(0.0, min(1.0, float(dot)))
        from math import acos, degrees
        theta = degrees(acos(dot))
        if not (20.0 <= theta <= 85.0):
            # too head-on (<20°) or too parallel (>70°): don't slide
            return moved_pos, best_n

        # Tangent direction to the blocking normal
        slide_dir = fwd_norm - best_n * fwd_norm.dot(best_n)
        # On too-steep surfaces, slide only in XY
        if best_n.dot(_UP) < floor_cos:
            slide_dir = Vector((slide_dir.x, slide_dir.y, 0.0))
        if slide_dir.length <= 1.0e-12:
            return moved_pos, best_n
        slide_dir.normalize()

        # Slow down while sliding (half advance and halve horizontal velocity)
        remaining *= 0.65
        self.vel.x *= 0.65
        self.vel.y *= 0.65

        # One slide attempt using the same 3-ray scheme
        ray_len2 = remaining + r
        best_d2 = None
        for z in (r, clamp(h * 0.5, r, h - r), h - r):
            o2 = moved_pos + Vector((0.0, 0.0, z))
            h2, n2, _o, d2 = self._raycast_any_norm(static_bvh, dynamic_map, o2, slide_dir, ray_len2)
            if h2 is None:
                continue
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2

        if best_d2 is None:
            return moved_pos + slide_dir * remaining, best_n

        allow2 = max(0.0, best_d2 - r)
        if allow2 > 1.0e-9:
            moved_pos = moved_pos + slide_dir * allow2

        return moved_pos, best_n

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



    def _dynamic_contact_influence(
        self,
        dynamic_map,                   # {obj: (LocalBVH_like, approx_radius)}
        v_lin_map,                     # {obj: Vector} linear velocity (m/s)
        v_ang_map,                     # unused (kept for signature compatibility)
        pos: Vector,                   # current character base position
        skip_obj=None                  # don't double-count support obj
    ):
        """
        Ultra-simple, CPU-friendly influence:
          • One nearest-point query per candidate (mid sample only)
          • Tiny push-out if overlapping
          • Mild carry from mover linear velocity only (no angular), with rules:
              - Floor-like contact (n·up >= 0.6): carry XY freely; clamp vertical add to 0
              - Wall-like (0.0 <= n·up < 0.6): carry only XY tangent; no normal add
              - Ceiling-like (n·up < 0.0): no carry (prevents upward boosts)
          • Per-step carry clamp to avoid jitter/tunneling (≤ 0.5 * radius in any axis)

        Cost: ≤ 1 find_nearest per chosen mover (we still gate movers).
        """
        if not dynamic_map:
            return Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0))

        r = float(self.cfg.radius)
        h = float(self.cfg.height)
        up = self._up

        # Capsule mid sample
        mid_z = max(r, min(h - r, h * 0.5))
        sample_mid = pos + Vector((0.0, 0.0, mid_z))

        # Bounding sphere of capsule for quick center gating
        cap_center = pos + Vector((0.0, 0.0, h * 0.5))
        cap_bs_rad = (h * 0.5 + r)

        # Choose the few nearest centers (up to 3) using a quick gate
        candidates = []
        for obj, (_lbvh, approx_rad) in dynamic_map.items():
            if obj is None or _lbvh is None:
                continue
            if skip_obj is not None and obj == skip_obj:
                continue
            try:
                c = obj.matrix_world.translation
            except Exception:
                continue
            # quick sphere-sphere gate with a small slack
            if (c - cap_center).length <= (float(approx_rad) + cap_bs_rad + 0.4):
                candidates.append((obj, (c - cap_center).length_squared))
        if not candidates:
            return Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0))

        candidates.sort(key=lambda t: t[1])
        # Keep at most 3 movers for this step
        movers = [t[0] for t in candidates[:3]]

        vel_add = Vector((0.0, 0.0, 0.0))
        push_out = Vector((0.0, 0.0, 0.0))

        # Per-step clamp to avoid jitter & “teleport carry”
        carry_cap = max(0.001, 0.5 * r)  # meters per step

        for obj in movers:
            lbvh, _ = dynamic_map.get(obj, (None, None))
            if lbvh is None:
                continue

            # Single nearest query (mid sample)
            try:
                hit_co, hit_n, _idx, dist = lbvh.find_nearest(sample_mid, distance=r + 0.20)
            except Exception:
                continue
            if hit_co is None or hit_n is None:
                continue

            n = hit_n.normalized()
            # Orient normal from surface toward the sample
            if (sample_mid - hit_co).dot(n) < 0.0:
                n = -n

            # 1) Tiny push-out if overlapping
            if dist < r:
                push_out += n * min((r - dist), 0.20)

            # 2) Linear carry only (no angular): calm & predictable
            v_lin = v_lin_map.get(obj, Vector((0.0, 0.0, 0.0))) if v_lin_map else Vector((0.0, 0.0, 0.0))
            if v_lin.length_squared <= 1.0e-12:
                continue

            # Classify contact
            n_up = n.dot(up)
            v_xy = Vector((v_lin.x, v_lin.y, 0.0))

            if n_up >= 0.6:
                # Floor-like: carry XY; block vertical boosts
                v_add = Vector((v_xy.x, v_xy.y, 0.0))
            elif n_up >= 0.0:
                # Wall-like: only tangent XY carry (no “suck” into wall)
                # Remove normal component from XY
                vn = v_xy.dot(n)
                if vn > 0.0:
                    v_xy = v_xy - n * vn
                v_add = Vector((v_xy.x, v_xy.y, 0.0))
            else:
                # Ceiling-like: no carry (prevents launch)
                v_add = Vector((0.0, 0.0, 0.0))

            # Per-axis clamp to keep things stable at high mover speeds
            v_add.x = max(-carry_cap / max(1.0e-6, 1.0), min(carry_cap / max(1.0e-6, 1.0), v_add.x))
            v_add.y = max(-carry_cap / max(1.0e-6, 1.0), min(carry_cap / max(1.0e-6, 1.0), v_add.y))
            v_add.z = 0.0  # never add vertical (no boosts)

            vel_add += v_add

        return vel_add, push_out



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

        # 3.5) Minimal dynamic contact influence (any side) + tiny pushout
        if dynamic_map:
            v_extra, p_out = self._dynamic_contact_influence(
                dynamic_map,
                platform_linear_velocity_map,
                platform_ang_velocity_map,  # ignored internally
                pos,
                skip_obj=self.ground_obj
            )
            if p_out.length_squared > 0.0:
                pos += p_out  # positional correction before casts
            if v_extra.length_squared > 0.0:
                # Horizontal-only carry; no vertical boosts
                self.vel.x += v_extra.x
                self.vel.y += v_extra.y


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

        # 7) Horizontal move (minimal 3-ray sweep + step-up + slide)
        hvel = Vector((self.vel.x + carry.x, self.vel.y + carry.y, 0.0))

        # Clamp uphill component on too-steep when grounded — XR ONLY (no local compute)
        if self.on_ground and self.ground_norm is not None and not self.on_walkable:
            try:
                from ..xr_systems.xr_ports.kcc import queue_clamp_xy
                g = self.ground_norm
                queue_clamp_xy(self, float(hvel.x), float(hvel.y),
                               (float(g.x), float(g.y), float(g.z)),
                               float(floor_cos))
            except Exception:
                pass
            xy = getattr(self, "_xr_clamp_xy", None)
            if isinstance(xy, (tuple, list)) and len(xy) == 2:
                hvel = Vector((float(xy[0]), float(xy[1]), 0.0))

        dz = self.vel.z * dt
        hvel_len2 = hvel.x*hvel.x + hvel.y*hvel.y

        # Early-out if idle (still do vertical block/snap below)
        if self.on_ground and abs(self.vel.z) < 1.0e-6 and hvel_len2 < 1.0e-10:
            pass
        else:
            move_total = (math.sqrt(hvel_len2) * dt) if hvel_len2 > 0.0 else 0.0
            if move_total <= 1.0e-9:
                fwd_norm = Vector((0.0, 0.0, 0.0))
            else:
                fwd_norm = (hvel / max(1.0e-12, (move_total / dt))).normalized()

            # Cap sub-steps (lower than before) to reduce casts while avoiding tunneling
            r = float(cfg.radius)
            # Allow up to ~2 radii per sub-step; cap at 2 sub-steps
            max_step = max(r * 2.0, r)
            sub_steps = 1 if hvel_len2 <= (max_step / max(dt, 1e-12))**2 else 2
            step_len = move_total / float(sub_steps) if sub_steps > 0 else 0.0

            for _sub in range(sub_steps):
                if step_len <= 1.0e-9 or fwd_norm.length <= 1.0e-9:
                    break

                # (A) Low-only forward probe to gate step-up (cheap)
                low_origin = pos + Vector((0.0, 0.0, r))
                hL, nL, _oL, dL = self._raycast_any_norm(static_bvh, dynamic_map,
                                                         low_origin, fwd_norm, step_len + r)

                # (B) Step-up: raise→short forward→drop
                did_step = False
                if self.on_ground and cfg.step_height > 0.0 and (hL is not None):
                    did_step, pos = self._try_step_up(
                        static_bvh, dynamic_map, pos, fwd_norm, step_len,
                        nL.normalized(), dL
                    )
                    if did_step:
                        continue  # go next sub-step

                # (C) Minimal 3-ray forward sweep (feet/mid/head)
                pos, _block_n = self._forward_sweep_min3(static_bvh, dynamic_map, pos, fwd_norm, step_len)


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
