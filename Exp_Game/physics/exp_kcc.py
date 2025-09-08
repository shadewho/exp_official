# Exp_Game/physics/exp_kcc.py
import math
import mathutils
from mathutils import Vector
from .exp_physics import (
    capsule_collision_resolve,
)

def clamp(v, lo, hi): return max(lo, min(hi, v))

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
    def __init__(self, obj, scene_cfg):
        self.obj = obj
        self.cfg = KCCConfig(scene_cfg)
        self.vel = Vector((0.0, 0.0, 0.0))
        self.on_ground = False
        self.ground_norm = Vector((0,0,1))
        self.ground_obj = None
        self._coyote = 0.0
        self._jump_buf = 0.0

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

        v = Vector((x, y, 0.0))
        if v.length > 1e-6:
            v.normalize()

        Rz = mathutils.Matrix.Rotation(camera_yaw, 4, 'Z')
        world = (Rz @ Vector((v.x, v.y, 0.0)))
        return world.xy, (run_key in keys_pressed)

    def _accelerate(self, cur_xy, wish_dir_xy, target_speed, accel, dt):
        wish = Vector((wish_dir_xy[0], wish_dir_xy[1]))
        desired = wish * target_speed
        t = clamp(accel * dt, 0.0, 1.0)  # critically-damped lerp
        return cur_xy.lerp(desired, t)

    def _raycast_down_any(self, static_bvh, dynamic_map, origin, max_dist):
        best = (None, None, None, 1e9)  # (loc, norm, obj, dist)
        if static_bvh:
            hit = static_bvh.ray_cast(origin + Vector((0,0,1.0)), Vector((0,0,-1)), max_dist+1.0)
            if hit[0] is not None:
                best = (hit[0], hit[1], None, (origin - hit[0]).length)
        if dynamic_map:
            for obj, (bvh_like, _) in dynamic_map.items():
                hit = bvh_like.ray_cast(origin + Vector((0,0,1.0)), Vector((0,0,-1)), max_dist+1.0)
                if hit[0] is not None:
                    d = (origin - hit[0]).length
                    if d < best[3]:
                        best = (hit[0], hit[1], obj, d)
        return best if best[0] is not None else (None, None, None, None)

    def _slope_ok(self, n):
        up = Vector((0,0,1))
        ang = math.degrees(up.angle(n))
        return ang <= self.cfg.slope_limit_deg

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
        # 1) Input
        wish_dir_xy, is_running = self._input_vector(keys_pressed, prefs, camera_yaw)
        target_speed = self.cfg.max_run if is_running else self.cfg.max_walk

        # 2) Timers (coyote & jump buffer)
        self._coyote = max(0.0, self._coyote - dt)
        self._jump_buf = max(0.0, self._jump_buf - dt)

        # 3) Horizontal acceleration
        cur_xy = Vector((self.vel.x, self.vel.y))
        accel = self.cfg.accel_ground if self.on_ground else self.cfg.accel_air
        new_xy = self._accelerate(cur_xy, wish_dir_xy, target_speed, accel, dt)
        self.vel.x, self.vel.y = new_xy.x, new_xy.y

        # 4) Gravity
        if not self.on_ground:
            self.vel.z += self.cfg.gravity * dt
        else:
            self.vel.z = min(self.vel.z, 0.0)

        # 5) Moving platform carry (v + ω×r), computed as **velocity** (m/s)
        carry = Vector((0,0,0))
        if self.on_ground and self.ground_obj:
            v_lin = Vector((0,0,0))
            if platform_linear_velocity_map and self.ground_obj in platform_linear_velocity_map:
                v_lin = platform_linear_velocity_map[self.ground_obj]

            v_rot = Vector((0,0,0))
            if platform_ang_velocity_map and self.ground_obj in platform_ang_velocity_map:
                omega = platform_ang_velocity_map[self.ground_obj]  # rad/s, world
                # r = world-space vector from platform origin to character
                r = (self.obj.matrix_world.translation -
                     self.ground_obj.matrix_world.translation)
                v_rot = omega.cross(r)

                # Also yaw-rotate the character with the platform (keep upright)
                # This gives "physically correct" spinning with the platform.
                yaw_delta = omega.z * dt
                eul = self.obj.rotation_euler
                eul.z += yaw_delta
                self.obj.rotation_euler = eul

            carry = v_lin + v_rot

        # 6) Integrate
        move = Vector((self.vel.x + carry.x, self.vel.y + carry.y, self.vel.z)) * dt
        self.obj.location += move

        # 7) Resolve collisions
        if static_bvh:
            capsule_collision_resolve(
                static_bvh, self.obj,
                radius=self.cfg.radius,
                heights=(0.2, self.cfg.height*0.5, self.cfg.height-0.1),
                max_iterations=2,
                push_strength=0.5
            )
        if dynamic_map:
            for bvh_like, _ in dynamic_map.values():
                capsule_collision_resolve(
                    bvh_like, self.obj,
                    radius=self.cfg.radius,
                    heights=(0.2, self.cfg.height*0.5, self.cfg.height-0.1),
                    max_iterations=1,
                    push_strength=0.5
                )

        # 8) Ground snap
        loc, norm, gobj, dist = self._raycast_down_any(static_bvh, dynamic_map, self.obj.location, self.cfg.snap_down)
        if loc and norm and self._slope_ok(norm) and self.vel.z <= 0.0:
            self.obj.location.z = loc.z
            self.vel.z = 0.0
            self.on_ground = True
            self.ground_norm = norm
            self.ground_obj = gobj
            self._coyote = self.cfg.coyote_time
        else:
            was_grounded = self.on_ground
            self.on_ground = False
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
