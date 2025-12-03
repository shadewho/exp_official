# Exploratory/Exp_Game/physics/exp_kcc.py
"""
Kinematic Character Controller - Full Physics Offload Architecture

Worker does the ENTIRE physics step:
  1. Input → Velocity acceleration
  2. Gravity
  3. Jump
  4. Horizontal collision (3D DDA on cached grid)
  5. Step-up
  6. Wall slide
  7. Ceiling check
  8. Ground detection

Main thread is THIN:
  - Apply previous frame's worker result
  - Handle dynamic movers (frame-perfect)
  - Snapshot state + input
  - Submit KCC_PHYSICS_STEP job
  - Write position to Blender
"""
import math
import mathutils
from mathutils import Vector
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

# ---- Small helpers ---------------------------------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

_UP = Vector((0.0, 0.0, 1.0))

# ---- GPU Visualization (Global Handler) ------------------------------------

_kcc_draw_handler = None
_kcc_vis_data = None  # Shared visualization data

def _draw_kcc_visual():
    """GPU draw callback for KCC visualization."""
    global _kcc_vis_data

    if _kcc_vis_data is None:
        return

    scene = bpy.context.scene
    if not getattr(scene, 'dev_debug_kcc_visual', False):
        return

    # Get toggle states
    show_capsule = getattr(scene, 'dev_debug_kcc_visual_capsule', True)
    show_normals = getattr(scene, 'dev_debug_kcc_visual_normals', True)
    show_ground = getattr(scene, 'dev_debug_kcc_visual_ground', True)
    show_movement = getattr(scene, 'dev_debug_kcc_visual_movement', True)

    # Enable depth test for 3D objects
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.blend_set('ALPHA')

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    # ─────────────────────────────────────────────────────────────────────
    # CAPSULE (Two spheres: feet + head)
    # PERFORMANCE: Single layer, fewer segments, batched drawing
    # ─────────────────────────────────────────────────────────────────────
    if show_capsule and 'capsule_spheres' in _kcc_vis_data:
        capsule_color = _kcc_vis_data.get('capsule_color', (0.0, 1.0, 0.0, 0.3))
        spheres = _kcc_vis_data['capsule_spheres']

        segments = 16  # REDUCED to 16 for max performance (still smooth)

        # Batch all circles together for fewer shader binds
        all_verts = []

        for sphere_center, sphere_radius in spheres:
            # XY circle
            for i in range(segments + 1):
                angle = (i / segments) * 2.0 * math.pi
                x = sphere_center[0] + sphere_radius * math.cos(angle)
                y = sphere_center[1] + sphere_radius * math.sin(angle)
                z = sphere_center[2]
                all_verts.append((x, y, z))

            # XZ circle
            for i in range(segments + 1):
                angle = (i / segments) * 2.0 * math.pi
                x = sphere_center[0] + sphere_radius * math.cos(angle)
                y = sphere_center[1]
                z = sphere_center[2] + sphere_radius * math.sin(angle)
                all_verts.append((x, y, z))

            # YZ circle
            for i in range(segments + 1):
                angle = (i / segments) * 2.0 * math.pi
                x = sphere_center[0]
                y = sphere_center[1] + sphere_radius * math.cos(angle)
                z = sphere_center[2] + sphere_radius * math.sin(angle)
                all_verts.append((x, y, z))

        # Single batched draw call for all circles
        if all_verts:
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": all_verts})
            shader.bind()
            shader.uniform_float("color", capsule_color)
            batch.draw(shader)

    # ─────────────────────────────────────────────────────────────────────
    # HIT NORMALS (Cyan arrows)
    # PERFORMANCE: Single line per normal, batched drawing
    # ─────────────────────────────────────────────────────────────────────
    if show_normals and 'hit_normals' in _kcc_vis_data:
        normal_color = (0.0, 1.0, 1.0, 0.9)  # Cyan, more opaque
        normals = _kcc_vis_data['hit_normals']

        arrow_length = 0.4
        all_lines = []

        for origin, normal in normals:
            end = (
                origin[0] + normal[0] * arrow_length,
                origin[1] + normal[1] * arrow_length,
                origin[2] + normal[2] * arrow_length
            )
            all_lines.extend([origin, end])

        # Single batched draw call for all normals
        if all_lines:
            batch = batch_for_shader(shader, 'LINES', {"pos": all_lines})
            shader.bind()
            shader.uniform_float("color", normal_color)
            batch.draw(shader)

    # ─────────────────────────────────────────────────────────────────────
    # GROUND RAY (Magenta = hit, Purple = miss)
    # PERFORMANCE: Simple line, single draw call
    # ─────────────────────────────────────────────────────────────────────
    if show_ground and 'ground_ray' in _kcc_vis_data:
        ray_data = _kcc_vis_data['ground_ray']
        origin = ray_data['origin']
        end = ray_data['end']
        hit = ray_data['hit']

        ray_color = (1.0, 0.0, 1.0, 0.9) if hit else (0.5, 0.0, 0.5, 0.7)  # Magenta/Purple

        # Single line for ray
        verts = [origin, end]
        batch = batch_for_shader(shader, 'LINES', {"pos": verts})
        shader.bind()
        shader.uniform_float("color", ray_color)
        batch.draw(shader)

        # Simple circle at hit point if hit
        if hit and 'hit_point' in ray_data:
            hit_point = ray_data['hit_point']
            segments = 12
            hit_radius = 0.08
            verts = []

            for i in range(segments + 1):
                angle = (i / segments) * 2.0 * math.pi
                x = hit_point[0] + hit_radius * math.cos(angle)
                y = hit_point[1] + hit_radius * math.sin(angle)
                z = hit_point[2]
                verts.append((x, y, z))

            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": verts})
            shader.bind()
            shader.uniform_float("color", ray_color)
            batch.draw(shader)

    # ─────────────────────────────────────────────────────────────────────
    # MOVEMENT VECTORS (Green = intended, Red = actual)
    # PERFORMANCE: Simple lines, separate batches by color
    # ─────────────────────────────────────────────────────────────────────
    if show_movement and 'movement_vectors' in _kcc_vis_data:
        vectors = _kcc_vis_data['movement_vectors']
        origin = vectors.get('origin')

        if origin and 'intended' in vectors:
            intended = vectors['intended']
            intended_color = (0.0, 1.0, 0.0, 0.9)  # Green
            verts = [origin, intended]
            batch = batch_for_shader(shader, 'LINES', {"pos": verts})
            shader.bind()
            shader.uniform_float("color", intended_color)
            batch.draw(shader)

        if origin and 'actual' in vectors:
            actual = vectors['actual']
            actual_color = (1.0, 0.0, 0.0, 0.9)  # Red
            verts = [origin, actual]
            batch = batch_for_shader(shader, 'LINES', {"pos": verts})
            shader.bind()
            shader.uniform_float("color", actual_color)
            batch.draw(shader)

    # Reset GPU state
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')

def enable_kcc_visualizer():
    """Register the KCC visualization draw handler."""
    global _kcc_draw_handler

    if _kcc_draw_handler is None:
        _kcc_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_kcc_visual, (), 'WINDOW', 'POST_VIEW'
        )
        _tag_all_view3d_for_redraw()

def disable_kcc_visualizer():
    """Unregister the KCC visualization draw handler."""
    global _kcc_draw_handler, _kcc_vis_data

    if _kcc_draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_kcc_draw_handler, 'WINDOW')
        except Exception:
            pass
        _kcc_draw_handler = None

    _kcc_vis_data = None
    _tag_all_view3d_for_redraw()

def _tag_all_view3d_for_redraw():
    """Tag all VIEW_3D areas for redraw."""
    wm = getattr(bpy.context, "window_manager", None)
    if not wm:
        return
    for win in wm.windows:
        scr = win.screen
        if not scr:
            continue
        for area in scr.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

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
    Full Physics Offload KCC:
    - Worker computes entire physics step (no prediction needed)
    - Main thread only applies results and handles dynamic movers
    - 1-frame latency is acceptable (33ms at 30Hz)
    """

    def __init__(self, obj, scene_cfg):
        self.obj = obj
        self.cfg = KCCConfig(scene_cfg)

        # Physics state
        self.pos          = obj.location.copy()
        self.vel          = Vector((0.0, 0.0, 0.0))
        self.on_ground    = False
        self.on_walkable  = True
        self.ground_norm  = _UP.copy()
        self.ground_obj   = None
        self._coyote      = 0.0
        self._jump_buf    = 0.0

        # Cached constants
        self._up = _UP
        self._floor_cos = math.cos(math.radians(self.cfg.slope_limit_deg))

        # Worker result caching (1-frame latency pattern)
        self._cached_physics_result = None
        self._last_physics_job_id = None

        # Visualization cache
        self._vis_colliding = False
        self._vis_was_stuck = False
        self._vis_hit_normals = []
        self._vis_intended_move = (0.0, 0.0, 0.0)
        self._vis_actual_move = (0.0, 0.0, 0.0)

        # Frame counter for debug output
        self._physics_frame = 0

    # --------------------
    # Input calculation (main thread - always immediate)
    # --------------------

    def _input_vector(self, keys_pressed, prefs, camera_yaw):
        """Calculate wish direction from input keys and camera yaw."""
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

        # Normalize
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
            wish_x = world3.x * inv_xy
            wish_y = world3.y * inv_xy
        else:
            wish_x = wish_y = 0.0

        return (wish_x, wish_y), (run_key in keys_pressed)

    # --------------------
    # Dynamic mover handling (main thread - frame-perfect)
    # --------------------

    def _handle_dynamic_movers(self, dynamic_map, v_lin_map, v_ang_map, pos, dt):
        """
        Handle dynamic platform carry on main thread (frame-perfect accuracy).
        Returns (velocity_add, push_out, rotation_delta_z).
        """
        if not dynamic_map:
            return Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0)), 0.0

        r = float(self.cfg.radius)
        h = float(self.cfg.height)
        up = self._up

        # Capsule mid sample
        mid_z = max(r, min(h - r, h * 0.5))
        sample_mid = pos + Vector((0.0, 0.0, mid_z))

        # Bounding sphere for quick gate
        cap_center = pos + Vector((0.0, 0.0, h * 0.5))
        cap_bs_rad = (h * 0.5 + r)

        # Find nearby movers
        candidates = []
        for obj, (_lbvh, approx_rad) in dynamic_map.items():
            if obj is None or _lbvh is None:
                continue
            if self.ground_obj is not None and obj == self.ground_obj:
                continue  # Skip ground object (handled separately)
            try:
                c = obj.matrix_world.translation
            except Exception:
                continue
            if (c - cap_center).length <= (float(approx_rad) + cap_bs_rad + 0.4):
                candidates.append((obj, (c - cap_center).length_squared))

        if not candidates:
            return Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0)), 0.0

        candidates.sort(key=lambda t: t[1])
        movers = [t[0] for t in candidates[:3]]

        vel_add = Vector((0.0, 0.0, 0.0))
        push_out = Vector((0.0, 0.0, 0.0))
        rot_delta_z = 0.0
        carry_cap = max(0.001, 0.5 * r)

        for obj in movers:
            lbvh, _ = dynamic_map.get(obj, (None, None))
            if lbvh is None:
                continue

            try:
                hit_co, hit_n, _idx, dist = lbvh.find_nearest(sample_mid, distance=r + 0.20)
            except Exception:
                continue
            if hit_co is None or hit_n is None:
                continue

            n = hit_n.normalized()
            if (sample_mid - hit_co).dot(n) < 0.0:
                n = -n

            # Push-out if overlapping
            if dist < r:
                push_out += n * min((r - dist), 0.20)

            # Linear carry
            v_lin = v_lin_map.get(obj, Vector((0.0, 0.0, 0.0))) if v_lin_map else Vector((0.0, 0.0, 0.0))
            if v_lin.length_squared <= 1.0e-12:
                continue

            n_up = n.dot(up)
            v_xy = Vector((v_lin.x, v_lin.y, 0.0))

            if n_up >= 0.6:
                v_add = Vector((v_xy.x, v_xy.y, 0.0))
            elif n_up >= 0.0:
                vn = v_xy.dot(n)
                if vn > 0.0:
                    v_xy = v_xy - n * vn
                v_add = Vector((v_xy.x, v_xy.y, 0.0))
            else:
                v_add = Vector((0.0, 0.0, 0.0))

            v_add.x = clamp(v_add.x, -carry_cap, carry_cap)
            v_add.y = clamp(v_add.y, -carry_cap, carry_cap)
            v_add.z = 0.0

            vel_add += v_add

        return vel_add, push_out, rot_delta_z

    def _get_platform_carry(self, platform_linear_velocity_map, platform_ang_velocity_map, pos, rot, dt):
        """Get velocity carry from ground platform (main thread, frame-perfect)."""
        carry = Vector((0.0, 0.0, 0.0))
        rot_delta_z = 0.0

        if self.on_ground and self.ground_obj:
            v_lin = platform_linear_velocity_map.get(self.ground_obj, Vector((0.0, 0.0, 0.0))) if platform_linear_velocity_map else Vector((0.0, 0.0, 0.0))
            v_rot = Vector((0.0, 0.0, 0.0))

            if platform_ang_velocity_map and self.ground_obj in platform_ang_velocity_map:
                omega = platform_ang_velocity_map[self.ground_obj]
                r_vec = (pos - self.ground_obj.matrix_world.translation)
                v_rot = omega.cross(r_vec)
                rot_delta_z = omega.z * dt

            carry = v_lin + v_rot

        return carry, rot_delta_z

    # --------------------
    # Worker result application
    # --------------------

    def _apply_physics_result(self, result, context=None, dynamic_map=None):
        """
        Apply physics result from worker to character state.
        Also checks collision against dynamic meshes (not in static grid).
        """
        if result is None:
            return

        # Extract result data
        new_pos = result.get("pos")
        new_vel = result.get("vel")
        on_ground = result.get("on_ground", False)
        on_walkable = result.get("on_walkable", True)
        ground_normal = result.get("ground_normal", (0.0, 0.0, 1.0))
        coyote = result.get("coyote_remaining", 0.0)
        jump_consumed = result.get("jump_consumed", False)

        # Cache visualization data from debug info
        debug = result.get("debug", {})
        self._vis_colliding = debug.get('h_blocked', False)
        self._vis_was_stuck = debug.get('was_stuck', False)

        # Apply state
        if new_pos:
            self.pos = Vector(new_pos)
        if new_vel:
            self.vel = Vector(new_vel)

        self.on_ground = on_ground
        self.on_walkable = on_walkable
        self.ground_norm = Vector(ground_normal)
        self._coyote = coyote

        if jump_consumed:
            self._jump_buf = 0.0

        # ─────────────────────────────────────────────────────────────────────
        # DYNAMIC MESH COLLISION CHECK (main thread, after worker result)
        # Worker only checks static grid - dynamic meshes need separate check
        # ─────────────────────────────────────────────────────────────────────
        if dynamic_map:
            self._check_dynamic_collision(dynamic_map)

        # ─────────────────────────────────────────────────────────────────────
        # PLATFORM CARRY (after dynamic collision sets ground_obj)
        # This must happen AFTER physics to avoid stutter - we ADD platform
        # motion to the physics result rather than baking it into velocity
        # ─────────────────────────────────────────────────────────────────────
        platform_lin_map = getattr(self, '_pending_platform_lin_map', None)
        platform_ang_map = getattr(self, '_pending_platform_ang_map', None)
        pending_dt = getattr(self, '_pending_dt', 1.0/30.0)

        if self.on_ground and self.ground_obj and platform_lin_map:
            v_lin = platform_lin_map.get(self.ground_obj)
            if v_lin and v_lin.length_squared > 1e-12:
                # Add platform velocity directly to position (not velocity)
                # This keeps the character fixed relative to the platform
                self.pos.x += v_lin.x * pending_dt
                self.pos.y += v_lin.y * pending_dt
                self.pos.z += v_lin.z * pending_dt

            # Handle angular velocity (rotation around platform center)
            if platform_ang_map:
                omega = platform_ang_map.get(self.ground_obj)
                if omega and abs(omega.z) > 1e-9:
                    # Rotate position around platform center
                    import math
                    platform_center = self.ground_obj.matrix_world.translation
                    rel_pos = self.pos - platform_center
                    angle = omega.z * pending_dt
                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)
                    new_x = rel_pos.x * cos_a - rel_pos.y * sin_a
                    new_y = rel_pos.x * sin_a + rel_pos.y * cos_a
                    self.pos.x = platform_center.x + new_x
                    self.pos.y = platform_center.y + new_y

        # Debug output (PERFORMANCE: Check flag BEFORE any string formatting)
        if context and getattr(context.scene, 'dev_debug_kcc_offload', False):
            from ..developer.dev_debug_gate import should_print_debug
            if should_print_debug("kcc_offload"):
                # Only extract debug data if we're actually printing
                debug = result.get("debug", {})
                # Use efficient string formatting (single f-string, no concatenation)
                print(f"[KCC] APPLY pos=({self.pos.x:.2f},{self.pos.y:.2f},{self.pos.z:.2f}) "
                      f"ground={on_ground} blocked={debug.get('h_blocked', False)} "
                      f"step={debug.get('did_step_up', False)} | "
                      f"{debug.get('calc_time_us', 0):.0f}us {debug.get('rays_cast', 0)}rays "
                      f"{debug.get('triangles_tested', 0)}tris")

    def _check_dynamic_collision(self, dynamic_map):
        """
        Check collision against dynamic meshes:
        1. Ground detection (raycast down) - allows standing on dynamic meshes
        2. Horizontal collision (push out) - prevents walking through
        Called after applying worker physics result.
        """
        import math
        if not dynamic_map:
            return

        # Clear hit normals for this frame
        self._vis_hit_normals = []

        r = float(self.cfg.radius)
        h = float(self.cfg.height)
        snap_down = float(self.cfg.snap_down)

        # Bounding sphere for quick gate
        cap_center = self.pos + Vector((0.0, 0.0, h * 0.5))
        cap_bs_rad = (h * 0.5 + r)

        # ─────────────────────────────────────────────────────────────────────
        # 1. GROUND DETECTION ON DYNAMIC MESHES
        # ─────────────────────────────────────────────────────────────────────
        # Raycast down from feet to find dynamic ground
        best_ground_z = None
        best_ground_n = None
        best_ground_obj = None

        ray_origin = self.pos + Vector((0.0, 0.0, 0.5))  # Start slightly above feet
        ray_max = snap_down + 0.5 + 0.1  # How far down to check

        for obj, (lbvh, approx_rad) in dynamic_map.items():
            if obj is None or lbvh is None:
                continue

            try:
                c = obj.matrix_world.translation
            except Exception:
                continue

            # Quick distance gate (generous for ground check)
            if (c - cap_center).length > (float(approx_rad) + cap_bs_rad + 2.0):
                continue

            # Raycast down using BVH
            try:
                hit_co, hit_n, _idx, dist = lbvh.ray_cast(ray_origin, Vector((0.0, 0.0, -1.0)), ray_max)
            except Exception:
                continue

            if hit_co is not None and hit_n is not None:
                hit_z = hit_co.z
                # Check if this is a valid ground (normal pointing up enough)
                if hit_n.z >= self._floor_cos:
                    if best_ground_z is None or hit_z > best_ground_z:
                        best_ground_z = hit_z
                        best_ground_n = hit_n.copy()
                        best_ground_obj = obj

        # Apply dynamic ground if found and within snap distance
        if best_ground_z is not None:
            ground_diff = best_ground_z - self.pos.z
            # If we're close enough to ground OR falling onto it
            if abs(ground_diff) <= snap_down or (self.vel.z <= 0 and ground_diff >= -0.1):
                self.pos.z = best_ground_z
                self.on_ground = True
                self.on_walkable = True
                self.ground_norm = best_ground_n
                self.ground_obj = best_ground_obj
                self.vel.z = max(0.0, self.vel.z)
                self._coyote = self.cfg.coyote_time

        # ─────────────────────────────────────────────────────────────────────
        # 2. HORIZONTAL COLLISION (push out from sides)
        # ─────────────────────────────────────────────────────────────────────
        sample_heights = [r, h * 0.5, h - r]

        total_push = Vector((0.0, 0.0, 0.0))
        push_count = 0

        for obj, (lbvh, approx_rad) in dynamic_map.items():
            if obj is None or lbvh is None:
                continue
            # Skip our current ground object for horizontal checks
            if best_ground_obj is not None and obj == best_ground_obj:
                continue

            try:
                c = obj.matrix_world.translation
            except Exception:
                continue

            # Quick distance gate
            if (c - cap_center).length > (float(approx_rad) + cap_bs_rad + 0.5):
                continue

            # Check collision at multiple heights
            for sample_z in sample_heights:
                sample_pt = self.pos + Vector((0.0, 0.0, sample_z))

                try:
                    hit_co, hit_n, _idx, dist = lbvh.find_nearest(sample_pt, distance=r + 0.15)
                except Exception:
                    continue

                if hit_co is None or hit_n is None:
                    continue

                # Push out if overlapping
                if dist < r:
                    n = hit_n.normalized()
                    # Make sure normal points away from mesh
                    if (sample_pt - hit_co).dot(n) < 0.0:
                        n = -n
                    push_dist = min((r - dist) + 0.02, 0.3)  # Max 0.3m push
                    total_push += n * push_dist
                    push_count += 1

                    # Cache hit normal for visualization
                    self._vis_hit_normals.append((
                        (hit_co.x, hit_co.y, hit_co.z),
                        (n.x, n.y, n.z)
                    ))

        # Apply accumulated horizontal push
        if push_count > 0:
            # Average the push if multiple hits
            total_push /= push_count
            # Only apply horizontal component (Z push handled by ground detection)
            self.pos.x += total_push.x
            self.pos.y += total_push.y

            # Remove velocity component into the push
            push_xy = Vector((total_push.x, total_push.y, 0.0))
            if push_xy.length > 0.01:
                push_n = push_xy.normalized()
                vn = self.vel.x * push_n.x + self.vel.y * push_n.y
                if vn < 0.0:  # Moving into the obstacle
                    self.vel.x -= push_n.x * vn
                    self.vel.y -= push_n.y * vn

    # --------------------
    # Job building
    # --------------------

    def _build_physics_job(self, wish_dir, is_running, jump_requested, dt, context=None):
        """Build KCC_PHYSICS_STEP job data for worker."""
        cfg = self.cfg

        # Extract debug flags from scene (if context available)
        debug_flags = {}
        if context and hasattr(context, 'scene'):
            scene = context.scene
            debug_flags = {
                "step_up": getattr(scene, "dev_debug_physics_step_up", False),
                "ground": getattr(scene, "dev_debug_physics_ground", False),
                "capsule": getattr(scene, "dev_debug_physics_capsule", False),
                "enhanced": getattr(scene, "dev_debug_physics_enhanced", False),
            }

        return {
            # Current state
            "pos": (self.pos.x, self.pos.y, self.pos.z),
            "vel": (self.vel.x, self.vel.y, self.vel.z),
            "on_ground": self.on_ground,
            "on_walkable": self.on_walkable,
            "ground_normal": (self.ground_norm.x, self.ground_norm.y, self.ground_norm.z),

            # Input this frame
            "wish_dir": wish_dir,  # (dx, dy) normalized
            "is_running": is_running,
            "jump_requested": jump_requested,

            # Physics config
            "config": {
                "radius": float(cfg.radius),
                "height": float(cfg.height),
                "gravity": float(cfg.gravity),
                "max_walk": float(cfg.max_walk),
                "max_run": float(cfg.max_run),
                "accel_ground": float(cfg.accel_ground),
                "accel_air": float(cfg.accel_air),
                "step_height": float(cfg.step_height),
                "snap_down": float(cfg.snap_down),
                "slope_limit_deg": float(cfg.slope_limit_deg),
                "jump_speed": float(cfg.jump_speed),
                "coyote_time": float(cfg.coyote_time),
            },

            # Timing
            "dt": dt,

            # Timers
            "coyote_remaining": self._coyote,
            "jump_buffer_remaining": self._jump_buf,

            # Debug flags
            "debug_flags": debug_flags,
        }

    # ---- Main step ----------------------------------------------------------

    def step(
        self,
        dt: float,
        prefs,
        keys_pressed,
        camera_yaw: float,
        static_bvh,  # Not used - worker has cached grid
        dynamic_map,
        platform_linear_velocity_map=None,
        platform_ang_velocity_map=None,
        engine=None,
        context=None,
        physics_frame=0,
    ):
        """
        Same-Frame Physics Offload step:
        1. Handle dynamic movers (frame-perfect, main thread)
        2. SNAPSHOT current state + input
        3. SUBMIT job to worker
        4. POLL for result (same-frame, with timeout)
        5. APPLY result immediately
        6. Write position to Blender

        This eliminates the 1-frame latency of the previous architecture.
        Worker computation is ~100-200µs, well within frame budget.
        """
        import time
        rot = self.obj.rotation_euler.copy()

        # Sync position from Blender (in case of external changes)
        self.pos = self.obj.location.copy()

        # Store velocity maps for post-physics platform carry
        self._pending_platform_lin_map = platform_linear_velocity_map
        self._pending_platform_ang_map = platform_ang_velocity_map
        self._pending_dt = dt

        # ─────────────────────────────────────────────────────────────────────
        # 1. Handle dynamic movers on main thread (push-out only, carry after physics)
        # ─────────────────────────────────────────────────────────────────────
        # Note: Platform carry is now applied AFTER physics result to prevent stutter
        if dynamic_map:
            # Push-out from nearby dynamic movers (no velocity carry here)
            v_extra, p_out, _ = self._handle_dynamic_movers(
                dynamic_map,
                None,  # Don't apply velocity here - do it after physics
                None,
                self.pos,
                dt
            )
            if p_out.length_squared > 0.0:
                self.pos += p_out

        # ─────────────────────────────────────────────────────────────────────
        # 2. SNAPSHOT current state + input
        # ─────────────────────────────────────────────────────────────────────
        # Increment frame counter
        self._physics_frame += 1

        # Frame number output (separate from KCC logs)
        if context and getattr(context.scene, 'dev_debug_frame_numbers', False):
            from ..developer.dev_debug_gate import should_print_debug
            if should_print_debug("frame_numbers"):
                from ..props_and_utils.exp_time import get_game_time
                game_time = get_game_time()
                print(f"[FRAME {self._physics_frame:04d}] t={game_time:.3f}s")

        wish_dir, is_running = self._input_vector(keys_pressed, prefs, camera_yaw)
        jump_requested = (self._jump_buf > 0.0)

        # Cache intended movement for visualization
        target_speed = self.cfg.max_run if is_running else self.cfg.max_walk
        self._vis_intended_move = (wish_dir[0] * target_speed * dt, wish_dir[1] * target_speed * dt, 0.0)

        # Cache position before physics for actual movement calculation
        pos_before = self.pos.copy()

        # Decrement timers
        self._jump_buf = max(0.0, self._jump_buf - dt)

        # ─────────────────────────────────────────────────────────────────────
        # 3. SUBMIT job and 4. POLL for same-frame result
        # ─────────────────────────────────────────────────────────────────────
        if engine:
            job_data = self._build_physics_job(wish_dir, is_running, jump_requested, dt, context)
            job_id = engine.submit_job("KCC_PHYSICS_STEP", job_data)
            self._last_physics_job_id = job_id

            # Same-frame polling: wait for our result
            # Worker computes in ~100-200µs typically, so this should succeed quickly
            # Using adaptive polling: busy-poll first, then sleep if needed
            poll_start = time.perf_counter()
            poll_timeout = 0.003  # 3ms max wait (plenty for ~200µs worker)
            result_found = False
            poll_count = 0

            while True:
                elapsed = time.perf_counter() - poll_start
                if elapsed >= poll_timeout:
                    break

                results = engine.poll_results(max_results=10)
                for result in results:
                    if result.job_id == job_id and result.job_type == "KCC_PHYSICS_STEP":
                        # Found our result! Apply immediately
                        if result.success:
                            self._apply_physics_result(result.result, context, dynamic_map)
                        result_found = True
                        break
                    else:
                        # Cache other results for their handlers
                        self._cache_other_result(result)
                if result_found:
                    break

                poll_count += 1

                # Adaptive sleep: busy-poll first 3 times, then add tiny sleeps
                # This minimizes latency while avoiding CPU spin
                if poll_count >= 3:
                    time.sleep(0.00005)  # 50µs (smaller than before)

            # Debug output (PERFORMANCE: Only compute timing if debug enabled)
            if context and getattr(context.scene, 'dev_debug_kcc_offload', False):
                from ..developer.dev_debug_gate import should_print_debug
                if should_print_debug("kcc_offload"):
                    poll_time_us = (time.perf_counter() - poll_start) * 1_000_000
                    if result_found:
                        print(f"[KCC] SAME-FRAME pos=({self.pos.x:.2f},{self.pos.y:.2f},{self.pos.z:.2f}) "
                              f"poll={poll_time_us:.0f}us")
                    else:
                        print(f"[KCC] TIMEOUT pos=({self.pos.x:.2f},{self.pos.y:.2f},{self.pos.z:.2f}) "
                              f"poll={poll_time_us:.0f}us - using previous state")
        else:
            # NO ENGINE FALLBACK - Physics requires engine
            if context and getattr(context.scene, 'dev_debug_kcc_offload', False):
                from ..developer.dev_debug_gate import should_print_debug
                if should_print_debug("kcc_offload"):
                    print(f"[KCC] WARNING: No engine available - physics step skipped")

        # ─────────────────────────────────────────────────────────────────────
        # 5. Write position to Blender
        # ─────────────────────────────────────────────────────────────────────
        self.obj.location = self.pos
        if abs(rot.z - self.obj.rotation_euler.z) > 1e-9:
            self.obj.rotation_euler = rot

        # ─────────────────────────────────────────────────────────────────────
        # 6. Update visualization (if enabled)
        # ─────────────────────────────────────────────────────────────────────
        # Cache actual movement for visualization
        self._vis_actual_move = (self.pos.x - pos_before.x, self.pos.y - pos_before.y, self.pos.z - pos_before.z)

        # Enable visualizer if toggle is on
        if context and getattr(context.scene, 'dev_debug_kcc_visual', False):
            if _kcc_draw_handler is None:
                enable_kcc_visualizer()
            self.update_visualization_data(context)
        elif _kcc_draw_handler is not None:
            disable_kcc_visualizer()

    def _cache_other_result(self, result):
        """Cache non-KCC results for processing by other handlers."""
        # Store in a list for the game loop to process
        if not hasattr(self, '_other_results'):
            self._other_results = []
        self._other_results.append(result)

    def get_cached_other_results(self):
        """Get and clear cached non-KCC results."""
        results = getattr(self, '_other_results', [])
        self._other_results = []
        return results

    # ---- Jumping ------------------------------------------------------------

    def request_jump(self):
        """Buffer a jump request (allows jump buffering before landing)."""
        self._jump_buf = self.cfg.jump_buffer

    def try_consume_jump(self):
        """
        Try to execute a buffered jump.
        Note: With full offload, actual jump execution happens in worker.
        This method now just ensures buffer is set for worker to check.
        """
        # With full offload, jump is consumed by worker based on _jump_buf
        # This method kept for compatibility with existing jump key handling
        return self._jump_buf > 0.0

    # ---- Engine result caching ----------------------------------------------

    def cache_physics_result(self, result):
        """
        Cache physics result from engine worker (for next frame's application).
        Called by game loop when KCC_PHYSICS_STEP job completes.

        Args:
            result: Dictionary with keys:
                - pos: (x, y, z)
                - vel: (vx, vy, vz)
                - on_ground: bool
                - on_walkable: bool
                - ground_normal: (nx, ny, nz)
                - coyote_remaining: float
                - jump_consumed: bool
                - debug: {...}
        """
        self._cached_physics_result = result

    # ---- Deprecated methods (kept for compatibility during transition) ------

    def cache_input_result(self, wish_dir_xy, is_running):
        """DEPRECATED: Use cache_physics_result instead."""
        pass

    def cache_slope_platform_result(self, delta_z, slide_xy, is_sliding, carry, rot_delta_z):
        """DEPRECATED: Use cache_physics_result instead."""
        pass

    def cache_raycast_result(self, hit, hit_location, hit_normal, hit_distance):
        """DEPRECATED: Use cache_physics_result instead."""
        pass

    def cache_forward_sweep_result(self, result_dict):
        """DEPRECATED: Use cache_physics_result instead."""
        pass

    # ---- Debug visualization cleanup ----------------------------------------

    def cleanup_debug_handlers(self):
        """
        Clean up GPU debug draw handlers for KCC visualization.
        Called when modal operator exits.
        """
        disable_kcc_visualizer()

    def update_visualization_data(self, context):
        """
        Update visualization data for GPU drawing.
        Called every frame when dev_debug_kcc_visual is enabled.

        PERFORMANCE: Only updates if position changed significantly.
        """
        global _kcc_vis_data

        if not getattr(context.scene, 'dev_debug_kcc_visual', False):
            return

        # PERFORMANCE: Skip update if position hasn't changed much
        if hasattr(self, '_last_vis_pos'):
            pos_delta = (self.pos - self._last_vis_pos).length
            if pos_delta < 0.01:  # Less than 1cm movement, skip update
                return
        self._last_vis_pos = self.pos.copy()

        # Determine capsule color based on state (increased opacity)
        if hasattr(self, '_vis_was_stuck') and self._vis_was_stuck:
            capsule_color = (1.0, 0.0, 0.0, 0.8)  # Red = stuck (depenetrating)
        elif hasattr(self, '_vis_colliding') and self._vis_colliding:
            capsule_color = (1.0, 1.0, 0.0, 0.8)  # Yellow = colliding
        elif self.on_ground:
            capsule_color = (0.0, 1.0, 0.0, 0.8)  # Green = grounded
        else:
            capsule_color = (0.0, 0.5, 1.0, 0.8)  # Blue = airborne

        # Build visualization data
        vis_data = {
            'capsule_color': capsule_color,
            'capsule_spheres': [],
            'hit_normals': [],
            'ground_ray': None,
            'movement_vectors': {}
        }

        # Capsule spheres (feet + head)
        r = float(self.cfg.radius)
        h = float(self.cfg.height)

        feet_center = (self.pos.x, self.pos.y, self.pos.z + r)
        head_center = (self.pos.x, self.pos.y, self.pos.z + h - r)

        vis_data['capsule_spheres'] = [
            (feet_center, r),
            (head_center, r)
        ]

        # Hit normals (from cached collision data)
        if hasattr(self, '_vis_hit_normals') and self._vis_hit_normals:
            vis_data['hit_normals'] = self._vis_hit_normals

        # Ground ray
        ray_origin = (self.pos.x, self.pos.y, self.pos.z + 0.5)
        ray_end = (self.pos.x, self.pos.y, self.pos.z - self.cfg.snap_down)

        vis_data['ground_ray'] = {
            'origin': ray_origin,
            'end': ray_end,
            'hit': self.on_ground
        }

        if self.on_ground:
            vis_data['ground_ray']['hit_point'] = (self.pos.x, self.pos.y, self.pos.z)

        # Movement vectors (from cached velocity data)
        if hasattr(self, '_vis_intended_move') and self._vis_intended_move:
            origin = (self.pos.x, self.pos.y, self.pos.z + h * 0.5)
            vis_data['movement_vectors']['origin'] = origin

            intended = self._vis_intended_move
            vis_data['movement_vectors']['intended'] = (
                origin[0] + intended[0],
                origin[1] + intended[1],
                origin[2] + intended[2]
            )

        if hasattr(self, '_vis_actual_move') and self._vis_actual_move:
            origin = (self.pos.x, self.pos.y, self.pos.z + h * 0.5)
            if 'origin' not in vis_data['movement_vectors']:
                vis_data['movement_vectors']['origin'] = origin

            actual = self._vis_actual_move
            vis_data['movement_vectors']['actual'] = (
                origin[0] + actual[0],
                origin[1] + actual[1],
                origin[2] + actual[2]
            )

        _kcc_vis_data = vis_data
        _tag_all_view3d_for_redraw()
