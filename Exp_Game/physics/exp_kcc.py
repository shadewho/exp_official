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
    """GPU draw callback for KCC visualization (optimized for single draw call)."""
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

    # Enable depth test and blending for 3D objects
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.blend_set('ALPHA')

    # IMPORTANT: Set line width for visibility (default is 1 pixel)
    line_width = getattr(scene, 'dev_debug_kcc_visual_line_width', 2.5)
    gpu.state.line_width_set(line_width)

    # PERFORMANCE: Use per-vertex color shader for single batched draw call
    shader = gpu.shader.from_builtin('FLAT_COLOR')

    # Collect ALL geometry and colors for single draw call
    all_verts = []
    all_colors = []

    # ─────────────────────────────────────────────────────────────────────
    # CAPSULE (Three spheres: feet + mid + head)
    # ─────────────────────────────────────────────────────────────────────
    if show_capsule and 'capsule_spheres' in _kcc_vis_data:
        capsule_color = _kcc_vis_data.get('capsule_color', (0.0, 1.0, 0.0, 0.5))
        spheres = _kcc_vis_data['capsule_spheres']

        segments = 12  # Optimized segment count

        for sphere_center, sphere_radius in spheres:
            # XY circle (horizontal around character)
            for i in range(segments):
                angle1 = (i / segments) * 2.0 * math.pi
                angle2 = ((i + 1) / segments) * 2.0 * math.pi
                x1 = sphere_center[0] + sphere_radius * math.cos(angle1)
                y1 = sphere_center[1] + sphere_radius * math.sin(angle1)
                z1 = sphere_center[2]
                x2 = sphere_center[0] + sphere_radius * math.cos(angle2)
                y2 = sphere_center[1] + sphere_radius * math.sin(angle2)
                z2 = sphere_center[2]
                all_verts.extend([(x1, y1, z1), (x2, y2, z2)])
                all_colors.extend([capsule_color, capsule_color])

            # XZ circle (front-facing)
            for i in range(segments):
                angle1 = (i / segments) * 2.0 * math.pi
                angle2 = ((i + 1) / segments) * 2.0 * math.pi
                x1 = sphere_center[0] + sphere_radius * math.cos(angle1)
                y1 = sphere_center[1]
                z1 = sphere_center[2] + sphere_radius * math.sin(angle1)
                x2 = sphere_center[0] + sphere_radius * math.cos(angle2)
                y2 = sphere_center[1]
                z2 = sphere_center[2] + sphere_radius * math.sin(angle2)
                all_verts.extend([(x1, y1, z1), (x2, y2, z2)])
                all_colors.extend([capsule_color, capsule_color])

        # Add vertical lines connecting the spheres (feet→mid→head)
        if len(spheres) >= 2:
            # Connect adjacent spheres
            for i in range(len(spheres) - 1):
                lower_center, lower_radius = spheres[i]
                upper_center, upper_radius = spheres[i + 1]

                for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
                    x1 = lower_center[0] + lower_radius * math.cos(angle)
                    y1 = lower_center[1] + lower_radius * math.sin(angle)
                    z1 = lower_center[2]
                    x2 = upper_center[0] + upper_radius * math.cos(angle)
                    y2 = upper_center[1] + upper_radius * math.sin(angle)
                    z2 = upper_center[2]
                    all_verts.extend([(x1, y1, z1), (x2, y2, z2)])
                    all_colors.extend([capsule_color, capsule_color])

    # ─────────────────────────────────────────────────────────────────────
    # HIT NORMALS (Cyan arrows)
    # ─────────────────────────────────────────────────────────────────────
    if show_normals and 'hit_normals' in _kcc_vis_data:
        normal_color = (0.0, 1.0, 1.0, 0.9)  # Cyan
        normals = _kcc_vis_data['hit_normals']
        arrow_length = 0.4

        for origin, normal in normals:
            end = (
                origin[0] + normal[0] * arrow_length,
                origin[1] + normal[1] * arrow_length,
                origin[2] + normal[2] * arrow_length
            )
            all_verts.extend([origin, end])
            all_colors.extend([normal_color, normal_color])

    # ─────────────────────────────────────────────────────────────────────
    # GROUND RAY (UNIFIED: Cyan = static hit, Orange = dynamic hit, Purple = miss)
    # ─────────────────────────────────────────────────────────────────────
    if show_ground and 'ground_ray' in _kcc_vis_data:
        ray_data = _kcc_vis_data['ground_ray']
        origin = ray_data['origin']
        end = ray_data['end']
        hit = ray_data['hit']
        is_dynamic = ray_data.get('is_dynamic', False)

        # UNIFIED: Different colors for static vs dynamic ground
        if hit:
            if is_dynamic:
                ray_color = (1.0, 0.5, 0.0, 0.9)  # Orange = dynamic ground
            else:
                ray_color = (0.0, 1.0, 1.0, 0.9)  # Cyan = static ground
        else:
            ray_color = (0.5, 0.0, 0.5, 0.7)  # Purple = miss

        # Add ray line
        all_verts.extend([origin, end])
        all_colors.extend([ray_color, ray_color])

        # Add circle at hit point if hit
        if hit and 'hit_point' in ray_data:
            hit_point = ray_data['hit_point']
            segments = 12
            hit_radius = 0.08

            # Convert circle to LINES format for batching
            for i in range(segments):
                angle1 = (i / segments) * 2.0 * math.pi
                angle2 = ((i + 1) / segments) * 2.0 * math.pi
                x1 = hit_point[0] + hit_radius * math.cos(angle1)
                y1 = hit_point[1] + hit_radius * math.sin(angle1)
                z1 = hit_point[2]
                x2 = hit_point[0] + hit_radius * math.cos(angle2)
                y2 = hit_point[1] + hit_radius * math.sin(angle2)
                z2 = hit_point[2]
                all_verts.extend([(x1, y1, z1), (x2, y2, z2)])
                all_colors.extend([ray_color, ray_color])

    # ─────────────────────────────────────────────────────────────────────
    # MOVEMENT VECTORS (Green = intended, Red = actual)
    # ─────────────────────────────────────────────────────────────────────
    if show_movement and 'movement_vectors' in _kcc_vis_data:
        vectors = _kcc_vis_data['movement_vectors']
        origin = vectors.get('origin')
        vector_scale = getattr(scene, 'dev_debug_kcc_visual_vector_scale', 3.0)

        if origin and 'intended' in vectors:
            intended = vectors['intended']
            direction = (
                (intended[0] - origin[0]) * vector_scale,
                (intended[1] - origin[1]) * vector_scale,
                (intended[2] - origin[2]) * vector_scale
            )
            scaled_end = (
                origin[0] + direction[0],
                origin[1] + direction[1],
                origin[2] + direction[2]
            )
            intended_color = (0.0, 1.0, 0.0, 0.9)  # Green
            all_verts.extend([origin, scaled_end])
            all_colors.extend([intended_color, intended_color])

        if origin and 'actual' in vectors:
            actual = vectors['actual']
            direction = (
                (actual[0] - origin[0]) * vector_scale,
                (actual[1] - origin[1]) * vector_scale,
                (actual[2] - origin[2]) * vector_scale
            )
            scaled_end = (
                origin[0] + direction[0],
                origin[1] + direction[1],
                origin[2] + direction[2]
            )
            actual_color = (1.0, 0.0, 0.0, 0.9)  # Red
            all_verts.extend([origin, scaled_end])
            all_colors.extend([actual_color, actual_color])

    # ─────────────────────────────────────────────────────────────────────
    # VERTICAL BODY INTEGRITY RAY (Orange = clear, Red = EMBEDDED!)
    # ─────────────────────────────────────────────────────────────────────
    if 'vertical_ray' in _kcc_vis_data and _kcc_vis_data['vertical_ray']:
        v_ray = _kcc_vis_data['vertical_ray']
        origin = v_ray['origin']
        end = v_ray['end']
        blocked = v_ray.get('blocked', False)

        # IMPORTANT: Bright red when embedded, bright orange when clear
        ray_color = (1.0, 0.0, 0.0, 1.0) if blocked else (1.0, 0.6, 0.0, 1.0)  # Red/Orange

        # Draw THICK vertical line (multiple parallel lines for thickness)
        thickness_offset = 0.02
        offsets = [
            (0.0, 0.0),           # Center
            (thickness_offset, 0.0),   # +X
            (-thickness_offset, 0.0),  # -X
            (0.0, thickness_offset),   # +Y
            (0.0, -thickness_offset),  # -Y
        ]

        for ox, oy in offsets:
            origin_offset = (origin[0] + ox, origin[1] + oy, origin[2])
            end_offset = (end[0] + ox, end[1] + oy, end[2])
            all_verts.extend([origin_offset, end_offset])
            all_colors.extend([ray_color, ray_color])

        # Add LARGE visible markers at origin and end (circles)
        marker_size = 0.15  # Much larger
        segments = 12  # More segments for smoother circle
        for marker_pos in [origin, end]:
            for i in range(segments):
                angle1 = (i / segments) * 2.0 * math.pi
                angle2 = ((i + 1) / segments) * 2.0 * math.pi
                x1 = marker_pos[0] + marker_size * math.cos(angle1)
                y1 = marker_pos[1] + marker_size * math.sin(angle1)
                x2 = marker_pos[0] + marker_size * math.cos(angle2)
                y2 = marker_pos[1] + marker_size * math.sin(angle2)
                all_verts.extend([(x1, y1, marker_pos[2]), (x2, y2, marker_pos[2])])
                all_colors.extend([ray_color, ray_color])

        # Add crosshair at both ends for extra visibility
        cross_size = 0.10
        for marker_pos in [origin, end]:
            # Horizontal line
            all_verts.extend([
                (marker_pos[0] - cross_size, marker_pos[1], marker_pos[2]),
                (marker_pos[0] + cross_size, marker_pos[1], marker_pos[2])
            ])
            all_colors.extend([ray_color, ray_color])
            # Vertical line
            all_verts.extend([
                (marker_pos[0], marker_pos[1] - cross_size, marker_pos[2]),
                (marker_pos[0], marker_pos[1] + cross_size, marker_pos[2])
            ])
            all_colors.extend([ray_color, ray_color])

    # ═════════════════════════════════════════════════════════════════════
    # SINGLE BATCHED DRAW CALL (PERFORMANCE OPTIMIZED)
    # ═════════════════════════════════════════════════════════════════════
    if all_verts:
        batch = batch_for_shader(shader, 'LINES', {"pos": all_verts, "color": all_colors})
        shader.bind()
        batch.draw(shader)

    # Reset GPU state
    gpu.state.line_width_set(1.0)
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
    - Main thread applies results and platform carry
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

        # ═══════════════════════════════════════════════════════════════════
        # PLATFORM SYSTEM (Relative Position)
        # When on a dynamic platform, store position RELATIVE to the platform.
        # Each frame: worldPos = platform.matrix @ relativePos
        # Cost: One matrix multiply (~16 muls + 12 adds) - extremely fast
        # ═══════════════════════════════════════════════════════════════════
        self._platform_relative_pos = None  # (x, y, z) in platform's local space
        self._platform_obj_id = None        # id() of platform we're attached to

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
        self._vis_vertical_ray = None  # Vertical body integrity ray

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
        self._vis_vertical_ray = debug.get('vertical_ray', None)

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
        # UNIFIED PHYSICS: Ground source from worker (static or dynamic)
        # Worker now handles ALL collision detection (static grid + dynamic meshes)
        # We just need to extract which object we're standing on for platform carry
        # ─────────────────────────────────────────────────────────────────────
        ground_hit_source = result.get("ground_hit_source")  # "static", "dynamic_{obj_id}", or None
        self.ground_obj = None  # Default to no ground object
        self._vis_ground_source = ground_hit_source  # Cache for visualization (static vs dynamic)

        if ground_hit_source and ground_hit_source.startswith("dynamic_"):
            # Extract object ID from "dynamic_{obj_id}" format (worker uses id(obj))
            try:
                obj_id = int(ground_hit_source[8:])  # Remove "dynamic_" prefix, convert to int
                # CRITICAL: Look up in ALL proxy meshes, not just active ones!
                # If we only check dynamic_map, edge cases where AABB fails cause lookup
                # to fail, breaking the "standing_on" override in update_dynamic_meshes
                import bpy
                scene = bpy.context.scene
                for pm in scene.proxy_meshes:
                    dyn_obj = pm.mesh_object
                    if dyn_obj and pm.is_moving and id(dyn_obj) == obj_id:
                        self.ground_obj = dyn_obj
                        break
            except (ValueError, TypeError):
                pass  # Invalid ID format, ground_obj stays None

        # ─────────────────────────────────────────────────────────────────────
        # PLATFORM SYSTEM: Handled via relative position in step() method
        # Player position is stored relative to platform, then transformed
        # to world space each frame. No velocity tracking needed.
        # ─────────────────────────────────────────────────────────────────────

        # Log worker messages to fast buffer (worker collected these during computation)
        worker_logs = result.get("logs", [])
        if worker_logs:
            from ..developer.dev_logger import log_worker_messages
            log_worker_messages(worker_logs)

        # Debug output (FAST BUFFER LOGGING - no console, no I/O)
        if context and getattr(context.scene, 'dev_debug_kcc_physics', False):
            from ..developer.dev_logger import log_game
            # Extract debug data
            debug = result.get("debug", {})
            # Log to fast memory buffer (~1μs, no I/O)
            # Determine visual state for logging (matches visualizer colors)
            was_stuck = debug.get('was_stuck', False)
            h_blocked = debug.get('h_blocked', False)

            # Visual state indicator (matches capsule color)
            if was_stuck:
                state = "STUCK"
            elif h_blocked:
                state = "BLOCK"
            elif on_ground:
                state = "GROUND"
            else:
                state = "AIR"

            log_game("KCC", f"{state} pos=({self.pos.x:.2f},{self.pos.y:.2f},{self.pos.z:.2f}) "
                            f"step={debug.get('did_step_up', False)} | "
                            f"{debug.get('calc_time_us', 0):.0f}us {debug.get('rays_cast', 0)}rays "
                            f"{debug.get('triangles_tested', 0)}tris")

    # --------------------
    # Job building
    # --------------------

    def _build_physics_job(self, wish_dir, is_running, jump_requested, dt, context=None, dynamic_map=None):
        """Build KCC_PHYSICS_STEP job data for worker."""
        cfg = self.cfg

        # Extract debug flags from scene (if context available)
        # UNIFIED PHYSICS: All flags control unified physics (static + dynamic identical)
        debug_flags = {}
        if context and hasattr(context, 'scene'):
            scene = context.scene
            debug_flags = {
                # Unified physics (all show source: static/dynamic)
                "physics": getattr(scene, "dev_debug_physics", False),
                "ground": getattr(scene, "dev_debug_physics_ground", False),
                "horizontal": getattr(scene, "dev_debug_physics_horizontal", False),
                "body": getattr(scene, "dev_debug_physics_body", False),
                "ceiling": getattr(scene, "dev_debug_physics_ceiling", False),
                "step": getattr(scene, "dev_debug_physics_step", False),
                "slide": getattr(scene, "dev_debug_physics_slide", False),
                "slopes": getattr(scene, "dev_debug_physics_slopes", False),
                # Dynamic mesh system (unified with static)
                "dynamic_cache": getattr(scene, "dev_debug_dynamic_cache", False),
            }

        # Serialize dynamic mesh transforms (64 bytes per mesh) - lightweight, per-frame
        # Mesh triangles are cached via targeted broadcast_job (one-time, guaranteed delivery)
        dynamic_transforms = {}
        if dynamic_map:
            for dyn_obj in dynamic_map.keys():
                try:
                    matrix = dyn_obj.matrix_world
                    obj_id = id(dyn_obj)
                    # Serialize as 16-element tuple (row-major)
                    dynamic_transforms[obj_id] = (
                        matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3],
                        matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3],
                        matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3],
                        matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3],
                    )
                except Exception:
                    continue

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

            # Dynamic mesh transforms (per-frame, lightweight - 64 bytes per mesh)
            # Mesh triangles are cached via targeted broadcast_job (guaranteed delivery)
            "dynamic_transforms": dynamic_transforms,
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
        platform_linear_velocity_map=None,  # DEPRECATED - kept for API compat
        platform_ang_velocity_map=None,     # DEPRECATED - kept for API compat
        engine=None,
        context=None,
        physics_frame=0,
    ):
        """
        Same-Frame Physics Offload step:
        1. PLATFORM SYSTEM - relative position tracking (one matrix multiply)
        2. SNAPSHOT current state + input
        3. SUBMIT job to worker
        4. POLL for result (same-frame, with timeout)
        5. APPLY result immediately
        6. Write position to Blender
        """
        import time
        rot = self.obj.rotation_euler.copy()

        # Sync position from Blender (in case of external changes)
        self.pos = self.obj.location.copy()

        # ═══════════════════════════════════════════════════════════════════
        # PLATFORM SYSTEM (Relative Position - How Real Game Engines Do It)
        # ═══════════════════════════════════════════════════════════════════
        # When on a dynamic platform:
        #   - Store position RELATIVE to platform (once, on landing)
        #   - Each frame: worldPos = platform.matrix @ relativePos
        # Cost: ONE matrix multiply per frame (~16 muls + 12 adds)
        # Benefits: No velocity tracking, no timing issues, frame-perfect
        # ═══════════════════════════════════════════════════════════════════

        platform_attached = False

        if self.on_ground and self.ground_obj:
            current_platform_id = id(self.ground_obj)

            # Check if we just landed on a NEW platform
            if self._platform_obj_id != current_platform_id:
                # LANDING: Compute relative position (one-time)
                try:
                    inv_matrix = self.ground_obj.matrix_world.inverted()
                    local_pos = inv_matrix @ self.pos
                    self._platform_relative_pos = (local_pos.x, local_pos.y, local_pos.z)
                    self._platform_obj_id = current_platform_id

                    # Log platform attachment
                    if context and getattr(context.scene, 'dev_debug_dynamic_cache', False):
                        from ..developer.dev_logger import log_game
                        log_game("PLATFORM", f"ATTACH obj={current_platform_id} rel=({local_pos.x:.2f},{local_pos.y:.2f},{local_pos.z:.2f})")
                except Exception:
                    self._platform_relative_pos = None
                    self._platform_obj_id = None

            # EACH FRAME: Transform relative pos to world pos
            if self._platform_relative_pos is not None:
                try:
                    rel = self._platform_relative_pos
                    world_pos = self.ground_obj.matrix_world @ Vector((rel[0], rel[1], rel[2]))
                    self.pos = world_pos
                    platform_attached = True
                except Exception:
                    # Platform gone or invalid - detach
                    self._platform_relative_pos = None
                    self._platform_obj_id = None
        else:
            # NOT on ground or no ground_obj - detach from platform
            if self._platform_obj_id is not None:
                if context and getattr(context.scene, 'dev_debug_dynamic_cache', False):
                    from ..developer.dev_logger import log_game
                    log_game("PLATFORM", f"DETACH obj={self._platform_obj_id}")
            self._platform_relative_pos = None
            self._platform_obj_id = None

        # ─────────────────────────────────────────────────────────────────────
        # 1. SNAPSHOT current state + input
        # ─────────────────────────────────────────────────────────────────────
        # Increment frame counter
        self._physics_frame += 1

        # Frame number output (separate from KCC logs)
        if context and getattr(context.scene, 'dev_debug_frame_numbers', False):
            from ..developer.dev_logger import log_game
            from ..props_and_utils.exp_time import get_game_time
            game_time = get_game_time()
            log_game("FRAME", f"{self._physics_frame:04d} t={game_time:.3f}s")

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
            job_data = self._build_physics_job(wish_dir, is_running, jump_requested, dt, context, dynamic_map)
            job_id = engine.submit_job("KCC_PHYSICS_STEP", job_data)
            self._last_physics_job_id = job_id

            # Same-frame polling: wait for our result
            # Worker computes in ~100-200µs typically, but dynamic mesh transforms can add 1-2ms
            # Using adaptive polling: busy-poll first, then sleep if needed
            poll_start = time.perf_counter()
            poll_timeout = 0.005  # 5ms max wait (dynamic mesh adds latency)
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

            # Debug output (FAST BUFFER LOGGING)
            if context and getattr(context.scene, 'dev_debug_kcc_physics', False):
                from ..developer.dev_logger import log_game
                poll_time_us = (time.perf_counter() - poll_start) * 1_000_000
                if result_found:
                    log_game("KCC", f"SAME-FRAME pos=({self.pos.x:.2f},{self.pos.y:.2f},{self.pos.z:.2f}) "
                                    f"poll={poll_time_us:.0f}us")
                else:
                    log_game("KCC", f"TIMEOUT pos=({self.pos.x:.2f},{self.pos.y:.2f},{self.pos.z:.2f}) "
                                    f"poll={poll_time_us:.0f}us - using previous state")
        else:
            # NO ENGINE FALLBACK - Physics requires engine
            if context and getattr(context.scene, 'dev_debug_kcc_physics', False):
                from ..developer.dev_logger import log_game
                log_game("KCC", "WARNING: No engine available - physics step skipped")

        # ─────────────────────────────────────────────────────────────────────
        # 5. Write position to Blender
        # ─────────────────────────────────────────────────────────────────────
        self.obj.location = self.pos
        if abs(rot.z - self.obj.rotation_euler.z) > 1e-9:
            self.obj.rotation_euler = rot

        # ─────────────────────────────────────────────────────────────────────
        # 5b. UPDATE relative position after physics (player moved on platform)
        # ─────────────────────────────────────────────────────────────────────
        # If player is on a platform and moved (walked, jumped, etc.), we need
        # to recompute their relative position so they stay in the new spot
        if self._platform_obj_id is not None and self.ground_obj:
            try:
                inv_matrix = self.ground_obj.matrix_world.inverted()
                local_pos = inv_matrix @ self.pos
                self._platform_relative_pos = (local_pos.x, local_pos.y, local_pos.z)
            except Exception:
                pass  # Keep old relative pos if matrix inversion fails

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
            'movement_vectors': {},
            'vertical_ray': None
        }

        # Capsule spheres (feet + head)
        r = float(self.cfg.radius)
        h = float(self.cfg.height)

        feet_center = (self.pos.x, self.pos.y, self.pos.z + r)
        mid_center = (self.pos.x, self.pos.y, self.pos.z + min(h * 0.5, h - r))  # Mid-height collision sphere
        head_center = (self.pos.x, self.pos.y, self.pos.z + h - r)

        vis_data['capsule_spheres'] = [
            (feet_center, r),
            (mid_center, r),
            (head_center, r)
        ]

        # Hit normals (from cached collision data)
        if hasattr(self, '_vis_hit_normals') and self._vis_hit_normals:
            vis_data['hit_normals'] = self._vis_hit_normals

        # Ground ray - UNIFIED: shows source (static vs dynamic) with different colors
        ray_origin = (self.pos.x, self.pos.y, self.pos.z + 0.5)
        ray_end = (self.pos.x, self.pos.y, self.pos.z - self.cfg.snap_down)

        # Get ground source (static, dynamic_*, or None)
        ground_source = getattr(self, '_vis_ground_source', None)
        is_dynamic_ground = ground_source and ground_source.startswith("dynamic_")

        vis_data['ground_ray'] = {
            'origin': ray_origin,
            'end': ray_end,
            'hit': self.on_ground,
            'source': ground_source,  # "static", "dynamic_{obj_id}", or None
            'is_dynamic': is_dynamic_ground
        }

        if self.on_ground:
            vis_data['ground_ray']['hit_point'] = (self.pos.x, self.pos.y, self.pos.z)

        # Vertical body integrity ray (from worker)
        if self._vis_vertical_ray:
            vis_data['vertical_ray'] = self._vis_vertical_ray

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