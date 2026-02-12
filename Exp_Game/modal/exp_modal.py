# File: exp_modal.py
import bpy
import math
import time
import sys

from ..physics.exp_raycastutils import create_bvh_tree
from ..startup_and_reset.exp_spawn import spawn_user
from .exp_view_helpers import (_update_axis_resolution_on_release, _resolved_move_keys, _axis_of_key,
                                smooth_rotate_towards_camera, _maybe_rebind_view3d, _bind_view3d_once)
from ..animations.state_machine import CharacterStateMachine, AnimState
from ..audio.exp_audio import (get_global_audio_state_manager, clean_audio_temp)
from ..startup_and_reset.exp_startup import (center_cursor_in_3d_view, clear_old_dynamic_references,
                          record_user_settings, apply_performance_settings, restore_user_settings,
                          move_armature_and_children_to_scene, revert_to_original_scene,
                            ensure_timeline_at_zero, ensure_object_mode)  
from ..interactions.exp_interactions import set_interact_pressed, set_action_pressed, reset_all_interactions
from ..reactions.exp_reactions import reset_all_tasks
from ..props_and_utils.exp_time import init_time
from ..reactions.exp_custom_ui import register_ui_draw, unregister_ui_draw, clear_all_text, show_controls_info
from ..systems.exp_counters_timers import reset_all_counters, reset_all_timers
from ..startup_and_reset.exp_game_reset import (capture_scene_state, reset_property_reactions, capture_initial_cam_state,
                              restore_initial_session_state, capture_initial_character_state, restore_scene_state)
from ..audio import exp_globals
from ..mouse_and_movement.exp_cursor import (
    setup_cursor_region,
    handle_mouse_move,
    release_cursor_clip,
    ensure_cursor_hidden_if_mac,
    force_restore_cursor,
    cache_blender_hwnd,
    is_blender_focused,
    is_delta_sane,
)
from ..physics.exp_kcc import KinematicCharacterController
from ..startup_and_reset.exp_fullscreen import exit_fullscreen_once
from .exp_loop import GameLoop
from ..physics.exp_view_fpv import reset_fpv_rot_scale
from .exp_engine_bridge import (
    init_engine,
    shutdown_engine,
    init_animations,
    shutdown_animations,
    # NOTE: cache_animations_in_workers removed (2025-01) - animations computed locally
)

def _first_view3d_r3d():
    """
    Return a valid SpaceView3D.region_3d from any open VIEW_3D/WINDOW,
    or None if none exists yet (e.g., during fullscreen swaps).
    """
    import bpy
    wm = bpy.context.window_manager
    if not wm:
        return None
    for win in wm.windows:
        scr = getattr(win, "screen", None)
        if not scr:
            continue
        for area in scr.areas:
            if area.type != 'VIEW_3D':
                continue
            space = area.spaces.active
            r3d = getattr(space, "region_3d", None)
            if r3d:
                return r3d
    return None


# ============================================================================
# ACTIVE MODAL OPERATOR TRACKING
# Global reference for systems that need engine access
# ============================================================================

_active_modal_operator = None

def get_active_modal_operator():
    """
    Return the currently active ExpModal operator instance.
    Used by systems that need access to the engine or operator state.
    Returns None if no modal is running.
    """
    return _active_modal_operator


class ExpModal(bpy.types.Operator):
    """Windowed (minimized) game start."""

    bl_idname = "view3d.exp_modal"
    bl_label = "Third Person Orbit"
    bl_options = {'BLOCKING', 'GRAB_CURSOR'}

    # ---------------------------
    #  Properties
    # ---------------------------
    time_scale: bpy.props.IntProperty(
        name="Time Scale",
        default=1,           # keep your current feel
        min=1,
        max=5,
        description="Controls how quickly time passes in the simulation (integer)"
    )

    launched_from_ui: bpy.props.BoolProperty(
        name="Launched from UI",
        default=False,
    )
    
    # -----------------------------
    # Internal State Variables
    # -----------------------------
    pitch: bpy.props.FloatProperty(name="Pitch", default=0.0)
    yaw: bpy.props.FloatProperty(name="Yaw", default=0.0)
    z_velocity: bpy.props.FloatProperty(name="Z Velocity", default=0.0)
    gravity: float = -9.8

    delta_time: float = 0.0  # Updated each frame based on real time * time_scale
    _last_time: float = 0.0  # Used to calculate real time between frames

    # --- Fixed 30 Hz physics scheduling (wall-clock) ---
    physics_hz: int = 30
    physics_dt: float = 1.0 / 30.0
    _next_physics_tick: float = 0.0  # monotonic time for next physics step


    #----------------------------------------------------
    #catch up - keeps slower systems in sync (no slo mo)
    max_catchup_steps: int = 3
    def _physics_steps_due(self) -> int:
        """
        Return how many 30 Hz steps are due *now*, capped to max_catchup_steps.
        Any extra debt is dropped and the schedule realigns to 'now'.
        """
        now = time.perf_counter()
        steps = 0
        # PERFORMANCE: Use class attribute directly instead of getattr() every call
        # Saves ~1-2µs per frame (30-60µs/sec)
        cap = self.max_catchup_steps

        while now >= self._next_physics_tick and steps < cap:
            steps += 1
            self._next_physics_tick += self.physics_dt

        # If we still lag past the next tick after executing 'cap' steps,
        # drop any remaining debt to avoid a spiral and realign.
        if now >= self._next_physics_tick:
            self._next_physics_tick = now + self.physics_dt

        return steps
    #------------------------------------------------------
    is_grounded: bool = False
    is_jumping: bool = False
    jump_timer: float = 0.0
    jump_duration: float = 0.5
    jump_cooldown: bool = False
    jump_key_released: bool = True

    # Rotation smoothing for character turning
    rotation_smoothness: float = 0.65 #higher value = faster turn

    # References to objects / data
    target_object = None
    bvh_tree = None
    char_state_machine = None

    # PERFORMANCE: Class-level defaults for engine/animation attributes
    # Avoids expensive hasattr() checks in hot paths (saves ~2-5µs per check)
    engine = None
    anim_controller = None
    _pending_interaction_job_id = None
    _loop = None
    dt: float = 0.016  # Default 60fps frame time until first update

    # Keep track of pressed keys
    keys_pressed = set()
    action_pressed: bool = False
    _press_seq: int = 0
    _axis_last = {'x': None, 'y': None}          # selected key per axis
    _axis_candidates = {'x': {}, 'y': {}}        # held keys → press order

    # Timer handle for the modal operator
    _timer = None

    #physics controller and time
    physics_controller = None
    fixed_clock = None

    # ========== ENGINE SYNCHRONIZATION TRACKING ==========
    # Track current physics frame number for engine sync
    _physics_frame: int = 0

    # Map job_id → {"frame": int, "timestamp": float} for latency tracking
    _pending_jobs = {}

    # Sync metrics for performance monitoring
    _sync_jobs_submitted = 0
    _sync_results_received = 0
    _sync_frame_latencies = []
    _sync_time_latencies = []
    _sync_last_report_frame = 0

    # Comprehensive test manager (controlled by scene.dev_debug_sync_test)
    _test_manager = None
    # ====================================================

    # ========== CURSOR/FOCUS STATE MACHINE (Windows only for now) ==========
    # States
    STATE_RUNNING = 'RUNNING'
    STATE_PAUSED = 'PAUSED'

    _game_state: str = 'RUNNING'
    _last_focus_check: float = 0.0
    _focus_check_interval: float = 1.0  # Check focus once per second
    _focus_lost_logged: bool = False    # Prevent log spam
    _delta_anomaly_logged: bool = False # Prevent log spam
    # =====================================================================

    # store the user-chosen keys from preferences only
    pref_forward_key:  str = "W"
    pref_backward_key: str = "S"
    pref_left_key:     str = "A"
    pref_right_key:    str = "D"
    pref_jump_key:     str = "SPACE"
    pref_run_key:      str = "LEFT_SHIFT"
    pref_interact_key: str = "E"
    pref_action_key:   str = "LEFTMOUSE"
    pref_reset_key:    str = "R"
    pref_end_game_key: str = "ESC"
    

    # ---------------------------------
    # Initialization For Dynamic Mesh
    # ---------------------------------
    def ensure_dynamic_fields_exist(self):
        """
        Make sure we have dictionaries for dynamic features,
        even if we don't have any dynamic meshes.
        """
        # NOTE: platform_delta_map and platform_motion_map removed (dead code)
        if not hasattr(self, "platform_prev_positions"):
            self.platform_prev_positions = {}
        if not hasattr(self, "platform_prev_quaternions"):
            self.platform_prev_quaternions = {}
        if not hasattr(self, "moving_meshes"):
            self.moving_meshes = []
        if not hasattr(self, "grounded_platform"):
            self.grounded_platform = None

    _initial_game_state = {}


    #loop initialization
    _loop = None


    # ═══════════════════════════════════════════════════════════════════════════
    # EMERGENCY CLEANUP - Critical Failure Recovery
    # ═══════════════════════════════════════════════════════════════════════════
    # This is the SINGLE entry point for crash recovery. All exception handlers
    # call this method. It attempts full cleanup via cancel(), falling back to
    # minimal cleanup (cursor + engine) if cancel() itself fails.
    #
    # Call this from ANY exception handler in invoke() or modal().
    # ═══════════════════════════════════════════════════════════════════════════

    def _emergency_cleanup(self, context, error_msg: str = "Unknown error"):
        """
        Emergency cleanup when modal crashes. Attempts full cancel() cleanup,
        falls back to minimal cursor/engine cleanup if that fails.

        Args:
            context: Blender context
            error_msg: Error message for logging
        """
        print(f"\n[CRITICAL FAILURE] {error_msg}")
        print("="*60)
        print("  Running emergency cleanup...")
        print("="*60)

        # Try full cleanup via cancel()
        try:
            self.cancel(context)
            print("  ✓ Full cleanup completed via cancel()")
        except Exception as cancel_error:
            # cancel() failed - do minimal cleanup
            print(f"  ✗ cancel() failed: {cancel_error}")
            print("  Attempting minimal cleanup...")

            # 1. Restore cursor (most important for UX)
            try:
                force_restore_cursor()
                print("  ✓ Cursor restored")
            except Exception as cursor_error:
                print(f"  ✗ Cursor restore failed: {cursor_error}")

            # 2. Shutdown engine (prevent worker process leaks)
            if hasattr(self, 'engine') and self.engine:
                try:
                    self.engine.shutdown()
                    self.engine = None
                    print("  ✓ Engine shutdown")
                except Exception as engine_error:
                    print(f"  ✗ Engine shutdown failed: {engine_error}")

            # 3. Clear global operator references
            global _active_modal_operator
            _active_modal_operator = None
            exp_globals.ACTIVE_MODAL_OP = None

        print("="*60 + "\n")


    # ---------------------------
    # Main Modal Functions
    # ---------------------------
    def invoke(self, context, event):
        global _active_modal_operator
        # NOTE: _active_modal_operator is set at the END of invoke, just before return {'RUNNING_MODAL'}
        # This ensures if invoke fails at any point, the global reference is never set to a dead operator

        scene = context.scene

        # --- proxy records for hidden meshes - to avoid performance culling conflicts ---
        if not hasattr(self, "_proxy_mesh_original_states"):
            self._proxy_mesh_original_states = {}
        else:
            self._proxy_mesh_original_states.clear()

        if not hasattr(self, "_force_hide_names"):
            self._force_hide_names = set()
        else:
            self._force_hide_names.clear()
        # --------------------------------------------------------------------------------

        # ========== FAST BUFFER LOGGER SETUP ==========
        # Initialize fast memory buffer logger for game diagnostics
        from ..developer.dev_logger import start_session
        start_session()
        # ============================================

        # Capture both camera/character state
        capture_initial_cam_state(self, context)

        #clear UI if any
        try:
            # EXEC_DEFAULT so it doesn't pop up a dialog
            bpy.ops.view3d.remove_package_display('EXEC_DEFAULT')
        except Exception as e:
            from ..developer.dev_logger import log_game
            log_game("MODAL", f"remove_package_display failed: {e}")

        addon_prefs = context.preferences.addons["Exploratory"].preferences

        if not self.launched_from_ui:
            ensure_object_mode(context)

        #----------------------------
        #Scene state and preference setup
        #----------------------------
        record_user_settings(scene)
        capture_scene_state(self,context)

        performance_level = addon_prefs.performance_level  # e.g., "LOW", "MEDIUM", "HIGH"
        apply_performance_settings(scene, performance_level)

        ensure_timeline_at_zero()

        # A) Clean out old audio temp BEFORE build (skip when slots-lock is ON)
        if not getattr(context.scene, "character_slots_lock", False):
            clean_audio_temp()

        result = bpy.ops.exploratory.build_character('EXEC_DEFAULT')
        if 'FINISHED' not in result:
            self.report({'ERROR'}, "BuildCharacter operator did not finish. Cannot proceed.")
            return {'CANCELLED'}

        # --- CRITICAL CHECK ---
        if not context.scene.target_armature:
            self.report({'ERROR'}, "No armature assigned to scene.target_armature! Cancelling game.")
            return {'CANCELLED'}

        #---CUSTOM_UI---
        clear_all_text()
        register_ui_draw()


        # 2) Override our local sensitivity from user preferences
        self.sensitivity = addon_prefs.mouse_sensitivity * .001

        # 3) Store each user-chosen key in local variables
        self.pref_forward_key  = addon_prefs.key_forward
        self.pref_backward_key = addon_prefs.key_backward
        self.pref_left_key     = addon_prefs.key_left
        self.pref_right_key    = addon_prefs.key_right
        self.pref_jump_key     = addon_prefs.key_jump
        self.pref_run_key      = addon_prefs.key_run
        self.pref_action_key   = addon_prefs.key_action
        self.pref_interact_key = addon_prefs.key_interact
        self.pref_reset_key    = addon_prefs.key_reset
        self.pref_end_game_key = addon_prefs.key_end_game

        #intialize the movement and game settings
        scene.mobility_game.allow_movement = True
        scene.mobility_game.allow_jump = True
        scene.mobility_game.allow_sprint = True

        # If performance mode is on => optionally do something
        if addon_prefs.performance_mode:
            pass
        #4A) Initialize dynamic system
        clear_old_dynamic_references(self)

        # 4B) Clear any old dynamic references
        self.ensure_dynamic_fields_exist()

        # 4C) Spawn the user/object if needed
        spawn_user()

        #capture character state after spawning
        capture_initial_character_state(self, context)

        #4D Game time
        init_time()  # Start the game clock at 0

        # show control hints for 10 seconds
        show_controls_info(10.0)

        # 4E) *** RESET ALL INTERACTIONS AND TASKS***
        reset_all_interactions(context.scene)
        reset_all_tasks()
        reset_all_counters(context.scene)
        reset_all_timers(context.scene)
        reset_property_reactions(context.scene)

        # 4F) Clear health cache (fresh start, ENABLE_HEALTH reactions haven't run yet)
        from ..reactions.exp_health import clear_all_health
        clear_all_health()

        # 5) Center the cursor for reading mouse deltas
        center_cursor_in_3d_view(context)

        # 6) Grab target object from scene property
        self.target_object = context.scene.target_armature
        if self.target_object:
            # --- Initialize camera pitch & yaw robustly ---
            self.pitch = math.radians(float(getattr(context.scene, "pitch_angle", 15.0)))

            # Try, in order: region_data → space_data.region_3d → any VIEW_3D r3d → armature yaw
            r3d = getattr(context, "region_data", None)
            if r3d is None:
                try:
                    r3d = context.space_data.region_3d
                except Exception:
                    r3d = None
            if r3d is None:
                r3d = _first_view3d_r3d()

            if r3d is not None:
                try:
                    self.yaw = r3d.view_rotation.to_euler().z
                except Exception:
                    self.yaw = self.target_object.matrix_world.to_euler('XYZ').z if self.target_object else 0.0
            else:
                self.yaw = self.target_object.matrix_world.to_euler('XYZ').z if self.target_object else 0.0
        # Set up the animation system
        self.char_state_machine = CharacterStateMachine()
        init_animations(self, context)
        move_armature_and_children_to_scene(self.target_object, context.scene)

        # --- Physics controller + HARD-LOCKED 30 Hz scheduler ---
        self.physics_controller = KinematicCharacterController(self.target_object, context.scene.char_physics)

        # Absolute, real-time 30 Hz physics (independent of time_scale)
        self.physics_hz = 30
        self.physics_dt = 1.0 / float(self.physics_hz)

        self._next_physics_tick = time.perf_counter() + self.physics_dt

        # FixedStepClock not needed for scheduling; keep None to avoid confusion
        self.fixed_clock = None



        # 7) Setup modal cursor region and hide the system cursor
        cache_blender_hwnd()  # Cache window handle for focus detection
        self._last_focus_check = time.perf_counter()
        self._focus_lost_logged = False
        self._delta_anomaly_logged = False
        self._game_state = self.STATE_RUNNING  # Start in running state
        setup_cursor_region(context, self)
        # 8) Create a timer (runs as fast as possible, interval=0.0)
        self._timer = context.window_manager.event_timer_add(1.0/30.0, window=context.window)
        self._last_time = time.perf_counter()

        # 9) Add ourselves to Blender's modal event loop
        exp_globals.ACTIVE_MODAL_OP = self
        context.window_manager.modal_handler_add(self)

        # 10) build a BVH tree for ground collisions
        
        all_static_meshes = []
        self.moving_meshes = []  # We'll store dynamic objects here

        for entry in context.scene.proxy_meshes:
            if not entry.mesh_object or entry.mesh_object.type != 'MESH':
                continue

            if entry.is_moving:
                # dynamic => store it for per-frame checks
                self.moving_meshes.append(entry.mesh_object)
            else:
                # static => goes into the big BVH
                all_static_meshes.append(entry.mesh_object)

        # Build the BVH only for the static subset:
        if all_static_meshes:
            self.bvh_tree = create_bvh_tree(all_static_meshes)
        else:
            self.bvh_tree = None

        # Extract triangles for raycast offloading
        from ..physics.exp_geometry import extract_static_triangles, print_geometry_report, build_uniform_grid
        if all_static_meshes:
            self.static_triangles = extract_static_triangles(all_static_meshes)
            print_geometry_report(self.static_triangles, context)

            # Phase 2: Build spatial acceleration grid
            # Cell size auto-computed based on triangle density (targets ~50 tris/cell)
            self.spatial_grid = build_uniform_grid(self.static_triangles, cell_size=None, context=context)
        else:
            self.static_triangles = []
            self.spatial_grid = None

        #define dynamic platforms (vertical movement)
        self.platform_prev_positions = {}
        for dyn_obj in self.moving_meshes:
            if dyn_obj:
                # Store its initial position
                self.platform_prev_positions[dyn_obj] = dyn_obj.matrix_world.translation.copy()

        self.platform_prev_quaternions = {}  # OPTIMIZED: 4 floats vs 16 for matrices
        for dyn_obj in self.moving_meshes:
            if dyn_obj:
                self.platform_prev_quaternions[dyn_obj] = dyn_obj.matrix_world.to_quaternion()

        for entry in context.scene.proxy_meshes:
            if entry.hide_during_game and entry.mesh_object:
                name = entry.mesh_object.name
                self._proxy_mesh_original_states[name] = entry.mesh_object.hide_viewport
                entry.mesh_object.hide_viewport = True
                self._force_hide_names.add(name)

        # 11) Clear any previously pressed keys
        self.keys_pressed.clear()
        self._press_seq = 0
        self._axis_last = {'x': None, 'y': None}
        self._axis_candidates = {'x': {}, 'y': {}}

        # ═══════════════════════════════════════════════════════════════════
        # ENGINE INITIALIZATION - CRITICAL GATE
        # ═══════════════════════════════════════════════════════════════════
        try:
            success, error_msg = init_engine(self, context)
            if not success:
                self.report({'ERROR'}, f"{error_msg} - aborting game")
                return {'CANCELLED'}

            # Cache animations in worker now that engine is running
            from .exp_engine_bridge import cache_animations_in_workers
            if self.anim_controller and self.anim_controller.cache.count > 0:
                cache_success = cache_animations_in_workers(self, context)
                if not cache_success:
                    self.report({'ERROR'}, "Failed to cache animations in worker - aborting game")
                    return {'CANCELLED'}

            # Initialize interaction offload tracking
            self._pending_interaction_job_id = None  # Track pending INTERACTION_CHECK_BATCH job
            self._interaction_map = []  # Maps worker indices to scene interactions

            # Game loop initialization
            self._loop = GameLoop(self)
            _bind_view3d_once(self, context)

        except Exception as e:
            # Engine initialization failed - run emergency cleanup
            import traceback
            traceback.print_exc()
            self._emergency_cleanup(context, f"Engine init failed: {e}")
            self.report({'ERROR'}, f"Engine initialization failed: {e}")
            return {'CANCELLED'}

        # Set global operator reference ONLY after all initialization succeeds
        # This ensures failed invoke() never leaves a dangling reference
        _active_modal_operator = self

        # ========== CRITICAL: Reset timing AFTER all init to prevent startup stutter ==========
        # This must be the LAST thing before return, after engine init, to avoid catch-up bursts
        now = time.perf_counter()
        self._last_time = now
        self._next_physics_tick = now + self.physics_dt
        # ======================================================================================

        return {'RUNNING_MODAL'}

    # ========== PAUSE/RESUME (Windows only for now) ==========

    def _pause_game(self, context):
        """
        Pause the game when focus is lost. Windows only.
        Releases cursor confinement, shows cursor, clears input state.
        """
        if sys.platform != 'win32':
            return  # No-op on Mac/Linux for now

        if self._game_state == self.STATE_PAUSED:
            return  # Already paused

        from ..developer.dev_logger import log_game
        log_game("CURSOR_STATE", "Pausing game - releasing cursor")

        # Release cursor confinement and show cursor
        release_cursor_clip()
        try:
            context.window.cursor_modal_restore()
        except Exception:
            pass

        # Clear all pressed keys to prevent stuck input
        self.keys_pressed.clear()
        self._axis_last = {'x': None, 'y': None}
        self._axis_candidates = {'x': {}, 'y': {}}

        self._game_state = self.STATE_PAUSED

    def _resume_game(self, context):
        """
        Resume the game when user clicks back. Windows only.
        Re-applies cursor confinement, hides cursor, resets mouse tracking.
        """
        if sys.platform != 'win32':
            return  # No-op on Mac/Linux for now

        if self._game_state == self.STATE_RUNNING:
            return  # Already running

        from ..developer.dev_logger import log_game
        log_game("CURSOR_STATE", "Resuming game - re-confining cursor")

        # Re-confine cursor and hide it
        from ..mouse_and_movement.exp_cursor import confine_cursor_to_window
        confine_cursor_to_window()
        try:
            context.window.cursor_modal_set('NONE')
        except Exception:
            pass

        # Reset mouse tracking to prevent delta anomaly on first move
        self.last_mouse_x = None
        self.last_mouse_y = None

        # ========== CRITICAL: Reset timing to prevent catch-up stutter ==========
        now = time.perf_counter()
        self._last_time = now                      # Reset delta_time calculation
        self._next_physics_tick = now + self.physics_dt  # Reset physics scheduler
        # ========================================================================

        # Reset detection flags
        self._delta_anomaly_logged = False
        self._focus_lost_logged = False

        self._game_state = self.STATE_RUNNING

    def modal(self, context, event):
        try:
            # End game hotkey - always works, even when paused
            if event.type == self.pref_end_game_key and event.value == 'PRESS':
                self.cancel(context)
                return {'CANCELLED'}

            # ========== PAUSED STATE HANDLING (Windows only) ==========
            if self._game_state == self.STATE_PAUSED:
                # When paused, only listen for click to resume
                if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                    # Check if Blender is now focused before resuming
                    if is_blender_focused():
                        self._resume_game(context)
                    else:
                        from ..developer.dev_logger import log_game
                        log_game("CURSOR_STATE", "Click detected but Blender not focused, waiting...")
                # Skip all game logic while paused
                return {'RUNNING_MODAL'}
            # ==========================================================

            if event.type == 'TIMER':
                # ========== FOCUS DETECTION (1/sec, Windows only) ==========
                if sys.platform == 'win32':
                    now = time.perf_counter()
                    if now - self._last_focus_check >= self._focus_check_interval:
                        self._last_focus_check = now
                        if not is_blender_focused():
                            # Focus lost - pause the game
                            self._pause_game(context)
                            return {'RUNNING_MODAL'}
                # ===========================================================

                if self._loop:
                    self._loop.on_timer(context)

                # ========== ENGINE HEARTBEAT ==========
                # Send heartbeat to confirm engine is alive
                # (Polling moved to GameLoop.on_timer() for proper frame sync)
                if hasattr(self, 'engine') and self.engine:
                    self.engine.send_heartbeat()
                # ======================================

                return {'RUNNING_MODAL'}

            # Key events
            elif event.type in {
                self.pref_forward_key, self.pref_backward_key, self.pref_left_key,
                self.pref_right_key,  self.pref_run_key,       self.pref_jump_key,
                self.pref_interact_key, self.pref_reset_key,   self.pref_action_key,
            }:
                # Option A: delegate to loop (keeps modal slim)
                if self._loop:
                    self._loop.handle_key_input(event)
                else:
                    self.handle_key_input(event)

            # Mouse -> camera rotation
            elif event.type == 'MOUSEMOVE':
                # ========== DELTA SANITY CHECK (fast buffer logging) ==========
                if not is_delta_sane(self.last_mouse_x, self.last_mouse_y,
                                     event.mouse_x, event.mouse_y):
                    if not self._delta_anomaly_logged:
                        from ..developer.dev_logger import log_game
                        dx = abs(event.mouse_x - (self.last_mouse_x or 0))
                        dy = abs(event.mouse_y - (self.last_mouse_y or 0))
                        log_game("CURSOR", f"Delta anomaly - mouse jumped dx={dx}, dy={dy}")
                        self._delta_anomaly_logged = True
                elif self._delta_anomaly_logged:
                    # Only reset flag when returning to normal (not every frame)
                    self._delta_anomaly_logged = False
                # ========================================================

                # Update yaw/pitch only — keep camera/FPV on the TIMER clock.
                handle_mouse_move(self, context, event)
                ensure_cursor_hidden_if_mac(context)

            return {'RUNNING_MODAL'}

        except Exception as e:
            # Modal crashed - run emergency cleanup
            import traceback
            traceback.print_exc()
            self._emergency_cleanup(context, f"Modal crashed: {e}")
            self.report({'ERROR'}, f"Game crashed: {e}")
            return {'CANCELLED'}


    def cancel(self, context):

        # ========== GAME LOOP SHUTDOWN ==========
        if hasattr(self, '_loop') and self._loop:
            self._loop.shutdown()
        # ========================================

        # ========== ANIMATION SHUTDOWN ==========
        shutdown_animations(self, context)
        # ========================================

        # ========== ENGINE SHUTDOWN ==========
        shutdown_engine(self, context)
        self._pending_jobs.clear()
        # ====================================

        # ========== KCC VISUALIZATION CLEANUP ==========
        # Remove GPU draw handlers for KCC visualization
        if hasattr(self, 'physics_controller') and self.physics_controller:
            self.physics_controller.cleanup_debug_handlers()
        # ====================================

        # ========== CAMERA STATE CLEANUP ==========
        # Clean up per-operator camera smoothers/latches/view state
        from ..physics.exp_view import cleanup_camera_state
        cleanup_camera_state(id(self))
        # ====================================

        # ========== DYNAMIC MESH STATE CLEANUP ==========
        from ..physics.exp_dynamic import cleanup_dynamic_mesh_state
        cleanup_dynamic_mesh_state(self)
        # ====================================

        # ========== AUDIO CLEANUP ==========
        from ..audio.exp_audio import reset_audio_managers, clean_audio_temp
        from ..audio.exp_globals import clear_modal_reference
        reset_audio_managers()
        clear_modal_reference()  # Clear ACTIVE_MODAL_OP only on game END (not reset)
        clean_audio_temp()
        # ====================================

        # ========== FONT CACHE CLEANUP ==========
        from ..reactions.exp_fonts import clear_font_cache
        clear_font_cache()
        # ====================================

        # ========== STATS CLEANUP ==========
        from ..developer.dev_stats import reset_stats
        reset_stats()
        # ====================================

        # ========== TRACKER CALLBACKS CLEANUP ==========
        from ..props_and_utils.trackers import clear_tracker_callbacks
        clear_tracker_callbacks()
        # ====================================

        # ========== BLEND SYSTEM CLEANUP ==========
        from ..animations.blend_system import shutdown_blend_system
        shutdown_blend_system()
        # ====================================

        # ========== UTILITY STORE CACHE CLEANUP ==========
        from ..props_and_utils.exp_utility_store import clear_uid_index_cache
        clear_uid_index_cache()
        # ====================================

        # ========== CROSSHAIRS CLEANUP ==========
        from ..reactions.exp_crosshairs import disable_crosshairs
        disable_crosshairs()
        # ====================================

        # ========== PROJECTILE CLEANUP ==========
        from ..reactions.exp_projectiles import clear as clear_projectiles
        clear_projectiles()
        # ====================================

        # ========== HEALTH CLEANUP ==========
        from ..reactions.exp_health import clear_all_health, disable_health_ui
        clear_all_health()
        disable_health_ui()
        # ====================================

        # NOTE: Do NOT call reset_fullscreen_state() here!
        # exit_fullscreen_once() (called below) needs the _state["orig_ui"] data
        # to properly restore the UI. It handles its own cleanup after restoration.

        # ========== LATENCY TRACKING CLEANUP ==========
        if hasattr(self, '_sync_frame_latencies'):
            self._sync_frame_latencies.clear()
        if hasattr(self, '_sync_time_latencies'):
            self._sync_time_latencies.clear()
        # ====================================

        # ========== DIAGNOSTICS LOG EXPORT ==========
        # Export fast buffer logger diagnostics if enabled
        from ..developer.dev_logger import export_game_log, clear_log, get_buffer_size
        if context.scene.dev_export_session_log and get_buffer_size() > 0:
            import os
            log_path = "C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt"
            log_dir = os.path.dirname(log_path)
            if os.path.exists(log_dir):
                export_game_log(log_path)
            clear_log()
        # ====================================

        # Restore the cursor modal state
        release_cursor_clip()
        context.window.cursor_modal_restore()

        #if in fullscreen, exit
        exit_fullscreen_once()

        # Remove the modal timer if it exists
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        
        # Clear any pressed keys
        self.keys_pressed.clear()
        
        # Restore scene state and user preferences
        restore_user_settings(context.scene)
        # NOTE: Audio already cleaned up by reset_audio_managers() above (line 801)
        # Don't call stop_all_sounds() or get_global_audio_state_manager().stop_current_sound() again

        # Restore proxy mesh visibility (only if NOT launched from UI, since scene will be reverted)
        if not self.launched_from_ui:
            for entry in context.scene.proxy_meshes:
                if entry.hide_during_game and entry.mesh_object:
                    original = self._proxy_mesh_original_states.get(entry.mesh_object.name)
                    if original is not None:
                        entry.mesh_object.hide_viewport = original

        #-----------------------------------------------
        ### Cleanup scene ###
        # When launched from UI, revert to original scene
        if self.launched_from_ui:
            revert_to_original_scene(context)
        #-----------------------------------------------

        # Unregister UI draw handler (also clears text)
        unregister_ui_draw()

        #reset --------------------reset --------#
        if not self.launched_from_ui:
            restore_scene_state(self, context)
            reset_fpv_rot_scale(context.scene)

        if not self.launched_from_ui:
            bpy.ops.exploratory.reset_game(
                'INVOKE_DEFAULT',
                skip_restore=True
            )



        #reset --------------------reset --------#

        # ─── Restore camera (always) and character (if launched_from_ui) ──
        restore_initial_session_state(self, context)

        # If launched from the custom UI, schedule UI popups
        if self.launched_from_ui:
            # Helper function: find a valid VIEW_3D area and WINDOW region.
            def get_valid_view3d_override():
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == 'VIEW_3D':
                            for region in area.regions:
                                if region.type == 'WINDOW':
                                    return {
                                        'window': window,
                                        'screen': window.screen,
                                        'area': area,
                                        'region': region
                                    }
                return None

            # Delayed callback function to invoke the UI operators.
            def delayed_ui_popups():
                override_ctx = get_valid_view3d_override()
                if override_ctx is not None:
                    with bpy.context.temp_override(**override_ctx):
                        bpy.ops.view3d.popup_social_details('INVOKE_REGION_WIN')
                else:
                    from ..developer.dev_logger import log_game
                    log_game("MODAL", "No valid VIEW3D context found for UI popups.")
                return None

            # Register the timer to delay UI calls.
            bpy.app.timers.register(delayed_ui_popups, first_interval=0.5)

        # Clear global modal operator references
        global _active_modal_operator
        _active_modal_operator = None
        exp_globals.ACTIVE_MODAL_OP = None

    # ---------------------------
    # Update / Logic Methods
    # ---------------------------
    def update_time(self):
        """Stable, scaled delta_time based on a monotonic clock."""
        current_time = time.perf_counter()
        if self._last_time == 0.0:
            self._last_time = current_time
        real_dt = current_time - self._last_time
        self._last_time = current_time

        # Clamp to avoid huge spikes after stalls (e.g., window focus)
        real_dt = max(0.0, min(real_dt, 0.050))  # 50 ms max per tick

        self.delta_time = real_dt * float(self.time_scale)

    def _physics_due(self) -> bool:
        """Return True if a 30 Hz physics step is due (wall-clock)."""
        return time.perf_counter() >= self._next_physics_tick

    def update_movement_and_gravity(self, context, steps: int = 0):
        """
        Execute 'steps' physics iterations at a fixed 30 Hz dt.
        Scheduling (deciding how many to run) is done by _physics_steps_due().
        """
        scene = context.scene
        if not self.target_object or steps <= 0:
            return

        if getattr(scene, "view_mode", 'THIRD') == 'LOCKED':
            from ..physics.exp_locked_view import run_locked_view
            run_locked_view(self, context, steps)
            return
        
        # Cache addon prefs once (lazy init), then reuse every call
        prefs = getattr(self, "_prefs", None)
        if prefs is None:
            prefs = context.preferences.addons["Exploratory"].preferences
            self._prefs = prefs

        # Bind hot attributes to locals (fewer attribute lookups in the loop)
        pc = self.physics_controller
        dt = self.physics_dt
        yaw = self.yaw
        static_bvh = self.bvh_tree
        dyn_objects_map = getattr(self, "dynamic_objects_map", None)
        v_lin_map = getattr(self, "platform_linear_velocity_map", {})

        # Resolve movement keys once for this batch
        resolved_keys = _resolved_move_keys(self)

        # Fixed 30 Hz steps
        for _ in range(int(steps)):
            # Increment physics frame counter (for logging correlation)
            self._physics_frame += 1

            pc.try_consume_jump()

            pc.step(
                dt=dt,
                prefs=prefs,
                keys_pressed=resolved_keys,
                camera_yaw=yaw,
                static_bvh=static_bvh,
                dynamic_map=dyn_objects_map,
                platform_linear_velocity_map=v_lin_map,
                engine=getattr(self, 'engine', None),  # Pass engine for offloading
                context=context,  # Pass context for debug output
                physics_frame=self._physics_frame,  # Pass frame number for correlation
            )

            # Keep existing flags for animations/audio
            self.z_velocity        = pc.vel.z
            self.is_grounded       = pc.on_ground
            self.grounded_platform = pc.ground_obj

        # THIRD-PERSON ONLY:
        # Rotate character toward camera once per frame *only* in third person.
        # In FIRST person this fights the FPV path and causes sway/jitter.
        if (
            getattr(scene, "view_mode", 'THIRD') != 'FIRST'
            and pc
            and (pc.vel.x * pc.vel.x + pc.vel.y * pc.vel.y) > 1e-4
        ):
            smooth_rotate_towards_camera(self)

    # ---------------------------
    # Event Handlers
    # ---------------------------
    def handle_key_input(self, event):
        scene = bpy.context.scene
        mg = scene.mobility_game

        # 1) Interact key => edge-triggered (only fires once per press, not on hold)
        if event.type == self.pref_interact_key:
            if event.value == 'PRESS':
                if getattr(self, '_interact_key_released', True):
                    set_interact_pressed(True)
                    self._interact_key_released = False
            elif event.value == 'RELEASE':
                set_interact_pressed(False)
                self._interact_key_released = True

        # 2) Action key => edge-triggered (only fires once per press, not on hold)
        if event.type == self.pref_action_key:
            if event.value == 'PRESS':
                if getattr(self, '_action_key_released', True):
                    self.action_pressed = True
                    set_action_pressed(True)
                    self._action_key_released = False
            elif event.value == 'RELEASE':
                self.action_pressed = False
                set_action_pressed(False)
                self._action_key_released = True

        # 2.5) Reset key
        if event.type == self.pref_reset_key and event.value == 'PRESS':
            bpy.ops.exploratory.reset_game('INVOKE_DEFAULT')
            return

        # 3) Movement lock
        if event.type in {self.pref_forward_key, self.pref_backward_key, self.pref_left_key, self.pref_right_key}:
            if not mg.allow_movement:
                return

        # 4) Sprint lock
        if event.type == self.pref_run_key and not mg.allow_sprint:
            return

        # 5) Jump: **edge-triggered** and buffer-driven
        if event.type == self.pref_jump_key:
            if not mg.allow_jump:
                return

            if event.value == 'PRESS':
                # Edge only: arm the jump buffer once per press
                if self.jump_key_released and self.physics_controller:
                    self.physics_controller.request_jump()
                self.jump_key_released = False

            elif event.value == 'RELEASE':
                self.jump_key_released = True

        # 6) Maintain keys_pressed set AND last-pressed-per-axis
        if event.value == 'PRESS':
            self.keys_pressed.add(event.type)

            ax = _axis_of_key(self, event.type)
            if ax:
                self._press_seq += 1
                self._axis_candidates[ax][event.type] = self._press_seq
                self._axis_last[ax] = event.type  # last-pressed wins on that axis

        elif event.value == 'RELEASE':
            self.keys_pressed.discard(event.type)
            _update_axis_resolution_on_release(self, event.type)

    # ---------------------------
    # Engine Synchronization Methods
    # ---------------------------
    def submit_engine_job(self, job_type: str, data: dict) -> int:
        """
        Submit a job to the engine with frame tagging for sync tracking.
        Returns job_id or -1 if submission failed.
        """
        # PERFORMANCE: Direct access - class-level default is None
        if not self.engine:
            return -1

        job_id = self.engine.submit_job(job_type, data)

        if job_id is not None and job_id >= 0:
            # Track job submission with frame number and timestamp
            self._pending_jobs[job_id] = {
                "frame": self._physics_frame,
                "timestamp": time.perf_counter()
            }
            self._sync_jobs_submitted += 1

        return job_id if job_id is not None else -1

    def process_engine_result(self, result):
        """
        Process a single engine result and track latency metrics.
        Returns True if result was successfully processed.
        """
        if result.job_id not in self._pending_jobs:
            return False

        # Calculate latencies
        job_info = self._pending_jobs[result.job_id]
        frame_latency = self._physics_frame - job_info["frame"]
        time_latency_ms = (time.perf_counter() - job_info["timestamp"]) * 1000.0

        # Track metrics in global stats
        self._sync_frame_latencies.append(frame_latency)
        self._sync_time_latencies.append(time_latency_ms)
        self._sync_results_received += 1

        # Track metrics in test manager (per-scenario)
        # Test manager removed - use standalone stress test operators

        # Clean up
        del self._pending_jobs[result.job_id]

        # Evict orphaned jobs older than 5 seconds (worker crash / lost result)
        now = time.perf_counter()
        stale_ids = [
            jid for jid, info in self._pending_jobs.items()
            if now - info["timestamp"] > 5.0
        ]
        for jid in stale_ids:
            del self._pending_jobs[jid]

        if frame_latency > 2:
            from ..developer.dev_logger import log_game
            log_game("ENGINE_SYNC", f"Stale result - Frame latency: {frame_latency} frames ({time_latency_ms:.1f}ms)")

        return True
    