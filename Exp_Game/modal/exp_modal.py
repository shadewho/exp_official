# File: exp_modal.py
########################################
import bpy
import math
import time
from ..physics.exp_raycastutils import create_bvh_tree
from ..startup_and_reset.exp_spawn import spawn_user
from .exp_view_helpers import (_update_axis_resolution_on_release, _resolved_move_keys, _axis_of_key,
                                smooth_rotate_towards_camera, _maybe_rebind_view3d, _bind_view3d_once)
from ..animations.exp_animations import AnimationStateManager, set_global_animation_manager
from ..audio.exp_audio import (get_global_audio_state_manager, clean_audio_temp)
from ..startup_and_reset.exp_startup import (center_cursor_in_3d_view, clear_old_dynamic_references,
                          record_user_settings, apply_performance_settings, restore_user_settings,
                          move_armature_and_children_to_scene, revert_to_original_scene,
                            ensure_timeline_at_zero, ensure_object_mode, clear_all_exp_custom_strips)  
from ..interactions.exp_interactions import set_interact_pressed, reset_all_interactions
from ..reactions.exp_reactions import reset_all_tasks
from ..props_and_utils.exp_time import init_time, FixedStepClock
from ..reactions.exp_custom_ui import register_ui_draw, clear_all_text, show_controls_info
from ..systems.exp_objectives import reset_all_objectives
from ..startup_and_reset.exp_game_reset import (capture_scene_state, reset_property_reactions, capture_initial_cam_state,
                              restore_initial_session_state, capture_initial_character_state, restore_scene_state)
from ..audio import exp_globals
from ..audio.exp_globals import stop_all_sounds
from ...Exp_UI.download_and_explore.cleanup import cleanup_downloaded_worlds, cleanup_downloaded_datablocks
from ..systems.exp_performance import init_performance_state, restore_performance_state
from ..mouse_and_movement.exp_cursor import (
    setup_cursor_region,
    handle_mouse_move,
    release_cursor_clip,
    ensure_cursor_hidden_if_mac,
)
from ..physics.exp_kcc import KinematicCharacterController
from ..systems.exp_live_performance import init_live_performance
from ..startup_and_reset.exp_fullscreen import exit_fullscreen_once
from ..systems.exp_threads import ThreadEngine
from .exp_loop import GameLoop



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
        cap = int(getattr(self, "max_catchup_steps", 3))

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
    animation_manager = None

    # Keep track of pressed keys
    keys_pressed = set()
    _press_seq: int = 0
    _axis_last = {'x': None, 'y': None}          # selected key per axis
    _axis_candidates = {'x': {}, 'y': {}}        # held keys → press order

    # Timer handle for the modal operator
    _timer = None

    #physics controller and time
    physics_controller = None
    fixed_clock = None


    # store the user-chosen keys from preferences only
    pref_forward_key:  str = "W"
    pref_backward_key: str = "S"
    pref_left_key:     str = "A"
    pref_right_key:    str = "D"
    pref_jump_key:     str = "SPACE"
    pref_run_key:      str = "LEFT_SHIFT"
    pref_interact_key: str = "LEFTMOUSE"
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
        if not hasattr(self, "platform_motion_map"):
            self.platform_motion_map = {}
        if not hasattr(self, "platform_delta_map"):
            self.platform_delta_map = {}
        if not hasattr(self, "platform_prev_positions"):
            self.platform_prev_positions = {}
        if not hasattr(self, "platform_prev_matrices"):
            self.platform_prev_matrices = {}
        if not hasattr(self, "moving_meshes"):
            self.moving_meshes = []
        if not hasattr(self, "grounded_platform"):
            self.grounded_platform = None

    _initial_game_state = {}



    #Threading intialization#
    _thread_eng = None
    _frame_seq: int = 0
    _cam_allowed_last: float = 3.0 

    #loop initialization
    _loop = None


    # ---------------------------
    # Main Modal Functions
    # ---------------------------
    def invoke(self, context, event):
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


        # Capture both camera/character state
        capture_initial_cam_state(self, context)

        #set UI mode to disable background caching
        context.scene.ui_current_mode = 'GAME'

        #clear UI if any
        try:
            # EXEC_DEFAULT so it doesn’t pop up a dialog
            bpy.ops.view3d.remove_package_display('EXEC_DEFAULT')
        except Exception as e:
            print(f"[WARN] remove_package_display failed: {e}")

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

        result = bpy.ops.exploratory.build_character('EXEC_DEFAULT')
        if 'FINISHED' not in result:
            self.report({'ERROR'}, "BuildCharacter operator did not finish. Cannot proceed.")
            return {'CANCELLED'}

        # --- CRITICAL CHECK ---
        if not context.scene.target_armature:
            self.report({'ERROR'}, "No armature assigned to scene.target_armature! Cancelling game.")
            return {'CANCELLED'}
        
        #--clear custom action strips--#
        if not self.launched_from_ui:
            try:
                clear_all_exp_custom_strips()
            except Exception as e:
                print(f"[WARN] clear_all_exp_custom_strips failed: {e}")

        # A) Clean out old audio temp (skip when audio-lock is ON)
        if not context.scene.character_audio_lock:
            clean_audio_temp()

        # C) Now build audio
        audio_result = bpy.ops.exploratory.build_audio('EXEC_DEFAULT')

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

        # ── Initialize cull state ─────────────────────────────
        init_performance_state(self, context)

        #4D Game time
        init_time()  # Start the game clock at 0

        # show control hints for 10 seconds
        show_controls_info(10.0)

        # 4E) *** RESET ALL INTERACTIONS AND TASKS***
        reset_all_interactions(context.scene)
        reset_all_tasks()
        reset_all_objectives(context.scene)
        reset_property_reactions(context.scene)   


        # 5) Center the cursor for reading mouse deltas
        center_cursor_in_3d_view(context)

        # 6) Grab target object from scene property
        self.target_object = context.scene.target_armature
        if self.target_object:
            # Initialize camera pitch & yaw from the current view
            self.pitch = math.radians(context.scene.pitch_angle)
            view_rotation = context.space_data.region_3d.view_rotation.to_euler()
            self.yaw = view_rotation.z

            # Set up the animation manager
            self.animation_manager = AnimationStateManager()
            set_global_animation_manager(self.animation_manager)
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
        setup_cursor_region(context, self)

        # 8) Create a timer (runs as fast as possible, interval=0.0)
        self._timer = context.window_manager.event_timer_add(1.0/30.0, window=context.window)
        self._last_time = time.perf_counter()

        # 9) Add ourselves to Blender’s modal event loop
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

        #define dynamic platforms (vertical movement)
        self.platform_prev_positions = {}
        for dyn_obj in self.moving_meshes:
            if dyn_obj:
                # Store its initial position
                self.platform_prev_positions[dyn_obj] = dyn_obj.matrix_world.translation.copy()

        self.platform_prev_matrices = {}
        for dyn_obj in self.moving_meshes:
            if dyn_obj:
                self.platform_prev_matrices[dyn_obj] = dyn_obj.matrix_world.copy()

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

        
        # Live performance overlay helper
        init_live_performance(self)

        # Threading engine 
        self._thread_eng = ThreadEngine(max_workers=2)

        #game loop initialization
        self._loop = GameLoop(self)
        
        _bind_view3d_once(self, context)

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # End game hotkey
        if event.type == self.pref_end_game_key and event.value == 'PRESS':
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            if self._loop:
                self._loop.on_timer(context)
            return {'RUNNING_MODAL'}

        # Key events
        elif event.type in {
            self.pref_forward_key, self.pref_backward_key, self.pref_left_key,
            self.pref_right_key,  self.pref_run_key,       self.pref_jump_key,
            self.pref_interact_key, self.pref_reset_key,
        }:
            # Option A: delegate to loop (keeps modal slim)
            if self._loop:
                self._loop.handle_key_input(event)
            else:
                self.handle_key_input(event)

        # Mouse -> camera rotation
        elif event.type == 'MOUSEMOVE':
            handle_mouse_move(self, context, event)
            ensure_cursor_hidden_if_mac(context)

        return {'RUNNING_MODAL'}


    def cancel(self, context):
        
        # Restore the cursor modal state
        release_cursor_clip()
        context.window.cursor_modal_restore()

        #if in fullscreen, exit
        exit_fullscreen_once()

        #restore UI mode
        context.scene.ui_current_mode = 'BROWSE'
        
        # Remove the modal timer if it exists
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        
        # Clear any pressed keys
        self.keys_pressed.clear()
        
        # Restore scene state and user preferences
        restore_user_settings(context.scene)
        stop_all_sounds()
        get_global_audio_state_manager().stop_current_sound()

        # Restore proxy mesh visibility
        for entry in context.scene.proxy_meshes:
            if entry.hide_during_game and entry.mesh_object:
                original = self._proxy_mesh_original_states.get(entry.mesh_object.name)
                if original is not None:
                    entry.mesh_object.hide_viewport = original

        #-----------------------------------------------
        ### Cleanup scene and temporary blend file ###
        revert_to_original_scene(context)   #1
        cleanup_downloaded_worlds()         #2
        cleanup_downloaded_datablocks()     #3
        #-----------------------------------------------

        clear_all_text()


        #reset --------------------reset --------#
        if not self.launched_from_ui:
            restore_scene_state(self, context)

        if not self.launched_from_ui:
            bpy.ops.exploratory.reset_game(
                'INVOKE_DEFAULT',
                skip_restore=True
            )
        if not self.launched_from_ui:
            restore_performance_state(self, context)


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
                        bpy.ops.view3d.add_package_display('INVOKE_REGION_WIN')
                        bpy.ops.view3d.popup_social_details('INVOKE_REGION_WIN')
                else:
                    print("No valid VIEW3D context found for UI popups.")
                return None  # Stop the timer

            # Register the timer to delay UI calls.
            bpy.app.timers.register(delayed_ui_popups, first_interval=0.5)
    
        #----------------------------
        #loop and threading shutdown
        #----------------------------
        if self._loop:
            try:
                self._loop.shutdown()
            except Exception:
                pass
        if self._thread_eng:
            try: self._thread_eng.shutdown()
            except Exception: pass
            self._thread_eng = None

        return {'CANCELLED'}


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
        if not self.target_object or steps <= 0:
            return

        prefs = context.preferences.addons["Exploratory"].preferences

        # We resolve movement keys once and reuse inside the loop
        resolved_keys = None

        for _ in range(int(steps)):
            # One fixed 30 Hz step (independent of time_scale)
            self.physics_controller.try_consume_jump()

            # Compute once and cache for this batch if not done yet
            if resolved_keys is None:
                resolved_keys = _resolved_move_keys(self)

            self.physics_controller.step(
                dt=self.physics_dt,
                prefs=prefs,
                keys_pressed=resolved_keys,
                camera_yaw=self.yaw,
                static_bvh=self.bvh_tree,
                dynamic_map=getattr(self, "dynamic_bvh_map", None),
                platform_linear_velocity_map=getattr(self, "platform_linear_velocity_map", {}),
                platform_ang_velocity_map=getattr(self, "platform_ang_velocity_map", {}),
                platform_motion_map=getattr(self, "platform_motion_map", {}),
            )

            # Keep existing flags for animations/audio
            self.z_velocity        = self.physics_controller.vel.z
            self.is_grounded       = self.physics_controller.on_ground
            self.grounded_platform = self.physics_controller.ground_obj

        # Rotate character toward camera ONCE per frame if actually moving
        if self.physics_controller and (self.physics_controller.vel.x**2 + self.physics_controller.vel.y**2) > 1e-6:
            smooth_rotate_towards_camera(self)



    # ---------------------------
    # Event Handlers
    # ---------------------------
    def handle_key_input(self, event):
        scene = bpy.context.scene
        mg = scene.mobility_game

        # 1) Interact key => set/clear
        if event.type == self.pref_interact_key:
            if event.value == 'PRESS':
                set_interact_pressed(True)
            elif event.value == 'RELEASE':
                set_interact_pressed(False)

        # 2) Reset key
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