# File: exp_modal.py
########################################
import bpy
import math
import time
from .props_and_utils.exp_utilities import get_game_world
from .physics.exp_raycastutils import create_bvh_tree
from .startup_and_reset.exp_spawn import spawn_user
from .physics.exp_view import update_view, shortest_angle_diff
from .animations.exp_animations import AnimationStateManager, set_global_animation_manager
from .audio.exp_audio import (get_global_audio_state_manager, clear_temp_sounds, 
                        get_global_audio_manager, clean_audio_temp)
from .startup_and_reset.exp_startup import (center_cursor_in_3d_view, clear_old_dynamic_references,
                          record_user_settings, apply_performance_settings, restore_user_settings,
                          move_armature_and_children_to_scene,
                            revert_to_original_workspace, revert_to_original_scene,
                            ensure_timeline_at_zero, ensure_object_mode)  
from .animations.exp_custom_animations import update_all_custom_managers
from .interactions.exp_interactions import check_interactions, set_interact_pressed, reset_all_interactions, approximate_bounding_sphere_radius
from .reactions.exp_reactions import update_transform_tasks, update_property_tasks, reset_all_tasks
from .props_and_utils.exp_time import init_time, update_time, get_game_time
from .reactions.exp_custom_ui import register_ui_draw, update_text_reactions, clear_all_text, show_controls_info
from .systems.exp_objectives import update_all_objective_timers, reset_all_objectives
from .startup_and_reset.exp_game_reset import (capture_scene_state, reset_property_reactions, capture_initial_cam_state,
                              restore_initial_session_state, capture_initial_character_state, restore_scene_state)
from .audio import exp_globals
from .audio.exp_globals import stop_all_sounds, update_sound_tasks
from ..Exp_UI.download_and_explore.cleanup import cleanup_downloaded_worlds, cleanup_downloaded_datablocks
from .systems.exp_performance import init_performance_state, update_performance_culling, restore_performance_state
from .mouse_and_movement.exp_cursor import setup_cursor_region, handle_mouse_move, release_cursor_clip, ensure_cursor_hidden_if_mac
from .physics.exp_kcc import KinematicCharacterController
from .props_and_utils.exp_time import FixedStepClock
from .physics.exp_dynamic import update_dynamic_meshes
from .systems.exp_live_performance import (
    init_live_performance,
    perf_frame_begin,
    perf_mark,
    perf_frame_end,
)

class ExpModal(bpy.types.Operator):
    """Windowed (minimized) game start."""

    bl_idname = "view3d.exp_modal"
    bl_label = "Third Person Orbit"
    bl_options = {'BLOCKING', 'GRAB_CURSOR'}

    # ---------------------------
    #  Properties
    # ---------------------------
    speed: bpy.props.FloatProperty(
        name="Speed",
        default=2.0,
        min=1.0,
        max=3.0,
        description="Character movement speed"
    )
    time_scale: bpy.props.IntProperty(
        name="Time Scale",
        default=3,           # keep your current feel
        min=1,
        max=5,
        description="Controls how quickly time passes in the simulation (integer)"
    )
    camera_distance: bpy.props.FloatProperty(
        name="Camera Distance",
        default=5.0,
        min=1.0,
        max=20.0,
        description="Default camera distance from the character"
    )

    launched_from_ui: bpy.props.BoolProperty(
        name="Launched from UI",
        default=False,
    )
    should_revert_workspace: bpy.props.BoolProperty(
        name="Revert Workspace on Cancel",
        description="If false, do not revert the workspace when canceling the modal",
        default=True
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

    # ---------------------------
    # Main Modal Functions
    # ---------------------------
    def invoke(self, context, event):
        scene = context.scene

        # ─── Capture both camera/character state ───────
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

        performance_level = addon_prefs.performance_level  # e.g., "LOW", "MEDIUM", "HIGH", "CUSTOM"
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

        # We do not allow catch-up bursts; exactly one step per 33.333 ms.
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

        # Initialize dictionary to record original visibility states
        self._proxy_mesh_original_states = {}
        for entry in context.scene.proxy_meshes:
            if entry.hide_during_game and entry.mesh_object:
                # Record the original visibility (using hide_viewport)
                self._proxy_mesh_original_states[entry.mesh_object.name] = entry.mesh_object.hide_viewport
                # Hide the proxy mesh during the game
                entry.mesh_object.hide_viewport = True

        # 11) Clear any previously pressed keys
        self.keys_pressed.clear()
        
        # Live performance overlay helper
        init_live_performance(self)

        return {'RUNNING_MODAL'}

    def modal(self, context, event):

        """Called repeatedly by Blender for each event (mouse, keyboard, timer, etc.)."""
        # 1) End game when user presses the assigned end-game key
        if event.type == self.pref_end_game_key and event.value == 'PRESS':
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':

            # ---- START FRAME (for live perf overlay) ----
            perf_frame_begin(self)

            # A) Timebases
            t0 = perf_mark(self, 'time')
            self.update_time()
            _ = update_time()  # keep your global game clock increasing
            perf_mark(self, 'time', t0)

            # Decide once: should we do a 30 Hz physics tick this frame?
            do_phys = self._physics_due()

            # B) Animation & audio (pure state) - per frame, uses scaled delta_time
            t0 = perf_mark(self, 'anim_audio')
            self.animation_manager.update(self.keys_pressed, self.delta_time, self.is_grounded)
            current_anim_state = self.animation_manager.anim_state
            audio_state_mgr = get_global_audio_state_manager()
            audio_state_mgr.update_audio_state(current_anim_state)
            perf_mark(self, 'anim_audio', t0)

            # C) FIRST: anything that moves objects this frame (treat as "physical")
            if do_phys:
                t0 = perf_mark(self, 'custom_tasks')
                update_all_custom_managers(self.delta_time)  # your content-driven movers
                update_transform_tasks()
                update_property_tasks()
                perf_mark(self, 'custom_tasks', t0)

                # D) Dynamic proxies/BVHs + platform velocities (feeds KCC)
                t0 = perf_mark(self, 'dynamic_meshes')
                update_dynamic_meshes(self)
                perf_mark(self, 'dynamic_meshes', t0)

                # E) Distance-based culling (can be heavy; do on physics tick)
                t0 = perf_mark(self, 'culling')
                update_performance_culling(self, context)
                perf_mark(self, 'culling', t0)

            # --- Character gating / input filtering (unchanged) ---
            mg = context.scene.mobility_game
            if not mg.allow_movement:
                for k in (self.pref_forward_key, self.pref_backward_key,
                          self.pref_left_key, self.pref_right_key):
                    if k in self.keys_pressed:
                        self.keys_pressed.remove(k)
            if not mg.allow_sprint:
                if self.pref_run_key in self.keys_pressed:
                    self.keys_pressed.remove(self.pref_run_key)

            # F) Movement & gravity (fixed 30 Hz physics; camera updates inside)
            t0 = perf_mark(self, 'physics')
            self.update_movement_and_gravity(context)  # will step only if due
            perf_mark(self, 'physics', t0)

            # H) Interactions/UI/SFX (per frame)
            t0 = perf_mark(self, 'interact_ui_audio')
            check_interactions(context)
            update_text_reactions()
            update_sound_tasks()
            perf_mark(self, 'interact_ui_audio', t0)

            # ---- END FRAME ----
            perf_frame_end(self, context)

        # B) Key events => only user preference keys (forward/back/left/right/run/jump).
        elif event.type in {
            self.pref_forward_key,
            self.pref_backward_key,
            self.pref_left_key,
            self.pref_right_key,
            self.pref_run_key,
            self.pref_jump_key,
            self.pref_interact_key,
            self.pref_reset_key,
        }:
            self.handle_key_input(event)

        # D) Mouse move => camera rotation
        elif event.type == 'MOUSEMOVE':
            handle_mouse_move(self, context, event)
            
            #Hide cursor on macOS
            ensure_cursor_hidden_if_mac(context)

        return {'RUNNING_MODAL'}


    def cancel(self, context):
        
        # Restore the cursor modal state
        release_cursor_clip()
        context.window.cursor_modal_restore()

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

        # Revert to the original workspace if the flag is True
        if self.should_revert_workspace:
            revert_to_original_workspace(context)


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

    def update_animations(self):
        """Send scaled delta_time to the animation manager."""
        if self.animation_manager:
            self.animation_manager.update(self.keys_pressed, self.delta_time, self.is_grounded, self.z_velocity)

    def update_movement_and_gravity(self, context):
        """
        HARD-LOCK physics to 30 Hz (real time). Never runs faster than 30 Hz:
        - We execute at most one KCC step per 33.333 ms wall-clock.
        - We DO NOT "catch up" with multiple steps if the UI stalls.
        - Animations/audio remain per-frame on scaled delta_time.
        """
        if not self.target_object:
            return

        # Only step exactly on (or after) the scheduled 30 Hz tick
        now = time.perf_counter()
        if now < self._next_physics_tick:
            return  # not time yet; keep physics frozen this frame

        prefs = context.preferences.addons["Exploratory"].preferences

        # One step at constant dt (independent of time_scale)
        self.physics_controller.try_consume_jump()
        self.physics_controller.step(
            dt=self.physics_dt,
            prefs=prefs,
            keys_pressed=self.keys_pressed,
            camera_yaw=self.yaw,
            static_bvh=self.bvh_tree,
            dynamic_map=getattr(self, "dynamic_bvh_map", None),
            platform_linear_velocity_map=getattr(self, "platform_linear_velocity_map", {}),
            platform_ang_velocity_map=getattr(self, "platform_ang_velocity_map", {}),
        )

        # Keep old flags for compatibility with animations/audio
        self.z_velocity = self.physics_controller.vel.z
        self.is_grounded = self.physics_controller.on_ground
        self.grounded_platform = self.physics_controller.ground_obj

        # Camera update at physics rate (30 Hz)
        update_view(
            context,
            self.target_object,
            self.pitch,
            self.yaw,
            self.bvh_tree,
            context.scene.orbit_distance,
            context.scene.zoom_factor,
            dynamic_bvh_map=getattr(self, "dynamic_bvh_map", None)
        )

        # Rotate character toward camera if moving
        self.smooth_rotate_towards_camera()

        # Schedule the next 30 Hz tick WITHOUT catch-up bursts
        self._next_physics_tick += self.physics_dt
        if now >= self._next_physics_tick:
            # If we fell behind, drop missed steps and re-align a period from now.
            self._next_physics_tick = now + self.physics_dt


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

        # 6) Maintain keys_pressed set for animations & movement
        if event.value == 'PRESS':
            self.keys_pressed.add(event.type)
        elif event.value == 'RELEASE':
            self.keys_pressed.discard(event.type)

    def handle_jump(self, event):
        """Manage jump start, cooldown, etc."""
        if (event.value == 'PRESS'
            and self.is_grounded
            and not self.is_jumping
            and not self.jump_cooldown
            and self.jump_key_released):
            self.is_jumping = True
            self.z_velocity = 7.0
            self.jump_cooldown = True
            self.jump_key_released = False

        elif event.value == 'RELEASE':
            self.jump_key_released = True

        # If already jumping, apply gravity
        if self.is_jumping:
            self.z_velocity -= self.gravity * self.delta_time
            # If velocity dips below 0, and we're not too high => reset jump
            if (self.z_velocity <= 0
                and self.target_object
                and self.target_object.location.z <= 2.0):
                self.is_jumping = False
                self.z_velocity = 0.0

    # ---------------------------
    # Utility / Helper Methods
    # ---------------------------
    def smooth_rotate_towards_camera(self):
        if not self.target_object:
            return

        # Only rotate if movement keys are pressed.
        pressed_keys = {
            self.pref_forward_key,
            self.pref_backward_key,
            self.pref_left_key,
            self.pref_right_key,
        }
        actual_pressed = self.keys_pressed.intersection(pressed_keys)
        if not actual_pressed:
            return

        desired_yaw = self.determine_desired_yaw(actual_pressed)
        current_yaw = self.target_object.rotation_euler.z
        # Use the helper to get the shortest difference.
        yaw_diff = shortest_angle_diff(current_yaw, desired_yaw)
        if abs(yaw_diff) > 0.001:
            self.target_object.rotation_euler.z += yaw_diff * self.rotation_smoothness


    def determine_desired_yaw(self, actual_pressed):
        """Calculate desired yaw based on user-chosen forward/back/left/right keys + camera yaw."""
        base_yaw = self.yaw

        # Check combos: forward+right => ~45°, forward+left => -45°, etc.
        # We'll unify the user-chosen keys to match typical movement combos
        # Or any logic you wish to implement:
        if (self.pref_forward_key in actual_pressed and self.pref_right_key in actual_pressed):
            return base_yaw - math.radians(45)
        if (self.pref_forward_key in actual_pressed and self.pref_left_key in actual_pressed):
            return base_yaw + math.radians(45)
        if (self.pref_backward_key in actual_pressed and self.pref_right_key in actual_pressed):
            return base_yaw - math.radians(135)
        if (self.pref_backward_key in actual_pressed and self.pref_left_key in actual_pressed):
            return base_yaw + math.radians(135)

        if self.pref_forward_key in actual_pressed:
            return base_yaw
        if self.pref_backward_key in actual_pressed:
            return base_yaw + math.pi
        if self.pref_left_key in actual_pressed:
            return base_yaw + (math.pi / 2)
        if self.pref_right_key in actual_pressed:
            return base_yaw - (math.pi / 2)

        # fallback
        return base_yaw