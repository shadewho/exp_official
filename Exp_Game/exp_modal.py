# File: exp_modal.py
########################################
import os
import bpy
import math
import time
from .exp_movement import move_character
from .exp_raycastutils import create_bvh_tree
from .exp_spawn import spawn_user
from .exp_view import update_view
from .exp_animations import AnimationStateManager, set_global_animation_manager
from .exp_audio import (get_global_audio_state_manager, clear_temp_sounds, 
                        get_global_audio_manager, clean_audio_temp)
from .exp_startup import (center_cursor_in_3d_view, clear_old_dynamic_references,
                          record_user_settings, apply_performance_settings, restore_user_settings,
                          move_armature_and_children_to_scene,
                            revert_to_original_workspace)  
from ..exp_preferences import ExploratoryAddonPreferences, get_addon_path
from .exp_custom_animations import update_all_custom_managers
from .exp_interactions import check_interactions, set_interact_pressed, reset_all_interactions, approximate_bounding_sphere_radius
from .exp_reactions import update_transform_tasks, update_property_tasks, reset_all_tasks
from .exp_time import init_time, update_time, get_game_time
from .exp_custom_ui import register_ui_draw, update_text_reactions, clear_all_text
from .exp_objectives import update_all_objective_timers, reset_all_objectives
from .exp_game_reset import capture_scene_state
from . import exp_globals
from .exp_globals import stop_all_sounds, update_sound_tasks
from ..Exp_UI.helper_functions import cleanup_downloaded_worlds

class ExpModal(bpy.types.Operator):
    """A modal operator that controls third-person movement and camera,
    using a consistent time scale for movement and animation."""

    bl_idname = "view3d.exp_modal"
    bl_label = "Third Person Orbit"
    bl_options = {'GRAB_CURSOR', 'BLOCKING'}

    # ---------------------------
    # User-Adjustable Properties
    # ---------------------------
    speed: bpy.props.FloatProperty(
        name="Speed",
        default=2.0,
        min=1.0,
        max=3.0,
        description="Character movement speed"
    )
    time_scale: bpy.props.FloatProperty(
        name="Time Scale",
        default=3.0,
        min=1.0,
        max=5.0,
        description="Controls how quickly time passes in the simulation"
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
    # ---------------------------
    # Internal State Variables
    # ---------------------------
    pitch: bpy.props.FloatProperty(name="Pitch", default=0.0)
    yaw: bpy.props.FloatProperty(name="Yaw", default=0.0)
    z_velocity: bpy.props.FloatProperty(name="Z Velocity", default=0.0)
    gravity: float = -9.8

    delta_time: float = 0.0  # Updated each frame based on real time * time_scale
    _last_time: float = 0.0  # Used to calculate real time between frames

    is_grounded: bool = False
    is_jumping: bool = False
    jump_timer: float = 0.0
    jump_duration: float = 0.5
    jump_cooldown: bool = False
    jump_key_released: bool = True

    # Rotation smoothing for character turning
    rotation_smoothness: float = 0.35

    # References to objects / data
    target_object = None
    bvh_tree = None
    animation_manager = None

    # Keep track of pressed keys
    keys_pressed = set()

    # Timer handle for the modal operator
    _timer = None

    # (New) We'll store the user-chosen keys from preferences only
    pref_forward_key:  str = "W"
    pref_backward_key: str = "S"
    pref_left_key:     str = "A"
    pref_right_key:    str = "D"
    pref_jump_key:     str = "SPACE"
    pref_run_key:      str = "LEFT_SHIFT"

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
        addon_prefs = context.preferences.addons["Exploratory"].preferences

        #----------------------------
        #Scene state and preference setup
        #----------------------------
        record_user_settings(scene)
        capture_scene_state(self,context)

        performance_level = addon_prefs.performance_level  # e.g., "LOW", "MEDIUM", "HIGH", "CUSTOM"
        apply_performance_settings(scene, performance_level)

        result = bpy.ops.exploratory.build_character('EXEC_DEFAULT')
        if 'FINISHED' not in result:
            self.report({'ERROR'}, "BuildCharacter operator did not finish. Cannot proceed.")
            return {'CANCELLED'}

        # --- CRITICAL CHECK ---
        if not context.scene.target_armature:
            self.report({'ERROR'}, "No armature assigned to scene.target_armature! Cancelling game.")
            return {'CANCELLED'}
        

        #---------------------
        #reset audio temp folder
        #---------------------
        # A) Clean out old temp audio
        clean_audio_temp()

        # B) Build character
        result = bpy.ops.exploratory.build_character('EXEC_DEFAULT')
        if 'FINISHED' not in result:
            ...

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
        self.pref_reset_key = addon_prefs.key_reset

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

        #4D Game time
        init_time()  # Start the game clock at 0

        # 4E) *** RESET ALL INTERACTIONS AND TASKS***
        reset_all_interactions(context.scene)
        reset_all_tasks()
        reset_all_objectives(context.scene)   


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


        # 7) Hide the cursor
        context.window.cursor_modal_set('NONE')

        # 8) Create a timer (runs as fast as possible, interval=0.0)
        self._timer = context.window_manager.event_timer_add(0.01, window=context.window)
        self._last_time = time.time()

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

        # 11) Clear any previously pressed keys
        self.keys_pressed.clear()

        return {'RUNNING_MODAL'}

    def modal(self, context, event):

        """Called repeatedly by Blender for each event (mouse, keyboard, timer, etc.)."""
        # A) Possibly cancel with Right Mouse or ESC
        
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            
            # A) Update time so self.delta_time is now the current frame’s dt
            self.update_time()
            delta = update_time()  # We get how many real seconds passed
            

            # B) Update the animation manager once
            self.animation_manager.update(self.keys_pressed, self.delta_time, self.is_grounded)
            current_anim_state = self.animation_manager.anim_state

            # C) Update audio
            audio_state_mgr = get_global_audio_state_manager()
            audio_state_mgr.update_audio_state(current_anim_state)

            # C.1) Dynamic Mesh Updates with Caching and Mesh-Specific Collision Size
            if not hasattr(self, "cached_dynamic_bvhs"):
                self.cached_dynamic_bvhs = {}
            self.dynamic_bvh_map = {}
            MATRIX_THRESHOLD = 1e-3  # Adjust as needed to detect significant movement
            for dyn_obj in self.moving_meshes:
                if dyn_obj and dyn_obj.type == 'MESH':
                    current_matrix = dyn_obj.matrix_world.copy()
                    cached = self.cached_dynamic_bvhs.get(dyn_obj)
                    if cached is not None:
                        last_matrix, cached_bvh, dyn_radius = cached
                        # If the object's transform hasn’t changed significantly, reuse the BVH and radius.
                        if (current_matrix.to_translation() - last_matrix.to_translation()).length < MATRIX_THRESHOLD:
                            self.dynamic_bvh_map[dyn_obj] = (cached_bvh, dyn_radius)
                            continue
                    # Otherwise, rebuild the BVH for this dynamic mesh
                    dyn_bvh = create_bvh_tree([dyn_obj])
                    if dyn_bvh:
                        # Compute the collision size using the mesh's actual geometry
                        dyn_radius = approximate_bounding_sphere_radius(dyn_obj)
                        self.dynamic_bvh_map[dyn_obj] = (dyn_bvh, dyn_radius)
                        self.cached_dynamic_bvhs[dyn_obj] = (current_matrix, dyn_bvh, dyn_radius)

            # C.2) Vertical Dictionary Positional Update
            for dyn_obj in self.moving_meshes:
                old_pos = self.platform_prev_positions.get(dyn_obj, None)
                new_pos = dyn_obj.matrix_world.translation.copy()  # current center
                if old_pos is not None:
                    # Calculate the object's lateral (sideways) motion vector.
                    motion_vec = new_pos - old_pos
                    if not hasattr(self, 'platform_motion_map'):
                        self.platform_motion_map = {}
                    self.platform_motion_map[dyn_obj] = motion_vec
                # Update stored position for next frame
                self.platform_prev_positions[dyn_obj] = new_pos

            # C.3) Update Platform Delta Map
            self.platform_delta_map = {}
            for dyn_obj in self.moving_meshes:
                old_mat = self.platform_prev_matrices[dyn_obj]
                new_mat = dyn_obj.matrix_world.copy()
                delta_mat = new_mat @ old_mat.inverted()
                self.platform_delta_map[dyn_obj] = delta_mat
                self.platform_prev_matrices[dyn_obj] = new_mat

            # C.4) Apply Platform Delta if Grounded
            self.apply_platform_delta_if_grounded()


            # update movement real time (holding keys)
            mg = context.scene.mobility_game
            
            if not mg.allow_movement:
                for k in (self.pref_forward_key, self.pref_backward_key,
                        self.pref_left_key, self.pref_right_key):
                    if k in self.keys_pressed:
                        self.keys_pressed.remove(k)

            # 2) If sprint is disallowed, remove SHIFT
            if not mg.allow_sprint:
                if self.pref_run_key in self.keys_pressed:
                    self.keys_pressed.remove(self.pref_run_key)

            # D) Movement, jumping, etc.
            self.update_movement_and_gravity(context)
            self.update_jumping(context)

            # E) Check interactions
            check_interactions(context)

            # F) update custom actions, transform tasks, etc.
            update_all_custom_managers(self.delta_time)
            update_transform_tasks()
            update_property_tasks()

            # G) Update the UI messages
            update_text_reactions()

            #H Sound reaction tasks
            update_sound_tasks()


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
            self.handle_mouse_move(context, event)

        return {'RUNNING_MODAL'}
    
    # ---------------------------
    # rotation delta
    # ---------------------------
    def apply_platform_delta_if_grounded(self):
        if not self.target_object:
            return

        best_obj = getattr(self, 'grounded_platform', None)
        if not best_obj:
            return

        delta_mat = self.platform_delta_map.get(best_obj)
        if not delta_mat:
            return

        loc = self.target_object.location
        self.target_object.location = delta_mat @ loc

    def cancel(self, context):
        """Cleanup code when user cancels with ESC or Right-click."""
        context.window.cursor_modal_restore()
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        self.keys_pressed.clear()

        #-------------------------------------
        #reset scene state and user preferences
        #--------------------------------------
        restore_user_settings(context.scene)
        stop_all_sounds()

        #Cleanup scene and temp blend file
        cleanup_downloaded_worlds()

        revert_to_original_workspace(context)
        
        # Only re-open the UI if we launched from the UI button.
        if self.launched_from_ui:
            bpy.ops.view3d.add_package_display('INVOKE_DEFAULT')

        if self.launched_from_ui:
            bpy.ops.view3d.popup_social_details('INVOKE_DEFAULT')
        
        return {'CANCELLED'}


    # ---------------------------
    # Update / Logic Methods
    # ---------------------------
    def update_time(self):
        """Update delta_time based on real system clock, times our time_scale."""
        current_time = time.time()
        real_dt = current_time - self._last_time
        self._last_time = current_time
        self.delta_time = real_dt * self.time_scale

    def update_animations(self):
        """Send scaled delta_time to the animation manager."""
        if self.animation_manager:
            self.animation_manager.update(self.keys_pressed, self.delta_time, self.is_grounded)

    def update_movement_and_gravity(self, context):
        """Apply character movement & gravity, using the user’s scaled dt."""
        if not self.target_object:
            return

         # Move
        self.z_velocity, self.is_grounded = move_character(
            op=self,
            target_object=self.target_object,
            keys_pressed=self.keys_pressed,
            bvh_tree=self.bvh_tree,
            delta_time=self.delta_time,
            speed=self.speed,
            gravity=self.gravity,
            z_velocity=self.z_velocity,
            jump_timer=self.jump_timer,
            is_jumping=self.is_jumping,
            is_grounded=self.is_grounded,
            jump_duration=self.jump_duration,
            sensitivity=self.sensitivity,
            pitch=self.pitch,
            yaw=self.yaw,
            context=context,
            dynamic_bvh_map=self.dynamic_bvh_map,
            platform_motion_map=self.platform_motion_map
        )


        # Update camera
        update_view(
            context,
            self.target_object,
            self.pitch,
            self.yaw,
            self.bvh_tree,
            self.camera_distance,
            context.scene.zoom_factor
        )

        # Possibly rotate character to face yaw
        self.smooth_rotate_towards_camera()

    def update_jumping(self, context):
        """Handle jump logic (timer, cooldown)."""
        # If currently in a jump => increment jump_timer
        if self.is_jumping:
            self.jump_timer += self.delta_time
            if self.jump_timer >= self.jump_duration:
                self.is_jumping = False
                self.jump_timer = 0.0
                self.jump_cooldown = True

        # Once grounded => reset jump cooldown
        if self.jump_cooldown and not self.is_jumping and self.is_grounded:
            self.jump_cooldown = False


    # ---------------------------
    # Event Handlers
    # ---------------------------
    def handle_key_input(self, event):
        scene = bpy.context.scene
        mg = scene.mobility_game

        # 1) If user pressed Interact Key => call set_interact_pressed
        if event.type == self.pref_interact_key:
            if event.value == 'PRESS':
                set_interact_pressed(True)
            elif event.value == 'RELEASE':
                set_interact_pressed(False)
        #reset key
        if event.type == self.pref_reset_key:
            if event.value == 'PRESS':
                bpy.ops.exploratory.reset_game('INVOKE_DEFAULT')
                return

        # 2) Movement lock logic
        if event.type in {self.pref_forward_key, self.pref_backward_key, self.pref_left_key, self.pref_right_key}:
            if not mg.allow_movement:
                return  # skip these

        # 3) Sprint lock logic
        if event.type == self.pref_run_key:
            if not mg.allow_sprint:
                return

        # 4) Jump lock logic
        if event.type == self.pref_jump_key:
            if not mg.allow_jump:
                return
            if event.value == 'PRESS':
                if self.is_grounded and not self.is_jumping and not self.jump_cooldown:
                    self.is_jumping = True
                    self.z_velocity = 7.0
                    self.jump_key_released = False
            elif event.value == 'RELEASE':
                self.jump_key_released = True


        # Finally, add or remove from self.keys_pressed as before
        if event.value == 'PRESS':
            self.keys_pressed.add(event.type)
        elif event.value == 'RELEASE':
            self.keys_pressed.discard(event.type)

    def handle_mouse_move(self, context, event):
        # Try to get the region from the context
        region = context.region
        # If it's not available, search for the 3D View 'WINDOW' region manually
        if region is None:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for reg in area.regions:
                        if reg.type == 'WINDOW':
                            region = reg
                            break
                    if region:
                        break
        # If still not found, exit the function
        if region is None:
            return

        region_x, region_y = region.x, region.y
        width, height = region.width, region.height

        mouse_abs_x = event.mouse_x
        mouse_abs_y = event.mouse_y

        center_x = region_x + (width // 2)
        center_y = region_y + (height // 2)

        delta_x = mouse_abs_x - center_x
        delta_y = mouse_abs_y - center_y

        self.yaw -= delta_x * self.sensitivity
        self.pitch -= delta_y * self.sensitivity

        # Clamp pitch to avoid flipping
        self.pitch = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, self.pitch))

        # Warp the cursor to keep it centered
        context.window.cursor_warp(center_x, center_y)


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
        """Smoothly rotate the character to match the camera yaw if user-chosen movement keys are pressed."""
        if not self.target_object:
            return

        # Only rotate if user-chosen forward/back/left/right or run are pressed
        # (i.e. if user is actually trying to move)
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
        current_z = self.target_object.rotation_euler.z
        yaw_diff = desired_yaw - current_z
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