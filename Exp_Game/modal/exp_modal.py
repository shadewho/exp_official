# File: exp_modal.py
import bpy
import math
import time
import random
import sys
import io

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
from ..interactions.exp_interactions import set_interact_pressed, set_action_pressed, reset_all_interactions
from ..reactions.exp_reactions import reset_all_tasks
from ..props_and_utils.exp_time import init_time
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
from ..startup_and_reset.exp_fullscreen import exit_fullscreen_once
from .exp_loop import GameLoop
from ..physics.exp_view_fpv import reset_fpv_rot_scale
from ..engine import EngineCore

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


# ============================================================================
# ENGINE SYNC TEST MANAGER
# Comprehensive testing system for engine-modal synchronization under load
# ============================================================================

class EngineSyncTestManager:
    """
    Manages comprehensive stress testing of the multiprocessing engine.
    Tests engine synchronization with 30Hz modal under realistic game loads.
    """

    def __init__(self):
        # Test scenarios - each tests different load characteristics
        self.scenarios = [
            {
                "name": "BASELINE",
                "duration_frames": 60,  # 2 seconds at 30Hz
                "jobs_per_frame": 1,
                "compute_ms_min": 0.0,
                "compute_ms_max": 0.1,
                "description": "Lightweight test - verify zero-latency capability",
                "target_grade": "A",
            },
            {
                "name": "LIGHT_LOAD",
                "duration_frames": 90,  # 3 seconds
                "jobs_per_frame": 5,
                "compute_ms_min": 1.0,
                "compute_ms_max": 3.0,
                "description": "Normal gameplay - 5 AI agents pathfinding",
                "target_grade": "A",
            },
            {
                "name": "MEDIUM_LOAD",
                "duration_frames": 90,  # 3 seconds
                "jobs_per_frame": 15,
                "compute_ms_min": 2.0,
                "compute_ms_max": 5.0,
                "description": "Busy gameplay - 10 AI + 5 physics predictions",
                "target_grade": "B",
            },
            {
                "name": "HEAVY_LOAD",
                "duration_frames": 90,  # 3 seconds
                "jobs_per_frame": 30,
                "compute_ms_min": 3.0,
                "compute_ms_max": 8.0,
                "description": "Worst case - 20 AI + 10 physics + batch operations",
                "target_grade": "B",
            },
            {
                "name": "BURST_TEST",
                "duration_frames": 90,  # 3 seconds
                "jobs_per_frame": 5,  # Normal load
                "burst_every_n_frames": 30,  # Every second
                "burst_job_count": 50,
                "compute_ms_min": 2.0,
                "compute_ms_max": 5.0,
                "description": "Burst stress - simulate enemy spawns/explosions",
                "target_grade": "B",
            },
            {
                "name": "CATCHUP_STRESS",
                "duration_frames": 120,  # 4 seconds
                "jobs_per_frame": 10,
                "compute_ms_min": 2.0,
                "compute_ms_max": 5.0,
                "force_delays": True,  # Trigger catchup frames
                "delay_every_n_frames": 15,  # Every 0.5 seconds
                "delay_duration_ms": 100,  # 100ms delay = guaranteed catchup
                "description": "Catchup stress - force modal inconsistency and verify sync",
                "target_grade": "B",
            },
        ]

        # Current test state
        self.current_scenario_index = 0
        self.scenario_start_frame = 0
        self.total_test_frames = sum(s["duration_frames"] for s in self.scenarios)

        # Per-scenario metrics
        self.scenario_metrics = []
        for scenario in self.scenarios:
            self.scenario_metrics.append({
                "name": scenario["name"],
                "jobs_submitted": 0,
                "results_received": 0,
                "frame_latencies": [],
                "time_latencies": [],
                "start_frame": 0,
                "end_frame": 0,
                # Catchup tracking
                "catchup_events": 0,  # Times we had 2+ physics steps in one timer event
                "total_catchup_steps": 0,  # Total extra steps (steps - 1)
                "max_catchup": 0,  # Max steps in single event
                "frame_attribution_errors": 0,  # Results arriving at wrong frame
            })

        # Test start time
        self.test_start_time = time.perf_counter()

        print("\n" + "="*70)
        print("ENGINE SYNC STRESS TEST - STARTING")
        print("="*70)
        print(f"Total duration: {self.total_test_frames} frames ({self.total_test_frames/30:.1f}s at 30Hz)")
        print(f"Scenarios: {len(self.scenarios)}")
        for i, scenario in enumerate(self.scenarios):
            print(f"  [{i+1}] {scenario['name']}: {scenario['duration_frames']}f - {scenario['description']}")
        print("="*70 + "\n")

    def get_current_scenario(self):
        """Get the currently active test scenario."""
        if self.current_scenario_index >= len(self.scenarios):
            return None
        return self.scenarios[self.current_scenario_index]

    def should_advance_scenario(self, current_frame):
        """Check if we should move to the next scenario."""
        scenario = self.get_current_scenario()
        if not scenario:
            return False

        frames_in_scenario = current_frame - self.scenario_start_frame
        return frames_in_scenario >= scenario["duration_frames"]

    def advance_scenario(self, current_frame):
        """Move to the next test scenario."""
        if self.current_scenario_index < len(self.scenarios):
            # Close out current scenario metrics
            self.scenario_metrics[self.current_scenario_index]["end_frame"] = current_frame

            # Advance to next scenario
            self.current_scenario_index += 1
            self.scenario_start_frame = current_frame

            if self.current_scenario_index < len(self.scenarios):
                scenario = self.scenarios[self.current_scenario_index]
                self.scenario_metrics[self.current_scenario_index]["start_frame"] = current_frame

                print(f"\n{'='*70}")
                print(f"SCENARIO {self.current_scenario_index + 1}/{len(self.scenarios)}: {scenario['name']}")
                print(f"{'='*70}")
                print(f"Description: {scenario['description']}")
                print(f"Duration: {scenario['duration_frames']} frames")
                print(f"Jobs/frame: {scenario['jobs_per_frame']}")
                print(f"Compute: {scenario['compute_ms_min']}-{scenario['compute_ms_max']}ms")
                if "burst_every_n_frames" in scenario:
                    print(f"Bursts: {scenario['burst_job_count']} jobs every {scenario['burst_every_n_frames']} frames")
                print(f"Target: Grade {scenario['target_grade']}")
                print(f"{'='*70}\n")

    def generate_jobs_for_frame(self, current_frame, operator, physics_steps=1):
        """Generate and submit jobs based on current scenario."""
        scenario = self.get_current_scenario()
        if not scenario:
            return  # Test complete

        # Check if we should advance to next scenario
        if self.should_advance_scenario(current_frame):
            self.advance_scenario(current_frame)
            scenario = self.get_current_scenario()
            if not scenario:
                return

        frames_in_scenario = current_frame - self.scenario_start_frame
        jobs_to_submit = scenario["jobs_per_frame"]

        # CATCHUP TRACKING: Record if this is a catchup event
        if physics_steps > 1:
            metrics = self.scenario_metrics[self.current_scenario_index]
            metrics["catchup_events"] += 1
            metrics["total_catchup_steps"] += (physics_steps - 1)
            metrics["max_catchup"] = max(metrics["max_catchup"], physics_steps)
            print(f"[CATCHUP] Frame {current_frame}: {physics_steps} physics steps in one timer event")

        # Check for burst
        if "burst_every_n_frames" in scenario:
            if frames_in_scenario > 0 and frames_in_scenario % scenario["burst_every_n_frames"] == 0:
                jobs_to_submit += scenario["burst_job_count"]
                print(f"[BURST] Frame {current_frame}: Submitting {scenario['burst_job_count']} additional jobs")

        # Check for forced delay (to trigger catchup frames)
        if scenario.get("force_delays", False):
            delay_interval = scenario.get("delay_every_n_frames", 15)
            if frames_in_scenario > 0 and frames_in_scenario % delay_interval == 0:
                delay_ms = scenario.get("delay_duration_ms", 100)
                print(f"[DELAY] Frame {current_frame}: Injecting {delay_ms}ms delay to force catchup")
                time.sleep(delay_ms / 1000.0)

        # Submit jobs
        for i in range(jobs_to_submit):
            # Generate realistic compute time
            compute_ms = random.uniform(scenario["compute_ms_min"], scenario["compute_ms_max"])

            job_id = operator.submit_engine_job("COMPUTE_HEAVY", {
                "iterations": int(compute_ms * 10),  # Rough calibration for 1-10ms jobs
                "data": list(range(50)),
                "frame": current_frame,
                "scenario": scenario["name"],
            })

            if job_id >= 0:
                self.scenario_metrics[self.current_scenario_index]["jobs_submitted"] += 1

    def record_result(self, result, frame_latency, time_latency_ms):
        """Record metrics for a completed job."""
        # Find which scenario this job belongs to
        scenario_name = result.result.get("scenario", "UNKNOWN") if result.success else "UNKNOWN"

        for i, metrics in enumerate(self.scenario_metrics):
            if metrics["name"] == scenario_name:
                metrics["results_received"] += 1
                metrics["frame_latencies"].append(frame_latency)
                metrics["time_latencies"].append(time_latency_ms)
                break

    def is_complete(self):
        """Check if all test scenarios are complete."""
        return self.current_scenario_index >= len(self.scenarios)

    def calculate_grade(self, avg_latency, max_latency, stale_pct):
        """Calculate grade based on metrics."""
        if avg_latency <= 1.0 and max_latency <= 2 and stale_pct < 5.0:
            return "A"
        elif avg_latency <= 1.5 and max_latency <= 3 and stale_pct < 10.0:
            return "B"
        else:
            return "F"

    def print_final_report(self):
        """Print comprehensive test results."""
        test_duration = time.perf_counter() - self.test_start_time

        print("\n" + "="*70)
        print("ENGINE SYNC STRESS TEST - FINAL REPORT")
        print("="*70)
        print(f"Total test duration: {test_duration:.1f}s")
        print(f"Scenarios completed: {len(self.scenarios)}")
        print("="*70)

        all_passed = True

        for i, (scenario, metrics) in enumerate(zip(self.scenarios, self.scenario_metrics)):
            print(f"\n{'─'*70}")
            print(f"SCENARIO {i+1}: {scenario['name']}")
            print(f"{'─'*70}")
            print(f"Description: {scenario['description']}")
            print(f"Target Grade: {scenario['target_grade']}")
            print(f"Duration: {metrics['end_frame'] - metrics['start_frame']} frames")
            print(f"\nMetrics:")
            print(f"  Jobs Submitted:    {metrics['jobs_submitted']}")
            print(f"  Results Received:  {metrics['results_received']}")
            print(f"  Pending/Lost:      {metrics['jobs_submitted'] - metrics['results_received']}")

            if metrics['frame_latencies']:
                avg_lat = sum(metrics['frame_latencies']) / len(metrics['frame_latencies'])
                max_lat = max(metrics['frame_latencies'])
                min_lat = min(metrics['frame_latencies'])
                stale_count = sum(1 for x in metrics['frame_latencies'] if x > 2)
                stale_pct = (stale_count / len(metrics['frame_latencies'])) * 100

                print(f"\nFrame Latency:")
                print(f"  Average:           {avg_lat:.2f} frames")
                print(f"  Min:               {min_lat} frames")
                print(f"  Max:               {max_lat} frames")
                print(f"  Stale (>2 frames): {stale_count} ({stale_pct:.1f}%)")

                if metrics['time_latencies']:
                    avg_time = sum(metrics['time_latencies']) / len(metrics['time_latencies'])
                    max_time = max(metrics['time_latencies'])
                    min_time = min(metrics['time_latencies'])

                    print(f"\nTime Latency:")
                    print(f"  Average:           {avg_time:.2f}ms")
                    print(f"  Min:               {min_time:.2f}ms")
                    print(f"  Max:               {max_time:.2f}ms")

                # Catchup statistics
                if metrics["catchup_events"] > 0:
                    print(f"\nCatchup Events:")
                    print(f"  Total Events:      {metrics['catchup_events']}")
                    print(f"  Extra Steps:       {metrics['total_catchup_steps']}")
                    print(f"  Max Steps/Event:   {metrics['max_catchup']}")
                    total_frames = metrics['end_frame'] - metrics['start_frame']
                    if total_frames > 0:
                        catchup_pct = (metrics["catchup_events"] / total_frames) * 100
                        print(f"  Catchup Rate:      {catchup_pct:.1f}% of timer events")

                # Calculate grade
                actual_grade = self.calculate_grade(avg_lat, max_lat, stale_pct)
                target_grade = scenario['target_grade']

                # Determine pass/fail
                grade_order = {"A": 3, "B": 2, "F": 1}
                passed = grade_order.get(actual_grade, 0) >= grade_order.get(target_grade, 0)

                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"\nResult: Grade {actual_grade} (Target: {target_grade}) - {status}")

                if not passed:
                    all_passed = False
            else:
                print("\n✗ FAIL - No results received!")
                all_passed = False

        # Overall verdict
        print(f"\n{'='*70}")
        if all_passed:
            print("OVERALL VERDICT: ✓ PASS - ENGINE IS PRODUCTION READY")
            print("The engine can handle realistic game loads while maintaining sync.")
        else:
            print("OVERALL VERDICT: ✗ FAIL - ENGINE NEEDS IMPROVEMENT")
            print("The engine cannot maintain acceptable sync under load.")
            print("\nPossible issues:")
            print("  - Queue size too small (increase JOB_QUEUE_SIZE)")
            print("  - Too few workers (increase WORKER_COUNT)")
            print("  - Jobs too computationally heavy (optimize algorithms)")
            print("  - System CPU overloaded (close background apps)")
        print("="*70 + "\n")

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


    # ---------------------------
    # Main Modal Functions
    # ---------------------------
    def invoke(self, context, event):
        global _active_modal_operator
        _active_modal_operator = self

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
        # Engine must be 100% ready before modal starts.
        # Philosophy: Engine = orchestrator, Modal = puppet
        # If engine fails, game cannot start. Period.
        # ═══════════════════════════════════════════════════════════════════

        startup_logs = context.scene.dev_startup_logs

        if startup_logs:
            print("\n" + "="*70)
            print("  GAME STARTUP SEQUENCE - ENGINE FIRST")
            print("="*70)

        # ─── STEP 1: Spawn Engine ───
        if startup_logs:
            print("\n[STARTUP 1/5] Spawning engine workers...")

        if not hasattr(self, 'engine'):
            self.engine = EngineCore()

        self.engine.start()

        if startup_logs:
            stats = self.engine.get_stats()
            print(f"[STARTUP 1/5] ✓ Engine started")
            print(f"              Workers: {stats['workers_alive']}/{stats['workers_total']} spawned")

        # ─── STEP 2: Verify Workers Alive ───
        if startup_logs:
            print(f"\n[STARTUP 2/5] Verifying workers alive...")

        if not self.engine.is_alive():
            error_msg = "Engine workers failed to spawn"
            if startup_logs:
                print(f"[STARTUP 2/5] ✗ FAILED: {error_msg}")
                print("="*70 + "\n")
            self.report({'ERROR'}, f"{error_msg} - aborting game")
            self.engine.shutdown()
            return {'CANCELLED'}

        if startup_logs:
            print(f"[STARTUP 2/5] ✓ All workers alive and running")

        # ─── STEP 3: PING Verification (Comprehensive Readiness) ───
        if startup_logs:
            print(f"\n[STARTUP 3/5] Verifying worker responsiveness (PING check)...")

        if not self.engine.wait_for_readiness(timeout=5.0):
            error_msg = "Engine workers not responding to PING"
            if startup_logs:
                print(f"[STARTUP 3/5] ✗ FAILED: {error_msg}")
                print("="*70 + "\n")
            self.report({'ERROR'}, f"{error_msg} - aborting game")
            self.engine.shutdown()
            return {'CANCELLED'}

        if startup_logs:
            print(f"[STARTUP 3/5] ✓ All workers responding to PING")

        # Initialize engine sync tracking
        self._physics_frame = 0
        self._pending_jobs = {}
        self._sync_jobs_submitted = 0
        self._sync_results_received = 0
        self._sync_frame_latencies = []
        self._sync_time_latencies = []
        self._sync_last_report_frame = 0

        # Note: EngineSyncTestManager removed - use standalone stress test operators instead
        # (Developer Tools → Manual Stress Tests)
        # Those test the ENGINE CORE in isolation, not engine+modal integration

        # ─── STEP 4: Cache Spatial Grid (Physics Requirement) ───
        if startup_logs:
            print(f"\n[STARTUP 4/5] Caching spatial grid in workers...")

        if self.spatial_grid and self.engine and self.engine.is_alive():
            import pickle
            import time as time_module

            # Measure grid serialization (one-time cost)
            pickle_start = time_module.perf_counter()
            pickled = pickle.dumps({"grid": self.spatial_grid})
            pickle_time = (time_module.perf_counter() - pickle_start) * 1000
            pickle_size_kb = len(pickled) / 1024

            if startup_logs:
                print(f"[STARTUP 4/5] Grid size: {pickle_size_kb:.1f} KB")
                print(f"              Serialization time: {pickle_time:.1f}ms")

            # Send CACHE_GRID jobs to ensure all workers receive the cache
            # Submit 8 jobs (2x worker count) to increase probability each worker gets one
            # Workers share a job queue, so we need extra jobs to ensure coverage
            # Duplicate cache jobs are safe - workers just overwrite _cached_grid
            for i in range(8):
                self.engine.submit_job("CACHE_GRID", {"grid": self.spatial_grid})

            if startup_logs:
                print(f"[STARTUP 4/5] Grid jobs submitted, waiting for all workers to confirm...")

            # CRITICAL: Wait for all unique workers to confirm grid receipt
            # This now tracks which workers confirmed, not just result count
            if not self.engine.verify_grid_cache(timeout=5.0):
                error_msg = "Not all workers cached spatial grid (see console for details)"
                if startup_logs:
                    print(f"[STARTUP 4/5] ✗ FAILED: {error_msg}")
                    print("="*70 + "\n")
                self.report({'ERROR'}, f"{error_msg} - aborting game")
                self.engine.shutdown()
                return {'CANCELLED'}

            if startup_logs:
                print(f"[STARTUP 4/5] ✓ Grid successfully cached in all workers")
        else:
            if startup_logs:
                print(f"[STARTUP 4/5] ⊘ No spatial grid (skipped)")

        # ─── STEP 5: Final Readiness Confirmation (Lock-Step Gate) ───
        if startup_logs:
            print(f"\n[STARTUP 5/5] Final readiness check (lock-step synchronization)...")

        final_status = self.engine.get_full_readiness_status(grid_required=bool(self.spatial_grid))

        if not final_status["ready"]:
            error_msg = final_status["message"]
            if startup_logs:
                print(f"[STARTUP 5/5] ✗ FAILED: {error_msg}")
                print(f"\n              Checks:")
                for check_name, passed in final_status["checks"].items():
                    status = "✓" if passed else "✗"
                    print(f"              {status} {check_name}")
                if final_status["details"]["critical"]:
                    print(f"\n              Critical Issues:")
                    for issue in final_status["details"]["critical"]:
                        print(f"              • {issue}")
                if final_status["details"]["warnings"]:
                    print(f"\n              Warnings:")
                    for warning in final_status["details"]["warnings"]:
                        print(f"              • {warning}")
                print("="*70 + "\n")
            self.report({'ERROR'}, f"Engine not ready: {error_msg} - aborting game")
            self.engine.shutdown()
            return {'CANCELLED'}

        if startup_logs:
            print(f"[STARTUP 5/5] ✓ {final_status['message']}")
            print(f"\n              All Checks Passed:")
            for check_name, passed in final_status["checks"].items():
                print(f"              ✓ {check_name}")
            print(f"\n" + "="*70)
            print(f"  ENGINE READY - MODAL STARTING")
            print("="*70 + "\n")

        # ═══════════════════════════════════════════════════════════════════
        # ENGINE CONFIRMED READY - MODAL CAN NOW INITIALIZE
        # ═══════════════════════════════════════════════════════════════════

        # Initialize interaction offload tracking
        self._pending_interaction_job_id = None  # Track pending INTERACTION_CHECK_BATCH job
        self._interaction_map = []  # Maps worker indices to scene interactions

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
            # Update yaw/pitch only — keep camera/FPV on the TIMER clock.
            handle_mouse_move(self, context, event)
            ensure_cursor_hidden_if_mac(context)

        return {'RUNNING_MODAL'}


    def cancel(self, context):

        # ========== ENGINE SHUTDOWN ==========
        # Shutdown multiprocessing engine gracefully
        if hasattr(self, 'engine') and self.engine:
            if context.scene.dev_debug_engine:
                print("[ExpModal] Shutting down multiprocessing engine...")

            # Print comprehensive test report if test manager is active
            # Test manager removed - use standalone stress test operators
            elif context.scene.dev_debug_engine:
                # Only print sync report if engine debug is enabled
                self._print_sync_report()

            self.engine.shutdown()
            self.engine = None
            if context.scene.dev_debug_engine:
                print("[ExpModal] Engine shutdown complete")
        # ====================================

        # ========== KCC VISUALIZATION CLEANUP ==========
        # Remove GPU draw handlers for KCC visualization
        if hasattr(self, 'physics_controller') and self.physics_controller:
            self.physics_controller.cleanup_debug_handlers()
        # ====================================

        # ========== DIAGNOSTICS LOG EXPORT ==========
        # Export fast buffer logger diagnostics if enabled
        from ..developer.dev_logger import export_game_log, clear_log, get_buffer_size
        if context.scene.dev_export_session_log and get_buffer_size() > 0:
            export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
            clear_log()
        # ====================================

        # Restore the cursor modal state
        release_cursor_clip()
        context.window.cursor_modal_restore()

        #if in fullscreen, exit
        exit_fullscreen_once()

        #restore UI mode
        context.scene.ui_current_mode = 'GAME'
        
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
                    print("No valid VIEW3D context found for UI popups.")
                return None

            # Register the timer to delay UI calls.
            bpy.app.timers.register(delayed_ui_popups, first_interval=0.5)

        # Clear global modal operator reference
        global _active_modal_operator
        _active_modal_operator = None

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
        v_ang_map = getattr(self, "platform_ang_velocity_map", {})

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
                platform_ang_velocity_map=v_ang_map,
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

        # 1) Interact key => set/clear
        if event.type == self.pref_interact_key:
            if event.value == 'PRESS':
                set_interact_pressed(True)
            elif event.value == 'RELEASE':
                set_interact_pressed(False)

        # 2) Action key => set/clear (global flag)
        if event.type == self.pref_action_key:
            if event.value == 'PRESS':
                self.action_pressed = True  # local (optional for you)
                set_action_pressed(True)    # global → interactions system
            elif event.value == 'RELEASE':
                self.action_pressed = False
                set_action_pressed(False)

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
        if not hasattr(self, 'engine') or not self.engine:
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

        # Optional: Print latency for debugging (can be disabled in production)
        if frame_latency > 2:
            print(f"[EngineSync] WARNING: Stale result - Frame latency: {frame_latency} frames ({time_latency_ms:.1f}ms)")

        return True

    def _print_sync_report(self):
        """Print comprehensive synchronization statistics."""
        if self._sync_jobs_submitted == 0:
            print("\n[EngineSync] No jobs submitted during session")
            return

        print("\n" + "="*60)
        print("ENGINE SYNCHRONIZATION REPORT")
        print("="*60)
        print(f"Total Physics Frames:     {self._physics_frame}")
        print(f"Jobs Submitted:           {self._sync_jobs_submitted}")
        print(f"Results Received:         {self._sync_results_received}")
        print(f"Pending Jobs:             {len(self._pending_jobs)}")

        if self._sync_frame_latencies:
            avg_frame_lat = sum(self._sync_frame_latencies) / len(self._sync_frame_latencies)
            max_frame_lat = max(self._sync_frame_latencies)
            stale_count = sum(1 for x in self._sync_frame_latencies if x > 2)
            stale_pct = (stale_count / len(self._sync_frame_latencies)) * 100

            print(f"\nFrame Latency:")
            print(f"  Average:                {avg_frame_lat:.2f} frames")
            print(f"  Maximum:                {max_frame_lat} frames")
            print(f"  Stale (>2 frames):      {stale_count} ({stale_pct:.1f}%)")

        if self._sync_time_latencies:
            avg_time_lat = sum(self._sync_time_latencies) / len(self._sync_time_latencies)
            max_time_lat = max(self._sync_time_latencies)
            min_time_lat = min(self._sync_time_latencies)

            print(f"\nTime Latency:")
            print(f"  Average:                {avg_time_lat:.2f}ms")
            print(f"  Min:                    {min_time_lat:.2f}ms")
            print(f"  Max:                    {max_time_lat:.2f}ms")

        # Grade the synchronization quality
        if self._sync_frame_latencies:
            if avg_frame_lat <= 1.0 and max_frame_lat <= 2 and stale_pct < 5:
                grade = "A (Excellent)"
            elif avg_frame_lat <= 1.5 and max_frame_lat <= 3 and stale_pct < 10:
                grade = "B (Good)"
            else:
                grade = "F (Needs Improvement)"

            print(f"\nSync Quality Grade:       {grade}")

        print("="*60 + "\n")