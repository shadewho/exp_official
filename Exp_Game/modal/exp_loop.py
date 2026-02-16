#Exp_Game/modal/exp_loop.py

import time
import bpy
from ..props_and_utils.exp_time import update_real_time, tick_sim_time, get_game_time
from ..animations.state_machine import AnimState
from .exp_engine_bridge import (
    update_animations_state,
    submit_animation_jobs,
    process_animation_result,
    poll_animation_results_with_timeout,
    get_cached_anim_other_results,
)
from ..reactions.exp_reactions import update_property_tasks
from ..reactions.exp_projectiles import (
    update_hitscan_tasks,  # Just despawns visual clones
    submit_hitscan_batch,
    submit_projectile_update,
    process_hitscan_results,
    process_projectile_results,
    interpolate_projectile_visuals,
    init_impact_cache,
    reset_impact_prop_writers,
)
from ..reactions.exp_transforms import (
    update_transform_tasks,
    submit_transform_batch,
    apply_transform_results,
    poll_transform_result_with_timeout,
    get_cached_other_results as get_cached_transform_results,
)
from ..reactions.exp_tracking import (
    update_tracking_tasks,
    submit_tracking_batch,
    apply_tracking_results,
)
from ..physics.exp_dynamic import update_dynamic_meshes
from ..physics.exp_view import (
    update_camera_for_operator,
    submit_camera_occlusion_early,
    poll_camera_result_with_timeout,
    cache_camera_worker_result,
)
from ..interactions.exp_interactions import check_interactions, apply_interaction_check_result, init_aabb_cache, init_reaction_node_cache
from ..interactions.exp_tracker_eval import (
    set_current_operator,
    submit_tracker_evaluation,
    process_tracker_result,
    cache_trackers_in_worker,
)
from ..reactions.exp_bindings import serialize_reaction_bindings
from ..reactions.exp_custom_ui import update_text_reactions
from ..reactions.exp_parenting import process_pending_parenting, clear_pending_parenting
from ..audio.exp_globals import update_sound_tasks
from ..audio.exp_audio import get_global_audio_state_manager
from ..developer.dev_stats import get_stats_tracker
from ..animations.blend_system import (
    init_blend_system,
    shutdown_blend_system,
    get_blend_system,
)


# ============================================================================
# RESULT DISPATCH HANDLERS - Separated for O(1) dict lookup instead of if/elif
# ============================================================================

def _handle_echo_result(op, result, stats):
    """Handle ECHO test result - no action needed."""
    pass


def _handle_camera_occlusion_result(op, result, stats):
    """Handle CAMERA_OCCLUSION_FULL result."""
    try:
        from ..developer.dev_logger import log_game
        op_key = id(op)
        job_id = result.job_id
        hit = result.result.get("hit", False)
        hit_distance = result.result.get("hit_distance", None)
        hit_source = result.result.get("hit_source", "STATIC")
        calc_time_us = result.result.get("calc_time_us", 0.0)
        static_tris = result.result.get("static_triangles_tested", 0)
        static_cells = result.result.get("static_cells_traversed", 0)
        dynamic_tris = result.result.get("dynamic_triangles_tested", 0)

        cache_camera_worker_result(op_key, job_id, hit, hit_distance, hit_source, calc_time_us)

        # Record stats for summary
        stats.record_camera_ray(calc_time_us=calc_time_us, hit=hit)

        # Check if grid is cached in worker (diagnostic)
        grid_cached = result.result.get("grid_cached", True)
        if not grid_cached:
            log_game("CAMERA", "WARNING: Grid not cached in worker! Static geometry will not be tested.")

        # Log unified camera result (uses master Hz control)
        dynamic_meshes = result.result.get("dynamic_meshes_tested", 0)
        method = result.result.get("method", "CAMERA_UNIFIED")
        if hit:
            log_game("CAMERA", f"HIT({hit_source}) dist={hit_distance:.3f}m | static_tris={static_tris} cells={static_cells} dyn_meshes={dynamic_meshes} | {calc_time_us:.0f}us [{method}]")
        else:
            log_game("CAMERA", f"MISS | static_tris={static_tris} cells={static_cells} dyn_meshes={dynamic_meshes} | {calc_time_us:.0f}us [{method}]")

    except Exception as e:
        print(f"[GameLoop] Error caching camera occlusion result: {e}")


def _handle_interaction_check_result(op, result, stats):
    """Handle INTERACTION_CHECK_BATCH result."""
    try:
        apply_interaction_check_result(result, bpy.context)
    except Exception as e:
        print(f"[GameLoop] Error applying interaction check result: {e}")
        import traceback
        traceback.print_exc()
        # CRITICAL: Clear pending job on error to prevent infinite loop
        op._pending_interaction_job_id = None


def _handle_compute_heavy_result(op, result, stats):
    """Handle COMPUTE_HEAVY stress test - no action needed."""
    pass


def _handle_kcc_physics_result(op, result, stats):
    """Handle KCC_PHYSICS_STEP result (late results - main processing is same-frame)."""
    try:
        from ..developer.dev_logger import log_game
        # Extract debug data for stats
        debug = result.result.get("debug", {})
        calc_time = debug.get("calc_time_us", 0.0)
        rays = debug.get("rays_cast", 0)
        tris = debug.get("triangles_tested", 0)
        h_blocked = debug.get("h_blocked", False)
        did_step = debug.get("did_step_up", False)
        did_slide = debug.get("did_slide", False)
        hit_ceiling = debug.get("hit_ceiling", False)

        # Record stats for summary
        stats.record_kcc_step(
            calc_time_us=calc_time,
            rays=rays,
            tris=tris,
            blocked=h_blocked,
            step_up=did_step,
            slide=did_slide,
            ceiling=hit_ceiling
        )

        # Engine health logging
        dynamic_active = debug.get("dynamic_meshes_active", 0)
        dynamic_xform_time = debug.get("dynamic_transform_time_us", 0.0)
        if dynamic_active > 0:
            log_game("ENGINE",
                f"MAIN: worker_time={calc_time:.0f}µs (xform={dynamic_xform_time:.0f}µs) "
                f"rays={rays} tris={tris} dyn={dynamic_active}")

    except Exception as e:
        print(f"[GameLoop] Error processing KCC stats: {e}")


def _handle_cache_grid_result(op, result, stats):
    """Handle CACHE_GRID result."""
    from ..developer.dev_debug_gate import should_print_debug
    if should_print_debug("kcc_physics"):
        if result.result.get("success"):
            tris = result.result.get("triangles", 0)
            cells = result.result.get("cells", 0)
            print(f"[KCC] Grid cached: {tris:,} triangles, {cells:,} cells")
        else:
            print(f"[KCC] Grid cache ERROR: {result.result.get('error', 'Unknown error')}")


def _handle_cache_dynamic_mesh_result(op, result, stats):
    """Handle CACHE_DYNAMIC_MESH result."""
    if result.result.get("success"):
        worker_logs = result.result.get("logs", [])
        if worker_logs:
            from ..developer.dev_logger import log_worker_messages
            log_worker_messages(worker_logs)


def _handle_cache_animations_result(op, result, stats):
    """Handle CACHE_ANIMATIONS result."""
    if result.result.get("success"):
        worker_logs = result.result.get("logs", [])
        if worker_logs:
            from ..developer.dev_logger import log_worker_messages
            log_worker_messages(worker_logs)


def _handle_animation_compute_batch_result(op, result, stats):
    """Handle ANIMATION_COMPUTE_BATCH result - apply computed poses to bpy."""
    try:
        objects_applied = process_animation_result(op, result)
        if objects_applied > 0:
            from ..developer.dev_logger import log_game
            log_game("ANIM", f"Applied poses to {objects_applied} objects")
    except Exception as e:
        print(f"[GameLoop] Error applying animation batch result: {e}")


def _handle_clear_animation_cache_result(op, result, stats):
    """Handle CLEAR_ANIMATION_CACHE result."""
    pass  # No action needed, just acknowledge


def _handle_evaluate_trackers_result(op, result, stats):
    """Handle EVALUATE_TRACKERS result."""
    try:
        process_tracker_result(result)
    except Exception as e:
        print(f"[GameLoop] Error applying tracker result: {e}")


def _handle_cache_trackers_result(op, result, stats):
    """Handle CACHE_TRACKERS result."""
    if result.result:
        count = result.result.get("tracker_count", 0)
        worker_logs = result.result.get("logs", [])
        if worker_logs:
            from ..developer.dev_logger import log_worker_messages
            log_worker_messages(worker_logs)


def _handle_hitscan_batch_result(op, result, stats):
    """Handle HITSCAN_BATCH result."""
    try:
        process_hitscan_results(result)
        # Process worker logs
        worker_logs = result.result.get("logs", [])
        if worker_logs:
            from ..developer.dev_logger import log_worker_messages
            log_worker_messages(worker_logs)
    except Exception as e:
        print(f"[GameLoop] Error processing hitscan results: {e}")


def _handle_projectile_update_result(op, result, stats):
    """Handle PROJECTILE_UPDATE_BATCH result."""
    try:
        process_projectile_results(result)
        # Process worker logs
        worker_logs = result.result.get("logs", [])
        if worker_logs:
            from ..developer.dev_logger import log_worker_messages
            log_worker_messages(worker_logs)
    except Exception as e:
        print(f"[GameLoop] Error processing projectile results: {e}")


def _handle_transform_batch_result(op, result, stats):
    """Handle TRANSFORM_BATCH result."""
    try:
        apply_transform_results(result)
    except Exception as e:
        print(f"[GameLoop] Error applying transform results: {e}")


def _handle_tracking_batch_result(op, result, stats):
    """Handle TRACKING_BATCH result."""
    try:
        apply_tracking_results(result)
    except Exception as e:
        print(f"[GameLoop] Error applying tracking results: {e}")


# Dispatch table for O(1) result routing (replaces if/elif chain)
_RESULT_HANDLERS = {
    "ECHO": _handle_echo_result,
    "CAMERA_OCCLUSION_FULL": _handle_camera_occlusion_result,
    "INTERACTION_CHECK_BATCH": _handle_interaction_check_result,
    "COMPUTE_HEAVY": _handle_compute_heavy_result,
    "KCC_PHYSICS_STEP": _handle_kcc_physics_result,
    "CACHE_GRID": _handle_cache_grid_result,
    "CACHE_DYNAMIC_MESH": _handle_cache_dynamic_mesh_result,
    "CACHE_ANIMATIONS": _handle_cache_animations_result,
    "ANIMATION_COMPUTE_BATCH": _handle_animation_compute_batch_result,
    "CLEAR_ANIMATION_CACHE": _handle_clear_animation_cache_result,
    "EVALUATE_TRACKERS": _handle_evaluate_trackers_result,
    "CACHE_TRACKERS": _handle_cache_trackers_result,
    "HITSCAN_BATCH": _handle_hitscan_batch_result,
    "PROJECTILE_UPDATE_BATCH": _handle_projectile_update_result,
    "TRANSFORM_BATCH": _handle_transform_batch_result,
    "TRACKING_BATCH": _handle_tracking_batch_result,
}


class GameLoop:
    """
    Per-frame/TIMER orchestrator for ExpModal.
    - No bpy writes in workers; only here on the main thread.
    - Reuses operator's state/caches to avoid churn.
    """

    def __init__(self, op):
        self.op = op
        self._last_anim_state = None

        # Summary print timers (for 1Hz stats)
        self._last_engine_summary = time.perf_counter()
        self._last_kcc_summary = time.perf_counter()

        # COLD START DIAGNOSTICS - track first 100 frames to detect bad starts
        self._startup_diag_frames = 100  # How many frames to analyze
        self._startup_diag_frame = 0
        self._startup_diag_intervals = []  # Frame intervals in ms
        self._startup_diag_steps = []  # Physics steps per frame
        self._startup_diag_start = time.perf_counter()
        self._startup_diag_last_time = self._startup_diag_start
        self._startup_diag_logged = False

        # Reset stats tracker on game start
        get_stats_tracker().reset_all()

        # Initialize blend system with animation cache from controller
        # PERFORMANCE: Direct attribute access (class-level default is None)
        anim_cache = op.anim_controller.cache if op.anim_controller else None
        init_blend_system(anim_cache)

        # Cache tracker node graph in workers
        import bpy
        cache_trackers_in_worker(op, bpy.context)

        # Serialize reaction bindings (data node connections)
        serialize_reaction_bindings(bpy.context.scene)

        # Initialize AABB cache for collision interactions (Phase 1.2)
        init_aabb_cache(bpy.context.scene)

        # Initialize impact cache for projectiles/hitscans (eliminates node graph iteration)
        init_impact_cache(bpy.context.scene)
        reset_impact_prop_writers()  # Clear stale impact data from previous session

        # Initialize reaction node cache (eliminates node graph iteration for dynamic inputs)
        init_reaction_node_cache(bpy.context.scene)

    def shutdown(self):
        """Clean up game loop resources (called when game stops)."""
        shutdown_blend_system()
        clear_pending_parenting()

    def _log_startup_diagnostics(self, log_game):
        """
        Log startup timing diagnostics to help identify cold start issues.
        Called after first N frames to compare good vs bad starts.
        """
        if not self._startup_diag_intervals:
            return

        intervals = self._startup_diag_intervals
        steps = self._startup_diag_steps

        # Frame interval stats
        avg_interval = sum(intervals) / len(intervals)
        min_interval = min(intervals)
        max_interval = max(intervals)

        # Count hiccups (frames > 50ms = missed frame)
        hiccups = sum(1 for i in intervals if i > 50)

        # Count fast frames (< 20ms = catching up)
        fast_frames = sum(1 for i in intervals if i < 20)

        # Physics steps histogram
        steps_0 = sum(1 for s in steps if s == 0)
        steps_1 = sum(1 for s in steps if s == 1)
        steps_2 = sum(1 for s in steps if s == 2)
        steps_3 = sum(1 for s in steps if s >= 3)

        # Effective FPS
        total_time = sum(intervals) / 1000  # seconds
        effective_fps = len(intervals) / total_time if total_time > 0 else 0

        # Determine health status
        # Bad signs: many hiccups, low FPS, lots of catchup steps
        is_healthy = hiccups < 5 and effective_fps > 25 and steps_3 < 10
        status = "GOOD" if is_healthy else "COLD_START_DETECTED"

        log_game("STARTUP-DIAG", f"{status} frames={len(intervals)} fps={effective_fps:.1f} avg={avg_interval:.1f}ms min={min_interval:.1f}ms max={max_interval:.1f}ms hiccups={hiccups} fast={fast_frames}")
        log_game("STARTUP-DIAG", f"PHYSICS_STEPS 0={steps_0} 1={steps_1} 2={steps_2} 3+={steps_3}")

        # If cold start detected, log first 10 frame intervals for debugging
        if not is_healthy:
            first_10 = [f"{i:.0f}" for i in intervals[:10]]
            log_game("STARTUP-DIAG", f"FIRST_10_INTERVALS_MS: {' '.join(first_10)}")

    # ---------- Public API ----------

    def on_timer(self, context):
        """
        Runs once per Blender TIMER event.
        Uses bounded catch-up for physics to prevent time dilation under stalls.
        """
        op = self.op

        # Increment frame counter for fast logger
        from ..developer.dev_logger import increment_frame, log_game
        increment_frame()

        # COLD START DIAGNOSTICS - collect timing data for first N frames
        if self._startup_diag_frame < self._startup_diag_frames:
            now = time.perf_counter()
            interval_ms = (now - self._startup_diag_last_time) * 1000
            self._startup_diag_intervals.append(interval_ms)
            self._startup_diag_last_time = now
            self._startup_diag_frame += 1
        elif not self._startup_diag_logged:
            # Log startup diagnostics summary
            self._log_startup_diagnostics(log_game)
            self._startup_diag_logged = True

        # A) Timebases (scaled + wall clock)
        op.update_time()        # scaled per-frame dt (UI / interpolation)
        _ = update_real_time()  # wall-clock (kept for diagnostics/overlays)

        # A2) Local projectile visual interpolation (every frame for smooth motion)
        # This runs before physics ticks to keep visuals smooth between worker updates
        # PERFORMANCE: Direct access - dt has class-level default of 0.016
        frame_dt = op.dt * float(op.time_scale)
        interpolate_projectile_visuals(frame_dt)

        # A3) Hitscan visual cleanup (every frame for immediate despawn)
        update_hitscan_tasks()

        # B) Decide how many 30 Hz physics steps are due (bounded catch-up)
        steps = op._physics_steps_due()
        op._perf_last_physics_steps = steps

        # COLD START DIAGNOSTICS - track physics steps
        if self._startup_diag_frame <= self._startup_diag_frames and len(self._startup_diag_steps) < self._startup_diag_frames:
            self._startup_diag_steps.append(steps)

        if steps:
            # Shared SIM dt per 30 Hz step; aggregated dt for bulk systems
            dt_sim = op.physics_dt * float(op.time_scale)
            agg_dt = dt_sim * steps

            # Advance SIM time by the actual amount we'll simulate this frame
            tick_sim_time(agg_dt)

            # B1) Custom/scripted tasks (SIM-timed): cheap → step-per-step
            for _ in range(steps):
                update_transform_tasks()  # Updates t values only (computation in worker)
                update_property_tasks()
                # NOTE: Projectile physics now handled by worker (PROJECTILE_UPDATE_BATCH)
                # update_projectile_tasks() removed - worker-only mode
                # NOTE: update_hitscan_tasks() moved to A3 (runs every frame)
                update_tracking_tasks(dt_sim)

            # B1a) Submit transform batch to worker (after t values updated)
            engine = getattr(op, "engine", None)
            transform_job_id = None
            if engine and engine.is_alive():
                transform_job_id = submit_transform_batch(engine)

            # B1a2) CRITICAL: Poll transform results BEFORE dynamic meshes/physics!
            # This ensures platform positions are updated before KCC reads them.
            # Without this, characters sink/shift on moving platforms because
            # physics runs against OLD platform position, then platform moves.
            #
            # Timeout: 0.5ms - worker completes in ~100-200µs, 0.5ms is 2.5x safety margin
            # (reduced from 5ms to avoid wasting frame time)
            if transform_job_id is not None and engine and engine.is_alive():
                poll_transform_result_with_timeout(engine, transform_job_id, timeout=0.0005)

            # B1a3) Submit tracking batch to worker (non-character object movement)
            # Character autopilot handled in update_tracking_tasks (just injects keys)
            if engine and engine.is_alive():
                submit_tracking_batch(engine, dt_sim)

            # B1b) Dynamic proxies + mesh caching (MUST run before hitscan/projectile)
            # This ensures:
            # 1. CACHE_DYNAMIC_MESH jobs are submitted to worker (mesh triangles)
            # 2. dynamic_objects_map is populated (for get_dynamic_transforms())
            update_dynamic_meshes(op)

            # B1c) Submit projectile/hitscan worker jobs (after dynamic meshes ready)
            engine = getattr(op, "engine", None)
            if engine and engine.is_alive():
                submit_hitscan_batch(engine)
                submit_projectile_update(engine, dt_sim)

            # B4) Animations & audio: once with aggregated dt
            self._update_character_animation(op, agg_dt)

        # C) Input gating based on game mobility flags (unchanged)
        mg = context.scene.mobility_game

        if not mg.allow_movement:
            for k in (op.pref_forward_key, op.pref_backward_key,
                    op.pref_left_key, op.pref_right_key):
                if k in op.keys_pressed:
                    op.keys_pressed.remove(k)
        if not mg.allow_sprint and op.pref_run_key in op.keys_pressed:
            op.keys_pressed.remove(op.pref_run_key)

        # D) Physics integration (execute exactly 'steps' iterations at 30 Hz)
        op.update_movement_and_gravity(context, steps)

        # D2) Poll engine results (apply non-camera results)
        self._poll_and_apply_engine_results()

        # E) Camera update
        _view_mode = getattr(context.scene, "view_mode", 'THIRD')
        if _view_mode == 'LOCKED':
            # LOCKED: fixed camera, no obstruction raycasting needed
            from ..physics.exp_locked_view import update_locked_camera
            update_locked_camera(context, op)
        elif _view_mode == 'FIRST':
            # FIRST: camera at FPV bone, no obstruction raycasting needed
            from ..physics.exp_view_fpv import update_first_person_camera
            update_first_person_camera(context, op)
        else:
            # THIRD: full camera pipeline with engine occlusion
            camera_job_id = submit_camera_occlusion_early(op, context)
            if camera_job_id is not None:
                poll_camera_result_with_timeout(op, context, camera_job_id, timeout=0.001)
            update_camera_for_operator(context, op)

        # F) Interactions/UI/SFX (only if sim actually advanced)
        if steps:
            # Set operator for input state access
            set_current_operator(op)

            # Submit tracker evaluation to worker (async)
            # Results are processed in _poll_and_apply_engine_results
            submit_tracker_evaluation(op, context)

            check_interactions(context)
            process_pending_parenting()  # Apply delayed parenting operations
            update_text_reactions()
            update_sound_tasks()



    def handle_key_input(self, event):
        """
        Delegate to operator's existing key handler (or migrate into here later).
        This keeps behavior identical and lets you move it later if you want.
        """
        self.op.handle_key_input(event)

    # ---------- Internals ----------

    def _update_character_animation(self, op, agg_dt: float):
        """
        Update character animation state machine and play animations.

        WORKER-OFFLOADED FLOW:
        1. Update state machine with current input/physics state
        2. If state changed, play new animation via controller
        3. Update animation state (times, fades) - main thread only
        4. Submit ANIMATION_COMPUTE jobs to engine
        5. Results are processed in _poll_and_apply_engine_results()
        """
        # Skip if no state machine or controller
        # PERFORMANCE: Direct access - class-level defaults are None
        if not op.char_state_machine:
            return
        if not op.anim_controller:
            return

        sm = op.char_state_machine
        ctrl = op.anim_controller
        armature = bpy.context.scene.target_armature

        if not armature:
            return

        # Get current game time for one-shot timing
        game_time = get_game_time()

        # Update state machine
        new_state, state_changed = sm.update(
            keys_pressed=op.keys_pressed,
            delta_time=agg_dt,
            is_grounded=op.is_grounded,
            vertical_velocity=op.z_velocity,
            game_time=game_time
        )

        # If state changed, play the new animation
        if state_changed:
            action_name = sm.get_action_name()
            if action_name and ctrl.has_animation(action_name):
                props = sm.get_state_properties()

                # Play on armature with crossfade (blend_in from slot)
                ctrl.play(
                    armature.name,
                    action_name,
                    weight=1.0,
                    speed=props['speed'],
                    looping=props['loop'],
                    fade_in=props.get('blend_in', 0.15),
                    replace=True
                )

                # Start one-shot tracking if needed
                if props['is_one_shot']:
                    anim = ctrl.cache.get(action_name)
                    if anim:
                        sm.start_one_shot(anim.duration / props['speed'], game_time)

            # Update audio state
            if new_state != self._last_anim_state:
                audio_state_mgr = get_global_audio_state_manager()
                audio_state_mgr.update_audio_state(new_state)
                self._last_anim_state = new_state

        # WORKER-OFFLOADED ANIMATION FLOW:
        # ALL animation computation runs on dedicated worker (worker 0).
        # Main thread ONLY does:
        #   1. State updates (times, fades)
        #   2. Job submission
        #   3. Result polling with timeout (same-frame sync)
        #   4. bpy writes (applying poses)
        #
        # Benefits:
        # - Frees main thread from sampling/blending math
        # - Parallelizes computation while main thread does other work
        # - Same-frame sync via polling ensures no animation lag

        # 1. Update blend system layer timings
        blend_system = get_blend_system()
        if blend_system:
            blend_system.update(agg_dt)

        # 2. Update animation state (times, fades) - required before job submission
        update_animations_state(op, agg_dt)

        # 3. Submit animation compute jobs to worker 0
        engine = getattr(op, "engine", None)
        if engine and engine.is_alive():
            submit_animation_jobs(op)

            # 4. SAME-FRAME SYNC: Poll for results immediately (2ms timeout)
            # Worker computes in ~100-500µs, so 2ms is plenty of headroom
            # This ensures animations update THIS frame, not next frame
            poll_animation_results_with_timeout(op, timeout=0.002)

        # 5. Apply blend system OVERLAY on top of locomotion (additive layers)
        armature = bpy.context.scene.target_armature
        if blend_system and armature:
            blend_system.apply_to_armature(armature)

    def _poll_and_apply_engine_results(self):
        """
        Poll multiprocessing engine for completed jobs and apply results.
        Tracks frame-level synchronization metrics and prints summaries.

        Uses O(1) dispatch table lookup instead of if/elif chain for ~5-10µs savings per result.

        Note: KCC_PHYSICS_STEP results are now handled same-frame in exp_kcc.py,
        so we skip them here. Other results cached during KCC polling are also processed.
        """
        op = self.op
        engine = getattr(op, "engine", None)
        if not engine:
            return

        # Non-blocking poll (up to 100 results per frame)
        # PERF: Iterate directly instead of list() conversion to avoid allocation
        results = engine.poll_results(max_results=100)

        # Also include any results cached by KCC during same-frame polling
        pc = getattr(op, 'physics_controller', None)
        cached_kcc = pc.get_cached_other_results() if pc else []

        # Include any results cached by transform polling during same-frame sync
        transform_cached = get_cached_transform_results()

        stats = get_stats_tracker()

        # Phase 1: Engine visibility - output worker load distribution and job type stats
        output_debug = getattr(engine, 'output_debug_stats', None)
        if output_debug:
            output_debug()

        # Process all results using dispatch table
        for result in results:
            stats.record_engine_job_completed()
            op.process_engine_result(result)

            if result.success:
                # O(1) dispatch table lookup instead of if/elif chain
                handler = _RESULT_HANDLERS.get(result.job_type)
                if handler:
                    handler(op, result, stats)
            else:
                # Job failed - handle camera stuck state
                if result.job_type == "CAMERA_OCCLUSION_FULL":
                    from ..physics.exp_view import _view_state_for
                    op_key = id(op)
                    view = _view_state_for(op_key)
                    if view.get("pending_job_id") == result.job_id:
                        view["pending_job_id"] = None
                        view["pending_job_submit_time"] = 0.0
                        print(f"[Camera ERROR] Cleared stuck pending job {result.job_id}: {result.error}")

        # Process cached results from KCC and transform polling
        for result in cached_kcc:
            stats.record_engine_job_completed()
            op.process_engine_result(result)
            if result.success:
                handler = _RESULT_HANDLERS.get(result.job_type)
                if handler:
                    handler(op, result, stats)

        for result in transform_cached:
            stats.record_engine_job_completed()
            op.process_engine_result(result)
            if result.success:
                handler = _RESULT_HANDLERS.get(result.job_type)
                if handler:
                    handler(op, result, stats)

        # Include any results cached by animation polling during same-frame sync
        anim_cached = get_cached_anim_other_results(op)
        for result in anim_cached:
            stats.record_engine_job_completed()
            op.process_engine_result(result)
            if result.success:
                handler = _RESULT_HANDLERS.get(result.job_type)
                if handler:
                    handler(op, result, stats)
