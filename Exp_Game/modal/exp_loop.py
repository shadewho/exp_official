#Exp_Game/modal/exp_loop.py

import time
import bpy
from ..props_and_utils.exp_time import update_real_time, tick_sim_time, get_game_time
from ..animations.state_machine import AnimState
from .exp_engine_bridge import (
    update_animations_state,
    submit_animation_jobs,
    poll_animation_results_with_timeout,
    get_cached_other_results,
    process_animation_result,
)
from ..reactions.exp_reactions import update_property_tasks
from ..reactions.exp_projectiles import update_projectile_tasks, update_hitscan_tasks
from ..reactions.exp_transforms import update_transform_tasks
from ..reactions.exp_tracking import update_tracking_tasks
from ..systems.exp_performance import update_performance_culling, apply_cull_result
from ..physics.exp_dynamic import update_dynamic_meshes
from ..physics.exp_view import (
    update_camera_for_operator,
    submit_camera_occlusion_early,
    poll_camera_result_with_timeout,
    cache_camera_worker_result,
)
from ..interactions.exp_interactions import check_interactions, apply_interaction_check_result
from ..interactions.exp_tracker_eval import (
    set_current_operator,
    submit_tracker_evaluation,
    process_tracker_result,
    cache_trackers_in_worker,
)
from ..reactions.exp_custom_ui import update_text_reactions
from ..audio.exp_globals import update_sound_tasks
from ..audio.exp_audio import get_global_audio_state_manager
from ..developer.dev_stats import get_stats_tracker
from ..animations.runtime_ik import apply_runtime_ik
from ..animations.blend_system import (
    init_blend_system,
    shutdown_blend_system,
    get_blend_system,
)


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

        # Reset stats tracker on game start
        get_stats_tracker().reset_all()

        # Initialize blend system with animation cache from controller
        anim_cache = None
        if hasattr(op, 'anim_controller') and op.anim_controller:
            anim_cache = op.anim_controller.cache
        init_blend_system(anim_cache)

        # Cache tracker node graph in workers
        import bpy
        cache_trackers_in_worker(op, bpy.context)

    def shutdown(self):
        """Clean up game loop resources (called when game stops)."""
        shutdown_blend_system()
    # ---------- Public API ----------

    def on_timer(self, context):
        """
        Runs once per Blender TIMER event.
        Uses bounded catch-up for physics to prevent time dilation under stalls.
        """
        op = self.op

        # Increment frame counter for fast logger
        from ..developer.dev_logger import increment_frame
        increment_frame()

        # A) Timebases (scaled + wall clock)
        op.update_time()        # scaled per-frame dt (UI / interpolation)
        _ = update_real_time()  # wall-clock (kept for diagnostics/overlays)

        # B) Decide how many 30 Hz physics steps are due (bounded catch-up)
        steps = op._physics_steps_due()
        op._perf_last_physics_steps = steps

        if steps:
            # Shared SIM dt per 30 Hz step; aggregated dt for bulk systems
            dt_sim = op.physics_dt * float(op.time_scale)
            agg_dt = dt_sim * steps

            # Advance SIM time by the actual amount we'll simulate this frame
            tick_sim_time(agg_dt)

            # B1) Custom/scripted tasks (SIM-timed): cheap → step-per-step
            for _ in range(steps):
                update_transform_tasks()
                update_property_tasks()
                update_projectile_tasks(dt_sim)
                update_hitscan_tasks()
                update_tracking_tasks(dt_sim)

            # B2) Dynamic proxies + platform v/ω (heavy): once per frame
            update_dynamic_meshes(op)

            # B3) Distance-based culling (has its own throttling): once
            update_performance_culling(op, context)

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

        # D3) CRITICAL: Submit camera job AFTER physics so it uses final character position
        # This prevents one-frame lag where camera clips through walls on collision
        # Previous approach: submitted early (before physics) → used stale position → one frame of clipping
        # New approach: submit after physics → use final position → zero-frame latency
        camera_job_id = submit_camera_occlusion_early(op, context)

        # D4) Explicit camera sync - poll with timeout immediately after submission
        # Worker completes in ~200µs, polling adds ~300µs total (0.9% of 33ms frame budget)
        if camera_job_id is not None:
            poll_camera_result_with_timeout(op, context, camera_job_id, timeout=0.003)

        # E) Camera update (uses cached worker result with FINAL physics position)
        update_camera_for_operator(context, op)

        # F) Interactions/UI/SFX (only if sim actually advanced)
        if steps:
            # Set operator for input state access
            set_current_operator(op)

            # Submit tracker evaluation to worker (async)
            # Results are processed in _poll_and_apply_engine_results
            submit_tracker_evaluation(op, context)

            check_interactions(context)
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
        if not hasattr(op, 'char_state_machine') or not op.char_state_machine:
            return
        if not hasattr(op, 'anim_controller') or not op.anim_controller:
            return

        sm = op.char_state_machine
        ctrl = op.anim_controller
        armature = bpy.context.scene.target_armature

        if not armature:
            return

        # Check if locomotion is locked by BlendSystem (forced animation playing)
        blend_system = get_blend_system()
        locomotion_locked = blend_system and blend_system.is_locomotion_locked()

        # Only update state machine if not locked
        if not locomotion_locked:
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

                    # Get blend time from scene property (default 0.15s)
                    blend_time = bpy.context.scene.character_actions.blend_time

                    # Play on armature with crossfade
                    ctrl.play(
                        armature.name,
                        action_name,
                        weight=1.0,
                        speed=props['speed'],
                        looping=props['loop'],
                        fade_in=blend_time,
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
        # 1. Update state (times, fades) on main thread
        update_animations_state(op, agg_dt)

        # 1b. Update blend system layer timings (but don't apply yet)
        blend_system = get_blend_system()
        if blend_system:
            blend_system.update(agg_dt)

        # 2. Submit jobs to engine
        submit_animation_jobs(op)

        # 3. Same-frame sync: poll for results immediately
        # Worker compute is fast (~100µs), so we wait up to 2ms
        # This applies the base locomotion animation to the armature
        poll_animation_results_with_timeout(op, timeout=0.002)

        # 4. Apply blend system OVERLAY on top of locomotion
        # This must happen AFTER worker results are applied so overlays work correctly
        armature = bpy.context.scene.target_armature
        if blend_system and armature:
            blend_system.apply_to_armature(armature)

        # 5. Runtime IK overlay (after all animation poses are applied)
        # IK modifies bones on top of animation - only if enabled in scene
        if armature:
            apply_runtime_ik(armature, agg_dt)

    def _poll_and_apply_engine_results(self):
        """
        Poll multiprocessing engine for completed jobs and apply results.
        Tracks frame-level synchronization metrics and prints summaries.

        Note: KCC_PHYSICS_STEP results are now handled same-frame in exp_kcc.py,
        so we skip them here. Other results cached during KCC polling are also processed.
        """
        op = self.op
        engine = getattr(op, "engine", None)
        if not engine:
            return

        # Non-blocking poll (up to 100 results per frame)
        results = list(engine.poll_results(max_results=100))

        # Also include any results cached by KCC during same-frame polling
        pc = getattr(op, 'physics_controller', None)
        if pc:
            cached_results = pc.get_cached_other_results()
            results.extend(cached_results)

        # Include any results cached by animation polling during same-frame sync
        anim_cached = get_cached_other_results(op)
        results.extend(anim_cached)

        stats = get_stats_tracker()

        # Record completions for engine stats
        for _ in results:
            stats.record_engine_job_completed()

        # Phase 1: Engine visibility - output worker load distribution and job type stats
        # This calls the engine's own debug output method which tracks per-worker and per-job-type stats
        if hasattr(engine, 'output_debug_stats'):
            engine.output_debug_stats()  # Gets context from bpy.context internally

        for result in results:
            # Process result and track latency metrics
            if hasattr(op, 'process_engine_result'):
                op.process_engine_result(result)

            # Route result to appropriate handler based on job type
            if result.success:
                if result.job_type == "ECHO":
                    # Echo test - no action needed
                    pass
                elif result.job_type == "CULL_BATCH":
                    # Apply performance culling result
                    try:
                        apply_cull_result(op, result.result)
                    except Exception as e:
                        print(f"[GameLoop] Error applying cull result: {e}")
                elif result.job_type == "CAMERA_OCCLUSION_FULL":
                    # Cache camera occlusion result - only if it matches current pending job
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
                elif result.job_type == "INTERACTION_CHECK_BATCH":
                    # Apply interaction check result (full offload with state management)
                    try:
                        apply_interaction_check_result(result, bpy.context)
                    except Exception as e:
                        print(f"[GameLoop] Error applying interaction check result: {e}")
                        import traceback
                        traceback.print_exc()
                        # CRITICAL: Clear pending job on error to prevent infinite loop
                        if hasattr(op, '_pending_interaction_job_id'):
                            op._pending_interaction_job_id = None
                elif result.job_type == "COMPUTE_HEAVY":
                    # Stress test - no action needed
                    pass
                elif result.job_type == "KCC_PHYSICS_STEP":
                    # KCC results are now processed same-frame in exp_kcc.py
                    # This branch only handles late results (should be rare)
                    # Still record stats for debug output
                    try:
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
                        from ..developer.dev_logger import log_game
                        dynamic_active = debug.get("dynamic_meshes_active", 0)
                        dynamic_xform_time = debug.get("dynamic_transform_time_us", 0.0)
                        if dynamic_active > 0:
                            log_game("ENGINE",
                                f"MAIN: worker_time={calc_time:.0f}µs (xform={dynamic_xform_time:.0f}µs) "
                                f"rays={rays} tris={tris} dyn={dynamic_active}")

                        # KCC Debug output uses fast buffer logger (log_game)
                        # NO console prints during gameplay (destroys performance)
                        # All KCC physics logs are handled in exp_kcc.py via log_game()

                    except Exception as e:
                        print(f"[GameLoop] Error processing KCC stats: {e}")

                elif result.job_type == "CACHE_GRID":
                    # Grid caching confirmation
                    from ..developer.dev_debug_gate import should_print_debug
                    if should_print_debug("kcc_physics"):
                        if result.result.get("success"):
                            tris = result.result.get("triangles", 0)
                            cells = result.result.get("cells", 0)
                            print(f"[KCC] Grid cached: {tris:,} triangles, {cells:,} cells")
                        else:
                            print(f"[KCC] Grid cache ERROR: {result.result.get('error', 'Unknown error')}")

                elif result.job_type == "CACHE_DYNAMIC_MESH":
                    # Dynamic mesh caching confirmation
                    # Process worker logs (cache events)
                    if result.result.get("success"):
                        worker_logs = result.result.get("logs", [])
                        if worker_logs:
                            from ..developer.dev_logger import log_worker_messages
                            log_worker_messages(worker_logs)

                elif result.job_type == "CACHE_ANIMATIONS":
                    # Animation caching confirmation
                    if result.result.get("success"):
                        worker_logs = result.result.get("logs", [])
                        if worker_logs:
                            from ..developer.dev_logger import log_worker_messages
                            log_worker_messages(worker_logs)

                elif result.job_type == "ANIMATION_COMPUTE_BATCH":
                    # Apply computed bone transforms from batched worker job
                    try:
                        objects_applied = process_animation_result(op, result)

                        # Process worker logs and log result
                        if result.result:
                            result_data = result.result
                            worker_logs = result_data.get("logs", [])
                            total_objects = result_data.get("total_objects", 0)
                            total_bones = result_data.get("total_bones", 0)
                            total_anims = result_data.get("total_anims", 0)
                            calc_time = result_data.get("calc_time_us", 0.0)

                            from ..developer.dev_logger import log_game
                            log_game("ANIMATIONS", f"BATCH_RESULT job={result.job_id} objs={total_objects} bones={total_bones} anims={total_anims} time={calc_time:.0f}µs")

                            if worker_logs:
                                from ..developer.dev_logger import log_worker_messages
                                log_worker_messages(worker_logs)

                    except Exception as e:
                        print(f"[GameLoop] Error applying animation batch result: {e}")

                elif result.job_type == "EVALUATE_TRACKERS":
                    # Apply tracker evaluation results from worker
                    try:
                        process_tracker_result(result)
                    except Exception as e:
                        print(f"[GameLoop] Error applying tracker result: {e}")

                elif result.job_type == "CACHE_TRACKERS":
                    # Tracker cache confirmation from worker
                    if result.result:
                        count = result.result.get("tracker_count", 0)
                        worker_logs = result.result.get("logs", [])
                        if worker_logs:
                            from ..developer.dev_logger import log_worker_messages
                            log_worker_messages(worker_logs)

            else:
                # Job failed - already logged in process_engine_result
                # CRITICAL: Clear pending_job_id for camera jobs to prevent permanent stuck state
                if result.job_type == "CAMERA_OCCLUSION_FULL":
                    from ..physics.exp_view import _view_state_for
                    op_key = id(op)
                    view = _view_state_for(op_key)
                    if view.get("pending_job_id") == result.job_id:
                        view["pending_job_id"] = None
                        view["pending_job_submit_time"] = 0.0
                        print(f"[Camera ERROR] Cleared stuck pending job {result.job_id}: {result.error}")

        # Periodic sync stats reporting (disabled when test manager is active)
        # Test manager has its own comprehensive reporting
        pass