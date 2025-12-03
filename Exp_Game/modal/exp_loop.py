#Exp_Game/modal/exp_loop.py

import time
import bpy
from ..props_and_utils.exp_time import update_real_time, tick_sim_time
from ..animations.exp_custom_animations import update_all_custom_managers
from ..animations.exp_animations import nla_is_locked
from ..reactions.exp_reactions import update_property_tasks
from ..reactions.exp_projectiles import update_projectile_tasks, update_hitscan_tasks
from ..reactions.exp_transforms import update_transform_tasks
from ..reactions.exp_tracking import update_tracking_tasks
from ..systems.exp_performance import update_performance_culling, apply_cull_result
from ..physics.exp_dynamic import update_dynamic_meshes, apply_dynamic_activation_result
from ..physics.exp_view import (
    update_camera_for_operator,
    submit_camera_occlusion_early,
    poll_camera_result_with_timeout,
    cache_camera_worker_result,
)
from ..interactions.exp_interactions import check_interactions, apply_interaction_check_result
from ..reactions.exp_custom_ui import update_text_reactions
from ..audio.exp_globals import update_sound_tasks
from ..audio.exp_audio import get_global_audio_state_manager
from ..developer.dev_stats import get_stats_tracker


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
        self._last_camera_summary = time.perf_counter()

        # Reset stats tracker on game start
        get_stats_tracker().reset_all()
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
                # >>> guard: skip NLA custom managers while reset/start wipes <<<
                if not nla_is_locked():
                    update_all_custom_managers(dt_sim)
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
            # >>> guard: skip character animation update while reset/start wipes <<<
            if not nla_is_locked() and op.animation_manager:
                op.animation_manager.update(
                    op.keys_pressed, agg_dt, op.is_grounded, op.z_velocity
                )
                cur_state = op.animation_manager.anim_state
                if cur_state is not None and cur_state != self._last_anim_state:
                    audio_state_mgr = get_global_audio_state_manager()
                    audio_state_mgr.update_audio_state(cur_state)
                    self._last_anim_state = cur_state

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
            check_interactions(context)
            update_text_reactions()
            update_sound_tasks()



    def handle_key_input(self, event):
        """
        Delegate to operator’s existing key handler (or migrate into here later).
        This keeps behavior identical and lets you move it later if you want.
        """
        self.op.handle_key_input(event)

    # ---------- Internals ----------

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
                if result.job_type == "FRAME_SYNC_TEST":
                    # Sync test - no action needed, metrics already tracked
                    pass
                elif result.job_type == "ECHO":
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
                            print(f"[Camera WARNING] Grid not cached in worker! Static geometry will not be tested.")

                        # Debug output - check Hz setting for mode
                        scene = bpy.context.scene
                        camera_enabled = getattr(scene, "dev_debug_camera_offload", False)
                        camera_hz = getattr(scene, "dev_debug_camera_offload_hz", 5)

                        if camera_enabled:
                            if camera_hz >= 30:
                                # Verbose mode: per-frame output
                                method = result.result.get("method", "CAMERA_FULL")
                                if hit:
                                    print(f"[Camera] job={job_id} HIT({hit_source}) dist={hit_distance:.3f}m | tris={static_tris} cells={static_cells} dyn={dynamic_tris} | {calc_time_us:.0f}us")
                                else:
                                    print(f"[Camera] job={job_id} MISS | tris={static_tris} cells={static_cells} dyn={dynamic_tris} | {calc_time_us:.0f}us")
                            else:
                                # Summary mode: print at configured Hz
                                now = time.perf_counter()
                                interval = 1.0 / camera_hz
                                if (now - self._last_camera_summary) >= interval:
                                    self._last_camera_summary = now
                                    summary = stats.get_camera_summary()

                                    print(f"[Camera] {summary['rays_per_sec']:.0f} rays/sec | "
                                          f"hit_rate: {summary['hit_rate']*100:.0f}% | "
                                          f"avg: {summary['avg_calc_us']:.0f}us")

                    except Exception as e:
                        print(f"[GameLoop] Error caching camera occlusion result: {e}")
                elif result.job_type == "DYNAMIC_MESH_ACTIVATION":
                    # Apply dynamic mesh activation result
                    try:
                        apply_dynamic_activation_result(op, result)
                    except Exception as e:
                        print(f"[GameLoop] Error applying dynamic mesh activation result: {e}")
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

                        # Debug output - check Hz setting for mode
                        scene = bpy.context.scene
                        kcc_enabled = getattr(scene, "dev_debug_kcc_offload", False)
                        kcc_hz = getattr(scene, "dev_debug_kcc_offload_hz", 1)

                        if kcc_enabled:
                            if kcc_hz >= 30:
                                # Verbose mode: per-frame output
                                pos = result.result.get("pos", (0, 0, 0))
                                on_ground = result.result.get("on_ground", False)

                                flags = []
                                if h_blocked: flags.append("BLOCKED")
                                if did_step: flags.append("STEP")
                                if did_slide: flags.append("SLIDE")
                                if hit_ceiling: flags.append("CEILING")
                                flag_str = " ".join(flags) if flags else "CLEAR"

                                print(f"[KCC] pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) ground={on_ground} {flag_str} | {calc_time:.0f}us {rays}rays {tris}tris")
                            else:
                                # Summary mode: print at configured Hz
                                now = time.perf_counter()
                                interval = 1.0 / kcc_hz
                                if (now - self._last_kcc_summary) >= interval:
                                    self._last_kcc_summary = now
                                    summary = stats.get_kcc_summary()

                                    events = []
                                    if summary["blocked_count"]: events.append(f"blocked:{summary['blocked_count']}")
                                    if summary["step_up_count"]: events.append(f"step:{summary['step_up_count']}")
                                    if summary["slide_count"]: events.append(f"slide:{summary['slide_count']}")
                                    if summary["ceiling_count"]: events.append(f"ceiling:{summary['ceiling_count']}")
                                    event_str = " ".join(events) if events else "clear"

                                    print(f"[KCC] {summary['steps_per_sec']:.0f} steps/sec | "
                                          f"avg: {summary['avg_calc_us']:.0f}us {summary['avg_rays']:.0f}rays {summary['avg_tris']:.0f}tris | "
                                          f"{event_str}")

                    except Exception as e:
                        print(f"[GameLoop] Error processing KCC stats: {e}")

                elif result.job_type == "CACHE_GRID":
                    # Grid caching confirmation
                    from ..developer.dev_debug_gate import should_print_debug
                    if should_print_debug("kcc_offload"):
                        if result.result.get("success"):
                            tris = result.result.get("triangles", 0)
                            cells = result.result.get("cells", 0)
                            print(f"[KCC] Grid cached: {tris:,} triangles, {cells:,} cells")
                        else:
                            print(f"[KCC] Grid cache ERROR: {result.result.get('error', 'Unknown error')}")

                # TODO: Add handlers for other game logic job types
                # elif result.job_type == "AI_PATHFIND":
                #     self._apply_pathfinding_result(result)
                # elif result.job_type == "PHYSICS_PREDICT":
                #     self._apply_physics_prediction(result)
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