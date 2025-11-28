#Exp_Game/modal/exp_loop.py

from ..props_and_utils.exp_time import update_real_time, tick_sim_time
from ..animations.exp_custom_animations import update_all_custom_managers
from ..animations.exp_animations import nla_is_locked
from ..reactions.exp_reactions import update_property_tasks
from ..reactions.exp_projectiles import update_projectile_tasks, update_hitscan_tasks
from ..reactions.exp_transforms import update_transform_tasks
from ..systems.exp_performance import update_performance_culling, apply_cull_result
from ..physics.exp_dynamic import update_dynamic_meshes, apply_dynamic_activation_result
from ..physics.exp_view import update_camera_for_operator
from ..interactions.exp_interactions import check_interactions, apply_interaction_check_result
from ..reactions.exp_custom_ui import update_text_reactions
from ..audio.exp_globals import update_sound_tasks
from ..audio.exp_audio import get_global_audio_state_manager


class GameLoop:
    """
    Per-frame/TIMER orchestrator for ExpModal.
    - No bpy writes in workers; only here on the main thread.
    - Reuses operator's state/caches to avoid churn.
    """

    def __init__(self, op):
        self.op = op
        self._last_anim_state = None
    # ---------- Public API ----------

    def on_timer(self, context):
        """
        Runs once per Blender TIMER event.
        Uses bounded catch-up for physics to prevent time dilation under stalls.
        """
        op = self.op

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

        # E) Camera update
        update_camera_for_operator(context, op)

        # F) Interactions/UI/SFX (only if sim actually advanced)
        if steps:
            check_interactions(context)
            update_text_reactions()
            update_sound_tasks()

        # G) Poll multiprocessing engine results EVERY frame (not just physics frames)
        # This ensures async results like raycast tests get processed promptly
        self._poll_and_apply_engine_results()



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
        Tracks frame-level synchronization metrics.
        """
        op = self.op
        engine = getattr(op, "engine", None)
        if not engine:
            return

        # Non-blocking poll (up to 100 results per frame)
        results = engine.poll_results(max_results=100)

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
                elif result.job_type == "DYNAMIC_MESH_ACTIVATION":
                    # Apply dynamic mesh activation result
                    try:
                        apply_dynamic_activation_result(op, result)
                    except Exception as e:
                        print(f"[GameLoop] Error applying dynamic mesh activation result: {e}")
                elif result.job_type == "INTERACTION_CHECK_BATCH":
                    # Apply interaction check result (diagnostic only in Sprint 1.2)
                    try:
                        apply_interaction_check_result(result)
                    except Exception as e:
                        print(f"[GameLoop] Error applying interaction check result: {e}")
                elif result.job_type == "COMPUTE_HEAVY":
                    # Stress test - no action needed
                    pass
                elif result.job_type == "KCC_INPUT_VECTOR":
                    # Cache input vector result for next frame's use
                    try:
                        pc = getattr(op, 'physics_controller', None)
                        if pc:
                            wish_dir_xy = result.result.get("wish_dir_xy", (0.0, 0.0))
                            is_running = result.result.get("is_running", False)
                            pc.cache_input_result(wish_dir_xy, is_running)

                            # Debug output (frequency-gated)
                            from ..developer.dev_debug_gate import should_print_debug
                            if should_print_debug("kcc_offload"):
                                calc_time = result.result.get("calc_time_us", 0.0)
                                print(f"[KCC Offload] Cached result: wish_dir={wish_dir_xy}, is_running={is_running}, calc_time={calc_time:.2f}µs")
                    except Exception as e:
                        print(f"[GameLoop] Error caching KCC input result: {e}")
                elif result.job_type == "KCC_SLOPE_PLATFORM_MATH":
                    # Cache slope/platform math result for next frame's use
                    try:
                        pc = getattr(op, 'physics_controller', None)
                        if pc:
                            delta_z = result.result.get("delta_z", 0.0)
                            slide_xy = result.result.get("slide_xy", (0.0, 0.0))
                            is_sliding = result.result.get("is_sliding", False)
                            carry = result.result.get("carry", (0.0, 0.0, 0.0))
                            rot_delta_z = result.result.get("rot_delta_z", 0.0)
                            pc.cache_slope_platform_result(delta_z, slide_xy, is_sliding, carry, rot_delta_z)

                            # Debug output (frequency-gated)
                            from ..developer.dev_debug_gate import should_print_debug
                            if should_print_debug("kcc_offload"):
                                calc_time = result.result.get("calc_time_us", 0.0)
                                if is_sliding:
                                    print(f"[KCC Offload] Cached slope/platform: SLIDING slide_xy={slide_xy}, delta_z={delta_z:.4f}, carry={carry}, rot_delta={rot_delta_z:.4f}, calc_time={calc_time:.2f}µs")
                                else:
                                    print(f"[KCC Offload] Cached slope/platform: delta_z={delta_z:.4f}, carry={carry}, rot_delta={rot_delta_z:.4f}, calc_time={calc_time:.2f}µs")
                    except Exception as e:
                        print(f"[GameLoop] Error caching KCC slope/platform result: {e}")
                elif result.job_type == "KCC_RAYCAST":
                    # Cache raycast result for next frame's use (brute force method)
                    try:
                        pc = getattr(op, 'physics_controller', None)
                        hit = result.result.get("hit", False)
                        hit_location = result.result.get("hit_location", None)
                        hit_normal = result.result.get("hit_normal", None)
                        hit_distance = result.result.get("hit_distance", None)

                        if pc:
                            pc.cache_raycast_result(hit, hit_location, hit_normal, hit_distance)

                        # Always print for startup test (this job type is only used for testing)
                        calc_time = result.result.get("calc_time_us", 0.0)
                        triangles_tested = result.result.get("triangles_tested", 0)
                        method = result.result.get("method", "BRUTE_FORCE")
                        if hit:
                            print(f"[Raycast {method}] HIT dist={hit_distance:.4f}m, tested={triangles_tested:,} tris, time={calc_time:.2f}µs")
                        else:
                            print(f"[Raycast {method}] MISS, tested={triangles_tested:,} tris, time={calc_time:.2f}µs")
                    except Exception as e:
                        print(f"[GameLoop] Error caching KCC raycast result: {e}")

                elif result.job_type == "KCC_RAYCAST_GRID":
                    # Cache raycast result for next frame's use (grid-accelerated method)
                    try:
                        pc = getattr(op, 'physics_controller', None)
                        hit = result.result.get("hit", False)
                        hit_location = result.result.get("hit_location", None)
                        hit_normal = result.result.get("hit_normal", None)
                        hit_distance = result.result.get("hit_distance", None)

                        if pc:
                            pc.cache_raycast_result(hit, hit_location, hit_normal, hit_distance)

                        # Always print for startup test (this job type is only used for testing)
                        calc_time = result.result.get("calc_time_us", 0.0)
                        triangles_tested = result.result.get("triangles_tested", 0)
                        cells_traversed = result.result.get("cells_traversed", 0)
                        method = result.result.get("method", "GRID_DDA")

                        if hit:
                            print(f"[Raycast {method}] HIT dist={hit_distance:.4f}m, tested={triangles_tested} tris, cells={cells_traversed}, time={calc_time:.2f}µs")
                        else:
                            print(f"[Raycast {method}] MISS, tested={triangles_tested} tris, cells={cells_traversed}, time={calc_time:.2f}µs")

                        # Check for error
                        if "error" in result.result:
                            print(f"[Raycast {method}] ERROR: {result.result['error']}")
                    except Exception as e:
                        print(f"[GameLoop] Error caching KCC raycast grid result: {e}")

                elif result.job_type == "CACHE_GRID":
                    # Grid caching confirmation
                    from ..developer.dev_debug_gate import should_print_debug
                    if should_print_debug("raycast_offload"):
                        if result.result.get("success"):
                            tris = result.result.get("triangles", 0)
                            cells = result.result.get("cells", 0)
                            print(f"[Grid Cache] Worker confirmed: {tris:,} triangles, {cells:,} cells cached")
                        else:
                            print(f"[Grid Cache] ERROR: {result.result.get('error', 'Unknown error')}")

                elif result.job_type == "KCC_RAYCAST_CACHED":
                    # Cached raycast result (uses pre-cached grid, minimal serialization)
                    try:
                        pc = getattr(op, 'physics_controller', None)
                        hit = result.result.get("hit", False)
                        hit_location = result.result.get("hit_location", None)
                        hit_normal = result.result.get("hit_normal", None)
                        hit_distance = result.result.get("hit_distance", None)

                        if pc:
                            pc.cache_raycast_result(hit, hit_location, hit_normal, hit_distance)

                        # Always print for startup test (this job type is only used for testing)
                        calc_time = result.result.get("calc_time_us", 0.0)
                        triangles_tested = result.result.get("triangles_tested", 0)
                        cells_traversed = result.result.get("cells_traversed", 0)
                        method = result.result.get("method", "CACHED_DDA")

                        if hit:
                            print(f"[Raycast {method}] HIT dist={hit_distance:.4f}m, tested={triangles_tested} tris, cells={cells_traversed}, time={calc_time:.2f}µs")
                        else:
                            print(f"[Raycast {method}] MISS, tested={triangles_tested} tris, cells={cells_traversed}, time={calc_time:.2f}µs")

                        if "error" in result.result:
                            print(f"[Raycast {method}] ERROR: {result.result['error']}")
                    except Exception as e:
                        print(f"[GameLoop] Error handling cached raycast result: {e}")

                # TODO: Add handlers for other game logic job types
                # elif result.job_type == "AI_PATHFIND":
                #     self._apply_pathfinding_result(result)
                # elif result.job_type == "PHYSICS_PREDICT":
                #     self._apply_physics_prediction(result)
            else:
                # Job failed - already logged in process_engine_result
                pass

        # Periodic sync stats reporting (disabled when test manager is active)
        # Test manager has its own comprehensive reporting
        pass