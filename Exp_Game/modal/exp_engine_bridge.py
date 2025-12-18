# Exp_Game/modal/exp_engine_bridge.py
"""
Engine Bridge - Modal-Engine Communication Layer

This module handles all engine lifecycle and communication:
- Engine initialization (5-step startup sequence)
- Engine shutdown with cache cleanup
- Job submission with sync tracking
- Result processing with latency metrics
- Animation system lifecycle (bake on start, clear on end)

Keeps engine concerns separate from modal game logic.
"""

import time
import pickle
import bpy
from ..engine import EngineCore
from ..engine.animations.baker import bake_action
from ..animations.controller import AnimationController


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

def init_engine(modal, context) -> tuple[bool, str]:
    """
    Initialize engine with 5-step startup sequence.

    Args:
        modal: ExpModal operator instance
        context: Blender context

    Returns:
        (success: bool, message: str)
    """
    startup_logs = context.scene.dev_startup_logs

    if startup_logs:
        print("\n" + "="*70)
        print("  GAME STARTUP SEQUENCE - ENGINE FIRST")
        print("="*70)

    # ─── STEP 1: Spawn Engine ───
    if startup_logs:
        print("\n[STARTUP 1/5] Spawning engine workers...")

    if not hasattr(modal, 'engine'):
        modal.engine = EngineCore()

    modal.engine.start()

    if startup_logs:
        stats = modal.engine.get_stats()
        print(f"[STARTUP 1/5] ✓ Engine started")
        print(f"              Workers: {stats['workers_alive']}/{stats['workers_total']} spawned")

    # ─── STEP 2: Verify Workers Alive ───
    if startup_logs:
        print(f"\n[STARTUP 2/5] Verifying workers alive...")

    if not modal.engine.is_alive():
        error_msg = "Engine workers failed to spawn"
        if startup_logs:
            print(f"[STARTUP 2/5] ✗ FAILED: {error_msg}")
            print("="*70 + "\n")
        modal.engine.shutdown()
        return False, error_msg

    if startup_logs:
        print(f"[STARTUP 2/5] ✓ All workers alive and running")

    # ─── STEP 3: PING Verification ───
    if startup_logs:
        print(f"\n[STARTUP 3/5] Verifying worker responsiveness (PING check)...")

    if not modal.engine.wait_for_readiness(timeout=5.0):
        error_msg = "Engine workers not responding to PING"
        if startup_logs:
            print(f"[STARTUP 3/5] ✗ FAILED: {error_msg}")
            print("="*70 + "\n")
        modal.engine.shutdown()
        return False, error_msg

    if startup_logs:
        print(f"[STARTUP 3/5] ✓ All workers responding to PING")

    # Initialize sync tracking on modal
    _init_sync_tracking(modal)

    # ─── STEP 4: Cache Spatial Grid ───
    if startup_logs:
        print(f"\n[STARTUP 4/5] Caching spatial grid in workers...")

    if modal.spatial_grid and modal.engine and modal.engine.is_alive():
        # Measure grid serialization (one-time cost)
        pickle_start = time.perf_counter()
        pickled = pickle.dumps({"grid": modal.spatial_grid})
        pickle_time = (time.perf_counter() - pickle_start) * 1000
        pickle_size_kb = len(pickled) / 1024

        if startup_logs:
            print(f"[STARTUP 4/5] Grid size: {pickle_size_kb:.1f} KB")
            print(f"              Serialization time: {pickle_time:.1f}ms")

        # Send CACHE_GRID jobs to all workers
        for i in range(8):
            modal.engine.submit_job("CACHE_GRID", {"grid": modal.spatial_grid})

        if startup_logs:
            print(f"[STARTUP 4/5] Grid jobs submitted, waiting for all workers to confirm...")

        if not modal.engine.verify_grid_cache(timeout=5.0):
            error_msg = "Not all workers cached spatial grid"
            if startup_logs:
                print(f"[STARTUP 4/5] ✗ FAILED: {error_msg}")
                print("="*70 + "\n")
            modal.engine.shutdown()
            return False, error_msg

        if startup_logs:
            print(f"[STARTUP 4/5] ✓ Grid successfully cached in all workers")
    else:
        if startup_logs:
            print(f"[STARTUP 4/5] ⊘ No spatial grid (skipped)")

    # ─── STEP 5: Final Readiness Confirmation ───
    if startup_logs:
        print(f"\n[STARTUP 5/5] Final readiness check (lock-step synchronization)...")

    final_status = modal.engine.get_full_readiness_status(grid_required=bool(modal.spatial_grid))

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
        modal.engine.shutdown()
        return False, f"Engine not ready: {error_msg}"

    if startup_logs:
        print(f"[STARTUP 5/5] ✓ {final_status['message']}")
        print(f"\n              All Checks Passed:")
        for check_name, passed in final_status["checks"].items():
            print(f"              ✓ {check_name}")
        print(f"\n" + "="*70)
        print(f"  ENGINE READY - MODAL STARTING")
        print("="*70 + "\n")

    return True, "Engine ready"


def shutdown_engine(modal, context):
    """
    Shutdown engine gracefully with cache cleanup.

    Args:
        modal: ExpModal operator instance
        context: Blender context
    """
    if not hasattr(modal, 'engine') or not modal.engine:
        return

    debug = context.scene.dev_debug_engine

    if debug:
        print("[ExpModal] Shutting down multiprocessing engine...")

    # Clear dynamic mesh caches in all workers before shutdown
    try:
        for _ in range(8):
            modal.engine.submit_job("CLEAR_DYNAMIC_CACHE", {"clear_all": True})
        if debug:
            print("[ExpModal] Sent CLEAR_DYNAMIC_CACHE to all workers")
    except Exception as e:
        if debug:
            print(f"[ExpModal] Warning: Failed to clear dynamic cache: {e}")

    # Print sync report if debug enabled
    if debug:
        print_sync_report(modal)

    modal.engine.shutdown()
    modal.engine = None

    if debug:
        print("[ExpModal] Engine shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════
# JOB SUBMISSION & RESULT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def _init_sync_tracking(modal):
    """Initialize sync tracking variables on modal."""
    modal._physics_frame = 0
    modal._pending_jobs = {}
    modal._sync_jobs_submitted = 0
    modal._sync_results_received = 0
    modal._sync_frame_latencies = []
    modal._sync_time_latencies = []
    modal._sync_last_report_frame = 0


def submit_engine_job(modal, job_type: str, data: dict) -> int:
    """
    Submit a job to the engine with frame tagging for sync tracking.

    Args:
        modal: ExpModal operator instance
        job_type: Type of job to submit
        data: Job data dict

    Returns:
        job_id or -1 if submission failed
    """
    if not hasattr(modal, 'engine') or not modal.engine:
        return -1

    job_id = modal.engine.submit_job(job_type, data)

    if job_id is not None and job_id >= 0:
        modal._pending_jobs[job_id] = {
            "frame": modal._physics_frame,
            "timestamp": time.perf_counter()
        }
        modal._sync_jobs_submitted += 1

    return job_id if job_id is not None else -1


def process_engine_result(modal, result) -> bool:
    """
    Process a single engine result and track latency metrics.

    Args:
        modal: ExpModal operator instance
        result: EngineResult object

    Returns:
        True if result was successfully processed
    """
    if result.job_id not in modal._pending_jobs:
        return False

    # Calculate latencies
    job_info = modal._pending_jobs[result.job_id]
    frame_latency = modal._physics_frame - job_info["frame"]
    time_latency_ms = (time.perf_counter() - job_info["timestamp"]) * 1000.0

    # Track metrics
    modal._sync_frame_latencies.append(frame_latency)
    modal._sync_time_latencies.append(time_latency_ms)
    modal._sync_results_received += 1

    # Clean up
    del modal._pending_jobs[result.job_id]

    # Warn on stale results
    if frame_latency > 2:
        print(f"[EngineSync] WARNING: Stale result - Frame latency: {frame_latency} frames ({time_latency_ms:.1f}ms)")

    return True


def print_sync_report(modal):
    """Print comprehensive synchronization statistics."""
    if not hasattr(modal, '_sync_jobs_submitted') or modal._sync_jobs_submitted == 0:
        print("\n[EngineSync] No jobs submitted during session")
        return

    print("\n" + "="*60)
    print("ENGINE SYNCHRONIZATION REPORT")
    print("="*60)
    print(f"Total Physics Frames:     {modal._physics_frame}")
    print(f"Jobs Submitted:           {modal._sync_jobs_submitted}")
    print(f"Results Received:         {modal._sync_results_received}")
    print(f"Pending Jobs:             {len(modal._pending_jobs)}")

    if modal._sync_frame_latencies:
        avg_frame_lat = sum(modal._sync_frame_latencies) / len(modal._sync_frame_latencies)
        max_frame_lat = max(modal._sync_frame_latencies)
        stale_count = sum(1 for x in modal._sync_frame_latencies if x > 2)
        stale_pct = (stale_count / len(modal._sync_frame_latencies)) * 100

        print(f"\nFrame Latency:")
        print(f"  Average:                {avg_frame_lat:.2f} frames")
        print(f"  Maximum:                {max_frame_lat} frames")
        print(f"  Stale (>2 frames):      {stale_count} ({stale_pct:.1f}%)")

    if modal._sync_time_latencies:
        avg_time_lat = sum(modal._sync_time_latencies) / len(modal._sync_time_latencies)
        max_time_lat = max(modal._sync_time_latencies)
        min_time_lat = min(modal._sync_time_latencies)

        print(f"\nTime Latency:")
        print(f"  Average:                {avg_time_lat:.2f}ms")
        print(f"  Min:                    {min_time_lat:.2f}ms")
        print(f"  Max:                    {max_time_lat:.2f}ms")

    # Grade sync quality
    if modal._sync_frame_latencies:
        if avg_frame_lat <= 1.0 and max_frame_lat <= 2 and stale_pct < 5:
            grade = "A (Excellent)"
        elif avg_frame_lat <= 1.5 and max_frame_lat <= 3 and stale_pct < 10:
            grade = "B (Good)"
        else:
            grade = "F (Needs Improvement)"

        print(f"\nSync Quality Grade:       {grade}")

    print("="*60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATION LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

def init_animations(modal, context) -> tuple[bool, str]:
    """
    Initialize animation system - bake all actions and cache in workers.

    Args:
        modal: ExpModal operator instance
        context: Blender context

    Returns:
        (success: bool, message: str)
    """
    startup_logs = context.scene.dev_startup_logs
    scene = context.scene
    armature = scene.target_armature

    if startup_logs:
        print("\n[ANIMATIONS] Initializing animation system...")

    # Create fresh controller
    modal.anim_controller = AnimationController()

    # Initialize pending animation jobs tracking
    modal._pending_anim_jobs = {}

    if not armature:
        if startup_logs:
            print("[ANIMATIONS] ⊘ No target armature - animations disabled")
        return True, "No armature"

    if armature.type != 'ARMATURE':
        if startup_logs:
            print("[ANIMATIONS] ⊘ Target is not an armature - animations disabled")
        return True, "Invalid armature"

    # Bake ALL actions in the blend file
    start_time = time.perf_counter()
    baked_count = 0
    failed = []

    for action in bpy.data.actions:
        try:
            anim = bake_action(action, armature)
            modal.anim_controller.add_animation(anim)
            baked_count += 1
        except Exception as e:
            failed.append(f"{action.name}: {e}")

    bake_elapsed = (time.perf_counter() - start_time) * 1000

    if startup_logs:
        print(f"[ANIMATIONS] ✓ Baked {baked_count} actions in {bake_elapsed:.0f}ms")
        if failed:
            print(f"[ANIMATIONS] ⚠ {len(failed)} actions failed to bake")

    # NOTE: Worker caching happens later via cache_animations_in_workers()
    # This is called after init_engine() in exp_modal.py

    return True, f"Baked {baked_count} actions"


def cache_animations_in_workers(modal, context) -> bool:
    """
    Cache baked animations in all workers.
    MUST be called AFTER init_engine() since it needs workers to be running.

    Args:
        modal: ExpModal operator instance
        context: Blender context

    Returns:
        True if caching succeeded
    """
    startup_logs = context.scene.dev_startup_logs

    if not hasattr(modal, 'anim_controller') or not modal.anim_controller:
        return True  # No animations to cache

    if modal.anim_controller.cache.count == 0:
        # Log to confirm animation system is initialized but has no animations
        print(f"[ANIM_DIAG] cache_animations_in_workers: No animations baked (cache.count=0)")
        return True  # No animations baked

    # Diagnostic: List all cached animations
    print(f"[ANIM_DIAG] cache_animations_in_workers: Caching {modal.anim_controller.cache.count} animations:")
    for name in modal.anim_controller.cache.names:
        print(f"[ANIM_DIAG]   - {name}")

    if not hasattr(modal, 'engine') or not modal.engine or not modal.engine.is_alive():
        if startup_logs:
            print("[ANIMATIONS] ⚠ Engine not available - skipping worker cache")
        return False

    cache_data = modal.anim_controller.get_cache_data_for_workers()
    baked_count = modal.anim_controller.cache.count

    if startup_logs:
        print(f"[ANIMATIONS] Caching {baked_count} animations in workers...")

    # Broadcast cache job to all workers (targeted delivery)
    submitted = modal.engine.broadcast_job("CACHE_ANIMATIONS", cache_data)

    # Wait for all cache jobs to complete
    if submitted > 0:
        confirmed_workers = set()
        start_time = time.perf_counter()
        timeout = 1.0

        while len(confirmed_workers) < submitted:
            if time.perf_counter() - start_time > timeout:
                if startup_logs:
                    print(f"[ANIMATIONS] ⚠ Timeout ({len(confirmed_workers)}/{submitted} workers)")
                break

            results = list(modal.engine.poll_results(max_results=50))
            for result in results:
                if result.job_type == "CACHE_ANIMATIONS" and result.success:
                    confirmed_workers.add(result.worker_id)

            if len(confirmed_workers) >= submitted:
                break
            time.sleep(0.002)

        if startup_logs:
            elapsed = (time.perf_counter() - start_time) * 1000
            print(f"[ANIMATIONS] ✓ Cached in {len(confirmed_workers)}/{submitted} workers ({elapsed:.0f}ms)")

    return True


def shutdown_animations(modal, context):
    """
    Shutdown animation system - clear all cached data.

    Args:
        modal: ExpModal operator instance
        context: Blender context
    """
    debug = context.scene.dev_debug_engine

    if hasattr(modal, 'anim_controller') and modal.anim_controller:
        if debug:
            cache_count = modal.anim_controller.cache.count
            print(f"[ANIMATIONS] Clearing {cache_count} cached animations...")

        modal.anim_controller.clear_all()
        modal.anim_controller = None

        if debug:
            print("[ANIMATIONS] Animation system shutdown complete")

    # Clear pending jobs
    if hasattr(modal, '_pending_anim_jobs'):
        modal._pending_anim_jobs.clear()


def update_animations_state(modal, delta_time: float):
    """
    Update animation state (times, fades). Call at start of frame.
    This prepares data for worker submission.

    Args:
        modal: ExpModal operator instance
        delta_time: Time since last frame in seconds
    """
    if hasattr(modal, 'anim_controller') and modal.anim_controller:
        modal.anim_controller.update_state(delta_time)


def submit_animation_jobs(modal) -> int:
    """
    Submit ANIMATION_COMPUTE jobs to engine for all active animations.
    Call after update_animations_state().

    Args:
        modal: ExpModal operator instance

    Returns:
        Number of jobs submitted
    """
    if not hasattr(modal, 'anim_controller') or not modal.anim_controller:
        return 0

    if not hasattr(modal, 'engine') or not modal.engine or not modal.engine.is_alive():
        return 0

    # Get job data for all objects with active animations
    jobs_data = modal.anim_controller.get_compute_job_data()

    if not jobs_data:
        # Log when no animation jobs (helps debug if controller has no active anims)
        from ..developer.dev_logger import log_game
        active_states = len(modal.anim_controller._states)
        cache_count = modal.anim_controller.cache.count
        log_game("ANIMATIONS", f"NO_JOBS: states={active_states} cache={cache_count}")
        return 0

    # Submit jobs
    count = 0
    from ..developer.dev_logger import log_game
    for object_name, job_data in jobs_data.items():
        job_id = modal.engine.submit_job("ANIMATION_COMPUTE", job_data)
        if job_id is not None and job_id >= 0:
            modal._pending_anim_jobs[job_id] = object_name
            count += 1
            # Log job submission details
            playing = job_data.get("playing", [])
            anim_names = [p["anim_name"] for p in playing]
            log_game("ANIMATIONS", f"SUBMIT job={job_id} obj={object_name} anims={anim_names}")

    return count


def process_animation_result(modal, result) -> bool:
    """
    Process an ANIMATION_COMPUTE result and apply to Blender.
    Call when receiving result from engine.

    Args:
        modal: ExpModal operator instance
        result: EngineResult object

    Returns:
        True if result was processed successfully
    """
    if not hasattr(modal, '_pending_anim_jobs'):
        return False

    job_id = result.job_id
    if job_id not in modal._pending_anim_jobs:
        return False

    object_name = modal._pending_anim_jobs.pop(job_id)

    if not result.success:
        return False

    result_data = result.result
    bone_transforms = result_data.get("bone_transforms", {})

    if bone_transforms and hasattr(modal, 'anim_controller') and modal.anim_controller:
        modal.anim_controller.apply_worker_result(object_name, bone_transforms)

    return True


def poll_animation_results_with_timeout(modal, timeout: float = 0.002) -> int:
    """
    Poll for ANIMATION_COMPUTE results with short timeout for same-frame sync.
    Similar to camera same-frame sync - worker is fast (~100µs) so we can wait.

    Non-animation results that arrive during polling are cached for later processing
    by _poll_and_apply_engine_results().

    Args:
        modal: ExpModal operator instance
        timeout: Max time to wait in seconds (default 2ms)

    Returns:
        Number of animation results applied
    """
    if not hasattr(modal, 'engine') or not modal.engine:
        return 0

    if not hasattr(modal, '_pending_anim_jobs') or not modal._pending_anim_jobs:
        return 0

    # Initialize cached results list if needed (for non-animation results)
    if not hasattr(modal, '_cached_other_results'):
        modal._cached_other_results = []

    # Small pre-poll delay to give worker time to start (like camera does)
    time.sleep(0.00015)  # 150µs

    start_time = time.perf_counter()
    applied = 0

    while time.perf_counter() - start_time < timeout:
        # Non-blocking poll
        results = list(modal.engine.poll_results(max_results=20))

        for result in results:
            if result.job_type == "ANIMATION_COMPUTE":
                if process_animation_result(modal, result):
                    applied += 1
                # Process worker logs
                if result.success:
                    worker_logs = result.result.get("logs", [])
                    bones_count = result.result.get("bones_count", 0)
                    anims_blended = result.result.get("anims_blended", 0)
                    calc_time = result.result.get("calc_time_us", 0.0)
                    # Always log result received (diagnostics)
                    from ..developer.dev_logger import log_game
                    log_game("ANIMATIONS", f"RESULT job={result.job_id} bones={bones_count} anims={anims_blended} time={calc_time:.0f}µs logs={len(worker_logs)}")
                    if worker_logs:
                        from ..developer.dev_logger import log_worker_messages
                        log_worker_messages(worker_logs)
            else:
                # Cache non-animation results for later processing
                modal._cached_other_results.append(result)

        # All pending animation jobs completed
        if not modal._pending_anim_jobs:
            break

        # Brief sleep to avoid busy spin
        time.sleep(0.0001)  # 100µs

    return applied


def get_cached_other_results(modal) -> list:
    """
    Get and clear any results cached during animation polling.
    Call from _poll_and_apply_engine_results() to process these.
    """
    if not hasattr(modal, '_cached_other_results'):
        return []
    results = modal._cached_other_results
    modal._cached_other_results = []
    return results