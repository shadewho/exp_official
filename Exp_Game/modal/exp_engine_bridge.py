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
    Silent operation - no console output.

    Args:
        modal: ExpModal operator instance
        context: Blender context

    Returns:
        (success: bool, message: str)
    """
    # STEP 1: Spawn Engine
    if modal.engine is None:
        modal.engine = EngineCore()

    modal.engine.start()

    # STEP 2: Verify Workers Alive
    if not modal.engine.is_alive():
        modal.engine.shutdown()
        return False, "Engine workers failed to spawn"

    # STEP 3: PING Verification (adaptive timeout for cold start)
    if not modal.engine.wait_for_readiness(timeout=3.0, extended_timeout=30.0):
        modal.engine.shutdown()
        return False, "Engine workers not responding to PING"

    # Initialize sync tracking
    _init_sync_tracking(modal)

    # STEP 4: Cache Spatial Grid
    if modal.spatial_grid and modal.engine and modal.engine.is_alive():
        modal.engine.broadcast_job("CACHE_GRID", {"grid": modal.spatial_grid})

        if not modal.engine.verify_grid_cache(timeout=5.0, extended_timeout=45.0):
            modal.engine.shutdown()
            return False, "Not all workers cached spatial grid"

    # STEP 5: Final Readiness Confirmation
    final_status = modal.engine.get_full_readiness_status(grid_required=bool(modal.spatial_grid))

    if not final_status["ready"]:
        modal.engine.shutdown()
        return False, f"Engine not ready: {final_status['message']}"

    return True, "Engine ready"


def shutdown_engine(modal, context):
    """
    Shutdown engine gracefully with cache cleanup.
    Silent operation - no console output.

    Args:
        modal: ExpModal operator instance
        context: Blender context
    """
    if not modal.engine:
        return

    # Clear dynamic mesh caches in all workers before shutdown
    try:
        modal.engine.broadcast_job("CLEAR_DYNAMIC_CACHE", {"clear_all": True})
    except Exception:
        pass  # Ignore errors during shutdown

    modal.engine.shutdown()
    modal.engine = None


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
    # PERFORMANCE: Direct access - class-level default is None
    if not modal.engine:
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

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATION LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

def init_animations(modal, context) -> tuple[bool, str]:
    """
    Initialize animation system - bake all actions and cache in workers.

    NO ARMATURE DEPENDENCY - bakes directly from FCurves.
    Supports both bone animations and object animations.

    Args:
        modal: ExpModal operator instance
        context: Blender context

    Returns:
        (success: bool, message: str)
    """
    from ..developer.dev_logger import log_game

    # Create fresh controller
    modal.anim_controller = AnimationController()

    # Bake ALL actions in the blend file - no armature dependency
    total_start = time.perf_counter()
    baked_count = 0
    baked_bones = 0
    baked_objects = 0
    failed = []
    total_frames = 0
    total_bones = 0

    for action in bpy.data.actions:
        try:
            action_start = time.perf_counter()
            anim = bake_action(action)
            action_elapsed = (time.perf_counter() - action_start) * 1000

            modal.anim_controller.add_animation(anim)
            baked_count += 1

            # Track stats
            frame_count = int(action.frame_range[1] - action.frame_range[0]) + 1
            bone_count = anim.num_bones
            total_frames += frame_count
            total_bones += bone_count

            if anim.has_bones:
                baked_bones += 1
            if anim.has_object:
                baked_objects += 1

            # Log per-action timing with type info
            anim_type = []
            if anim.has_bones:
                anim_type.append(f"{bone_count}bones")
            if anim.has_object:
                anim_type.append("obj")
            type_str = "+".join(anim_type) if anim_type else "empty"

            log_game("ANIM-CACHE", f"BAKED {action.name}: {type_str} x {frame_count}frames = {action_elapsed:.0f}ms")

        except Exception as e:
            failed.append(f"{action.name}: {e}")

    bake_elapsed = (time.perf_counter() - total_start) * 1000

    # Summary log
    log_game("ANIM-CACHE", f"BAKE_TOTAL {baked_count}actions ({baked_bones}bone/{baked_objects}obj) {total_bones}bones {total_frames}frames = {bake_elapsed:.0f}ms")

    # NOTE: Worker caching is done AFTER init_engine() in exp_modal.py
    # This function only handles baking - caching requires engine to be running

    return True, f"Baked {baked_count} actions"


def shutdown_animations(modal, context):
    """
    Shutdown animation system - clear all cached data.
    Silent operation - no console output.

    Args:
        modal: ExpModal operator instance
        context: Blender context
    """
    # Clear animation cache in workers
    clear_animation_cache_in_workers(modal)

    # Clear local controller
    if modal.anim_controller:
        modal.anim_controller.clear_all()
        modal.anim_controller = None


def update_animations_state(modal, delta_time: float):
    """
    Update animation state (times, fades). Call at start of frame.
    This prepares data for worker computation via submit_animation_jobs().

    Args:
        modal: ExpModal operator instance
        delta_time: Time since last frame in seconds
    """
    # PERFORMANCE: Direct access - class-level default is None
    if modal.anim_controller:
        modal.anim_controller.update_state(delta_time)


# =============================================================================
# WORKER ANIMATION OFFLOADING (Reinstated 2025-01)
# =============================================================================
# Animation computation offloaded to dedicated worker (worker 0) to free main thread.
# Benefits:
# - Main thread only does bpy writes (unavoidable)
# - Worker handles all sampling, blending, quaternion math
# - Reduces main thread load for scenes with many animated objects
#
# Flow:
#   1. init_animations() bakes all actions and calls cache_animations_in_workers()
#   2. Each frame: submit_animation_jobs() sends batch job to worker 0
#   3. Worker computes blended poses for all objects
#   4. Main thread polls result via process_animation_result() and applies to bpy
# =============================================================================

# Dedicated animation worker (worker 0) has the animation cache
ANIMATION_WORKER_ID = 0


def cache_animations_in_workers(modal, context) -> bool:
    """
    Cache baked animations in the ANIMATION WORKER ONLY.
    MUST be called AFTER init_engine() since it needs workers to be running.

    OPTIMIZATION: Only send cache to worker 0 (animation worker).
    - Saves memory (no duplication across workers)
    - Animation jobs are always routed to this worker anyway

    Args:
        modal: ExpModal operator instance
        context: Blender context

    Returns:
        True if caching succeeded
    """
    from ..developer.dev_logger import log_game

    if not modal.anim_controller:
        return True  # No animations to cache

    if modal.anim_controller.cache.count == 0:
        log_game("ANIM-CACHE", "EMPTY no animations baked")
        return True  # No animations baked

    if not modal.engine or not modal.engine.is_alive():
        log_game("ANIM-CACHE", "SKIP engine not available")
        return False

    # Log animation list
    anim_count = modal.anim_controller.cache.count
    anim_names = list(modal.anim_controller.cache.names)
    log_game("ANIM-CACHE", f"CACHING {anim_count} animations: {anim_names}")

    # Log animation worker designation
    log_game("ANIM-WORKER", f"DESIGNATED worker={ANIMATION_WORKER_ID} for all animation jobs")

    # Measure serialization time
    serialize_start = time.perf_counter()
    cache_data = modal.anim_controller.get_cache_data_for_workers()
    serialize_elapsed = (time.perf_counter() - serialize_start) * 1000

    # Estimate data size
    import sys
    try:
        data_size_kb = sys.getsizeof(str(cache_data)) / 1024
    except:
        data_size_kb = 0

    log_game("ANIM-CACHE", f"SERIALIZED {anim_count} anims in {serialize_elapsed:.0f}ms (~{data_size_kb:.0f}KB)")

    # Send cache to ANIMATION WORKER ONLY (not all workers)
    transfer_start = time.perf_counter()
    job_id = modal.engine.submit_job(
        "CACHE_ANIMATIONS",
        cache_data,
        check_overload=False,
        target_worker=ANIMATION_WORKER_ID
    )

    # Wait for cache confirmation from animation worker
    if job_id is not None and job_id >= 0:
        log_game("ANIM-CACHE", f"SUBMITTED job_id={job_id} waiting for confirmation...")
        start_time = time.perf_counter()
        timeout = 5.0  # 5 second timeout
        confirmed = False

        while not confirmed:
            if time.perf_counter() - start_time > timeout:
                log_game("ANIM-WORKER", f"TIMEOUT worker={ANIMATION_WORKER_ID} cache not confirmed")
                break

            results = modal.engine.poll_results(max_results=50)
            for result in results:
                log_game("ANIM-CACHE", f"POLL_RESULT type={result.job_type} success={result.success} worker_id={result.worker_id}")
                if result.job_type == "CACHE_ANIMATIONS" and result.success:
                    if result.worker_id == ANIMATION_WORKER_ID:
                        confirmed = True
                        break

            if confirmed:
                break
            time.sleep(0.002)

        transfer_elapsed_ms = (time.perf_counter() - transfer_start) * 1000

        if confirmed:
            log_game("ANIM-WORKER", f"CACHE_OK worker={ANIMATION_WORKER_ID} {anim_count}anims ({transfer_elapsed_ms:.0f}ms)")
            return True
        else:
            log_game("ANIM-WORKER", f"CACHE_FAIL worker={ANIMATION_WORKER_ID} timeout after {transfer_elapsed_ms:.0f}ms")
            return False

    log_game("ANIM-CACHE", f"SUBMIT_FAILED job_id={job_id}")
    return False


def submit_animation_jobs(modal) -> int:
    """
    Submit ANIMATION_COMPUTE_BATCH job to ANIMATION WORKER for all active animations.
    BATCHED: One IPC round-trip for ALL animated objects (O(1) instead of O(n)).
    TARGETED: Always routed to ANIMATION_WORKER_ID (worker 0) which has the cache.

    Call after update_animations_state().

    Args:
        modal: ExpModal operator instance

    Returns:
        Number of objects included in batch (0 if no batch submitted)
    """
    if not modal.anim_controller:
        return 0

    if not modal.engine or not modal.engine.is_alive():
        return 0

    # Get job data for all objects with active animations
    jobs_data = modal.anim_controller.get_compute_job_data()

    if not jobs_data:
        return 0  # No active animations

    from ..developer.dev_logger import log_game

    # BATCHED + TARGETED: Submit ONE job to ANIMATION WORKER
    batch_data = {"objects": jobs_data}
    job_id = modal.engine.submit_job(
        "ANIMATION_COMPUTE_BATCH",
        batch_data,
        target_worker=ANIMATION_WORKER_ID
    )

    if job_id is not None and job_id >= 0:
        # Track pending batch job (store list of object names for result processing)
        modal._pending_anim_batch_job = {
            "job_id": job_id,
            "object_names": list(jobs_data.keys())
        }

        # Log batch submission with worker info
        obj_count = len(jobs_data)
        total_anims = sum(len(d.get("playing", [])) for d in jobs_data.values())
        log_game("ANIM", f"SUBMIT job={job_id} objs={obj_count} anims={total_anims} -> worker={ANIMATION_WORKER_ID}")

        return obj_count

    return 0


def process_animation_result(modal, result) -> int:
    """
    Process an ANIMATION_COMPUTE_BATCH result and apply to Blender.

    Args:
        modal: ExpModal operator instance
        result: EngineResult from engine.poll_results()

    Returns:
        Number of objects with transforms applied
    """
    if result.job_type != "ANIMATION_COMPUTE_BATCH":
        return 0

    if not result.success:
        return 0

    # Clear pending job tracker
    if hasattr(modal, '_pending_anim_batch_job'):
        modal._pending_anim_batch_job = None

    result_data = result.result or {}
    results_dict = result_data.get("results", {})

    if not results_dict:
        return 0

    objects_applied = 0

    for obj_name, obj_result in results_dict.items():
        bone_transforms = obj_result.get("bone_transforms", {})
        object_transform = obj_result.get("object_transform")

        # Apply via controller's apply_worker_result method
        count = modal.anim_controller.apply_worker_result(
            obj_name,
            bone_transforms,
            object_transform
        )

        if count > 0:
            objects_applied += 1

    return objects_applied


def poll_animation_results_with_timeout(modal, timeout: float = 0.002) -> int:
    """
    Poll for ANIMATION_COMPUTE_BATCH result with short timeout for same-frame sync.
    BATCHED: Only ONE result to wait for regardless of object count.

    Non-animation results that arrive during polling are cached for later processing
    by _poll_and_apply_engine_results().

    Args:
        modal: ExpModal operator instance
        timeout: Max time to wait in seconds (default 2ms)

    Returns:
        Number of objects whose poses were applied
    """
    if not modal.engine or not modal.engine.is_alive():
        return 0

    # Check for pending batch job
    if not hasattr(modal, '_pending_anim_batch_job') or not modal._pending_anim_batch_job:
        return 0

    # Initialize cached results list if needed (for non-animation results)
    if not hasattr(modal, '_cached_anim_other_results'):
        modal._cached_anim_other_results = []

    # Small pre-poll delay to give worker time to start
    time.sleep(0.00015)  # 150µs

    start_time = time.perf_counter()
    objects_applied = 0

    while time.perf_counter() - start_time < timeout:
        # Non-blocking poll
        results = modal.engine.poll_results(max_results=20)

        for result in results:
            if result.job_type == "ANIMATION_COMPUTE_BATCH":
                objects_applied = process_animation_result(modal, result)

                # Process worker logs
                if result.success:
                    result_data = result.result or {}
                    worker_logs = result_data.get("logs", [])
                    total_objects = result_data.get("objects_processed", 0)
                    total_bones = result_data.get("total_bones_computed", 0)
                    total_anims = result_data.get("total_anims", 0)
                    calc_time_us = result_data.get("calc_time_us", 0)

                    # Log batch result
                    from ..developer.dev_logger import log_game
                    log_game("ANIM", f"BATCH objs={total_objects} bones={total_bones} anims={total_anims} {calc_time_us:.0f}µs worker={result.worker_id}")

                    # Process worker logs (includes CACHE_MISS, NUMPY_READY, etc.)
                    if worker_logs:
                        from ..developer.dev_logger import log_worker_messages
                        log_worker_messages(worker_logs)
            else:
                # Cache non-animation results for later processing
                modal._cached_anim_other_results.append(result)

        # Batch job completed (only one to wait for)
        if not hasattr(modal, '_pending_anim_batch_job') or not modal._pending_anim_batch_job:
            break

        # Brief sleep to avoid busy spin
        time.sleep(0.0001)  # 100µs

    return objects_applied


def get_cached_anim_other_results(modal) -> list:
    """
    Get and clear any results cached during animation polling.
    Call from _poll_and_apply_engine_results() to process these.
    """
    if not hasattr(modal, '_cached_anim_other_results'):
        return []
    results = modal._cached_anim_other_results
    modal._cached_anim_other_results = []
    return results


def clear_animation_cache_in_workers(modal):
    """
    Clear animation cache in workers on game end.

    Args:
        modal: ExpModal operator instance
    """
    if not modal.engine or not modal.engine.is_alive():
        return

    # Broadcast to all workers (they share the cache clearing logic)
    modal.engine.broadcast_job("CLEAR_ANIMATION_CACHE", {})


# NOTE: Pose Library Caching removed (2025) - not used