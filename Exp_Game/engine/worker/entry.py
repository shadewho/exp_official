# Exp_Game/engine/worker/entry.py
"""
Worker process entry point.
Contains the main worker loop and job dispatcher.
This module is loaded by worker_bootstrap.py and runs in isolated worker processes.
IMPORTANT: Uses sys.path manipulation for imports to work with spec_from_file_location.
"""

import time
import traceback
import sys
import os
from queue import Empty
import numpy as np

# Add engine folder to path for worker submodule imports
# This is necessary because bootstrap uses spec_from_file_location
_worker_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_worker_dir)
if _engine_dir not in sys.path:
    sys.path.insert(0, _engine_dir)

# Import from sibling modules (using absolute imports with sys.path)
from worker.math import (
    compute_aabb,
    build_triangle_grid,
)

from worker.physics import handle_kcc_physics_step
from worker.jobs import handle_camera_occlusion
from worker.reactions.transforms import handle_transform_batch
from worker.reactions.hitscan import handle_hitscan_batch
from worker.reactions.projectiles import handle_projectile_update_batch
from worker.reactions.tracking import handle_tracking_batch

# Animation imports (worker-safe, no bpy)
from animations.blend import (
    sample_bone_animation,
    blend_bone_poses,
    sample_object_animation,
    blend_object_transforms,
)


# ============================================================================
# DEBUG FLAG
# ============================================================================
# Controlled by scene.dev_debug_engine in the Developer Tools panel (main thread)
DEBUG_ENGINE = False


# ============================================================================
# WORKER-SIDE CACHES
# ============================================================================

# Grid is sent once via CACHE_GRID job and stored here for all subsequent raycasts.
# This avoids 3MB serialization per raycast (20ms overhead eliminated).
_cached_grid = None

# Will hold dynamic mesh data after CACHE_DYNAMIC_MESH jobs
_cached_dynamic_meshes = {}

# Persistent transform cache: {obj_id: (matrix_16_tuple, world_aabb, inv_matrix)}
# Main thread sends transform updates only when meshes MOVE.
# Worker caches last known transform for each mesh.
_cached_dynamic_transforms = {}

# Animation cache: {anim_name: dict} - stores raw animation dicts, NOT BakedAnimation objects
# Sent once via CACHE_ANIMATIONS job at game start
# Dicts contain: bone_transforms, bone_names, object_transforms, duration, fps, animated_mask
_cached_animations = {}

# Flag for lazy numpy conversion - reset when cache is populated
_animations_numpy_ready = False


# ============================================================================
# ANIMATION HELPER FUNCTIONS
# ============================================================================

def _ensure_numpy_arrays():
    """
    Convert cached animation data from lists to numpy arrays (one-time operation).
    Called lazily on first animation compute after cache is populated.
    Returns list of log messages to report status.
    """
    global _animations_numpy_ready

    logs = []

    if _animations_numpy_ready:
        return logs

    convert_start = time.perf_counter()

    total_bones = 0
    animated_bones = 0

    for anim_name, anim_data in _cached_animations.items():
        # Convert bone_transforms list to numpy array
        bt = anim_data.get("bone_transforms")
        if bt is not None and not isinstance(bt, np.ndarray):
            if len(bt) > 0:
                anim_data["bone_transforms"] = np.array(bt, dtype=np.float32)
                if len(anim_data["bone_transforms"].shape) > 1:
                    total_bones += anim_data["bone_transforms"].shape[1]
            else:
                anim_data["bone_transforms"] = np.empty((0, 0, 10), dtype=np.float32)

        # Convert animated_mask list to numpy array
        am = anim_data.get("animated_mask")
        if am is not None and not isinstance(am, np.ndarray):
            anim_data["animated_mask"] = np.array(am, dtype=bool)
            animated_bones += int(np.sum(anim_data["animated_mask"]))

        # Convert object_transforms if present
        ot = anim_data.get("object_transforms")
        if ot is not None and not isinstance(ot, np.ndarray):
            anim_data["object_transforms"] = np.array(ot, dtype=np.float32)

    convert_ms = (time.perf_counter() - convert_start) * 1000
    _animations_numpy_ready = True

    logs.append(("ANIM-WORKER", f"NUMPY_READY {len(_cached_animations)} anims | {total_bones} bones ({animated_bones} animated) | convert={convert_ms:.1f}ms"))

    return logs


def _compute_single_object_pose(object_name: str, playing_list: list, logs: list) -> dict:
    """
    Compute blended pose for a single object using NUMPY VECTORIZED operations.

    Args:
        object_name: Name of the object
        playing_list: List of playing animation dicts
        logs: List to append log messages to

    Returns:
        {
            "bone_transforms": {bone_name: (10-float tuple), ...},
            "bones_count": int,
            "object_transform": (10-float tuple) or None,
            "anims_blended": int,
        }
    """
    if not playing_list:
        return {
            "bone_transforms": {},
            "bones_count": 0,
            "object_transform": None,
            "anims_blended": 0,
        }

    # Ensure numpy arrays are ready (one-time conversion)
    numpy_logs = _ensure_numpy_arrays()
    logs.extend(numpy_logs)

    # Sample each playing animation using numpy
    poses = []
    bone_weights = []
    object_transforms = []
    object_weights = []
    bone_names = None

    for p in playing_list:
        anim_name = p.get("anim_name")
        anim_time = p.get("time", 0.0)
        weight = p.get("weight", 1.0)
        looping = p.get("looping", True)

        if weight < 0.001:
            continue

        # Get cached animation DICT (not BakedAnimation object!)
        anim_data = _cached_animations.get(anim_name)
        if anim_data is None:
            logs.append(("ANIM-WORKER", f"CACHE_MISS obj={object_name} anim='{anim_name}' cached={list(_cached_animations.keys())}"))
            continue

        # Sample BONE transforms (for armatures)
        pose, sample_stats = sample_bone_animation(anim_data, anim_time, looping)
        if pose.size > 0:
            poses.append(pose)
            bone_weights.append(weight)

            if bone_names is None:
                bone_names = anim_data.get("bone_names", [])

        # Sample OBJECT transforms (for mesh, empty, etc.)
        obj_transform = sample_object_animation(anim_data, anim_time, looping)
        if obj_transform is not None:
            object_transforms.append(obj_transform)
            object_weights.append(weight)

    # Build result
    result = {
        "bone_transforms": {},
        "bones_count": 0,
        "object_transform": None,
        "anims_blended": max(len(poses), len(object_transforms)),
    }

    # Blend bone poses (for armatures)
    if poses:
        blended = blend_bone_poses(poses, bone_weights)

        bone_transforms = {}
        if bone_names and blended.size > 0:
            # Optimized: single numpy conversion + C-level zip instead of Python loop
            # ~5x faster for 100+ bones (avoids per-element tolist() calls)
            bone_transforms = dict(zip(bone_names[:len(blended)], map(tuple, blended.tolist())))

        result["bone_transforms"] = bone_transforms
        result["bones_count"] = len(bone_transforms)

    # Blend object transforms (for non-armature objects)
    if object_transforms:
        blended_obj = blend_object_transforms(object_transforms, object_weights)
        if blended_obj is not None:
            result["object_transform"] = tuple(blended_obj.tolist())

    return result


# ============================================================================
# JOB DISPATCHER
# ============================================================================

def process_job(job) -> dict:
    """
    Process a single job and return result as a plain dict (pickle-safe).
    IMPORTANT: NO bpy access here!
    """
    global _cached_grid, _cached_dynamic_meshes, _cached_dynamic_transforms, _cached_animations, _animations_numpy_ready
    start_time = time.perf_counter()

    try:
        # ===================================================================
        # JOB TYPE DISPATCH
        # ===================================================================

        if job.job_type == "ECHO":
            # Simple echo test
            result_data = {
                "echoed": job.data,
                "worker_msg": "Job processed successfully"
            }

        elif job.job_type == "PING":
            # Worker verification ping - used during startup to confirm worker responsiveness
            result_data = {
                "pong": True,
                "worker_id": job.data.get("worker_check", -1),
                "timestamp": time.time(),
                "worker_msg": "Worker alive and responsive"
            }

        elif job.job_type == "CACHE_GRID":
            # Cache spatial grid for subsequent raycast jobs
            # This is sent ONCE at game start to avoid 3MB serialization per raycast
            grid = job.data.get("grid", None)
            if grid is not None:
                _cached_grid = grid
                tri_count = len(grid.get("triangles", []))
                cell_count = len(grid.get("cells", {}))
                result_data = {
                    "success": True,
                    "triangles": tri_count,
                    "cells": cell_count,
                    "message": "Grid cached successfully"
                }
            else:
                result_data = {
                    "success": False,
                    "error": "No grid data provided"
                }

        elif job.job_type == "CACHE_DYNAMIC_MESH":
            # Cache dynamic mesh triangles in LOCAL space
            # This is sent ONCE per dynamic mesh (or when mesh changes)
            # Per-frame, only transform matrices are sent (64 bytes vs 3MB!)
            #
            # OPTIMIZATION: Pre-compute local AABB once here.
            # Per-frame we only transform 8 AABB corners instead of N*3 vertices.
            obj_id = job.data.get("obj_id")
            triangles = job.data.get("triangles", [])
            radius = job.data.get("radius", 1.0)

            if obj_id is not None and triangles:
                # DIAGNOSTIC: Check if already cached (potential duplicate caching)
                was_cached = obj_id in _cached_dynamic_meshes

                # Compute local-space AABB ONCE (O(N) here, O(8) per frame)
                local_aabb = compute_aabb(triangles)

                # Build spatial grid for O(cells) ray testing instead of O(N)
                # This is the key optimization for high-poly meshes!
                tri_grid = build_triangle_grid(triangles, local_aabb)
                grid_cells = len(tri_grid["cells"]) if tri_grid else 0

                _cached_dynamic_meshes[obj_id] = {
                    "triangles": triangles,   # List of (v0, v1, v2) tuples in local space
                    "local_aabb": local_aabb, # Pre-computed local AABB for fast world transform
                    "radius": radius,         # Bounding sphere radius for quick rejection
                    "grid": tri_grid          # Spatial grid for fast ray-triangle culling
                }
                tri_count = len(triangles)
                grid_res = tri_grid["resolution"] if tri_grid else 0
                if DEBUG_ENGINE:
                    print(f"[Worker] Dynamic mesh cached: obj_id={obj_id} tris={tri_count:,} radius={radius:.2f}m grid={grid_res}³={grid_cells}cells")

                # Log to diagnostics (one-time cache event)
                status = "RE-CACHED" if was_cached else "CACHED"
                total_cached = len(_cached_dynamic_meshes)
                cache_log = ("DYN-CACHE", f"{status} obj_id={obj_id} tris={tri_count} grid={grid_res}³={grid_cells}cells radius={radius:.2f}m total={total_cached}")
                logs = [cache_log]

                result_data = {
                    "success": True,
                    "obj_id": obj_id,
                    "triangle_count": tri_count,
                    "radius": radius,
                    "message": "Dynamic mesh cached successfully",
                    "logs": logs  # Return all logs to main thread
                }
            else:
                result_data = {
                    "success": False,
                    "error": "Missing obj_id or triangles data"
                }

        elif job.job_type == "CLEAR_DYNAMIC_CACHE":
            # Clear dynamic mesh caches - sent on game end/reset to prevent memory leaks
            # Can clear specific mesh (obj_id provided) or all meshes (obj_id=None)
            obj_id = job.data.get("obj_id", None)
            clear_all = job.data.get("clear_all", False)

            cleared_meshes = 0
            cleared_transforms = 0

            if clear_all or obj_id is None:
                # Clear ALL dynamic caches
                cleared_meshes = len(_cached_dynamic_meshes)
                cleared_transforms = len(_cached_dynamic_transforms)
                _cached_dynamic_meshes.clear()
                _cached_dynamic_transforms.clear()
                if DEBUG_ENGINE:
                    print(f"[Worker] Cleared ALL dynamic caches: {cleared_meshes} meshes, {cleared_transforms} transforms")
            else:
                # Clear specific mesh
                if obj_id in _cached_dynamic_meshes:
                    del _cached_dynamic_meshes[obj_id]
                    cleared_meshes = 1
                if obj_id in _cached_dynamic_transforms:
                    del _cached_dynamic_transforms[obj_id]
                    cleared_transforms = 1
                if DEBUG_ENGINE:
                    print(f"[Worker] Cleared dynamic cache for obj_id={obj_id}")

            result_data = {
                "success": True,
                "cleared_meshes": cleared_meshes,
                "cleared_transforms": cleared_transforms,
                "remaining_meshes": len(_cached_dynamic_meshes),
                "remaining_transforms": len(_cached_dynamic_transforms),
                "message": "Dynamic cache cleared"
            }

        elif job.job_type == "COMPUTE_HEAVY":
            # Stress test - simulate realistic game calculation
            # (e.g., pathfinding, physics prediction, AI decision)
            iterations = job.data.get("iterations", 10)
            data = job.data.get("data", [])

            if DEBUG_ENGINE:
                print(f"[Worker] COMPUTE_HEAVY job - iterations={iterations}, data_size={len(data)}")

            # Simulate realistic computation (1-5ms per job)
            total = 0
            for i in range(iterations):
                for val in data:
                    total += val * i
                    total = (total * 31 + val) % 1000000

            result_data = {
                "iterations_completed": iterations,
                "data_size": len(data),
                "result": total,
                "worker_msg": f"Completed {iterations} iterations",
                "scenario": job.data.get("scenario", "UNKNOWN"),
                "frame": job.data.get("frame", -1),
            }

        elif job.job_type == "CULL_BATCH":
            # Performance culling - distance-based object visibility
            entry_ptr = job.data.get("entry_ptr", 0)
            obj_names = job.data.get("obj_names", [])
            obj_positions = job.data.get("obj_positions", [])
            ref_loc = job.data.get("ref_loc", (0, 0, 0))
            thresh = job.data.get("thresh", 10.0)
            start = job.data.get("start", 0)
            max_count = job.data.get("max_count", 100)

            rx, ry, rz = ref_loc
            t2 = float(thresh) * float(thresh)
            n = len(obj_names)

            if n == 0:
                result_data = {"entry_ptr": entry_ptr, "next_idx": start, "changes": []}
            else:
                i = 0
                changes = []
                idx = start % n

                while i < n and len(changes) < max_count:
                    name = obj_names[idx]
                    px, py, pz = obj_positions[idx]
                    dx = px - rx
                    dy = py - ry
                    dz = pz - rz
                    far = (dx*dx + dy*dy + dz*dz) > t2
                    changes.append((name, far))
                    i += 1
                    idx = (idx + 1) % n

                result_data = {"entry_ptr": entry_ptr, "next_idx": idx, "changes": changes}

        elif job.job_type == "INTERACTION_CHECK_BATCH":
            # Interaction proximity & collision checks
            calc_start = time.perf_counter()

            interactions = job.data.get("interactions", [])
            player_position = job.data.get("player_position", (0, 0, 0))

            triggered_indices = []
            px, py, pz = player_position

            for i, inter_data in enumerate(interactions):
                inter_type = inter_data.get("type")

                if inter_type == "PROXIMITY":
                    obj_a_pos = inter_data.get("obj_a_pos")
                    obj_b_pos = inter_data.get("obj_b_pos")
                    threshold = inter_data.get("threshold", 0.0)

                    if obj_a_pos and obj_b_pos:
                        ax, ay, az = obj_a_pos
                        bx, by, bz = obj_b_pos
                        dx = ax - bx
                        dy = ay - by
                        dz = az - bz
                        dist_squared = dx*dx + dy*dy + dz*dz
                        threshold_squared = threshold * threshold

                        if dist_squared <= threshold_squared:
                            triggered_indices.append(i)

                elif inter_type == "COLLISION":
                    aabb_a = inter_data.get("aabb_a")
                    aabb_b = inter_data.get("aabb_b")
                    margin = inter_data.get("margin", 0.0)

                    if aabb_a and aabb_b:
                        a_minx, a_maxx, a_miny, a_maxy, a_minz, a_maxz = aabb_a
                        b_minx, b_maxx, b_miny, b_maxy, b_minz, b_maxz = aabb_b

                        a_minx -= margin
                        a_maxx += margin
                        a_miny -= margin
                        a_maxy += margin
                        a_minz -= margin
                        a_maxz += margin

                        b_minx -= margin
                        b_maxx += margin
                        b_miny -= margin
                        b_maxy += margin
                        b_minz -= margin
                        b_maxz += margin

                        overlap_x = (a_minx <= b_maxx) and (a_maxx >= b_minx)
                        overlap_y = (a_miny <= b_maxy) and (a_maxy >= b_miny)
                        overlap_z = (a_minz <= b_maxz) and (a_maxz >= b_minz)

                        if overlap_x and overlap_y and overlap_z:
                            triggered_indices.append(i)

            calc_end = time.perf_counter()
            calc_time_us = (calc_end - calc_start) * 1_000_000

            result_data = {
                "triggered_indices": triggered_indices,
                "count": len(interactions),
                "calc_time_us": calc_time_us
            }

        elif job.job_type == "KCC_PHYSICS_STEP":
            # Full KCC physics step - delegated to worker.physics module
            result_data = handle_kcc_physics_step(
                job.data,
                _cached_grid,
                _cached_dynamic_meshes,
                _cached_dynamic_transforms
            )

        elif job.job_type == "CAMERA_OCCLUSION_FULL":
            # Camera occlusion - delegated to worker.jobs module
            result_data = handle_camera_occlusion(
                job.data,
                _cached_grid,
                _cached_dynamic_meshes,
                _cached_dynamic_transforms
            )

        elif job.job_type == "TRANSFORM_BATCH":
            # Transform interpolation batch - delegated to worker.reactions.transforms
            result_data = handle_transform_batch(job.data)

        elif job.job_type == "HITSCAN_BATCH":
            # Hitscan raycast batch - delegated to worker.reactions.hitscan
            result_data = handle_hitscan_batch(
                job.data,
                _cached_grid,
                _cached_dynamic_meshes,
                _cached_dynamic_transforms
            )

        elif job.job_type == "PROJECTILE_UPDATE_BATCH":
            # Projectile physics batch - delegated to worker.reactions.projectiles
            result_data = handle_projectile_update_batch(
                job.data,
                _cached_grid,
                _cached_dynamic_meshes,
                _cached_dynamic_transforms
            )

        elif job.job_type == "TRACKING_BATCH":
            # Object tracking batch - delegated to worker.reactions.tracking
            result_data = handle_tracking_batch(
                job.data,
                _cached_grid,
                _cached_dynamic_meshes,
                _cached_dynamic_transforms
            )

        elif job.job_type == "CACHE_ANIMATIONS":
            # Cache animation DICTS for subsequent ANIMATION_COMPUTE_BATCH jobs
            # Sent ONCE at game start to avoid serialization overhead per frame
            # IMPORTANT: Store raw dicts, NOT BakedAnimation objects!
            # Numpy conversion happens lazily via _ensure_numpy_arrays()
            animations_data = job.data.get("animations", {})

            # Clear existing cache and reset numpy flag
            _cached_animations.clear()
            _animations_numpy_ready = False

            cached_count = 0
            total_bones = 0
            cached_names = []

            for anim_name, anim_dict in animations_data.items():
                # Store dict directly - numpy conversion is lazy
                _cached_animations[anim_name] = anim_dict
                cached_count += 1
                cached_names.append(anim_name)
                total_bones += len(anim_dict.get("bone_names", []))

            # DEBUG: Print to worker stdout to confirm caching happened
            print(f"[Worker CACHE_ANIMATIONS] Cached {cached_count} animations: {cached_names}")

            result_data = {
                "success": True,
                "cached_count": cached_count,
                "cached_names": cached_names,
                "total_bones": total_bones,
                "message": f"Cached {cached_count} animations",
                "logs": [("ANIM-CACHE", f"WORKER_CACHED {cached_count} anims: {cached_names}")]
            }

        elif job.job_type == "ANIMATION_COMPUTE_BATCH":
            # Compute blended poses for all animated objects in one batch
            # Input: {"objects": {obj_name: {"playing": [{anim_name, time, weight, looping}, ...]}}}
            # Output: {"results": {obj_name: {"bone_transforms": {...}, "object_transform": ...}}}
            calc_start = time.perf_counter()
            logs = []
            objects_data = job.data.get("objects", {})

            results = {}
            total_bones = 0
            total_anims = 0

            for object_name, obj_data in objects_data.items():
                playing_list = obj_data.get("playing", [])

                # Use helper function that handles numpy conversion and blending
                obj_result = _compute_single_object_pose(object_name, playing_list, logs)

                results[object_name] = {
                    "bone_transforms": obj_result["bone_transforms"],
                    "object_transform": obj_result.get("object_transform"),
                }

                total_bones += obj_result["bones_count"]
                total_anims += obj_result["anims_blended"]

            calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

            result_data = {
                "success": True,
                "results": results,
                "objects_processed": len(results),
                "total_bones_computed": total_bones,
                "total_anims": total_anims,
                "calc_time_us": calc_time_us,
                "logs": logs,
            }

        elif job.job_type == "CLEAR_ANIMATION_CACHE":
            # Clear animation cache - sent on game end/reset
            cleared_count = len(_cached_animations)
            _cached_animations.clear()
            _animations_numpy_ready = False

            result_data = {
                "success": True,
                "cleared_count": cleared_count,
                "message": "Animation cache cleared"
            }

        else:
            # Unknown job type - still succeed but note it
            result_data = {
                "message": f"Unknown job type '{job.job_type}' - no handler registered",
                "data": job.data
            }

        processing_time = time.perf_counter() - start_time

        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "result": result_data,
            "success": True,
            "error": None,
            "timestamp": time.perf_counter(),
            "processing_time": processing_time
        }

    except Exception as e:
        processing_time = time.perf_counter() - start_time

        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "result": None,
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            "timestamp": time.perf_counter(),
            "processing_time": processing_time
        }


# ============================================================================
# WORKER MAIN LOOP
# ============================================================================

def worker_loop(job_queue, result_queue, worker_id, shutdown_event):
    """
    Main loop for a worker process.
    This is the entry point called by multiprocessing.Process.
    """
    if DEBUG_ENGINE:
        print(f"[Engine Worker {worker_id}] Started")

    jobs_processed = 0

    try:
        while not shutdown_event.is_set():
            try:
                # Wait for a job (with timeout so we can check shutdown_event)
                job = job_queue.get(timeout=0.1)

                # Check if this job is targeted at a specific worker
                target = getattr(job, 'target_worker', -1)
                if target >= 0 and target != worker_id:
                    # This job is for a different worker - put it back and try again
                    try:
                        job_queue.put_nowait(job)
                    except Exception:
                        pass  # Queue full, job will be lost (shouldn't happen)
                    continue

                if DEBUG_ENGINE:
                    print(f"[Engine Worker {worker_id}] Processing job {job.job_id} (type: {job.job_type})")

                # Process the job
                result = process_job(job)

                # Add worker_id to result before sending (CRITICAL for grid cache verification)
                result["worker_id"] = worker_id

                # Send result back
                result_queue.put(result)

                jobs_processed += 1

                if DEBUG_ENGINE:
                    print(f"[Engine Worker {worker_id}] Completed job {job.job_id} in {result['processing_time']*1000:.2f}ms")

            except Empty:
                # Queue is empty, just continue
                continue
            except Exception as e:
                # Handle any queue errors or unexpected issues
                if not shutdown_event.is_set():
                    if DEBUG_ENGINE:
                        print(f"[Engine Worker {worker_id}] Error: {e}")
                        traceback.print_exc()
                continue

    finally:
        if DEBUG_ENGINE:
            print(f"[Engine Worker {worker_id}] Shutting down (processed {jobs_processed} jobs)")
