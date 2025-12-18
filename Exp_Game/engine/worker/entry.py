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

# Animation cache: {anim_name: {bones: {bone_name: [frames]}, duration, fps, ...}}
# Sent once via CACHE_ANIMATIONS job. Per-frame, only times/weights are sent.
_cached_animations = {}


# ============================================================================
# ANIMATION COMPUTE (Worker-side blending)
# ============================================================================

def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def _slerp(q1, q2, t: float):
    """
    Spherical linear interpolation between two quaternions.
    Quaternions are (w, x, y, z) format.
    """
    import math
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    # Dot product
    dot = w1*w2 + x1*x2 + y1*y2 + z1*z2

    # Take shorter path
    if dot < 0.0:
        w2, x2, y2, z2 = -w2, -x2, -y2, -z2
        dot = -dot

    # Very close - use linear interpolation
    if dot > 0.9995:
        w = w1 + t * (w2 - w1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        z = z1 + t * (z2 - z1)
        # Normalize
        length = (w*w + x*x + y*y + z*z) ** 0.5
        if length > 0:
            w, x, y, z = w/length, x/length, y/length, z/length
        return (w, x, y, z)

    # Standard slerp
    theta_0 = math.acos(min(1.0, max(-1.0, dot)))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    if abs(sin_theta_0) < 1e-10:
        return (w1, x1, y1, z1)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (
        s0 * w1 + s1 * w2,
        s0 * x1 + s1 * x2,
        s0 * y1 + s1 * y2,
        s0 * z1 + s1 * z2,
    )


def _blend_transform(t1, t2, weight: float):
    """
    Blend two transforms. weight=0 returns t1, weight=1 returns t2.
    Transform format: (qw, qx, qy, qz, lx, ly, lz, sx, sy, sz)
    """
    q = _slerp(t1[0:4], t2[0:4], weight)
    l = (_lerp(t1[4], t2[4], weight),
         _lerp(t1[5], t2[5], weight),
         _lerp(t1[6], t2[6], weight))
    s = (_lerp(t1[7], t2[7], weight),
         _lerp(t1[8], t2[8], weight),
         _lerp(t1[9], t2[9], weight))
    return q + l + s


def _interpolate_transform(frames, frame_float: float):
    """Interpolate between frames at a fractional frame index."""
    if not frames:
        return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    frame_low = int(frame_float)
    frame_high = frame_low + 1
    t = frame_float - frame_low

    # Clamp indices
    max_idx = len(frames) - 1
    frame_low = min(frame_low, max_idx)
    frame_high = min(frame_high, max_idx)

    if frame_low == frame_high or t < 0.001:
        return tuple(frames[frame_low])

    return _blend_transform(frames[frame_low], frames[frame_high], t)


def _sample_animation(anim_data: dict, anim_time: float, loop: bool = True):
    """
    Sample animation at a specific time.

    Returns:
        Dict[bone_name, Transform] - bone transforms at this time
    """
    duration = anim_data.get("duration", 0.0)
    fps = anim_data.get("fps", 30.0)
    bones_data = anim_data.get("bones", {})

    if duration <= 0:
        # Static pose - just return first frame
        return {
            name: tuple(frames[0]) if frames else (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            for name, frames in bones_data.items()
        }

    # Handle looping
    if loop:
        anim_time = anim_time % duration
    else:
        anim_time = max(0.0, min(anim_time, duration))

    frame_float = anim_time * fps

    # Sample each bone
    result = {}
    for bone_name, frames in bones_data.items():
        if frames:
            result[bone_name] = _interpolate_transform(frames, frame_float)

    return result


def _blend_bone_poses(poses_with_weights):
    """
    Blend multiple bone poses by weight.

    Args:
        poses_with_weights: List of (pose_dict, weight) tuples

    Returns:
        Blended pose dict {bone_name: Transform}
    """
    if not poses_with_weights:
        return {}

    if len(poses_with_weights) == 1:
        return poses_with_weights[0][0]

    # Normalize weights
    total_weight = sum(w for _, w in poses_with_weights)
    if total_weight <= 0:
        return poses_with_weights[0][0]

    # Collect all bone names
    all_bones = set()
    for pose, _ in poses_with_weights:
        all_bones.update(pose.keys())

    result = {}
    for bone_name in all_bones:
        bone_data = [(pose[bone_name], w / total_weight)
                     for pose, w in poses_with_weights if bone_name in pose]

        if not bone_data:
            continue

        if len(bone_data) == 1:
            result[bone_name] = bone_data[0][0]
            continue

        # Iterative blending
        blended = bone_data[0][0]
        acc_weight = bone_data[0][1]

        for transform, weight in bone_data[1:]:
            if acc_weight + weight > 0:
                blend_t = weight / (acc_weight + weight)
                blended = _blend_transform(blended, transform, blend_t)
                acc_weight += weight

        result[bone_name] = blended

    return result


def _handle_animation_compute(job_data: dict) -> dict:
    """
    Handle ANIMATION_COMPUTE job - compute blended pose from playing animations.

    Input job_data:
        {
            "object_name": str,
            "playing": [
                {"anim_name": str, "time": float, "weight": float, "looping": bool},
                ...
            ]
        }

    Returns:
        {
            "success": bool,
            "bone_transforms": {bone_name: (10-float tuple), ...},
            "bones_count": int,
            "anims_blended": int,
            "calc_time_us": float,
            "logs": [(category, message), ...]
        }
    """
    calc_start = time.perf_counter()
    logs = []

    object_name = job_data.get("object_name", "Unknown")
    playing_list = job_data.get("playing", [])

    if not playing_list:
        return {
            "success": True,
            "bone_transforms": {},
            "bones_count": 0,
            "anims_blended": 0,
            "calc_time_us": 0.0,
            "logs": []
        }

    # Sample each playing animation
    poses_with_weights = []
    anim_names = []

    for p in playing_list:
        anim_name = p.get("anim_name")
        anim_time = p.get("time", 0.0)
        weight = p.get("weight", 1.0)
        looping = p.get("looping", True)

        if weight < 0.001:
            continue

        # Get cached animation
        anim_data = _cached_animations.get(anim_name)
        if anim_data is None:
            logs.append(("ANIMATIONS", f"CACHE MISS: '{anim_name}' not in worker cache ({len(_cached_animations)} cached)"))
            continue

        # Sample at current time
        pose = _sample_animation(anim_data, anim_time, looping)
        if pose:
            poses_with_weights.append((pose, weight))
            anim_names.append(f"{anim_name}:{weight:.0%}")

    if not poses_with_weights:
        return {
            "success": True,
            "bone_transforms": {},
            "bones_count": 0,
            "anims_blended": 0,
            "calc_time_us": 0.0,
            "logs": logs  # Include any debug logs (e.g., cache misses)
        }

    # Blend all poses
    blended = _blend_bone_poses(poses_with_weights)

    calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

    # Build log message
    anim_str = " + ".join(anim_names) if len(anim_names) <= 3 else f"{len(anim_names)} anims"
    logs.append(("ANIMATIONS", f"{object_name}: {anim_str} | {len(blended)} bones | {calc_time_us:.0f}µs"))

    return {
        "success": True,
        "bone_transforms": blended,
        "bones_count": len(blended),
        "anims_blended": len(poses_with_weights),
        "calc_time_us": calc_time_us,
        "logs": logs
    }


# ============================================================================
# JOB DISPATCHER
# ============================================================================

def process_job(job) -> dict:
    """
    Process a single job and return result as a plain dict (pickle-safe).
    IMPORTANT: NO bpy access here!
    """
    global _cached_grid, _cached_dynamic_meshes, _cached_dynamic_transforms
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

        elif job.job_type == "CACHE_ANIMATIONS":
            # Cache baked animations for subsequent ANIMATION_COMPUTE jobs
            # Sent ONCE at game start. Per-frame, only times/weights are sent.
            animations_data = job.data.get("animations", {})

            if animations_data:
                # Clear existing cache and store new animations
                _cached_animations.clear()
                for anim_name, anim_dict in animations_data.items():
                    _cached_animations[anim_name] = anim_dict

                anim_count = len(_cached_animations)
                total_bones = sum(
                    len(anim.get("bones", {}))
                    for anim in _cached_animations.values()
                )

                if DEBUG_ENGINE:
                    print(f"[Worker] Animations cached: {anim_count} anims, {total_bones} bone channels")

                # Log for diagnostics
                logs = [("ANIMATIONS", f"CACHED {anim_count} animations, {total_bones} bone channels")]

                result_data = {
                    "success": True,
                    "animation_count": anim_count,
                    "total_bone_channels": total_bones,
                    "message": "Animations cached successfully",
                    "logs": logs
                }
            else:
                result_data = {
                    "success": False,
                    "error": "No animation data provided"
                }

        elif job.job_type == "ANIMATION_COMPUTE":
            # Compute blended pose from multiple playing animations
            # Input: list of {anim_name, time, weight} for each playing animation
            # Output: blended bone transforms ready to apply via bpy
            result_data = _handle_animation_compute(job.data)

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
            