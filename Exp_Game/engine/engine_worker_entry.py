# Exp_Game/engine/engine_worker_entry.py
"""
Worker process entry point - ISOLATED from addon imports.
This module is called directly by multiprocessing and does NOT import bpy.
IMPORTANT: This file has NO relative imports to avoid triggering addon __init__.py
"""

import time
import traceback
from queue import Empty
from dataclasses import dataclass
from typing import Any, Optional

# ============================================================================
# INLINE DEFINITIONS - Use DICTS instead of dataclasses for pickle safety
# ============================================================================

# Workers receive jobs as objects (sent from main thread)
# but RETURN results as plain dicts (pickle-safe)

# Debug flag (hardcoded to avoid config import)
# Controlled by scene.dev_debug_engine in the Developer Tools panel (main thread)
DEBUG_ENGINE = False


def process_job(job) -> dict:
    """
    Process a single job and return result as a plain dict (pickle-safe).
    IMPORTANT: NO bpy access here!
    """
    start_time = time.perf_counter()  # Higher precision than time.time()

    try:
        # ===================================================================
        # THIS IS WHERE YOU'LL ADD YOUR LOGIC LATER
        # For now, just echo back to prove it works
        # ===================================================================

        if job.job_type == "ECHO":
            # Simple echo test
            result_data = {
                "echoed": job.data,
                "worker_msg": "Job processed successfully"
            }

        elif job.job_type == "FRAME_SYNC_TEST":
            # Frame synchronization test - lightweight job for latency measurement
            # Echoes back frame number and timestamp to measure round-trip time
            result_data = {
                "frame": job.data.get("frame", -1),
                "submit_timestamp": job.data.get("timestamp", 0.0),
                "process_timestamp": time.time(),
                "worker_id": job.data.get("worker_id", -1),
                "worker_msg": "Sync test completed"
            }

        elif job.job_type == "COMPUTE_HEAVY":
            # Stress test - simulate realistic game calculation
            # (e.g., pathfinding, physics prediction, AI decision)
            iterations = job.data.get("iterations", 10)  # Reduced to 10 for realistic 1-5ms jobs
            data = job.data.get("data", [])

            # DIAGNOSTIC: Print actual iteration value being used
            if DEBUG_ENGINE:
                print(f"[Worker] COMPUTE_HEAVY job - iterations={iterations}, data_size={len(data)}")

            # Simulate realistic computation (1-5ms per job)
            # This mimics real game calculations like:
            # - AI pathfinding node evaluation
            # - Physics collision prediction
            # - Batch distance calculations
            total = 0
            for i in range(iterations):
                for val in data:
                    total += val * i
                    # Add some realistic computation
                    total = (total * 31 + val) % 1000000

            result_data = {
                "iterations_completed": iterations,
                "data_size": len(data),
                "result": total,
                "worker_msg": f"Completed {iterations} iterations",
                # Echo back metadata for tracking
                "scenario": job.data.get("scenario", "UNKNOWN"),
                "frame": job.data.get("frame", -1),
            }

        elif job.job_type == "CULL_BATCH":
            # Performance culling - distance-based object visibility
            # This is pure math (NO bpy access) and can run in parallel
            entry_ptr = job.data.get("entry_ptr", 0)
            obj_names = job.data.get("obj_names", [])
            obj_positions = job.data.get("obj_positions", [])
            ref_loc = job.data.get("ref_loc", (0, 0, 0))
            thresh = job.data.get("thresh", 10.0)
            start = job.data.get("start", 0)
            max_count = job.data.get("max_count", 100)

            # Compute distance-based culling (inline to avoid imports)
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
                    # distance^2 compare avoids sqrt
                    far = (dx*dx + dy*dy + dz*dz) > t2
                    changes.append((name, far))
                    i += 1
                    idx = (idx + 1) % n

                result_data = {"entry_ptr": entry_ptr, "next_idx": idx, "changes": changes}

        elif job.job_type == "DYNAMIC_MESH_ACTIVATION":
            # Track worker execution for verification
            calc_start = time.perf_counter()
            # Dynamic mesh proximity checks - distance-based activation gating
            # Pure math (NO bpy access) - determines which meshes should be active
            mesh_positions = job.data.get("mesh_positions", [])
            mesh_objects = job.data.get("mesh_objects", [])  # List of (obj, prev_active) tuples
            player_position = job.data.get("player_position", (0, 0, 0))
            base_distances = job.data.get("base_distances", [])

            px, py, pz = player_position
            activation_decisions = []

            for i, (mesh_pos, (obj_name, prev_active), base_dist) in enumerate(zip(mesh_positions, mesh_objects, base_distances)):
                # Special case: base_dist = 0 means NO distance gating (always active)
                if base_dist <= 0.0:
                    activation_decisions.append((obj_name, True, prev_active))
                    continue

                mx, my, mz = mesh_pos

                # Calculate squared distance (avoid sqrt)
                dx = mx - px
                dy = my - py
                dz = mz - pz
                dist_squared = dx*dx + dy*dy + dz*dz

                # Hysteresis: avoid activation flapping
                # If previously active, add 10% margin before deactivating
                # If previously inactive, subtract 10% margin before activating
                margin = base_dist * 0.10
                if prev_active:
                    threshold = base_dist + margin
                else:
                    threshold = max(0.0, base_dist - margin)

                # Compare squared distances (no sqrt needed)
                should_activate = (dist_squared <= (threshold * threshold))

                activation_decisions.append((obj_name, should_activate, prev_active))

            calc_end = time.perf_counter()
            calc_time_us = (calc_end - calc_start) * 1_000_000  # microseconds

            result_data = {
                "activation_decisions": activation_decisions,
                "count": len(activation_decisions),
                "calc_time_us": calc_time_us  # Prove worker did the work
            }

        elif job.job_type == "INTERACTION_CHECK_BATCH":
            # Interaction proximity & collision checks
            # Pure math (NO bpy access) - determines which interactions are triggered
            calc_start = time.perf_counter()

            interactions = job.data.get("interactions", [])
            player_position = job.data.get("player_position", (0, 0, 0))

            triggered_indices = []
            px, py, pz = player_position

            for i, inter_data in enumerate(interactions):
                inter_type = inter_data.get("type")

                if inter_type == "PROXIMITY":
                    # Distance check
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
                    # AABB overlap check
                    aabb_a = inter_data.get("aabb_a")  # (minx, maxx, miny, maxy, minz, maxz)
                    aabb_b = inter_data.get("aabb_b")
                    margin = inter_data.get("margin", 0.0)

                    if aabb_a and aabb_b:
                        # Unpack AABBs
                        a_minx, a_maxx, a_miny, a_maxy, a_minz, a_maxz = aabb_a
                        b_minx, b_maxx, b_miny, b_maxy, b_minz, b_maxz = aabb_b

                        # Apply margin
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

                        # Check overlap
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

        else:
            # Unknown job type - still succeed but note it
            result_data = {
                "message": f"Unknown job type '{job.job_type}' - no handler registered",
                "data": job.data
            }

        processing_time = time.perf_counter() - start_time

        # Return plain dict (pickle-safe)
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
        # Capture any errors and return them safely
        processing_time = time.perf_counter() - start_time

        # Return plain dict (pickle-safe)
        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "result": None,
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            "timestamp": time.perf_counter(),
            "processing_time": processing_time
        }


def worker_loop(job_queue, result_queue, worker_id, shutdown_event):
    """
    Main loop for a worker process.
    This is the entry point called by multiprocessing.Process.
    """
    # DIAGNOSTIC: Print worker info on startup
    import sys
    print(f"[Engine Worker {worker_id}] ========== STARTUP DIAGNOSTIC ==========")
    print(f"[Engine Worker {worker_id}] Loaded from: {__file__}")
    print(f"[Engine Worker {worker_id}] Python executable: {sys.executable}")
    print(f"[Engine Worker {worker_id}] DEBUG_ENGINE = {DEBUG_ENGINE}")
    print(f"[Engine Worker {worker_id}] COMPUTE_HEAVY default iterations: 10 (line 48)")
    print(f"[Engine Worker {worker_id}] ========================================")

    if DEBUG_ENGINE:
        print(f"[Engine Worker {worker_id}] Started")

    jobs_processed = 0

    try:
        while not shutdown_event.is_set():
            try:
                # Wait for a job (with timeout so we can check shutdown_event)
                job = job_queue.get(timeout=0.1)

                if DEBUG_ENGINE:
                    print(f"[Engine Worker {worker_id}] Processing job {job.job_id} (type: {job.job_type})")

                # Process the job
                result = process_job(job)

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
