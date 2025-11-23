# Exp_Game/engine/engine_worker.py
"""
Worker process logic - runs in separate process.
IMPORTANT: This code runs in a separate process - NO bpy access allowed!

NOTE: This file is kept for documentation purposes.
The actual worker entry point is engine_worker_entry.py which avoids bpy imports.
"""

import time
import traceback
from .engine_types import EngineJob, EngineResult
from .engine_config import DEBUG_ENGINE


def process_job(job: EngineJob) -> EngineResult:
    """
    Process a single job and return result.

    This is where actual work happens. Currently just a placeholder
    that demonstrates the job was received and processed.

    Args:
        job: The job to process

    Returns:
        EngineResult with the processed result
    """
    start_time = time.time()

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
        else:
            # Unknown job type - still succeed but note it
            result_data = {
                "message": f"Unknown job type '{job.job_type}' - no handler registered",
                "data": job.data
            }

        processing_time = time.time() - start_time

        return EngineResult(
            job_id=job.job_id,
            job_type=job.job_type,
            result=result_data,
            success=True,
            processing_time=processing_time
        )

    except Exception as e:
        # Capture any errors and return them safely
        processing_time = time.time() - start_time

        return EngineResult(
            job_id=job.job_id,
            job_type=job.job_type,
            result=None,
            success=False,
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            processing_time=processing_time
        )


def worker_loop(job_queue, result_queue, worker_id, shutdown_event):
    """
    Main loop for a worker process.

    Runs continuously, pulling jobs from job_queue and putting results in result_queue.

    Args:
        job_queue: multiprocessing.Queue for receiving jobs
        result_queue: multiprocessing.Queue for sending results
        worker_id: Integer ID for this worker (for logging)
        shutdown_event: multiprocessing.Event that signals shutdown
    """
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
                    print(f"[Engine Worker {worker_id}] Completed job {job.job_id} in {result.processing_time*1000:.2f}ms")

            except Exception as e:
                # Handle any queue errors or unexpected issues
                if not shutdown_event.is_set():
                    if DEBUG_ENGINE:
                        print(f"[Engine Worker {worker_id}] Error: {e}")
                continue

    finally:
        if DEBUG_ENGINE:
            print(f"[Engine Worker {worker_id}] Shutting down (processed {jobs_processed} jobs)")
