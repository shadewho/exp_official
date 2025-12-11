#!/usr/bin/env python3
"""
Standalone worker process entry point.
This file is executed as __main__ by multiprocessing and has NO addon imports.
"""

if __name__ == '__main__':
    import time
    import traceback
    from queue import Empty

    # Get arguments from command line (passed by multiprocessing)
    # We'll receive: worker_id, engine_dir via environment or other means

    # For now, let's use a simpler approach - define everything here
    from dataclasses import dataclass
    from typing import Any, Optional

    @dataclass
    class EngineJob:
        job_id: int
        job_type: str
        data: Any
        timestamp: float = None

        def __post_init__(self):
            if self.timestamp is None:
                self.timestamp = time.time()

    @dataclass
    class EngineResult:
        job_id: int
        job_type: str
        result: Any
        success: bool = True
        error: Optional[str] = None
        timestamp: float = None
        processing_time: float = 0.0

        def __post_init__(self):
            if self.timestamp is None:
                self.timestamp = time.time()

    def process_job(job):
        """Process a single job - NO bpy access!"""
        start_time = time.time()

        try:
            if job.job_type == "ECHO":
                result_data = {
                    "echoed": job.data,
                    "worker_msg": "Job processed successfully"
                }
            else:
                result_data = {
                    "message": f"Unknown job type '{job.job_type}'",
                    "data": job.data
                }

            return EngineResult(
                job_id=job.job_id,
                job_type=job.job_type,
                result=result_data,
                success=True,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return EngineResult(
                job_id=job.job_id,
                job_type=job.job_type,
                result=None,
                success=False,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                processing_time=time.time() - start_time
            )

    def worker_main(job_queue, result_queue, worker_id, shutdown_event):
        """Main worker loop"""
        print(f"[Engine Worker {worker_id}] Started")

        jobs_processed = 0

        try:
            while not shutdown_event.is_set():
                try:
                    job = job_queue.get(timeout=0.1)
                    print(f"[Engine Worker {worker_id}] Processing job {job.job_id}")

                    result = process_job(job)
                    result_queue.put(result)

                    jobs_processed += 1
                    print(f"[Engine Worker {worker_id}] Completed job {job.job_id}")

                except Empty:
                    continue
                except Exception as e:
                    if not shutdown_event.is_set():
                        print(f"[Engine Worker {worker_id}] Error: {e}")
                    continue

        finally:
            print(f"[Engine Worker {worker_id}] Shutting down (processed {jobs_processed} jobs)")

    # This gets called by multiprocessing
    # Arguments are passed via pickling
    import multiprocessing
    multiprocessing.freeze_support()
