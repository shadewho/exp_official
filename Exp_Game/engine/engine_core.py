# Exp_Game/engine/engine_core.py
"""
Core engine manager - spawns and manages worker processes.
This runs on the main thread and coordinates with the modal operator.
"""

import multiprocessing as mp
import time
import os
from pathlib import Path
from typing import Optional, List, Dict
from .engine_types import EngineJob, EngineResult, EngineHeartbeat
from .engine_config import (
    WORKER_COUNT,
    JOB_QUEUE_SIZE,
    RESULT_QUEUE_SIZE,
    HEARTBEAT_INTERVAL,
    SHUTDOWN_TIMEOUT,
    DEBUG_ENGINE
)


def _run_worker_from_file(bootstrap_path, worker_path, job_queue, result_queue, worker_id, shutdown_event):
    """
    Module-level function that loads and runs worker bootstrap.
    This function is picklable since it's defined at module level.
    """
    import importlib.util

    # Load bootstrap module
    spec = importlib.util.spec_from_file_location("_worker_bootstrap", bootstrap_path)
    bootstrap_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap_mod)

    # Call bootstrap function
    bootstrap_mod.bootstrap_worker(worker_path, job_queue, result_queue, worker_id, shutdown_event)


class EngineCore:
    """
    Main engine manager that spawns and coordinates worker processes.

    This class handles:
    - Starting worker processes
    - Submitting jobs to workers
    - Polling results from workers
    - Heartbeat monitoring
    - Graceful shutdown

    Usage:
        engine = EngineCore()
        engine.start()

        # Submit a job
        job_id = engine.submit_job("ECHO", {"message": "Hello"})

        # Poll for results
        results = engine.poll_results()

        # Shutdown
        engine.shutdown()
    """

    def __init__(self):
        # Ensure we use 'spawn' method for cross-platform consistency
        # (Windows only supports spawn, so we use it everywhere)
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, that's fine
            pass

        # Communication queues
        self._job_queue: Optional[mp.Queue] = None
        self._result_queue: Optional[mp.Queue] = None

        # Worker processes
        self._workers: List[mp.Process] = []
        self._shutdown_event: Optional[mp.Event] = None

        # State tracking
        self._running = False
        self._start_time = 0.0
        self._next_job_id = 1
        self._jobs_submitted = 0
        self._jobs_completed = 0

        # Heartbeat tracking
        self._last_heartbeat = 0.0
        self._heartbeat_count = 0

        # Determine actual worker count (don't exceed CPU cores)
        cpu_count = os.cpu_count() or 1
        self._worker_count = min(WORKER_COUNT, max(1, cpu_count - 1))

        if DEBUG_ENGINE:
            print(f"[Engine Core] Initialized (will use {self._worker_count} workers)")

    def start(self):
        """
        Start the engine and spawn worker processes.
        Call this when the modal operator starts.
        """
        if self._running:
            if DEBUG_ENGINE:
                print("[Engine Core] Already running, ignoring start request")
            return

        if DEBUG_ENGINE:
            print(f"[Engine Core] Starting with {self._worker_count} workers...")

        # Create communication infrastructure
        self._job_queue = mp.Queue(maxsize=JOB_QUEUE_SIZE)
        self._result_queue = mp.Queue(maxsize=RESULT_QUEUE_SIZE)
        self._shutdown_event = mp.Event()

        # Get paths for bootstrap and worker
        bootstrap_file = str(Path(__file__).parent / "worker_bootstrap.py")
        worker_entry_file = str(Path(__file__).parent / "engine_worker_entry.py")

        # Spawn worker processes using isolated bootstrap
        # Pass bootstrap and worker paths as strings to avoid pickling issues
        self._workers = []
        for i in range(self._worker_count):
            worker = mp.Process(
                target=_run_worker_from_file,
                args=(bootstrap_file, worker_entry_file, self._job_queue, self._result_queue, i, self._shutdown_event),
                name=f"ExpEngine-Worker-{i}"
            )
            worker.start()
            self._workers.append(worker)

        self._running = True
        self._start_time = time.time()
        self._last_heartbeat = time.time()

        if DEBUG_ENGINE:
            print(f"[Engine Core] Started successfully with {len(self._workers)} workers")

    def is_alive(self) -> bool:
        """
        Check if the engine is running.

        Returns:
            True if engine is running and at least one worker is alive
        """
        if not self._running:
            return False

        # Check if any workers are still alive
        alive_count = sum(1 for w in self._workers if w.is_alive())

        if alive_count == 0 and self._running:
            if DEBUG_ENGINE:
                print("[Engine Core] WARNING: No workers alive!")
            return False

        return True

    def get_stats(self) -> Dict:
        """
        Get current engine statistics.

        Returns:
            Dictionary with engine stats
        """
        uptime = time.time() - self._start_time if self._running else 0.0
        alive_workers = sum(1 for w in self._workers if w.is_alive()) if self._workers else 0

        return {
            "running": self._running,
            "alive": self.is_alive(),
            "workers_total": self._worker_count,
            "workers_alive": alive_workers,
            "jobs_submitted": self._jobs_submitted,
            "jobs_completed": self._jobs_completed,
            "jobs_pending": self._jobs_submitted - self._jobs_completed,
            "uptime": uptime,
            "heartbeat_count": self._heartbeat_count
        }

    def submit_job(self, job_type: str, data: any) -> Optional[int]:
        """
        Submit a job to be processed by workers.

        Args:
            job_type: String identifier for the job type (e.g., "ECHO", "PATHFIND")
            data: Job data (must be picklable - NO bpy objects!)

        Returns:
            job_id if submitted successfully, None if queue is full or engine not running
        """
        if not self._running:
            if DEBUG_ENGINE:
                print("[Engine Core] Cannot submit job - engine not running")
            return None

        job_id = self._next_job_id
        self._next_job_id += 1

        job = EngineJob(
            job_id=job_id,
            job_type=job_type,
            data=data
        )

        try:
            self._job_queue.put_nowait(job)
            self._jobs_submitted += 1

            if DEBUG_ENGINE:
                print(f"[Engine Core] Submitted job {job_id} (type: {job_type})")

            return job_id

        except Exception as e:
            if DEBUG_ENGINE:
                print(f"[Engine Core] Failed to submit job: {e}")
            return None

    def poll_results(self, max_results: int = 16) -> List[EngineResult]:
        """
        Poll for completed results from workers (non-blocking).

        Args:
            max_results: Maximum number of results to retrieve per poll

        Returns:
            List of EngineResult objects (may be empty)
        """
        if not self._running:
            return []

        results = []

        for _ in range(max_results):
            try:
                result_dict = self._result_queue.get_nowait()

                # Convert dict to EngineResult object
                result = EngineResult(
                    job_id=result_dict["job_id"],
                    job_type=result_dict["job_type"],
                    result=result_dict["result"],
                    success=result_dict["success"],
                    error=result_dict.get("error"),
                    timestamp=result_dict.get("timestamp", time.time()),
                    processing_time=result_dict.get("processing_time", 0.0)
                )

                results.append(result)
                self._jobs_completed += 1

                if DEBUG_ENGINE:
                    status = "SUCCESS" if result.success else "FAILED"
                    print(f"[Engine Core] Received result for job {result.job_id} ({status})")

            except:
                # Queue is empty
                break

        return results

    def send_heartbeat(self):
        """
        Generate and log a heartbeat (called periodically by modal).
        This confirms the engine is alive and responsive.
        """
        if not self._running:
            return

        now = time.time()

        if now - self._last_heartbeat >= HEARTBEAT_INTERVAL:
            self._last_heartbeat = now
            self._heartbeat_count += 1

            heartbeat = EngineHeartbeat(
                timestamp=now,
                worker_count=sum(1 for w in self._workers if w.is_alive()),
                jobs_processed=self._jobs_completed,
                uptime=now - self._start_time
            )

            if DEBUG_ENGINE:
                print(f"[Engine Core] HEARTBEAT #{self._heartbeat_count} - "
                      f"Workers: {heartbeat.worker_count}/{self._worker_count}, "
                      f"Jobs: {heartbeat.jobs_processed}, "
                      f"Uptime: {heartbeat.uptime:.1f}s")

    def shutdown(self):
        """
        Gracefully shutdown the engine and terminate worker processes.
        Call this when the modal operator ends.
        """
        if not self._running:
            return

        if DEBUG_ENGINE:
            print("[Engine Core] Shutting down...")

        self._running = False

        # Signal workers to stop
        if self._shutdown_event:
            self._shutdown_event.set()

        # Wait for workers to finish (with timeout)
        start_shutdown = time.time()
        for worker in self._workers:
            remaining_time = SHUTDOWN_TIMEOUT - (time.time() - start_shutdown)
            if remaining_time > 0:
                worker.join(timeout=remaining_time)

            # Force terminate if still alive
            if worker.is_alive():
                if DEBUG_ENGINE:
                    print(f"[Engine Core] Force terminating worker {worker.name}")
                worker.terminate()
                worker.join(timeout=0.5)

        # Clear queues
        if self._job_queue:
            while not self._job_queue.empty():
                try:
                    self._job_queue.get_nowait()
                except:
                    break

        if self._result_queue:
            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except:
                    break

        # Clean up
        self._workers.clear()
        self._job_queue = None
        self._result_queue = None
        self._shutdown_event = None

        if DEBUG_ENGINE:
            print(f"[Engine Core] Shutdown complete (processed {self._jobs_completed} jobs in {time.time() - self._start_time:.1f}s)")
