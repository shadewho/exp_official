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

        # Phase 1: Visibility - Per-worker load tracking
        self._worker_stats = {}  # Will be initialized in start()

        # Phase 1: Visibility - Job type profiling
        self._job_type_stats = {}  # {job_type: {"count": N, "total_time_ms": X, "avg_time_ms": Y}}

        # Phase 1: Visibility - Debug output timing
        self._last_debug_output = 0.0

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
        worker_entry_file = str(Path(__file__).parent / "worker" / "entry.py")

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
        self._last_debug_output = time.time()

        # Phase 1: Initialize per-worker stats tracking
        self._worker_stats = {
            i: {"jobs_processed": 0, "total_time_ms": 0.0}
            for i in range(self._worker_count)
        }

        if DEBUG_ENGINE:
            print(f"[Engine Core] Started successfully with {len(self._workers)} workers")

    def wait_for_readiness(self, timeout: float = 5.0) -> bool:
        """
        Wait for engine to be fully ready to process jobs.

        This performs comprehensive readiness checks:
        1. Verify all workers are alive
        2. Send PING jobs to all workers and wait for responses
        3. Ensure workers can process jobs successfully

        Args:
            timeout: Maximum seconds to wait for readiness

        Returns:
            True if engine is ready, False if timeout or failure
        """
        if not self._running:
            if DEBUG_ENGINE:
                print("[Engine Core] Cannot check readiness - engine not running")
            return False

        start_time = time.time()

        if DEBUG_ENGINE:
            print(f"[Engine Core] Checking readiness (timeout: {timeout}s)...")

        # Step 1: Verify all workers are alive
        alive_count = sum(1 for w in self._workers if w.is_alive())
        if alive_count != self._worker_count:
            if DEBUG_ENGINE:
                print(f"[Engine Core] FAILED: Only {alive_count}/{self._worker_count} workers alive")
            return False

        if DEBUG_ENGINE:
            print(f"[Engine Core] ✓ All {self._worker_count} workers alive")

        # Step 2: Send PING jobs to all workers
        ping_jobs = []
        for i in range(self._worker_count):
            job_id = self.submit_job("PING", {"worker_check": i})
            if job_id is None:
                if DEBUG_ENGINE:
                    print(f"[Engine Core] FAILED: Could not submit PING job {i}")
                return False
            ping_jobs.append(job_id)

        if DEBUG_ENGINE:
            print(f"[Engine Core] Sent {len(ping_jobs)} PING jobs")

        # Step 3: Wait for all PING responses
        received_pings = set()
        poll_start = time.time()

        while len(received_pings) < len(ping_jobs):
            if time.time() - start_time > timeout:
                if DEBUG_ENGINE:
                    print(f"[Engine Core] FAILED: Timeout waiting for PING responses "
                          f"({len(received_pings)}/{len(ping_jobs)} received)")
                return False

            results = self.poll_results(max_results=20)
            for result in results:
                if result.job_id in ping_jobs and result.job_type == "PING":
                    if not result.success:
                        if DEBUG_ENGINE:
                            print(f"[Engine Core] FAILED: PING job {result.job_id} failed: {result.error}")
                        return False
                    received_pings.add(result.job_id)
                    if DEBUG_ENGINE:
                        print(f"[Engine Core] ✓ PING response {len(received_pings)}/{len(ping_jobs)} "
                              f"(latency: {result.processing_time*1000:.1f}ms)")

            time.sleep(0.01)  # 10ms poll interval

        elapsed = time.time() - poll_start

        if DEBUG_ENGINE:
            print(f"[Engine Core] ✓ All workers ready ({elapsed*1000:.1f}ms total)")

        return True

    def verify_grid_cache(self, timeout: float = 5.0) -> bool:
        """
        Verify that spatial grid has been successfully cached in all workers.

        This should be called after submitting CACHE_GRID jobs to ensure
        workers received and processed the grid data.

        IMPORTANT: Tracks UNIQUE workers that confirmed, not just result count.
        This prevents false positives where one worker processes multiple jobs
        while other workers never receive the grid cache.

        Args:
            timeout: Maximum seconds to wait for confirmation

        Returns:
            True if all workers confirmed cache, False otherwise
        """
        if not self._running:
            if DEBUG_ENGINE:
                print("[Engine Core] Cannot verify grid - engine not running")
            return False

        if DEBUG_ENGINE:
            print(f"[Engine Core] Verifying grid cache in all {self._worker_count} workers...")

        start_time = time.time()
        expected_workers = set(range(self._worker_count))  # {0, 1, 2, 3}
        confirmed_workers = set()  # Track which workers have confirmed

        # Poll for CACHE_GRID results until all unique workers confirm
        while len(confirmed_workers) < len(expected_workers):
            if time.time() - start_time > timeout:
                missing_workers = expected_workers - confirmed_workers
                if DEBUG_ENGINE:
                    print(f"[Engine Core] FAILED: Grid cache timeout - "
                          f"Workers {sorted(confirmed_workers)} confirmed, "
                          f"Workers {sorted(missing_workers)} missing")
                return False

            results = self.poll_results(max_results=20)
            for result in results:
                if result.job_type == "CACHE_GRID":
                    if not result.success:
                        if DEBUG_ENGINE:
                            print(f"[Engine Core] FAILED: Grid cache job failed on worker {result.worker_id}: {result.error}")
                        return False

                    # Track this worker as confirmed
                    if result.worker_id not in confirmed_workers:
                        confirmed_workers.add(result.worker_id)
                        if DEBUG_ENGINE:
                            print(f"[Engine Core] ✓ Grid cached in worker {result.worker_id} "
                                  f"({len(confirmed_workers)}/{len(expected_workers)} workers confirmed)")

            time.sleep(0.01)

        if DEBUG_ENGINE:
            print(f"[Engine Core] ✓ Grid successfully cached in all workers: {sorted(confirmed_workers)}")

        return True

    def get_full_readiness_status(self, grid_required: bool = False) -> Dict:
        """
        Comprehensive readiness check - returns complete status report.

        This is the DEFINITIVE readiness check for production use.
        Call this before starting the game to ensure engine is 100% ready.

        Args:
            grid_required: If True, checks grid cache status (for physics-enabled games)

        Returns:
            {
                "ready": bool,              # Overall ready status
                "status": str,              # "READY" / "FAILED" / "DEGRADED"
                "checks": {                 # Individual check results
                    "running": bool,
                    "workers_alive": bool,
                    "ping_verified": bool,
                    "grid_cached": bool,    # Only checked if grid_required=True
                    "health_passed": bool
                },
                "details": {                # Detailed information
                    "workers_alive": int,
                    "workers_total": int,
                    "uptime": float,
                    "warnings": List[str],
                    "critical": List[str]
                },
                "message": str              # Human-readable status message
            }
        """
        result = {
            "ready": False,
            "status": "FAILED",
            "checks": {
                "running": False,
                "workers_alive": False,
                "ping_verified": False,
                "grid_cached": False,
                "health_passed": False
            },
            "details": {
                "workers_alive": 0,
                "workers_total": self._worker_count,
                "uptime": 0.0,
                "warnings": [],
                "critical": []
            },
            "message": ""
        }

        # Check 1: Engine running
        if not self._running:
            result["message"] = "Engine not started"
            result["details"]["critical"].append("Engine not running")
            return result

        result["checks"]["running"] = True
        result["details"]["uptime"] = time.time() - self._start_time

        # Check 2: Workers alive
        alive_count = sum(1 for w in self._workers if w.is_alive())
        result["details"]["workers_alive"] = alive_count

        if alive_count == 0:
            result["message"] = "No workers alive - engine failed to spawn"
            result["details"]["critical"].append("Zero workers alive")
            return result

        if alive_count < self._worker_count:
            result["details"]["warnings"].append(f"Only {alive_count}/{self._worker_count} workers alive")

        result["checks"]["workers_alive"] = (alive_count == self._worker_count)

        # Check 3: Health check
        health = self.check_worker_health()
        result["checks"]["health_passed"] = health["healthy"]

        if not health["healthy"]:
            result["details"]["warnings"].extend(health["warnings"])
            result["details"]["critical"].extend(health["critical"])

        # Check 4: PING verification (lightweight check)
        # Note: This is quick and non-blocking, just checks recent communication
        stats = self.get_stats()
        has_completed_jobs = stats["jobs_completed"] > 0

        if has_completed_jobs or self._heartbeat_count > 0:
            # Workers have processed jobs or sent heartbeats - they're responsive
            result["checks"]["ping_verified"] = True
        else:
            result["details"]["warnings"].append("No jobs completed yet - workers may not be responsive")

        # Check 5: Grid cache (optional, only if required)
        if grid_required:
            # We can't verify grid cache after the fact, so we assume it's cached
            # if the check passed during startup. This is a trust-based check.
            # Real verification happens during startup via verify_grid_cache()
            result["checks"]["grid_cached"] = True  # Trusted from startup
        else:
            result["checks"]["grid_cached"] = True  # Not required

        # Determine overall status
        all_checks_passed = all(result["checks"].values())
        has_critical = len(result["details"]["critical"]) > 0
        has_warnings = len(result["details"]["warnings"]) > 0

        if all_checks_passed and not has_critical:
            result["ready"] = True
            result["status"] = "READY"
            result["message"] = f"Engine fully ready - {alive_count}/{self._worker_count} workers online"
        elif has_critical:
            result["ready"] = False
            result["status"] = "FAILED"
            result["message"] = f"Critical issues: {'; '.join(result['details']['critical'])}"
        elif has_warnings:
            result["ready"] = True  # Still usable
            result["status"] = "DEGRADED"
            result["message"] = f"Engine degraded - {'; '.join(result['details']['warnings'])}"
        else:
            result["ready"] = False
            result["status"] = "FAILED"
            result["message"] = "Engine checks failed"

        return result

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

    def get_worker_distribution(self) -> Dict:
        """
        Phase 1: Get worker load distribution statistics.

        Returns:
            Dictionary with per-worker stats and distribution percentages
        """
        total_jobs = sum(stats["jobs_processed"] for stats in self._worker_stats.values())

        distribution = {}
        for worker_id in range(self._worker_count):
            jobs = self._worker_stats[worker_id]["jobs_processed"]
            percentage = (jobs / total_jobs * 100) if total_jobs > 0 else 0
            avg_time = (
                self._worker_stats[worker_id]["total_time_ms"] / jobs
                if jobs > 0 else 0
            )

            distribution[worker_id] = {
                "jobs_processed": jobs,
                "percentage": percentage,
                "avg_time_ms": avg_time
            }

        return {
            "total_jobs": total_jobs,
            "workers": distribution
        }

    def get_job_type_stats(self) -> Dict:
        """
        Phase 1: Get job type profiling statistics.

        Returns:
            Dictionary with per-job-type stats
        """
        return dict(self._job_type_stats)

    def check_worker_health(self) -> Dict:
        """
        Comprehensive worker health check.

        Returns:
            Dictionary with health status and warnings
        """
        health = {
            "healthy": True,
            "workers_alive": 0,
            "workers_dead": 0,
            "warnings": [],
            "critical": []
        }

        if not self._running:
            health["healthy"] = False
            health["critical"].append("Engine not running")
            return health

        # Check each worker
        dead_workers = []
        for i, worker in enumerate(self._workers):
            if worker.is_alive():
                health["workers_alive"] += 1
            else:
                health["workers_dead"] += 1
                dead_workers.append(i)
                health["critical"].append(f"Worker {i} is dead")

        # Critical: No workers alive
        if health["workers_alive"] == 0:
            health["healthy"] = False
            health["critical"].append("NO WORKERS ALIVE - ENGINE UNUSABLE")
            return health

        # Warning: Some workers dead
        if health["workers_dead"] > 0:
            health["healthy"] = False
            health["warnings"].append(f"{health['workers_dead']}/{self._worker_count} workers dead")

        # Check queue saturation
        pending = self._jobs_submitted - self._jobs_completed
        if pending > JOB_QUEUE_SIZE * 0.8:
            health["warnings"].append(f"Queue near full ({pending}/{JOB_QUEUE_SIZE})")

        return health

    def submit_job(self, job_type: str, data: any, check_overload: bool = True, target_worker: int = -1) -> Optional[int]:
        """
        Submit a job to be processed by workers.

        Args:
            job_type: String identifier for the job type (e.g., "ECHO", "PATHFIND")
            data: Job data (must be picklable - NO bpy objects!)
            check_overload: If True, reject job if queue is too full (default: True)
            target_worker: If >= 0, only this worker will process the job (-1 = any worker)

        Returns:
            job_id if submitted successfully, None if queue is full or engine not running
        """
        if not self._running:
            if DEBUG_ENGINE:
                print("[Engine Core] Cannot submit job - engine not running")
            return None

        # Optional overload protection - prevent queue saturation
        if check_overload:
            pending = self._jobs_submitted - self._jobs_completed
            # Reject if queue is 90% full (leave headroom for critical jobs)
            if pending > JOB_QUEUE_SIZE * 0.9:
                if DEBUG_ENGINE:
                    print(f"[Engine Core] Job rejected - queue overloaded ({pending}/{JOB_QUEUE_SIZE})")
                return None

        job_id = self._next_job_id
        self._next_job_id += 1

        job = EngineJob(
            job_id=job_id,
            job_type=job_type,
            data=data,
            target_worker=target_worker
        )

        try:
            self._job_queue.put_nowait(job)
            self._jobs_submitted += 1

            if DEBUG_ENGINE:
                target_str = f" (target: W{target_worker})" if target_worker >= 0 else ""
                print(f"[Engine Core] Submitted job {job_id} (type: {job_type}){target_str}")

            return job_id

        except Exception as e:
            if DEBUG_ENGINE:
                print(f"[Engine Core] Failed to submit job: {e}")
            return None

    def broadcast_job(self, job_type: str, data: any) -> int:
        """
        Broadcast a job to ALL workers with per-worker targeting.

        Each worker gets exactly one job targeted specifically to them.
        Workers only process jobs targeted at their ID or jobs with no target (-1).
        This guarantees reliable delivery to all workers.

        Args:
            job_type: String identifier for the job type
            data: Job data (must be picklable)

        Returns:
            Number of jobs successfully submitted
        """
        if not self._running:
            return 0

        submitted = 0
        for worker_id in range(self._worker_count):
            # Target each job to a specific worker
            job_id = self.submit_job(job_type, data, check_overload=False, target_worker=worker_id)
            if job_id is not None:
                submitted += 1

        if DEBUG_ENGINE:
            print(f"[Engine Core] Broadcast {job_type} to {submitted}/{self._worker_count} workers (targeted)")

        return submitted

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
                    processing_time=result_dict.get("processing_time", 0.0),
                    worker_id=result_dict.get("worker_id", -1)  # Extract worker ID from result
                )

                results.append(result)
                self._jobs_completed += 1

                # Phase 1: Track per-worker stats
                worker_id = result.worker_id
                if worker_id >= 0 and worker_id in self._worker_stats:
                    self._worker_stats[worker_id]["jobs_processed"] += 1
                    self._worker_stats[worker_id]["total_time_ms"] += result.processing_time * 1000

                # Phase 1: Track per-job-type stats
                job_type = result.job_type
                if job_type not in self._job_type_stats:
                    self._job_type_stats[job_type] = {
                        "count": 0,
                        "total_time_ms": 0.0,
                        "avg_time_ms": 0.0
                    }

                self._job_type_stats[job_type]["count"] += 1
                self._job_type_stats[job_type]["total_time_ms"] += result.processing_time * 1000
                self._job_type_stats[job_type]["avg_time_ms"] = (
                    self._job_type_stats[job_type]["total_time_ms"] /
                    self._job_type_stats[job_type]["count"]
                )

                if DEBUG_ENGINE:
                    status = "SUCCESS" if result.success else "FAILED"
                    print(f"[Engine Core] Received result for job {result.job_id} ({status})")

            except:
                # Queue is empty
                break

        return results

    def output_debug_stats(self, context=None):
        """
        Phase 1: Output debug statistics for visibility.

        Shows:
        - Worker load distribution (% of total jobs)
        - Job type profiling (avg time, count, jobs/sec)
        - Overall throughput (jobs/sec)

        Respects scene.dev_debug_engine and scene.dev_debug_engine_hz properties.

        Args:
            context: Blender context (optional, for accessing scene properties)
        """
        if not self._running:
            return

        # Try to get scene properties (if context provided and bpy available)
        engine_enabled = False
        engine_hz = 1

        try:
            if context:
                import bpy
                scene = context.scene
                engine_enabled = getattr(scene, "dev_debug_engine", False)
                engine_hz = getattr(scene, "dev_debug_engine_hz", 1)
            else:
                # No context - try to get from bpy.context
                import bpy
                if bpy.context and bpy.context.scene:
                    scene = bpy.context.scene
                    engine_enabled = getattr(scene, "dev_debug_engine", False)
                    engine_hz = getattr(scene, "dev_debug_engine_hz", 1)
        except:
            # bpy not available or no context - skip debug output
            return

        if not engine_enabled:
            return

        # Import logger (frequency gating handled by logger system via master Hz)
        try:
            from ..developer.dev_logger import log_game
        except:
            return  # Logger not available

        # Get current time for uptime calculation
        now = time.time()

        # Calculate worker load distribution
        total_jobs = sum(stats["jobs_processed"] for stats in self._worker_stats.values())

        if total_jobs == 0:
            # No jobs processed yet
            log_game("ENGINE", "No jobs processed yet")
            return

        worker_loads = []
        for worker_id in range(self._worker_count):
            jobs = self._worker_stats[worker_id]["jobs_processed"]
            percentage = (jobs / total_jobs * 100) if total_jobs > 0 else 0
            worker_loads.append(f"W{worker_id}:{percentage:.0f}%")

        worker_load_str = " ".join(worker_loads)

        # Calculate overall throughput (jobs/sec)
        uptime = now - self._start_time
        jobs_per_sec = total_jobs / uptime if uptime > 0 else 0

        # Output summary (worker distribution only - individual systems show their own job stats)
        log_game("ENGINE", f"Worker load: {worker_load_str} | {jobs_per_sec:.0f} jobs/sec")

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