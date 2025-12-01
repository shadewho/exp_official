# Exp_Game/developer/dev_stats.py
"""
Debug Statistics Tracker

Accumulates metrics between summary prints for meaningful 1Hz output.
Instead of per-frame spam, provides rolling averages and event counts.

Usage:
    from ..developer.dev_stats import get_stats_tracker
    stats = get_stats_tracker()

    # Record a KCC physics step
    stats.record_kcc_step(calc_time_us=145, rays=6, tris=234, blocked=True, step_up=False, slide=True)

    # Get summary (resets counters)
    summary = stats.get_kcc_summary()  # Returns dict with averages and counts
"""

import time


class DebugStatsTracker:
    """
    Singleton tracker for debug statistics.
    Accumulates metrics between summary outputs.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.reset_all()

    def reset_all(self):
        """Reset all statistics."""
        self._start_time = time.perf_counter()

        # Engine stats
        self._engine_jobs_submitted = 0
        self._engine_jobs_completed = 0
        self._engine_total_latency_ms = 0.0
        self._engine_last_summary_time = time.perf_counter()

        # KCC stats
        self._kcc_steps = 0
        self._kcc_total_calc_us = 0.0
        self._kcc_total_rays = 0
        self._kcc_total_tris = 0
        self._kcc_blocked_count = 0
        self._kcc_step_up_count = 0
        self._kcc_slide_count = 0
        self._kcc_ceiling_count = 0
        self._kcc_last_summary_time = time.perf_counter()

        # Camera stats
        self._camera_rays = 0
        self._camera_hits = 0
        self._camera_total_calc_us = 0.0
        self._camera_last_summary_time = time.perf_counter()

    # ════════════════════════════════════════════════════════════════════════
    # ENGINE STATS
    # ════════════════════════════════════════════════════════════════════════

    def record_engine_job_submitted(self):
        """Record a job submission."""
        self._engine_jobs_submitted += 1

    def record_engine_job_completed(self, latency_ms: float = 0.0):
        """Record a job completion with optional latency."""
        self._engine_jobs_completed += 1
        self._engine_total_latency_ms += latency_ms

    def get_engine_summary(self) -> dict:
        """
        Get engine summary and reset counters.
        Returns dict with jobs/sec, avg latency, etc.
        """
        now = time.perf_counter()
        elapsed = max(0.001, now - self._engine_last_summary_time)

        jobs_per_sec = self._engine_jobs_completed / elapsed
        avg_latency = (self._engine_total_latency_ms / max(1, self._engine_jobs_completed))

        summary = {
            "jobs_submitted": self._engine_jobs_submitted,
            "jobs_completed": self._engine_jobs_completed,
            "jobs_per_sec": jobs_per_sec,
            "avg_latency_ms": avg_latency,
            "elapsed_sec": elapsed,
        }

        # Reset counters
        self._engine_jobs_submitted = 0
        self._engine_jobs_completed = 0
        self._engine_total_latency_ms = 0.0
        self._engine_last_summary_time = now

        return summary

    # ════════════════════════════════════════════════════════════════════════
    # KCC PHYSICS STATS
    # ════════════════════════════════════════════════════════════════════════

    def record_kcc_step(self, calc_time_us: float = 0.0, rays: int = 0, tris: int = 0,
                        blocked: bool = False, step_up: bool = False,
                        slide: bool = False, ceiling: bool = False):
        """Record a KCC physics step result."""
        self._kcc_steps += 1
        self._kcc_total_calc_us += calc_time_us
        self._kcc_total_rays += rays
        self._kcc_total_tris += tris

        if blocked:
            self._kcc_blocked_count += 1
        if step_up:
            self._kcc_step_up_count += 1
        if slide:
            self._kcc_slide_count += 1
        if ceiling:
            self._kcc_ceiling_count += 1

    def get_kcc_summary(self) -> dict:
        """
        Get KCC summary and reset counters.
        Returns dict with avg timing, event counts, etc.
        """
        now = time.perf_counter()
        elapsed = max(0.001, now - self._kcc_last_summary_time)
        steps = max(1, self._kcc_steps)

        summary = {
            "steps": self._kcc_steps,
            "steps_per_sec": self._kcc_steps / elapsed,
            "avg_calc_us": self._kcc_total_calc_us / steps,
            "avg_rays": self._kcc_total_rays / steps,
            "avg_tris": self._kcc_total_tris / steps,
            "blocked_count": self._kcc_blocked_count,
            "step_up_count": self._kcc_step_up_count,
            "slide_count": self._kcc_slide_count,
            "ceiling_count": self._kcc_ceiling_count,
            "elapsed_sec": elapsed,
        }

        # Reset counters
        self._kcc_steps = 0
        self._kcc_total_calc_us = 0.0
        self._kcc_total_rays = 0
        self._kcc_total_tris = 0
        self._kcc_blocked_count = 0
        self._kcc_step_up_count = 0
        self._kcc_slide_count = 0
        self._kcc_ceiling_count = 0
        self._kcc_last_summary_time = now

        return summary

    # ════════════════════════════════════════════════════════════════════════
    # CAMERA STATS
    # ════════════════════════════════════════════════════════════════════════

    def record_camera_ray(self, calc_time_us: float = 0.0, hit: bool = False):
        """Record a camera occlusion ray result."""
        self._camera_rays += 1
        self._camera_total_calc_us += calc_time_us
        if hit:
            self._camera_hits += 1

    def get_camera_summary(self) -> dict:
        """Get camera summary and reset counters."""
        now = time.perf_counter()
        elapsed = max(0.001, now - self._camera_last_summary_time)
        rays = max(1, self._camera_rays)

        summary = {
            "rays": self._camera_rays,
            "rays_per_sec": self._camera_rays / elapsed,
            "hits": self._camera_hits,
            "hit_rate": self._camera_hits / rays,
            "avg_calc_us": self._camera_total_calc_us / rays,
            "elapsed_sec": elapsed,
        }

        # Reset counters
        self._camera_rays = 0
        self._camera_hits = 0
        self._camera_total_calc_us = 0.0
        self._camera_last_summary_time = now

        return summary


def get_stats_tracker() -> DebugStatsTracker:
    """Get the global stats tracker singleton."""
    return DebugStatsTracker()


def reset_stats():
    """Reset all statistics (call on game start)."""
    get_stats_tracker().reset_all()
