# Exp_Game/engine/engine_types.py
"""
Data structures for engine communication.
All types must be picklable (no bpy references!).
"""

from dataclasses import dataclass
from typing import Any, Optional
import time


@dataclass
class EngineJob:
    """
    A job to be processed by the engine.

    Attributes:
        job_id: Unique identifier for this job
        job_type: String identifier for what kind of job this is (e.g., "PATHFIND", "PHYSICS")
        data: Job-specific data (must be picklable - no bpy objects!)
        timestamp: When this job was created
        target_worker: If set, only this worker should process the job (-1 = any worker)
    """
    job_id: int
    job_type: str
    data: Any
    timestamp: float = None
    target_worker: int = -1  # -1 = any worker can process, else specific worker ID

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class EngineResult:
    """
    A result returned from the engine.

    Attributes:
        job_id: Matches the job_id from the EngineJob
        job_type: What kind of job this was
        result: Job-specific result data (must be picklable)
        success: Whether the job completed successfully
        error: Error message if success=False
        timestamp: When this result was created
        processing_time: How long the job took (seconds)
        worker_id: ID of the worker that processed this job (for tracking)
    """
    job_id: int
    job_type: str
    result: Any
    success: bool = True
    error: Optional[str] = None
    timestamp: float = None
    processing_time: float = 0.0
    worker_id: int = -1  # ID of worker that processed this job

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class EngineHeartbeat:
    """
    Periodic heartbeat signal from engine to confirm it's alive.

    Attributes:
        timestamp: When this heartbeat was sent
        worker_count: Number of active workers
        jobs_processed: Total jobs processed since startup
        uptime: Seconds since engine started
    """
    timestamp: float
    worker_count: int
    jobs_processed: int
    uptime: float