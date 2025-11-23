# Exp_Game/engine/__init__.py
"""
Multiprocessing engine - offloads heavy computations from main thread.
Provides TRUE parallelism by bypassing Python's GIL.
"""

# Public API exports
from .engine_core import EngineCore
from .engine_types import EngineJob, EngineResult, EngineHeartbeat
from .engine_config import (
    WORKER_COUNT,
    JOB_QUEUE_SIZE,
    RESULT_QUEUE_SIZE,
    HEARTBEAT_INTERVAL,
    SHUTDOWN_TIMEOUT,
    DEBUG_ENGINE
)

__all__ = [
    'EngineCore',
    'EngineJob',
    'EngineResult',
    'EngineHeartbeat',
    'WORKER_COUNT',
    'JOB_QUEUE_SIZE',
    'RESULT_QUEUE_SIZE',
    'HEARTBEAT_INTERVAL',
    'SHUTDOWN_TIMEOUT',
    'DEBUG_ENGINE',
]
