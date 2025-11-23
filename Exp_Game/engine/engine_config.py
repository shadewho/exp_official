# Exp_Game/engine/engine_config.py
"""
Configuration for the multiprocessing engine.
Adjust these values based on your needs.
"""

# Number of worker processes to spawn
# Set to 4 workers - safe for most systems running Blender
# (Most Blender-capable systems have 4+ cores)
WORKER_COUNT = 4

# Queue sizes (how many jobs/results can be buffered)
# Lower values = less memory, less latency
# Higher values = more buffering for burst loads
# For stress testing, we need massive queues to handle burst submissions
JOB_QUEUE_SIZE = 10000     # Large queue for stress test bursts
RESULT_QUEUE_SIZE = 10000  # Large queue for result bursts

# Heartbeat interval (seconds)
# How often the engine sends "I'm alive" signals
HEARTBEAT_INTERVAL = 1.0

# Shutdown timeout (seconds)
# How long to wait for workers to finish before forcing termination
SHUTDOWN_TIMEOUT = 2.0

# Debug flag - set to True to see detailed logging
# Controlled by scene.dev_debug_engine in the Developer Tools panel
DEBUG_ENGINE = False
