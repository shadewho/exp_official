"""
Worker bootstrap - COMPLETELY ISOLATED from addon package.
This file has NO imports from the addon to avoid triggering __init__.py
"""
import importlib.util


def bootstrap_worker(worker_file_path, job_queue, result_queue, worker_id, shutdown_event):
    """
    Bootstrap function that runs in worker process.
    Loads worker module WITHOUT importing addon package to avoid bpy.
    """
    # DIAGNOSTIC: Print what we're loading
    import os
    print(f"[Bootstrap {worker_id}] ========== LOADING WORKER ==========")
    print(f"[Bootstrap {worker_id}] Worker file path: {worker_file_path}")
    print(f"[Bootstrap {worker_id}] File exists: {os.path.exists(worker_file_path)}")
    print(f"[Bootstrap {worker_id}] File size: {os.path.getsize(worker_file_path) if os.path.exists(worker_file_path) else 'N/A'} bytes")
    print(f"[Bootstrap {worker_id}] ====================================")

    spec = importlib.util.spec_from_file_location("_worker_module", worker_file_path)
    worker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker_module)
    worker_module.worker_loop(job_queue, result_queue, worker_id, shutdown_event)
