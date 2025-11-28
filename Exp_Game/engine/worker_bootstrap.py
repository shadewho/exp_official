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
    spec = importlib.util.spec_from_file_location("_worker_module", worker_file_path)
    worker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker_module)
    worker_module.worker_loop(job_queue, result_queue, worker_id, shutdown_event)
