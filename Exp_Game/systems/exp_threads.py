# Exp_Game/systems/exp_threads.py
import threading, queue, concurrent.futures, time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, List
from math import sqrt

# Toggle all debug prints here
DEBUG_THREADS = False

@dataclass
class JobResult:
    key: str
    version: int
    payload: Any


class ThreadEngine:
    """
    A small, safe job system:
      • submit_latest(key, version, fn, *args, **kw) enqueues the latest job per key
      • poll_results() returns ready JobResult objects for the main thread to apply
      • shutdown() cancels future work

    Notes:
      - Only print statements were added for visibility; logic remains the same.
      - Do not call bpy/Depsgraph from worker threads; pass snapshots only.
    """

    def __init__(self, max_workers: int = 1):
        if DEBUG_THREADS:
            print(f"[ThreadEngine] init with max_workers={max_workers}")
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="EXP"
        )
        self._lock = threading.Lock()
        self._inflight: Dict[str, Tuple[int, concurrent.futures.Future]] = {}
        self._latest_args: Dict[str, Tuple[int, Callable, tuple, dict]] = {}
        self._results = queue.SimpleQueue()
        self._cancel_event = threading.Event()

    def cancel_token(self) -> threading.Event:
        return self._cancel_event

    def submit_latest(self, key: str, version: int, fn: Callable, *args, **kw):
        """
        Enqueue or coalesce work under a logical key.
        If a job for 'key' is running, remember only the newest args (debounce).
        """
        with self._lock:
            running = self._inflight.get(key)
            if running is None:
                if DEBUG_THREADS:
                    print(f"[ThreadEngine] submit key={key} v{version}")
                fut = self._executor.submit(self._run_and_capture, key, version, fn, args, kw)
                self._inflight[key] = (version, fut)
            else:
                # Debounce: remember the newest args; older queued work is discarded
                if DEBUG_THREADS:
                    print(f"[ThreadEngine] coalesce key={key} (queue newer v{version})")
                self._latest_args[key] = (version, fn, args, kw)

    def _run_and_capture(self, key: str, version: int, fn: Callable, args: tuple, kw: dict):
        if self._cancel_event.is_set():
            if DEBUG_THREADS:
                print(f"[ThreadEngine] cancel seen before run key={key} v{version}")
            return

        thr = threading.current_thread().name
        t0 = time.perf_counter()
        if DEBUG_THREADS:
            print(f"[ThreadEngine] START key={key} v{version} on {thr}")

        payload = fn(*args, **kw)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        if DEBUG_THREADS:
            print(f"[ThreadEngine] END   key={key} v{version} on {thr} took {dt_ms:.2f}ms")

        # Push result (even if None) for the main thread to decide
        self._results.put(JobResult(key=key, version=version, payload=payload))

        # If a newer job was requested while we ran, schedule it immediately
        with self._lock:
            nxt = self._latest_args.pop(key, None)
            if nxt:
                v2, f2, a2, k2 = nxt
                if DEBUG_THREADS:
                    print(f"[ThreadEngine] chain next key={key} v{v2}")
                fut2 = self._executor.submit(self._run_and_capture, key, v2, f2, a2, k2)
                self._inflight[key] = (v2, fut2)
            else:
                self._inflight.pop(key, None)

    def poll_results(self, max_per_poll: int = 16):
        """Non-blocking: yield up to N ready results."""
        out = []
        for _ in range(max_per_poll):
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                break
        if DEBUG_THREADS and out:
            print(f"[ThreadEngine] polled {len(out)} result(s)")
        return out

    def shutdown(self):
        if DEBUG_THREADS:
            print("[ThreadEngine] shutdown requested")
        self._cancel_event.set()
        # Best-effort cancel queued work
        with self._lock:
            for key, (_, fut) in list(self._inflight.items()):
                try:
                    fut.cancel()
                    if DEBUG_THREADS:
                        print(f"[ThreadEngine] cancel inflight key={key}")
                except Exception:
                    pass
            self._inflight.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if DEBUG_THREADS:
            print("[ThreadEngine] executor shut down")


def compute_cull_batch(
    entry_ptr: int,
    obj_names: List[str],
    obj_positions: List[Tuple[float, float, float]],
    ref_loc: Tuple[float, float, float],
    thresh: float,
    start: int,
    max_count: int,
) -> Dict[str, object]:
    """
    Thread-safe distance checks for per-object culling.

    Returns:
      {
        "entry_ptr": entry_ptr,
        "next_idx": int,                   # round-robin cursor for caller
        "changes": List[Tuple[str,bool]],  # (obj_name, desired_hidden)
      }
    }
    """
    rx, ry, rz = ref_loc
    t2 = float(thresh) * float(thresh)

    n = len(obj_names)
    if n == 0:
        return {"entry_ptr": entry_ptr, "next_idx": start, "changes": []}

    i = 0
    out = []
    idx = start % n

    while i < n and len(out) < max_count:
        name = obj_names[idx]
        px, py, pz = obj_positions[idx]
        dx = px - rx; dy = py - ry; dz = pz - rz
        # distance^2 compare avoids sqrt; identical decision as true distance
        far = (dx*dx + dy*dy + dz*dz) > t2
        out.append((name, far))
        i += 1
        idx = (idx + 1) % n

    return {"entry_ptr": entry_ptr, "next_idx": idx, "changes": out}