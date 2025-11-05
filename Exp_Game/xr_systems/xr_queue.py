import time
"""Developers manifesto: develop a strong output of the character, actions, 
environment, utilties etc so that we can eliminate guess work from XR development.
Visualize and output statistics and save time debugging and confusion in the 
future. Gate everything by ON/OFF toggles so that we can quickly enable/disable 
and don't bloat the system when not needed. It's critical every new component 
and is backed by data and visualization to help us understand the system and develop it.
Use /Developers module and the framwork within to build these systems out.
Use DevHud to output text and graphs as needed. 
"""
# Module-scoped, single-thread use from Blender TIMER
_current = None            # {"op": op, "seq": int, "sim_time": float}
_jobs = []                 # [{"id","name","payload"}]
_callbacks = {}            # job_id -> callable(result_dict)
_seq_counter = 0

# Pending batches already sent (we'll poll replies later, never block)
# seq -> {"callbacks": {job_id: cb}, "t_send": float}
_pending_batches = {}

def xr_begin_frame(op, sim_time: float) -> int:
    """Start a new XR batch for this Blender TIMER tick (send-only later)."""
    global _current, _jobs, _callbacks, _seq_counter
    _seq_counter = (_seq_counter + 1) % 1_000_000_000
    _current = {"op": op, "seq": _seq_counter, "sim_time": float(sim_time)}
    _jobs.clear()
    _callbacks.clear()
    return _seq_counter

def xr_enqueue(name: str, payload: dict, apply_cb=None) -> str:
    """Queue a job for this frame. apply_cb(result_dict) runs when a reply arrives (maybe next frame)."""
    if _current is None:
        raise RuntimeError("xr_begin_frame must be called before xr_enqueue().")
    job_id = f"{name}:{_current['seq']}:{len(_jobs)}"
    _jobs.append({"id": job_id, "name": name, "payload": payload})
    if callable(apply_cb):
        _callbacks[job_id] = apply_cb
    # account work in stats (request issued)
    stats = getattr(_current["op"], "_xr_stats", None)
    if isinstance(stats, dict):
        stats["req"] = int(stats.get("req", 0)) + 1
    return job_id

def xr_flush_frame(op) -> int:
    """
    SEND-ONLY. Fire the batch to XR and return immediately.
    We never wait for the reply here. Replies are handled by xr_poll().
    """
    global _current, _jobs, _callbacks, _pending_batches
    if _current is None or not _jobs:
        _current = None
        _jobs.clear()
        _callbacks.clear()
        return 0

    xr = getattr(op, "_xr", None)
    if xr is None:
        raise RuntimeError("XR not initialized but XR jobs were queued.")

    # Build batch (echo the frame_seq so runtime can return it)
    batch = {
        "type": "frame_input",
        "frame_seq": int(_current["seq"]),
        "t": float(_current["sim_time"]),
        "jobs": list(_jobs),
    }

    t0 = time.perf_counter()
    # Non-public send is fine here; NEVER read here (no stall)
    try:
        xr._send(batch)  # fire-and-forget
    except Exception:
        pass

    # Keep callbacks & send time so we can apply later in xr_poll()
    _pending_batches[_current["seq"]] = {
        "callbacks": dict(_callbacks),
        "t_send": t0,
    }

    # Clear current batch state
    _jobs.clear()
    _callbacks.clear()
    _current = None
    return 0

def xr_poll(op, max_loops: int = 64) -> int:
    """
    Drain as many XR replies as are available right now (non-blocking),
    applying their callbacks. Hard-cap to avoid starvation.
    Returns number of callbacks run.
    """
    xr = getattr(op, "_xr", None)
    if xr is None:
        return 0

    applied = 0
    loops = 0
    HARD_CAP = max(8, int(max_loops or 64))

    while loops < HARD_CAP:
        loops += 1
        try:
            resp = xr._readline(timeout_s=0.0)  # non-blocking
        except Exception:
            break

        # Nothing more buffered right now → stop
        if not isinstance(resp, dict):
            break

        if resp.get("type") != "frame_cmds":
            continue

        seq  = int(resp.get("frame_seq", -1))
        pend = _pending_batches.pop(seq, None)
        now  = time.perf_counter()

        # Update RTT/phase stats
        stats = getattr(op, "_xr_stats", None)
        if isinstance(stats, dict):
            if pend is not None:
                lat_ms = (now - float(pend.get("t_send", now))) * 1000.0
                stats["last_lat_ms"] = float(lat_ms)
                prev = float(stats.get("lat_ema_ms", 0.0))
                stats["lat_ema_ms"] = lat_ms if prev == 0.0 else (prev * 0.85 + lat_ms * 0.15)
                stats["last_ok_wall"] = now
            try:
                xr_tick = int(resp.get("tick", -1))
                period  = float(resp.get("period", 1.0/30.0))
                bl      = int(getattr(op, "_xr_bl_step", 0))
                if xr_tick >= 0:
                    phase_frames = bl - xr_tick
                    stats["phase_ms"] = float(phase_frames * period * 1000.0)
            except Exception:
                pass

        # Dispatch per-job callbacks with _frame_seq injected
        cbs = pend["callbacks"] if (pend and isinstance(pend.get("callbacks"), dict)) else {}
        for jr in resp.get("jobs", []):
            jid = jr.get("id")
            cb  = cbs.pop(jid, None)
            if callable(cb):
                res = jr.get("result")
                if not isinstance(res, dict):
                    res = {} if res is None else {"value": res}
                res["_frame_seq"] = seq
                try:
                    cb(res)
                except Exception:
                    pass
                applied += 1
                if isinstance(stats, dict):
                    stats["ok"] = int(stats.get("ok", 0)) + 1

    return applied
