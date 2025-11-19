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

    try:
        jobs = stats.setdefault("jobs", {})
        rec  = jobs.setdefault(name, {"req": 0, "ok": 0, "fail": 0})
        rec["req"] += 1
    except Exception:
        pass

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

    # backlog metrics
    try:
        stats = getattr(op, "_xr_stats", None)
        if isinstance(stats, dict):
            pend_now = len(_pending_batches)
            stats["pend_now"] = int(pend_now)
            stats["pend_max"] = int(max(int(stats.get("pend_max", 0) or 0), pend_now))
    except Exception:
        pass
    return 0

def xr_poll(op, max_loops: int = 64) -> int:
    """
    Drain as many XR replies as are available right now (non-blocking),
    applying their callbacks. Hard-cap to avoid starvation.
    Returns number of callbacks run.

    IMPORTANT:
      • Updates stats['pend_now'] AFTER popping replies.
      • Pushes a trace record for each reply so Δframes p95 is real.
    """
    xr = getattr(op, "_xr", None)
    if xr is None:
        # keep backlog sane when XR is off
        try:
            stats = getattr(op, "_xr_stats", None)
            if isinstance(stats, dict):
                pn = len(_pending_batches)
                stats["pend_now"] = int(pn)
                stats["pend_max"] = int(max(int(stats.get("pend_max", 0) or 0), pn))
        except Exception:
            pass
        return 0

    applied = 0
    loops   = 0
    HARD_CAP = max(8, int(max_loops or 64))

    while loops < HARD_CAP:
        loops += 1
        try:
            resp = xr._readline(timeout_s=0.0)  # non-blocking
        except Exception:
            break

        if not isinstance(resp, dict):
            break
        if resp.get("type") != "frame_cmds":
            continue

        seq  = int(resp.get("frame_seq", -1))
        pend = _pending_batches.pop(seq, None)
        now  = time.perf_counter()

        stats = getattr(op, "_xr_stats", None)
        lat_ms = None
        bl = xr_tick = None
        phase_ms_val = None

        if isinstance(stats, dict):
            # RTT / EMA / last_ok
            if pend is not None:
                lat_ms = (now - float(pend.get("t_send", now))) * 1000.0
                stats["last_lat_ms"] = float(lat_ms)
                prev = float(stats.get("lat_ema_ms", 0.0))
                stats["lat_ema_ms"] = lat_ms if prev == 0.0 else (prev * 0.85 + lat_ms * 0.15)
                stats["last_ok_wall"] = now
                if lat_ms <= 12.0:
                    stats["sameframe_ok"] = int(stats.get("sameframe_ok", 0)) + 1
                else:
                    stats["nextframe_ok"]  = int(stats.get("nextframe_ok", 0)) + 1

            # Phase (Blender step vs XR tick)
            try:
                xr_tick = int(resp.get("tick", -1))
                period  = float(resp.get("period", 1.0/30.0))
                bl      = int(getattr(op, "_xr_bl_step", 0))
                if xr_tick >= 0:
                    phase_frames = bl - xr_tick
                    phase_ms_val = float(phase_frames * period * 1000.0)
                    stats["phase_frames"] = int(phase_frames)
                    stats["phase_ms"]     = phase_ms_val
            except Exception:
                pass

            # Runtime diagnostics bubble through
            try:
                diag = resp.get("diag", {})
                if isinstance(diag, dict):
                    if "proc_ms"    in diag: stats["proc_ms"]    = float(diag.get("proc_ms", 0.0) or 0.0)
                    if "jobs_n"     in diag: stats["jobs_n"]     = int(diag.get("jobs_n", 0) or 0)
                    if "jobs_ms"    in diag: stats["jobs_ms"]    = float(diag.get("jobs_ms", 0.0) or 0.0)
                    if "job_max_ms" in diag: stats["job_max_ms"] = float(diag.get("job_max_ms", 0.0) or 0.0)
            except Exception:
                pass

            # Refresh backlog AFTER popping
            try:
                pn = len(_pending_batches)
                stats["pend_now"] = int(pn)
                stats["pend_max"] = int(max(int(stats.get("pend_max", 0) or 0), pn))
            except Exception:
                pass

        # >>> Push trace so HUD/health can compute Δframes p95 truthfully
        try:
            if (bl is not None) and (xr_tick is not None) and (xr_tick >= 0):
                if lat_ms is None:
                    lat_ms = float(stats.get("last_lat_ms", 0.0)) if isinstance(stats, dict) else 0.0
                _push_trace(op, bl, xr_tick, float(lat_ms or 0.0), float(phase_ms_val or 0.0))
        except Exception:
            pass

        # Dispatch per-job callbacks
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
                    try:
                        jname = jr.get("name", None) or jr.get("id", "")
                        if not jname and isinstance(jid, str) and ":" in jid:
                            jname = jid.split(":", 1)[0]
                        if jname:
                            jobs = stats.setdefault("jobs", {})
                            rec  = jobs.setdefault(jname, {"req": 0, "ok": 0, "fail": 0})
                            rec["ok"] += 1
                    except Exception:
                        pass

    # Final backlog snapshot after draining
    try:
        stats = getattr(op, "_xr_stats", None)
        if isinstance(stats, dict):
            pn = len(_pending_batches)
            stats["pend_now"] = int(pn)
            stats["pend_max"] = int(max(int(stats.get("pend_max", 0) or 0), pn))
    except Exception:
        pass

    return applied



def _push_trace(op, bl_step: int, xr_tick: int, lat_ms: float, phase_ms: float):
    """Append one record to op._frame_trace so HUD/health can compute Δframes p95 truthfully."""
    try:
        rec = {
            "wall": time.perf_counter(),
            "step": int(bl_step),
            "tick": int(xr_tick),
            "status": "OK",
            "rtt_ms": float(lat_ms),
            "phase_ms": float(phase_ms),
            "gate_ms": 0.0,
        }
        buf = getattr(op, "_frame_trace", None)
        if not isinstance(buf, list):
            buf = []
            setattr(op, "_frame_trace", buf)
        buf.append(rec)
        if len(buf) > 200:
            del buf[:len(buf) - 200]
    except Exception:
        pass

