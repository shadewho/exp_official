# File: Exp_Game/modal/exp_loop.py
import time
from ..props_and_utils.exp_time import update_real_time, tick_sim_time, get_game_time
from ..animations.exp_custom_animations import update_all_custom_managers
from ..animations.exp_animations import nla_is_locked
from ..reactions.exp_reactions import update_property_tasks
from ..reactions.exp_projectiles import update_projectile_tasks, update_hitscan_tasks
from ..reactions.exp_transforms import update_transform_tasks
from ..systems.exp_performance import update_performance_culling
from ..physics.exp_dynamic import update_dynamic_meshes
from ..physics.exp_view import view_queue_third_job, apply_view_from_xr
from ..interactions.exp_interactions import check_interactions
from ..reactions.exp_custom_ui import update_text_reactions
from ..reactions.exp_tracking import update_tracking_tasks
from ..audio.exp_globals import update_sound_tasks
from ..audio.exp_audio import get_global_audio_state_manager
from ..Developers.exp_dev_interface import devhud_frame_begin, devhud_frame_end
from ..xr_systems.xr_queue import xr_begin_frame, xr_flush_frame, xr_poll


class GameLoop:
    """
    Per-frame/TIMER orchestrator for ExpModal.
    - No bpy writes in workers; only here on the main thread.
    - XR is started lazily on first simulated frame and stopped on shutdown.
    """

    def __init__(self, op):
        self.op = op
        self._last_anim_state = None
        self._xr_started = False

    # ---------- Public API ----------

    def on_timer(self, context):
        """
        Runs once per Blender TIMER event.
        Blender is authoritative and never stalls. XR follows Blender:
        we enqueue XR work and fire it off, then poll replies later in the frame
        without blocking. View is split: queue -> flush+poll -> apply.
        """
        op = self.op

        # ---- START FRAME (Developer HUD) ----
        devhud_frame_begin(op)

        # A) Timebases (scaled + wall clock)
        op.update_time()        # scaled per-frame dt (UI / interpolation)
        _ = update_real_time()  # wall-clock (for diagnostics only)

        # B) Decide how many 30 Hz physics steps are due (bounded catch-up)
        steps = op._physics_steps_due()
        op._perf_last_physics_steps = steps

        # Lazy XR start: start when we actually simulate
        if steps and not self._xr_started:
            try:
                from ..xr_systems.xr_client import XRClient
                op._xr = XRClient()
                if not op._xr.start():
                    try: op.report({'ERROR'}, f"XR failed to start: {op._xr.last_error or 'unknown error'}")
                    except Exception: pass
                    op.cancel(context); return
                self._xr_started = True
            except Exception as e:
                try: op.report({'ERROR'}, f"XR bootstrap exception: {e}")
                except Exception: pass
                op.cancel(context); return

        # B0) FRAME-BEGIN XR SYNC: reset per-tick flags BEFORE ANY POLL
        if self._xr_started:
            try:
                xr_begin_frame(op, get_game_time())
                op._view_enq_guard = 0              # allow exactly one enqueue per tick
                op._view_logged_this_tick = False   # allow one 'allowed' series emit per tick
                # Drain any leftover replies from the previous tick now that flags are reset
                xr_poll(op, max_loops=64)
            except Exception:
                pass

        if steps:
            # Shared SIM dt per 30 Hz step; aggregated dt for bulk systems
            dt_sim = op.physics_dt * float(op.time_scale)
            agg_dt = dt_sim * steps

            # Advance SIM time by the actual amount we’ll simulate this frame
            tick_sim_time(agg_dt)

            # B1) Custom/scripted tasks (SIM-timed): cheap → step-per-step
            for _ in range(steps):
                if not nla_is_locked():
                    update_all_custom_managers(dt_sim)
                update_transform_tasks()
                update_property_tasks()
                update_projectile_tasks(dt_sim)
                update_hitscan_tasks()
                update_tracking_tasks(dt_sim)

            # B3) Distance-based culling (may enqueue XR work): once
            update_performance_culling(op, context)

            # B4) Animations & audio: once with aggregated dt
            if not nla_is_locked() and op.animation_manager:
                op.animation_manager.update(
                    op.keys_pressed, agg_dt, op.is_grounded, op.z_velocity
                )
                cur_state = op.animation_manager.anim_state
                if cur_state is not None and cur_state != self._last_anim_state:
                    audio_state_mgr = get_global_audio_state_manager()
                    audio_state_mgr.update_audio_state(cur_state)
                    self._last_anim_state = cur_state

        # B2) Dynamic proxies: **always once per TIMER tick** (static & dynamic in tandem)
        update_dynamic_meshes(op)

        # C) Input gating (unchanged)
        mg = context.scene.mobility_game
        if not mg.allow_movement:
            for k in (op.pref_forward_key, op.pref_backward_key,
                    op.pref_left_key, op.pref_right_key):
                if k in op.keys_pressed:
                    op.keys_pressed.remove(k)
        if not mg.allow_sprint and op.pref_run_key in op.keys_pressed:
            op.keys_pressed.remove(op.pref_run_key)

        # D) Physics integration (execute exactly 'steps' iterations at 30 Hz)
        op.update_movement_and_gravity(context, steps)

        # E) VIEW — Phase 1: compute candidate & QUEUE XR job (no apply yet)
        if self._xr_started:
            # Only queue when XR is alive; authority stays on XR.
            view_queue_third_job(context, op)

        # F) Interactions/UI/SFX (only if sim actually advanced)
        if steps:
            check_interactions(context)
            update_text_reactions()
            update_sound_tasks()

        # >>> Flush XR then drain replies (non-blocking; drain pipe)
        if self._xr_started:
            try: xr_flush_frame(op)
            except Exception: pass
            try:
                op._xr_bl_step = int(getattr(op, "_xr_bl_step", 0)) + int(steps)
                xr_poll(op, max_loops=64)  # drain buffered replies this frame
            except Exception:
                pass

        # E2) VIEW — Phase 2: APPLY using the latest XR answer (same-frame when available)
        apply_view_from_xr(context, op)

        # ---- END FRAME ----
        devhud_frame_end(op, context)



    def handle_key_input(self, event):
        self.op.handle_key_input(event)

    def _apply_xr_frame_cmds(self, cmds: dict):
        """Very small bridge: apply minimal XR text sets to your HUD, if exposed."""
        sets = cmds.get("set", [])
        if not sets:
            # Still mark XR as responsive
            try:
                s = getattr(self.op, "_xr_stats", None)
                if isinstance(s, dict):
                    s["last_ok_wall"] = time.perf_counter()
            except Exception:
                pass
            return
        # Use your existing custom UI layer if it exposes set_text(id, value)
        try:
            from ..reactions import exp_custom_ui as hud
            setter = getattr(hud, "set_text", None)
            if callable(setter):
                for srec in sets:
                    if srec.get("kind") == "text":
                        setter(str(srec.get("id", "xr")), str(srec.get("value", "")))
            # Mark XR as responsive
            s = getattr(self.op, "_xr_stats", None)
            if isinstance(s, dict):
                s["last_ok_wall"] = time.perf_counter()
        except Exception:
            pass  # purely optional
    def _bootstrap_xr_sync(self, op):
        """
        Ensure metrics buckets exist and initialize phase tracking.
        """
        if not isinstance(getattr(op, "_xr_stats", None), dict):
            op._xr_stats = {
                "req": 0, "ok": 0, "fail": 0,
                "last_lat_ms": 0.0, "lat_ema_ms": 0.0,
                "last_ok_wall": 0.0,
                "scan_total": 0, "writes_total": 0,
                "last_batch": 0, "last_total": 0,
                # New: frame sync diagnostics
                "phase_frames": 0, "phase_ms": 0.0,
                "misses": 0, "dupes": 0, "gate_wait_ms_ema": 0.0,
            }
        if not isinstance(getattr(op, "_xr_phase", None), dict):
            op._xr_phase = {
                "step": 0,                # Blender's physics step counter
                "last_tick": -1,          # last XR tick observed
                "strict": False,          # set True to enable micro-gating
                "timeout_s": 0.004,       # XR reply wait for frame_input
                "gate_budget_s": 0.003,   # max micro-wait to align with tick
            }

    def _xr_gate_and_mark(self, op, dt_sim: float):
        """
        Send one frame_input per physics step, measure phase/latency.
        Optionally gate for a few sub-ms to align the XR tick to the expected step.
        """
        stats = op._xr_stats
        phase = op._xr_phase

        # Expected Blender step id
        step_id = int(phase.get("step", 0)) + 1
        phase["step"] = step_id

        # Send and wait briefly for reply
        t0 = time.perf_counter()
        req = {"type": "frame_input", "frame_id": step_id, "t": float(get_game_time()), "dt": float(dt_sim)}
        stats["req"] += 1
        resp = op._xr.request_frame(req, timeout_s=float(phase.get("timeout_s", 0.004)))
        lat_ms = (time.perf_counter() - t0) * 1000.0
        stats["last_lat_ms"] = float(lat_ms)
        stats["lat_ema_ms"] = lat_ms if stats["lat_ema_ms"] == 0.0 else (stats["lat_ema_ms"] * 0.85 + lat_ms * 0.15)

        ok = isinstance(resp, dict) and resp.get("type") == "frame_cmds"
        if not ok:
            stats["fail"] += 1
            self._xr_trace_push(op, step_id, None, "MISS", lat_ms, 0.0, 0.0)
            return

        stats["ok"] += 1
        xr_tick = int(resp.get("tick", -1))
        period  = float(resp.get("period", 1.0/30.0))
        phase_frames = step_id - xr_tick if xr_tick >= 0 else 0
        phase_ms = phase_frames * (period * 1000.0)
        stats["phase_frames"] = int(phase_frames)
        stats["phase_ms"] = float(phase_ms)
        stats["last_ok_wall"] = time.perf_counter()

        # Duplicate detection
        if xr_tick == phase.get("last_tick", -1):
            stats["dupes"] = int(stats.get("dupes", 0)) + 1

        # Optional micro-gating to align to expected tick (strict mode)
        gate_wait = 0.0
        if bool(phase.get("strict", False)) and xr_tick < step_id:
            budget = float(phase.get("gate_budget_s", 0.003))
            start_wait = time.perf_counter()
            # Poll with tiny sleeps until XR tick catches up or budget is exhausted
            while (time.perf_counter() - start_wait) < budget and xr_tick < step_id:
                ok2, tk = op._xr.ping()
                if ok2 and tk is not None:
                    xr_tick = int(tk)
                else:
                    break
                if xr_tick >= step_id:
                    break
                time.sleep(0.0005)
            gate_wait = (time.perf_counter() - start_wait) * 1000.0  # ms
            stats["gate_wait_ms_ema"] = gate_wait if stats["gate_wait_ms_ema"] == 0.0 else (stats["gate_wait_ms_ema"] * 0.85 + gate_wait * 0.15)

            # Recompute phase after potential wait
            phase_frames = step_id - xr_tick
            phase_ms = phase_frames * (period * 1000.0)
            stats["phase_frames"] = int(phase_frames)
            stats["phase_ms"] = float(phase_ms)
            if xr_tick < step_id:
                stats["misses"] = int(stats.get("misses", 0)) + 1

        phase["last_tick"] = xr_tick
        self._apply_xr_frame_cmds(resp)
        self._xr_trace_push(op, step_id, xr_tick, "OK", lat_ms, phase_ms, gate_wait)

    def _xr_trace_push(self, op, step_id, xr_tick, status, rtt_ms, phase_ms, gate_ms):
        """
        Lightweight ring buffer for debugging last ~200 steps.
        """
        rec = {
            "wall": time.perf_counter(),
            "step": int(step_id),
            "tick": (None if xr_tick is None else int(xr_tick)),
            "status": str(status),
            "rtt_ms": float(rtt_ms),
            "phase_ms": float(phase_ms),
            "gate_ms": float(gate_ms),
        }
        if not hasattr(op, "_frame_trace") or not isinstance(op._frame_trace, list):
            op._frame_trace = []
        op._frame_trace.append(rec)
        if len(op._frame_trace) > 200:
            op._frame_trace.pop(0)

    def shutdown(self):
        """
        Ensure XR is shut down (operator.cancel() also calls this).
        """
        xr = getattr(self.op, "_xr", None)
        if xr:
            try:
                xr.stop()
            except Exception:
                pass
            self.op._xr = None