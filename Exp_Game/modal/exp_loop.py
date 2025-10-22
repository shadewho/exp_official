#Exp_Game/modal/exp_loop.py

from ..systems.exp_live_performance import (
    perf_frame_begin, perf_mark, perf_frame_end
)
from ..props_and_utils.exp_time import update_real_time, tick_sim_time
from ..animations.exp_custom_animations import update_all_custom_managers
from ..animations.exp_animations import nla_is_locked
from ..reactions.exp_reactions import update_property_tasks
from ..reactions.exp_projectiles import update_projectile_tasks
from ..reactions.exp_transforms import update_transform_tasks
from ..systems.exp_performance import update_performance_culling, apply_cull_thread_result
from ..physics.exp_dynamic import update_dynamic_meshes
from ..physics.exp_view import update_camera_for_operator
from ..interactions.exp_interactions import check_interactions
from ..reactions.exp_custom_ui import update_text_reactions
from ..audio.exp_globals import update_sound_tasks
from ..audio.exp_audio import get_global_audio_state_manager


class GameLoop:
    """
    Per-frame/TIMER orchestrator for ExpModal.
    - No bpy writes in workers; only here on the main thread.
    - Reuses operator's state/caches to avoid churn.
    """

    def __init__(self, op):
        self.op = op
        self._last_anim_state = None
    # ---------- Public API ----------

    def on_timer(self, context):
        """
        Runs once per Blender TIMER event.
        Uses bounded catch-up for physics to prevent time dilation under stalls.
        """
        op = self.op

        # ---- START FRAME (for live perf overlay) ----
        perf_frame_begin(op)

        # A) Timebases (scaled + wall clock)
        t0 = perf_mark(op, 'time')
        op.update_time()        # scaled per-frame dt (UI / interpolation)
        _ = update_real_time()  # wall-clock (kept for diagnostics/overlays)
        perf_mark(op, 'time', t0)

        # B) Decide how many 30 Hz physics steps are due (bounded catch-up)
        steps = op._physics_steps_due()
        op._perf_last_physics_steps = steps

        if steps:
            # Shared SIM dt per 30 Hz step; aggregated dt for bulk systems
            dt_sim = op.physics_dt * float(op.time_scale)
            agg_dt = dt_sim * steps

            # Advance SIM time by the actual amount we’ll simulate this frame
            tick_sim_time(agg_dt)

            # B1) Custom/scripted tasks (SIM-timed): cheap → step-per-step
            t0 = perf_mark(op, 'custom_tasks')
            for _ in range(steps):
                # >>> guard: skip NLA custom managers while reset/start wipes <<<
                if not nla_is_locked():
                    update_all_custom_managers(dt_sim)
                update_transform_tasks()
                update_property_tasks()
                update_projectile_tasks(dt_sim)
            perf_mark(op, 'custom_tasks', t0)

            # B2) Dynamic proxies + platform v/ω (heavy): once per frame
            t0 = perf_mark(op, 'dynamic_meshes')
            update_dynamic_meshes(op)
            perf_mark(op, 'dynamic_meshes', t0)

            # B3) Distance-based culling (has its own throttling): once
            t0 = perf_mark(op, 'culling')
            update_performance_culling(op, context)
            perf_mark(op, 'culling', t0)

            # poll threaded results once per frame (apply cull batches)
            t0 = perf_mark(op, 'threads_poll')
            self._poll_and_apply_thread_results()
            perf_mark(op, 'threads_poll', t0)

            # B4) Animations & audio: once with aggregated dt
            t0 = perf_mark(op, 'anim_audio')
            # >>> guard: skip character animation update while reset/start wipes <<<
            if not nla_is_locked() and op.animation_manager:
                op.animation_manager.update(
                    op.keys_pressed, agg_dt, op.is_grounded, op.z_velocity
                )
                cur_state = op.animation_manager.anim_state
                if cur_state is not None and cur_state != self._last_anim_state:
                    audio_state_mgr = get_global_audio_state_manager()
                    audio_state_mgr.update_audio_state(cur_state)
                    self._last_anim_state = cur_state
            perf_mark(op, 'anim_audio', t0)

        # C) Input gating based on game mobility flags (unchanged)
        mg = context.scene.mobility_game
        if not mg.allow_movement:
            for k in (op.pref_forward_key, op.pref_backward_key,
                    op.pref_left_key, op.pref_right_key):
                if k in op.keys_pressed:
                    op.keys_pressed.remove(k)
        if not mg.allow_sprint and op.pref_run_key in op.keys_pressed:
            op.keys_pressed.remove(op.pref_run_key)

        # D) Physics integration (execute exactly 'steps' iterations at 30 Hz)
        t0 = perf_mark(op, 'physics')
        op.update_movement_and_gravity(context, steps)
        perf_mark(op, 'physics', t0)

        # E) Camera update
        t0 = perf_mark(op, 'camera')
        update_camera_for_operator(context, op)
        perf_mark(op, 'camera', t0)

        # F) Interactions/UI/SFX (only if sim actually advanced)
        if steps:
            t0 = perf_mark(op, 'interact_ui_audio')
            check_interactions(context)
            update_text_reactions()
            update_sound_tasks()
            perf_mark(op, 'interact_ui_audio', t0)

        # ---- END FRAME ----
        perf_frame_end(op, context)



    def handle_key_input(self, event):
        """
        Delegate to operator’s existing key handler (or migrate into here later).
        This keeps behavior identical and lets you move it later if you want.
        """
        self.op.handle_key_input(event)

    def shutdown(self):
        """
        Ensure the worker pool is closed (operator.cancel() also calls this).
        """
        eng = getattr(self.op, "_thread_eng", None)
        if eng:
            try:
                eng.shutdown()
            except Exception:
                pass

    # ---------- Internals ----------

    def _poll_and_apply_thread_results(self):
        """
        Collect worker results for systems OTHER than camera.
        """
        op = self.op
        eng = getattr(op, "_thread_eng", None)
        if not eng:
            return

        for res in eng.poll_results():

            # Apply threaded culling batches, etc.
            if res.key.startswith("cull:") and isinstance(res.payload, dict):
                try:
                    apply_cull_thread_result(op, res.payload)
                except Exception:
                    pass
