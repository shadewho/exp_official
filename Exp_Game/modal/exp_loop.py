#Exp_Game/modal/exp_loop.py
import time
import math
from mathutils import Vector
import bpy

from ..systems.exp_live_performance import (
    perf_frame_begin, perf_mark, perf_frame_end
)
from ..props_and_utils.exp_time import update_real_time, tick_sim_time
from ..animations.exp_custom_animations import update_all_custom_managers
from ..reactions.exp_reactions import update_transform_tasks, update_property_tasks
from ..systems.exp_performance import update_performance_culling, apply_cull_thread_result
from ..physics.exp_dynamic import update_dynamic_meshes
from ..physics.exp_view import compute_camera_allowed_distance
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
        # Keep last camera allowed distance as our local cache,
        # but also mirror it to op for backwards compat.
        self._cam_allowed_last = getattr(op, "_cam_allowed_last", 3.0)

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
                update_all_custom_managers(dt_sim)
                update_transform_tasks()
                update_property_tasks()
            perf_mark(op, 'custom_tasks', t0)

            # B2) Dynamic proxies + platform v/ω (heavy): once per frame
            t0 = perf_mark(op, 'dynamic_meshes')
            update_dynamic_meshes(op)
            perf_mark(op, 'dynamic_meshes', t0)

            # B3) Distance-based culling (has its own throttling): once
            t0 = perf_mark(op, 'culling')
            update_performance_culling(op, context)
            perf_mark(op, 'culling', t0)

            # B4) Animations & audio: once with aggregated dt
            t0 = perf_mark(op, 'anim_audio')
            if op.animation_manager:
                op.animation_manager.update(
                    op.keys_pressed, agg_dt, op.is_grounded, op.z_velocity
                )
                audio_state_mgr = get_global_audio_state_manager()
                audio_state_mgr.update_audio_state(op.animation_manager.anim_state)
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

        # E) Camera update (unchanged threading/apply path)
        t0 = perf_mark(op, 'camera')
        self._update_camera(context)
        self._poll_and_apply_thread_results()
        self._apply_camera_to_view(context)
        perf_mark(op, 'camera', t0)

        # F) Interactions/UI/SFX (SIM timers already advanced)
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

    def _update_camera(self, context):
        """
        Build camera request and (debounced) submit to worker (“camera” job).
        Uses a cached VIEW_3D near-clip value (set by the operator on invoke).
        """
        op = self.op
        if not op.target_object:
            return

        # Ensure local debounce state exists (kept on the loop instance)
        if not hasattr(self, "_cam_last_params"):
            self._cam_last_params = None
            self._cam_last_time = 0.0

        # 1) Direction from pitch/yaw
        dir_x = math.cos(op.pitch) * math.sin(op.yaw)
        dir_y = -math.cos(op.pitch) * math.cos(op.yaw)
        dir_z = math.sin(op.pitch)
        direction = Vector((dir_x, dir_y, dir_z))
        if direction.length > 1e-9:
            direction.normalize()

        # 2) Anchor at capsule top
        cp = context.scene.char_physics
        cap_h = float(getattr(cp, "height", 2.0))
        cap_r = float(getattr(cp, "radius", 0.30))
        anchor = op.target_object.location + Vector((0.0, 0.0, cap_h))

        # 3) Camera “thickness” from cached near-clip (fallback to 0.1)
        clip_start = getattr(op, "_clip_start_cached", 0.1)
        try:
            clip_start = float(clip_start)
            if clip_start <= 0.0:
                clip_start = 0.1
        except Exception:
            clip_start = 0.1
        r_cam = max(0.008, clip_start * 0.60)

        # 4) Desired boom and minimum
        desired_max = max(0.0, context.scene.orbit_distance + context.scene.zoom_factor)
        min_cam     = max(0.0006, cap_r * 0.04)

        # --- Cheap debounce: only solve when pose changed or on a short heartbeat ---
        now = time.perf_counter()
        params = (float(op.pitch), float(op.yaw),
                float(anchor.x), float(anchor.y), float(anchor.z),
                float(desired_max), float(r_cam))

        if self._cam_last_params is not None:
            PITCH_EPS  = 0.0015   # ~0.086°
            YAW_EPS    = 0.0015
            ANCHOR_EPS = 0.008    # 8 mm
            DIST_EPS   = 0.010    # 1 cm
            HEARTBEAT  = 0.08     # ~12.5 Hz fallback

            lp, ly, lx, ly2, lz, ldes, lrcam = self._cam_last_params
            same_pose = (
                abs(params[0] - lp) < PITCH_EPS and
                abs(params[1] - ly) < YAW_EPS   and
                math.sqrt((params[2]-lx)**2 + (params[3]-ly2)**2 + (params[4]-lz)**2) < ANCHOR_EPS and
                abs(params[5] - ldes) < DIST_EPS and
                abs(params[6] - lrcam) < 1e-6
            )
            if same_pose and (now - self._cam_last_time) < HEARTBEAT:
                # Skip submitting a new camera job this tick; still stash for viewport write.
                self._cam_anchor    = anchor
                self._cam_direction = direction
                return

        # Record pose/time for next tick
        self._cam_last_params = params
        self._cam_last_time   = now

        # 5) Submit background camera solve (coalesced by key/version)
        eng = getattr(op, "_thread_eng", None)
        if eng:
            if not hasattr(op, "_frame_seq"):
                op._frame_seq = 0
            op._frame_seq += 1
            eng.submit_latest(
                key="camera",
                version=op._frame_seq,
                fn=compute_camera_allowed_distance,
                char_key=op.target_object.name,
                anchor=anchor.copy(),
                direction=direction.copy(),
                r_cam=float(r_cam),
                desired_max=float(desired_max),
                min_cam=float(min_cam),
                static_bvh=getattr(op, "bvh_tree", None),
                dynamic_bvh_map=getattr(op, "dynamic_bvh_map", None),
            )

        # Stash for viewport application
        self._cam_anchor    = anchor
        self._cam_direction = direction



    def _poll_and_apply_thread_results(self):
        """
        Collect finished thread jobs. Apply:
          • camera distance
          • per-object culling batch writes
        """
        op = self.op
        eng = getattr(op, "_thread_eng", None)
        if not eng:
            return

        for res in eng.poll_results():
            if res.key == "camera" and isinstance(res.payload, (int, float)):
                self._cam_allowed_last = float(res.payload)
                op._cam_allowed_last = self._cam_allowed_last  # mirror for compatibility
                continue

            if res.key.startswith("cull:") and isinstance(res.payload, dict):
                # Delegate to exp_performance utility (operates on op._perf_runtime)
                try:
                    apply_cull_thread_result(op, res.payload)
                except Exception:
                    # Non-fatal; continue consuming other results
                    pass

    def _apply_camera_to_view(self, context):
        """
        Writes to exactly ONE cached VIEW_3D (rv3d) on the main thread.
        Skips redundant writes unless changes exceed tiny thresholds.
        """
        op = self.op
        if not hasattr(self, "_cam_anchor") or not hasattr(self, "_cam_direction"):
            return

        anchor    = self._cam_anchor
        direction = self._cam_direction
        desired_dist = getattr(op, "_cam_allowed_last", 3.0)

        # Rebind once if rv3d got invalid
        rv3d = getattr(op, "_view3d_rv3d", None)
        if rv3d is None:
            rebind_ok = False
            try:
                if hasattr(op, "_maybe_rebind_view3d"):
                    rebind_ok = bool(op._maybe_rebind_view3d(context))
            except Exception:
                rebind_ok = False
            rv3d = getattr(op, "_view3d_rv3d", None) if rebind_ok else None
            if rv3d is None:
                return

        # --- change thresholds (keep very small to preserve feel) ---
        POS_EPS   = 1e-4     # meters
        ANG_EPS   = 1e-4     # radians on quaternion delta
        DIST_EPS  = 1e-4     # meters

        # Build target rotation once
        target_rot = direction.to_track_quat('Z', 'Y')

        # Pull last applied values from self (cache) to avoid getattr on rv3d
        last_loc = getattr(self, "_rv3d_last_loc", None)
        last_rot = getattr(self, "_rv3d_last_rot", None)
        last_dst = getattr(self, "_rv3d_last_dst", None)

        # Decide if each channel needs a write
        need_loc = True
        if last_loc is not None:
            need_loc = (anchor - last_loc).length > POS_EPS

        need_rot = True
        if last_rot is not None:
            dq = last_rot.rotation_difference(target_rot)
            need_rot = abs(dq.angle) > ANG_EPS

        need_dst = True
        if last_dst is not None:
            need_dst = abs(desired_dist - last_dst) > DIST_EPS

        # Apply only what changed
        try:
            if need_loc:
                rv3d.view_location = anchor
                self._rv3d_last_loc = anchor.copy()
            if need_rot:
                rv3d.view_rotation = target_rot
                self._rv3d_last_rot = target_rot.copy()
            if need_dst:
                rv3d.view_distance = desired_dist
                self._rv3d_last_dst = float(desired_dist)
        except Exception:
            # If write failed, try a one-shot rebind and a single retry
            if hasattr(op, "_maybe_rebind_view3d") and op._maybe_rebind_view3d(context):
                try:
                    rv3d = op._view3d_rv3d
                    if need_loc:
                        rv3d.view_location = anchor
                        self._rv3d_last_loc = anchor.copy()
                    if need_rot:
                        rv3d.view_rotation = target_rot
                        self._rv3d_last_rot = target_rot.copy()
                    if need_dst:
                        rv3d.view_distance = desired_dist
                        self._rv3d_last_dst = float(desired_dist)
                except Exception:
                    pass

