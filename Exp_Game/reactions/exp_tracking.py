# Exploratory/Exp_Game/reactions/exp_tracking.py
import bpy
import math
from mathutils import Vector, Matrix
from ..props_and_utils.exp_time import get_game_time
from ..audio import exp_globals  # ACTIVE_MODAL_OP
from ..physics.exp_raycastutils import raycast_closest_any

# ─────────────────────────────────────────────────────────
# Lightweight tracking system
# • Character mover → “autopilot” by injecting keys for KCC (reuses your full physics)
# • Non-character mover → minimal sweep + slide + ground snap against proxy meshes
# • Re-evaluates target every SIM step (tracks moving targets)
# ─────────────────────────────────────────────────────────

_ACTIVE_TRACKS = []

def clear():
    """Stop all active tracks and release any injected keys."""
    op = getattr(exp_globals, "ACTIVE_MODAL_OP", None)
    for t in _ACTIVE_TRACKS:
        t._release_keys(op)
    _ACTIVE_TRACKS.clear()

class _TrackTask:
    __slots__ = (
        "r_id","from_obj","to_obj",
        "use_gravity","respect_proxy","speed","arrive_radius",
        "exclusive_control","allow_run","max_runtime","start_time",
        "_is_char_driver","_last_keys",
    )
    def __init__(self, r_id, from_obj, to_obj,
                 use_gravity=True, respect_proxy=True, speed=3.5,
                 arrive_radius=0.3, exclusive_control=True, allow_run=True,
                 max_runtime=0.0):
        self.r_id            = r_id
        self.from_obj        = from_obj
        self.to_obj          = to_obj
        self.use_gravity     = bool(use_gravity)
        self.respect_proxy   = bool(respect_proxy)
        self.speed           = float(speed)
        self.arrive_radius   = max(0.0, float(arrive_radius))
        self.exclusive_control = bool(exclusive_control)
        self.allow_run       = bool(allow_run)
        self.max_runtime     = max(0.0, float(max_runtime))
        self.start_time      = get_game_time()
        self._last_keys      = set()
        op = getattr(exp_globals, "ACTIVE_MODAL_OP", None)
        self._is_char_driver = (op is not None and op.target_object and self.from_obj == op.target_object)

    # --------- injected keys (character autopilot) ----------
    def _release_keys(self, op):
        if not self._is_char_driver or not op:
            return
        # remove our own injected keys only
        for k in list(self._last_keys):
            try:
                op.keys_pressed.discard(k)
            except Exception:
                pass
        self._last_keys.clear()

    def _inject_autopilot(self, op: "ExpModal", wish_world_xy: Vector, speed_hint: float):
        """
        Convert desired world-plane vector → camera-local → discrete keys.
        Only press 0–2 direction keys + optional run. Exclusive mode strips user WASD/Run for the tick.
        """
        if not op or wish_world_xy.length_squared <= 1.0e-10:
            self._release_keys(op)
            return

        # remove previous keys first (prevents stacking)
        self._release_keys(op)

        # Exclusive → clear user WASD/Run for this tick
        if self.exclusive_control:
            for base in (op.pref_forward_key, op.pref_backward_key, op.pref_left_key,
                         op.pref_right_key, op.pref_run_key):
                op.keys_pressed.discard(base)

        # rotate world → camera local
        Rz = Matrix.Rotation(-op.yaw, 4, 'Z')
        local = Rz @ Vector((wish_world_xy.x, wish_world_xy.y, 0.0))
        x, y = float(local.x), float(local.y)

        want = set()
        # pick per axis (threshold avoids jitter at arrival)
        thr = 0.18
        if   y >  thr: want.add(op.pref_forward_key)
        elif y < -thr: want.add(op.pref_backward_key)
        if   x >  thr: want.add(op.pref_right_key)
        elif x < -thr: want.add(op.pref_left_key)

        # decide run from speed hint vs scene config
        run_now = False
        try:
            max_run = float(bpy.context.scene.char_physics.max_speed_run)
            run_now = self.allow_run and (speed_hint >= max_run - 1e-3)
        except Exception:
            run_now = False
        if run_now:
            want.add(op.pref_run_key)

        # inject for this frame
        for k in want:
            op.keys_pressed.add(k)
        self._last_keys = want

    # --------- non-character mover (simple sweep/slide) ----------
    def _move_object_simple(self, mover: bpy.types.Object, goal_world: Vector, dt: float, op):
        if not mover or dt <= 0.0:
            return

        pos = mover.location.copy()
        to_vec = Vector((goal_world.x - pos.x, goal_world.y - pos.y, 0.0))
        dist = to_vec.length
        if dist <= 1.0e-9:
            return

        step_len = min(self.speed * dt, dist)
        fwd = to_vec.normalized()

        static_bvh  = op.bvh_tree if (self.respect_proxy and op) else None
        dynamic_map = op.dynamic_bvh_map if (self.respect_proxy and op) else None

        # Character-like capsule dimensions (borrow from scene)
        try:
            r = float(bpy.context.scene.char_physics.radius)
            h = float(bpy.context.scene.char_physics.height)
        except Exception:
            r, h = 0.22, 1.8

        # 3 vertical rays (feet/mid/head) to detect wall ahead
        ray_len = step_len + r
        best_d = None
        best_n = None
        for z in (r, max(r, min(h - r, 0.5 * h)), h - r):
            o = pos + Vector((0, 0, z))
            hit_loc, hit_n, _obj, d = raycast_closest_any(static_bvh, dynamic_map, o, fwd, ray_len)
            if hit_loc is not None and (best_d is None or d < best_d):
                best_d, best_n = d, hit_n.normalized()

        # advance until contact
        moved = 0.0
        if best_d is None:
            moved_pos = pos + fwd * step_len
        else:
            allow = max(0.0, best_d - r)
            moved = min(step_len, allow)
            moved_pos = pos + fwd * moved

            # simple slide if significant remainder
            remain = max(0.0, step_len - moved)
            if remain > (0.15 * r) and best_n is not None:
                slide = (fwd - best_n * fwd.dot(best_n))
                if slide.length > 1e-9:
                    slide.normalize()
                    # one more quick pass
                    best_d2 = None
                    for z in (r, max(r, min(h - r, 0.5 * h)), h - r):
                        o2 = moved_pos + Vector((0, 0, z))
                        h2, n2, _o, d2 = raycast_closest_any(static_bvh, dynamic_map, o2, slide, remain + r)
                        if h2 is not None and (best_d2 is None or d2 < best_d2):
                            best_d2 = d2
                    allow2 = remain if best_d2 is None else max(0.0, best_d2 - r)
                    moved_pos = moved_pos + slide * allow2

        # gravity / snap to ground (optional)
        if self.use_gravity:
            down = raycast_closest_any(static_bvh, dynamic_map, moved_pos, Vector((0, 0, -1)), max(0.0, 0.6))
            if down[0] is not None:
                moved_pos.z = down[0].z

        mover.location = moved_pos

    # --------- per-step update ----------
    def update(self, dt: float) -> bool:
        """Returns True when finished."""
        now = get_game_time()
        if self.max_runtime > 0.0 and (now - self.start_time) > self.max_runtime:
            return True

        # resolve target every step (moving targets)
        tgt = None
        try:
            if self.to_obj:  # object
                tgt = self.to_obj.matrix_world.translation
        except Exception:
            tgt = None

        mover = self.from_obj if self.from_obj else None
        if not mover or tgt is None:
            return True

        # stop condition
        cur = mover.location
        delta = Vector((tgt.x - cur.x, tgt.y - cur.y, 0.0))
        if delta.length <= self.arrive_radius:
            return True

        op = getattr(exp_globals, "ACTIVE_MODAL_OP", None)

        if self._is_char_driver and op is not None:
            # character autopilot → hand off to KCC via injected keys
            wish_xy = Vector((delta.x, delta.y, 0.0))
            self._inject_autopilot(op, wish_xy, self.speed)
        else:
            # non-character mover → do our own minimal sweep
            self._move_object_simple(mover, tgt, dt, op)

        return False


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def start(r):
    """
    Build and queue a tracking task from a ReactionDefinition instance.
    Expected ReactionDefinition fields (see exp_reaction_definition):
      track_from_use_character / track_from_object
      track_to_use_character   / track_to_object
      track_speed, track_arrive_radius, track_respect_proxy_meshes,
      track_use_gravity, track_exclusive_control, track_allow_run, track_max_runtime
    """
    scn = bpy.context.scene
    from_obj = scn.target_armature if getattr(r, "track_from_use_character", True) else getattr(r, "track_from_object", None)
    to_obj   = scn.target_armature if getattr(r, "track_to_use_character", False) else getattr(r, "track_to_object", None)

    # If we’re driving the character and another active character task exists, cancel it first (single driver policy).
    op = getattr(exp_globals, "ACTIVE_MODAL_OP", None)
    if op and from_obj and op.target_object and from_obj == op.target_object:
        for t in list(_ACTIVE_TRACKS):
            if t._is_char_driver:
                t._release_keys(op)
                _ACTIVE_TRACKS.remove(t)

    task = _TrackTask(
        r_id = r,
        from_obj = from_obj,
        to_obj   = to_obj,
        use_gravity   = getattr(r, "track_use_gravity", True),
        respect_proxy = getattr(r, "track_respect_proxy_meshes", True),
        speed         = getattr(r, "track_speed", 3.5),
        arrive_radius = getattr(r, "track_arrive_radius", 0.3),
        exclusive_control = getattr(r, "track_exclusive_control", True),
        allow_run     = getattr(r, "track_allow_run", True),
        max_runtime   = getattr(r, "track_max_runtime", 0.0),
    )
    _ACTIVE_TRACKS.append(task)


def update_tracking_tasks(dt: float):
    """
    Called on each SIM step (30 Hz). Injects keys for character tasks and moves non-character tasks.
    """
    if not _ACTIVE_TRACKS:
        return
    op = getattr(exp_globals, "ACTIVE_MODAL_OP", None)
    finished = []
    for t in _ACTIVE_TRACKS:
        try:
            done = t.update(dt)
        except Exception:
            done = True
        if done:
            t._release_keys(op)
            finished.append(t)
    # cleanup
    for t in finished:
        try:
            _ACTIVE_TRACKS.remove(t)
        except Exception:
            pass


def execute_tracking_reaction(r):
    """
    Start a 'Track To' task (character autopilot or object mover).
    """
    try:
        start(r)
    except Exception:
        pass
