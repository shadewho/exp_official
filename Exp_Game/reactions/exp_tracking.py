# Exploratory/Exp_Game/reactions/exp_tracking.py
"""
Lightweight tracking system - moves objects toward targets.

Character mover: "autopilot" by injecting keys for KCC (reuses full physics)
Non-character mover: Offloaded to worker (sweep/slide/gravity via TRACKING_BATCH)

Worker offload pattern:
  Main thread: collect tracks, submit job, apply results
  Worker: compute collision/movement (reuses unified_raycast)
"""
import bpy
from mathutils import Vector, Matrix
from ..props_and_utils.exp_time import get_game_time
from ..audio import exp_globals  # ACTIVE_MODAL_OP


# ─────────────────────────────────────────────────────────────────────────────
# Module state
# ─────────────────────────────────────────────────────────────────────────────
_ACTIVE_TRACKS = []
_tracking_obj_lookup: dict[int, bpy.types.Object] = {}  # obj_id -> bpy object
_pending_tracking_job_id = None


def clear():
    """Stop all active tracks and release any injected keys."""
    global _pending_tracking_job_id
    op = getattr(exp_globals, "ACTIVE_MODAL_OP", None)
    for t in _ACTIVE_TRACKS:
        t._release_keys(op)
    _ACTIVE_TRACKS.clear()
    _tracking_obj_lookup.clear()
    _pending_tracking_job_id = None


# ─────────────────────────────────────────────────────────────────────────────
# Track Task (character autopilot stays on main thread)
# ─────────────────────────────────────────────────────────────────────────────
class _TrackTask:
    __slots__ = (
        "r_id", "from_obj", "to_obj",
        "use_gravity", "respect_proxy", "speed", "arrive_radius",
        "exclusive_control", "allow_run", "max_runtime", "start_time",
        "_is_char_driver", "_last_keys", "_arrived",
    )

    def __init__(self, r_id, from_obj, to_obj,
                 use_gravity=True, respect_proxy=True, speed=3.5,
                 arrive_radius=0.3, exclusive_control=True, allow_run=True,
                 max_runtime=0.0):
        self.r_id = r_id
        self.from_obj = from_obj
        self.to_obj = to_obj
        self.use_gravity = bool(use_gravity)
        self.respect_proxy = bool(respect_proxy)
        self.speed = float(speed)
        self.arrive_radius = max(0.0, float(arrive_radius))
        self.exclusive_control = bool(exclusive_control)
        self.allow_run = bool(allow_run)
        self.max_runtime = max(0.0, float(max_runtime))
        self.start_time = get_game_time()
        self._last_keys = set()
        self._arrived = False
        op = getattr(exp_globals, "ACTIVE_MODAL_OP", None)
        self._is_char_driver = (op is not None and op.target_object and self.from_obj == op.target_object)

    # ─────────────────────────────────────────────────────────────────────────
    # Injected keys (character autopilot) - STAYS ON MAIN THREAD
    # ─────────────────────────────────────────────────────────────────────────
    def _release_keys(self, op):
        if not self._is_char_driver or not op:
            return
        for k in list(self._last_keys):
            try:
                op.keys_pressed.discard(k)
            except Exception:
                pass
        self._last_keys.clear()

    def _inject_autopilot(self, op, wish_world_xy: Vector, speed_hint: float):
        """
        Convert desired world-plane vector -> camera-local -> discrete keys.
        Only press 0-2 direction keys + optional run.
        """
        if not op or wish_world_xy.length_squared <= 1.0e-10:
            self._release_keys(op)
            return

        self._release_keys(op)

        if self.exclusive_control:
            for base in (op.pref_forward_key, op.pref_backward_key, op.pref_left_key,
                         op.pref_right_key, op.pref_run_key):
                op.keys_pressed.discard(base)

        Rz = Matrix.Rotation(-op.yaw, 4, 'Z')
        local = Rz @ Vector((wish_world_xy.x, wish_world_xy.y, 0.0))
        x, y = float(local.x), float(local.y)

        want = set()
        thr = 0.18
        if y > thr:
            want.add(op.pref_forward_key)
        elif y < -thr:
            want.add(op.pref_backward_key)
        if x > thr:
            want.add(op.pref_right_key)
        elif x < -thr:
            want.add(op.pref_left_key)

        run_now = False
        try:
            max_run = float(bpy.context.scene.char_physics.max_speed_run)
            run_now = self.allow_run and (speed_hint >= max_run - 1e-3)
        except Exception:
            run_now = False
        if run_now:
            want.add(op.pref_run_key)

        for k in want:
            op.keys_pressed.add(k)
        self._last_keys = want

    # ─────────────────────────────────────────────────────────────────────────
    # Character autopilot update (MAIN THREAD - just injects keys)
    # ─────────────────────────────────────────────────────────────────────────
    def update_character(self, dt: float) -> bool:
        """Update character autopilot. Returns True when finished."""
        now = get_game_time()
        if self.max_runtime > 0.0 and (now - self.start_time) > self.max_runtime:
            return True

        tgt = None
        try:
            if self.to_obj:
                tgt = self.to_obj.matrix_world.translation
        except Exception:
            tgt = None

        mover = self.from_obj
        if not mover or tgt is None:
            return True

        cur = mover.location
        delta = Vector((tgt.x - cur.x, tgt.y - cur.y, 0.0))
        if delta.length <= self.arrive_radius:
            return True

        op = getattr(exp_globals, "ACTIVE_MODAL_OP", None)
        wish_xy = Vector((delta.x, delta.y, 0.0))
        self._inject_autopilot(op, wish_xy, self.speed)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Worker offload: Submit / Apply pattern
# ─────────────────────────────────────────────────────────────────────────────

def submit_tracking_batch(engine, dt: float):
    """
    Submit non-character tracking tasks to worker.
    Character tasks handled separately (just inject keys).

    Returns job_id if submitted, None otherwise.
    """
    global _pending_tracking_job_id

    if not _ACTIVE_TRACKS:
        return None

    # Debug flag
    debug_tracking = getattr(bpy.context.scene, 'dev_debug_tracking', False)

    now = get_game_time()
    batch = []
    _tracking_obj_lookup.clear()

    # Get character physics params for capsule dimensions
    try:
        radius = float(bpy.context.scene.char_physics.radius)
        height = float(bpy.context.scene.char_physics.height)
    except Exception:
        radius, height = 0.22, 1.8

    for task in _ACTIVE_TRACKS:
        # Skip character tasks (handled via key injection)
        if task._is_char_driver:
            continue

        # Skip finished/invalid tasks
        if task.max_runtime > 0.0 and (now - task.start_time) > task.max_runtime:
            continue

        mover = task.from_obj
        if not mover:
            continue

        # Get target position
        tgt = None
        try:
            if task.to_obj:
                tgt = task.to_obj.matrix_world.translation
        except Exception:
            continue

        if tgt is None:
            continue

        obj_id = id(mover)
        _tracking_obj_lookup[obj_id] = mover

        batch.append({
            "obj_id": obj_id,
            "current_pos": tuple(mover.location),
            "goal_pos": (tgt.x, tgt.y, tgt.z),
            "speed": task.speed,
            "dt": dt,
            "radius": radius,
            "height": height,
            "arrive_radius": task.arrive_radius,
            "use_gravity": task.use_gravity,
            "respect_proxy": task.respect_proxy,
        })

    if not batch:
        return None

    _pending_tracking_job_id = engine.submit_job("TRACKING_BATCH", {
        "tracks": batch,
        "debug_logs": [] if debug_tracking else None,  # Only collect logs if debugging
    })

    # Debug logging
    if debug_tracking:
        from ..developer.dev_logger import log_game
        log_game("TRACKING", f"SUBMIT batch={len(batch)} job_id={_pending_tracking_job_id}")
        for b in batch:
            obj = _tracking_obj_lookup.get(b["obj_id"])
            name = obj.name if obj else "?"
            pos = b["current_pos"]
            goal = b["goal_pos"]
            log_game("TRACKING", f"  {name}: pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) -> goal=({goal[0]:.1f},{goal[1]:.1f},{goal[2]:.1f}) speed={b['speed']:.1f}")

    return _pending_tracking_job_id


def apply_tracking_results(result):
    """
    Apply computed positions from worker.
    Returns number of objects updated.
    """
    global _pending_tracking_job_id
    _pending_tracking_job_id = None

    # Debug flag
    debug_tracking = getattr(bpy.context.scene, 'dev_debug_tracking', False)

    if not result.success:
        if debug_tracking:
            from ..developer.dev_logger import log_game
            log_game("TRACKING", f"APPLY FAILED: {result.error}")
        return 0

    results = result.result.get("results", [])
    rays = result.result.get("rays", 0)
    tris = result.result.get("tris", 0)
    applied = 0

    # Mark arrived tasks
    arrived_ids = set()

    for r in results:
        obj_id = r["obj_id"]
        obj = _tracking_obj_lookup.get(obj_id)
        if not obj:
            continue

        # Apply new position (thin write - no computation)
        obj.location = Vector(r["new_pos"])
        applied += 1

        if r.get("arrived", False):
            arrived_ids.add(obj_id)

    # Mark tasks as arrived
    for task in _ACTIVE_TRACKS:
        if not task._is_char_driver and task.from_obj:
            if id(task.from_obj) in arrived_ids:
                task._arrived = True

    # Debug logging
    if debug_tracking:
        from ..developer.dev_logger import log_game
        arrived_count = len(arrived_ids)
        log_game("TRACKING", f"APPLY count={applied} arrived={arrived_count} rays={rays} tris={tris}")
        for r in results:
            obj = _tracking_obj_lookup.get(r["obj_id"])
            name = obj.name if obj else "?"
            pos = r["new_pos"]
            status = "ARRIVED" if r.get("arrived") else "moving"
            log_game("TRACKING", f"  {name}: ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) {status}")

        # Output worker logs
        worker_logs = result.result.get("worker_logs", [])
        for category, msg in worker_logs:
            log_game(category, msg)

    return applied


def get_pending_job_id():
    """Get the current pending job ID (for polling)."""
    return _pending_tracking_job_id


# ─────────────────────────────────────────────────────────────────────────────
# Update loop (called at 30 Hz)
# ─────────────────────────────────────────────────────────────────────────────

def update_tracking_tasks(dt: float):
    """
    Called on each SIM step (30 Hz).
    - Character tasks: inject keys (stays on main thread)
    - Non-character tasks: handled via submit/apply pattern (worker)
    """
    if not _ACTIVE_TRACKS:
        return

    # Debug flag
    debug_tracking = getattr(bpy.context.scene, 'dev_debug_tracking', False)

    op = getattr(exp_globals, "ACTIVE_MODAL_OP", None)
    finished = []
    char_count = 0
    obj_count = 0

    for t in _ACTIVE_TRACKS:
        try:
            if t._is_char_driver:
                # Character autopilot - inject keys (main thread)
                char_count += 1
                done = t.update_character(dt)
                if debug_tracking and t._last_keys:
                    from ..developer.dev_logger import log_game
                    keys_str = ",".join(sorted(t._last_keys))
                    log_game("TRACKING", f"AUTOPILOT keys=[{keys_str}] speed={t.speed:.1f}")
            else:
                # Non-character - check if arrived (position applied by worker)
                obj_count += 1
                done = t._arrived
                # Also check timeout
                now = get_game_time()
                if t.max_runtime > 0.0 and (now - t.start_time) > t.max_runtime:
                    done = True
        except Exception:
            done = True

        if done:
            t._release_keys(op)
            finished.append(t)

    # Log active track summary
    if debug_tracking and (char_count > 0 or obj_count > 0):
        from ..developer.dev_logger import log_game
        log_game("TRACKING", f"ACTIVE char={char_count} obj={obj_count} finished={len(finished)}")

    for t in finished:
        try:
            _ACTIVE_TRACKS.remove(t)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def start(r):
    """
    Build and queue a tracking task from a ReactionDefinition instance.
    """
    scn = bpy.context.scene
    from_obj = scn.target_armature if getattr(r, "track_from_use_character", True) else getattr(r, "track_from_object", None)
    to_obj = scn.target_armature if getattr(r, "track_to_use_character", False) else getattr(r, "track_to_object", None)

    op = getattr(exp_globals, "ACTIVE_MODAL_OP", None)
    if op and from_obj and op.target_object and from_obj == op.target_object:
        for t in list(_ACTIVE_TRACKS):
            if t._is_char_driver:
                t._release_keys(op)
                _ACTIVE_TRACKS.remove(t)

    task = _TrackTask(
        r_id=r,
        from_obj=from_obj,
        to_obj=to_obj,
        use_gravity=getattr(r, "track_use_gravity", True),
        respect_proxy=getattr(r, "track_respect_proxy_meshes", True),
        speed=getattr(r, "track_speed", 3.5),
        arrive_radius=getattr(r, "track_arrive_radius", 0.3),
        exclusive_control=getattr(r, "track_exclusive_control", True),
        allow_run=getattr(r, "track_allow_run", True),
        max_runtime=getattr(r, "track_max_runtime", 0.0),
    )
    _ACTIVE_TRACKS.append(task)


def execute_tracking_reaction(r):
    """Start a 'Track To' task (character autopilot or object mover)."""
    try:
        start(r)
    except Exception:
        pass
