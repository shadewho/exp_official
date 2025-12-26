# Exploratory/Exp_Game/reactions/exp_tracking.py
"""
Lightweight tracking system - moves objects toward targets.

Character mover: "autopilot" by injecting keys for KCC (reuses full physics)
Non-character mover: Offloaded to worker (sweep/slide/gravity via TRACKING_BATCH)

Worker offload pattern:
  Main thread: collect tracks, submit job, apply results
  Worker: compute collision/movement (reuses unified_raycast)

OPTIMIZATIONS:
  - Minimal main thread work (just collect data, apply results)
  - Object lookup dict persisted between frames (only add/remove as needed)
  - Debug flag cached once per frame
  - Batch removal of finished tasks (no O(n) removes in loop)
  - Direct tuple assignment for positions (no Vector creation)
"""
import bpy
from mathutils import Vector, Matrix
from ..props_and_utils.exp_time import get_game_time
from ..audio import exp_globals  # ACTIVE_MODAL_OP


# ─────────────────────────────────────────────────────────────────────────────
# Module state
# ─────────────────────────────────────────────────────────────────────────────
_ACTIVE_TRACKS = []
_tracking_obj_lookup: dict[int, bpy.types.Object] = {}  # obj_id -> bpy object (persisted)
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
        keys = self._last_keys
        if keys:
            pressed = op.keys_pressed
            for k in keys:
                pressed.discard(k)
            keys.clear()

    def _inject_autopilot(self, op, wish_world_xy: Vector, speed_hint: float):
        """
        Convert desired world-plane vector -> camera-local -> discrete keys.
        Only press 0-2 direction keys + optional run.
        """
        if not op or wish_world_xy.length_squared <= 1.0e-10:
            self._release_keys(op)
            return

        self._release_keys(op)

        pressed = op.keys_pressed
        if self.exclusive_control:
            pressed.discard(op.pref_forward_key)
            pressed.discard(op.pref_backward_key)
            pressed.discard(op.pref_left_key)
            pressed.discard(op.pref_right_key)
            pressed.discard(op.pref_run_key)

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
        if self.allow_run:
            try:
                max_run = float(bpy.context.scene.char_physics.max_speed_run)
                run_now = speed_hint >= max_run - 1e-3
            except Exception:
                pass
        if run_now:
            want.add(op.pref_run_key)

        for k in want:
            pressed.add(k)
        self._last_keys = want

    # ─────────────────────────────────────────────────────────────────────────
    # Character autopilot update (MAIN THREAD - just injects keys)
    # ─────────────────────────────────────────────────────────────────────────
    def update_character(self, now: float) -> bool:
        """Update character autopilot. Returns True when finished."""
        if self.max_runtime > 0.0 and (now - self.start_time) > self.max_runtime:
            return True

        to_obj = self.to_obj
        if not to_obj:
            return True

        try:
            tgt = to_obj.matrix_world.translation
        except Exception:
            return True

        mover = self.from_obj
        if not mover:
            return True

        cur = mover.location
        dx = tgt.x - cur.x
        dy = tgt.y - cur.y
        dist_sq = dx * dx + dy * dy
        if dist_sq <= self.arrive_radius * self.arrive_radius:
            return True

        op = getattr(exp_globals, "ACTIVE_MODAL_OP", None)
        wish_xy = Vector((dx, dy, 0.0))
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

    # Debug flag - cache once
    debug_tracking = getattr(bpy.context.scene, 'dev_debug_tracking', False)

    now = get_game_time()
    batch = []

    # Track which obj_ids are still active this frame
    active_obj_ids = set()

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

        to_obj = task.to_obj
        if not to_obj:
            continue

        # Get target position - minimal exception handling
        try:
            tgt = to_obj.matrix_world.translation
        except Exception:
            continue

        obj_id = id(mover)
        active_obj_ids.add(obj_id)

        # Only update lookup if not already present
        if obj_id not in _tracking_obj_lookup:
            _tracking_obj_lookup[obj_id] = mover

        # Use tuple directly - avoid Vector overhead
        loc = mover.location
        batch.append({
            "obj_id": obj_id,
            "current_pos": (loc.x, loc.y, loc.z),
            "goal_pos": (tgt.x, tgt.y, tgt.z),
            "speed": task.speed,
            "dt": dt,
            "arrive_radius": task.arrive_radius,
            "use_gravity": task.use_gravity,
            "use_collision": task.respect_proxy,
        })

    # Clean stale entries from lookup (objects no longer tracked)
    if len(_tracking_obj_lookup) > len(active_obj_ids):
        stale = [k for k in _tracking_obj_lookup if k not in active_obj_ids]
        for k in stale:
            del _tracking_obj_lookup[k]

    if not batch:
        return None

    _pending_tracking_job_id = engine.submit_job("TRACKING_BATCH", {
        "tracks": batch,
        "debug_logs": [] if debug_tracking else None,
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
            log_game("TRACKING", f"  {name}: pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) -> goal=({goal[0]:.2f},{goal[1]:.2f},{goal[2]:.2f}) collision={b['use_collision']} gravity={b['use_gravity']}")

    return _pending_tracking_job_id


def apply_tracking_results(result):
    """
    Apply computed positions from worker.
    Returns number of objects updated.
    """
    global _pending_tracking_job_id
    _pending_tracking_job_id = None

    # Debug flag - cache once
    debug_tracking = getattr(bpy.context.scene, 'dev_debug_tracking', False)

    if not result.success:
        if debug_tracking:
            from ..developer.dev_logger import log_game
            log_game("TRACKING", f"APPLY FAILED: {result.error}")
        return 0

    result_data = result.result
    results = result_data.get("results", [])
    applied = 0

    # Mark arrived tasks
    arrived_ids = set()
    lookup = _tracking_obj_lookup

    for r in results:
        obj_id = r["obj_id"]
        obj = lookup.get(obj_id)
        if not obj:
            continue

        # Direct tuple assignment - no Vector creation
        new_pos = r["new_pos"]
        obj.location = (new_pos[0], new_pos[1], new_pos[2])
        applied += 1

        if r.get("arrived", False):
            arrived_ids.add(obj_id)

    # Mark tasks as arrived - only iterate if we have arrivals
    if arrived_ids:
        for task in _ACTIVE_TRACKS:
            if not task._is_char_driver and task.from_obj:
                if id(task.from_obj) in arrived_ids:
                    task._arrived = True

    # Debug logging
    if debug_tracking:
        from ..developer.dev_logger import log_game
        rays = result_data.get("rays", 0)
        tris = result_data.get("tris", 0)
        calc_time = result_data.get("calc_time_us", 0)
        log_game("TRACKING", f"APPLY count={applied} arrived={len(arrived_ids)} rays={rays} tris={tris} calc_time={calc_time}us")

        for r in results:
            obj = lookup.get(r["obj_id"])
            name = obj.name if obj else "?"
            pos = r["new_pos"]
            status = "ARRIVED" if r.get("arrived") else "moving"
            log_game("TRACKING", f"  {name}: ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) {status}")

        # Output worker logs
        worker_logs = result_data.get("worker_logs", [])
        if worker_logs:
            from ..developer.dev_logger import log_worker_messages
            log_worker_messages(worker_logs)

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
    tracks = _ACTIVE_TRACKS
    if not tracks:
        return

    # Cache values once
    debug_tracking = getattr(bpy.context.scene, 'dev_debug_tracking', False)
    op = getattr(exp_globals, "ACTIVE_MODAL_OP", None)
    now = get_game_time()

    # Track finished indices for batch removal
    finished_indices = []
    char_count = 0
    obj_count = 0

    for i, t in enumerate(tracks):
        done = False

        if t._is_char_driver:
            # Character autopilot - inject keys (main thread)
            char_count += 1
            done = t.update_character(now)
            if debug_tracking and t._last_keys:
                from ..developer.dev_logger import log_game
                keys_str = ",".join(sorted(t._last_keys))
                log_game("TRACKING", f"AUTOPILOT keys=[{keys_str}] speed={t.speed:.1f}")
        else:
            # Non-character - check if arrived (position applied by worker)
            obj_count += 1
            done = t._arrived
            # Also check timeout
            if not done and t.max_runtime > 0.0:
                done = (now - t.start_time) > t.max_runtime

        if done:
            t._release_keys(op)
            finished_indices.append(i)

    # Log active track summary
    if debug_tracking and (char_count > 0 or obj_count > 0):
        from ..developer.dev_logger import log_game
        log_game("TRACKING", f"ACTIVE char={char_count} obj={obj_count} finished={len(finished_indices)}")

    # Batch removal - rebuild list excluding finished (O(n) instead of O(n²))
    if finished_indices:
        finished_set = set(finished_indices)
        _ACTIVE_TRACKS[:] = [t for i, t in enumerate(tracks) if i not in finished_set]


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
        # Remove existing character autopilot
        for i in range(len(_ACTIVE_TRACKS) - 1, -1, -1):
            t = _ACTIVE_TRACKS[i]
            if t._is_char_driver:
                t._release_keys(op)
                del _ACTIVE_TRACKS[i]

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
