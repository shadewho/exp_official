# Exploratory/Exp_Game/reactions/exp_tracking.py
"""
Tracking system - moves objects toward targets.

Character mover: "autopilot" by injecting keys for KCC (reuses full physics)
Non-character mover: Offloaded to worker (collision/gravity via TRACKING_BATCH)

OPTIMIZED for many simultaneous tracking objects:
  - Minimal main thread work (collect data, apply results)
  - Object lookup dict persisted between frames
  - Pre-computed squared arrive radius
  - Batch removal of finished tasks (O(n) not O(n²))
  - Direct tuple assignment for positions (no Vector creation)
  - Single debug flag check per frame
"""
import math
import bpy
from mathutils import Vector, Matrix
from ..props_and_utils.exp_time import get_game_time
from ..audio import exp_globals  # ACTIVE_MODAL_OP
from .exp_projectiles import get_dynamic_transforms  # reuse for dynamic mesh support


# ─────────────────────────────────────────────────────────────────────────────
# Module state
# ─────────────────────────────────────────────────────────────────────────────
_ACTIVE_TRACKS: list = []
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
        "r_id", "from_obj", "to_obj", "mode",
        "use_gravity", "respect_proxy", "speed", "arrive_radius", "arrive_radius_sq",
        "exclusive_control", "allow_run", "max_runtime", "start_time",
        "_is_char_driver", "_last_keys", "_arrived", "_obj_id",
    )

    def __init__(self, r_id, from_obj, to_obj, mode='DIRECT',
                 use_gravity=True, respect_proxy=True, speed=3.5,
                 arrive_radius=0.3, exclusive_control=True, allow_run=True,
                 max_runtime=0.0):
        self.r_id = r_id
        self.from_obj = from_obj
        self.to_obj = to_obj
        self.mode = mode  # 'DIRECT' or 'PATHFINDING'
        self.use_gravity = bool(use_gravity)
        self.respect_proxy = bool(respect_proxy)
        self.speed = float(speed)
        ar = max(0.0, float(arrive_radius))
        self.arrive_radius = ar
        self.arrive_radius_sq = ar * ar  # Pre-compute for fast distance checks
        self.exclusive_control = bool(exclusive_control)
        self.allow_run = bool(allow_run)
        self.max_runtime = max(0.0, float(max_runtime))
        self.start_time = get_game_time()
        self._last_keys = set()
        self._arrived = False
        self._obj_id = id(from_obj) if from_obj else 0  # Cache object ID
        op = getattr(exp_globals, "ACTIVE_MODAL_OP", None)
        self._is_char_driver = (op is not None and op.target_object and from_obj == op.target_object)

    def _release_keys(self, op):
        """Release any injected keys (character autopilot only)."""
        if not self._is_char_driver or not op:
            return
        keys = self._last_keys
        if keys:
            pressed = op.keys_pressed
            for k in keys:
                pressed.discard(k)
            keys.clear()

    def _inject_autopilot(self, op, dx: float, dy: float, speed_hint: float):
        """
        Convert desired world-plane direction -> camera-local -> discrete keys.
        Only press 0-2 direction keys + optional run.
        """
        if not op:
            return

        dist_sq = dx * dx + dy * dy
        if dist_sq <= 1.0e-10:
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

        # Transform world direction to camera-local using yaw
        yaw = getattr(op, 'yaw', 0.0)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        # Rotate by -yaw: x' = x*cos + y*sin, y' = -x*sin + y*cos
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw

        want = set()
        thr = 0.18
        if local_y > thr:
            want.add(op.pref_forward_key)
        elif local_y < -thr:
            want.add(op.pref_backward_key)
        if local_x > thr:
            want.add(op.pref_right_key)
        elif local_x < -thr:
            want.add(op.pref_left_key)

        # Check if we should run
        if self.allow_run:
            try:
                max_run = float(bpy.context.scene.char_physics.max_speed_run)
                if speed_hint >= max_run - 1e-3:
                    want.add(op.pref_run_key)
            except Exception:
                pass

        for k in want:
            pressed.add(k)
        self._last_keys = want

    def update_character(self, now: float, op) -> bool:
        """Update character autopilot. Returns True when finished."""
        if self.max_runtime > 0.0 and (now - self.start_time) > self.max_runtime:
            return True

        to_obj = self.to_obj
        if not to_obj:
            return True

        mover = self.from_obj
        if not mover:
            return True

        # Get positions directly - avoid try/except in hot path
        try:
            tgt = to_obj.matrix_world.translation
            cur = mover.location
        except Exception:
            return True

        dx = tgt.x - cur.x
        dy = tgt.y - cur.y
        dist_sq = dx * dx + dy * dy

        if dist_sq <= self.arrive_radius_sq:
            return True

        self._inject_autopilot(op, dx, dy, self.speed)
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

    tracks = _ACTIVE_TRACKS
    if not tracks:
        return None

    # Don't submit if a job is already pending
    if _pending_tracking_job_id is not None:
        return None

    # Cache values once
    debug_tracking = getattr(bpy.context.scene, 'dev_debug_tracking', False)
    now = get_game_time()
    lookup = _tracking_obj_lookup

    # Pre-allocate batch list (estimate: most tracks are non-character)
    batch = []
    active_obj_ids = set()

    for task in tracks:
        # Skip character tasks (handled via key injection)
        if task._is_char_driver:
            continue

        # Skip finished tasks
        if task._arrived:
            continue

        # Skip timed-out tasks
        if task.max_runtime > 0.0 and (now - task.start_time) > task.max_runtime:
            continue

        mover = task.from_obj
        to_obj = task.to_obj
        if not mover or not to_obj:
            continue

        # Get positions - minimal exception handling
        try:
            tgt = to_obj.matrix_world.translation
            loc = mover.location
        except Exception:
            continue

        obj_id = task._obj_id
        active_obj_ids.add(obj_id)

        # Only update lookup if not present
        if obj_id not in lookup:
            lookup[obj_id] = mover

        batch.append({
            "obj_id": obj_id,
            "current_pos": (loc.x, loc.y, loc.z),
            "goal_pos": (tgt.x, tgt.y, tgt.z),
            "speed": task.speed,
            "dt": dt,
            "arrive_radius": task.arrive_radius,
            "use_gravity": task.use_gravity,
            "use_collision": task.respect_proxy,
            "mode": task.mode,  # 'DIRECT' or 'PATHFINDING'
        })

    # Clean stale entries from lookup
    if len(lookup) > len(active_obj_ids):
        stale = [k for k in lookup if k not in active_obj_ids]
        for k in stale:
            del lookup[k]

    if not batch:
        return None

    # Get current dynamic mesh transforms for collision detection
    dynamic_transforms = get_dynamic_transforms()

    _pending_tracking_job_id = engine.submit_job("TRACKING_BATCH", {
        "tracks": batch,
        "dynamic_transforms": dynamic_transforms,
        "debug_logs": [] if debug_tracking else None,
    })

    if debug_tracking and _pending_tracking_job_id is not None:
        from ..developer.dev_logger import log_game
        log_game("TRACKING", f"SUBMIT batch={len(batch)} job_id={_pending_tracking_job_id}")

    return _pending_tracking_job_id


def apply_tracking_results(result):
    """
    Apply computed positions from worker.
    Returns number of objects updated.
    """
    global _pending_tracking_job_id
    _pending_tracking_job_id = None

    if not result.success:
        return 0

    result_data = result.result
    results = result_data.get("results", [])
    if not results:
        return 0

    lookup = _tracking_obj_lookup
    arrived_ids = set()
    applied = 0

    # Apply positions (hot loop - minimal overhead)
    for r in results:
        obj_id = r["obj_id"]
        obj = lookup.get(obj_id)
        if obj:
            new_pos = r["new_pos"]
            obj.location = (new_pos[0], new_pos[1], new_pos[2])
            applied += 1
            if r.get("arrived", False):
                arrived_ids.add(obj_id)

    # Mark tasks as arrived
    if arrived_ids:
        for task in _ACTIVE_TRACKS:
            if not task._is_char_driver and task._obj_id in arrived_ids:
                task._arrived = True

    # Debug logging (only if enabled)
    debug_tracking = getattr(bpy.context.scene, 'dev_debug_tracking', False)
    if debug_tracking:
        from ..developer.dev_logger import log_game
        rays = result_data.get("rays", 0)
        tris = result_data.get("tris", 0)
        calc_time = result_data.get("calc_time_us", 0)
        log_game("TRACKING", f"APPLY count={applied} arrived={len(arrived_ids)} rays={rays} tris={tris} {calc_time}us")

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
    - Non-character tasks: arrival checked via _arrived flag (set by apply_tracking_results)
    """
    tracks = _ACTIVE_TRACKS
    if not tracks:
        return

    # Cache values once
    op = getattr(exp_globals, "ACTIVE_MODAL_OP", None)
    now = get_game_time()

    # Track finished indices for batch removal
    finished_indices = []

    for i, t in enumerate(tracks):
        done = False

        if t._is_char_driver:
            # Character autopilot - inject keys (main thread)
            done = t.update_character(now, op)
        else:
            # Non-character - check if arrived or timed out
            done = t._arrived
            if not done and t.max_runtime > 0.0:
                done = (now - t.start_time) > t.max_runtime

        if done:
            t._release_keys(op)
            finished_indices.append(i)

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
        # Remove existing character autopilot (only one allowed at a time)
        for i in range(len(_ACTIVE_TRACKS) - 1, -1, -1):
            t = _ACTIVE_TRACKS[i]
            if t._is_char_driver:
                t._release_keys(op)
                del _ACTIVE_TRACKS[i]

    task = _TrackTask(
        r_id=r,
        from_obj=from_obj,
        to_obj=to_obj,
        mode=getattr(r, "track_mode", 'DIRECT'),
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
    start(r)


def get_active_count() -> int:
    """Return number of active tracking tasks (for diagnostics)."""
    return len(_ACTIVE_TRACKS)


def get_object_track_count() -> int:
    """Return number of non-character tracking tasks (worker-handled)."""
    return sum(1 for t in _ACTIVE_TRACKS if not t._is_char_driver)
