# Exploratory/Exp_Game/reactions/exp_tracking.py
"""
Tracking system - moves objects toward targets via engine worker.

ALL tracking is offloaded to the worker (TRACKING_BATCH).
Worker handles collision (static + dynamic meshes) and gravity
(velocity-based falling with ground detection).

OPTIMIZED for many simultaneous tracking objects:
  - Minimal main thread work (collect data, apply results)
  - Object lookup dict persisted between frames
  - Pre-computed squared arrive radius
  - Batch removal of finished tasks (O(n) not O(n^2))
  - Direct tuple assignment for positions (no Vector creation)
  - Single debug flag check per frame
"""
import bpy
from ..props_and_utils.exp_time import get_game_time
from .exp_projectiles import get_dynamic_transforms


# ─────────────────────────────────────────────────────────────────────────────
# Module state
# ─────────────────────────────────────────────────────────────────────────────
_ACTIVE_TRACKS: list = []
_tracking_obj_lookup: dict[int, bpy.types.Object] = {}  # obj_id -> bpy object (persisted)
_pending_tracking_job_id = None


def clear():
    """Stop all active tracks."""
    global _pending_tracking_job_id
    _ACTIVE_TRACKS.clear()
    _tracking_obj_lookup.clear()
    _pending_tracking_job_id = None


# ─────────────────────────────────────────────────────────────────────────────
# Track Task
# ─────────────────────────────────────────────────────────────────────────────
class _TrackTask:
    __slots__ = (
        "r_id", "from_obj", "to_obj", "mode",
        "use_gravity", "respect_proxy", "speed",
        "arrive_radius", "arrive_radius_sq",
        "max_runtime", "start_time",
        "_arrived", "_obj_id",
        "face_enabled", "face_target_obj", "face_axis",
    )

    def __init__(self, r_id, from_obj, to_obj, mode='DIRECT',
                 use_gravity=True, respect_proxy=True, speed=3.5,
                 arrive_radius=0.3, max_runtime=0.0,
                 face_enabled=False, face_target_obj=None, face_axis="NEG_Y"):
        self.r_id = r_id
        self.from_obj = from_obj
        self.to_obj = to_obj
        self.mode = mode
        self.use_gravity = bool(use_gravity)
        self.respect_proxy = bool(respect_proxy)
        self.speed = float(speed)
        ar = max(0.0, float(arrive_radius))
        self.arrive_radius = ar
        self.arrive_radius_sq = ar * ar
        self.max_runtime = max(0.0, float(max_runtime))
        self.start_time = get_game_time()
        self._arrived = False
        self._obj_id = id(from_obj) if from_obj else 0
        self.face_enabled = bool(face_enabled)
        self.face_target_obj = face_target_obj
        self.face_axis = face_axis


# ─────────────────────────────────────────────────────────────────────────────
# Worker offload: Submit / Apply
# ─────────────────────────────────────────────────────────────────────────────

def submit_tracking_batch(engine, dt: float):
    """
    Submit tracking tasks to worker for movement computation.
    Returns job_id if submitted, None otherwise.
    """
    global _pending_tracking_job_id

    tracks = _ACTIVE_TRACKS
    if not tracks:
        return None

    if _pending_tracking_job_id is not None:
        return None

    debug_tracking = getattr(bpy.context.scene, 'dev_debug_tracking', False)
    now = get_game_time()
    lookup = _tracking_obj_lookup

    batch = []
    active_obj_ids = set()

    for task in tracks:
        if task._arrived:
            continue

        if task.max_runtime > 0.0 and (now - task.start_time) > task.max_runtime:
            continue

        mover = task.from_obj
        to_obj = task.to_obj
        if not mover or not to_obj:
            continue

        try:
            tgt = to_obj.matrix_world.translation
            loc = mover.location
        except Exception:
            continue

        obj_id = task._obj_id
        active_obj_ids.add(obj_id)

        if obj_id not in lookup:
            lookup[obj_id] = mover

        entry = {
            "obj_id": obj_id,
            "current_pos": (loc.x, loc.y, loc.z),
            "goal_pos": (tgt.x, tgt.y, tgt.z),
            "speed": task.speed,
            "dt": dt,
            "arrive_radius": task.arrive_radius,
            "use_gravity": task.use_gravity,
            "use_collision": task.respect_proxy,
            "mode": task.mode,
        }

        if task.face_enabled and task.face_target_obj:
            try:
                ft = task.face_target_obj.matrix_world.translation
                entry["face_target_pos"] = (ft.x, ft.y, ft.z)
                entry["face_axis"] = task.face_axis
            except Exception:
                pass

        batch.append(entry)

    # Clean stale entries from lookup
    if len(lookup) > len(active_obj_ids):
        stale = [k for k in lookup if k not in active_obj_ids]
        for k in stale:
            del lookup[k]

    if not batch:
        return None

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

    for r in results:
        obj_id = r["obj_id"]
        obj = lookup.get(obj_id)
        if obj:
            new_pos = r["new_pos"]
            obj.location = (new_pos[0], new_pos[1], new_pos[2])

            face_z = r.get("face_euler_z")
            if face_z is not None:
                obj.rotation_euler.z = face_z
            face_x = r.get("face_euler_x")
            if face_x is not None:
                obj.rotation_euler.x = face_x

            applied += 1
            if r.get("arrived", False):
                arrived_ids.add(obj_id)

    if arrived_ids:
        for task in _ACTIVE_TRACKS:
            if task._obj_id in arrived_ids:
                task._arrived = True

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
    Checks for arrived/timed-out tasks and removes them.
    Movement is computed by the worker via submit_tracking_batch.
    """
    tracks = _ACTIVE_TRACKS
    if not tracks:
        return

    now = get_game_time()
    finished_indices = []

    for i, t in enumerate(tracks):
        done = t._arrived
        if not done and t.max_runtime > 0.0:
            done = (now - t.start_time) > t.max_runtime
        if done:
            finished_indices.append(i)

    if finished_indices:
        finished_set = set(finished_indices)
        _ACTIVE_TRACKS[:] = [t for i, t in enumerate(tracks) if i not in finished_set]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def start(r):
    """Build and queue a tracking task from a ReactionDefinition instance."""
    from .exp_bindings import resolve_object, resolve_bool

    from_obj = resolve_object(r, "track_from_object", r.track_from_object)
    to_obj = resolve_object(r, "track_to_object", r.track_to_object)

    # Resolve face target
    face_enabled = resolve_bool(r, "track_face_enabled", getattr(r, "track_face_enabled", False))
    face_target_obj = None
    face_axis = getattr(r, "track_face_axis", "NEG_Y")
    if face_enabled:
        face_target_obj = resolve_object(r, "track_face_object", r.track_face_object)

    # Remove existing track for this mover (only one track per object)
    if from_obj:
        mover_id = id(from_obj)
        _ACTIVE_TRACKS[:] = [t for t in _ACTIVE_TRACKS if t._obj_id != mover_id]

    task = _TrackTask(
        r_id=r,
        from_obj=from_obj,
        to_obj=to_obj,
        mode=getattr(r, "track_mode", 'DIRECT'),
        use_gravity=getattr(r, "track_use_gravity", True),
        respect_proxy=getattr(r, "track_respect_proxy_meshes", True),
        speed=getattr(r, "track_speed", 3.5),
        arrive_radius=getattr(r, "track_arrive_radius", 0.3),
        max_runtime=getattr(r, "track_max_runtime", 0.0),
        face_enabled=face_enabled,
        face_target_obj=face_target_obj,
        face_axis=face_axis,
    )
    _ACTIVE_TRACKS.append(task)


def execute_tracking_reaction(r):
    """Start a Track To task."""
    start(r)


def get_active_count() -> int:
    """Return number of active tracking tasks (for diagnostics)."""
    return len(_ACTIVE_TRACKS)
