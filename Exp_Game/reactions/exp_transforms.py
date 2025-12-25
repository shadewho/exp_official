# Exploratory/Exp_game/reactions/exp_transforms.py
"""
Transform reactions - worker-offloaded interpolation.

All lerp/slerp computation happens in the worker process.
Main thread only:
1. Schedules tasks and tracks timing
2. Submits batch to worker
3. Applies results (thin write loop)
"""

import bpy
from mathutils import Vector, Euler, Matrix
from ..props_and_utils.exp_time import get_game_time
from ..developer.dev_logger import log_game

# ------------------------------
# TransformTask + Manager
# ------------------------------

_active_transform_tasks = []
_pending_transform_job_id = None


class TransformTask:
    """Stores transform interpolation parameters. Computation happens in worker."""

    def __init__(self, obj, start_loc, start_rot, start_scl,
                 end_loc, end_rot, end_scl,
                 start_time, duration,
                 rot_interp='euler',
                 delta_euler=None):
        self.obj = obj
        self.obj_id = id(obj)  # For result matching

        # Store starting transforms (Euler = local space)
        self.start_loc = start_loc
        self.start_rot = start_rot           # Euler
        self.start_scl = start_scl

        # Targets
        self.end_loc = end_loc
        self.end_rot = end_rot               # Euler
        self.end_scl = end_scl

        # Rotation interpolation mode
        self.rot_interp  = rot_interp        # 'euler' | 'quat' | 'local_delta'
        self.delta_euler = delta_euler       # Euler (only for 'local_delta')

        # Quats for slerp path (used by 'quat')
        self.start_rot_q = start_rot.to_quaternion()
        self.end_rot_q   = end_rot.to_quaternion()

        self.start_time = start_time
        self.duration   = duration

        # Current t value (updated each frame, used by worker)
        self.current_t = 0.0

    def compute_t(self, now):
        """Compute current interpolation factor. Does NOT apply transforms."""
        if self.duration <= 0.0:
            self.current_t = 1.0
            return True  # finished

        t = (now - self.start_time) / self.duration
        if t >= 1.0:
            self.current_t = 1.0
            return True  # finished

        self.current_t = t
        return False

    def to_worker_data(self):
        """Serialize task data for worker job."""
        return {
            "obj_id": self.obj_id,
            "t": self.current_t,
            "start_loc": tuple(self.start_loc),
            "end_loc": tuple(self.end_loc),
            "start_rot_q": (
                self.start_rot_q.w,
                self.start_rot_q.x,
                self.start_rot_q.y,
                self.start_rot_q.z,
            ),
            "end_rot_q": (
                self.end_rot_q.w,
                self.end_rot_q.x,
                self.end_rot_q.y,
                self.end_rot_q.z,
            ),
            "start_scl": tuple(self.start_scl),
            "end_scl": tuple(self.end_scl),
            "rot_mode": self.rot_interp,
            "start_rot_e": tuple(self.start_rot),
            "end_rot_e": tuple(self.end_rot),
            "delta_euler": tuple(self.delta_euler) if self.delta_euler else (0.0, 0.0, 0.0),
        }


def schedule_transform(obj, end_loc, end_rot, end_scl, duration,
                       rot_interp='euler', delta_euler=None):
    """
    Schedule a transform interpolation task.

    rot_interp: 'euler' | 'quat' | 'local_delta'
    If rot_interp == 'local_delta', pass delta_euler (Euler local offset).
    """
    if not obj:
        return

    start_loc = obj.location.copy()
    start_rot = obj.rotation_euler.copy()
    start_scl = obj.scale.copy()

    start_time = get_game_time()

    task = TransformTask(
        obj,
        start_loc, start_rot, start_scl,
        end_loc, end_rot, end_scl,
        start_time, duration,
        rot_interp=rot_interp,
        delta_euler=delta_euler
    )
    _active_transform_tasks.append(task)

    log_game("TRANSFORMS", f"SCHEDULE obj={obj.name} mode={rot_interp} dur={duration:.2f}s")


def update_transform_tasks():
    """
    Called once per frame. Updates t values for all active tasks.
    Does NOT apply transforms - that happens after worker results arrive.
    """
    if not _active_transform_tasks:
        return

    now = get_game_time()
    for task in _active_transform_tasks:
        task.compute_t(now)


def submit_transform_batch(engine):
    """
    Submit all active transform tasks to worker for computation.
    Returns job_id or None if no tasks.
    """
    global _pending_transform_job_id

    if not _active_transform_tasks:
        return None

    if not engine or not engine.is_alive():
        return None

    # Don't submit if we're still waiting for previous results
    if _pending_transform_job_id is not None:
        return None

    # Build batch data
    transforms_data = [task.to_worker_data() for task in _active_transform_tasks]

    job_id = engine.submit_job("TRANSFORM_BATCH", {
        "transforms": transforms_data,
    })

    _pending_transform_job_id = job_id

    log_game("TRANSFORMS", f"SUBMIT batch={len(transforms_data)} job_id={job_id}")

    return job_id


def apply_transform_results(result):
    """
    Apply computed transforms from worker and remove finished tasks.
    Called from game loop when TRANSFORM_BATCH result arrives.
    """
    global _pending_transform_job_id
    _pending_transform_job_id = None

    if not result.success:
        log_game("TRANSFORMS", f"ERROR: {result.error}")
        return

    result_data = result.result
    if not result_data:
        return

    results = result_data.get("results", [])
    calc_time = result_data.get("calc_time_us", 0.0)

    # Build lookup by obj_id
    results_by_id = {r["obj_id"]: r for r in results}

    # Apply results and track finished tasks
    finished_indices = []
    finished_names = []

    for i, task in enumerate(_active_transform_tasks):
        r = results_by_id.get(task.obj_id)
        if not r:
            continue

        # Check if object still exists
        obj = task.obj
        if obj is None:
            finished_indices.append(i)
            continue

        # Apply transforms
        try:
            obj.location = Vector(r["loc"])
            obj.rotation_euler = Euler(r["rot_euler"], 'XYZ')
            obj.scale = Vector(r["scl"])
        except ReferenceError:
            # Object was deleted
            finished_indices.append(i)
            continue

        # Remove if finished
        if r.get("finished", False):
            finished_indices.append(i)
            try:
                finished_names.append(f"{obj.name}({task.rot_interp})")
            except:
                finished_names.append("(deleted)")

    # Remove finished tasks in reverse order
    for i in reversed(finished_indices):
        _active_transform_tasks.pop(i)

    # Process worker logs
    worker_logs = result_data.get("logs", [])
    if worker_logs:
        from ..developer.dev_logger import log_worker_messages
        log_worker_messages(worker_logs)

    # Build log message
    if finished_names:
        log_game("TRANSFORMS", f"APPLY count={len(results)} finished={len(finished_indices)} [{', '.join(finished_names)}] calc_time={calc_time:.0f}us")
    else:
        log_game("TRANSFORMS", f"APPLY count={len(results)} active={len(_active_transform_tasks)} calc_time={calc_time:.0f}us")


def get_pending_job_id():
    """Get the current pending job ID (for polling)."""
    return _pending_transform_job_id


def poll_transform_result_with_timeout(engine, job_id, timeout=0.002):
    """
    Poll for transform batch result with timeout (same-frame sync).

    CRITICAL: This must be called BEFORE KCC physics runs!
    Otherwise platforms move after physics reads their position,
    causing characters to sink/shift on moving platforms.

    Args:
        engine: The worker engine
        job_id: The transform job ID to wait for
        timeout: Max wait time in seconds (default 2ms)

    Returns:
        True if result was applied, False if timeout
    """
    import time

    if job_id is None or engine is None:
        return False

    poll_start = time.perf_counter()
    result_found = False
    cached_other_results = []

    while True:
        elapsed = time.perf_counter() - poll_start
        if elapsed >= timeout:
            break

        results = engine.poll_results(max_results=10)
        for result in results:
            if result.job_id == job_id and result.job_type == "TRANSFORM_BATCH":
                # Found our result! Apply immediately
                if result.success:
                    apply_transform_results(result)
                result_found = True
                break
            else:
                # Cache other results for later processing
                cached_other_results.append(result)

        if result_found:
            break

        # Small sleep to avoid spinning
        time.sleep(0.0001)  # 100µs

    # Re-queue cached results for later processing
    # (These will be picked up by _poll_and_apply_engine_results)
    if cached_other_results:
        _cache_other_transform_results(cached_other_results)

    # Debug logging
    if result_found:
        log_game("TRANSFORMS", f"SAME-FRAME sync job={job_id} poll={elapsed*1000:.1f}ms")

    return result_found


# Cache for other results encountered during transform polling
_cached_other_results = []

def _cache_other_transform_results(results):
    """Cache non-transform results for later processing."""
    global _cached_other_results
    _cached_other_results.extend(results)

def get_cached_other_results():
    """Get and clear cached non-transform results."""
    global _cached_other_results
    results = _cached_other_results
    _cached_other_results = []
    return results


def clear_transform_tasks():
    """Clear all transform tasks. Called on game stop."""
    global _pending_transform_job_id, _cached_other_results
    _active_transform_tasks.clear()
    _pending_transform_job_id = None
    _cached_other_results = []


# ------------------------------
# Reaction Execution
# ------------------------------

def execute_transform_reaction(reaction):
    """
    Applies a transform reaction to either:
      - the scene's target_armature (if use_character=True), or
      - the specified transform_object.
    """
    scene = bpy.context.scene

    # 1) Pick target: character vs. user-picked object
    if getattr(reaction, "use_character", False):
        target_obj = scene.target_armature
    else:
        target_obj = reaction.transform_object

    # 2) Bail if nothing to move
    if not target_obj:
        return

    # 3) Ensure Euler XYZ rotation mode
    target_obj.rotation_mode = 'XYZ'

    # 4) Clamp duration
    duration = reaction.transform_duration
    if duration < 0.0:
        duration = 0.0

    # 5) Dispatch based on transform_mode
    mode = reaction.transform_mode

    if mode == "OFFSET":
        apply_offset_transform(reaction, target_obj, duration)

    elif mode == "TO_LOCATION":
        apply_to_location_transform(reaction, target_obj, duration)

    elif mode == "TO_OBJECT":
        to_obj = (
            scene.target_armature
            if getattr(reaction, "transform_to_use_character", False)
            else reaction.transform_to_object
        )
        if not to_obj:
            return

        start_loc = target_obj.location.copy()
        start_rot = target_obj.rotation_euler.copy()
        start_scl = target_obj.scale.copy()

        end_loc = (
            to_obj.location.copy()
            if reaction.transform_use_location
            else start_loc
        )
        end_rot = (
            to_obj.rotation_euler.copy()
            if reaction.transform_use_rotation
            else start_rot
        )
        end_scl = (
            to_obj.scale.copy()
            if reaction.transform_use_scale
            else start_scl
        )
        schedule_transform(target_obj, end_loc, end_rot, end_scl, duration)

    elif mode == "LOCAL_OFFSET":
        apply_local_offset_transform(reaction, target_obj, duration)

    elif mode == "TO_BONE":
        apply_to_bone_transform(reaction, target_obj, duration)


def apply_offset_transform(reaction, target_obj, duration):
    """
    Global offset around the object's current origin, preserving spin.
    """
    loc_off = Vector(reaction.transform_location)
    rot_off = Euler(reaction.transform_rotation, 'XYZ')
    scl_off = Vector(reaction.transform_scale)

    start_loc = target_obj.location.copy()
    start_rot = target_obj.rotation_euler.copy()
    start_scl = target_obj.scale.copy()

    T_off = Matrix.Translation(loc_off)
    R_off = rot_off.to_matrix().to_4x4()
    S_off = Matrix.Diagonal((scl_off.x, scl_off.y, scl_off.z, 1.0))
    user_offset_mat = T_off @ R_off @ S_off

    pivot_world = target_obj.matrix_world.translation
    pivot_inv = Matrix.Translation(-pivot_world)
    pivot_fwd = Matrix.Translation(pivot_world)

    start_mat  = target_obj.matrix_world.copy()
    offset_mat = pivot_fwd @ user_offset_mat @ pivot_inv
    final_mat  = offset_mat @ start_mat

    end_loc, _end_rot_q, end_scl = final_mat.decompose()

    end_rot = Euler((
        start_rot.x + rot_off.x,
        start_rot.y + rot_off.y,
        start_rot.z + rot_off.z,
    ), 'XYZ')

    schedule_transform(target_obj, end_loc, end_rot, end_scl, duration)


def apply_to_bone_transform(reaction, target_obj, duration):
    """
    Copy transforms from a specific bone (by name) onto target_obj.
    """
    scn = bpy.context.scene

    use_char = bool(getattr(reaction, "transform_to_bone_use_character", True))
    arm_obj = scn.target_armature if use_char else getattr(reaction, "transform_to_armature", None)
    if not arm_obj or getattr(arm_obj, "type", "") != 'ARMATURE':
        return

    bone_name = (getattr(reaction, "transform_bone_name", "") or "").strip()
    if not bone_name:
        return
    try:
        pb = arm_obj.pose.bones.get(bone_name)
    except Exception:
        pb = None
    if not pb:
        return

    start_loc = target_obj.location.copy()
    start_rot = target_obj.rotation_euler.copy()
    start_scl = target_obj.scale.copy()

    world_mat = arm_obj.matrix_world @ pb.matrix
    try:
        end_loc_w, end_rot_q, end_scl_v = world_mat.decompose()
        end_rot_e = end_rot_q.to_euler('XYZ')
    except Exception:
        end_loc_w = world_mat.translation
        end_rot_e = world_mat.to_euler('XYZ')
        end_scl_v = Vector((1.0, 1.0, 1.0))

    use_loc = bool(getattr(reaction, "transform_use_location", True))
    use_rot = bool(getattr(reaction, "transform_use_rotation", True))
    use_scl = bool(getattr(reaction, "transform_use_scale", True))

    end_loc = end_loc_w if use_loc else start_loc
    end_rot = end_rot_e if use_rot else start_rot
    end_scl = end_scl_v if use_scl else start_scl

    schedule_transform(target_obj, end_loc, end_rot, end_scl, duration)


def apply_to_location_transform(reaction, target_obj, duration):
    """
    Interpret transform values as absolute world transforms.
    """
    end_loc = Vector(reaction.transform_location)
    end_rot = Euler(reaction.transform_rotation, 'XYZ')
    end_scl = Vector(reaction.transform_scale)

    schedule_transform(target_obj, end_loc, end_rot, end_scl, duration)


def apply_to_object_transform(reaction, target_obj, to_obj, duration):
    """
    Copy transforms from another object.
    """
    end_loc = to_obj.location.copy()
    end_rot = to_obj.rotation_euler.copy()
    end_scl = to_obj.scale.copy()

    schedule_transform(target_obj, end_loc, end_rot, end_scl, duration)


def apply_local_offset_transform(reaction, target_obj, duration):
    """
    LOCAL offset:
      - Rotation: about the object's CURRENT LOCAL axes.
      - Translation: along local axes.
      - Scale: component-wise in local space.
    """
    from math import tau
    loc_off = Vector(reaction.transform_location)
    rot_off = Euler(reaction.transform_rotation, 'XYZ')
    scl_off = Vector(reaction.transform_scale)

    start_loc = target_obj.location.copy()
    start_rot_eul = target_obj.rotation_euler.copy()
    start_rot_q = start_rot_eul.to_quaternion()
    start_scl = target_obj.scale.copy()

    delta_q_local = rot_off.to_quaternion()
    end_rot_q = start_rot_q @ delta_q_local
    end_rot_base = end_rot_q.to_euler('XYZ')

    def unwrap_to_target(base_val, target_val):
        k = round((target_val - base_val) / tau)
        return base_val + k * tau

    target_eul = (
        start_rot_eul.x + rot_off.x,
        start_rot_eul.y + rot_off.y,
        start_rot_eul.z + rot_off.z,
    )
    end_rot = Euler((
        unwrap_to_target(end_rot_base.x, target_eul[0]),
        unwrap_to_target(end_rot_base.y, target_eul[1]),
        unwrap_to_target(end_rot_base.z, target_eul[2]),
    ), 'XYZ')

    R_world_n = target_obj.matrix_world.to_3x3().normalized()
    end_loc = start_loc + (R_world_n @ loc_off)

    end_scl = Vector((
        start_scl.x * scl_off.x,
        start_scl.y * scl_off.y,
        start_scl.z * scl_off.z,
    ))

    schedule_transform(
        target_obj,
        end_loc, end_rot, end_scl,
        duration,
        rot_interp='local_delta',
        delta_euler=rot_off
    )
