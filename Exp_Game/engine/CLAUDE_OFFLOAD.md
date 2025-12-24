# Engine Offload Plan

> **Goal**: Keep the main thread FREE from computations. Offload everything possible to worker processes.

---

## Current Architecture Status

### Already Offloaded to Worker

| System | Job Type | Location | Status |
|--------|----------|----------|--------|
| KCC Physics | `KCC_PHYSICS_STEP` | `worker/physics.py` | Complete |
| Camera Occlusion | `CAMERA_OCCLUSION_FULL` | `worker/jobs.py` | Complete |
| Performance Culling | `CULL_BATCH` | `worker/jobs.py` | Complete |
| Animation Blending | `ANIMATION_COMPUTE_BATCH` | `engine/animations/blend.py` | Complete |
| Hitscan | `HITSCAN_BATCH` | `worker/reactions/hitscan.py` | Complete |
| Projectiles | `PROJECTILE_UPDATE_BATCH` | `worker/reactions/projectiles.py` | Complete |
| Tracker Evaluation | `EVALUATE_TRACKERS` | `worker/interactions/trackers.py` | Complete |
| Interaction Checks | `INTERACTION_CHECK_BATCH` | `worker/interactions/triggers.py` | Complete |
| Dynamic Mesh Cache | `CACHE_DYNAMIC_MESH` | `worker/entry.py` | Complete |
| Static Grid Cache | `CACHE_GRID` | `worker/entry.py` | Complete |

---

## Phase 1: Quick Wins (No Worker Changes)

### 1.1 World State Filtering - DONE
**File**: `interactions/exp_tracker_eval.py`

**Current Problem**:
```python
def collect_world_state(context):
    positions = {}
    for obj in bpy.data.objects:  # ITERATES ALL OBJECTS EVERY FRAME
        if obj.type in {'MESH', 'ARMATURE', 'EMPTY'}:
            loc = obj.matrix_world.translation
            positions[obj.name] = (loc.x, loc.y, loc.z)
```

**Solution**: Pre-filter at game start, only collect referenced objects.

```python
# At game start, extract from serialize_tracker_graph()
_tracked_object_names: set[str] = set()

def _extract_referenced_objects(tracker_data: list):
    """Extract object names referenced by tracker nodes."""
    global _tracked_object_names
    _tracked_object_names.clear()

    for tracker in tracker_data:
        tree = tracker.get("condition_tree", {})
        _extract_from_tree(tree)

def _extract_from_tree(node: dict):
    """Recursively extract object references from serialized node."""
    if not node:
        return

    # Distance tracker
    if node.get("object_a"):
        _tracked_object_names.add(node["object_a"])
    if node.get("object_b"):
        _tracked_object_names.add(node["object_b"])

    # Contact tracker
    if node.get("object"):
        _tracked_object_names.add(node["object"])
    for target in node.get("targets", []):
        _tracked_object_names.add(target)

    # Recurse into logic gate inputs
    for child in node.get("inputs", []):
        _extract_from_tree(child)

def collect_world_state(context):
    positions = {}
    # Only collect tracked objects
    for name in _tracked_object_names:
        obj = bpy.data.objects.get(name)
        if obj:
            loc = obj.matrix_world.translation
            positions[name] = (loc.x, loc.y, loc.z)
    # Always include character
    char = context.scene.target_armature
    if char:
        loc = char.matrix_world.translation
        positions[char.name] = (loc.x, loc.y, loc.z)
    ...
```

**Impact**: Reduces O(all_objects) to O(tracked_objects) per frame.

---

### 1.2 AABB Caching for Interactions - DONE
**File**: `interactions/exp_interactions.py`

**Current Problem**:
```python
def get_aabb(obj):
    # 8 matrix multiplications PER OBJECT PER FRAME
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return (min/max per axis...)
```

**Solution**: Cache AABB with matrix version check.

```python
_aabb_cache: dict[int, dict] = {}

def get_cached_aabb(obj) -> tuple:
    """Get AABB, using cache if transform unchanged."""
    key = id(obj)
    cached = _aabb_cache.get(key)

    # Check if matrix changed (fast tuple comparison)
    current_matrix = tuple(obj.matrix_world[i][j] for i in range(4) for j in range(4))

    if cached and cached["matrix"] == current_matrix:
        return cached["aabb"]

    # Recalculate
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    aabb = (
        min(pt.x for pt in corners), max(pt.x for pt in corners),
        min(pt.y for pt in corners), max(pt.y for pt in corners),
        min(pt.z for pt in corners), max(pt.z for pt in corners)
    )

    _aabb_cache[key] = {"aabb": aabb, "matrix": current_matrix}
    return aabb

def clear_aabb_cache():
    """Call on game end or scene change."""
    _aabb_cache.clear()
```

**Impact**: Static objects = 0 recalculations. Moving objects = 1 recalculation per frame only if transform changed.

---

### 1.3 Sound Distance Batching
**File**: `audio/exp_globals.py`

**Current Problem**: Each sound task calculates distance individually.

**Solution**: Batch all distance calculations in one pass.

```python
def update_sound_tasks():
    if not _sound_tasks:
        return

    now = get_game_time()
    char_pos = None

    # Get character position ONCE
    scn = bpy.context.scene
    if scn.target_armature:
        char_pos = scn.target_armature.location

    # Batch process all tasks
    to_remove = []
    for i, task in enumerate(_sound_tasks):
        # Duration check
        if task.mode == "DURATION" and (now - task.start_time) >= task.duration:
            task.handle.stop()
            to_remove.append(i)
            continue

        # Distance attenuation (single calculation per task)
        if task.use_distance and char_pos and task.dist_object:
            dist = (char_pos - task.dist_object.location).length
            if dist >= task.dist_max:
                task.handle.volume = 0.0
            else:
                # Linear falloff
                factor = 1.0 - (dist / task.dist_max)
                task.handle.volume = task.original_volume * factor

    # Remove finished
    for i in reversed(to_remove):
        _sound_tasks.pop(i)
```

**Impact**: Minor, but cleaner code path.

---

## Phase 2: Transform Tasks Offload

### 2.1 New Job Type: `TRANSFORM_BATCH`
**Files**:
- `worker/entry.py` (add handler)
- `worker/transforms.py` (new file)
- `reactions/exp_transforms.py` (refactor)

**Current Flow (Main Thread)**:
```
update_transform_tasks() called at 30Hz
  for each task:
    lerp location (Vector.lerp)
    slerp rotation (Quaternion.slerp)
    lerp scale (Vector.lerp)
    write to obj.location, obj.rotation_euler, obj.scale
```

**New Flow (Worker Offload)**:
```
Main Thread:
  1. Collect active transforms into batch
  2. Submit TRANSFORM_BATCH job
  3. Poll results
  4. Apply results (thin write loop)

Worker:
  1. Receive batch of transform data
  2. Compute all lerps/slerps (vectorized numpy)
  3. Return computed transforms
```

**Worker Handler** (`worker/transforms.py`):
```python
import numpy as np
from .math import slerp_quaternion, quaternion_to_euler, euler_to_quaternion, quaternion_multiply

def handle_transform_batch(job_data: dict) -> dict:
    """
    Batch compute transform interpolations.

    Input:
        transforms: [
            {
                "obj_id": int,
                "t": float,  # interpolation factor
                "start_loc": (x, y, z),
                "end_loc": (x, y, z),
                "start_rot_q": (w, x, y, z),
                "end_rot_q": (w, x, y, z),
                "start_scl": (x, y, z),
                "end_scl": (x, y, z),
                "rot_mode": "slerp" | "euler" | "local_delta",
                "delta_euler": (x, y, z) | None,
            }
        ]

    Output:
        results: [
            {
                "obj_id": int,
                "loc": (x, y, z),
                "rot_euler": (x, y, z),
                "scl": (x, y, z),
                "finished": bool,
            }
        ]
    """
    transforms = job_data.get("transforms", [])
    results = []

    for tf in transforms:
        t = tf["t"]
        obj_id = tf["obj_id"]
        finished = t >= 1.0

        # Location lerp
        loc = tuple(
            tf["start_loc"][i] + (tf["end_loc"][i] - tf["start_loc"][i]) * t
            for i in range(3)
        )

        # Rotation
        rot_mode = tf.get("rot_mode", "slerp")
        if rot_mode == "slerp":
            q = slerp_quaternion(tf["start_rot_q"], tf["end_rot_q"], t)
            rot_euler = quaternion_to_euler(q)
        elif rot_mode == "local_delta":
            # q(t) = q_start @ quat(euler(t * delta))
            delta = tf.get("delta_euler", (0, 0, 0))
            delta_t = tuple(d * t for d in delta)
            q_delta = euler_to_quaternion(delta_t)
            q = quaternion_multiply(tf["start_rot_q"], q_delta)
            rot_euler = quaternion_to_euler(q)
        else:
            # Per-channel euler lerp (start_rot_q is actually euler here)
            rot_euler = tuple(
                tf["start_rot_q"][i] + (tf["end_rot_q"][i] - tf["start_rot_q"][i]) * t
                for i in range(3)
            )

        # Scale lerp
        scl = tuple(
            tf["start_scl"][i] + (tf["end_scl"][i] - tf["start_scl"][i]) * t
            for i in range(3)
        )

        results.append({
            "obj_id": obj_id,
            "loc": loc,
            "rot_euler": rot_euler,
            "scl": scl,
            "finished": finished,
        })

    return {"results": results, "count": len(results)}
```

**Main Thread Refactor** (`reactions/exp_transforms.py`):
```python
from mathutils import Vector, Euler

# Store obj references by id for result application
_transform_obj_lookup: dict[int, bpy.types.Object] = {}

def submit_transform_batch(engine) -> int | None:
    """Submit active transforms to worker."""
    if not _active_transform_tasks:
        return None

    now = get_game_time()
    batch = []
    _transform_obj_lookup.clear()

    for task in _active_transform_tasks:
        if not task.obj:
            continue

        obj_id = id(task.obj)
        _transform_obj_lookup[obj_id] = task.obj

        t = (now - task.start_time) / task.duration if task.duration > 0 else 1.0
        t = max(0.0, min(1.0, t))

        batch.append({
            "obj_id": obj_id,
            "t": t,
            "start_loc": tuple(task.start_loc),
            "end_loc": tuple(task.end_loc),
            "start_rot_q": tuple(task.start_rot_q),
            "end_rot_q": tuple(task.end_rot_q),
            "start_scl": tuple(task.start_scl),
            "end_scl": tuple(task.end_scl),
            "rot_mode": task.rot_interp,
            "delta_euler": tuple(task.delta_euler) if task.delta_euler else None,
        })

    if not batch:
        return None

    return engine.submit_job("TRANSFORM_BATCH", {"transforms": batch})

def apply_transform_results(result):
    """Apply computed transforms from worker."""
    if not result.success:
        return 0

    results = result.result.get("results", [])
    applied = 0

    for r in results:
        obj = _transform_obj_lookup.get(r["obj_id"])
        if not obj:
            continue

        # Thin write - no computation here
        obj.location = Vector(r["loc"])
        obj.rotation_euler = Euler(r["rot_euler"], 'XYZ')
        obj.scale = Vector(r["scl"])
        applied += 1

    # Cleanup finished tasks
    finished_ids = {r["obj_id"] for r in results if r.get("finished")}
    for i in reversed(range(len(_active_transform_tasks))):
        task = _active_transform_tasks[i]
        if id(task.obj) in finished_ids:
            # Handle reset scheduling if needed
            if task.reset_enabled:
                # Schedule revert task...
                pass
            _active_transform_tasks.pop(i)

    return applied
```

---

## Phase 3: Tracking Movement Offload

### 3.1 New Job Type: `TRACKING_BATCH`
**Files**:
- `worker/entry.py` (add handler)
- `worker/tracking.py` (new file)
- `reactions/exp_tracking.py` (refactor)

**Current Flow (Main Thread)**:
```
update_tracking_tasks() at 30Hz
  for each track:
    if character: inject autopilot keys (stays on main)
    if object: _move_object_simple()
      raycast sweep/slide (uses static BVH) <-- OFFLOAD THIS
      gravity snap
      write to mover.location
```

**Problem**: `_move_object_simple()` does raycasts on main thread.

**New Flow (Worker Offload)**:
```
Main Thread:
  1. Separate character tracks (stay main) from object tracks
  2. Submit TRACKING_BATCH job with object positions and goals
  3. Poll results
  4. Apply new positions

Worker:
  1. Receive track data
  2. Compute sweep/slide using cached grid (same as KCC)
  3. Return new positions
```

**Worker Handler** (`worker/tracking.py`):
```python
from .raycast import unified_raycast
from .math import normalize_vec3

def handle_tracking_batch(job_data: dict, grid, dynamic_meshes, dynamic_transforms) -> dict:
    """
    Batch compute tracking movement with collision.

    Input:
        tracks: [
            {
                "obj_id": int,
                "current_pos": (x, y, z),
                "goal_pos": (x, y, z),
                "speed": float,
                "dt": float,
                "radius": float,
                "height": float,
                "use_gravity": bool,
                "respect_proxy": bool,
            }
        ]

    Output:
        results: [
            {
                "obj_id": int,
                "new_pos": (x, y, z),
                "arrived": bool,
            }
        ]
    """
    tracks = job_data.get("tracks", [])
    results = []

    for track in tracks:
        pos = list(track["current_pos"])
        goal = track["goal_pos"]
        speed = track["speed"]
        dt = track["dt"]
        radius = track.get("radius", 0.22)
        height = track.get("height", 1.8)
        arrive_radius = track.get("arrive_radius", 0.3)

        # Direction to goal (XY only)
        dx = goal[0] - pos[0]
        dy = goal[1] - pos[1]
        dist = (dx*dx + dy*dy) ** 0.5

        # Check arrival
        if dist <= arrive_radius:
            results.append({
                "obj_id": track["obj_id"],
                "new_pos": tuple(pos),
                "arrived": True,
            })
            continue

        # Normalize and compute step
        step_len = min(speed * dt, dist)
        fwd = (dx / dist, dy / dist, 0.0)

        # Sweep forward with collision
        if track.get("respect_proxy", True) and grid:
            new_pos = sweep_with_collision(
                pos, fwd, step_len, radius, height,
                grid, dynamic_meshes, dynamic_transforms
            )
        else:
            # No collision - simple move
            new_pos = [
                pos[0] + fwd[0] * step_len,
                pos[1] + fwd[1] * step_len,
                pos[2],
            ]

        # Gravity snap
        if track.get("use_gravity", True) and grid:
            new_pos = snap_to_ground(new_pos, grid, dynamic_meshes, dynamic_transforms)

        results.append({
            "obj_id": track["obj_id"],
            "new_pos": tuple(new_pos),
            "arrived": False,
        })

    return {"results": results, "count": len(results)}


def sweep_with_collision(pos, fwd, step_len, radius, height, grid, dyn_meshes, dyn_transforms):
    """
    Sweep capsule forward with wall collision and slide.
    Reuses KCC-style logic from physics.py.
    """
    # Similar to KCC horizontal movement
    # 3 vertical rays (feet/mid/head) to detect wall
    ray_len = step_len + radius
    best_d = None
    best_n = None

    for z in (radius, max(radius, 0.5 * height), height - radius):
        origin = (pos[0], pos[1], pos[2] + z)
        hit = unified_raycast(origin, fwd, ray_len, grid, dyn_meshes, dyn_transforms)
        if hit and hit.get("hit"):
            d = hit["distance"]
            if best_d is None or d < best_d:
                best_d = d
                best_n = hit.get("normal")

    # Advance until contact
    if best_d is None:
        return [pos[0] + fwd[0] * step_len, pos[1] + fwd[1] * step_len, pos[2]]

    allow = max(0.0, best_d - radius)
    moved = min(step_len, allow)
    new_pos = [pos[0] + fwd[0] * moved, pos[1] + fwd[1] * moved, pos[2]]

    # Simple slide if remainder
    remain = step_len - moved
    if remain > 0.1 * radius and best_n:
        # Slide direction
        slide = (fwd[0] - best_n[0] * (fwd[0]*best_n[0] + fwd[1]*best_n[1]),
                 fwd[1] - best_n[1] * (fwd[0]*best_n[0] + fwd[1]*best_n[1]),
                 0.0)
        slide_len = (slide[0]**2 + slide[1]**2) ** 0.5
        if slide_len > 1e-9:
            slide = (slide[0]/slide_len, slide[1]/slide_len, 0.0)
            new_pos[0] += slide[0] * remain
            new_pos[1] += slide[1] * remain

    return new_pos


def snap_to_ground(pos, grid, dyn_meshes, dyn_transforms):
    """Snap position to ground using downward raycast."""
    origin = (pos[0], pos[1], pos[2] + 0.1)
    direction = (0.0, 0.0, -1.0)
    max_dist = 0.6

    hit = unified_raycast(origin, direction, max_dist, grid, dyn_meshes, dyn_transforms)
    if hit and hit.get("hit"):
        return [pos[0], pos[1], hit["position"][2]]
    return pos
```

---

## Phase 4: Property Task Optimization

### 4.1 Batch Lerp Calculation
**File**: `reactions/exp_reactions.py`

Property tasks use `eval()`/`exec()` for dynamic bpy access, so the assignment MUST stay on main thread. However, we can batch the lerp calculations.

**Current**: Each task computes lerp individually.
**Optimization**: Pre-compute all lerps in one pass, then apply.

```python
def update_property_tasks():
    if not _active_property_tasks:
        return

    now = get_game_time()

    # Phase 1: Compute all alphas and lerps
    updates = []
    to_remove = []

    for i, task in enumerate(_active_property_tasks):
        if task.finished:
            to_remove.append(i)
            continue

        if task.duration <= 0:
            updates.append((task.path_str, task.new_val, True, task))
            to_remove.append(i)
            continue

        alpha = (now - task.start_time) / task.duration
        alpha = max(0.0, min(1.0, alpha))

        if alpha >= 1.0:
            updates.append((task.path_str, task.new_val, True, task))
            to_remove.append(i)
        else:
            val = _lerp_value(task.old_val, task.new_val, alpha)
            updates.append((task.path_str, val, False, None))

    # Phase 2: Apply all updates (bpy writes)
    for path_str, val, finished, task in updates:
        _assign_safely(path_str, val)

    # Phase 3: Cleanup and schedule reverts
    for i in reversed(to_remove):
        task = _active_property_tasks[i]
        if task.reset_enabled:
            revert_start = task.end_time + task.reset_delay
            pt2 = PropertyTask(
                path_str=task.path_str,
                old_val=task.new_val,
                new_val=task.old_val,
                start_time=revert_start,
                duration=task.duration,
                reset_enabled=False,
                reset_delay=0.0
            )
            _active_property_tasks.append(pt2)
        _active_property_tasks.pop(i)
```

**Impact**: Cleaner code, slightly better cache locality.

---

## Implementation Priority

| Phase | Item | Impact | Effort | Priority |
|-------|------|--------|--------|----------|
| 1.1 | World state filtering | HIGH | LOW | **DONE** |
| 1.2 | AABB caching | HIGH | LOW | **DONE** |
| 1.3 | Sound distance batching | LOW | LOW | P2 |
| 2.1 | Transform batch offload | MEDIUM | MEDIUM | **P1** |
| 3.1 | Tracking batch offload | MEDIUM | MEDIUM | **P1** |
| 4.1 | Property task batching | LOW | LOW | P2 |

---

## Testing Checklist

### Phase 1
- [ ] World state filtering: Verify trackers still fire correctly
- [ ] World state filtering: Verify character position always included
- [ ] AABB cache: Verify collision detection still works for moving objects
- [ ] AABB cache: Verify cache clears on game end
- [ ] Sound distance: Verify attenuation sounds correct

### Phase 2
- [ ] Transform batch: Verify smooth interpolation matches original
- [ ] Transform batch: Verify rotation modes (slerp, euler, local_delta)
- [ ] Transform batch: Verify multi-turn rotations work (720 degrees)
- [ ] Transform batch: Verify reset scheduling still works

### Phase 3
- [ ] Tracking batch: Verify objects reach goals
- [ ] Tracking batch: Verify collision avoidance works (walls)
- [ ] Tracking batch: Verify gravity snap works
- [ ] Tracking batch: Verify character autopilot still works (stays on main)

---

## Metrics to Track

1. **Frame time**: `time.perf_counter()` around game loop sections
2. **Worker utilization**: Jobs per frame, queue depth
3. **Main thread compute**: Time spent in `update_*` functions
4. **Latency**: Time from job submit to result apply

### Debug Logging
```python
from ..developer.dev_logger import log_game

log_game("OFFLOAD", f"TRANSFORM_BATCH submitted={count} calc_time={us:.0f}us")
log_game("OFFLOAD", f"TRACKING_BATCH submitted={count} calc_time={us:.0f}us")
log_game("OFFLOAD", f"WORLD_STATE objects={len(positions)} (filtered from {total})")
```

---

## Notes

- **Worker-safe code**: No `bpy` imports in worker modules
- **Serialization**: All data sent to worker must be tuples/dicts/primitives (no bpy objects)
- **Object references**: Use `id(obj)` as key, maintain lookup dict on main thread
- **Same-frame sync**: Use `poll_with_timeout()` pattern for latency-sensitive systems
- **Cache invalidation**: Clear caches on game end/reset

---

## Future Considerations

### Potential Phase 5: Audio Distance Offload
If many sound sources active, could batch distance calculations to worker.

### Potential Phase 6: IK Offload
Runtime IK currently on main thread. Could offload bone chain solving to worker.

### Potential Phase 7: GPU Compute
For very large scenes, consider GPU compute for:
- Batch AABB calculations
- Batch distance calculations
- Batch transform lerps

---

*Last updated: 2024-12-23*
