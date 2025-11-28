# Engine Offload Plan

**Goal**: Move all performance-critical computations from the main modal thread to multiprocessing engine workers without changing game feel.

---

## Core Principles

1. **Preserve Game Feel** - The game works and feels great. We optimize performance only.
2. **No Fallback Code** - Once offloaded, remove main thread implementation. No dual paths.
3. **Developer Tools Integration** - Every offload gets a debug toggle in Developer Tools panel.
4. **Incremental Validation** - One system at a time, fully tested before moving on.
5. **Performance Wins Only** - Only offload if it's a measurable performance improvement.

---

## Current State

### Engine Infrastructure
- 4 worker processes (true multiprocessing, bypasses GIL)
- ~1,300 jobs/sec sustained throughput
- Non-blocking submit/poll (zero main thread stalls)
- Worker grid caching for large data (send once, reuse forever)
- Predictive job submission (compensates for 1-frame latency)

### Completed Offloads

| System | Job Type | Status | Performance |
|--------|----------|--------|-------------|
| Performance Culling | `CULL_BATCH` | Production | 1000+ objects/frame |
| Dynamic Mesh Activation | `DYNAMIC_MESH_ACTIVATION` | Production | 13-17µs/batch |
| Interaction Proximity | `INTERACTION_CHECK_BATCH` | Production | 4-34µs/batch |
| KCC Input Vector | `KCC_INPUT_VECTOR` | Production | ~5µs/calc |
| KCC Slope/Platform Math | `KCC_SLOPE_PLATFORM_MATH` | Production | ~8µs/calc |
| KCC Ground Raycast (Static) | `KCC_RAYCAST_CACHED` | Production | ~50µs (was 20ms brute) |

### Developer Tools Debug Toggles (Current)
```
Developer Tools Panel
├── Enable All Debug Output (master)
├── Engine (Multiprocessing)
├── Performance Culling
├── Dynamic Mesh Offload
├── KCC Physics Offload
├── Raycast Offload
├── Physics & Character
├── Interactions & Reactions
├── Audio System
└── Animations & NLA
```

---

## Phase 1: Complete KCC Offload

### 1.1 KCC Ground Raycast for Dynamic Platforms
**Status**: NOT STARTED

**Current**: Dynamic platforms use main thread BVH rebuild every frame.

**Strategy**:
- Cache base mesh triangles in local space (once at startup)
- Send only transform matrices per frame (~64 bytes/platform)
- Worker applies transform to cached triangles, then raycasts

```
Job Type: CACHE_DYNAMIC_MESHES
Input: {mesh_id: local_triangles[]} (once at startup)
Output: Confirmation

Job Type: DYNAMIC_TRANSFORMS
Input: {mesh_id: matrix4x4}[] (every frame)
Output: Confirmation

Job Type: KCC_RAYCAST_DYNAMIC
Input: ray_origin, ray_dir, max_dist
Output: hit, location, normal, distance
```

**Performance Win**: Eliminates per-frame BVH rebuild on main thread.

**Debug Toggle**: `dev_debug_raycast_offload` (existing)

---

### 1.2 Full KCC Step Offload
**Status**: NOT STARTED

**Current**: `step()` runs on main thread with these remaining operations:
- Horizontal acceleration (`_accelerate()`)
- Dynamic contact influence
- Gravity application
- Jump logic
- Ground snapping
- Platform carry (linear + rotational)
- Position/rotation writes

**Strategy**:
Phase A - Offload remaining pure math:
- Acceleration calculations
- Gravity integration
- Jump velocity application
- Platform carry math

Phase B - Full step offload:
- Worker runs entire physics step
- Returns final position, velocity, rotation delta
- Main thread applies to bpy

```
Job Type: KCC_FULL_STEP
Input: {
    pos, vel, rot,                    # Current state
    wish_dir, is_running, dt,         # Input
    ground_hit, ground_normal,        # Raycast result (from cache)
    platform_velocities,              # Dynamic platform data
    cfg                               # Physics config
}
Output: {
    new_pos, new_vel, new_rot,        # Final state
    on_ground, ground_obj_id          # Ground state
}
```

**Performance Win**: Entire physics loop off main thread.

**Debug Toggle**: `dev_debug_kcc_offload` (existing)

---

## Phase 2: View/Camera Offload

### 2.1 Camera Occlusion Raycasts
**Status**: NOT STARTED

**Current** (`exp_view.py`):
- `_multi_ray_min_hit()` - Center ray against static + dynamic BVH
- `_los_blocked()` - Line-of-sight check
- `_binary_search_clear_los()` - Binary search for clear position
- `_camera_sphere_pushout_any()` - Nearest point queries

All use main thread BVH raycasts.

**Strategy**:
- Use same cached grid as KCC for static geometry
- Use cached dynamic mesh transforms for dynamic geometry
- Submit camera ray job at start of frame
- Apply result from previous frame (1-frame latency acceptable for camera)

```
Job Type: CAMERA_OCCLUSION
Input: {
    anchor, direction, desired_dist,  # Camera params
    r_cam,                            # Camera thickness
    dynamic_transforms                # Current frame transforms
}
Output: {
    allowed_dist,                     # Final distance after all checks
    hit_token                         # What was hit (for latching)
}
```

**Performance Win**: 5+ raycasts per frame moved off main thread.

**Debug Toggle**: `dev_debug_view_offload` (NEW)

---

### 2.2 Camera Math Offload
**Status**: NOT STARTED

**Current**:
- Direction calculation (pitch/yaw → vector)
- Anchor calculation (capsule top)
- Smoothing filter calculations
- Latch filter logic

**Strategy**:
- These are lightweight calculations
- May not be worth the serialization overhead
- Evaluate after 2.1 is complete

**Performance Win**: Likely minimal. Evaluate.

---

## Phase 3: Reaction System Offload

### 3.1 Transform Interpolation
**Status**: NOT STARTED

**Current**: Per-frame lerp for position/rotation/scale reactions.

**Strategy**:
```
Job Type: INTERPOLATION_BATCH
Input: task_definitions[] (start, end, progress, ease_type)
Output: List[(task_id, interpolated_value)]
```

**Performance Win**: Scales to hundreds of simultaneous animations.

**Debug Toggle**: `dev_debug_interpolation_offload` (NEW)

---

### 3.2 Projectile Physics
**Status**: NOT STARTED

**Current**: Per-step velocity integration + collision detection.

**Strategy**:
- Use cached static grid for collision
- Batch all projectiles in single job

```
Job Type: PROJECTILE_BATCH
Input: projectile_states[], dt
Output: updated_states[], hit_events[]
```

**Performance Win**: Unlimited projectiles without performance cost.

**Debug Toggle**: `dev_debug_projectile_offload` (NEW)

---

## Phase 4: Future Systems

### 4.1 AI Pathfinding
When AI is implemented, pathfinding runs entirely in workers.

### 4.2 Physics Prediction
Multi-step lookahead for AI decision making.

### 4.3 Particle Physics
Thousands of particles simulated in workers.

---

## Developer Tools Architecture

### Current Structure (Good)
```
dev_properties.py     - Property definitions
dev_panel.py          - UI panel layout
dev_debug_gate.py     - Frequency-gated printing
```

### Scalability Assessment

**Strengths**:
- Per-category toggles with Hz control
- Master toggle for enable all
- Consistent naming pattern (`dev_debug_*`)
- Frequency gating prevents console spam

**Improvements Needed**:

1. **Grouping** - As offloads grow, group related toggles:
   ```
   Physics Offload
   ├── KCC Physics
   ├── Raycast (Static)
   ├── Raycast (Dynamic)
   └── Ground Detection

   View Offload
   ├── Camera Occlusion
   └── Camera Math

   Reaction Offload
   ├── Interpolation
   └── Projectiles
   ```

2. **Performance Metrics** - Add optional timing display:
   - Jobs submitted per second
   - Average job latency
   - Cache hit rate

3. **Status Indicators** - Show offload health:
   - Green = Working normally
   - Yellow = High latency
   - Red = Errors

### Adding New Offload Debug Toggle

1. Add property in `dev_properties.py`:
```python
bpy.types.Scene.dev_debug_NEW_offload = bpy.props.BoolProperty(
    name="New Offload",
    description="Print NEW offload debug...",
    default=False
)
bpy.types.Scene.dev_debug_NEW_offload_hz = bpy.props.IntProperty(
    name="New Offload Hz",
    default=5, min=1, max=30
)
```

2. Add to master toggle update in `dev_properties.py`:
```python
scene.dev_debug_NEW_offload = enabled
```

3. Add to panel in `dev_panel.py`:
```python
row = col.row(align=True)
row.prop(scene, "dev_debug_NEW_offload", text="New Offload")
if scene.dev_debug_NEW_offload:
    row.prop(scene, "dev_debug_NEW_offload_hz", text="Hz")
```

4. Add to unregister in `dev_properties.py`

5. Use in code:
```python
from ..developer.dev_debug_gate import should_print_debug
if should_print_debug("NEW_offload"):
    print("[New Offload] Debug message")
```

---

## Implementation Priority

### Immediate (High Value)
1. **Phase 1.1** - Dynamic platform raycasts (completes ground detection)
2. **Phase 2.1** - Camera occlusion raycasts (5+ rays/frame)

### Near-term (Medium Value)
3. **Phase 1.2** - Full KCC step offload (entire physics loop)
4. **Phase 3.1** - Transform interpolation (reaction scaling)

### Future (When Needed)
5. **Phase 3.2** - Projectile physics
6. **Phase 4.x** - AI, prediction, particles

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Main thread physics time | ~2ms | <0.5ms |
| Main thread camera time | ~0.5ms | <0.1ms |
| Max objects without lag | ~500 | 2000+ |
| Max projectiles | ~10 | 100+ |

---

## File Reference

### Engine Core
- `Exp_Game/engine/engine_core.py` - Main engine manager
- `Exp_Game/engine/engine_worker_entry.py` - Worker job handlers
- `Exp_Game/engine/engine_types.py` - Job/Result types

### Physics
- `Exp_Game/physics/exp_kcc.py` - Character controller
- `Exp_Game/physics/exp_view.py` - Camera system
- `Exp_Game/physics/exp_geometry.py` - Grid building

### Developer Tools
- `Exp_Game/developer/dev_properties.py` - Debug toggles
- `Exp_Game/developer/dev_panel.py` - UI panel
- `Exp_Game/developer/dev_debug_gate.py` - Frequency gating

### Game Loop
- `Exp_Game/modal/exp_loop.py` - Result polling and application
- `Exp_Game/modal/exp_modal.py` - Engine lifecycle

---

## Notes

- All job types defined in `engine_worker_entry.py`
- All result handlers in `exp_loop.py` `_poll_and_apply_engine_results()`
- Grid caching: Send large data once, workers store in module-level variables
- Predictive submission: Calculate next-frame state, submit job, use result next frame
- 1-frame latency is acceptable for non-input-critical systems (camera, culling)

---

**Last Updated**: 2025-11-28
**Status**: Phase 1.1 Next (Dynamic Platform Raycasts)
