# Exp_Game Engine Offload Plan

**Project Goal**: Move computations and logic from the main modal thread to /engine workers without changing game feel, behavior, or interactions.

---

## Core Principles

### 1. **Preserve Game Feel (NON-NEGOTIABLE)**
- The game works amazing and feels great
- We are ONLY optimizing performance, NOT changing behavior
- Zero tolerance for gameplay regressions
- All offloaded systems must produce identical results to main thread versions
- Frame timing, input response, and physics behavior must remain unchanged

### 2. **No Fallback Code (Clean Architecture)**
- Once logic moves to workers, remove main thread implementation completely
- No dual code paths (worker + main thread)
- No runtime switches or "if worker_available" branches
- Exception: Development-time feature toggles in Developer Tools panel only

### 3. **Organized Structure**
- Keep worker job types clearly defined and documented
- Maintain consistent patterns across all offloaded systems
- Follow existing architecture (Snapshot → Submit → Poll → Apply)
- Group related offloads together in code organization

### 4. **Verification via Developer System**
- Every new offload MUST have dedicated debug toggle in Developer Tools N-Panel
- All worker output controlled via scene.dev_debug_* properties
- Console verification required for production deployment
- Test stress scenarios with debug enabled to confirm worker operation

### 5. **Incremental & Validated**
- One system at a time
- Fully test each offload before moving to next
- Verify performance gains are real (not theoretical)
- Document actual vs predicted performance improvements

---

## Current State (Baseline)

### Already Offloaded (PRODUCTION)
- **Performance Culling** (CULL_BATCH)
  - 1000+ objects per frame
  - Distance calculations in workers
  - Debug toggle: `scene.dev_debug_engine`
  - Status: ✅ Verified working, zero issues

### Main Thread Bottlenecks (Measured)
1. **Physics Raycasting** - 6-18 raycasts per frame (BVH queries)
2. **Camera Occlusion** - 5+ raycasts per frame (BVH queries)
3. **Dynamic Mesh Updates** - Distance checks + velocity calculations
4. **Interaction Checks** - Distance + AABB per frame
5. **Custom Tasks** - Projectiles, transforms, properties (30Hz each)

### Worker Infrastructure
- 4 worker processes (multiprocessing)
- 1,284 jobs/sec sustained throughput
- 1-5ms per job processing time
- Non-blocking submit/poll (zero main thread stalls)
- NO bpy access in workers (pure Python only)

---

## Offload Phases

---

## Phase 1: Low-Hanging Fruit
**Goal**: Prove offload patterns with easy wins, build confidence

### 1.1 Dynamic Mesh Proximity Checks
**Current Behavior**:
- Per-frame iteration calculating `distance(mesh.location, player.location)`
- Activates/deactivates dynamic meshes based on proximity threshold
- Small count (~10-50 meshes), but tight loop

**Offload Strategy**:
```
Job Type: DYNAMIC_MESH_ACTIVATION
Input: mesh_positions[], player_position, activation_radius
Output: List[(mesh_index, should_activate: bool)]
Worker: Pure distance² calculations (no sqrt)
Main Thread: Apply activation state to meshes
```

**Performance Win**: ⭐⭐ (LOW-MEDIUM)
- Frees up per-frame iteration
- Enables scaling to hundreds of dynamic meshes
- Expected: +2-5% main thread time freed

**Difficulty**: ⭐ (EASY)
- Copy CULL_BATCH pattern exactly
- Already have serialization pattern from performance culling

**Developer Tools Integration**:
```python
# New property in exp_dev_tools.py DevToolsProps
dev_debug_dynamic_offload: BoolProperty(
    name="Dynamic Mesh Offload",
    description="Log dynamic mesh activation job submissions and results",
    default=False
)
```

**Verification Steps**:
1. Enable debug toggle in N-Panel → Create → Developer Tools
2. Run game, observe console output showing:
   - Job submissions with mesh count
   - Worker results with activation decisions
   - Confirm matches previous main thread behavior
3. Disable toggle, confirm silent operation
4. Stress test: Add 100+ dynamic meshes, confirm no lag

**File Changes**:
- `exp_dynamic.py` - Remove main thread distance calculations, add worker job submission/polling
- `exp_dev_tools.py` - Add debug property
- `engine/engine_worker.py` - Add DYNAMIC_MESH_ACTIVATION handler
- `exp_loop.py` - Poll and apply results in game loop

**Success Criteria**:
- ✅ Dynamic meshes activate/deactivate at same distances as before
- ✅ Zero main thread distance calculations for proximity checks
- ✅ Console debug output confirms worker execution
- ✅ Performance: +2-5% main thread time freed (measure with profiler)

---

### 1.2 Interaction Proximity & AABB Checks
**Current Behavior**:
- Per-frame distance checks: `distance(interaction.location, player.location) < trigger_radius`
- AABB checks: `is_point_in_box(player.location, interaction.bbox_min, interaction.bbox_max)`
- Small-medium count (~5-50 interactions per scene)
- Returns list of triggered interaction indices

**Offload Strategy**:
```
Job Type: INTERACTION_CHECK_BATCH
Input: interaction_data[] (position, radius, aabb, type), player_position
Output: List[interaction_indices] that are triggered
Worker: Distance² + point-in-box checks (pure math)
Main Thread: Execute reactions for triggered interactions only
```

**Performance Win**: ⭐⭐ (LOW-MEDIUM)
- Tight loop freed from main thread
- Enables complex interaction systems (hundreds of triggers)
- Expected: +3-7% main thread time freed

**Difficulty**: ⭐ (EASY)
- Pure math: `distance² < radius²` and AABB checks
- No bpy dependency for trigger detection
- Reactions still execute on main thread (bpy-bound)

**Developer Tools Integration**:
```python
# New property in exp_dev_tools.py DevToolsProps
dev_debug_interaction_offload: BoolProperty(
    name="Interaction Offload",
    description="Log interaction check job submissions and triggered interactions",
    default=False
)
```

**Verification Steps**:
1. Enable debug toggle in N-Panel
2. Walk through scene triggering various interactions
3. Console confirms:
   - Job submissions with interaction count
   - Worker returns triggered indices
   - Reactions execute correctly (unchanged behavior)
4. Test all interaction types: proximity, collision, timer, key-press
5. Confirm no regressions in interaction feel/timing

**File Changes**:
- `exp_interactions.py` - Remove main thread checks, add worker job submission
- `exp_dev_tools.py` - Add debug property
- `engine/engine_worker.py` - Add INTERACTION_CHECK_BATCH handler
- `exp_loop.py` - Poll and apply results (execute reactions)

**Success Criteria**:
- ✅ All interaction types trigger at same distances/timing as before
- ✅ Zero main thread distance/AABB calculations
- ✅ Console debug confirms worker execution
- ✅ Performance: +3-7% main thread time freed

---

## Phase 2: Medium Complexity
**Goal**: Unlock scalability for moving platforms and interpolation systems

### 2.1 Platform Velocity Calculations
**Current Behavior**:
- Every frame: Calculate position delta from matrix differences
- Angular velocity from quaternion differencing
- Runs for all active dynamic meshes
- Results feed into physics system (platform carry velocity)

**Offload Strategy**:
```
Job Type: PLATFORM_VELOCITY_BATCH
Input: current_matrices[], previous_matrices[], delta_time
Output: List[(linear_velocity: Vector3, angular_velocity: Vector3)]
Worker: Matrix math + quaternion differencing (pure math)
Main Thread: Cache previous matrices, apply velocities to physics
```

**Performance Win**: ⭐⭐⭐ (MEDIUM)
- Runs every frame for all dynamic meshes
- Matrix math is CPU-intensive
- Expected: +5-10% main thread time freed

**Difficulty**: ⭐⭐ (MEDIUM)
- Serialize Matrix 4x4 (16 floats each)
- Quaternion math must be replicated in worker (no mathutils)
- Requires caching strategy for previous frame data

**Developer Tools Integration**:
```python
dev_debug_velocity_offload: BoolProperty(
    name="Platform Velocity Offload",
    description="Log platform velocity calculation jobs and results",
    default=False
)
```

**Verification Steps**:
1. Enable debug toggle
2. Test moving platforms with character standing on them
3. Console confirms:
   - Job submissions with matrix data
   - Worker returns velocities
   - Physics applies platform carry correctly
4. Verify: Character rides platforms identically to before (THIS IS CRITICAL)
5. Test rotating platforms (angular velocity must be exact)

**File Changes**:
- `exp_dynamic.py` - Remove velocity calculations, add worker jobs
- `exp_dev_tools.py` - Add debug property
- `engine/engine_worker.py` - Add PLATFORM_VELOCITY_BATCH handler (implement quaternion math)
- `exp_loop.py` - Poll and apply velocities

**Success Criteria**:
- ✅ Character rides platforms with identical feel/behavior
- ✅ No regressions in platform physics (sliding, snapping, etc.)
- ✅ Console confirms worker velocity calculations
- ✅ Performance: +5-10% main thread time freed

---

### 2.2 Transform & Property Interpolation (Bulk)
**Current Behavior**:
- Per-frame lerp calculations for transform tasks (position/rotation)
- Per-frame lerp for property tasks (float/vector/int/bool)
- Small count typically, but can scale to hundreds
- Main thread calculates new values, then writes to bpy properties

**Offload Strategy**:
```
Job Type: INTERPOLATION_BATCH
Input: task_definitions[] (start, end, current_time, duration, ease_type), current_time
Output: List[(task_id, new_value)]
Worker: Pure lerp/ease calculations (math only)
Main Thread: Write results to bpy property paths
```

**Performance Win**: ⭐⭐ (LOW-MEDIUM)
- Small count typically, but unlocks hundreds of simultaneous animations
- Expected: +2-5% main thread time freed (scales with task count)

**Difficulty**: ⭐⭐ (MEDIUM)
- Serialize task states (start/end values, timing, ease functions)
- Implement easing functions in worker (linear, ease-in/out, etc.)
- Main thread only writes final values

**Developer Tools Integration**:
```python
dev_debug_interpolation_offload: BoolProperty(
    name="Interpolation Offload",
    description="Log interpolation task calculations and results",
    default=False
)
```

**Verification Steps**:
1. Enable debug toggle
2. Trigger transform/property animations (reactions, custom tasks)
3. Console confirms:
   - Job submissions with task data
   - Worker returns interpolated values
   - Main thread applies to properties
4. Verify: Animation timing and smoothness identical to before
5. Test all ease types (linear, ease-in, ease-out, etc.)

**File Changes**:
- `exp_reactions.py` - Remove lerp calculations, add worker jobs
- `exp_dev_tools.py` - Add debug property
- `engine/engine_worker.py` - Add INTERPOLATION_BATCH handler (implement easing)
- `exp_loop.py` - Poll and apply property writes

**Success Criteria**:
- ✅ All animations/interpolations have identical timing/smoothness
- ✅ Zero main thread lerp calculations
- ✅ Console confirms worker execution
- ✅ Performance: +2-5% main thread time freed

---

## Phase 3: Complex Offloading
**Goal**: Unlock advanced gameplay systems (projectiles, predictions)

### 3.1 Projectile Physics Prediction
**Current Behavior**:
- `update_projectile_tasks()` runs per physics step (up to 3x per frame)
- Each projectile: velocity integration, gravity, collision detection
- Can have multiple simultaneous projectiles (arrows, grenades, magic)

**Offload Strategy**:
```
Job Type: PROJECTILE_STEP_BATCH
Input: projectile_states[] (position, velocity, gravity, lifetime), simplified_collision_geometry, delta_time, steps
Output: List[new_projectile_states] (position, velocity, hit_info)
Worker: Full physics loop (integration + simplified collision)
Main Thread: Apply positions, execute hit reactions
```

**Performance Win**: ⭐⭐⭐⭐ (HIGH)
- Multiple projectiles = multiple physics loops per frame
- Unlocks complex projectile gameplay without performance cost
- Expected: +10-15% main thread time freed (scales with projectile count)

**Difficulty**: ⭐⭐⭐ (MEDIUM-HIGH)
- Serialize projectile states
- Implement physics integration in worker
- Simplified collision (spheres/planes) initially (NO BVH raycasting yet)
- OR wait for Phase 4 (BVH serialization)

**Developer Tools Integration**:
```python
dev_debug_projectile_offload: BoolProperty(
    name="Projectile Offload",
    description="Log projectile physics calculations and hit detection",
    default=False
)
```

**Verification Steps**:
1. Enable debug toggle
2. Spawn projectiles (arrows, grenades, etc.)
3. Console confirms:
   - Job submissions with projectile states
   - Worker returns updated positions/hits
   - Main thread applies results
4. Verify: Projectile trajectories identical to before (THIS IS CRITICAL)
5. Test hit detection (must trigger reactions correctly)

**File Changes**:
- `exp_reactions.py` - Remove projectile physics, add worker jobs
- `exp_dev_tools.py` - Add debug property
- `engine/engine_worker.py` - Add PROJECTILE_STEP_BATCH handler
- `exp_loop.py` - Poll and apply projectile updates

**Success Criteria**:
- ✅ Projectiles have identical trajectories/timing
- ✅ Hit detection unchanged (same impact points)
- ✅ Console confirms worker physics execution
- ✅ Performance: +10-15% main thread time freed

**Note**: Initial implementation uses simplified collision. Full accuracy requires Phase 4 (BVH serialization).

---

### 3.2 Physics Prediction / Lookahead (Future AI)
**Current Behavior**:
- Physics only runs for current frame (no prediction)
- AI systems would need to predict jump trajectories, platform motion, etc.

**Offload Strategy**:
```
Job Type: PHYSICS_PREDICT
Input: current_state, input_sequence[], static_geometry, dynamic_geometry, steps
Output: predicted_states[] (position, velocity, grounding per step)
Worker: Replicate KCC physics loop for N steps ahead
Main Thread: Use predictions for AI decision-making (non-critical)
```

**Performance Win**: ⭐⭐⭐⭐⭐ (VERY HIGH)
- Enables AI navigation without main thread cost
- Predictive collision avoidance
- Jump trajectory validation
- Expected: Unlocks AI systems that would otherwise tank performance

**Difficulty**: ⭐⭐⭐⭐ (HIGH)
- Requires BVH serialization OR simplified collision geometry
- Worker must replicate full KCC physics logic
- Speculative results may be invalidated by main thread changes
- Depends on Phase 4 for full accuracy

**Developer Tools Integration**:
```python
dev_debug_physics_predict: BoolProperty(
    name="Physics Prediction",
    description="Log physics prediction jobs and speculative results",
    default=False
)
```

**Verification Steps**:
1. Enable debug toggle
2. Trigger prediction jobs (AI planning, etc.)
3. Console confirms:
   - Job submissions with state + inputs
   - Worker returns predicted trajectory
   - Predictions used for AI decisions
4. Verify: Predictions are accurate (compare to actual physics results)

**File Changes**:
- NEW: `exp_ai.py` or similar (prediction consumer)
- `exp_dev_tools.py` - Add debug property
- `engine/engine_worker.py` - Add PHYSICS_PREDICT handler (replicate KCC)
- `exp_loop.py` - Poll predictions, feed to AI systems

**Success Criteria**:
- ✅ Predictions match actual physics within acceptable tolerance
- ✅ Console confirms worker prediction execution
- ✅ Performance: AI systems run without main thread impact
- ✅ Main thread physics unchanged (predictions are optional)

**Dependencies**: Phase 4 (BVH serialization) for full accuracy. Can implement with simplified collision first.

---

## Phase 4: The Ultimate Unlock
**Goal**: Offload BVH raycasting (THE main thread bottleneck)

### 4.1 Static BVH Raycasting (Serialized BVH)
**Current Behavior**:
- Physics: 6-18 raycasts per frame (sweep, step-up, grounding)
- Camera: 5+ raycasts per frame (occlusion, line-of-sight)
- ALL raycasts use bpy-bound BVH (cannot run in workers)
- THIS IS THE #1 PERFORMANCE BOTTLENECK

**Offload Strategy**:
```
Job Type: BVH_RAYCAST_BATCH
Input: ray_origins[], ray_directions[], max_distance, serialized_bvh
Output: List[(hit_position, hit_normal, hit_distance)] or None per ray
Worker: Pure Python BVH traversal + raycast algorithm
Main Thread: Submit ray queries, apply results
```

**Performance Win**: ⭐⭐⭐⭐⭐ (VERY HIGH)
- Frees 60-80% of main thread physics/camera time
- Expected: +20-40 FPS in complex scenes
- Unlocks full parallelization of physics and camera

**Difficulty**: ⭐⭐⭐⭐⭐ (VERY HIGH)
- Serialize entire BVH data structure (vertices, faces, tree hierarchy)
- Implement BVH traversal in pure Python (no mathutils, no bpy)
- Large memory footprint (entire level geometry in workers)
- One-time serialization cost at game start
- Complex implementation (200-500+ lines of code)

**Developer Tools Integration**:
```python
dev_debug_bvh_offload: BoolProperty(
    name="BVH Raycast Offload",
    description="Log BVH raycast job submissions and hit results",
    default=False
)
```

**Verification Steps**:
1. Enable debug toggle
2. Run game (physics + camera active)
3. Console confirms:
   - BVH serialization at game start (one-time)
   - Ray batch submissions every frame
   - Worker returns hit results
   - Main thread applies (character movement, camera position)
4. Verify: Physics feel IDENTICAL (grounding, sliding, step-up)
5. Verify: Camera behavior IDENTICAL (occlusion, smoothing)
6. Stress test: Complex geometry, many simultaneous raycasts

**File Changes**:
- `exp_kcc.py` - Remove BVH raycasts, add worker job submissions
- `exp_view.py` - Remove BVH raycasts, add worker job submissions
- `exp_dev_tools.py` - Add debug property
- NEW: `engine/bvh_serializer.py` - Serialize BVH at game start
- NEW: `engine/bvh_raycast.py` - Pure Python BVH traversal + raycast
- `engine/engine_worker.py` - Add BVH_RAYCAST_BATCH handler
- `exp_loop.py` - Poll and apply raycast results

**Success Criteria**:
- ✅ Physics feel 100% identical (grounding, collisions, sliding)
- ✅ Camera behavior 100% identical (occlusion, smoothing)
- ✅ Console confirms worker raycast execution
- ✅ Performance: +20-40 FPS (or +60-80% main thread time freed)
- ✅ Zero main thread BVH raycasts

**Alternative Approach**:
Consider third-party libraries if available without bpy dependency:
- `embree-python` (Intel raytracing library)
- `rtree` (spatial indexing)
- Custom C extension with Python bindings

**Note**: This is ALL-OR-NOTHING. No fallback code. Either workers handle all raycasts or keep on main thread.

---

## Phase 5: Future Systems
**Goal**: Build advanced systems enabled by worker offloading

### 5.1 AI Pathfinding (When Implemented)
**Offload Strategy**:
```
Job Type: PATHFIND
Input: start_position, goal_position, navmesh/waypoint_data
Output: path[] (list of waypoints)
Worker: A* or navmesh pathfinding
Main Thread: AI follows path
```

**Performance Win**: ⭐⭐⭐⭐⭐ (VERY HIGH)
**Difficulty**: ⭐⭐⭐ (MEDIUM)

---

### 5.2 Particle System Physics (When Implemented)
**Offload Strategy**:
```
Job Type: PARTICLE_STEP_BATCH
Input: particle_states[] (position, velocity, lifetime, forces)
Output: updated_particle_states[]
Worker: Integrate physics for hundreds/thousands of particles
Main Thread: Apply positions (instancing/geometry nodes)
```

**Performance Win**: ⭐⭐⭐⭐⭐ (VERY HIGH)
**Difficulty**: ⭐⭐ (MEDIUM)

---

## Implementation Order (Recommended)

### Sprint 1: Prove the Pattern (Start Here)
1. Dynamic Mesh Proximity (Difficulty: ⭐, Win: ⭐⭐)
2. Interaction Proximity/AABB (Difficulty: ⭐, Win: ⭐⭐)

**Timeline**: 1-3 days
**Goal**: Prove batch distance/AABB patterns work. Build confidence.

---

### Sprint 2: Unlock Scalability
3. Platform Velocity Calculations (Difficulty: ⭐⭐, Win: ⭐⭐⭐)
4. Transform/Property Interpolation (Difficulty: ⭐⭐, Win: ⭐⭐)

**Timeline**: 3-7 days
**Goal**: Enable hundreds of moving platforms and simultaneous animations.

---

### Sprint 3: Projectile Freedom
5. Projectile Physics Prediction (Difficulty: ⭐⭐⭐, Win: ⭐⭐⭐⭐)

**Timeline**: 5-10 days
**Goal**: Unlock complex projectile gameplay. Use simplified collision initially.

---

### Sprint 4: Physics Lookahead
6. Physics Prediction/Lookahead (Difficulty: ⭐⭐⭐⭐, Win: ⭐⭐⭐⭐⭐)

**Timeline**: 7-14 days
**Goal**: Enable AI and predictive systems. Depends on Sprint 5 for full accuracy.

---

### Sprint 5: The Big One
7. Static BVH Raycasting (Difficulty: ⭐⭐⭐⭐⭐, Win: ⭐⭐⭐⭐⭐)

**Timeline**: 14-30 days
**Goal**: Offload THE bottleneck. This is the ultimate performance unlock.

---

## Expected Performance Gains (Cumulative)

| After Sprint | Main Thread Freed | FPS Impact | Systems Unlocked |
|--------------|-------------------|------------|------------------|
| Sprint 1 | +5-10% | +0-2 FPS | Large interaction systems |
| Sprint 2 | +15-20% | +2-5 FPS | Hundreds of moving platforms |
| Sprint 3 | +25-35% | +5-10 FPS | Complex projectile gameplay |
| Sprint 4 | +30-40% | +8-15 FPS | AI navigation, predictions |
| Sprint 5 | +60-80% | +20-40 FPS | Fully parallelized physics/camera |

**Note**: Percentages are predicted based on current profiling. Actual gains will be measured and documented.

---

## Critical Rules

### DO:
✅ Implement worker version fully before removing main thread code
✅ Test worker version thoroughly with stress tests
✅ Add debug toggle for EVERY new offload in Developer Tools panel
✅ Verify console output confirms worker execution
✅ Measure actual performance gains (profiler/FPS counter)
✅ Maintain identical game feel and behavior
✅ Keep code organized and follow existing patterns
✅ Document actual vs predicted results

### DON'T:
❌ Keep dual code paths (main thread + worker)
❌ Add runtime switches between worker/main thread modes
❌ Use "if worker_available" branches (except dev tools toggles)
❌ Change game feel, timing, or behavior
❌ Skip verification steps
❌ Move to next phase before current is complete
❌ Add features or "improvements" beyond offloading

### Exception:
⚠️ During development: Use feature toggles in Developer Tools panel ONLY. Remove before production.

---

## Developer Tools Panel Structure

```python
# exp_dev_tools.py - DevToolsProps additions

# Existing
dev_debug_engine: BoolProperty(name="Engine (Multiprocessing)", ...)

# Phase 1
dev_debug_dynamic_offload: BoolProperty(name="Dynamic Mesh Offload", ...)
dev_debug_interaction_offload: BoolProperty(name="Interaction Offload", ...)

# Phase 2
dev_debug_velocity_offload: BoolProperty(name="Platform Velocity Offload", ...)
dev_debug_interpolation_offload: BoolProperty(name="Interpolation Offload", ...)

# Phase 3
dev_debug_projectile_offload: BoolProperty(name="Projectile Offload", ...)
dev_debug_physics_predict: BoolProperty(name="Physics Prediction", ...)

# Phase 4
dev_debug_bvh_offload: BoolProperty(name="BVH Raycast Offload", ...)
```

**Panel Layout** (N-Panel → Create → Developer Tools):
```
Developer Tools
├── Engine (Multiprocessing)          [existing]
├── Performance Culling Offload       [existing, rename from generic engine]
├── Dynamic Mesh Offload             [Phase 1.1]
├── Interaction Offload              [Phase 1.2]
├── Platform Velocity Offload        [Phase 2.1]
├── Interpolation Offload            [Phase 2.2]
├── Projectile Offload               [Phase 3.1]
├── Physics Prediction               [Phase 3.2]
└── BVH Raycast Offload              [Phase 4.1]
```

All debug properties default to `False` (silent operation in production).

---

## Verification Checklist (Per Offload)

Before marking any offload as "complete", verify:

1. **Functionality**:
   - [ ] Game behavior identical to before offload
   - [ ] No regressions in feel, timing, or interactions
   - [ ] Edge cases tested (high object counts, complex scenarios)

2. **Performance**:
   - [ ] Main thread time reduction measured (profiler)
   - [ ] FPS gains documented (if applicable)
   - [ ] Worker throughput confirmed (jobs/sec)

3. **Debug System**:
   - [ ] Debug toggle added to Developer Tools panel
   - [ ] Console output shows job submissions
   - [ ] Console output shows worker results
   - [ ] Toggle off = silent operation

4. **Code Quality**:
   - [ ] Main thread implementation removed (no fallback)
   - [ ] Follows existing patterns (Snapshot → Submit → Poll → Apply)
   - [ ] Code organized and documented
   - [ ] No dead code or commented-out old implementation

5. **Stress Testing**:
   - [ ] High object/task counts tested
   - [ ] No crashes or queue rejections
   - [ ] Performance scales linearly with workload

---

## Communication Protocol

When implementing any offload, provide:

1. **Current Status**: What you're working on (e.g., "Implementing Phase 1.1 - Dynamic Mesh Proximity")
2. **Changes Made**: List of files modified and what changed
3. **Verification**: Console output showing debug toggle working
4. **Performance**: Measured gains (before/after FPS or profiler results)
5. **Next Steps**: What's next or any blockers

---

## Context for Future Conversations

This document is the **single source of truth** for the offload project. Reference it when:
- Planning next steps
- Discussing implementation details
- Verifying completion of phases
- Measuring performance gains
- Adding new offload systems

**Key Context**:
- Game works amazing, preserve feel/behavior at all costs
- No fallback code (clean architecture)
- Every offload needs debug toggle in Developer Tools
- Verify via console output before marking complete
- Organized structure, follow existing patterns
- Incremental progress, one system at a time

---

## Success Criteria (Project Complete)

The offload project is complete when:

1. **Phase 5 Complete**: BVH raycasting offloaded (or determined infeasible)
2. **Performance**: +60-80% main thread freed, +20-40 FPS measured
3. **Zero Regressions**: Game feels identical to original
4. **Clean Codebase**: No fallback code, all debug toggles working
5. **Documentation**: All actual gains documented vs predictions
6. **Future-Ready**: Architecture supports new systems (AI, particles, etc.)

---

## Current Status

**Phase**: ✅ Sprint 1 Complete
**Completed**:
- Sprint 1.1 - Dynamic Mesh Activation (working perfectly, 13-17µs per batch)
- Sprint 1.2 - Interaction Proximity/AABB Checks (working perfectly, 4-34µs per batch)

**Next Task**: Developer Health Monitoring (modal/engine/worker diagnostics)
**Then**: Sprint 2.1 - Platform Velocity OR Sprint 4 - BVH Raycasting
**Blockers**: None
**Notes**: Architecture proven solid. Need comprehensive health dashboard before continuing offload work.

---

**Last Updated**: 2025-11-23 (Sprint 1 Complete)
**Version**: 1.1
