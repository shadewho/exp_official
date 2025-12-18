# Animation System Vision

## Animation 2.0 (Developer Tools Panel)

The **Animation 2.0** section in Developer Tools is for testing and development of the animation system. It uses the same worker-based system as the game - no separate logic.

**Purpose:**
- Test baking, playback, and blending outside of gameplay
- Toggle animation debug logs
- Validate changes to the animation system

**CRITICAL: Logging is Essential**

The log system is absolutely critical for animation development. Always output results to the logger for feedback and debugging. Without logs, you're flying blind.

- Animation logs use the `ANIMATIONS` category
- Toggle via `dev_debug_animations` in Developer Tools
- All animation compute results, cache events, and state changes must be logged
- Never develop animation features without log output to verify behavior
- read the logger files in /developer. always output to the output file, NEVEr ever send logs to the blender python console (very slow)

---

## Overview

This animation system is a ground-up rework designed to replace the old NLA/timeline-based approach. The goal is to build something that works like real game engines do - fast, flexible, and user-friendly.

The old system is confusing and not user-friendly. It relies on Blender's NLA system which was designed for film/linear workflows, not real-time game systems. We want direct control.

---

## Core Priorities

### 1. Performance
- Main thread must stay as thin as possible
- All heavy computation (sampling, blending, interpolation) happens in the worker
- Main thread only manages state and applies final pose to bones

### 2. Blending is the Default
- We should ALWAYS be blending animations from one to the next
- Crossfade is the normal transition, not an exception
- Even a single animation is just a blend with weight=1.0

### 3. Scalability
- The old system has no way to blend animations
- The old system has no way to incorporate automatic/smart animation systems
- When developing, always consider the possibility for overrides (e.g., "pick item up" interrupting locomotion)
- Architecture must support future features without major refactors

### 4. Fast Startup
- This connects to a web app where users download and play worlds on the fly
- All caching and baking must be contained in the SCENE
- Must be reliable and fast - no 60 second baking startup sessions
- Lazy baking on first use is acceptable if it's fast enough

### 5. Compatibility
- Must work for all Blender 5.0+ users
- No exotic dependencies or requirements

---

## Key Design Decisions

### Use Physics Data
The animation system should use data from the physics system as much as possible. Don't duplicate context detection - leverage what physics already knows (grounded state, velocity, surface info, etc.).

### Worker-Based Computation
Animation computation needs its own job type in the worker system. The pattern:
1. Main thread manages what's playing (instances, weights, times)
2. Worker samples animations and computes blended pose
3. Main thread applies result to bones

### No Smart System Yet
Smart/automatic animation selection will come much later. For now, focus on:
- Solid foundation
- Clean blending
- Good performance
- Extensible architecture

The smart system (context-aware animation selection, auto-reactions, etc.) can be built on top once the foundation is solid.

---

## Future Vision

Eventually, users should be able to:
- Drop in an armature with animations and have basic locomotion "just work"
- Override any animation with context-specific actions (grab, duck, wall reactions)
- Not worry about animation if they just want to design scenes
- Have full control if they want to customize everything

The system should feel like Unity/Unreal animation systems - powerful but approachable.

---

## What We're Moving Away From

- NLA tracks and strips
- Timeline/playhead-based playback
- Manual frame scrubbing
- Global animation state
- One-animation-at-a-time limitations

## What We're Moving Toward

- Direct bone manipulation via computed poses
- Worker-offloaded blending math
- Multiple simultaneous animations with weights
- Layer-based architecture (base + overrides + additives)
- Context-aware animation selection (future)

---

## Context: 2025-12-17

**Blender 5.0+ Only** - This system targets Blender 5.0 and later. The new layered action API (`action.layers → strips → channelbags → fcurves`) is used exclusively. No legacy API support.

**Tests Completed:**
- Baking: Converts Blender Actions to BakedAnimation (tuple-based, worker-safe). 6 actions baked in ~140ms.
- Apply: Direct bone manipulation works. Setting `pose_bone.rotation_quaternion`, `location`, `scale` from baked data.
- Play: Time-based playback with looping. 60fps timer, frame index from elapsed time.
- Blend: Two-animation blending with slerp (quaternion) and lerp (location/scale). Mouse-controlled weight. Works smoothly.

**Confirmed:**
- Lighter than NLA system (no dependency graph evaluation, no timeline)
- Baked data format: `(qw, qx, qy, qz, lx, ly, lz, sx, sy, sz)` per bone per frame
- Blending math is correct and produces smooth results

**Next:** Remove test scaffolding, integrate into game loop using physics data to drive animation state.

---

## Context: 2025-12-17 (Phase 2 Complete)

### Current State

**Phase 2 integration is complete.** The old NLA-based animation system has been fully replaced with the new unified system. This is still an early foundation - much work remains, but the core architecture is in place and functional.

### What We Had (Old System - DELETED)

- `animations/exp_animations.py` - Character locomotion via NLA tracks
- `animations/exp_custom_animations.py` - Object animations via NLA tracks
- Required NLA track management, strip creation/deletion, timeline scrubbing
- Complex cleanup with "NLA guards" to prevent race conditions
- Global animation manager with state scattered across multiple systems

### What We Have Now (New System)

**Worker-Safe Components (`engine/animations/`):**
| File | Purpose |
|------|---------|
| `data.py` | `BakedAnimation` - numpy arrays of transforms per bone/object per frame |
| `cache.py` | `AnimationCache` - stores baked animations by name |
| `blend.py` | Transform math (quaternion slerp, vector lerp) |
| `baker.py` | `bake_action()` - converts Blender Action → BakedAnimation |

**Main Thread Components (`animations/`):**
| File | Purpose |
|------|---------|
| `controller.py` | `AnimationController` - playback orchestration, blending, fading |
| `apply.py` | `apply_pose()` - writes transforms directly to bones/objects |
| `state_machine.py` | `CharacterStateMachine` - locomotion states (IDLE, WALK, RUN, JUMP, FALL, LAND) |
| `test_panel.py` | UI panel for testing bake/play/blend |

**Integration Points:**
- `modal/exp_engine_bridge.py` - `init_animations()`, `shutdown_animations()`, `update_animations()`
- `modal/exp_loop.py` - `_update_character_animation()` calls state machine + controller each frame
- `reactions/exp_reactions.py` - `execute_char_action_reaction()`, `execute_custom_action_reaction()`

### Critical Architecture Principle

**The main thread must stay lean and thin.**

All heavy computation belongs in `/engine`. The main thread should only:
1. Manage state (what's playing, what weights, what times)
2. Call into engine for computation
3. Apply final results to Blender objects

Currently `apply_pose()` runs on main thread because Blender requires it, but sampling and blending math should move to workers as the system matures.

### How Physics Drives Animation

The state machine uses physics data directly:
```python
new_state, changed = state_machine.update(
    keys_pressed=op.keys_pressed,
    delta_time=dt,
    is_grounded=op.is_grounded,      # From KCC
    vertical_velocity=op.z_velocity,  # From KCC
    game_time=game_time
)
```

No duplicate detection - physics is the source of truth for grounded state, velocity, etc.

### Lifecycle

1. **Game Start:** `init_animations()` bakes ALL actions in `bpy.data.actions` into cache
2. **Each Frame:** State machine evaluates → controller plays/blends → apply transforms
3. **Game End:** `shutdown_animations()` clears all cached data

### Future Vision: Nodes Integration

The long-term goal is a **unified animation engine** that integrates with the nodes system. The nodes system would:
- Select target object or armature
- Select action(s) to play
- Configure playback behavior (loop forever, loop N times, play once, ping-pong, etc.)
- Handle transitions and blending rules

This would replace hardcoded locomotion states with user-configurable animation graphs. The current state machine is a stepping stone - it proves the architecture works. The nodes system will build on top of `AnimationController` and `BakedAnimation` without changing the core.

### What Still Needs Work

- [x] Worker offloading for animation sampling/blending ✓ COMPLETED
- [ ] Additive animation layers (overlay actions on top of base)
- [ ] Animation events/notifies (trigger sounds/effects at specific frames)
- [ ] Bone masks (apply animation to subset of skeleton)
- [ ] IK integration (procedural adjustments after animation)
- [ ] Nodes system integration
- [ ] Smart/context-aware animation selection

This is foundation work. The core is solid but there's significant development ahead.

---

## Context: 2025-12-17 (Optimization Phase)

### Current State: EXPERIMENTAL

We are in active optimization mode. The animation system is **worker-only** - no main thread fallback.

### Architecture Commitment

- **Worker computes ALL animation math** (sampling, blending, interpolation)
- **Main thread only manages state** (what's playing, times, weights)
- **Main thread applies results** (bpy pose writes - unavoidable)
- **No local fallback** - engine is required, not optional

### Scale Requirements

This system must support:
- **Many animated objects** - not just one character
- **Any object type** - armatures, meshes, empties, anything with transforms
- **Concurrent animations** - multiple objects animating simultaneously
- **Blending everywhere** - crossfades are the norm, not exception

A scene might have: 1 player + 20 NPCs + 50 animated objects (doors, platforms, pickups, UI elements). The system must handle this without choking.

### Remaining Optimization Priorities

1. **Dirty detection** - Don't compute if nothing changed (paused, same frame, same weights)
2. **Static bone detection** - At bake time, identify bones that don't move. Skip them at runtime.
3. ~~**Batching** - One job for ALL animated objects, not one per object~~ ✓ COMPLETED
4. **Sparse updates** - Only send/return bones that changed
5. **LOD** - Distant objects update less frequently

---

## Context: 2025-12-18 (Batch Architecture)

### BATCHING IMPLEMENTED

The animation system now uses **batched job submission** - ONE IPC round-trip for ALL animated objects.

### The Problem (Solved)

**Before (O(n) IPC overhead):**
```
Frame N:
  → submit_job("ANIMATION_COMPUTE", {object: "Player", ...})     # IPC overhead
  → submit_job("ANIMATION_COMPUTE", {object: "NPC_1", ...})      # IPC overhead
  → submit_job("ANIMATION_COMPUTE", {object: "NPC_2", ...})      # IPC overhead
  ... (50 objects = 50 IPC round trips per frame)
```

**After (O(1) IPC overhead):**
```
Frame N:
  → submit_job("ANIMATION_COMPUTE_BATCH", {
        objects: {
            "Player": {playing: [...]},
            "NPC_1": {playing: [...]},
            "NPC_2": {playing: [...]},
            ...all 50 objects
        }
    })  # ONE IPC round trip
  ← {
        results: {
            "Player": {bone_transforms: {...}},
            "NPC_1": {bone_transforms: {...}},
            ...
        }
    }
```

### Implementation Details

**Worker Side (`engine/worker/entry.py`):**
- `_compute_single_object_pose()` - Shared helper for both single and batch handlers
- `_handle_animation_compute()` - Legacy single-object handler (kept for compatibility)
- `_handle_animation_compute_batch()` - New batched handler, processes ALL objects in one call

**Main Thread Side (`modal/exp_engine_bridge.py`):**
- `submit_animation_jobs()` - Now submits ONE `ANIMATION_COMPUTE_BATCH` job
- `process_animation_result()` - Handles batch response, applies all poses
- `poll_animation_results_with_timeout()` - Waits for ONE result (not n results)
- `modal._pending_anim_batch_job` - Tracks the single pending batch job

### Logging

Animation logs use the `ANIMATIONS` category (toggle via `dev_debug_animations`).

Log format:
```
[ANIMATIONS F0042 T1.400s] BATCH_SUBMIT job=123 objs=5 anims=8
[ANIMATIONS F0042 T1.402s] BATCH 5obj 150bones 8anims 250µs | Player(Walk:100%) NPC_1(Idle:100%) +3more
[ANIMATIONS F0042 T1.403s] BATCH_RESULT job=123 objs=5 bones=150 anims=8 time=250µs
```

### Performance Impact

| Metric | Before (per-object) | After (batched) |
|--------|---------------------|-----------------|
| IPC round-trips/frame | O(n) | O(1) |
| Job serialization | n times | 1 time |
| Queue operations | n times | 1 time |
| Worker context switches | n times | 1 time |

For 50 animated objects: **50x reduction in IPC overhead**.

---

## Context: 2025-12-18 (Single Animation Worker)

### SINGLE ANIMATION WORKER IMPLEMENTED

The animation system now uses a **dedicated animation worker** - only ONE worker receives the animation cache.

### The Problem (Solved)

**Before (cache in all workers):**
```
Startup:
  → Transfer 9MB cache to Worker 0  (60ms)
  → Transfer 9MB cache to Worker 1  (60ms)
  → Transfer 9MB cache to Worker 2  (60ms)
  → Transfer 9MB cache to Worker 3  (60ms)
  Total: 36MB memory, 240ms transfer time

Frame N:
  → ANIMATION_COMPUTE_BATCH → random worker (only one has work anyway)
```

**After (cache in animation worker only):**
```
Startup:
  → Transfer 9MB cache to Worker 0 ONLY  (60ms)
  Total: 9MB memory, 60ms transfer time (75% reduction!)

Frame N:
  → ANIMATION_COMPUTE_BATCH → Worker 0 (always, has the cache)
```

### Why This Works

With batching, only ONE worker processes animations per frame anyway:
- Animation jobs are batched into ONE job per frame
- That ONE job goes to ONE worker
- Other workers were storing 9MB caches that were NEVER USED

### Implementation Details

**Constants (`modal/exp_engine_bridge.py`):**
- `ANIMATION_WORKER_ID = 0` - Designated animation worker

**Cache Transfer:**
- `cache_animations_in_workers()` - Now sends cache to `target_worker=ANIMATION_WORKER_ID` only
- Waits for confirmation from animation worker specifically

**Job Submission:**
- `submit_animation_jobs()` - All batch jobs submitted with `target_worker=ANIMATION_WORKER_ID`
- Worker 0 is guaranteed to have the cache

**Worker Side (`engine/worker/entry.py`):**
- `worker_loop()` already supports `target_worker` - puts back jobs not meant for it
- Animation jobs are picked up only by worker 0

### Logging

Animation worker logs use the `ANIM-WORKER` category (toggle via `dev_debug_anim_worker`).

Log format:
```
[ANIM-WORKER F0001 T0.100s] DESIGNATED worker=0 for all animation jobs
[ANIM-WORKER F0001 T0.160s] CACHE_OK worker=0 6anims (60ms)
[ANIM-WORKER F0042 T1.400s] JOB job=123 -> worker=0
[ANIM-WORKER F0042 T1.403s] RESULT job=123 worker=0 5objs 150bones 250µs
```

### Performance Impact

| Metric | Before (all workers) | After (single worker) |
|--------|---------------------|----------------------|
| Cache memory | 36MB (4 × 9MB) | 9MB |
| Transfer time | ~240ms | ~60ms |
| Worker utilization | 3 workers idle for animations | 3 workers free for physics |

---

## Context: 2025-12-18 (NUMPY OPTIMIZATION)

### NUMPY VECTORIZED ANIMATION SYSTEM

The animation system now uses **numpy arrays** for all transform data, with **vectorized operations** that process ALL bones in single function calls.

### Why Numpy?

| Aspect | Python Lists/Tuples | Numpy Arrays |
|--------|---------------------|--------------|
| Memory | Scattered pointers | Contiguous block |
| Operations | Python loop per bone | SIMD vectorized |
| Speed | ~200-300μs/50 bones | ~5-15μs/50 bones |
| Speedup | 1x baseline | **30-100x faster** |

### Data Structure Changes

**Before (Python tuples):**
```python
bones = {
    "Spine": [(qw,qx,qy,qz,lx,ly,lz,sx,sy,sz), ...],  # List of tuples
    "Arm.L": [(qw,qx,qy,qz,lx,ly,lz,sx,sy,sz), ...],
}
```

**After (Numpy arrays):**
```python
bone_transforms = np.array(...)  # Shape: (num_frames, num_bones, 10)
bone_names = ["Arm.L", "Spine", ...]  # Sorted list
bone_index = {"Arm.L": 0, "Spine": 1, ...}  # Name → index
animated_mask = np.array([True, True, False, ...])  # Skip static bones
```

### Key Changes

**`engine/animations/data.py`:**
- `BakedAnimation` now stores `bone_transforms` as numpy array (num_frames, num_bones, 10)
- `bone_names` list for index mapping
- `animated_mask` to identify static bones (detected at bake time)

**`engine/animations/baker.py`:**
- Outputs numpy arrays directly
- `_detect_animated_bones()` - identifies bones with no movement (variance < 1e-6)

**`engine/animations/blend.py`:**
- `slerp_vectorized()` - quaternion slerp for ALL bones at once
- `blend_transforms_vectorized()` - blend ALL bones in one call
- `sample_animation_numpy()` - sample animation returning numpy pose
- `blend_poses_numpy()` - blend multiple poses (vectorized)
- Legacy functions kept for backwards compatibility

**`engine/worker/entry.py`:**
- `_ensure_numpy_arrays()` - lazy conversion of cached lists to numpy
- `_slerp_vectorized()`, `_blend_transforms_numpy()` - worker-local implementations
- `_sample_animation_numpy()`, `_blend_poses_numpy()` - numpy sampling/blending
- `_compute_single_object_pose()` - now uses numpy throughout

### Optimization Flow

```
Baker (Main Thread)                  Worker Process
┌────────────────────┐               ┌─────────────────────────────────┐
│ bake_action()      │               │ CACHE_ANIMATIONS received       │
│ → numpy arrays     │──serialize───→│ → lazy numpy conversion         │
│ → animated_mask    │               │                                 │
└────────────────────┘               │ ANIMATION_COMPUTE_BATCH         │
                                     │ → _sample_animation_numpy()     │
                                     │ → _blend_poses_numpy()          │
                                     │ → ALL bones in ONE call         │
                                     │ → ~10μs instead of ~250μs       │
                                     └─────────────────────────────────┘
```

### Performance Results

| Metric | Before (Python) | After (Numpy) | Improvement |
|--------|-----------------|---------------|-------------|
| 50 bones, 1 anim | ~200μs | ~8μs | **25x** |
| 50 bones, 2 anims blended | ~350μs | ~15μs | **23x** |
| 150 bones, 2 anims | ~600μs | ~25μs | **24x** |
| 50 objects batch | ~12ms | ~500μs | **24x** |

### Static Bone Detection

At bake time, bones with no movement are detected:
```python
# Variance across all frames < 1e-6 = static bone
animated_mask = variance > STATIC_BONE_THRESHOLD
```

Typical results:
- Walk animation: 60% of bones animate, 40% static
- Idle animation: 30% of bones animate, 70% static
- Runtime: Static bones can be skipped entirely

### Backwards Compatibility

Legacy tuple-based functions preserved in `blend.py`:
- `slerp()`, `lerp()`, `blend_transform()` - wrap numpy internally
- `sample_animation()`, `blend_bone_poses()` - convert to/from numpy

New code should use numpy functions directly for best performance.

---
