# Dynamic Mesh System - Development Guide

**Status**: WORK IN PROGRESS - Partially offloaded, needs optimization

---

## Goal

Fully offload dynamic mesh physics to the worker engine. Unify static and dynamic collision systems so we don't maintain two separate physics codepaths. Dynamic meshes should eventually be a simplified rigid body system that interacts with the character in any way (platforms, pushers, elevators, doors, etc.).

**We are NOT there yet.**

---
---
## Important
read CLAUDE_GAME and CLAUDE_ENGINE CONTEXT
---
## Current State

### What Works
- Triangle caching to worker (one-time per mesh)
- Transform matrices sent per-frame for active meshes
- Ground detection on dynamic meshes
- Platform carry (player moves with platform)
- AABB-based activation gating
- "Standing on" override to prevent deactivation while on platform

### What Doesn't Work Well
- **Bouncing/instability** - Player falls into mesh, gets caught, bounces back up
- **Frame latency issues** - Suspected timing problems between main thread and worker
- **Inconsistent ground detection** - Works some frames, fails others
- **Edge cases** - AABB boundaries cause activation flapping

---

## Architecture

### Main Thread (exp_dynamic.py)
```
1. Cache ALL dynamic mesh triangles to worker (one-time, game start)
2. For each dynamic mesh:
   a. Check if player standing on it (force active)
   b. Else check AABB proximity (2m margin)
   c. If active: send transform, compute velocity
   d. If inactive: skip (save CPU)
```

### Worker (engine_worker_entry.py)
```
1. Receive transform matrices for active meshes
2. Transform cached local triangles to world space
3. Build unified_dynamic_meshes list with AABB + bounding sphere
4. Test rays against both static grid AND dynamic meshes
5. Return closest hit with source metadata (static vs dynamic_{obj_id})
```

### KCC Physics (exp_kcc.py)
```
1. Build job with dynamic_transforms from active meshes
2. Submit to worker, poll result
3. Parse ground_hit_source to identify platform
4. Apply platform carry velocity if on dynamic ground
```

---

## Key Files

| File | Purpose |
|------|---------|
| `engine/engine_worker_entry.py` | Worker physics, transform, collision |
| `physics/exp_dynamic.py` | Main thread activation, caching, velocity |
| `physics/exp_kcc.py` | KCC job building, result parsing, platform carry |
| `modal/exp_loop.py` | Game loop ordering |

---

## Known Issues To Fix

1. **Bouncing** - Root cause unclear. May be timing, may be ground detection, may be snap logic.

2. **Frame latency** - Main thread and worker may have sync issues. Transforms might be stale.

3. **AABB edge cases** - When mesh moves, AABB shifts, player falls out of activation zone briefly.

4. **Ground detection inconsistency** - `GROUND_MISS` and `GROUND_ABOVE_REJECT` happen when they shouldn't.

---

## Development Workflow

**ALWAYS use the dev logger to debug and test results.**

1. Enable in N-panel:
   - `dev_debug_dynamic_mesh` - Transform timing, activation state
   - `dev_debug_dynamic_collision` - Ground hits, AABB rejections
   - `dev_debug_unified_physics` - Combined physics summary

2. Export logs to: `C:\Users\spenc\Desktop\engine_output_files\diagnostics_latest.txt`

3. Look for patterns:
   - `static_only` when should have dynamic = transforms not being sent
   - `GROUND_MISS` when standing on platform = collision detection failing
   - `AABB DEACTIVATED` followed by fall = activation edge case

---

## Unified Physics Principle

Static and dynamic meshes should share:
- Same ray-triangle intersection function
- Same ground detection logic
- Same collision response
- Same step-up/slide behavior

They differ only in:
- Data source (static=cached grid, dynamic=per-frame transform)
- Culling method (static=spatial grid O(1), dynamic=AABB per mesh)
- When tested (static=always, dynamic=when active)

**Do NOT create separate physics code for dynamic meshes.**

---

## Next Steps

1. Diagnose and fix bouncing (priority)
2. Investigate frame latency between transform send and collision test
3. Consider always-active mode for platforms player is near (larger margin?)
4. Profile and optimize transform cost
5. Add more diagnostic logging to pinpoint failures

---

**End of Document**
