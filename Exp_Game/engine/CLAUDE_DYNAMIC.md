# Dynamic Mesh System - Current State

**Last Updated**: 2025-12-20

---

## 1. Main Thread Must Stay Thin

**This is non-negotiable.** All physics computation runs in worker threads.

**Main thread responsibilities:**
- Cache mesh triangles to worker (one-time per mesh)
- Send transform updates for meshes that moved (dirty check)
- Compute velocities for platform carry
- Apply physics results from worker

**Main thread does NOT:**
- Run raycasts
- Test triangles
- Perform collision detection
- Do spatial queries

Worker handles ALL physics decisions. Main thread is I/O only.

---

## 2. Testing & Debugging with Logger

**Always verify changes with the logging system.**

- Enable debug flags in Developer Tools panel
- Key categories: `DYN-CACHE`, `DYN-OPT`, `GROUND`, `PHYSICS`, `HORIZONTAL`
- Logs export to: `C:\Users\spenc\Desktop\engine_output_files\diagnostics_latest.txt`

**Log format examples:**
```
[DYN-CACHE] CACHED platform_01: 128 tris, radius=2.50m
[DYN-OPT] rays=12 early_outs=8 saved=66%
[GROUND F0042 T1.401s] HIT source=dynamic_12345 z=3.50m
[PHYSICS F0042] total=180us | rays=12 tris=45 | ground=dynamic
```

**Before/after any change:** Run the game, check logs, confirm behavior matches expectations.

---

## 3. Optimizations & Fixes

### Completed (2025-12-15)

| Optimization | Description | Savings |
|-------------|-------------|---------|
| Direct trig rotation | Replaced `Matrix.Rotation()` with `cos`/`sin` for wish direction | ~50-100us/frame |
| Quaternion angular velocity | Replaced 4x4 matrix inversion with quaternion (O(1) vs O(n^3)) | ~20-50us/platform |
| Cached velocity lookup | Avoid duplicate dict access + sqrt in proactive detection | ~5-15us/frame |
| Squared length check | Check `len_sq < threshold^2` before calling `sqrt()` | ~2-10us/frame |
| Bounding radius helper | Extracted `_compute_bounding_radius()` - no duplicate code | Cleaner code |
| Quaternion storage | Store 4 floats instead of 16 for previous rotation | 75% memory reduction |

### Bug Fix (2025-12-17): Multi-Worker Rotation Stale Cache

**Problem:** Platform rotation sync was inconsistent - sometimes character spun correctly with platform, sometimes way too fast.

**Root Cause:** `_platform_prev_yaw` was a worker-side global. With 4 workers, each had its own stale copy. When jobs were distributed across workers, `prev_yaw` came from whichever frame that worker last processed (not the actual previous frame).

**Fix:** Moved `prev_yaw` tracking to main thread:
- Main thread stores `_platform_prev_yaws` dict in exp_kcc.py
- Passed to worker via job data (`platform_prev_yaws`)
- Worker returns `platform_current_yaw` and `platform_rot_obj_id`
- Main thread stores returned value for next frame

**Files changed:** `exp_kcc.py`, `engine/worker/physics.py`

### Optimization (2025-12-17): Dirty Transform Tracking

**Problem:** All dynamic mesh transforms (64 bytes each) were sent every frame, even for stationary meshes.

**Fix:** Main thread tracks `_prev_dynamic_transforms` and only sends transforms that changed:
- Compare current transform tuple to previous
- Only include in job data if different
- Worker's persistent `cached_dynamic_transforms` keeps using cached values for unchanged meshes

**Savings:** With 4 meshes where only 1 is moving:
- Before: 256 bytes/frame (4 × 64)
- After: 64 bytes/frame (1 × 64)
- 75% reduction in transform serialization

**Files changed:** `exp_kcc.py`

**Dead code removed:**
- `platform_delta_map` - computed but never read
- `platform_delta_quat_map` - computed but never read
- `platform_motion_map` - populated but never read
- 4 deprecated `cache_*` methods in exp_kcc.py

### Bug Fix (2025-12-20): Stationary Dynamic Mesh Never Active

**Problem:** Stationary dynamic meshes (ones that never move) would never interact with physics until manually moved. The mesh would show `transform_cache=3` when 4 meshes existed.

**Root Cause:** Race condition + dirty check interaction:
1. Frame 1: All transforms sent, but physics job times out or worker hasn't cached mesh triangles yet
2. Worker discards transforms for meshes not yet in `cached_dynamic_meshes`
3. Main thread already updated `_prev_dynamic_transforms` (thinks transform was sent)
4. Frame 2+: Stationary mesh passes dirty check (`prev == current`), transform NOT resent
5. Mesh never gets its transform in worker cache → invisible to physics forever

**Fix:** Added 30-frame warmup period where all transforms are always sent:
- `_transform_warmup_frames = 30` counter in exp_kcc.py
- During warmup: skip dirty check, send all transforms every frame
- After warmup: resume normal dirty checking for efficiency
- This ensures worker receives transforms after mesh cache is fully populated

**Files changed:** `exp_kcc.py`

---

## 4. What Works

**Platform Riding:** Player can stand on moving platforms and be carried correctly. Uses relative position storage - player position stored in platform's local space on landing, transformed back to world each frame.

**Platform Rotation:** Player rotates with spinning platforms. Main thread tracks `prev_yaw` per platform, passes to worker, worker computes delta and returns current yaw. Consistent frame-to-frame tracking regardless of which worker processes each job.

**Horizontal Collisions:** Dynamic meshes push the player horizontally. Uses proactive detection - rays cast toward approaching mesh surfaces based on mesh velocity. Speed-scaled detection range handles fast-moving meshes.

**Unified Raycast:** Static and dynamic meshes use identical code paths. `unified_raycast()` tests both, returns closest hit regardless of source. No special-case logic.

---

## 5. Known Issues (Needs Work)

### Reset Spawn Location Drift
- Resetting game while interacting with dynamic mesh causes spawn location to differ
- Likely related to platform carry state not being fully cleared on reset
- Check `exp_game_reset.py` for platform state cleanup


---

## 6. Development Priorities

**Speed and efficiency are paramount.**

1. **Measure first** - Use logger to identify actual bottlenecks before optimizing
2. **Worker-side solutions** - Never add physics to main thread
3. **Early-out patterns** - Skip unnecessary work when possible
4. **Cache everything** - Triangles cached once, transforms only sent when changed
5. **Avoid allocations** - Reuse data structures, minimize per-frame `new` operations

**Key files:**
- `engine_worker_entry.py` - Worker physics, unified raycast
- `exp_dynamic.py` - Main thread dynamic mesh handling
- `exp_kcc.py` - Character controller, job submission
- `dev_logger.py` - Fast buffer logging system

---

## Quick Reference

**Enable dynamic mesh logging:**
1. Developer Tools panel > Debug Toggles
2. Enable `dev_debug_dynamic_cache` and `dev_debug_dynamic_opt`
3. Run game, interact with dynamic meshes
4. Check `diagnostics_latest.txt`

**Architecture summary:**
```
Main Thread (thin)          Worker (all physics)
------------------          -------------------
Send transforms    -->      Receive transforms
                            Update AABB cache
                            Unified raycast (static + dynamic)
                            KCC physics step
                            Platform detection
Apply result       <--      Return pos/vel/ground
```
