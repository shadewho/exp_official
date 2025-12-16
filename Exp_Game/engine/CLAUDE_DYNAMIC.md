# Dynamic Mesh System - Current State

**Last Updated**: 2025-12-16

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

## 3. Optimizations Completed (2025-12-15)

| Optimization | Description | Savings |
|-------------|-------------|---------|
| Direct trig rotation | Replaced `Matrix.Rotation()` with `cos`/`sin` for wish direction | ~50-100us/frame |
| Quaternion angular velocity | Replaced 4x4 matrix inversion with quaternion (O(1) vs O(n^3)) | ~20-50us/platform |
| Cached velocity lookup | Avoid duplicate dict access + sqrt in proactive detection | ~5-15us/frame |
| Squared length check | Check `len_sq < threshold^2` before calling `sqrt()` | ~2-10us/frame |
| Bounding radius helper | Extracted `_compute_bounding_radius()` - no duplicate code | Cleaner code |
| Quaternion storage | Store 4 floats instead of 16 for previous rotation | 75% memory reduction |

**Dead code removed:**
- `platform_delta_map` - computed but never read
- `platform_delta_quat_map` - computed but never read
- `platform_motion_map` - populated but never read
- 4 deprecated `cache_*` methods in exp_kcc.py

---

## 4. What Works

**Platform Riding:** Player can stand on moving platforms and be carried correctly. Uses relative position storage - player position stored in platform's local space on landing, transformed back to world each frame.

**Horizontal Collisions:** Dynamic meshes push the player horizontally. Uses proactive detection - rays cast toward approaching mesh surfaces based on mesh velocity. Speed-scaled detection range handles fast-moving meshes.

**Unified Raycast:** Static and dynamic meshes use identical code paths. `unified_raycast()` tests both, returns closest hit regardless of source. No special-case logic.

---

## 5. Known Issues (Needs Work)

### Multiple Dynamic Meshes Simultaneously
- Current system not fully tested with multiple moving meshes at once
- May have issues with priority/ordering when player contacts multiple meshes

### Reset Spawn Location Drift
- Resetting game while interacting with dynamic mesh causes spawn location to differ
- Likely related to platform carry state not being fully cleared on reset
- Check `exp_game_reset.py` for platform state cleanup

### Clip-Through / Sinking Into Meshes
- Character tends to sink into dynamic meshes during contact
- Penetration resolution may be insufficient
- May need stronger push-out force or better overlap detection

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
