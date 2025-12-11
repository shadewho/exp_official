# Dynamic Mesh Physics Offload - Development Guide

**Last Updated**: 2025-12-11
**Status**: FULLY UNIFIED - Zero main thread collision, AABB-gated activation

---

## Mission Statement

**Goal**: Unify static and dynamic mesh physics into a single, high-performance system running entirely in the worker engine. Free the main thread from ALL physics computation. The only difference between static and dynamic meshes should be:
1. **AABB activation** - Dynamic meshes activate when player within bounding box + margin
2. **Physical influence** - Dynamic meshes can push/carry the character

Everything else (collision detection, ray testing, step-up, ground detection) uses **identical physics code** for both static and dynamic.

---

## Architecture (FULLY CLEANED UP)

### **Key Principle: One Collision System, Two Data Sources**

```
 WORKER: Unified Physics Engine

  For EACH collision check:
  1. Test static grid (fast 3D DDA)
  2. Test dynamic meshes (sphere cull - tris)
  3. Return closest hit + source metadata

   Used by ALL physics checks:
  - Horizontal collision (3 rays + width + slope)
  - Ground detection
  - Ceiling check
  - Body integrity ray
  - Step-up

  Single collision response per check
  Result includes ground_hit_source for platform
```

### **Main Thread Responsibilities (MINIMAL)**

```
MAIN THREAD:
  1. Snapshot Blender state (transforms)
  2. Submit physics job to worker
  3. Poll for worker result
  4. Apply position/velocity to character
  5. Apply platform carry velocity (if on dynamic ground)

  NO collision detection on main thread!
```

---

## Cleanup Summary (2025-12-10)

### **DELETED from exp_kcc.py:**
- `_handle_dynamic_movers()` method (was ~90 lines) - Push-out now handled by worker's horizontal collision
- `_check_dynamic_collision()` method (was ~130 lines) - All collision now in worker

### **SIMPLIFIED in exp_dynamic.py:**
- LocalBVH **KEPT** but only for camera/projectiles/tracking (NOT used by KCC)
- `dynamic_bvh_map` - For camera/projectiles (with LocalBVH)
- `dynamic_objects_map` - Simpler map for KCC platform carry lookup
- Triangle caching to worker (unchanged)
- Velocity calculation (unchanged)

### **ADDED to worker (engine_worker_entry.py):**
- `UNIFIED` log category showing unified physics status
- Confirms static+dynamic testing in single path

---

## Key Files Reference

### **Worker Engine**
- `engine_worker_entry.py` - All worker physics computation
  - `ray_sphere_intersect()` - Quick sphere rejection test
  - `ray_aabb_intersect()` - AABB rejection (tighter than sphere)
  - `compute_bounding_sphere()` - Sphere calculation
  - `compute_aabb()` - AABB calculation
  - `ray_triangle_intersect()` - With backface culling (~50% speedup)
  - `unified_raycast()` - Full unified raycast
  - `test_dynamic_meshes_ray()` - Simple ray helper (uses AABB + sphere)
  - Transform + bounds computation (`unified_dynamic_meshes`)
  - Horizontal rays with inline dynamic testing
  - Body integrity ray with dynamic testing
  - Ceiling check with dynamic testing
  - Ground detection with dynamic testing

### **Main Thread**
- `exp_kcc.py` - KCC physics controller
  - `_apply_physics_result()` - Parse ground_hit_source, apply platform carry
  - `_build_physics_job()` - Package transforms for worker
  - `step()` - Submit to worker, poll results
  - **NO collision detection methods** (all deleted)

- `exp_dynamic.py` - Dynamic mesh management
  - AABB-gated activation (zero-latency, main thread)
  - Triangle caching triggers (one-time per mesh)
  - Velocity calculation for platform carry
  - LocalBVH creation (for camera/projectiles ONLY)

### **Developer Tools**
- `dev_logger.py` - Fast buffer logging system
- `dev_debug_gate.py` - Frequency gating
- `dev_properties.py` - Debug toggles and Hz control
- `dev_panel.py` - Developer Tools UI panel

---

## Logging Categories

| Category | Debug Property | What It Shows |
|----------|---------------|---------------|
| `UNIFIED` | `unified_physics` | Unified physics status (static+dynamic) |
| `DYN-CACHE` | `dynamic_cache` | One-time mesh caching events |
| `DYN-MESH` | `dynamic_mesh` | Transform timing + activation state |
| `DYN-COLLISION` | `dynamic_collision` | Ground collision sources |
| `PHYS-BODY` | `body_ray` | Body integrity ray |
| `ENGINE` | `engine` | Worker timing breakdown |
| `KCC` | `kcc_offload` | KCC state per frame |

### **Expected Log Output (with unified physics):**
```
[UNIFIED] total=1234µs (xform=280µs) | static+dynamic=2 | rays=12 tris=450 | ground=dynamic_Platform
[DYN-MESH] TRANSFORM active=2 tris=536 time=280.5µs
[DYN-MESH] AABB ACTIVATED: Platform1 bounds=[(5.0,3.0,0.0)->(8.0,6.0,2.0)]
[KCC] GROUND pos=(5.23,3.12,1.00) step=False | 1234us 12rays 450tris
```

---

## Performance Benefits

| Aspect | Before (Split) | After (Unified) |
|--------|----------------|-----------------|
| **Code paths** | 2 (static + dynamic) | 1 (unified) |
| **Main thread collision** | BVH raycasts per mesh | NONE (zero) |
| **Dynamic testing** | Brute force ALL tris | AABB cull + backface cull |
| **Collision response** | Ran twice (bug!) | Single response per check |
| **Maintenance** | Fix bugs in 2 places | Fix bugs once |
| **Push-out logic** | Separate method | Handled by horizontal rays |
| **Backface culling** | None | ~50% fewer triangle tests |
| **AABB culling** | Only sphere | Tight box rejection |

---

## Performance Profile

**Expected (with AABB + backface culling):**

```
Transform + Bounds:  ~300us for 268 triangles (computes AABB + sphere)
Dynamic Ray Tests:   AABB culling for ray-mesh rejection
Triangle Tests:      Backface culling skips ~50% of triangles
Worker Total:        ~1500us per frame (5% of 30Hz budget)
Main Thread:         Apply result only (~50us)
Poll Timeout:        5ms (allows for dynamic mesh overhead)
```

**Optimizations Applied (2025-12-10):**
1. **AABB Culling** - Tighter than sphere for elongated meshes
2. **5ms Timeout** - Increased from 3ms to accommodate transform overhead

**Bug Fixes Applied (2025-12-11):**
1. **Ground Detection Fix** - Only accept hits BELOW player (prevents teleport-up to ceilings)
2. **Backface Culling Reverted** - Was breaking body integrity ray (shoots upward)
3. **AABB Debug Logging** - Shows mesh bounds and ray rejection reasons
4. **AABB Activation Fix** - Zero-latency main thread AABB check replaces worker-based distance gating (had 1-frame latency bug causing bouncing)

---

## Testing the Unified System

### **Enable Diagnostics**
In N-panel Developer Tools:
1. Set Master Frequency to 30 Hz (verbose) or 1 Hz (recommended)
2. Enable "Export Diagnostics Log to File"
3. Enable relevant categories:
   - Engine Diagnostics (shows UNIFIED logs)
   - Transform & Collision (dynamic mesh)
   - KCC Debug

### **What to Look For**
1. **Ground detection**: Walk onto dynamic platform
   - Log: `[UNIFIED] PHYSICS: ... ground=dynamic_PlatformName`
2. **Platform carry**: Character should move with platform
3. **Horizontal collision**: Walk into dynamic wall, should stop
4. **Ceiling check**: Jump under dynamic ceiling, should stop
5. **No main thread collision**: Should see no `_check_dynamic_collision` or `_handle_dynamic_movers` in profiler

### **Log Output Location**
`C:\Users\spenc\Desktop\engine_output_files\diagnostics_latest.txt`

---

## Data Flow Diagram

```
Main Thread                     Worker Process
    |                               |
    +- Update dynamic meshes -----> |
    |   (exp_dynamic.py)            |
    |   - AABB check (zero latency) |
    |   - Calc velocities           |
    |   - Send transforms           |
    |                               |
    +- Submit KCC job -----------> |
    |   (pos, vel, wish_dir)        |
    |                               +- Receive transforms
    |                               +- Transform local->world
    |                               +- Compute bounding spheres
    |                               |
    |                               +- Run unified physics:
    |                               |   - Horizontal rays (static + dyn)
    |                               |   - Width rays (static + dyn)
    |                               |   - Slope rays (static + dyn)
    |                               |   - Step-up check
    |                               |   - Body integrity (static + dyn)
    |                               |   - Ceiling check (static + dyn)
    |                               |   - Ground detect (static + dyn)
    |                               |
    |                               +- Build result
    |                               |   (pos, vel, ground_hit_source)
    |                               |
    | <-------- Return result ------+
    |
    +- Apply result to character
    +- Parse ground_hit_source
    +- Lookup ground object
    +- Apply platform carry velocity
```

---

## Future Optimizations

| Priority | Optimization | Impact | Status |
|----------|-------------|--------|--------|
| **1** | Backface Culling | 2x collision | Not Started |
| **2** | Pre-allocate Transform Lists | 1.3x transform | Not Started |
| **3** | Early-Out on Hit | 1.5x collision | Not Started |
| **4** | Parallel Physics Jobs | 2-3x total | Not Started |
| **5** | Numba JIT | 10-50x hot paths | Not Started |

### **Backface Culling Implementation:**
```python
# In test_dynamic_meshes_ray, before ray_triangle_intersect:
normal = compute_triangle_normal(tri[0], tri[1], tri[2])
if (normal[0] * ray_direction[0] + normal[1] * ray_direction[1] + normal[2] * ray_direction[2]) > 0:
    continue  # Skip backface
```

---

## Architecture Summary

**The unified physics system achieves:**

1. **Main thread does ZERO collision detection** - All offloaded to worker
2. **Static and dynamic use IDENTICAL physics code** - Single code path per check
3. **Efficient bounding sphere culling** - Skip entire meshes that ray misses
4. **ground_hit_source metadata** - Main thread knows what character stands on
5. **Platform carry preserved** - Uses ground_hit_source to apply velocity
6. **Comprehensive logging** - UNIFIED category confirms unity
7. **Clean codebase** - No deprecated methods, no duplicate logic

---

**End of Document**
