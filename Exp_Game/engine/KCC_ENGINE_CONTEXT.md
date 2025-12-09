# KCC Physics Engine - Current State & Active Issues

**Status:** Core offload functional, smooth with prints off, collision resolution needs work
**Last Updated:** 2025-12-02
**Session Logs:** `C:\Users\spenc\Desktop\engine_output_files\diagnostics_latest.txt`

---
## ⚠️ CRITICAL: Development Workflow

**ALWAYS MAKE CHANGES TO:** `C:\Users\spenc\Desktop\Exploratory\addons\Exploratory`

**NEVER EDIT:** `C:\Users\spenc\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\Exploratory`

## 🎯 PRIMARY MANDATE

**FREE THE MAIN THREAD - OFFLOAD EVERYTHING POSSIBLE**

All physics computation MUST happen in worker (`engine_worker_entry.py`). Main thread (`exp_kcc.py`) is ONLY for:
- Submitting jobs
- Polling results (3ms timeout, ~100-200µs typical)
- Applying to Blender (bpy writes)
- GPU visualization (batched, single layer, performance-invisible)

**All solutions must be:**
- ✅ Robust and computationally friendly
- ✅ Never overwhelm the engine (respect 30Hz budget)
- ✅ Fully visualized via developer system
- ✅ Optimized for zero gameplay impact when enabled

---

## 🔥 CRITICAL PROJECT: DYNAMIC BVH OFFLOAD TO WORKER

**Status:** PLANNED - Ready to implement
**Goal:** Unify static and dynamic mesh physics in worker thread
**Impact:** Full physics parity + main thread freed from dynamic collision

### The Problem

**Current Architecture (Asymmetric):**
- Static geometry: Full physics in worker (capsule sweep, step-up, wall slide, body ray, ceiling, ground, slopes)
- Dynamic geometry: Partial physics on main thread (ground + horizontal push-out ONLY)
- **Result:** Dynamic meshes missing body integrity ray, step-up, wall slide, ceiling check, steep slopes

**Why This Matters:**
- Character can embed in dynamic platforms (no body ray)
- Can't step up on dynamic objects (no step-up logic)
- Slides through dynamic walls (no wall slide)
- Main thread doing physics work it shouldn't
- Two separate physics implementations = maintenance nightmare

### The Solution: Cache Local Geometry, Send Transforms

**Key Insight:** Dynamic meshes are RIGID BODIES (no deformation). We cache triangles in LOCAL SPACE once, send only transform matrices per frame.

**Architecture:**

```
PHASE 1: One-Time Cache (Startup/Registration)
─────────────────────────────────────────────────
New Job: CACHE_DYNAMIC_MESH
Input: obj_id, triangles (local space), radius
Worker stores: _cached_dynamic_meshes[obj_id] = {
    triangles: [(v0, v1, v2), ...],  # Local coordinates
    radius: float,
    spatial_cells: {}  # Optional mini-grid if >100 tris
}
Sent: ONCE per object (like CACHE_GRID for static)
Cost: One-time, negligible

PHASE 2: Per-Frame Transform Update (Lightweight!)
─────────────────────────────────────────────────
Modified KCC_PHYSICS_STEP job adds:
dynamic_transforms = {
    obj_id: matrix_4x4  # 16 floats = 64 bytes
}

Data size: 10 meshes = 640 bytes/frame
          20 meshes = 1,280 bytes/frame
          (vs 3MB if sending full triangles!)

PHASE 3: Worker Unified Physics
─────────────────────────────────────────────────
# Transform dynamic triangles ONCE per frame
for obj_id, matrix in dynamic_transforms.items():
    local_tris = _cached_dynamic_meshes[obj_id]['triangles']
    world_tris[obj_id] = [transform_tri(t, matrix) for t in local_tris]
    # Cost: 3 vertices × matrix multiply × N tris
    # 50 triangles = ~2.5µs

# ALL physics checks test BOTH static AND dynamic
Body integrity ray: test static grid + dynamic tris
Capsule sweep: test static grid + dynamic tris
Ground detection: test static grid + dynamic tris
Ceiling check: test static grid + dynamic tris
Step-up: test static grid + dynamic tris
Wall slide: test static grid + dynamic tris
Steep slopes: test static grid + dynamic tris

Result: IDENTICAL PHYSICS DNA
```

### Performance Budget

**Per-Frame Overhead (10 dynamic meshes × 50 tris each):**
- Transform triangles: ~7.5µs (3 vertices × 16 floats × 50 tris × 10 meshes)
- Serialize matrices: ~5µs (640 bytes)
- Test during collision: ~10-20µs (already testing nearby geometry)
- **Total added: <30µs to 100-200µs baseline = 15% increase**
- **Target: Stay under 300µs total worker time**

**Memory in Worker:**
- 10 dynamic meshes × 50 triangles = 18 KB triangle data
- Spatial grids (optional): ~50 KB per large mesh
- **Total: <1 MB for typical scenes**

**Activation Gating (Already Implemented):**
- Only send transforms for ACTIVE meshes (distance-gated with hysteresis)
- Typical: 3-5 active meshes near player
- Far meshes: Zero cost (not sent, not tested)

### Synchronization Requirements (CRITICAL)

**Same-Frame Polling (Already Working):**
- Main thread submits KCC_PHYSICS_STEP with dynamic transforms
- Worker computes full physics (static + dynamic unified)
- Main thread polls with 3ms timeout (~200µs typical)
- **Zero frame offset** - result applied same frame
- Platform carry applied AFTER worker result (velocity application)

**No 1-Frame Latency:**
- Dynamic transforms sent WITH physics job (not separate)
- Worker sees current-frame positions
- Result returned same frame
- Critical for smooth platform riding

**Frame Budget Enforcement:**
- Worker execution MUST stay <500µs to maintain smoothness
- If worker exceeds budget: reduce active dynamic mesh count or simplify geometry
- Main thread waits max 3ms (prevents frame drops)

### Developer System & Logging (CRITICAL)

**New Debug Categories (Add to dev_properties.py):**

1. **`dev_debug_dynamic_cache`** - Dynamic mesh caching
   - Logs: CACHE_DYNAMIC_MESH job (one-time per object)
   - Format: `[DYN-CACHE] obj={name} tris={count} radius={r:.2f}m cached`
   - Shows: Object registered, triangle count, bounding radius
   - Hz: N/A (one-time events)

2. **`dev_debug_dynamic_transform`** - Per-frame transform updates
   - Logs: Transform matrix submission to worker
   - Format: `[DYN-XFORM] active={count} bytes={size} objects=[id1,id2,...]`
   - Shows: How many active, data size, which objects
   - Hz: 1-30 Hz (default 1 Hz to avoid spam)

3. **`dev_debug_dynamic_physics`** - Dynamic collision results
   - Logs: Worker-side dynamic mesh collision tests
   - Format: `[DYN-PHYS] obj={name} hit={bool} dist={d:.3f}m normal=({x},{y},{z})`
   - Shows: Which dynamic mesh was tested, hit result, distance, normal
   - Hz: 1-30 Hz (default 5 Hz)

4. **`dev_debug_physics_unified`** - Combined static+dynamic stats
   - Logs: Per-frame summary of ALL collision tests
   - Format: `[PHYS-UNIFIED] static_tris={n} dynamic_tris={m} total_tests={k} time={t}µs`
   - Shows: Triangle counts tested, performance breakdown
   - Hz: 1-30 Hz (default 5 Hz)

**Enhanced Existing Categories:**

5. **`dev_debug_physics_body_integrity`** (ENHANCE)
   - ADD: Dynamic mesh embedding detection
   - Format: `[PHYS-BODY] EMBEDDED obj={name} type={static|dynamic} dist={d:.3f}m`
   - Shows: Whether embedding is from static or dynamic mesh
   - Differentiates: Static vs dynamic collision sources

6. **`dev_debug_kcc_offload`** (ENHANCE)
   - ADD: Dynamic mesh count to main summary line
   - Format: `[KCC F0042] GROUND🟢 pos=(x,y,z) | {time}µs static_grid={bool} dynamic_active={count}`
   - Shows: How many dynamic meshes active this frame

**Performance Logging:**

7. **`dev_debug_dynamic_performance`** - Detailed timing breakdown
   - Logs: Transform time, test time, per-mesh breakdown
   - Format: `[DYN-PERF] transform={t1}µs test_body={t2}µs test_capsule={t3}µs obj_breakdown=[...]`
   - Shows: Exactly where time is spent
   - Hz: 1-30 Hz (default 1 Hz)

**Output to diagnostics_latest.txt:**
- All categories respect Master Hz control
- Fast buffer logger (1000x faster than print)
- Zero gameplay impact when enabled
- Export after session for analysis

### Shared Physics DNA - Implementation Strategy

**Core Principle:** Static and dynamic are IDENTICAL except for transform source.

**Unified Collision Function:**
```python
def test_collision_unified(ray_origin, ray_dir, max_dist):
    """Test ray against ALL geometry (static + dynamic)."""
    best_hit = None
    best_dist = max_dist
    best_source = None  # "static" or dynamic obj_id

    # Test static grid (DDA traversal)
    for tri in static_grid_cells:
        hit, dist, normal = ray_triangle_intersect(ray_origin, ray_dir, tri)
        if hit and dist < best_dist:
            best_hit = (dist, normal, tri)
            best_dist = dist
            best_source = "static"

    # Test dynamic meshes (transformed triangles)
    for obj_id, tris in transformed_dynamic_tris.items():
        for tri in tris:
            hit, dist, normal = ray_triangle_intersect(ray_origin, ray_dir, tri)
            if hit and dist < best_dist:
                best_hit = (dist, normal, tri)
                best_dist = dist
                best_source = obj_id  # Dynamic mesh ID

    return best_hit, best_source
```

**Apply to ALL Physics Checks:**
- Body integrity ray: `test_collision_unified(feet_pos, up_dir, body_height)`
- Horizontal collision: `test_collision_unified(capsule_pos, fwd_dir, move_len)`
- Ground detection: `test_collision_unified(feet_pos, down_dir, snap_down)`
- Ceiling check: `test_collision_unified(head_pos, up_dir, move_z)`
- Step-up: `test_collision_unified(elevated_pos, fwd_dir, step_dist)`

**Result:** Every physics feature automatically works on both static and dynamic.

### Optimization Strategies

**1. Selective Transformation (Lazy Eval):**
```python
# Only transform dynamic meshes that are:
# - Active (distance-gated)
# - Moving this frame (delta_matrix != identity)
# - Near collision test (within capsule radius + mesh radius)
```

**2. Spatial Acceleration for Large Meshes:**
```python
# Build mini-grid for dynamic meshes with >100 triangles
if len(triangles) > 100:
    build_spatial_cells(triangles)  # Once at cache time
    # Per-frame: only transform cells near test point
```

**3. Early Distance Rejection:**
```python
# Before transforming triangles, check bounding sphere
mesh_center = transform_point(local_center, matrix)
if distance(test_point, mesh_center) > mesh_radius + test_radius:
    continue  # Skip entire mesh
```

**4. Activation Hysteresis (Already Working):**
```python
# Worker updates activation with 10% margin
# Prevents thrashing when player at boundary
# Typical: 3-5 active meshes instead of 10-20
```

### Success Metrics

**Physics Parity (Goal: 100%):**
- ✅ Body integrity ray works on dynamic
- ✅ Step-up works on dynamic platforms
- ✅ Wall slide works on dynamic walls
- ✅ Ceiling check works on dynamic ceilings
- ✅ Steep slopes work on dynamic ramps
- ✅ All collision feels identical to static

**Performance (Goal: <300µs total):**
- ✅ Worker execution stays under 300µs with 5 active dynamic meshes
- ✅ Main thread overhead stays under 50µs
- ✅ 30 Hz locked, zero stuttering
- ✅ Smooth platform riding (no jitter)

**Code Quality (Goal: Single Implementation):**
- ✅ Delete `_check_dynamic_collision()` from main thread (redundant)
- ✅ Single unified collision test function
- ✅ Static and dynamic use same code paths
- ✅ Maintenance burden reduced by 50%

**Logging & Visibility (Goal: Complete Transparency):**
- ✅ Every dynamic mesh interaction logged with source type
- ✅ Frame-by-frame timeline shows static vs dynamic collisions
- ✅ Performance breakdown shows transform + test times
- ✅ Embedding detection reports static vs dynamic source
- ✅ Diagnostics export shows full picture for Claude analysis

### Implementation Checklist

**Phase 1: Worker Infrastructure**
- [ ] Add `_cached_dynamic_meshes` global dict
- [ ] Implement `CACHE_DYNAMIC_MESH` job handler
- [ ] Add triangle transform helper function
- [ ] Add dynamic mesh activation filtering

**Phase 2: Job Data Extension**
- [ ] Modify `KCC_PHYSICS_STEP` to accept `dynamic_transforms`
- [ ] Serialize transform matrices in job submission
- [ ] Add activation gating (only send active meshes)

**Phase 3: Unified Collision Testing**
- [ ] Create `test_collision_unified()` helper
- [ ] Extend body integrity ray to test dynamic
- [ ] Extend horizontal collision to test dynamic
- [ ] Extend ground detection to test dynamic
- [ ] Extend ceiling check to test dynamic
- [ ] Extend step-up to test dynamic

**Phase 4: Developer System**
- [ ] Add 4 new debug properties (cache, transform, physics, unified)
- [ ] Enhance 2 existing properties (body, kcc_offload)
- [ ] Add performance timing property
- [ ] Add all logging categories to `_CATEGORY_MAP`
- [ ] Test Master Hz gating on all categories

**Phase 5: Main Thread Cleanup**
- [ ] Delete `_check_dynamic_collision()` method
- [ ] Remove dynamic collision call from `_apply_physics_result()`
- [ ] Keep platform carry application (velocity, not collision)
- [ ] Update comments to reflect new architecture

**Phase 6: Testing & Validation**
- [ ] Test body ray on dynamic platform (should detect embedding)
- [ ] Test step-up on dynamic stairs (should climb)
- [ ] Test wall slide on dynamic wall (should slide)
- [ ] Test ceiling on dynamic moving platform (should block)
- [ ] Profile worker time (<300µs target)
- [ ] Export diagnostics, verify unified logging
- [ ] Verify zero frame offset (smooth platform riding)

**Philosophy:**
Dynamic and static share the same physics DNA. The ONLY difference is where triangles come from (cached grid vs transformed cache). Everything else is IDENTICAL. One implementation, one maintenance burden, full parity.

---

## 🚨 ACTIVE ISSUES (Priority Order)

### 1. Mid-Height Collision Resolution
**Problem:** Capsule collision detection inconsistent at mid-height (between feet and head spheres)
**Impact:** Character passes through or clips into geometry at torso level
**Root Cause:** Only 2 sphere sweeps (feet + head), no coverage in middle third of capsule

**Current Work:** Vertical body integrity ray implemented (feet→head detection)
- ✅ Detects mesh embedding between capsule spheres
- ✅ Visualizer: Orange=clear, Red=embedded (thick beam with markers)
- ✅ Logging: `[EMBEDDED] hit=X.XXm pct=XX.X%` with penetration depth
- ❌ **Next:** Use ray to assist collision resolution for midpoint contacts
- ❌ **Next:** Handle dynamic meshes crossing between spheres
- **Philosophy:** Ray bridges the gap - helps main spheres handle what they missed

### 2. Dynamic BVH Character Rotation
**Problem:** Dynamic BVHs don't rotate the character when platforms rotate
**Impact:** Character slides off rotating platforms instead of rotating with them

### 3. Dynamic BVH Speed & Falling
**Problem:** Dynamic BVHs fail when:
- Character falls onto them from above
- Platform applies speed/acceleration to character
**Impact:** Character falls through or gets stuck in moving platforms

### 4. Slope Limits Ineffective
**Problem:** Slope angle limits don't prevent uphill movement on steep surfaces
**Impact:** Character can climb walls they shouldn't be able to
**Expected:** `steep_slide_gain` and `steep_min_speed` should force sliding, not allow climbing

### 5. Capsule Strength Inconsistency
**Problem:** Overall collision strength varies by contact height (mid/head especially weak)
**Impact:** Unpredictable collision response, character squeezes through tight spaces
**Related:** Issue #1 (mid-height coverage)

---

## 🛠️ DEVELOPER SYSTEM WORKFLOW (CRITICAL)

**New Standard:** Visualize → Export → Analyze → Fix → Repeat

### Debug Output Pipeline
```
1. Enable debug categories (N-panel → Developer Tools)
2. Enable "Export Diagnostics Log" toggle
3. Play game (diagnostics go to memory buffer)
4. Stop game → auto-exports to C:\Users\spenc\Desktop\engine_output_files\diagnostics_latest.txt
5. Read log file in Claude → analyze frame-by-frame
6. Make targeted changes based on DATA
```

### Current Debug Categories
- `dev_debug_kcc_offload` - Worker physics (CRITICAL - use this first)
- `dev_debug_kcc_visual` - GPU visualizer (capsule, normals, ground, movement)
- `dev_debug_camera_offload` - Camera raycast results
- `dev_debug_engine` - Engine job processing

**All categories have Hz throttling (1-30Hz) - default 5Hz for readability**

### GPU Visualizer (dev_debug_kcc_visual)
**Current State:** Single-layer batched drawing, performance-invisible

**Shows:**
- Capsule spheres (color-coded: green=grounded, yellow=colliding, red=stuck, blue=airborne)
- Hit normals (cyan arrows)
- Ground ray (magenta=hit, purple=miss)
- Movement vectors (green=intended, red=actual)

**When adding new features:**
- ✅ ALWAYS add visualization for new mechanics
- ✅ Add individual toggle for each visual element
- ✅ Use batched drawing (single `batch.draw()` per element type)
- ✅ Test performance impact (should be zero)

### Never Be Shy to Develop Debug Tools
**Philosophy:** Better visualization = faster iteration = better results

**When stuck on an issue:**
1. Add specific visualization for that mechanic
2. Add detailed logging to engine output file
3. Capture session → analyze → understand root cause
4. Fix based on data, not guesses

---

## 📐 Architecture Overview

```
Main Thread (exp_kcc.py)               Worker (engine_worker_entry.py)
┌─────────────────────────┐           ┌──────────────────────────────┐
│ Submit KCC_PHYSICS_STEP │──────────>│ Capsule sweep (2-3 spheres)  │
│ Poll (3ms timeout)      │           │ Step-up detection            │
│ Apply result to Blender │<──────────│ Ground detection             │
│ GPU visualization       │           │ Slope handling               │
│ Platform carry          │           │ Wall slide                   │
│                         │           │ Spatial grid (DDA)           │
│                         │           │ Dynamic mesh physics         │
└─────────────────────────┘           └──────────────────────────────┘
```

**Key Files:**
- `physics/exp_kcc.py` - Main thread wrapper + GPU visualizer
- `engine/engine_worker_entry.py` - Worker physics handler (KCC_PHYSICS_STEP)
- `developer/dev_properties.py` - Debug toggles
- `developer/dev_panel.py` - N-panel UI

**Current Performance:**
- Worker execution: ~100-200µs typical
- Main thread overhead: ~50µs (minimal)
- System smooth when prints disabled
- 30Hz locked timestep

---

## 🔬 Investigation Strategy

**For each issue:**

1. **Reproduce** - Get consistent repro in test scene
2. **Visualize** - Add GPU overlay for the specific mechanic
3. **Log** - Add detailed worker output to engine file
4. **Capture** - Export session log
5. **Analyze** - Read frame-by-frame, identify pattern
6. **Fix** - Make targeted change in worker
7. **Verify** - Compare before/after logs

**Example Output Format:**
```
[KCC F0042] CAPSULE 68 h_tris | clear 1 planes | step ATTEMPTED climb=0.125m
[KCC F0042] GROUND ON_GROUND dist=0.001m normal=(0.00,0.00,1.00)
[KCC F0042] SLOPE angle=12.3° walkable=True slide=False
```

---

## 💡 Solution Guidelines

**When fixing collision issues:**
- Computation MUST stay in worker (no bpy access)
- Test performance impact (log execution time)
- Visualize the change (add to GPU overlay)
- Verify with engine output logs (before/after comparison)

**When adding sphere sweeps:**
- Consider cost: each sweep = N triangle tests
- Spatial grid helps, but more sweeps = more work
- Balance coverage vs performance
- Mid-height sphere likely needed for issue #1

**When adjusting slope handling:**
- Steep slope detection in worker
- `steep_slide_gain` = downward acceleration (default 18.0 m/s²)
- `steep_min_speed` = minimum slide speed (default 2.5 m/s)
- Must prevent uphill velocity, not just add downward force

**When fixing dynamic BVH issues:**
- Dynamic mesh physics computation happens in worker (same as static geometry)
- Platform carry application happens AFTER worker result (main thread)
- Platform rotation = angular velocity applied to character position
- Falling collision = need predictive check, not reactive
- Speed application = velocity delta, not position offset

---


**For each fix:**
- Profile worker execution time (should stay <500µs)
- Add visualization if new mechanic
- Export session logs before/after
- Document in this file

---

## 📊 Current System State

**What's Working:**
- ✅ Core physics offload (~95% computation in worker)
- ✅ Same-frame polling (low latency)
- ✅ Spatial grid acceleration (DDA traversal)
- ✅ Basic collision (feet + head spheres)
- ✅ Ground detection and snapping
- ✅ Step-up on elevated obstacles
- ✅ GPU visualization (performance-invisible)
- ✅ Engine output export system
- ✅ Smooth gameplay when prints disabled


**Philosophy:**
Data-driven development. Visualize everything. Never guess. Test methodically. Respect the engine.
