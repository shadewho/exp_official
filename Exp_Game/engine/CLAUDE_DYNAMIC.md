# Dynamic Mesh Physics Offload System

**Status:** Planning Phase
**Last Updated:** 2025-12-09
**Critical Goal:** Unify static and dynamic proxy mesh physics in worker, free main thread

---


## 🎯 PRIMARY OBJECTIVES

### 1. Unified Physics DNA
**Goal:** Static and dynamic proxy meshes must share IDENTICAL physics logic.

```
SAME collision detection code
SAME step-up logic
SAME wall slide logic
SAME body integrity checks
SAME ceiling detection
SAME steep slope handling
SAME ground detection
```

**Only difference:** Dynamic meshes move the character (platform carry), static meshes don't.

### 2. Free the Main Thread
**Current (BAD):**
- Main thread: BVH building, collision detection, push-out computation
- Worker: Static physics only
- Result: Main thread doing heavy work, dynamic physics incomplete

**Target (GOOD):**
- Main thread: Extract matrices (16 floats), submit job, poll result, apply platform carry
- Worker: ALL collision detection for both static AND dynamic
- Result: Main thread <50µs overhead, worker handles everything

### 3. Performance & Observability
- Worker execution: <300µs total (static + dynamic combined)
- Zero frame latency (same-frame polling)
- Complete logging pipeline for debugging
- Frame-accurate timeline export
- Performance breakdown by collision type

---

## 🏗️ ARCHITECTURE OVERVIEW

### Cache-and-Transform Pattern

**Phase 1: One-Time Caching (Startup)**
```
Main Thread                           Worker Process
    |                                      |
    | Extract triangles (LOCAL SPACE)     |
    | mesh.loop_triangles → [(v0,v1,v2)] |
    |                                      |
    +---> CACHE_DYNAMIC_MESH job -------->+
          {obj_id, triangles, radius}     |
                                           |
                                    Store in:
                                    _cached_dynamic_meshes[obj_id] = {
                                        "triangles": [...],
                                        "radius": float
                                    }
```

**Cost:** One-time per object (~1ms per 100 triangles)
**Frequency:** Once at startup or when mesh registered

---

**Phase 2: Per-Frame Transform Update (Lightweight)**
```
Main Thread (every frame)             Worker Process
    |                                      |
    | Get matrix_world from Blender       |
    | Serialize to 16-float tuple         |
    |                                      |
    +---> KCC_PHYSICS_STEP job --------->+
          {                                |
              ...existing fields...        |
              "dynamic_transforms": {      |
                  obj_id: (16 floats)      | Transform triangles:
              }                            | world_tri = local_tri × matrix
          }                                |
                                           | For each cached mesh:
                                           |   for tri in cached[obj_id]:
                                           |       v0_w = matrix × v0_local
                                           |       v1_w = matrix × v1_local
                                           |       v2_w = matrix × v2_local
```

**Cost per frame:**
- 10 meshes × 64 bytes = 640 bytes serialization (~5µs)
- 10 meshes × 50 triangles × transform = ~7.5µs
- **Total: <15µs overhead**

---

**Phase 3: Unified Collision Testing**
```
Worker: test_collision_unified(ray_origin, ray_dir, max_dist)
    |
    +---> Test Static Grid (DDA traversal)
    |     for cell in grid_cells:
    |         for tri in cell_triangles:
    |             hit = ray_triangle_intersect(...)
    |             track_closest(hit)
    |
    +---> Test Dynamic Meshes (transformed triangles)
          for obj_id, tris in transformed_dynamic:
              for tri in tris:
                  hit = ray_triangle_intersect(...)  # SAME FUNCTION!
                  track_closest(hit)
    |
    +---> Return best_hit (static or dynamic)
```

**All physics checks use this unified function:**
- Body integrity ray (feet → head vertical)
- Horizontal capsule sweep (3 rays: feet, mid, head)
- Ground detection (raycast down)
- Ceiling check (raycast up)
- Step-up detection (elevated forward ray)
- Wall slide (tangent projection)

**Result:** Dynamic and static are physically IDENTICAL.

---

## 📊 LOGGING & OBSERVABILITY FRAMEWORK

### Critical Principle: Complete Transparency

**Every dynamic mesh interaction must be logged with:**
1. Frame number
2. Timestamp
3. Object ID/name
4. Collision type (body/ground/horizontal/ceiling)
5. Hit distance
6. Hit normal
7. Source type (static vs dynamic)

### Log Categories (Master Hz Controlled)

#### 1. **DYN-CACHE** - Dynamic Mesh Caching
**Purpose:** Track one-time registration of dynamic meshes to worker

**Toggle:** `dev_debug_dynamic_cache`
**Hz:** N/A (one-time events only)
**Output File:** `diagnostics_latest.txt`

**Format:**
```
[DYN-CACHE] obj={name} id={id} tris={count} radius={r:.2f}m cached
[DYN-CACHE] obj=Platform_01 id=12345 tris=48 radius=2.35m cached
```

**When logged:**
- On first activation of dynamic mesh
- On mesh re-registration (if geometry changes)

---

#### 2. **DYN-XFORM** - Per-Frame Transform Updates
**Purpose:** Track which dynamic meshes are active and data sent to worker

**Toggle:** `dev_debug_dynamic_transform`
**Hz:** 1-30 Hz (default 1 Hz to avoid spam)
**Output File:** `diagnostics_latest.txt`

**Format:**
```
[DYN-XFORM F#### T##.###s] active={count} bytes={size} objects=[id1,id2,...]
[DYN-XFORM F0042 T1.400s] active=3 bytes=192 objects=[12345,12346,12347]
```

**When logged:**
- Every frame when dynamic meshes are active
- Shows which objects are within activation distance
- Byte count = 64 × active_count

---

#### 3. **DYN-PHYS** - Dynamic Collision Results
**Purpose:** Track worker-side collision tests against dynamic meshes

**Toggle:** `dev_debug_dynamic_physics`
**Hz:** 1-30 Hz (default 5 Hz)
**Output File:** `diagnostics_latest.txt`

**Format:**
```
[DYN-PHYS F#### T##.###s] obj={name} type={test} hit={bool} dist={d:.3f}m normal=({x:.2f},{y:.2f},{z:.2f})
[DYN-PHYS F0042 T1.400s] obj=Platform_01 type=ground hit=True dist=0.025m normal=(0.00,0.00,1.00)
[DYN-PHYS F0043 T1.433s] obj=MovingWall type=horizontal hit=True dist=0.180m normal=(-1.00,0.00,0.00)
```

**Test types:**
- `body` - Body integrity ray
- `ground` - Ground detection
- `horizontal` - Capsule sweep collision
- `ceiling` - Upward collision
- `step` - Step-up attempt
- `slide` - Wall slide

---

#### 4. **PHYS-UNIFIED** - Combined Static+Dynamic Stats
**Purpose:** Per-frame summary showing total collision work

**Toggle:** `dev_debug_physics_unified`
**Hz:** 1-30 Hz (default 5 Hz)
**Output File:** `diagnostics_latest.txt`

**Format:**
```
[PHYS-UNIFIED F#### T##.###s] static_tris={n} dynamic_tris={m} total_tests={k} time={t}µs breakdown=[s_time={a}µs d_time={b}µs]
[PHYS-UNIFIED F0042 T1.400s] static_tris=124 dynamic_tris=48 total_tests=172 time=145µs breakdown=[s_time=98µs d_time=47µs]
```

**Breakdown:**
- `static_tris` - Triangles tested from static grid
- `dynamic_tris` - Triangles tested from dynamic meshes
- `total_tests` - Combined collision tests
- `time` - Total worker computation time
- `s_time` - Time spent on static collision
- `d_time` - Time spent on dynamic collision

---

#### 5. **PHYS-BODY** (Enhanced) - Embedding Detection
**Purpose:** Detect mesh embedding, show static vs dynamic source

**Toggle:** `dev_debug_physics_body_integrity` (existing, enhanced)
**Hz:** 1-30 Hz (default 5 Hz)
**Output File:** `diagnostics_latest.txt`

**Format (NEW):**
```
[PHYS-BODY F#### T##.###s] EMBEDDED source={static|dynamic} obj={name} dist={d:.3f}m pct={p:.1f}%
[PHYS-BODY F0042 T1.400s] EMBEDDED source=dynamic obj=Platform_01 dist=0.125m pct=6.9%
[PHYS-BODY F0043 T1.433s] CLEAR height={h:.2f}m
```

**Shows:**
- Whether embedding is from static or dynamic mesh
- Which specific object (if dynamic)
- Penetration depth and percentage of capsule height

---

#### 6. **KCC** (Enhanced) - Main Physics Summary
**Purpose:** Frame summary line showing dynamic mesh activity

**Toggle:** `dev_debug_kcc_offload` (existing, enhanced)
**Hz:** 1-30 Hz (default 5 Hz)
**Output File:** `diagnostics_latest.txt`

**Format (ENHANCED):**
```
[KCC F#### T##.###s] {state} pos=({x},{y},{z}) step={bool} | {time}µs {rays}rays {tris}tris | dynamic_active={count} dynamic_hits={hits}
[KCC F0042 T1.400s] GROUND🟢 pos=(10.5,5.2,3.0) step=False | 145µs 4rays 172tris | dynamic_active=3 dynamic_hits=2
```

**New fields:**
- `dynamic_active` - How many dynamic meshes were tested
- `dynamic_hits` - How many dynamic collisions occurred

---

#### 7. **DYN-PERF** - Performance Breakdown
**Purpose:** Detailed timing for dynamic mesh operations

**Toggle:** `dev_debug_dynamic_performance`
**Hz:** 1-30 Hz (default 1 Hz)
**Output File:** `diagnostics_latest.txt`

**Format:**
```
[DYN-PERF F#### T##.###s] transform={t1}µs test_body={t2}µs test_capsule={t3}µs test_ground={t4}µs obj_breakdown=[obj:{time}µs, ...]
[DYN-PERF F0042 T1.400s] transform=7µs test_body=12µs test_capsule=18µs test_ground=10µs obj_breakdown=[12345:15µs, 12346:12µs, 12347:10µs]
```

**Breakdown:**
- `transform` - Time to transform all dynamic triangles
- `test_body` - Time for body integrity ray vs dynamic
- `test_capsule` - Time for horizontal sweep vs dynamic
- `test_ground` - Time for ground detection vs dynamic
- `obj_breakdown` - Per-object timing

---

### Additional Log File: Dynamic Mesh Timeline

**File:** `C:\Users\spenc\Desktop\engine_output_files\dynamic_timeline_latest.txt`

**Purpose:** Separate detailed log just for dynamic mesh events (optional, for deep debugging)

**Toggle:** `dev_export_dynamic_timeline`
**Format:** Same categories as above, but ONLY dynamic-related events

**When to use:**
- Debugging specific dynamic mesh behavior
- Analyzing platform carry synchronization
- Profiling dynamic collision performance
- When diagnostics_latest.txt is too noisy

---

## 🔬 DEBUGGING WORKFLOW

### Standard Workflow: Unified Diagnostics

```
1. Enable debug categories:
   - dev_debug_kcc_offload (main summary)
   - dev_debug_dynamic_physics (collision details)
   - dev_debug_physics_unified (combined stats)

2. Set Master Hz to 5 Hz (readable rate)

3. Enable "Export Diagnostics Log"

4. Play game session

5. Stop → auto-exports to diagnostics_latest.txt

6. Read file in Claude:
   - Search for "DYN-PHYS" to see dynamic collisions
   - Search for "PHYS-UNIFIED" to see combined stats
   - Search for "EMBEDDED" to find mesh penetration
   - Check KCC lines for dynamic_hits count

7. Analyze frame-by-frame timeline
```

### Deep Debugging: Dynamic Timeline

```
1. Enable ALL dynamic categories:
   - dev_debug_dynamic_cache
   - dev_debug_dynamic_transform
   - dev_debug_dynamic_physics
   - dev_debug_dynamic_performance

2. Enable "Export Dynamic Timeline" (separate file)

3. Set Master Hz to 30 Hz (every frame)

4. Play short test session (5-10 seconds)

5. Read dynamic_timeline_latest.txt:
   - Frame-by-frame dynamic mesh activity
   - Performance breakdown per object
   - Transform updates timing
   - Collision test results

6. Compare with diagnostics_latest.txt for full picture
```

---

## ⚡ PERFORMANCE METRICS & TARGETS

### Worker Execution Time Budget

**Current (Static Only):** ~100-200µs
**Target (Static + Dynamic):** <300µs
**Budget for Dynamic:** <100µs additional

**Breakdown:**
```
Static collision:        100-200µs (existing)
Dynamic transform:       5-10µs (10 meshes × 50 tris)
Dynamic collision:       30-50µs (body + capsule + ground)
Overhead (activation):   5-10µs (filtering active meshes)
─────────────────────────────────────────────────────
Total:                   140-270µs (well under 300µs target)
```

### Per-Frame Data Size

**Static grid:** 0 bytes (cached once at startup)
**Dynamic meshes:**
- 1 mesh: 64 bytes (matrix)
- 5 meshes: 320 bytes (typical active count)
- 10 meshes: 640 bytes (max realistic)
- 20 meshes: 1,280 bytes (stress test)

**Serialization overhead:** ~5µs for 10 meshes

### Activation Gating (Already Working)

**Distance gating ensures only nearby meshes are active:**
- Typical scene: 20 total dynamic meshes
- Activation radius: `register_distance` (default 15m)
- Typical active: 3-5 meshes near player
- Far meshes: Zero cost (not sent, not tested)

**Hysteresis:** 10% margin prevents thrashing at boundary

---

## 🎯 SUCCESS METRICS

### Physics Parity (Goal: 100%)
- [ ] Body integrity ray detects dynamic mesh embedding
- [ ] Step-up works on dynamic platforms/stairs
- [ ] Wall slide works on dynamic walls (same as static)
- [ ] Ceiling check blocks on dynamic ceilings
- [ ] Steep slope sliding works on dynamic ramps
- [ ] Ground detection snaps to dynamic surfaces
- [ ] All collision feels IDENTICAL to static

### Performance (Goal: <300µs Worker, 30Hz Locked)
- [ ] Worker execution <300µs with 5 active dynamic meshes
- [ ] Worker execution <400µs with 10 active dynamic meshes
- [ ] Main thread overhead <50µs (matrix extraction + job submit)
- [ ] Zero stuttering at 30Hz
- [ ] Smooth platform riding (no jitter/sliding)

### Code Quality (Goal: Single Implementation)
- [ ] Delete `_check_dynamic_collision()` from main thread
- [ ] Delete `_handle_dynamic_movers()` from main thread
- [ ] Delete `exp_bvh_local.py` (LocalBVH wrapper)
- [ ] Single unified collision function in worker
- [ ] Static and dynamic use identical code paths
- [ ] Maintenance burden reduced 50%+

### Observability (Goal: Complete Transparency)
- [ ] Every dynamic collision logged with source type
- [ ] Frame-by-frame timeline exported
- [ ] Performance breakdown per object
- [ ] Embedding detection shows static vs dynamic
- [ ] Master Hz control works for all categories
- [ ] Diagnostics export complete and readable

---

## 🛠️ IMPLEMENTATION PHASES

### Phase 1: Worker Infrastructure (No Breaking Changes)
**Goal:** Worker can handle dynamic meshes (parallel to existing system)

**Tasks:**
1. Add `_cached_dynamic_meshes` global dict
2. Add `CACHE_DYNAMIC_MESH` job handler
3. Add triangle transform helper function
4. Add `test_collision_unified()` function
5. Extend body integrity ray to test dynamic
6. Extend capsule sweep to test dynamic
7. Extend ground detection to test dynamic
8. Extend ceiling check to test dynamic
9. Extend step-up to test dynamic
10. Extend wall slide to test dynamic

**Files modified:**
- `engine/engine_worker_entry.py` (~200 lines added)

**Testing:**
- Worker logs show dynamic collision detection
- No changes to main thread yet
- Can verify worker capability in isolation

---

### Phase 2: Main Thread Cutover (The Switch)
**Goal:** Main thread uses worker for dynamic collision

**Tasks:**
1. Modify `update_dynamic_meshes()` in exp_dynamic.py:
   - Replace LocalBVH building with triangle extraction
   - Add cache registration (CACHE_DYNAMIC_MESH job)
   - Keep velocity computation (unchanged)
   - Keep activation system (unchanged)

2. Modify `step()` in exp_kcc.py:
   - Add `dynamic_transforms` dict to job_data
   - Serialize active mesh matrices

3. Delete collision methods in exp_kcc.py:
   - Delete `_check_dynamic_collision()` (lines 655-797)
   - Delete `_handle_dynamic_movers()` (lines 432-521)
   - Remove calls to both methods

4. Keep platform carry:
   - Lines 599-622 stay UNCHANGED
   - Velocity application remains on main thread

**Files modified:**
- `physics/exp_dynamic.py` (~50 lines modified)
- `physics/exp_kcc.py` (~200 lines deleted, ~20 lines added)

**Testing:**
- Dynamic collision now handled by worker
- Main thread freed up
- Physics parity achieved

---

### Phase 3: Cleanup & Developer System
**Goal:** Remove dead code, add full observability

**Tasks:**
1. Delete `physics/exp_bvh_local.py` entirely
2. Remove LocalBVH imports from all files
3. Add debug properties to `developer/dev_properties.py`
4. Add log categories to `developer/dev_logger.py`
5. Add UI toggles to `developer/dev_panel.py`
6. Add worker logging for all dynamic collision events
7. Test Master Hz control on all new categories

**Files modified:**
- `physics/exp_bvh_local.py` (DELETE)
- `developer/dev_properties.py` (~40 lines added)
- `developer/dev_logger.py` (~10 lines added)
- `developer/dev_panel.py` (~30 lines added)
- `engine/engine_worker_entry.py` (add logging calls)

**Testing:**
- All debug categories work
- Diagnostics export shows complete picture
- No performance impact from logging

---

### Phase 4: Validation & Optimization
**Goal:** Verify performance, fix any issues

**Tasks:**
1. Profile worker execution time
2. Test with 1, 5, 10, 20 active dynamic meshes
3. Verify 30Hz lock maintained
4. Test platform carry smoothness
5. Export diagnostics, analyze timeline
6. Optimize any bottlenecks found

**Success criteria:**
- Worker <300µs with 5 meshes
- Worker <400µs with 10 meshes
- Zero stuttering
- Smooth platform riding

---

## 🔍 SYNCHRONIZATION & LATENCY

### Same-Frame Polling (Zero Latency)

**Current architecture (KEEP THIS):**
```
Frame N:
  1. Main thread submits KCC_PHYSICS_STEP with dynamic_transforms
  2. Worker receives job, transforms dynamic triangles
  3. Worker computes full physics (static + dynamic unified)
  4. Main thread polls with 3ms timeout (~200µs typical)
  5. Main thread receives result SAME FRAME
  6. Main thread applies result + platform carry

Result: Zero frame offset
```

**Why this matters:**
- Platform carry applied immediately after collision detection
- Character position synced with moving platform
- No 1-frame latency = smooth riding

### Transform Synchronization

**Matrices sent WITH physics job (not separate):**
```python
# This ensures worker sees current-frame positions
job_data = {
    "pos": current_position,
    "vel": current_velocity,
    "dynamic_transforms": {
        obj_id: current_matrix_world  # THIS FRAME'S matrix
    }
}
```

**Critical:** Do NOT send transforms in separate job (would cause frame offset)

### Activation Latency (Acceptable)

**Activation system has 1-frame latency (by design):**
```
Frame N: Submit DYNAMIC_MESH_ACTIVATION job
Frame N: Use activation state from Frame N-1 (acceptable)
Frame N+1: Apply activation result from Frame N
```

**Why this is OK:**
- Activation is distance-based (smooth hysteresis)
- Player can't move fast enough to cause issues
- 1-frame delay = 33ms at 30Hz (imperceptible)

---

## 📁 FILE STRUCTURE

### Files Modified
```
engine/
  engine_worker_entry.py      - Core physics implementation (HEAVY CHANGES)
  CLAUDE_DYNAMIC.md           - This file (NEW)

physics/
  exp_kcc.py                  - Job submission + cleanup (MODERATE CHANGES)
  exp_dynamic.py              - Triangle extraction + caching (MODERATE CHANGES)
  exp_bvh_local.py            - LocalBVH wrapper (DELETE ENTIRE FILE)

developer/
  dev_properties.py           - New debug toggles (LIGHT CHANGES)
  dev_logger.py               - New log categories (LIGHT CHANGES)
  dev_panel.py                - New UI toggles (LIGHT CHANGES)
```

### New Log Files (Optional)
```
C:\Users\spenc\Desktop\engine_output_files\
  diagnostics_latest.txt      - Main diagnostics (ENHANCED)
  dynamic_timeline_latest.txt - Dynamic-only timeline (NEW, OPTIONAL)
```

---

## 💡 DESIGN PRINCIPLES

### 1. Unified Physics DNA
Static and dynamic share IDENTICAL collision code. The ONLY difference is where triangles come from (cached grid vs transformed cache).

### 2. Cache Once, Transform Lightweight
Triangles cached in local space once. Per-frame: send 64 bytes, transform in worker. Never send full geometry every frame.

### 3. Same-Frame Results
Zero frame latency. Transforms sent WITH physics job, result returned same frame, applied immediately.

### 4. Complete Observability
Every collision logged with source type. Frame-by-frame timeline. Performance breakdown. Zero guesswork.

### 5. Performance Budget Discipline
Stay under 300µs worker time. Distance gate active meshes. Profile and optimize continuously.

### 6. Surgical Changes
Keep what works (activation, velocity, platform carry). Replace only collision detection. Minimal disruption.

---

## 🚨 CRITICAL REMINDERS

1. **Never send bpy objects to worker** - Serialize to tuples/dicts only
2. **Always use fast buffer logger in game loop** - Never print() during gameplay
3. **Test with diagnostics export enabled** - Capture full timeline for debugging
4. **Respect Master Hz control** - Don't spam logs, keep output readable
5. **Profile worker time continuously** - Watch for performance regression
6. **Verify same-frame polling** - Platform carry must be smooth
7. **Distance gate activation** - Don't test far meshes, respect register_distance
8. **Cache triangles in local space** - Rigid bodies = no re-caching needed

---

## 📖 RELATED DOCUMENTATION

- **Main game docs:** `Exp_Game/CLAUDE_GAME.md`
- **Logger system:** `Exp_Game/developer/CLAUDE_LOGGER.md`
- **Engine context:** `Exp_Game/engine/CLAUDE_ENGINE_CONTEXT.md`
- **Implementation plan:** `C:\Users\spenc\.claude\plans\luminous-strolling-parnas.md`

---

**Philosophy:** Dynamic and static are the same. One unified physics system. Complete transparency through logging. Main thread freed. Worker handles everything. Zero compromises.
