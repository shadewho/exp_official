# Dynamic Mesh System - Context & Current State

**Last Updated**: 2025-12-14

---

## Core Architectural Principles

### **PRIORITY #1: NEVER MAIN THREAD**
**CRITICAL BOUNDARY**: Physics computation ONLY runs in worker threads.
- Main thread = input/output ONLY (send transforms, apply results)
- Workers = 100% of physics, collision detection, raycasts, spatial queries
- **NEVER** add physics logic to main thread - this is a hard architectural boundary
- If you're considering main thread physics, STOP and find worker solution

### **PRIORITY #2: Performance Per Frame**
Every solution must be optimized for frame-time cost:
- Main thread: Target < 100µs per frame for dynamic mesh handling
- Worker thread: Target < 200µs per KCC physics step (including dynamic meshes)
- Minimize allocations, prefer cache-friendly data structures
- Use spatial acceleration (grids, AABBs) for broad-phase culling

### **PRIORITY #3: Diagnostics & Logging**
**NO BLIND TESTING** - Must have visibility into what's happening:
- Every system needs debug flags and detailed logging
- Use fast buffer logger (`dev_logger.py`) for frame-by-frame diagnostics
- Log positions, velocities, collisions, push forces, platform attachments
- Analyze logs to understand problems before implementing solutions
- Logs exported to `diagnostics_latest.txt` for Claude analysis

---

## Current Architecture

### Main Thread (exp_kcc.py) - ULTRA LIGHTWEIGHT
**What runs here:**
1. **Platform carry via relative position** (lines 668-718)
   - Store player position relative to platform in local space (on landing)
   - Each frame: `worldPos = platform.matrix @ relativePos` (ONE matrix multiply)
   - Cost: ~16 multiplies + 12 adds per frame
   - No velocity tracking needed

2. **Input snapshot** (lines 720-738)
   - Read keyboard state, compute wish direction
   - Package into job data

3. **Job submission to worker** (lines 747-751)
   - Send current state (pos, vel, ground state)
   - Send input (wish_dir, jump, running)
   - Send dynamic transforms (64 bytes per MOVING mesh only)

4. **Same-frame polling** (lines 754-787)
   - Wait for worker result (~100-200µs)
   - Busy-poll first 3 iterations, then 50µs sleeps

5. **Apply worker result** (line 772, method at 448-547)
   - Extract new pos, vel, ground state
   - Identify which mesh player is standing on (`ground_hit_source`)
   - Write to Blender objects

**What does NOT run here:**
- Zero physics computation
- Zero collision detection
- Zero raycasts or spatial queries
- No triangle testing
- No velocity calculations
- No anti-tunneling logic

**Total main thread cost**: < 1% of frame time

---

### Worker Threads (engine_worker_entry.py) - ALL PHYSICS

**Persistent caches (per-worker):**
- `_cached_dynamic_meshes` - Triangle data, local-space grids (sent once)
- `_cached_dynamic_transforms` - Transform matrices + world AABBs (updated when meshes move)

**Per-frame work:**
1. **Transform cache update** (lines 1772-1786)
   - Receive transform updates from main thread (only for moving meshes)
   - Compute world-space AABBs from transforms
   - Store in `_cached_dynamic_transforms[obj_id]`

2. **Build unified mesh list** (lines 1788-1863)
   - Create `unified_dynamic_meshes` from ALL cached transforms
   - Each mesh: `{obj_id, triangles, matrix, inv_matrix, aabb, bounding_sphere, grid}`
   - Stationary meshes use cached transforms (zero main thread cost)

3. **Full KCC physics step** (lines 1865-2500+)
   - Input → Velocity (acceleration)
   - Gravity
   - Jump
   - Horizontal collision (walls/obstacles) - UNIFIED raycast
   - Step-up detection
   - Wall slide
   - Ceiling check
   - Ground detection - UNIFIED raycast
   - Vertical body integrity
   - Slope handling

4. **Unified raycast** (`unified_raycast()` at lines 882-1190)
   - **Phase 1**: Test static grid (3D DDA traversal)
   - **Phase 2**: Test ALL dynamic meshes (AABB cull → sphere cull → grid-accelerated ray testing)
   - Returns closest hit with `source`: `"static"` or `"dynamic"`
   - Returns `obj_id` for dynamic hits (used by platform system)

5. **Return result** (lines 2500+)
   - New pos, vel, ground state
   - `ground_hit_source`: `"static"`, `"dynamic_{obj_id}"`, or `None`
   - Debug data, worker logs

**Total worker cost**: ~100-200µs per frame (including dynamic meshes)

---

## Unified Physics System

**Key principle**: Static and dynamic meshes use IDENTICAL code paths.

**Single raycast function for ALL geometry:**
```python
unified_raycast(ray_origin, ray_direction, max_dist, grid_data, dynamic_meshes)
```

**What's unified:**
- Same ray-triangle intersection (Möller-Trumbore algorithm)
- Same spatial grid acceleration
- Same normal computation
- Same collision response (ground, horizontal, ceiling, step, slide)
- Closest hit wins regardless of source

**Why unity matters:**
- Prevents divergent behavior between static/dynamic
- Easier debugging (one code path to trace)
- Consistent performance characteristics
- No "activation" logic needed

**Logging format:**
```
[GROUND F0042 T1.401s] HIT source=static z=3.10m normal=(0,0,1) | player_z=3.10 tris=12
[GROUND F0043 T1.435s] HIT source=dynamic_12345 z=3.50m normal=(0,0,1) | player_z=3.50 tris=8
[PHYSICS F0042 T1.403s] total=200us | static+dynamic=3 | rays=8 tris=45 | ground=static
```

---

## Current Problems & Diagnosis Needs

### Problem 1: Mesh Tunneling at Medium/High Speeds
**Symptom**: Player clips through dynamic meshes moving at 3+ m/s, especially horizontal movement.

**Why it's happening (hypothesis):**
- Raycasts are point-in-time samples (no motion prediction)
- Fast-moving mesh can "jump over" player between frames
- At 30 fps, 0.033s per frame - mesh moving 5 m/s travels 0.167m per frame
- If player radius is 0.3m, mesh can pass through before raycast detects it

**What we need to diagnose:**
- Log mesh positions frame-by-frame
- Log player position vs mesh AABB each frame
- Log when player is INSIDE mesh AABB (should never happen)
- Compare previous frame mesh position to current frame

**Potential solutions (need data first):**
1. Swept AABB collision (union of prev + current mesh AABB)
2. CCD (continuous collision detection) - raycast along mesh motion path
3. Velocity-based prediction (if mesh moving toward player, extend collision margin)

### Problem 2: Frame Lag or Stuttering on Moving Platforms
**Symptom**: When riding a dynamic platform, movement feels stuttery or laggy.

**Possible causes:**
1. **Timing issue**: Platform carry happens BEFORE physics, but platform moved this frame?
2. **Main thread vs worker**: Platform transform updated on main thread, but worker sees old transform?
3. **Velocity mismatch**: Platform moves, but relative position doesn't update properly?
4. **Frame delay**: Worker computes with frame N data, but platform is already at frame N+1?

**What we need to diagnose:**
- Log platform transform each frame (main thread)
- Log platform transform worker receives (worker side)
- Log player world position before/after platform carry
- Log player relative position to platform
- Compare timestamps: when did platform move vs when did physics compute

**Logging needed:**
```
[PLATFORM F0042 T1.400s MAIN] obj=12345 pos=(10.0,5.0,2.0) matrix=...
[PLATFORM F0042 T1.401s WORKER] obj=12345 received matrix=...
[PLATFORM F0042 T1.402s] player_local=(0.5,0.0,0.1) player_world=(10.5,5.0,2.1)
```

### Problem 3: Horizontal Mesh Impacts Don't Push
**Symptom**: Wall moving horizontally doesn't push player - player tunnels through or gets stuck.

**Why it's happening (hypothesis):**
- Horizontal raycasts fire in player movement direction, not mesh movement direction
- Mesh moves INTO player space, but no collision detected because player isn't moving toward mesh
- Need to detect when mesh AABB overlaps player capsule

**What we need to diagnose:**
- Log horizontal raycast results each frame
- Log mesh AABB vs player capsule overlap
- Log mesh movement direction and speed
- Detect when mesh moves but player doesn't

---

## Logging System Requirements

### Current Logger (`dev_logger.py`)
- Fast in-memory buffer (~1µs per log vs ~1000µs for console print)
- Frequency gating via `dev_debug_master_hz` (1-30 Hz)
- Category toggles: `physics`, `ground`, `horizontal`, `body`, `ceiling`, `dynamic_cache`, etc.
- Export to `diagnostics_latest.txt` on session end

### Needed: Enhanced Dynamic Mesh Logging

**New debug category: `DYNAMIC_MESH_INTERACTION`**

**What to log:**
1. **Mesh state per frame:**
   - Position, velocity, AABB bounds
   - Distance to player
   - Is mesh moving? Speed and direction
   - Is player inside mesh AABB? (red flag)

2. **Collision events:**
   - Ray hit dynamic mesh (which ray: ground, horizontal, ceiling?)
   - Hit distance, normal, position
   - Was hit expected? (player moving toward mesh)
   - Was hit unexpected? (mesh moved into player)

3. **Platform carry:**
   - Landing event (compute relative position)
   - Each frame carry (relative → world transform)
   - Detach event (leave ground or mesh gone)
   - Velocity transfer (if any)

4. **Push/penetration resolution:**
   - Player inside mesh detected
   - Push direction and magnitude
   - Velocity added to resolve

**Log format:**
```
[DYN-MESH F0042 T1.400s] obj=12345 pos=(10.0,5.0,2.0) vel=(2.0,0.0,0.0) speed=2.0m/s
[DYN-MESH F0042 T1.400s] AABB: min=(9.0,4.0,1.5) max=(11.0,6.0,2.5) | player=(10.5,5.0,2.0) INSIDE=False
[DYN-COLLISION F0042 T1.401s] HORIZONTAL HIT obj=12345 dist=0.3m normal=(1,0,0) | expected=True
[DYN-PLATFORM F0042 T1.402s] CARRY obj=12345 rel=(0.5,0.0,0.1) world=(10.5,5.0,2.1)
```

---

## Development Strategy: Diagnosis First

**Current approach is wrong**: We've been implementing solutions without understanding the problem.

**Correct approach:**
1. **Add comprehensive logging** for dynamic mesh interactions
2. **Run test scenarios** with moving meshes at various speeds
3. **Analyze logs** to see exactly when/why tunneling occurs
4. **Identify root cause** from data, not guesses
5. **Design solution** based on actual problem
6. **Implement with logging** to verify fix
7. **Test and iterate** using log data

**Test scenarios needed:**
1. Slow platform (1 m/s) - should work perfectly
2. Medium platform (3 m/s) - may show issues
3. Fast platform (5+ m/s) - likely tunneling
4. Rotating platform - angular velocity transfer
5. Horizontal wall push - moving wall at various speeds
6. Oscillating platform - direction changes

**For each test:**
- Enable all dynamic mesh debug flags
- Export logs after 5-10 seconds
- Read logs to understand what happened
- Compare expected vs actual behavior

---

## Key Files

### Worker-Side Physics
- **`engine_worker_entry.py`** - All physics computation, unified raycast, KCC step
  - Lines 882-1190: `unified_raycast()` - single function for all geometry
  - Lines 1671-2500+: `KCC_PHYSICS_STEP` job handler - full physics simulation
  - Lines 1750-1863: Dynamic mesh transform cache and unified mesh building

### Main Thread (Lightweight)
- **`exp_kcc.py`** - Character controller, job packaging, result application
  - Lines 668-718: Platform carry system (relative position)
  - Lines 747-787: Job submission and same-frame polling
  - Lines 448-547: `_apply_physics_result()` - extract and apply worker results

- **`exp_dynamic.py`** - Dynamic mesh registration, transform updates
  - Main thread maintains list of dynamic meshes
  - Sends triangle data to worker ONCE via broadcast
  - Sends transform updates only when meshes MOVE

### Logging
- **`dev_logger.py`** - Fast buffer logger system
- **`dev_debug_gate.py`** - Frequency gating (Hz control)
- **`dev_properties.py`** - Debug property definitions
- **`dev_panel.py`** - Developer Tools UI panel

---

## Technical Notes

### Why Multi-Worker Breaks Per-Worker State
- 4 worker threads (W0-W3) share KCC job queue
- Load balancing distributes jobs: Frame 0 → W0, Frame 1 → W1, Frame 2 → W2, etc.
- If each worker tracks "previous frame" state, they never see their own previous frame
- **Solution**: State must be sent WITH job data OR tracked on main thread

### Transform Cache Architecture
Worker `_cached_dynamic_transforms[obj_id]`:
- Stores `(matrix_4x4, world_aabb)` for each dynamic mesh
- Updated only when main thread sends new transform
- Stationary meshes = zero main thread cost (worker uses cached data)
- ALL cached meshes tested every frame (no "activation" needed)

### Spatial Grid Acceleration
- Static: One global grid for all static geometry
- Dynamic: Per-mesh local-space grid (8x8x8 cells)
- Ray transformed to local space → grid traversal → O(cells) instead of O(tris)
- Cached grids built ONCE when mesh triangles received

### Platform Carry System
**Main thread only** (not physics):
1. **Landing**: `worldPos` → `localPos` via `inv_matrix @ pos` (one-time)
2. **Each frame**: `localPos` → `worldPos` via `matrix @ pos` (one matrix multiply)
3. **Detach**: Clear when leave ground or mesh gone

No velocity tracking, no timing issues, frame-perfect positioning.

---

## Immediate Next Steps

### 1. Enhanced Logging Implementation
Add `DYNAMIC_MESH_INTERACTION` logging category:
- Track mesh positions, velocities, AABBs each frame
- Log all dynamic mesh ray hits (source, distance, normal)
- Log platform carry events (attach, carry, detach)
- Log player vs mesh AABB overlap detection

### 2. Diagnostic Test Suite
Create test scene with:
- Slow platform (1 m/s horizontal)
- Medium platform (3 m/s horizontal)
- Fast platform (5-8 m/s horizontal)
- Rotating platform (angular motion)
- Oscillating wall (back and forth)

### 3. Log Analysis Protocol
For each test:
1. Enable dynamic mesh debug flags
2. Run for 10 seconds
3. Export to `diagnostics_latest.txt`
4. Analyze frame-by-frame to find exact moment of failure
5. Identify pattern: timing issue? velocity issue? position issue?

### 4. Data-Driven Solution Design
Based on log analysis, identify:
- **If tunneling**: When does mesh position jump past player?
- **If stuttering**: Is there frame delay in transform updates?
- **If stuck**: Is player inside mesh AABB? How did they get there?

Only AFTER understanding problem from logs, design solution.

---

## Anti-Patterns to Avoid

### ❌ Don't: Implement Without Logging
- Never add features without diagnostic visibility
- Can't debug what you can't see

### ❌ Don't: Guess at Solutions
- "Maybe we need swept AABB?" - based on what data?
- "Maybe it's a timing issue?" - prove it with logs first

### ❌ Don't: Add Main Thread Physics
- **NEVER** - this is a hard boundary
- If solution requires main thread, find different solution

### ❌ Don't: Blind Performance Optimization
- Measure first, optimize second
- Log timing data to identify actual bottlenecks

### ✅ Do: Log Everything
- Frame-by-frame positions, velocities, collisions
- Timestamp everything
- Export and analyze before implementing

### ✅ Do: Test Incrementally
- One scenario at a time
- One speed at a time
- Isolate variables

### ✅ Do: Worker-Side Solutions
- ALL physics computation in workers
- Main thread = I/O only

### ✅ Do: Unified Approach
- Static and dynamic use same code paths
- Prevents divergent behavior

---

**Status**: Ready for diagnostic logging implementation and systematic problem analysis.
