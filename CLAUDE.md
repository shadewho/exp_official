# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL: Development Workflow

**ALWAYS MAKE CHANGES TO:** `C:\Users\spenc\Desktop\Exploratory\addons\Exploratory`

**NEVER EDIT:** `C:\Users\spenc\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\Exploratory`

The AppData location is where Blender loads the addon from, but it's NOT the development directory. The user has a custom install script that copies from Desktop → AppData.

**Workflow:**
1. Make ALL code changes to Desktop version
2. User runs install script (copies Desktop → AppData)
3. User reloads Blender to test

---

## 🚨 ENGINE-FIRST MANDATE (CRITICAL)

**The engine is the HEART of Exploratory - a companion to the modal that takes computational stress off the main thread.**

### THE NEW WAY: Engine Offloading is REQUIRED

**When implementing ANY new feature or task:**
1. ✅ **ALWAYS check if work can be offloaded to the engine FIRST**
2. ✅ **Default to engine offload unless there's a specific reason not to**
3. ✅ **The goal is to FREE the main thread for smooth modal operation**
4. ✅ **Engine offload is NOT optional - it's the architecture standard**

**Why this matters:**
- Main thread must stay responsive for Blender's modal operator
- Python GIL limits threading - multiprocessing bypasses it completely
- Smooth gameplay requires main thread only doing bpy writes and coordination
- Heavy computation (distance checks, raycasts, pathfinding, etc.) MUST be offloaded

**Before starting ANY task, ask:**
- "Can this computation run without bpy access?"
- "Would this benefit from running in parallel?"
- "Will this block the main thread?"

**If YES to any → Use the engine.**

### Engine Architecture

```
Main Thread (Blender Modal)          Worker Processes (4 cores)
┌─────────────────────────┐         ┌──────────────────────────┐
│  ExpModal               │         │  Worker 0-3              │
│  ├─ Snapshot data  ────►├────────►│  ├─ Heavy computation    │
│  ├─ Submit job          │         │  ├─ NO bpy access!       │
│  ├─ Poll results   ◄────┤◄────────┤  └─ Return pickled data  │
│  └─ Apply to Blender    │         └──────────────────────────┘
└─────────────────────────┘
```

**Files:** `Exp_Game/engine/`
- `engine_core.py` - Main thread manager
- `engine_worker_entry.py` - Worker process handler (NO bpy imports!)
- `engine_types.py` - Job/Result data structures

**Current Production Workloads:**
- `KCC_PHYSICS_STEP` - Full character physics computation
- `CAMERA_OCCLUSION_FULL` - Camera raycast occlusion
- `CULL_BATCH` - Performance culling (1000+ objects)
- `DYNAMIC_MESH_ACTIVATION` - Dynamic mesh distance gating
- `INTERACTION_CHECK_BATCH` - Proximity/collision checks

---

## 🔥 ENGINE HEALTH & READINESS (ABSOLUTELY CRITICAL)

**PHILOSOPHY:** The engine is the BOSS. The modal is the SERVANT. The engine must be 100% ready before the modal starts ANYTHING. This is non-negotiable.

### ✅ ENGINE-FIRST STARTUP SYSTEM (IMPLEMENTED)

**STATUS:** Fully operational 5-step startup gate ensures engine is completely ready before game starts.

**The Startup Sequence:**
```
======================================================================
  GAME STARTUP SEQUENCE - ENGINE FIRST
======================================================================

[STARTUP 1/5] Spawning engine workers...
[STARTUP 2/5] Verifying workers alive...
[STARTUP 3/5] Verifying worker responsiveness (PING check)...
[STARTUP 4/5] Caching spatial grid in workers...
[STARTUP 5/5] Final readiness check (lock-step synchronization)...

======================================================================
  ENGINE READY - MODAL STARTING
======================================================================
```

**What Happens at Each Step:**

1. **Worker Spawn** (`engine_core.py:start()`)
   - Spawns 4 worker processes
   - Creates job/result queues
   - Verifies all workers started successfully
   - **ABORT if any worker fails to spawn**

2. **Alive Check** (`engine_core.py:is_alive()`)
   - Confirms all worker processes are running
   - Checks process.is_alive() for each worker
   - **ABORT if any worker is dead**

3. **PING Verification** (`engine_core.py:wait_for_readiness()`)
   - Submits PING jobs to all workers
   - Waits for responses (confirms workers are processing jobs)
   - Timeout: 5 seconds
   - **ABORT if workers don't respond**

4. **Grid Cache Verification** (`engine_core.py:verify_grid_cache()`)
   - Submits 8 CACHE_GRID jobs (2x worker count for coverage)
   - **Tracks UNIQUE workers that confirm** (not just result count)
   - Uses worker_id field to verify each of the 4 workers cached the grid
   - Timeout: 5 seconds
   - **ABORT if not all 4 workers confirm cache**
   - This prevents the "grid not cached in worker" bug where jobs go to uncached workers

5. **Final Health Check** (`engine_core.py:get_full_readiness_status()`)
   - Comprehensive verification of all systems
   - Checks: running, workers_alive, ping_verified, grid_cached, health_passed
   - **ABORT if ANY check fails**

**Debug Output:** Enable with `dev_startup_logs` property in Developer Tools panel.

**Files:**
- `Exp_Game/modal/exp_modal.py` - 5-step startup gate (lines ~840-930)
- `Exp_Game/engine/engine_core.py` - Readiness verification methods
- `Exp_Game/developer/dev_properties.py` - dev_startup_logs property

---

### 🔧 Worker ID Tracking System

**CRITICAL:** All engine results now include `worker_id` field to track which worker processed each job.

**Why This Matters:**
- Workers share a job queue - any worker can grab any job
- Grid cache verification MUST ensure each unique worker has the cache
- Without tracking, one worker could process all CACHE_GRID jobs while others remain uncached
- Camera/physics jobs sent to uncached workers would fail silently

**Implementation:**
1. `engine_types.py` - EngineResult includes `worker_id: int` field
2. `engine_worker_entry.py` - Workers return `worker_id` in result dict
3. `engine_core.py:poll_results()` - Extracts `worker_id` when converting dict to EngineResult
4. `engine_core.py:verify_grid_cache()` - Uses set-based tracking to verify unique workers

**Example:**
```python
confirmed_workers = set()  # Track which workers confirmed
for result in results:
    if result.job_type == "CACHE_GRID":
        confirmed_workers.add(result.worker_id)
# Only pass if confirmed_workers == {0, 1, 2, 3}
```

---

### ⚠️ Current Issues (Still Need Attention)

⚠️ **Performance Inconsistency**
- Game startup sometimes smooth, sometimes sluggish
- Worker job completion times vary unpredictably
- Occasional job stalls/timeouts (especially camera occlusion)
- Camera timeout recovery works but shouldn't be needed so often

⚠️ **Job Allocation Not Optimized**
- Workers share a single job queue (no load balancing)
- Jobs may pile up on one worker while others idle
- No prioritization of critical jobs (physics > culling)
- Risk of overloading workers with too many simultaneous jobs

⚠️ **Worker Health Monitoring Limited**
- No detection of crashed/hung workers during gameplay
- No automatic worker recovery
- Failed jobs sometimes leave state inconsistent

### 🚨 Required Improvements (Before Adding More Offload)

**CRITICAL - DO THESE BEFORE ADDING NEW JOB TYPES:**

1. **Job Allocation Optimization**
   - Implement round-robin or least-loaded worker selection
   - Track pending jobs per worker to prevent overload
   - Add job priority system (physics = highest, culling = lower)
   - Limit max pending jobs per worker (e.g., 2-3 max)

2. **Worker Health Monitoring (Runtime)**
   - Periodic heartbeat from workers during gameplay
   - Detect and log crashed/hung workers
   - Automatic worker restart on failure
   - Graceful degradation if worker count drops

3. **Performance Profiling**
   - Track job completion time distribution
   - Identify slow job types
   - Log worker utilization percentage
   - Detect queue buildup early

4. **Overload Protection Per-System**
   - Each system must handle `submit_job()` returning `None` (queue full)
   - Implement fallbacks for rejected jobs (use cached data, skip frame)
   - Add per-system throttling to prevent engine flooding

### 🎯 Engine Philosophy: Companion to Modal

**The engine exists to serve the modal, not compete with it:**
- **Engine = Boss, Modal = Servant** - Engine proves readiness, modal waits
- **Main thread orchestrates, workers compute** - Offload everything possible
- **Modal stays smooth (30Hz locked), workers absorb variance** - Never block main thread
- **Failed jobs must not break game state** - Always have fallbacks
- **Engine should be invisible to the player when working correctly** - No stutters, no lag

**CRITICAL LESSON: Debug Prints Must Never Affect Behavior**
- Camera system had bugs hidden by debug print timing (accidentally adding delays)
- Symptoms: Works perfectly with debug ON, breaks with debug OFF
- Root causes found: 1) Frame ordering (camera before physics), 2) CPU starvation (polling too aggressive)
- Solution: Explicit synchronization + pre-poll delays → behavior identical regardless of debug state
- **Rule: If system works differently with prints on/off, there's a timing bug - fix the architecture, don't rely on prints**

**Before offloading a new system, ask:**
- ✅ **Is the engine healthy enough to handle it?** Check current load
- ✅ **Will this job type overload workers?** Profile execution time
- ✅ **What happens if this job fails/times out?** Implement fallback
- ✅ **Does this genuinely free the main thread?** Measure impact
- ✅ **Can workers handle this without bpy?** No Blender objects allowed
- ✅ **Does it work identically with debug on/off?** Test both - behavior must be identical

---

## ⚠️ PHYSICS SYSTEM (ENGINE-OFFLOADED - ACTIVE DEVELOPMENT)

**Status:** Physics computation fully offloaded to worker engine. Core mechanics functional but slope handling and smoothness need continued refinement.

### Architecture: Full Physics Offload

**Worker computes ENTIRE physics step:**
1. Input → velocity acceleration (with steep slope blocking)
2. Gravity
3. Jump
4. Horizontal collision (3D DDA spatial grid)
5. Step-up detection
6. Wall slide
7. Ceiling check
8. Ground detection
9. Steep slope sliding (strong downward force)

**Main thread is THIN:**
- Apply worker result
- Dynamic mesh collision (BVH raycasts - requires bpy)
- Platform carry (position offset)
- Write position to Blender

**Key Files:**
- `Exp_Game/physics/exp_kcc.py` - Main thread coordinator (thin)
- `Exp_Game/engine/engine_worker_entry.py` - Worker physics handler (KCC_PHYSICS_STEP)

### Same-Frame Polling Pattern

Physics uses **same-frame polling** to eliminate input latency:
```python
# Submit job
job_id = engine.submit_job("KCC_PHYSICS_STEP", job_data)

# Poll with 3ms timeout (worker typically completes in ~100-200µs)
poll_start = time.perf_counter()
while (time.perf_counter() - poll_start) < 0.003:
    results = engine.poll_results(max_results=10)
    for result in results:
        if result.job_id == job_id:
            self._apply_physics_result(result.result, context, dynamic_map)
            break
    if result_found:
        break
    time.sleep(0.00005)  # 50µs adaptive sleep
```

### Spatial Grid Acceleration

Worker uses **3D spatial grid with DDA traversal** for collision:
- Grid cached once at game start (8 `CACHE_GRID` jobs sent to ensure all 4 workers receive it)
- **Verification system ensures each unique worker has the cache** (uses worker_id tracking)
- Cell-based lookup: O(1) instead of O(n) triangle checks
- Typical performance: ~100-200µs per physics step

**Grid Stats (example):**
- Cell size: 2.0m
- Grid dimensions: 74 x 40 x 15 = 44,400 cells
- Non-empty: ~5,000 cells (12% fill)
- Triangles: ~70,000 references
- Build time: ~180ms (one-time cost)
- Cache size: ~3MB serialized

**Cache Verification (CRITICAL):**
- Workers share a job queue, so 4 jobs doesn't guarantee 4 unique workers
- Submit 8 jobs (2x worker count) to increase probability each worker gets one
- Track unique worker confirmations using worker_id field
- Game aborts if not all 4 workers confirm within 5 seconds
- See "ENGINE-FIRST STARTUP SYSTEM" section for details

### What's Working

✅ Basic movement and collision
✅ Ground detection and snapping
✅ Jump buffering and coyote time
✅ Step-up on obstacles
✅ Wall sliding
✅ Dynamic mesh collision (horizontal and ground)
✅ Platform carry (linear and angular)
✅ Same-frame polling (low latency)
✅ Steep slope upward blocking (removes upward velocity component)
✅ Steep slope downward sliding (accelerates down slope)

### Active Development Needed

⚠️ **Slope handling needs tuning**
- Upward blocking works but may feel inconsistent
- Slide strength tuned but needs playtesting
- Interaction between blocking and sliding needs refinement
- **ALL slope work must be done in worker** (`engine_worker_entry.py`)

⚠️ **Timing/smoothness not perfect**
- Some jitter in specific scenarios
- Physics step timing needs investigation
- May be related to engine performance inconsistency

⚠️ **Dynamic mesh platform carry**
- Movement relative to platform works but can be stuttery
- Platform carry applied after physics (may need position prediction)

### Physics Development Guidelines

**When working on physics:**
1. ✅ ALL physics computation MUST happen in worker (`engine_worker_entry.py`)
2. ✅ Main thread (`exp_kcc.py`) should ONLY apply results
3. ✅ Test with dev_debug_kcc_offload enabled to see worker output
4. ✅ Changes must not break same-frame polling pattern
5. ✅ Verify grid cache is working (check static_tris_tested > 0)

### Debug Output

Enable via Developer Tools panel (N-panel → Create → Developer Tools):
- `dev_debug_kcc_offload` - KCC physics debug
- `dev_debug_engine` - Engine job processing

**Output format:**
```
[KCC] APPLY pos=(19.42,-0.15,3.53) ground=False blocked=False step=False | 68us 4rays 0tris
```

---

## 🎥 CAMERA SYSTEM (ENGINE-OFFLOADED - PRODUCTION READY)

**Status:** Camera occlusion fully offloaded with explicit synchronization. Frame-perfect, debug-independent.

### Architecture

**Worker computes:**
- Static geometry raycast (cached spatial grid - DDA traversal)
- Dynamic mesh raycast (brute force triangle intersection)
- Returns hit distance for camera pull-in (~200µs typical)

**Main thread computes:**
- LoS verification (Blender BVH - very fast)
- Camera sphere pushout (Blender BVH - very fast)
- Apply to viewport (requires bpy)

### CRITICAL: Explicit Synchronization Pattern

**Camera uses same-frame explicit polling like physics:**
```python
# 1. Submit AFTER physics (uses final character position)
camera_job_id = submit_camera_occlusion_early(op, context)

# 2. Poll with timeout (3ms) + 150µs pre-delay
poll_camera_result_with_timeout(op, context, camera_job_id, timeout=0.003)

# 3. Update camera (guaranteed fresh result)
update_camera_for_operator(context, op)
```

**Why this order matters:**
- ❌ OLD: Submit before physics → used stale position → one-frame clipping
- ✅ NEW: Submit after physics → uses final position → zero-frame latency

**Pre-poll delay (150µs):**
- Gives worker time to grab job from queue and start processing
- Prevents polling loop from starving worker of CPU time
- Makes behavior identical whether debug prints are on or off
- Without this, debug prints accidentally provided the needed delay

### Camera Development Guidelines

**When working on camera:**
1. ✅ Camera MUST submit AFTER physics (uses final character position)
2. ✅ MUST use explicit polling with timeout (like physics does)
3. ✅ Pre-poll delay MUST stay at 150µs (prevents CPU starvation)
4. ✅ Never rely on debug prints for timing - behavior must be identical debug on/off
5. ✅ Heavy raycast work in worker (`CAMERA_OCCLUSION_FULL` job type)
6. ✅ Main thread (`exp_view.py`) only for fast BVH ops and bpy writes
7. ✅ Test with dev_debug_camera_offload both ON and OFF - must feel identical

---

## CRITICAL: BPY Import Guard

**Problem:** Workers crash with `ModuleNotFoundError: No module named '_bpy'`

**Solution:** Guard in `Exploratory/__init__.py`:
```python
import sys
if 'multiprocessing' in sys.modules and __name__ != '__main__':
    # Worker process - skip all bpy imports
    pass
else:
    # Normal Blender context - safe to import
    import bpy
    # ... ALL imports inside this else block ...
```

**ALL imports and functions that use bpy MUST be inside the `else` block!**

---

## Project Overview

**Exploratory** is a Blender addon that transforms Blender into a game engine and interactive experience platform.

**Target Blender Version**: 5.0.0+
**License**: GNU GPL v2

### High-Level Architecture

```
Exploratory/
├── __init__.py                 # Registration (HAS BPY IMPORT GUARD!)
├── Exp_Game/                   # Game Engine Core
│   ├── engine/                 # Multiprocessing engine (4 workers) ⚡ NEEDS OPTIMIZATION
│   ├── modal/                  # Game loop (30Hz fixed timestep)
│   ├── physics/                # KCC (offloaded to engine) 🔧 ACTIVE DEVELOPMENT
│   ├── animations/             # Character state machine
│   ├── interactions/           # Trigger system
│   ├── reactions/              # Event response system
│   ├── systems/                # Performance culling (offloaded)
│   └── developer/              # Debug toggles (N-panel)
├── Exp_Nodes/                  # Visual node editor
└── Exp_UI/                     # Web integration (exploratory.online)
```

### Game Loop (30Hz Fixed Timestep)

**File:** `Exp_Game/modal/exp_loop.py` (`GameLoop.on_timer`)

**Update order per frame:**
1. Time update (wall-clock + scaled)
2. Camera occlusion submit (early - gives workers time to process)
3. Custom tasks (scripted animations, property changes)
4. Dynamic meshes (BVH rebuild for moving platforms)
5. Performance culling submit (engine job)
6. Animations (character state machine)
7. Input handling
8. **Physics** (KCC via engine - same-frame polling)
9. **Engine poll** (apply completed jobs from earlier submissions)
10. Camera update (uses cached worker result)
11. Interactions/reactions
12. UI/audio

### Core Systems

**Performance Culling:**
- Distance-based object visibility (1000+ objects)
- Fully offloaded to engine workers
- Round-robin batching for sustained load

**Character Animation:**
- State machine: idle/walk/run/jump/fall/land
- NLA track integration
- Audio state synchronization

**Interactions/Reactions:**
- Triggers: PROXIMITY, COLLISION, INTERACT, ACTION, TIMER, ON_GAME_START
- Reactions: Actions, sounds, transforms, projectiles, UI, objectives
- Task-based execution with delays and interpolation

---

## Developer Tools

**Location:** `Exp_Game/developer/`

**Debug Categories (N-panel → Create → Developer Tools):**
- `dev_startup_logs` - **Startup sequence debug** (5-step engine-first verification) - **USE THIS TO DEBUG STARTUP ISSUES**
- `dev_debug_engine` - Engine/worker debug (job submission, completion, failures)
- `dev_debug_kcc_offload` - Physics offload debug (worker computation details)
- `dev_debug_camera_offload` - Camera offload debug (raycast results, timeouts)
- `dev_debug_performance` - Culling debug
- `dev_debug_physics` - Movement/collision debug
- `dev_debug_interactions` - Trigger/reaction debug
- `dev_debug_audio` - Audio debug
- `dev_debug_animations` - Animation debug

**⚠️ ALL debug output MUST:**
1. Be toggleable via `scene.dev_debug_*` properties
2. Default to False (silent console in production)
3. Use format: `[Category] message`

---

## Common Patterns

### Engine Offload Pattern

```python
# 1. SNAPSHOT (main thread - pickle-safe data only!)
data = {"positions": [(obj.location.x, obj.location.y, obj.location.z) for obj in objects]}

# 2. SUBMIT (non-blocking)
job_id = self.engine.submit_job("JOB_TYPE", data)

# 3. POLL (non-blocking, in game loop)
results = self.engine.poll_results()

# 4. APPLY (main thread, bpy writes)
for result in results:
    obj.hide_viewport = result.result["should_hide"]
```

### Adding New Engine Job Type

**BEFORE adding new job types:**
- ⚠️ Verify engine is stable enough (see ENGINE PERFORMANCE section)
- ⚠️ Ensure job won't overload workers
- ⚠️ Plan for job failure/timeout
- ⚠️ Profile job execution time

**1. Worker Handler** (`engine_worker_entry.py`):
```python
elif job.job_type == "MY_JOB":
    # Pure Python - NO bpy!
    result = compute_something(job.data)
    return {"result": result}
```

**2. Main Thread Handler** (`exp_loop.py` or system file):
```python
elif result.job_type == "MY_JOB":
    apply_my_result(op, result.result)
```

**3. Debug Logging:**
```python
if context.scene.dev_debug_MY_CATEGORY:
    print(f"[MyCategory] {message}")
```

### Modal Operator Pattern

- Store state as operator properties (not globals)
- Use `time.perf_counter()` for timing
- Engine lifecycle: start in `invoke()`, shutdown in `cancel()`
- Fixed timestep physics (30Hz, up to 3 catchup steps)

---

## Performance Considerations

**Main thread should ONLY:**
- Read Blender data (snapshots)
- Write Blender data (apply results)
- Coordinate engine jobs
- Handle user input
- Fast operations with bpy (BVH queries, etc.)

**Engine workers should do:**
- Heavy computation (distance, raycasts, pathfinding)
- Physics simulation
- AI decision-making
- Batch processing
- Anything that can run without bpy access

**Current bottlenecks:**
- Dynamic mesh BVH rebuild (per-frame, main thread) - CANNOT offload (needs bpy)
- Platform carry (needs bpy for matrix_world)
- Engine inconsistency (see ENGINE PERFORMANCE section)

---

## Known Limitations

**Engine:**
- ✅ **FIXED:** Startup readiness verification now in place (5-step gate)
- ✅ **FIXED:** Grid cache verification ensures all workers have cache
- ⚠️ **Still needs work:** Performance inconsistent across game sessions
- ⚠️ **Still needs work:** No worker load balancing (shared queue)
- ⚠️ **Still needs work:** Jobs can timeout unexpectedly (camera occlusion)
- ⚠️ **Still needs work:** No runtime health monitoring (only at startup)

**Physics (Active Development):**
- Slope handling needs continued tuning in worker
- Some timing/smoothness issues
- Dynamic platform carry can be stuttery

**Performance:**
- Moving platforms: Only vertical/rotational motion (no lateral sliding)
- Capsule collision only (no complex shapes)

**Multiprocessing:**
- Workers have NO bpy access (will crash)
- All data must be picklable (no Blender objects!)
- Workers share job queue - no guarantee which worker processes which job
- All workers must receive critical data (like grid cache) - use worker_id tracking

---

## Troubleshooting

### Workers Not Starting
**Symptom:** `ModuleNotFoundError: No module named '_bpy'`
**Fix:** Check BPY import guard in `__init__.py`, ensure ALL imports inside `else` block

### Game Won't Start - Startup Fails
**Symptom:** Game aborts during startup with error message
**Fix:** Enable `dev_startup_logs` in Developer Tools panel and restart game. The 5-step startup sequence will show exactly which step failed and why. Common issues:
- Workers not spawning → Check multiprocessing is available
- PING timeout → Workers crashed or deadlocked
- Grid cache timeout → Not all workers received cache (worker_id tracking issue)
- Health check failed → Engine in inconsistent state

### Camera Jobs Timing Out
**Symptom:** `[Camera TIMEOUT] Job X stuck for 101.2ms`
**Fix:** This is expected occasionally - timeout recovery is working. If constant (every 2-3 frames), investigate worker health and job queue depth.

### Physics Feels Inconsistent
**Symptom:** Character physics varies between game sessions
**Fix:** Likely engine performance issue - check worker job completion times with `dev_debug_engine`. Enable `dev_startup_logs` to verify all workers are healthy at startup.

---

## Architecture Philosophy

- **Engine-first** - Offload everything possible to workers
- **Modal companion** - Engine serves the modal, absorbs computational stress
- **TRUE parallelism** - Multiprocessing bypasses GIL
- **Responsive main thread** - Critical for smooth gameplay
- **Graceful degradation** - Failed jobs must not break game state
- **Fixed timestep** - Consistent physics (30Hz)
- **Blender-native** - Uses Blender's data model where appropriate

**The engine is not just a feature - it's the FOUNDATION that makes Exploratory possible.**

---

## 🚨 CRITICAL: Engine Health is EVERYTHING

**The #1 Priority for Exploratory Development:**

### On Startup
- ✅ **Engine MUST be 100% ready before modal starts** - Non-negotiable
- ✅ **All 4 workers MUST be verified** - PING check + grid cache
- ✅ **Any failure MUST abort game** - No half-ready states
- ✅ **Clear error messages** - Tell user exactly what failed
- ✅ **Debug with startup logs** - Enable `dev_startup_logs` to diagnose issues

### During Gameplay
- ⚠️ **Monitor engine health continuously** (future work)
- ⚠️ **Prevent engine overload** - All systems must respect engine capacity
- ⚠️ **Handle job rejection gracefully** - Always have fallbacks
- ⚠️ **Never trust workers to be instant** - Use timeouts and recovery
- ⚠️ **Profile new workloads before adding** - Don't break existing stability

### Before Adding ANY New Engine Workload

**Ask these questions IN ORDER:**

1. **Is the engine currently healthy?**
   - Check current job throughput (should be <200 jobs/sec sustained)
   - Check for timeout warnings (should be rare)
   - Verify queue depth stays at 0 (no buildup)
   - If engine is struggling → FIX THAT FIRST

2. **Can this task be offloaded without bpy?**
   - If NO → Keep it on main thread
   - If YES → Continue

3. **How expensive is this computation?**
   - Profile worker execution time (aim for <1ms per job)
   - Test with realistic data (not empty test cases)
   - If >5ms → Consider optimization or batching

4. **How often will this job run?**
   - Physics: 30 jobs/sec (critical path)
   - Camera: 30 jobs/sec (timeout recovery)
   - Culling: 1-5 jobs/sec (throttled)
   - If >30 jobs/sec → Add throttling

5. **What happens if this job fails/times out?**
   - MUST have fallback (cached data, skip frame, etc.)
   - MUST NOT break game state
   - MUST NOT cause crash
   - If no good fallback → Reconsider offloading

6. **Will this overload the workers?**
   - Calculate total load: your_jobs/sec + existing_jobs/sec
   - If total >500 jobs/sec → Risk of overload
   - If total >1000 jobs/sec → Guaranteed problems
   - Add per-system throttling if needed

### Engine Health Checklist (Use Before Each Release)

- [ ] Enable `dev_startup_logs` - verify 5-step startup passes
- [ ] Enable `dev_debug_engine` - verify workers healthy during gameplay
- [ ] Play for 5 minutes - check for timeout warnings
- [ ] Monitor jobs/sec - should stay 60-120 range for typical gameplay
- [ ] Check queue depth - should stay at 0 (occasional spikes to 1-2 OK)
- [ ] Verify no "Grid not cached" warnings
- [ ] Test game restart 3 times - startup should pass every time

**Remember: A broken engine = broken game. Engine health is not optional.**
