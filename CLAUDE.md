# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL: Development Workflow

 NEVEr ADD "print()" statements never never never. always use the diagnostics logs (Exp_Game\developer\CLAUDE_LOGGER.md)

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


**Before offloading a new system, ask:**
- ✅ **Is the engine healthy enough to handle it?** Check current load
- ✅ **Will this job type overload workers?** Profile execution time
- ✅ **What happens if this job fails/times out?** Implement fallback
- ✅ **Does this genuinely free the main thread?** Measure impact
- ✅ **Can workers handle this without bpy?** No Blender objects allowed
- ✅ **Does it work identically with debug on/off?** Test both - behavior must be identical

---

## 📁 ADDON STRUCTURE OVERVIEW

```
Exploratory/
├── __init__.py              # Registration + BPY import guard for workers
├── Exp_Game/                # Runtime game engine (THIS IS THE CORE)
│   ├── engine/              # Multiprocessing worker system
│   ├── modal/               # Game loop + modal operator
│   ├── physics/             # KCC, camera, dynamic meshes
│   ├── animations/          # State machine + blend system
│   ├── interactions/        # Trigger evaluation
│   ├── reactions/           # Response executors
│   ├── systems/             # Performance culling
│   ├── audio/               # Sound management
│   ├── developer/           # Debug tools + logging
│   ├── startup_and_reset/   # Game init/cleanup
│   └── props_and_utils/     # Shared utilities
├── Exp_Nodes/               # Visual node editor (DESIGN TIME ONLY)
└── Exp_UI/                  # Web API integration (exploratory.online)
```

---

## 🔄 MODAL + ENGINE RELATIONSHIP

- **Modal** (`exp_modal.py`) = Blender's game loop host, runs at 30Hz strictly
- **Engine** (`engine_core.py`) = 4 worker processes for heavy computation
- **Relationship**: Modal orchestrates, engine computes. Modal stays thin and fast.

**Frame flow:**
1. Modal timer fires (33ms interval)
2. Main thread snapshots data → submits jobs to engine
3. Workers compute in parallel (physics, camera, culling)
4. Main thread polls results → applies to Blender objects
5. Repeat

**Critical rule:** Main thread NEVER blocks waiting for workers. Use polling with timeouts.

---

## 🎨 NODE SYSTEM (Exp_Nodes/)

**IMPORTANT: Nodes are DESIGN-TIME ONLY. Never accessed during gameplay.**

- Nodes provide visual UI for building interactions/reactions
- At game start: node data is serialized into lightweight runtime structures
- During game: only serialized data is used, node graphs are never traversed
- This keeps runtime fast - no node lookups, no graph walking

**Key pattern:**
```python
# GAME START: Serialize once
tracker_data = serialize_tracker_graph(scene)  # Reads nodes → flat data

# RUNTIME: Use serialized data only
evaluate_trackers(tracker_data, world_state)   # No node access
```

---

## 🎬 ANIMATION SYSTEM

**Location:** `Exp_Game/animations/`

| File | Purpose |
|------|---------|
| `controller.py` | Animation state machine (idle/walk/run/jump/fall/land) |
| `blend_system.py` | Layer-based animation blending with IK support |
| `runtime_ik.py` | Runtime IK solving (arms/legs) |
| `states.py` | AnimState enum and transitions |

**Engine integration:** Animation evaluation can be offloaded for complex rigs. Worker computes bone transforms → main thread applies to armature.

---

## ⚡ INTERACTIONS & REACTIONS

**Interactions** (`Exp_Game/interactions/`) = Trigger detection
- PROXIMITY, COLLISION, INTERACT, ACTION, TIMER, ON_GAME_START
- Evaluated each frame against world state
- Fires reactions when conditions met

**Reactions** (`Exp_Game/reactions/`) = Response execution
- Sounds, transforms, projectiles, UI text, counters, timers, teleports
- Task-based with delays and interpolation
- Each reaction type has its own executor file

**Flow:** Interaction fires → schedules reaction batch → reactions execute over time

---

## 🌐 Exp_UI (Web Integration)

**Purpose:** Connect to exploratory.online web app

- Users upload worlds to website
- Other users can explore uploaded worlds
- Addon downloads .blend file → extracts scene → launches game
- `launched_from_ui` flag tracks this mode for proper cleanup

**Not used for local development/testing.**

---

## 🚀 STARTUP & CLEANUP PROCESS

### Game Start (`exp_modal.py:invoke()`)
1. Engine spawn (4 workers)
2. Worker alive verification
3. PING responsiveness check
4. Spatial grid cache to all workers
5. Final health check
6. Initialize game systems
7. Start 30Hz timer

### Game End (`exp_modal.py:cancel()`)
1. Game loop shutdown
2. Animation shutdown
3. Engine shutdown (workers terminate)
4. GPU handlers cleanup (KCC vis, crosshairs, UI)
5. State cleanup (camera, dynamic meshes, audio, fonts, IK, stats)
6. Fullscreen exit + UI restore
7. Scene state restore

**Critical:** Every module-level cache must be cleared on game end. Stale state causes bugs on restart.

---

## ⏱️ 30Hz GAME LOOP (STRICTLY 30Hz)

**File:** `Exp_Game/modal/exp_loop.py`

- Fixed 33.33ms timestep, never faster, never slower
- Up to 3 catchup steps if frames are missed
- All gameplay logic tied to this rate

**Frame order:**
1. Time update
2. Physics submit → poll → apply
3. Camera submit → poll → apply
4. Animations
5. Interactions/reactions
6. Performance culling
7. Audio/UI

---

## 📝 LOGGING SYSTEM (CRITICAL)

**⚠️ NEVER USE `print()` DURING GAMEPLAY - EXTREMELY SLOW**

Blender's Python console is blocking. Print statements during game cause stutters.

**USE THIS INSTEAD:** `Exp_Game/developer/dev_logger.py`

```python
from ..developer.dev_logger import log_game

log_game("CATEGORY", f"message here")  # Fast buffered logging
```

**How it works:**
- Logs buffer in memory during gameplay
- Export to file on game end (if enabled)
- Zero performance impact during play

**Full documentation:** `Exp_Game/developer/CLAUDE_LOGGER.md`

**Debug toggles:** All logging gated by `scene.dev_debug_*` properties. Default OFF.

---

## 🏗️ ENGINE STRUCTURE

**Location:** `Exp_Game/engine/`

| File | Purpose |
|------|---------|
| `engine_core.py` | Main thread: spawn workers, submit jobs, poll results |
| `engine_types.py` | EngineJob, EngineResult dataclasses |
| `engine_worker_entry.py` | Worker process: job dispatch + handlers |
| `worker/` | Organized worker-side modules (physics, interactions, etc.) |

**Organization matters:** Worker code is in `engine/worker/` subdirectory. Keep it organized - don't dump everything in one file.

**Current job types:**
- `KCC_PHYSICS_STEP` - Character physics
- `CAMERA_OCCLUSION_FULL` - Camera raycast
- `CULL_BATCH` - Visibility culling
- `CACHE_GRID` - Spatial grid initialization
- `CACHE_DYNAMIC_MESH` - Moving platform triangles
- `EVALUATE_TRACKERS` - Interaction condition checks

---

## 🎯 CORE PRINCIPLE

**The engine handles computation. The main thread stays lean.**

- If it doesn't need `bpy` → offload to engine
- If it's O(n) or worse → offload to engine
- If it takes >1ms → offload to engine

Main thread only: read Blender data, write Blender data, coordinate jobs.

This is why the game runs smoothly at 30Hz even with complex physics and 1000+ objects.
