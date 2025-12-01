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

**We are in a TRANSITION PHASE developing a robust multiprocessing helper engine.**

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

**Usage Pattern:**
```python
# Main thread - snapshot data
data = {"positions": [(obj.location.x, obj.location.y, obj.location.z) for obj in objects]}

# Submit to engine (non-blocking)
job_id = self.engine.submit_job("JOB_TYPE", data)

# Poll for results (non-blocking)
results = self.engine.poll_results()

# Apply to Blender (main thread)
for result in results:
    obj.hide_viewport = result.result["should_hide"]
```

**Current Production Workloads:**
- `KCC_PHYSICS_STEP` - Full character physics computation (see Physics System section)
- `CULL_BATCH` - Performance culling (1000+ objects)
- `DYNAMIC_MESH_ACTIVATION` - Dynamic mesh distance gating
- `INTERACTION_CHECK_BATCH` - Proximity/collision checks
- `CAMERA_OCCLUSION_FULL` - Camera raycast occlusion

---

## ⚠️ PHYSICS SYSTEM (COMPLETE REBUILD - IN TRANSITION)

**Status:** The entire physics system was rebuilt for full engine offload. Currently functional but in refinement phase.

### Architecture: Full Physics Offload

**Worker computes ENTIRE physics step:**
1. Input → velocity acceleration
2. Gravity
3. Jump
4. Horizontal collision (3D DDA spatial grid)
5. Step-up detection
6. Wall slide
7. Ceiling check
8. Ground detection
9. Steep slope sliding

**Main thread is THIN:**
- Apply worker result
- Dynamic mesh collision (BVH raycasts)
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
- Grid cached once at game start (`CACHE_GRID` job)
- Cell-based lookup: O(1) instead of O(n) triangle checks
- Typical performance: ~100-200µs per physics step

**Grid Stats (example):**
- Cell size: 2.0m
- Grid dimensions: 74 x 40 x 15 = 44,400 cells
- Non-empty: ~5,000 cells (12% fill)
- Triangles: ~70,000 references
- Build time: ~180ms (one-time cost)

### What's Working

✅ Basic movement and collision
✅ Ground detection and snapping
✅ Jump buffering and coyote time
✅ Step-up on obstacles
✅ Wall sliding
✅ Dynamic mesh collision (horizontal and ground)
✅ Platform carry (linear and angular)
✅ Same-frame polling (low latency)

### Known Issues (Active Work)

⚠️ **Slope handling not 100% reliable**
- Steep slope sliding direction corrected but may need tuning
- Slope blocking when running needs refinement

⚠️ **Timing/smoothness not perfect**
- Some jitter remains in specific scenarios
- Physics step timing needs more work

⚠️ **Dynamic mesh platform carry**
- Movement relative to platform works but can be stuttery
- Platform carry applied after physics (may need position prediction)

### Debug Output

Enable via Developer Tools panel (N-panel → Create → Developer Tools):
- `dev_debug_kcc_offload` - KCC physics debug
- `dev_debug_engine` - Engine job processing

**Output format:**
```
[KCC] APPLY pos=(19.42,-0.15,3.53) ground=False blocked=False step=False | 68us 4rays 0tris
```

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
│   ├── engine/                 # Multiprocessing engine (4 workers)
│   ├── modal/                  # Game loop (30Hz fixed timestep)
│   ├── physics/                # KCC (offloaded to engine)
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
2. Custom tasks (scripted animations, property changes)
3. Dynamic meshes (BVH rebuild for moving platforms)
4. **Performance culling submit** (engine job)
5. **Engine poll** (apply completed jobs)
6. Animations (character state machine)
7. Input handling
8. **Physics** (KCC via engine)
9. Camera update
10. Interactions/reactions
11. UI/audio

### Core Systems

**Interactions/Reactions:**
- Triggers: PROXIMITY, COLLISION, INTERACT, ACTION, TIMER, ON_GAME_START
- Reactions: Actions, sounds, transforms, projectiles, UI, objectives
- Task-based execution with delays and interpolation

**Performance Culling:**
- Distance-based object visibility (1000+ objects)
- Fully offloaded to engine workers
- Round-robin batching for sustained load

**Character Animation:**
- State machine: idle/walk/run/jump/fall/land
- NLA track integration
- Audio state synchronization

---

## Developer Tools

**Location:** `Exp_Game/developer/`

**Debug Categories (N-panel → Create → Developer Tools):**
- `dev_debug_engine` - Engine/worker debug
- `dev_debug_kcc_offload` - Physics offload debug
- `dev_debug_performance` - Culling debug
- `dev_debug_physics` - Movement/collision debug
- `dev_debug_interactions` - Trigger/reaction debug
- `dev_debug_audio` - Audio debug
- `dev_debug_animations` - Animation debug
- `dev_debug_all` - Master toggle

**⚠️ ALL debug output MUST:**
1. Be toggleable via `scene.dev_debug_*` properties
2. Default to False (silent console in production)
3. Use format: `[Category] message`

---

## Common Patterns

### Engine Offload Pattern

```python
# 1. SNAPSHOT (main thread)
data = snapshot_blender_data()

# 2. SUBMIT (non-blocking)
job_id = engine.submit_job("JOB_TYPE", data)

# 3. POLL (non-blocking, in game loop)
results = engine.poll_results()

# 4. APPLY (main thread, bpy writes)
for result in results:
    apply_to_blender(result.result)
```

### Adding New Engine Job Type

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

**Engine workers should do:**
- Heavy computation (distance, raycasts, pathfinding)
- Physics simulation
- AI decision-making
- Batch processing

**Current bottlenecks:**
- Dynamic mesh BVH rebuild (per-frame, main thread) - CANNOT offload (needs bpy)
- Platform carry (needs bpy for matrix_world)
- Camera updates (needs bpy for object transforms)

---

## Known Limitations

**Physics (In Transition):**
- Slope handling needs refinement
- Some timing/smoothness issues remain
- Dynamic platform carry can be stuttery

**Performance:**
- Moving platforms: Only vertical/rotational motion (no lateral sliding)
- Capsule collision only (no complex shapes)

**Multiprocessing:**
- Workers have NO bpy access (will crash)
- All data must be picklable (no Blender objects!)
- 0-1 frame latency acceptable for most systems

---

## Troubleshooting

### Workers Not Starting
**Symptom:** `ModuleNotFoundError: No module named '_bpy'`
**Fix:** Check BPY import guard in `__init__.py`, ensure ALL imports inside `else` block

### Physics Issues
**Symptom:** Character behaves incorrectly
**Fix:** Enable `dev_debug_kcc_offload` to see per-frame physics output

### Silent Console
**Symptom:** No debug output
**Fix:** Enable debug categories in Developer Tools panel (N-panel → Create)

---

## Architecture Philosophy

- **Blender-native** - Uses Blender's data model
- **Modal-driven** - Timer-based game loop (not frame handlers)
- **Fixed timestep** - Consistent physics (30Hz)
- **Engine-first** - Offload everything possible to workers
- **TRUE parallelism** - Multiprocessing bypasses GIL
- **Responsive main thread** - Critical for smooth gameplay
