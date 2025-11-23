# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL: Development Workflow

**ALWAYS MAKE CHANGES TO:** `C:\Users\spenc\Desktop\Exploratory\addons\Exploratory`

**NEVER EDIT:** `C:\Users\spenc\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\Exploratory`

The AppData location is where Blender loads the addon from, but it's NOT the development directory. The user has a custom install script that copies from Desktop → AppData:

```python
# User's install script
SRC = r"C:\Users\spenc\Desktop\Exploratory\addons\Exploratory"
DEST = r"%APPDATA%\Blender Foundation\Blender\5.0\scripts\addons\Exploratory"
# Script removes old DEST and copies SRC → DEST
```

**Workflow:**
1. Make ALL code changes to Desktop version
2. User runs their install script (copies Desktop → AppData)
3. User reloads Blender or restarts to test

**DO NOT write directly to AppData - changes will be lost when the install script runs!**

---

## Project Overview

**Exploratory** is a Blender addon (v1.0.0) that transforms Blender into a game engine and interactive experience platform. Users can create first/third-person games, download community worlds from exploratory.online, and build interactive experiences using visual node-based programming.

**Target Blender Version**: 5.0.0+
**Author**: Spencer Shade
**Website**: https://exploratory.online/
**License**: GNU GPL v2

## High-Level Architecture

The addon consists of three major subsystems plus a multiprocessing engine:

```
Exploratory/
├── __init__.py                 # Main registration orchestrator (HAS BPY IMPORT GUARD!)
├── exp_preferences.py          # User settings (keybinds, character, performance)
├── build_character.py          # Character/armature building
├── update_addon.py             # Self-updating system
│
├── Exp_Game/                   # Game Engine Core (modal operator, physics, interactions)
│   ├── engine/                 # Multiprocessing engine for offloading computations
│   └── developer/              # Developer tools (debug toggles, diagnostics)
├── Exp_Nodes/                  # Visual Node Editor System
└── Exp_UI/                     # Web Integration & Community Features
```

---

## CRITICAL: Multiprocessing Engine System

### Overview

A TRUE multiprocessing engine (bypasses Python's GIL) was added to offload heavy computations from the main modal thread.

**Location:** `Exp_Game/engine/`

**Status:** ✅ **PRODUCTION READY** - Actively used for performance culling (1000+ objects)

### Architecture

```
Main Thread (Blender Modal)          Worker Processes (Separate Cores)
┌─────────────────────────┐         ┌──────────────────────────┐
│  ExpModal               │         │  Worker 0                │
│  ├─ Game loop          │         │  ├─ Process jobs         │
│  ├─ Submit jobs   ────►├────────►│  ├─ Return results       │
│  ├─ Poll results  ◄────┤◄────────┤  └─ NO bpy access!       │
│  └─ Apply results      │         └──────────────────────────┘
└─────────────────────────┘         ┌──────────────────────────┐
                                    │  Worker 1, 2, 3...       │
         Job Queue                  │  (4 workers total)       │
         ↓        ↑                 └──────────────────────────┘
    Submit    Poll
```

### Files

```
Exp_Game/engine/
├── __init__.py              # Public API exports
├── engine_core.py           # Main engine manager (EngineCore class)
├── engine_worker_entry.py   # Worker process entry point (NO bpy imports!)
├── engine_types.py          # Data structures (EngineJob, EngineResult, EngineHeartbeat)
├── engine_config.py         # Configuration (worker count, queue sizes, debug flags)
├── stress_test.py           # Comprehensive stress test suite
├── test_operator.py         # Blender UI operators (Quick Test, Stress Test)
└── ENGINE_STATUS.md         # Current status, test results, and next steps
```

### Key Classes

**EngineCore** (engine_core.py):
- Main manager running on Blender main thread
- Methods: `start()`, `shutdown()`, `submit_job()`, `poll_results()`, `send_heartbeat()`, `is_alive()`, `get_stats()`

**EngineJob** (engine_types.py):
```python
@dataclass
class EngineJob:
    job_id: int
    job_type: str
    data: Any  # Must be picklable - NO bpy objects!
    timestamp: float
```

**EngineResult** (engine_types.py):
```python
@dataclass
class EngineResult:
    job_id: int
    job_type: str
    result: Any  # Must be picklable
    success: bool
    error: Optional[str]
    timestamp: float
    processing_time: float
```

### Integration with Modal Operator

**Location:** `Exp_Game/modal/exp_modal.py`

**1. Import (line ~37):**
```python
from ..engine import EngineCore
```

**2. Startup in `invoke()` (lines ~424-436):**
```python
# ========== ENGINE INITIALIZATION ==========
if not hasattr(self, 'engine'):
    self.engine = EngineCore()

self.engine.start()

if not self.engine.is_alive():
    self.report({'WARNING'}, "Multiprocessing engine failed to start - continuing without engine")
else:
    print("[ExpModal] Multiprocessing engine started successfully")
# ===========================================
```

**3. Heartbeat & Polling in `modal()` (lines ~455-469):**
```python
# ========== ENGINE HEARTBEAT & POLLING ==========
if hasattr(self, 'engine') and self.engine:
    self.engine.send_heartbeat()

    results = self.engine.poll_results()
    if results:
        for result in results:
            if result.success:
                pass  # TODO: Handle successful results
            else:
                print(f"[ExpModal] Engine job {result.job_id} failed: {result.error}")
# ================================================
```

**4. Shutdown in `cancel()` (lines ~496-503):**
```python
# ========== ENGINE SHUTDOWN ==========
if hasattr(self, 'engine') and self.engine:
    print("[ExpModal] Shutting down multiprocessing engine...")
    self.engine.shutdown()
    self.engine = None
# ====================================
```

### CRITICAL: BPY Import Guard

**Problem:** Worker processes try to import the addon's `__init__.py`, which imports `bpy`, which doesn't exist in workers → crash with `ModuleNotFoundError: No module named '_bpy'`

**Solution:** Guard in `Exploratory/__init__.py` (lines ~29-58):
```python
import sys
if 'multiprocessing' in sys.modules and __name__ != '__main__':
    # Worker process - skip all bpy imports
    pass
else:
    # Normal Blender context - safe to import
    import bpy
    from .Exp_UI.internet.helpers import is_internet_available
    # ... all other imports inside this else block ...

    def version_check_timer():
        # ...

    def register():
        # ...

    def unregister():
        # ...
```

**IMPORTANT:** ALL imports and functions that depend on bpy MUST be inside the `else` block!

### Worker Process Isolation

**engine_worker_entry.py** is completely self-contained:
- NO relative imports (avoids triggering addon `__init__.py`)
- Defines `EngineJob` and `EngineResult` inline (duplicated to avoid imports)
- Contains `process_job()` and `worker_loop()` functions
- NO access to `bpy` module

**Production Job Types:**
- `"CULL_BATCH"` - Performance culling distance calculations (PRODUCTION USE)
- `"ECHO"` - Simple echo test (echoes data back)
- `"FRAME_SYNC_TEST"` - Lightweight sync test for latency measurement
- `"COMPUTE_HEAVY"` - Stress test job with configurable iterations

**To Add New Job Types:**
1. Edit `process_job()` in `engine_worker_entry.py`:
   ```python
   elif job.job_type == "MY_JOB_TYPE":
       # Pure Python logic - NO bpy!
       result_data = {"my_result": compute_something(job.data)}
   ```
2. Add handler in `exp_loop.py` `_poll_and_apply_engine_results()`:
   ```python
   elif result.job_type == "MY_JOB_TYPE":
       # Apply result to Blender (bpy access OK here)
       apply_my_result(op, result.result)
   ```
3. **CRITICAL**: Add debug logging controlled by `scene.dev_debug_*` properties
4. Test using Developer Tools panel (N-panel → Create → Developer Tools)

### Configuration

**File:** `Exp_Game/engine/engine_config.py`

```python
WORKER_COUNT = 4              # Number of worker processes (hardcoded for compatibility)
JOB_QUEUE_SIZE = 10000        # Max buffered jobs (large for burst loads)
RESULT_QUEUE_SIZE = 10000     # Max buffered results (large for burst loads)
HEARTBEAT_INTERVAL = 1.0      # Seconds between heartbeats
SHUTDOWN_TIMEOUT = 2.0        # Seconds to wait for graceful shutdown
DEBUG_ENGINE = False          # Debug logging (controlled by scene.dev_debug_engine)
```

**⚠️ CRITICAL - Debug Logging:**
- ALL debug output MUST be toggleable via Developer Tools panel
- Set `DEBUG_ENGINE = False` by default (silent console)
- Users enable debug categories in N-panel → Create → Developer Tools
- See "Developer Tools Module" section below

### Usage Pattern

**1. Submit Job (Main Thread):**
```python
# Snapshot data from Blender (bpy access OK here)
positions = [(obj.location.x, obj.location.y, obj.location.z) for obj in objects]

# Submit to workers (non-blocking)
job_id = self.engine.submit_job("COMPUTE", {"positions": positions})
```

**2. Process Job (Worker Process):**
```python
# In engine_worker_entry.py process_job()
if job.job_type == "COMPUTE":
    # Pure Python - NO bpy access!
    result = heavy_computation(job.data["positions"])
    return EngineResult(job.job_id, job.job_type, {"result": result}, success=True)
```

**3. Poll Results (Main Thread):**
```python
# Non-blocking poll
results = self.engine.poll_results()

for result in results:
    if result.success:
        # Apply to Blender objects (bpy access OK here)
        apply_result_to_scene(result.result)
```

### Current Status

**✅ Phase 1 Complete - Core Engine Functional:**
- Engine infrastructure built and stress tested
- Integrated into modal operator
- BPY import guard in place
- 4 worker processes (hardcoded for cross-platform safety)
- Heartbeat monitoring (1/sec)
- Graceful shutdown
- **Stress Test Results: Grade B (USABLE)**
  - 1,284 jobs/sec throughput
  - 100% completion rate (7,146/7,146 jobs)
  - 0 queue rejections
  - Cross-platform compatible (Windows confirmed, macOS/Linux expected)

---

## ✅ ENGINE-MODAL SYNCHRONIZATION (COMPLETED)

The multiprocessing engine is **fully synchronized** with the 30Hz modal game loop and **production-ready**.

### What Was Implemented

**1. Frame Tracking System** (`exp_modal.py`)
- `_physics_frame` counter increments with each physics step
- Tracks exact frame number for engine synchronization

**2. Frame-Tagged Job Submission**
- `submit_engine_job()` tags every job with frame number and timestamp
- `_pending_jobs` tracks submission frame for latency measurement

**3. Engine Polling in Game Loop** (`exp_loop.py`)
- Moved from `modal()` to `GameLoop.on_timer()` for proper frame sync
- `_poll_and_apply_engine_results()` processes results after physics
- Integrated into update order: threads_poll → **engine_poll** → animations

**4. Comprehensive Stress Test Suite**
- 6 test scenarios: BASELINE, LIGHT_LOAD, MEDIUM_LOAD, HEAVY_LOAD, BURST_TEST, CATCHUP_STRESS
- Total duration: ~18 seconds, automatic execution on game start
- Per-scenario metrics: frame/time latency, catchup events, stale results
- `EngineSyncTestManager` class manages all testing

### Test Results (Proven Capabilities)

**✅ Zero-Frame Latency Under Load**
- LIGHT_LOAD (5 jobs/frame): 0.00 avg, 0 max latency
- MEDIUM_LOAD (15 jobs/frame): 0.00 avg, 0 max latency
- HEAVY_LOAD (30 jobs/frame): 0.00 avg, 0 max latency

**✅ Catchup Frame Synchronization**
- HEAVY_LOAD scenario: **50% catchup rate** (half of timer events had 2-3 physics steps)
- Still achieved **0.00 frame latency** - results arrive at correct physics frames
- Proves frame attribution is accurate during modal inconsistency

**✅ Burst Load Tolerance**
- 50-job bursts every second: 0.00 avg latency, no saturation
- Queue handles simultaneous job submissions without degradation

**✅ Forced Modal Inconsistency**
- CATCHUP_STRESS: Injected 100ms delays every 15 frames
- 19.2% catchup rate, still 0.00 frame latency
- Engine maintains sync during deliberate timer irregularities

**✅ Sustained Throughput**
- 900+ jobs/sec sustained over multiple scenarios
- 4,600+ total jobs processed in 18 seconds
- No dropped jobs (except 5 in early test iteration, fixed)

**⚠️ Startup Overhead (Known Issue)**
- BASELINE scenario: Max 7 frames latency during first 60 frames
- 11.9% stale results in first 2 seconds (worker warm-up)
- Not a concern for gameplay (only affects initial startup)

### Active Production Workloads

**Performance Culling (CULL_BATCH):**
- **Purpose**: Distance-based object visibility culling
- **Location**: `exp_performance.py` → `engine_worker_entry.py`
- **Throughput**: 1000+ objects per frame, round-robin batching
- **Optimization**: Zero main-thread distance calculations (only bpy writes)
- **Main Thread**: Snapshot positions, apply hide_viewport results
- **Worker**: Pure math distance calculations, threshold comparisons

**Test Jobs:**
- `"ECHO"` - Simple echo test (0ms compute)
- `"FRAME_SYNC_TEST"` - Lightweight sync test (0-0.1ms compute)
- `"COMPUTE_HEAVY"` - Stress test job (1-8ms compute, configurable iterations)

### Remaining Engine Tests (TODO Before Production)

**CRITICAL (Do First):**
1. **Long-Duration Stability** - Run 5-minute test, monitor memory/latency over time
2. **Worker Failure Recovery** - Kill worker mid-test, verify engine continues with 3 workers
3. **Queue Saturation** - Submit 15k jobs at once, verify graceful rejection

**RECOMMENDED:**
4. **Frame Rate Impact** - Monitor actual physics timing, verify 30Hz maintained
5. **Shutdown Under Load** - Force shutdown at peak load, verify clean exit

**Success Criteria for Production:**
- Phase 1 tests pass → Safe for production use
- Know exact capacity limits and failure modes
- Confidence in fault tolerance

---

**Current Status:** ✅ **PRODUCTION** - Engine running performance culling for 1000+ objects

**Debug Output Control:**
- **Default**: Silent console (all debug flags = False)
- **Enable**: N-panel → Create → Developer Tools → "Engine (Multiprocessing)"
- **Sync Tests**: Toggle "Run Sync Stress Tests on Start" (off by default)

**Expected Behavior (Debug Enabled):**
- Game start: `[Engine Worker 0-3] Started`, `Workers: 4/4`
- During game: Heartbeat messages showing workers alive
- Test mode: Scenario progress and [CATCHUP]/[BURST]/[DELAY] messages
- Game end: `[Engine Core] Shutdown complete (processed N jobs in X.Xs)`

**Expected Behavior (Production - Debug Disabled):**
- **Silent console** - no engine messages
- Engine runs invisibly in background
- Performance culling works without any console output

**If Workers Crash:**
- Error: `ModuleNotFoundError: No module named '_bpy'`
- Cause: BPY import guard not working or workers importing addon
- Fix: Check that `__init__.py` has the guard and all imports are inside `else` block

---

## Developer Tools Module

**Location:** `Exp_Game/developer/`

**Purpose:** Centralized debug controls for developers and advanced users.

### Structure

```
Exp_Game/developer/
├── __init__.py          # Module initialization and exports
├── dev_properties.py    # Debug toggle properties (scene properties)
└── dev_panel.py         # UI panel in Create tab
```

### Debug Categories (All Toggleable from N-Panel)

**Console Debug Output:**
- `dev_debug_engine` - Engine/multiprocessing debug (workers, jobs, heartbeats)
- `dev_debug_performance` - Performance culling debug (distance calcs, batch processing)
- `dev_debug_physics` - Physics & character controller debug (movement, collisions)
- `dev_debug_interactions` - Interactions & reactions debug (triggers, task execution)
- `dev_debug_audio` - Audio system debug (playback, state changes)
- `dev_debug_animations` - Animation & NLA debug (state transitions, blending)

**Master Toggle:**
- `dev_debug_all` - Enable ALL debug categories at once (convenience toggle)

**Engine Diagnostics:**
- `dev_run_sync_test` - Run comprehensive engine stress tests on game start (~18 seconds)

### Adding New Debug Categories

**1. Add Property** (`dev_properties.py`):
```python
bpy.types.Scene.dev_debug_NEW_CATEGORY = bpy.props.BoolProperty(
    name="New Category Debug",
    description="Detailed description of what this debugs",
    default=False
)
```

**2. Add to UI** (`dev_panel.py`):
```python
col.prop(scene, "dev_debug_NEW_CATEGORY", text="New Category Name")
```

**3. Update Master Toggle** (`dev_properties.py` → `_update_all_debug_flags()`):
```python
scene.dev_debug_NEW_CATEGORY = enabled
```

**4. Use in Code:**
```python
if context.scene.dev_debug_NEW_CATEGORY:
    print("[NewCategory] Debug message here")
```

### ⚠️ CRITICAL DEBUG LOGGING REQUIREMENTS

**ALL future engine changes MUST:**
1. ✅ Add debug logging for significant events
2. ✅ Make logging toggleable via `scene.dev_debug_*` properties
3. ✅ Test with debug enabled/disabled before committing
4. ✅ Default to **silent console** (debug flags = False)
5. ✅ Document new debug flags in this file

**Example - Adding Engine Job Type:**
```python
# In engine_worker_entry.py
elif job.job_type == "NEW_JOB":
    if DEBUG_ENGINE:  # Controlled by scene.dev_debug_engine
        print(f"[Worker] Processing NEW_JOB: {job.data}")
    # ... processing logic
```

**Philosophy:**
- **Production**: Silent console, no spam
- **Development**: Toggle specific categories as needed
- **Organized**: Group related debug output by category
- **Scalable**: Easy to add new categories without refactoring

---

## Original Architecture (Pre-Engine)

### 1. Exp_Game: Game Engine Core

Modal operator-based game loop running at 30Hz with fixed timestep physics.

**Core Systems**:
- **modal/**: Main game loop (`exp_modal.py` modal operator + `exp_loop.py` frame updates)
- **physics/**: Kinematic Character Controller (KCC), BVH collision trees, camera modes
- **animations/**: Character state machine (idle/walk/run/jump/fall/land)
- **interactions/**: Trigger system (proximity, collision, interact key, timers)
- **reactions/**: Event response system (actions, sounds, transforms, UI, projectiles, objectives)
- **systems/**: Performance culling (`exp_performance.py`), objectives
- **audio/**: Sound management
- **engine/**: Multiprocessing engine (TRUE parallelism, bypasses GIL)
- **developer/**: Debug toggles and diagnostic tools (N-panel UI)

**Game Loop Pattern** (`GameLoop.on_timer` in exp_loop.py):
1. Fixed 30Hz physics updates (wall-clock locked, uses `time.perf_counter()`)
2. Up to 3 catchup steps per frame to prevent time dilation
3. Per-frame update order:
   - time → tasks → dynamic meshes
   - **culling submit** → **engine poll** (performance culling via multiprocessing)
   - animations → input → physics → camera
   - interactions → UI → audio

**Physics System** (exp_kcc.py):
- Capsule-based collision with BVH raycasting
- Static BVH built once at game start, dynamic BVH per-frame for moving platforms
- Ground detection, jump buffering, platform riding

**Interaction/Reaction System**:
- Interactions define triggers (PROXIMITY, COLLISION, INTERACT, ACTION, OBJECTIVE_UPDATE, TIMER_COMPLETE, ON_GAME_START, EXTERNAL)
- Reactions execute responses (custom actions, sounds, property changes, transforms, projectiles, UI text, objective updates, mobility locks, etc.)
- Task-based execution with delays and interpolation
- Nodes translate to these same property groups at game start

**Performance Culling System:**
- **File**: `Exp_Game/systems/exp_performance.py`
- **Engine Integration**: Uses multiprocessing engine for distance calculations
- **Architecture**: Zero main-thread distance calculations (max offloading)
- **Main Thread**: Snapshot positions, apply hide_viewport, placeholder toggling
- **Engine Workers**: Per-object distance calculations, threshold comparisons
- **Throughput**: 1000+ objects per frame with round-robin batching
- **Optimization**: Single-object proxy check for per-object mode (no redundant iteration)

**Legacy Threading (REMOVED):**
- ❌ `exp_threads.py` - DELETED (replaced by multiprocessing engine)
- ❌ `ThreadEngine` class - REMOVED (Python threading, GIL-limited)
- ✅ All culling now uses multiprocessing engine (true parallelism)

### 2. Exp_Nodes: Visual Node Programming

Custom node tree (`ExploratoryNodesTreeType`) for authoring interactions without code.

**Node Categories**:
- **Triggers** (blue): Proximity, Collision, Interact Key, Action Key, External, Objective Update, Timer Complete, On Game Start
- **Reactions** (green/yellow): Custom Action, Character Action, Play Sound, Property Value, Transform, Custom UI Text, Crosshairs, Projectile, Hitscan, Objective Counter/Timer, Mobility, Mesh Visibility, Reset Game, Action Keys, Parenting, Tracking
- **Objectives** (orange): Counter/Timer nodes
- **Utilities** (gray): Delay, Capture Float Vector

**Runtime Behavior**: Nodes are NOT executed during gameplay. At game start, node trees are translated into `scene.interactions` collection. The game engine reads from property groups, not node trees.

### 3. Exp_UI: Web Integration

Connects Blender to exploratory.online for downloading/uploading worlds.

**Key Features**:
- Token-based authentication (`auth/`)
- Package browsing and downloading (`download_and_explore/`)
- GPU-accelerated custom UI overlay (`interface/drawing/`)
- Social features: likes, comments, event voting (`packages/`, `events/`)
- Preferences persistence workaround for Blender 5.0 (`prefs_persistence.py`)

**API Base URL** (`main_config.py`):
- Production: `https://exploratory.online`
- Dev: `http://127.0.0.1:5000` (via `ADDON_ENV` environment variable)

---

## Critical Registration Order

**IMPORTANT**: Submodules must register in this exact order due to dependencies:
1. **Developer properties** (early registration, used by all systems)
2. Core preferences and operators
3. Preferences persistence handlers
4. **Exp_Game** (defines scene properties used by nodes)
5. **Exp_UI** (may reference Game properties)
6. **Exp_Nodes** (node sockets reference Game property types)

See `Exp_Game/__init__.py` `register()` function for implementation.

---

## Key Property Storage Locations

**Scene Properties** (game state):
- `scene.target_armature`: Player armature
- `scene.character_actions`: Action pointers (idle, walk, run, jump, fall, land)
- `scene.character_audio`: Audio settings
- `scene.interactions`: Interaction definitions
- `scene.reactions`: Global reaction library
- `scene.objectives`: Quest objectives
- `scene.proxy_meshes`: Collision mesh list
- `scene.mobility_game`: Movement locks

**Addon Preferences** (user settings):
- Keybinds (WASD, jump, run, interact, action)
- Mouse sensitivity
- Character customization (skin, actions, sounds)
- Performance presets (Low/Medium/High)

---

## Common Development Tasks

### Running the Addon
1. Copy `Exploratory/` to Blender addons directory
2. Enable in Preferences → Add-ons → "Exploratory"
3. Access via 3D Viewport → N-panel → "Exploratory" tab

### Testing the Engine
1. Start game (modal operator)
2. Check System Console for:
   - `[Engine Worker 0] Started` through `[Engine Worker 3] Started`
   - `Workers: 4/4` in heartbeat messages
3. End game - should see `[Engine Core] Shutdown complete`

### Building a Game
**Method 1 - UI Panels**:
1. Set `scene.target_armature` to player armature
2. Add proxy meshes to `scene.proxy_meshes` collection
3. Create interactions in Studio panel (`VIEW3D_PT_Exploratory_Studio`)
4. Add reactions to interactions
5. Press "Start Game" to launch modal operator

**Method 2 - Node Editor**:
1. Create "Exploratory Nodes" tree in Node Editor
2. Add trigger nodes (proximity, collision, etc.)
3. Connect reaction nodes
4. Nodes auto-translate to interactions on game start

### Debugging
- Enable performance overlay: `Exp_Game/systems/exp_live_performance.py` (shows frame timing, physics steps)
- Console output: Use `print()` statements (visible in Blender console)
- Check registration order if properties are missing
- Engine debug: Set `DEBUG_ENGINE = True` in `engine_config.py`

---

## Important Patterns

### Modal Operator Pattern
`ExpModal` (Exp_Game/modal/exp_modal.py) demonstrates best practices:
- Store state as operator properties (not global variables)
- Use `time.perf_counter()` for monotonic timing
- Fixed timestep physics (separate from render FPS)
- Cleanup in `cancel()` method
- Timer for periodic updates (`event_timer_add`)
- **NEW:** Engine lifecycle (start in invoke, heartbeat in modal, shutdown in cancel)

### Multiprocessing Pattern
**Main Thread (Blender):**
```python
# Snapshot (read-only bpy access)
data = snapshot_from_blender()

# Submit (non-blocking)
job_id = engine.submit_job("TYPE", data)

# Poll (non-blocking)
results = engine.poll_results()

# Apply (write bpy)
for result in results:
    apply_to_blender(result)
```

**Worker Process:**
```python
# NO bpy access allowed!
def process_job(job):
    # Pure Python computation
    result = compute(job.data)
    return EngineResult(...)
```

### BVH Tree Usage
1. Build once for static meshes (`create_bvh_tree` in exp_bvh_local.py)
2. Per-frame rebuild for dynamic meshes (moving platforms)
3. Raycast down from character for ground detection
4. Store as operator property to avoid rebuilding

### Task System (Reactions)
1. Interaction fires → creates task dictionary
2. Task added to global task list
3. Each frame, task update function runs
4. Task removes itself when complete
5. Supports delays and property interpolation

---

## Extension Points

### Adding New Job Types to Engine
1. Edit `engine_worker_entry.py` → `process_job()` function
2. Add new `if job.job_type == "YOUR_TYPE":` branch
3. Implement pure Python logic (NO bpy!)
4. Return `EngineResult` with result data

### Adding New Trigger Types
1. Define enum in `exp_interaction_definition.py`
2. Add properties to `InteractionDefinition` class
3. Implement check logic in `exp_interactions.py`
4. Create node class in `trigger_nodes.py`
5. Add to node menu in `node_editor.py`

### Adding New Reaction Types
1. Define properties in `exp_reaction_definition.py`
2. Implement execution in `exp_reactions.py` (or new module)
3. Create node class in `reaction_nodes.py`
4. Add to node menu in `node_editor.py`
5. Update UIList drawing in `trig_react_obj_lists.py`

### Customizing Character Physics
Physics tuning exposed in `Exp_Game/physics/exp_kcc.py`:
- `CharacterPhysicsConfigPG`: Gravity, jump force, acceleration, friction
- Editable via `VIEW3D_PT_Exploratory_PhysicsTuning` panel

---

## File Naming Conventions

**Prefixes**:
- `exp_`: Game engine implementation files
- `EXPLORATORY_`: Operator/panel class names
- `EXP_`: Panel/UI class names
- `_`: Private functions/variables

**Patterns**:
- `exp_<feature>.py`: Feature implementation
- `<feature>_definition.py`: Property group definitions
- `<feature>_nodes.py`: Node implementations

---

## Known Limitations & Considerations

**Blender 5.0 Compatibility**:
- Uses preferences persistence workaround (no `AddonPreferences` IDProperties)
- Sentinel enum pattern to avoid empty enum errors
- `mtime` caching for enum loading from .blend files

**Performance**:
- Distance-based culling system (multiprocessing engine - TRUE parallelism)
- Performance culling: 1000+ objects per frame with zero main-thread distance calculations
- **Sprint 1.1 COMPLETE**: Dynamic mesh activation offloaded (distance checks in workers)
- **Sprint 1.2 COMPLETE**: Interaction proximity/AABB checks offloaded (4-34µs per batch)
- Debug output: Toggleable via Developer Tools panel (silent by default)
- Moving platforms: Only vertical and rotational motion transfer (no lateral sliding)

**Physics**:
- Capsule-based collision only (no complex shapes)
- Fixed 30Hz timestep (up to 3 catchup steps)

**Character Customization**:
- Must use compatible armature with default bone structure
- Custom skins must be rigged to compatible skeleton

**Multiprocessing**:
- Workers have NO bpy access (will crash if attempted)
- All data must be picklable (no Blender objects!)
- 1-2 frame latency acceptable for AI/physics predictions
- Not for player input (must be immediate)

---

## Special Files

**combine.py**: Development tool (Tkinter GUI for combining Python files into single text file, excluded from distribution via .gitignore)

**update_addon.py**: Self-update mechanism that downloads updates from exploratory.online, preserves cache/database, handles Windows file locking

**__init__.py**: HAS CRITICAL BPY IMPORT GUARD - DO NOT REMOVE!

---

## Asset Locations

**Default Assets**:
- Character skin/armature: `Exp_Game/exp_assets/Skins/exp_default_char.blend`
- Default armature only: `Exp_Game/exp_assets/Armature/Armature.blend`
- Sounds: `exp_assets/Sounds/` (folder structure present)

**Downloaded Worlds**:
- Temporary location: `Exp_UI/World Downloads/`
- Auto-cleanup on game end

---

## Architecture Philosophy

- **Blender-native**: Uses Blender's data model (no external engine)
- **Modular**: Clear separation between game engine, nodes, and web UI
- **Property-based**: Scene properties for state (not global variables)
- **Timer-driven**: Modal operator with timer (not frame handlers)
- **Fixed timestep**: Consistent physics regardless of render FPS
- **TRUE parallelism**: Multiprocessing engine for heavy computations (NEW!)

---

## Troubleshooting

### Engine Workers Not Starting
**Symptom:** `Workers: 0/4`, `ModuleNotFoundError: No module named '_bpy'`

**Cause:** BPY import guard not working in `__init__.py`

**Fix:**
1. Check `__init__.py` lines 29-58 have the guard
2. Ensure ALL imports are inside the `else` block
3. Restart Blender completely
4. Check System Console for errors

### Engine Jobs Not Processing
**Symptom:** Jobs submitted but no results returned

**Cause:** Job type not implemented in worker

**Fix:**
1. Edit `engine_worker_entry.py`
2. Add handler for your job type in `process_job()`
3. Restart game to reload worker code

### Performance Issues
**Symptom:** Game running slow

**Solutions:**
1. Enable Developer Tools panel → Toggle debug categories to identify bottlenecks
2. Use performance culling entries for distant objects (supports 1000+ objects)
3. Check if engine is processing jobs (toggle `dev_debug_engine`)
4. Reduce physics step count if needed (Character Physics & View panel)
5. Consider offloading heavy calculations to multiprocessing engine

### No Debug Output / Silent Console
**Symptom:** Expected to see debug messages but console is silent

**Cause:** Debug flags are disabled by default (production mode)

**Fix:**
1. Open N-panel → Exploratory → Create tab
2. Open "Developer Tools" panel
3. Enable specific debug categories or use "Enable All Debug Output" toggle
4. Restart game to see debug messages
