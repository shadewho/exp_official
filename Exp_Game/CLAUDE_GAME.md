# CLAUDE.md - Exploratory Game Engine

This file provides guidance to Claude Code when working with the Exploratory game engine (Exp_Game module).

---

## 🎯 Core Philosophy

**Testing > Guessing**
It's more important to set up tests, get detailed results, and make informed changes than to poke around aimlessly. Solid foundations with visualized results beat guesswork.

**Visualization = Understanding**
The #1 goal is to **paint a complete picture** of what's happening in the game through logs and visualization. When you see something on screen, Claude should see it in the logs.

**Critical Development Goal:**
- Visualizer shows RED capsule → Logs show `STUCK🔴`
- Visualizer shows YELLOW capsule → Logs show `BLOCK🟡`
- Visualizer shows GREEN capsule → Logs show `GROUND🟢`
- Visualizer shows BLUE capsule → Logs show `AIR🔵`

**What you see = What Claude sees.**

Logs, files, and visual debug tools that help Claude see what's happening are CRITICAL. If Claude can't see it through data, Claude can't fix it. Always develop the logging and developer systems to create a 1:1 mapping between visual state and log output.

**Tests → Results → Changes**
This is the development loop. Not: Try → Hope → Try Again.

---

## 🚨 CRITICAL PRINCIPLE: Free the Main Thread

**The #1 Rule:** If a calculation can reasonably run off the main thread, it MUST be offloaded.

**Why:**
- Main thread handles Blender's modal operator (30Hz game loop)
- Python GIL blocks true threading
- Heavy computation on main thread = stuttering, lag, poor gameplay
- Smooth gameplay requires responsive main thread

**What stays on main thread:**
- ✅ Reading Blender data (bpy access)
- ✅ Writing to Blender (bpy.data modifications)
- ✅ Input handling
- ✅ Coordinating engine jobs
- ✅ Fast BVH operations (already optimized)

**What goes to worker engine:**
- ✅ Physics calculations (KCC)
- ✅ Raycasts against static geometry
- ✅ Distance checks (culling)
- ✅ Pathfinding
- ✅ AI decisions
- ✅ Any computation that doesn't need bpy

---

## ⚙️ The Worker Engine (Companion to Modal)

**Location:** `Exp_Game/engine/`

**What it is:**
A multiprocessing engine with 4 worker processes that runs in sync with the modal operator. It's a **companion**, not a separate system.

**Architecture:**
```
Main Thread (Modal - 30Hz)          Workers (4 cores)
┌─────────────────────────┐         ┌──────────────────────────┐
│  Snapshot data     ────►├────────►│  Heavy computation       │
│  Submit job             │         │  NO bpy access!          │
│  Poll results      ◄────┤◄────────┤  Return pickled data     │
│  Apply to Blender       │         └──────────────────────────┘
└─────────────────────────┘
```

**Current Production Workloads:**
- `KCC_PHYSICS_STEP` - Full character physics
- `CAMERA_OCCLUSION_FULL` - Camera raycast occlusion
- `CULL_BATCH` - Performance culling (1000+ objects)
- `DYNAMIC_MESH_ACTIVATION` - Distance-based mesh gating
- `INTERACTION_CHECK_BATCH` - Proximity checks

**Golden Rule:** Don't overwhelm the engine. Game runs at 30Hz - don't submit more jobs than workers can handle.

---

## 🛠️ Developer Module (N-Panel Toggles)

**Location:** `Exp_Game/developer/`

**Purpose:** Every debug system needs a toggle in the N-panel.

**When developing/testing:**
1. Add property to `dev_properties.py`
2. Add toggle to `dev_panel.py`
3. Use property to gate debug output
4. Default to `False` (silent in production)

**Structure:**
```python
# Property
bpy.types.Scene.dev_debug_my_system = bpy.props.BoolProperty(
    name="My System Debug",
    description="Debug output for my system",
    default=False
)

# In code
if context.scene.dev_debug_my_system:
    print("[MySystem] Debug message")
```

---

## 📊 Print Policy (MANDATORY)

**ALL prints for the game engine need:**

1. **A relevant toggle** - No print without a toggle
2. **A category tag** - Format: `[CATEGORY] message`
3. **A frequency (Hz)** - Throttle output to prevent spam

**Example:**
```python
# Property with Hz control
bpy.types.Scene.dev_debug_physics_ground_hz = bpy.props.IntProperty(
    name="Ground Hz",
    description="Output frequency (1-30 Hz)",
    default=5,
    min=1,
    max=30
)

# Throttled output
if context.scene.dev_debug_physics_ground:
    if should_print_at_hz("ground", scene.dev_debug_physics_ground_hz):
        print(f"[PHYS-GROUND] status | data")
```

**Why Hz throttling:**
- Game runs at 30 FPS
- Printing every frame = 30 messages/sec (too much)
- 5Hz = 5 messages/sec (readable)
- User can increase to 30Hz if they need every frame

---

## 📁 Fast Buffer Logger System (CRITICAL)

**⚠️ PERFORMANCE CRITICAL**: Console `print()` statements during gameplay cause severe performance degradation (~1000μs per call). This was causing frame drops, stuttering, and unacceptable game performance.

**📖 Full Documentation**: See `developer/CLAUDE_LOGGER.md` for complete logger system documentation.

### Quick Summary

**Location:** `C:\Users\spenc\Desktop\engine_output_files\`

**Files:**
- `diagnostics_latest.txt` - Game diagnostics log (exported after game stops)

**Key Rules:**
1. ✅ **Use logger for in-game diagnostics** - Anything in the game loop MUST use `log_game()` (1000x faster than print)
2. ✅ **Use print for one-time events** - Startup, errors, user actions (outside game loop) can use print()
3. ✅ **Master Hz control** - Set logging frequency (1-30 Hz) to control output verbosity
4. ✅ **Export on demand** - Enable "Export Diagnostics Log" toggle to save logs to file

### Why Logger vs Print?

| Aspect | Console Print | Fast Buffer Logger |
|--------|---------------|-------------------|
| **Speed** | ~1000μs (1 millisecond) | ~1μs (1 microsecond) |
| **Impact** | 100 prints/frame = **3 FPS** | 100 logs/frame = **minimal** |
| **Gameplay** | Severe stuttering, unplayable | Smooth, full diagnostics |
| **Performance** | 1x baseline | **1000x faster** |

**How it works:**
1. Enable "Export Diagnostics Log" toggle in dev panel
2. Set Master Hz (1 Hz recommended for most debugging)
3. Play game with debug categories enabled
4. Logs written to memory buffer (zero I/O during gameplay)
5. Stop game → auto-exports all diagnostics to file
6. User tells Claude: "Read C:\Users\spenc\Desktop\engine_output_files\diagnostics_latest.txt"
7. Claude analyzes full session timeline with frame numbers and timestamps

**Workflow:**
```
Test → Capture Data → Share with Claude → Analyze → Make Changes → Repeat
```

**Critical**: ALL in-game diagnostics now go through the fast buffer logger. Never use print() in the game loop.

---

## 🎨 Character Visualizer (3D Viewport Overlay)

**Location:** `Exp_Game/physics/exp_kcc.py` (GPU rendering - single batched draw call)
**Toggle:** `dev_debug_kcc_visual` in N-panel

**What it shows:**
- **Capsule shape** (two spheres - feet + head)
  - 🟢 Green = grounded
  - 🟡 Yellow = colliding
  - 🔴 Red = stuck (depenetrating)
  - 🔵 Blue = airborne
- **Hit normals** - Cyan arrows showing collision surfaces
- **Ground ray** - Magenta (hit) or purple (miss)
- **Movement vectors** - Green (intended) vs Red (actual)

**Philosophy: Logs Mirror Visual State**

The visualizer and logs are **perfectly synchronized** - what you see on screen is exactly what Claude sees in logs:

```
[KCC F1544 T51.467s] BLOCK🟡 pos=(10.5,5.2,3.1) step=False | 120us 4rays 12tris
[KCC F1545 T51.500s] AIR🔵 pos=(10.5,5.2,2.9) step=False | 115us 4rays 12tris
[KCC F1546 T51.533s] GROUND🟢 pos=(10.5,5.2,3.0) step=False | 118us 4rays 12tris
[KCC F1547 T51.567s] STUCK🔴 pos=(10.5,5.2,3.1) step=False | 142us 8rays 24tris
```

**What you see = What Claude sees.**

When you report "the capsule turned red and got stuck," Claude can read the logs and see `STUCK🔴` entries with the exact frame, position, and diagnostic data. It's like having a video, but through data.

**Goal:** Continue developing visualization and logging systems to paint a complete picture of game state. Every visual element should have a corresponding log entry.

---

## 🧪 Development Mindset

### ✅ DO:

- **Try things** - Don't be too conservative
- **Take bird's-eye view** - Step back and assess honestly
- **Visualize everything** - Logs, overlays, exports
- **Test systematically** - Set up test, capture results, analyze
- **Build diagnostic tools** - More tools = faster iteration
- **Free the main thread** - Offload everything reasonable

### ❌ DON'T:

- **Guess without data** - Get logs first
- **Poke around aimlessly** - Test methodically
- **Add prints without toggles** - Always gate with dev properties
- **Overwhelm the engine** - Respect 30Hz budget
- **Skip visualization** - If you can't see it, you can't fix it
- **Be afraid to refactor** - Honest assessment > false progress

---

## 🔄 The Testing Loop (Gold Standard)

**1. Identify Problem**
- User describes bug
- User provides screenshot or description

**2. Build Diagnostic Tools**
- Add visualization (if needed)
- Add detailed logging (with toggle + Hz)
- Add engine output capture

**3. Reproduce and Capture**
- User plays game with debug enabled
- System captures full timeline
- Export to `diagnostics_latest.txt`

**4. Analyze**
- Claude reads exported log
- Claude sees frame-by-frame sequence
- Claude identifies root cause

**5. Make Changes**
- Targeted fix based on data
- Not guesswork

**6. Test Again**
- Verify fix with same diagnostic tools
- Compare before/after logs
- Confirm behavior change

**Repeat until solved.**

---

## 🎯 Success Metrics

**Good development session:**
- ✅ Clear problem identified
- ✅ Diagnostic tools built/used
- ✅ Data captured and analyzed
- ✅ Root cause found
- ✅ Fix applied
- ✅ Behavior verified

**Bad development session:**
- ❌ "Try this and see if it helps"
- ❌ No logs or visualization
- ❌ Guessing at root cause
- ❌ Can't verify if fix worked
- ❌ Confusion about what's happening

---

## 📚 Key Files Reference

**Engine:**
- `engine/engine_core.py` - Main thread manager
- `engine/engine_worker_entry.py` - Worker process handler
- `engine/engine_types.py` - Job/result data structures

**Physics:**
- `physics/exp_kcc.py` - KCC controller (main thread + visualization)
- Physics offloaded to: `engine/engine_worker_entry.py` (KCC_PHYSICS_STEP handler)

**Developer Tools:**
- `developer/dev_properties.py` - Debug toggle properties
- `developer/dev_panel.py` - N-panel UI

**Modal:**
- `modal/exp_modal.py` - Game loop operator (30Hz)
- `modal/exp_loop.py` - Per-frame update logic

---

## 💡 Remember

**"Paint a complete picture - What you see = What Claude sees"** - This is paramount.

The goal is always to develop logs and visualization systems so that when you see something on screen, Claude sees it in the logs with perfect clarity. Visual state should map 1:1 to log output.

**Core principle:**
- 🔴 RED capsule on screen → `STUCK🔴` in logs
- 🟡 YELLOW capsule on screen → `BLOCK🟡` in logs
- 🟢 GREEN capsule on screen → `GROUND🟢` in logs
- 🔵 BLUE capsule on screen → `AIR🔵` in logs

If Claude can't see what's happening through data, Claude can't help effectively. Every system should be observable, measurable, and exportable. Every visual element should have corresponding diagnostic output.

**Fast buffer logger + GPU visualizer + frame-accurate logs = complete picture of game state.**

This is how we build a solid, debuggable game engine - by ensuring transparency through data.
