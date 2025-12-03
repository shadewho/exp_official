# CLAUDE.md - Exploratory Game Engine

This file provides guidance to Claude Code when working with the Exploratory game engine (Exp_Game module).

---

## 🎯 Core Philosophy

**Testing > Guessing**
It's more important to set up tests, get detailed results, and make informed changes than to poke around aimlessly. Solid foundations with visualized results beat guesswork.

**Visualization > Confusion**
Logs, files, and visual debug tools that help Claude see what's happening are CRITICAL. If Claude can't see it, Claude can't fix it.

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

## 📁 Engine Output System (NEW)

**Location:** `C:\Users\spenc\Desktop\engine_output_files\`

**Files:**
- `kcc_latest.txt` - Current session log
- `kcc_previous.txt` - Previous session (backup)

**Purpose:**
Faster visualization and better detail of what's happening in-game. This is the **primary way** to share debug data with Claude.

**How it works:**
1. Enable "Export Session Log" toggle in dev panel
2. Play game with debug categories enabled
3. Stop game → auto-exports all console output to file
4. User tells Claude: "Read C:\Users\spenc\Desktop\engine_output_files\kcc_latest.txt"
5. Claude analyzes full session timeline

**Goal:** Eventually ALL possible prints go through engine output system for complete session capture.

**Workflow:**
```
Test → Capture Data → Share with Claude → Analyze → Make Changes → Repeat
```

---

## 🎨 Character Visualizer (NEW - 3D Viewport Overlay)

**Location:** `Exp_Game/physics/exp_kcc.py` (GPU rendering)
**Toggle:** `dev_debug_kcc_visual` in N-panel

**What it shows:**
- **Capsule shape** (two spheres - feet + head)
  - Green = grounded
  - Yellow = colliding
  - Red = stuck (depenetrating)
  - Blue = airborne
- **Hit normals** - Cyan arrows showing collision surfaces
- **Ground ray** - Magenta (hit) or purple (miss)
- **Movement vectors** - Green (intended) vs Red (actual)

**Philosophy:** Prints + Visualizer = Sequence Understanding

The visualizer is **coupled with prints** so Claude can see what you see in sequence. It's like having a video, but through data:
```
Frame 1544: Capsule=YELLOW Ray=MAGENTA(0.15m) Normals=2 | [PHYS-STEP] FAILED
Frame 1545: Capsule=BLUE Ray=PURPLE(miss) Normals=0 | [PHYS-GROUND] MISS
```

Claude can read this and visualize exactly what happened, frame by frame.

**Goal:** Continue developing character visualization system to paint a complete picture of physics state.

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
- Export to `kcc_latest.txt`

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

**"Visualization and painting pictures via logs and other results"** - This is paramount.

If Claude can't see what's happening through data, Claude can't help effectively. Every system should be observable, measurable, and exportable.

**The engine output system + character visualizer + detailed logs = complete picture of game state.**

This is how we build a solid game engine.
