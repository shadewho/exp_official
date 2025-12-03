# KCC Physics Engine - Current State & Active Issues

**Status:** Core offload functional, smooth with prints off, collision resolution needs work
**Last Updated:** 2025-12-02
**Session Logs:** `C:\Users\spenc\Desktop\engine_output_files\kcc_latest.txt`

---

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

## 🚨 ACTIVE ISSUES (Priority Order)

### 1. Mid-Height Collision Resolution
**Problem:** Capsule collision detection inconsistent at mid-height (between feet and head spheres)
**Impact:** Character passes through or clips into geometry at torso level
**Root Cause:** Only 2 sphere sweeps (feet + head), no coverage in middle third of capsule

### 2. Dynamic BVH Character Rotation
**Problem:** Dynamic BVHs don't rotate the character when platforms rotate
**Impact:** Character slides off rotating platforms instead of rotating with them
**Location:** Main thread dynamic mesh handling (requires bpy)

### 3. Dynamic BVH Speed & Falling
**Problem:** Dynamic BVHs fail when:
- Character falls onto them from above
- Platform applies speed/acceleration to character
**Impact:** Character falls through or gets stuck in moving platforms

### 4. Sliding Jitter
**Problem:** Wall sliding and slope sliding feel jittery/stuttery
**Impact:** Poor gameplay feel, especially on slopes
**Likely Cause:** Frame-to-frame velocity resolution inconsistency

### 5. Slope Limits Ineffective
**Problem:** Slope angle limits don't prevent uphill movement on steep surfaces
**Impact:** Character can climb walls they shouldn't be able to
**Expected:** `steep_slide_gain` and `steep_min_speed` should force sliding, not allow climbing

### 6. Capsule Strength Inconsistency
**Problem:** Overall collision strength varies by contact height (mid/head especially weak)
**Impact:** Unpredictable collision response, character squeezes through tight spaces
**Related:** Issue #1 (mid-height coverage)

---

## 🛠️ DEVELOPER SYSTEM WORKFLOW (CRITICAL)

**New Standard:** Visualize → Export → Analyze → Fix → Repeat

### Debug Output Pipeline
```
1. Enable debug categories (N-panel → Developer Tools)
2. Enable "Export Session Log" toggle
3. Play game (prints go to console + session log)
4. Stop game → auto-exports to C:\Users\spenc\Desktop\engine_output_files\kcc_latest.txt
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
│ Dynamic mesh (BVH)      │           │ Wall slide                   │
│ Platform carry          │           │ Spatial grid (DDA)           │
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
- Platform rotation = angular velocity applied to character
- Platform carry happens AFTER worker result (main thread)
- Falling collision = need predictive check, not reactive
- Speed application = velocity delta, not position offset

---

## 🎯 Next Steps

**Immediate priorities:**

1. **Mid-height collision** - Add third sphere sweep at capsule center.... maybe!! ask me about this later... is the third nessesary? or could we just bridge with a ray or something between foot and head capsules? i need performance and i need to make sure we are always optimally setting up high performance systems. is there a smart way to approach a stronger mid/head collisions? feet collisions work great **ask me about this next session!
2. **Slope limits** - Fix uphill blocking logic in worker
3. **Sliding jitter** - Investigate velocity resolution smoothing
4. **Dynamic BVH rotation** - Add angular carry in platform system
5. **Dynamic BVH falling** - Add predictive collision before position update

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

**What Needs Work:**
- ❌ Mid-height collision coverage
- ❌ Dynamic platform rotation carry
- ❌ Dynamic platform falling collision
- ❌ Slope limit enforcement
- ❌ Sliding smoothness
- ❌ Collision strength consistency

**Philosophy:**
Data-driven development. Visualize everything. Never guess. Test methodically. Respect the engine.
