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

**Philosophy:**
Data-driven development. Visualize everything and log everything in the log output file. Never guess. Test methodically. Respect the engine.
