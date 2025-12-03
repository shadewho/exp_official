# KCC Physics Engine - Worker Offload Context

**Status:** Functional base, ground-level step-up broken
**Last Updated:** 2025-12-02
**Session Logs:** `C:\Users\spenc\Desktop\engine_output_files\kcc_latest.txt`

---

## 🚨 CRITICAL ISSUE - Ground-Level Platforms

**Problem:** Step-up doesn't work for platforms touching the ground (z=1.01 base).

**What Works:**
- ✅ Elevated platforms (z=1.0 base → z=1.58 top) - character walks onto them
- ✅ Step-up triggers and detects collision on some frames

**What Doesn't Work:**
- ❌ Ground-level platforms (z=1.01 base → z=1.22 top) - no interaction
- ❌ Small obstacles sitting on ground - character walks through them
- ❌ Step-up reports `climb=0.000m` (finds wrong ground)

**Root Cause Identified:**
```
Character feet at z=1.01
Feet sphere center at z=1.31 (pos.z + radius)
Ground platform top at z=1.22
Platform vertical sides: z=1.01 to z=1.22

→ Feet sphere checks at z=1.31, ABOVE the short platform entirely!
→ Collision detection misses platforms shorter than radius (0.30m)
```

**Evidence from logs:**
- Frame 43: `27 h_tris | clear 1 planes` - collision detected
- Step-up triggers: `attempted SUCCESS climb=0.000m`
- Drop-down finds z=1.01 instead of z=1.22 (wrong ground)

**Why step-up fails:**
1. Horizontal collision sometimes works (sees 27 tris, finds 1 hit)
2. Step-up lifts character by 0.5m
3. Moves forward `radius * 1.5 = 0.45m`
4. **Drops down, finds ORIGINAL ground at z=1.01 (not platform top at z=1.22)**
5. Result: `climb=0.000m` (no height gain)

**Why it overshoots:**
Small platforms (<0.5m wide) are completely passed over during forward movement.

---

## Debug Flow - ALWAYS USE THIS

**1. Enable Session Log Export:**
- Developer Tools panel → "Export Session Log" toggle
- Plays game → Stop → Auto-exports to `C:\Users\spenc\Desktop\engine_output_files\kcc_latest.txt`

**2. Enable Debug Categories:**
- `dev_debug_kcc_offload` - Worker physics output (CRITICAL)
- All output goes to console AND session log file

**3. Read Logs in Claude:**
```
Read: C:\Users\spenc\Desktop\engine_output_files\kcc_latest.txt
Look for:
- [PHYS-CAPSULE] ... | X h_tris - how many triangles tested
- [PHYS-STEP] attempted SUCCESS climb=X.XXXm - step-up results
- [PHYS-GROUND] ON_GROUND dist=X.XXXm - ground detection
```

**4. Analyze Frame-by-Frame:**
- Frame headers show visual state: `[VIS] GREEN(grounded) | 2 cyan arrows`
- Compare triangles found vs collision detected
- Check if step-up triggers and what climb height is

**5. CRITICAL - Don't Guess:**
- ❌ Making changes without seeing logs
- ❌ Adding debug prints (causes lag/crashes)
- ✅ Read session log file FIRST
- ✅ Understand what's actually happening
- ✅ Make targeted changes based on data

---

## Architecture

```
Main Thread (exp_kcc.py)
├─ Submit KCC_PHYSICS_STEP job
├─ Poll with 3ms timeout
└─ Apply result to Blender

Worker (engine_worker_entry.py)
├─ Input → velocity
├─ Horizontal sweep (2 spheres: feet at z+radius, head at z+height-radius)
│   └─ sphere_triangle_intersect_sweep (line 1080) - CRITICAL
├─ Step-up attempt (line 1827)
│   ├─ Lift by step_height (0.5m)
│   ├─ Move forward radius*1.5 (0.45m)
│   └─ Drop down to find ground ← FINDING WRONG GROUND
├─ Ground detection (line 1781) - sphere cast down
└─ Return state
```

**Key Files:**
- `engine_worker_entry.py:1080` - `sphere_triangle_intersect_sweep()`
- `engine_worker_entry.py:1501` - `capsule_sweep_horizontal()` (2 spheres)
- `engine_worker_entry.py:1827` - `try_step_up()`
- `engine_worker_entry.py:1781` - `detect_ground()`

---

## What We Tried (and Why It Failed)

**Attempt 1: Proximity detection for parallel movement**
- Added check when sphere moves parallel to surface
- Result: Detected ground beneath character, triggered step-up every frame
- Caused: Massive performance hit, timeouts, jitter
- **REVERTED**

**Attempt 2: Third "low" sphere sweep**
- Added sphere at z=1.16 (below feet sphere) to catch short obstacles
- Result: Blender crash
- **REVERTED**

**Attempt 3: Reduce step-up forward distance**
- Changed from `radius * 3.0` to `radius * 0.7` (0.9m → 0.21m)
- Theory: Don't overshoot small platforms
- Result: Blender crash (combined with low sweep)
- **REVERTED to `radius * 1.5` (0.45m)**

**Current State:**
- Back to stable baseline: 2 sphere sweeps, normal step-up
- Ground-level platforms still don't work, but system is stable

---

## Next Session - Start Here

**The Real Problem:**
Step-up drop-down phase uses `detect_ground()` which does a sphere sweep DOWN from the new position. This sweep is finding the WRONG ground surface (original z=1.01 instead of platform top z=1.22).

**Why?**
Two possibilities:
1. **Overshoot:** Forward movement (0.45m) moves past the small platform entirely
2. **Sweep passes through:** Sphere sweep downward passes through thin platform geometry

**What to Investigate:**
1. Add **temporary** debug output ONLY to `try_step_up()` at line 1827:
   - Print lifted position
   - Print new_raised_pos after forward movement
   - Print ground_result["ground_z"] found by drop-down
   - This will show if we're overshooting or missing geometry

2. Check platform geometry in Blender:
   - Is it a solid box with thickness?
   - Or a thin plane?
   - What are the exact dimensions?

3. Test elevated platform that WORKS:
   - Why does it work?
   - What's different about the geometry or height?

**Approach:**
- ✅ Make changes based on DATA from logs
- ✅ One change at a time
- ✅ Test each change
- ❌ Don't change multiple things at once (can't tell what worked)

**Important:**
The elevated platform (z=1.0 to z=1.58) WORKS perfectly. The system CAN detect and step onto platforms. The issue is specifically with SHORT platforms (height < radius) that sit ON the ground. Focus on why the drop-down phase finds the wrong surface.
