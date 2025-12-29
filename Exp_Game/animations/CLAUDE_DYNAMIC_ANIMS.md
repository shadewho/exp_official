# Dynamic Animation System

**Status**: In Development
**Location**: Developer Tools Panel → Animation 2.0

---

## Vision

Simplify character animation so users don't have to spend endless hours animating.
Instead of hand-animating every action, users define **poses** and the system
figures out how to move between them **naturally and anatomically correct**.

```
USER DEFINES:                    SYSTEM HANDLES:
┌─────────────────┐              ┌─────────────────────────────────┐
│  Pose A: Idle   │              │  - How to get from A to B       │
│  Pose B: Grab   │   ───────►   │  - Anatomically valid path      │
│  Pose C: Throw  │              │  - No impossible joint angles   │
└─────────────────┘              │  - Full body coordination       │
                                 │  - Real-time, any starting pose │
                                 └─────────────────────────────────┘
```

---

## Core Goals

1. **Pose-to-Pose Animation**
   - User captures key poses (idle, reach, grab, throw, etc.)
   - System dynamically blends between them at runtime
   - Works from ANY current character state, not just predefined start poses

2. **Anatomically Constrained Movement**
   - Rig knows what movements are physically possible
   - Joint limits prevent impossible rotations
   - Coupling rules ensure body parts move together naturally
   - No twisted bones, no broken elbows, no impossible spine bends

3. **Full-Body IK Integration**
   - Not just limbs - the ENTIRE body participates
   - Spine, hips, shoulders all coordinate with limb movement
   - When arm reaches, shoulder rotates, spine may lean, etc.

4. **Engine-Computed (Worker-Based)**
   - ALL math runs in engine workers, not main thread
   - Same code path for testing AND gameplay
   - Scalable to many characters

---

## Animation 2.0 Test Suite

**IMPORTANT**: The Animation 2.0 section in Developer Tools is a TEST SUITE.
It is NOT production code. It exists to:

- Develop and validate the dynamic animation system
- Test pose blending with real armatures
- Debug IK and constraint issues
- Iterate on the movement profile system

When the system is ready, it will be exposed through the Node System for
actual gameplay use. The test suite ensures everything works before that.

### Test Suite Principles

1. **Always Engine-Backed**
   - Every test uses the worker system (POSE_BLEND_COMPUTE, IK_SOLVE_BATCH)
   - No main-thread math - same as production
   - If it works in tests, it works in game

2. **Real Armature Testing**
   - Tests run on the actual target armature
   - Real bone hierarchies, real transforms
   - Catches rig-specific issues early

3. **Diagnostic Logging**
   - All worker jobs log to diagnostics
   - Can trace exactly what's computed
   - Essential for debugging movement issues

---

## System Architecture

### Phase 1: Rig Structure (Complete)

The rig structure and joint limits are defined in:
- `animations/rig.md` - Full rig documentation with joint limits
- `engine/animations/default_limits.py` - Default joint limits as Python data

```
RIG DATA
├── Bone Hierarchy        - Parent/child relationships
├── Bone Lengths          - Precise measurements for IK
├── IK Chain Definitions  - arm_L, arm_R, leg_L, leg_R
└── Joint Rotation Limits - Min/max angles per bone per axis
```

### Phase 2: Movement Profiles (Current)

Define what each joint CAN and CANNOT do:

```
MOVEMENT PROFILE
├── Joint Limits          - Min/max angles per axis
│   ├── LeftForeArm: X=[0, 145°]      (elbow can't bend backwards)
│   ├── Spine1: X=[-30°, 45°]         (forward/back lean limits)
│   └── LeftArm: full ball joint with soft limits
│
├── Coupling Rules        - What moves together
│   ├── Arm raise > 90° → Shoulder blade rotates 30%
│   ├── Spine bend → Hips counter-rotate 20%
│   └── Head turn → Neck follows 50%
│
└── Rest Preferences      - Natural resting positions
    ├── Elbows: slight bend (5°)
    ├── Knees: slight bend (3°)
    └── Spine: slight S-curve
```

### Phase 3: Constrained Blending

Apply movement profile during pose transitions:

```
CONSTRAINED BLEND PIPELINE
┌────────────────────────────────────────────────────────────────┐
│ 1. Get current pose (any state)                                │
│ 2. Get target pose (user-defined)                              │
│ 3. For each frame of blend:                                    │
│    a. Interpolate all bones (slerp/lerp)                       │
│    b. Apply joint limits (clamp to valid range)                │
│    c. Apply coupling rules (adjust related bones)              │
│    d. Solve IK for target points (hands, feet)                 │
│    e. Validate full pose (no collisions, no impossibles)       │
│ 4. Apply result to armature                                    │
└────────────────────────────────────────────────────────────────┘
```

### Phase 4: Full-Body IK

Extend IK beyond just limbs:

```
FULL-BODY IK CHAINS
├── Limbs (current)
│   ├── arm_L: Shoulder → Arm → ForeArm → Hand
│   ├── arm_R: Shoulder → Arm → ForeArm → Hand
│   ├── leg_L: Thigh → Shin → Foot
│   └── leg_R: Thigh → Shin → Foot
│
├── Spine Chain (planned)
│   └── Hips → Spine → Spine1 → Spine2 → Chest
│
├── Head Chain (planned)
│   └── Chest → NeckLower → NeckUpper → Head
│
└── Full Body (goal)
    └── All chains coordinated, root motion included
```

---

## Worker Job Types

All computation happens in engine workers:

| Job Type | Purpose |
|----------|---------|
| `POSE_BLEND_COMPUTE` | Blend two poses with optional IK |
| `IK_SOLVE_BATCH` | Solve IK for multiple chains |
| `ANIMATION_COMPUTE_BATCH` | Blend keyframed animations |
| `CONSTRAINED_BLEND` | (Planned) Blend with joint limits |
| `FULL_BODY_IK` | (Planned) Whole-body IK solve |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      MAIN THREAD (LIGHT)                        │
│  - Cache poses at play start                                    │
│  - Submit jobs each frame                                       │
│  - Poll for results                                             │
│  - Apply pre-computed transforms to bones                       │
│  - NO MATH on main thread                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ submit jobs / receive results
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ENGINE WORKERS (HEAVY)                        │
│  - Quaternion slerp (vectorized numpy)                          │
│  - Location/scale lerp                                          │
│  - Joint limit clamping                                         │
│  - Coupling rule application                                    │
│  - IK solving (two-bone, multi-chain)                           │
│  - Full pose validation                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Files

| File | Purpose |
|------|---------|
| `animations/test_panel.py` | Animation 2.0 test suite UI and logic |
| `animations/pose_library.py` | Pose capture and storage |
| `animations/runtime_ik.py` | IK state management |
| `animations/rig.md` | Complete rig documentation with joint limits |
| `engine/worker/entry.py` | Worker job handlers |
| `engine/animations/ik.py` | IK solvers (worker-safe) |
| `engine/animations/blend.py` | Blend math (worker-safe) |
| `engine/animations/joint_limits.py` | Joint limit enforcement (worker-safe) |
| `engine/animations/default_limits.py` | Default rig joint limits |
| `developer/rig_visualizer.py` | Debug visualization |

---

## Development Roadmap

### Complete
- [x] Worker-based pose blending (POSE_BLEND_COMPUTE)
- [x] Worker-based IK solving (IK_SOLVE_BATCH)
- [x] Rig structure documentation (rig.md)
- [x] Joint limit definitions (default_limits.py)
- [x] Joint limit enforcement during blend (CACHE_JOINT_LIMITS)
- [x] Bone orientation data extraction and documentation (rig.md)
- [x] Two-bone IK with correct rotation math (ik_solver.py)
- [x] Self-verifying diagnostic logging pattern
- [x] Pose validation system (pose_validator.py)

### In Progress
- [ ] Pole vector control (knee/elbow bend direction)
- [ ] Coupling rules (bones that move together)
- [ ] Spine IK chain
- [ ] Full-body coordination

### Future
- [ ] Valid pose recording (learn from examples)
- [ ] Automatic path finding between poses
- [ ] Node System integration for gameplay
- [ ] Multi-character support

---

## CRITICAL: Self-Verifying Diagnostics (2025-12-26)

**LESSON LEARNED**: Diagnostic logs must VERIFY outputs, not just log inputs.

When debugging IK or any rotation-based system, **do not rely on screenshots or user feedback**. The logs themselves must tell you if something worked or failed.

### The Wrong Way (input-only logging)
```
log("applying rotation quaternion: (0.7, 0.1, 0.2, 0.3)")
log("target direction: (0.5, 0.8, 0.1)")
# No way to know if it actually worked!
```

### The Right Way (self-verifying logging)
```python
# BEFORE
dir_before = (tail - head).normalized()
log(f"BEFORE: dir=({dir_before.x:.3f},{dir_before.y:.3f},{dir_before.z:.3f})")
log(f"WANT:   dir=({target_dir.x:.3f},{target_dir.y:.3f},{target_dir.z:.3f})")

# Apply rotation...
bpy.context.view_layer.update()

# AFTER - VERIFY IT WORKED
dir_after = (tail - head).normalized()
angle_error = degrees(acos(dir_after.dot(target_dir)))
log(f"AFTER:  dir=({dir_after.x:.3f},{dir_after.y:.3f},{dir_after.z:.3f})")
log(f"VERIFY: angle_error={angle_error:.1f}deg")

if angle_error > 5:
    log("!!! ROTATION FAILED - bone not pointing at target !!!")
```

This pattern immediately reveals if the math is wrong - no screenshots needed.

---

## Bone Orientation Data (rig.md)

The `rig.md` file now contains **bone local axis orientations** - critical data for IK.

### Why This Matters
Each bone has its own local coordinate system:
- **Y-axis**: Points along the bone (head → tail)
- **X-axis**: Perpendicular - often the "twist" axis
- **Z-axis**: Perpendicular - often the "bend" axis

For IK, you MUST know which axis to rotate around:
- **Elbows**: Bend around forearm's local X or Z axis
- **Knees**: Bend around shin's local X axis

### Key Insight: Pole Vectors
The **pole vector** controls where the elbow/knee points:
- Legs: pole = `(0, 1, 0)` → knees point FORWARD
- Arms: pole = `(0, -1, 0)` → elbows point BACKWARD

Without the pole vector, `rotation_difference()` takes the shortest path, which may flip the joint in unexpected directions.

### Bone Orientation Table
See `rig.md` section "Bone Local Axis Orientations" for the complete table of every bone's X/Y/Z axes in world space at rest pose.

---

## IK Rotation Math (WORKING SOLUTION)

After many failed attempts, this is the correct approach for rotating a bone to point at a target:

```python
# 1. Get bone's current Y-axis in world space
bone_world = arm_matrix @ pose_bone.matrix
bone_y_world = Vector((bone_world[0][1], bone_world[1][1], bone_world[2][1])).normalized()

# 2. Get world rotation needed (shortest path)
world_rotation = bone_y_world.rotation_difference(target_dir)

# 3. Apply to get new world orientation
bone_world_quat = bone_world.to_quaternion()
new_world_quat = world_rotation @ bone_world_quat

# 4. Convert to local pose rotation
if pose_bone.parent:
    parent_world_quat = (arm_matrix @ pose_bone.parent.matrix).to_quaternion()
    parent_rest = pose_bone.parent.bone.matrix_local
    bone_rest_in_parent = parent_rest.inverted() @ pose_bone.bone.matrix_local
    bone_rest_in_parent_quat = bone_rest_in_parent.to_quaternion()

    rotation = bone_rest_in_parent_quat.inverted() @ parent_world_quat.inverted() @ new_world_quat
else:
    bone_rest_world_quat = (arm_matrix @ pose_bone.bone.matrix_local).to_quaternion()
    rotation = bone_rest_world_quat.inverted() @ new_world_quat

pose_bone.rotation_quaternion = rotation
```

### Why Previous Attempts Failed
1. **Matrix-based approach**: Building rotation matrices and transforming between spaces had sign/order errors
2. **rotation_difference alone**: Gives shortest path but doesn't account for bone's rest orientation
3. **Missing view_layer.update()**: Blender doesn't update bone positions until you call this

### Key Files
| File | Purpose |
|------|---------|
| `animations/ik_solver.py` | Two-bone IK solver with working rotation math |
| `animations/ik_state.py` | IK state capture and analysis |
| `animations/pose_validator.py` | Pose validation using joint limits |
| `animations/rig_probe.py` | Rig diagnostics and bone orientation dump |

---

## Design Rules

1. **MAIN THREAD FREE**
   - Only submit jobs and apply results
   - No slerp, no IK math, no limit checking on main thread

2. **TEST = PRODUCTION**
   - Animation 2.0 tests use exact same code as gameplay will
   - If it works in test suite, it works in game

3. **ANATOMICALLY VALID**
   - Every frame must be a possible human pose
   - No intermediate frames with broken joints

4. **FULL BODY**
   - Not just limbs - spine, hips, head all participate
   - Character moves as a unified whole

5. **USER SIMPLICITY**
   - User defines poses, not animations
   - System handles the movement between them

---

## Full-Body IK System (2025-12-27)

### Architecture

The Full-Body IK system coordinates the entire skeleton, not just isolated limbs.

**Rig Hierarchy:**
```
Root (world anchor at Z=0) ← IK targets relative to this
└── Hips (pelvis control) ← can translate/rotate for crouch/lean
    ├── Spine → Spine1 → Spine2 ← leans toward reach targets
    │   ├── NeckLower → NeckUpper → Head ← look-at
    │   ├── LeftShoulder → Arm → ForeArm → Hand ← arm IK
    │   └── RightShoulder → Arm → ForeArm → Hand ← arm IK
    ├── LeftThigh → Shin → Foot ← leg IK (grounded)
    └── RightThigh → Shin → Foot ← leg IK (grounded)
```

**Solve Order:**
1. **Hips Position** - Center of mass, crouch amount
2. **Leg IK** - Ground feet (compensate for Hips movement)
3. **Spine Chain** - Lean toward hand targets
4. **Arm IK** - Reach targets given new shoulder positions
5. **Head/Neck** - Look-at target

### IK Modes

| Mode | Description |
|------|-------------|
| `FULL_BODY` | Whole skeleton responds to all targets |
| `TWO_BONE` | Single limb chain (legacy, building block) |
| `FOOT_GROUND` | Keep feet planted while body moves |
| `LOOK_AT` | Head/neck tracking only |

### Self-Verifying Diagnostics

**CRITICAL**: The logs must tell us if the solve worked WITHOUT screenshots.

```
[FULL-BODY-IK F0001] ════════════════════════════════════════════
[FULL-BODY-IK F0001] SOLVE START
[FULL-BODY-IK F0001] ────────────────────────────────────────────
[FULL-BODY-IK F0001] BEFORE:
[FULL-BODY-IK F0001]   Root: pos=(0.00, 0.00, 0.00)
[FULL-BODY-IK F0001]   Hips: pos=(0.00, 0.06, 1.00) rel_to_root
[FULL-BODY-IK F0001]   Spine_lean: 0.0° (upright)
[FULL-BODY-IK F0001]   L_foot: (−0.10, 0.00, 0.00)
[FULL-BODY-IK F0001]   R_foot: (+0.10, 0.00, 0.00)
[FULL-BODY-IK F0001]   L_hand: (−0.50, 0.00, 1.20)
[FULL-BODY-IK F0001]   R_hand: (+0.50, 0.00, 1.20)
[FULL-BODY-IK F0001] ────────────────────────────────────────────
[FULL-BODY-IK F0001] TARGETS:
[FULL-BODY-IK F0001]   Hips: drop=0.30m (crouch)
[FULL-BODY-IK F0001]   L_foot: (−0.10, 0.00, 0.00) GROUNDED
[FULL-BODY-IK F0001]   R_foot: (+0.10, 0.00, 0.00) GROUNDED
[FULL-BODY-IK F0001]   R_hand: (+1.00, 0.50, 1.00) REACH
[FULL-BODY-IK F0001] ────────────────────────────────────────────
[FULL-BODY-IK F0001] SOLVING:
[FULL-BODY-IK F0001]   [1] Hips: (0.00, 0.06, 1.00) → (0.00, 0.06, 0.70)
[FULL-BODY-IK F0001]   [2] L_leg IK: thigh=45° shin=90° foot_err=0.1cm ✓
[FULL-BODY-IK F0001]   [3] R_leg IK: thigh=45° shin=90° foot_err=0.2cm ✓
[FULL-BODY-IK F0001]   [4] Spine: lean=15° toward reach
[FULL-BODY-IK F0001]   [5] R_arm IK: reach_err=2.5cm (at limit) ⚠
[FULL-BODY-IK F0001] ────────────────────────────────────────────
[FULL-BODY-IK F0001] AFTER:
[FULL-BODY-IK F0001]   Hips: (0.00, 0.06, 0.70) ✓ dropped 0.30m
[FULL-BODY-IK F0001]   L_foot: (−0.10, 0.00, 0.001) err=0.1cm ✓
[FULL-BODY-IK F0001]   R_foot: (+0.10, 0.00, 0.002) err=0.2cm ✓
[FULL-BODY-IK F0001]   R_hand: (+0.95, 0.48, 0.98) err=2.5cm ⚠
[FULL-BODY-IK F0001]   Joint_limits: 0 violations ✓
[FULL-BODY-IK F0001] ────────────────────────────────────────────
[FULL-BODY-IK F0001] RESULT: SUCCESS (4/5 constraints, 1 at limit)
[FULL-BODY-IK F0001] ════════════════════════════════════════════
```

**Key Metrics (must be in logs):**
- Position error per end-effector (cm)
- Joint limit violations (count + which joints)
- Solve time (µs)
- Constraint satisfaction ratio

### GPU Visualization

The rig visualizer must show Full-Body IK state:

| Element | Color | Description |
|---------|-------|-------------|
| Root anchor | White | World anchor position |
| Hips control | Yellow | Pelvis position + translation from rest |
| Spine lean | Orange arrow | Direction torso is leaning |
| Foot targets | Green/Red | Ground targets (green=reached, red=error) |
| Hand targets | Cyan/Red | Reach targets (cyan=reached, red=out of reach) |
| Look-at | Magenta | Head target direction |
| Center of mass | White cross | Balance point over feet |
| Ground plane | Gray grid | Reference plane for foot grounding |

### Files

| File | Purpose |
|------|---------|
| `animations/full_body_ik.py` | (NEW) Full-body IK solver |
| `engine/animations/full_body.py` | (NEW) Worker-side full-body solver |
| `developer/rig_visualizer.py` | Update for full-body visualization |
| `animations/test_panel.py` | Full-body IK test operators |

---

## Session Log: 2025-12-27 (Evening)

### Where We Stopped

Full-body IK system is now functional for **legs, arms, and hips**. Ready to add **spine IK chain** for extended reaching.

### What We Have Working

| Component | Status | Notes |
|-----------|--------|-------|
| **Leg IK** | ✅ Working | Direction-based rotation, knees bend FORWARD correctly |
| **Arm IK** | ✅ Working | Direction-based rotation, elbows bend BACKWARD correctly |
| **Cross-body Arms** | ✅ Working | Elbow circle geometry prevents arms clipping through body |
| **Hips Drop** | ✅ Working | World-to-local coordinate transform, drops in correct axis |
| **Region System** | ✅ Working | LEGS, LOWER_BODY, UPPER_BODY, FULL_BODY all unified |
| **Diagnostics** | ✅ Working | Reports FAILED when targets missed by >5cm |
| **Spine IK** | 🔜 Next | Graduated lean for extended reaching |

### What We Fixed This Session

1. **Knee Bend Direction** (critical fix)
   - **Problem**: Knees bent BACKWARD instead of FORWARD
   - **Root Cause**: Cross product order `cross(pole_vec, reach_dir)` gave wrong rotation axis
   - **Fix**: Changed to `cross(reach_dir, pole_vec)` in `solve_leg_ik()`
   - **File**: `engine/animations/ik.py:420`

2. **Leg IK Rotation Method**
   - **Problem**: Legs used raw quaternions from solver, didn't account for bone rest orientation
   - **Fix**: Changed to direction-based rotation like arms:
     - `solve_leg_ik()` returns `(thigh_dir, shin_dir, knee_pos)` - directions, not quaternions
     - Caller uses `_compute_bone_rotation_to_direction()` for thigh
     - Caller uses `_compute_child_rotation_for_ik()` for shin (accounts for parent rotation)
   - **File**: `full_body_ik.py:1809-1838, 1870-1897`

3. **Region Handling for "Legs Only"**
   - **Problem**: LEGS region was moving hips when it shouldn't
   - **Fix**: Removed 'LEGS' from `use_hips` check
   - **File**: `test_panel.py:828`

4. **Diagnostic Honesty**
   - **Problem**: Logs said "SUCCESS" when feet were 16-23cm off target
   - **Fix**: Added error checking - any target >5cm off now reports "FAILED" with `<<<MISSED TARGET>>>`
   - **File**: `full_body_ik.py:758-807`

### Key Insight: Direction-Based IK

The working pattern for limb IK:

```python
# Solver returns DIRECTIONS (world-space vectors)
upper_dir, lower_dir, joint_pos = solve_limb_ik(root, target, pole)

# Convert to bone rotations using bone's actual rest orientation
upper_quat = _compute_bone_rotation_to_direction(upper_bone, upper_dir, armature_matrix)
lower_quat = _compute_child_rotation_for_ik(lower_bone, lower_dir, upper_quat, armature_matrix)
```

This works because:
- Solver computes WHERE joints should be (pure geometry)
- Rotation conversion accounts for each bone's rest orientation
- Child bones account for parent's rotation (inherited transform)

### Next Steps

1. **Spine IK Chain** - Graduated lean toward reach targets
   - Distribute rotation across Spine → Spine1 → Spine2
   - Kick in when arm target is beyond comfortable reach
   - Respect joint limits (max ~45° forward, ~30° side)

2. **Train Claude Better on Movement**
   - Document the direction-based rotation pattern more clearly
   - Add examples of common movement scenarios
   - Create reference for pole vector conventions

3. **Real-Time IK Transitions**
   - Blend between IK states smoothly
   - Handle target changes mid-motion
   - Integrate with animation state machine

### Files Modified This Session

| File | Changes |
|------|---------|
| `engine/animations/ik.py` | Fixed knee cross product order |
| `animations/full_body_ik.py` | Direction-based leg rotation, honest diagnostics |
| `animations/test_panel.py` | Fixed region handling |

---

**Last Updated**: 2025-12-29

---

## Session Log: 2025-12-29 (IK Test-Driven Development)

### The Big Picture: Why This Matters

**Ultimate Goal**: A full-body smart IK system that can be used during gameplay for:
- **Grabbing objects** - Hand reaches to object, body compensates naturally
- **Looking at targets** - Head/neck track points of interest
- **Reaching for ledges** - Arms extend, spine leans, feet stay grounded
- **Interacting with world** - Door handles, levers, buttons at any height/angle
- **Dynamic reactions** - Flinching, dodging, pointing, gesturing

This is NOT just for testing - this is the foundation for the **Interaction/Reaction system**. When a player triggers "grab object at position X", the IK system must:
1. Solve the pose in real-time
2. Produce anatomically correct movement
3. Work from ANY starting pose
4. Blend smoothly with existing animations

**If the IK can't solve poses correctly in isolation, it will NEVER work in gameplay.**

### Current Development Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TEST-DRIVEN IK DEVELOPMENT                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. DEFINE TEST POSES                                                      │
│      └── test_suite.py generates IKTestCase objects                         │
│          ├── Hand targets (L/R at various heights)                          │
│          ├── Foot targets (steps, kicks, balance)                           │
│          ├── Look-at targets (head tracking)                                │
│          ├── Lean targets (spine bending)                                   │
│          └── Hip drop (crouch depth)                                        │
│                                                                             │
│   2. RUN TESTS IN BLENDER                                                   │
│      └── test_panel.py provides UI                                          │
│          ├── "Run All Tests" - executes full suite                          │
│          ├── GPU visualization of targets (color-coded)                     │
│          ├── Real-time pose application to armature                         │
│          └── Human judge marks pass/fail + notes                            │
│                                                                             │
│   3. AUTO-EVALUATE RESULTS                                                  │
│      └── Test suite measures:                                               │
│          ├── Target error (cm from goal)                                    │
│          ├── Joint limit violations                                         │
│          ├── Pole vector problems (elbow/knee direction)                    │
│          └── Elbow/knee angles                                              │
│                                                                             │
│   4. EXPORT SESSION DATA                                                    │
│      └── rig_test_data/output_data/                                         │
│          ├── session_YYYYMMDD_*.json - full test details                    │
│          └── session_YYYYMMDD_*_summary.csv - quick overview                │
│                                                                             │
│   5. ANALYZE FAILURES                                                       │
│      └── Read CSV, identify patterns:                                       │
│          ├── Which poses fail consistently?                                 │
│          ├── What problems repeat? (POLE_UP, POLE_WRONG_SIDE, etc.)         │
│          └── What's the root cause in the solver?                           │
│                                                                             │
│   6. FIX IK SOLVER                                                          │
│      └── engine/animations/ik_solver.py                                     │
│          ├── Adjust elbow/knee scoring logic                                │
│          ├── Add anatomical constraints                                     │
│          ├── Fix pole vector calculations                                   │
│          └── Improve edge case handling                                     │
│                                                                             │
│   7. RETEST → REPEAT                                                        │
│      └── Run tests again, verify fixes, find next issue                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What We've Learned (Hard-Won Lessons)

#### Problem 1: Elbow Points UP When Reaching Above Shoulder

**Symptom**: When hand target is above shoulder height (1.5m+), elbow rotates to point straight up instead of backward.

**Root Cause**: Original code used angle-based detection (`reach_up_component > 0.3`) which missed shallow upward angles. A target at 1.60m with shoulder at 1.50m gave ~0.29 - just under threshold.

**Fix**: Changed to HEIGHT-BASED detection:
```python
# OLD (broken)
reach_up_component = np.dot(reach_dir, char_up)
is_reaching_up = reach_up_component > 0.3  # WRONG - misses 1.60m targets

# NEW (working)
shoulder_z = float(np.dot(shoulder_pos, char_up))
target_z = float(np.dot(hand_target, char_up))
height_above_shoulder = target_z - shoulder_z
is_reaching_up = height_above_shoulder > 0.05  # ANY target 5cm+ above shoulder
```

**Additional Rule**: Elbow can NEVER be above hand target. Added massive penalty:
```python
elbow_z = float(np.dot(test_elbow, char_up))
if elbow_z > target_z + 0.05:  # Elbow more than 5cm above hand
    elbow_above_hand_penalty = (elbow_z - target_z) * 20.0  # MASSIVE penalty
```

#### Problem 2: Dead Zone Behind Body

**Symptom**: Hands behind head or back produce broken poses. Arms seem unable to reach backward.

**Root Cause**: Elbow circle sampling didn't handle behind-body reaches well. The "best" elbow position was often in front when it should be behind.

**Fix**: Special handling for behind-body reaches:
```python
if target_y < -0.1:  # Target behind shoulder
    is_reaching_behind = True
    # Bias elbow selection toward backward positions
```

#### Problem 3: Arms Cross Through Spine

**Symptom**: When reaching to opposite side of body, arm goes straight through spine instead of around it.

**Root Cause**: No spine crossover limit - solver didn't know arms can't pass through the torso.

**Fix**: Height-based crossover limits:
```python
max_crossover_chest = 0.35  # Max 35cm past center at chest height
max_crossover_belly = 0.08  # Max 8cm past center at belly height

# Interpolate based on target height
crossover_limit = lerp(max_crossover_belly, max_crossover_chest, height_factor)

# Clamp X position
if is_left_arm and target_x > crossover_limit:
    target_x = crossover_limit  # Can't reach further right
```

### Current IK Solver Architecture

**File**: `engine/animations/ik_solver.py`

```python
def solve_full_body_ik(snapshot, targets, params):
    """
    Main entry point for full-body IK.

    Solve order (critical for dependencies):
    1. Hips position (crouch/lean affects everything)
    2. Leg IK (feet must be grounded before upper body)
    3. Spine chain (lean toward reach targets)
    4. Arm IK (solve after spine since shoulder moves)
    5. Head/Neck (look-at after body is positioned)
    """

def solve_arm_ik(shoulder_pos, hand_target, upper_len, lower_len, ...):
    """
    Two-bone arm IK with elbow circle sampling.

    Key innovations:
    - Height-based "reaching up" detection
    - Elbow can never be above hand (anatomical rule)
    - Behind-body reach special handling
    - Spine crossover prevention
    - Elbow circle samples 24 positions, scores each
    """

def solve_leg_ik(hip_pos, foot_target, upper_len, lower_len, ...):
    """
    Two-bone leg IK with knee hinge constraint.

    Key rules:
    - Knee ONLY bends forward (hinge joint)
    - Pole vector always forward (+Y in character space)
    - Knee angle clamped to anatomical limits
    """
```

### Test Suite Structure

**File**: `animations/rig_test_data/test_suite.py`

```
TEST BINS (organized by difficulty)
├── _reach_bins()         - Hand targets at heights (fewer easy, more hard)
├── _foot_bins()          - Foot targets (steps + high kicks)
├── _crouch_bins()        - Hip drop tests (medium + deep)
├── _look_bins()          - Head tracking (baseline + extreme)
├── _lean_bins()          - Spine bending (4 directions)
├── _combined_bins()      - Multi-target poses
├── _athletic_bins()      - Kicks, punches, throws
├── _extreme_reach_bins() - Max extension tests
├── _balance_bins()       - One-leg, yoga poses
├── _interaction_bins()   - Doors, climbing, pushing
├── _asymmetric_bins()    - Different arm positions (reduced)
├── _spine_stress_bins()  - Extreme torso bends
├── _neck_stress_bins()   - Extreme head positions
├── _shoulder_stress_bins() - Behind head/back reaches
├── _extreme_arm_bins()   - Maximum arm reaches
└── _extreme_leg_bins()   - High kicks, splits, scorpions
```

**Recent Changes** (2025-12-29):
- Reduced easy baseline tests (single forward reach per height below shoulder)
- Expanded above-shoulder tests (full 3-direction coverage)
- Added 5 new leg-behind-body tests (SCORPION, HIGH BACK KICK, STANDING BOW, SIDE BACK KICK, EXTREME ARABESQUE)
- Trimmed asymmetric tests to focus on challenging poses

### GPU Visualization

**File**: `animations/test_panel.py`

The test panel draws IK targets in 3D so you can see exactly what the solver is trying to reach:

| Target Type | Color | Description |
|-------------|-------|-------------|
| L_HAND | Blue (0.2, 0.6, 1.0) | Left hand target |
| R_HAND | Orange (1.0, 0.4, 0.2) | Right hand target |
| L_FOOT | Green (0.2, 1.0, 0.4) | Left foot target |
| R_FOOT | Yellow (1.0, 1.0, 0.2) | Right foot target |
| LOOK_AT | Magenta (1.0, 0.2, 1.0) | Head/look target |
| LEAN | Purple (0.6, 0.2, 1.0) | Spine lean target |

Targets are drawn as spheres at world positions, root-relative to the armature.

### Key Files Reference

| File | Purpose |
|------|---------|
| `animations/rig.md` | Complete rig documentation (bone hierarchy, joint limits, axis orientations) |
| `animations/test_panel.py` | Test UI, GPU visualization, test execution |
| `animations/rig_test_data/test_suite.py` | Test case definitions and auto-evaluation |
| `engine/animations/ik_solver.py` | Core IK solver (runs in workers) |
| `engine/animations/default_limits.py` | Joint rotation limits |
| `animations/full_body_ik.py` | Full-body IK coordination |

### Development Phases

```
PHASE 1: Static Pose Solving ◄── WE ARE HERE
├── [x] Two-bone arm IK
├── [x] Two-bone leg IK
├── [x] Elbow/knee direction control
├── [x] Spine crossover prevention
├── [x] Behind-body reach handling
├── [~] Above-shoulder elbow-up fix (testing)
├── [ ] Spine lean chain
└── [ ] Head/neck look-at

PHASE 2: Pose-to-Pose Transitions
├── [ ] Blend between solved poses
├── [ ] Smooth target interpolation
├── [ ] Handle target changes mid-motion
└── [ ] Validate intermediate frames

PHASE 3: Real-Time Gameplay Integration
├── [ ] Connect to Interaction/Reaction system
├── [ ] Engine worker job type (IK_SOLVE_REALTIME)
├── [ ] Blend with animation state machine
└── [ ] Multi-character support

PHASE 4: Advanced Features
├── [ ] Finger IK for gripping
├── [ ] Physics-based settling
├── [ ] Obstacle avoidance
└── [ ] Procedural reactions (flinch, dodge)
```

### Next Steps

1. **Run updated test suite** - Verify elbow-up fix works with new height-based detection
2. **Add spine lean chain** - When arm can't reach target, lean torso to extend range
3. **Look-at system** - Head/neck tracking for look targets
4. **Pose-to-pose blending** - Smooth transitions between solved poses

### Critical Rule: Test Everything

**The IK system MUST work perfectly in the test suite before being used in gameplay.**

If a pose fails in testing:
1. It will fail in gameplay
2. Players will see broken animations
3. The interaction/reaction system will produce garbage

The test suite is not a "nice to have" - it's the quality gate. Every fix must be validated by running the full suite and checking the failure rate.

```
TARGET: 0 failures on core poses
CURRENT: ~7/29 failures (24% failure rate)
GOAL: <5% failure rate before gameplay integration
```


---

## Session Log: 2025-12-29 (CRITICAL BUG FIX - Stale Parent Transforms)

### THE ROOT CAUSE FOUND

**Problem**: Every change to IK rotation code seemed impossible to get right. Changes that should work didn't. The verification said "OK" but post-apply showed 50-130° errors.

**Root Cause**: **STALE PARENT TRANSFORMS**

When computing bone rotations in a chain (thigh → shin), the code queried Blender for parent state:

```python
# Line 1456 in _compute_bone_rotation_to_direction:
parent_posed_world = arm_matrix @ bone.parent.matrix  # ← STALE!
```

**But Blender hadn't been updated yet!** The sequence was:

```
1. Compute thigh_quat (stores in result.bone_transforms, NOT applied to Blender)
2. Compute shin_quat using bone.parent.matrix (which is STILL rest pose!)
3. Later... apply all transforms to Blender
4. view_layer.update()
5. Post-apply verification shows HUGE errors
```

### Why Verification Lied

The inline verification at line 1500-1512 did this:

```python
# Uses bone.parent.matrix - which is STALE (hasn't been updated)
parent_posed_world = arm_matrix @ bone.parent.matrix  # REST POSE!
parent_posed_quat = parent_posed_world.to_quaternion()
verify_y_world = parent_posed_quat @ verify_y  # Wrong parent = wrong result
```

**It was verifying against the OLD parent pose, not the NEW one we just computed!**

### The Evidence

From test session 2025-12-29 05:31:
```
conv_l_arm_verify_err = 0.000  ← "Verification passed!"
postapply_l_arm_error = 93.4°  ← Reality: 93° off!
```

The 0.000° verification used stale parent data. The 93.4° is reality after Blender computed the full chain.

### THE FIX: TransformChainTracker

Created `animations/transform_chain.py` - a new module that tracks our own transform chain WITHOUT querying Blender's stale data.

**Key Method**: `compute_rotation_to_direction(bone_name, target_dir)`

```python
class TransformChainTracker:
    """
    Tracks bone transforms without relying on Blender's stale data.

    Caches rest pose data once, then tracks pending rotations as we compute them.
    When computing a child's rotation, uses the tracked parent rotation.
    """

    def get_bone_world_quat(self, bone_name: str) -> Quaternion:
        """
        Get bone's world orientation, accounting for all pending parent rotations.

        This is THE KEY METHOD. It computes what the bone's world orientation
        WILL BE once all pending rotations are applied.

        DOES NOT query Blender's bone.matrix (which is stale).
        """
        # Build chain from root to this bone
        chain = []  # [root, ..., bone]

        # Start with armature world orientation
        world_quat = self.arm_quat.copy()

        # Walk down the chain
        for name in chain:
            # Apply rest pose relative to parent
            world_quat = world_quat @ rest_quat

            # Apply our PENDING rotation if we have one
            if name in self._pending_local_quats:
                world_quat = world_quat @ self._pending_local_quats[name]

        return world_quat
```

### How It's Used

```python
# OLD (BROKEN) - queries stale Blender data
thigh_quat = self._compute_bone_rotation_to_direction(thigh, thigh_dir, arm_matrix)
shin_quat = self._compute_child_rotation_for_ik(shin, shin_dir, thigh_quat, arm_matrix)

# NEW (FIXED) - tracks our own transforms
chain = TransformChainTracker(self.armature)
thigh_quat = chain.compute_rotation_to_direction("LeftThigh", Vector(thigh_dir))
shin_quat = chain.compute_rotation_to_direction("LeftShin", Vector(shin_dir))
```

The chain tracker:
1. Computes thigh rotation and STORES it
2. When computing shin, uses the STORED thigh rotation (not Blender's stale data)
3. Verification happens against our computed chain, not Blender's stale state

### Files Modified

| File | Changes |
|------|---------|
| `animations/transform_chain.py` | NEW - TransformChainTracker class |
| `animations/full_body_ik.py` | Import TransformChainTracker, use for all limb IK |
| `animations/full_body_ik.py` | Updated `_verify_post_apply_directions()` for new diagnostic format |

### Why This Was Impossible to Debug Before

1. **Every change you made was doomed** - You could have the perfect quaternion formula, but it's computing against the WRONG parent orientation.

2. **The verification gaslighted you** - It said "0.0° error ✓OK" so you thought the math was correct.

3. **Arms were worse than legs** - Longer chain = more accumulated staleness:
   - Leg: Hips → Thigh → Shin (2 bones, some error)
   - Arm: Spine2 → Shoulder → Arm → ForeArm (4 bones, massive error)

### Expected Results

After this fix:
- Chain verify error and post-apply error should MATCH (both < 1° if math is correct)
- If they don't match, there's a different bug (not staleness)
- Arm and leg errors should be similar magnitude (no more "arms are mysteriously worse")

### Key Insight

The fundamental problem was never the rotation math. It was that the rotation math was being applied to the WRONG input data. This is an **architectural bug**, not a **math bug**.

**RERUN THE TEST SUITE** to verify the fix works.

---

**Last Updated**: 2025-12-29