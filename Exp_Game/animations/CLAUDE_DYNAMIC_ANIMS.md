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

**Last Updated**: 2025-12-27 ~10:30 PM EST
