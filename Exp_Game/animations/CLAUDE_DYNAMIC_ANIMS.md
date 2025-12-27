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

**Last Updated**: 2025-12-26
