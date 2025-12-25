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

### Phase 1: Rig Calibration (Current)

Analyze the rig to understand its structure:

```
RIG CALIBRATION
├── Bone Orientations     - Which axis points down the bone
├── Bend Axes             - Which axis each joint bends on
├── Roll Values           - Bone roll for proper rotation
└── Chain Definitions     - IK chains (arm_L, arm_R, leg_L, leg_R)
```

Files:
- `engine/animations/rig_calibration.py` - Worker-safe analysis
- `developer/rig_analyzer.py` - UI and data extraction

### Phase 2: Movement Profiles (Next)

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
| `engine/worker/entry.py` | Worker job handlers |
| `engine/animations/ik.py` | IK solvers (worker-safe) |
| `engine/animations/blend.py` | Blend math (worker-safe) |
| `engine/animations/rig_calibration.py` | Rig analysis (worker-safe) |
| `developer/rig_analyzer.py` | Rig analyzer UI |
| `developer/rig_visualizer.py` | Debug visualization |

---

## Development Roadmap

### Now
- [x] Worker-based pose blending (POSE_BLEND_COMPUTE)
- [x] Worker-based IK solving (IK_SOLVE_BATCH)
- [x] Rig calibration system
- [ ] Joint limit definitions
- [ ] Joint limit enforcement during blend

### Next
- [ ] Coupling rules (bones that move together)
- [ ] Spine IK chain
- [ ] Full-body coordination

### Future
- [ ] Valid pose recording (learn from examples)
- [ ] Automatic path finding between poses
- [ ] Node System integration for gameplay
- [ ] Multi-character support

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

**Last Updated**: 2025-12-25
