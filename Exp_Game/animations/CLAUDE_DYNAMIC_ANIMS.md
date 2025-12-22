# Dynamic Animation & IK System

**Status**: IK system in progress - leg IK working, arm IK needs fixes

**Goal**: Procedurally grab objects, squat, trip, react to physics using IK + dynamic mesh system

**CRITICAL RULE**: Main thread stays FREE. All heavy math runs in engine workers.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MAIN THREAD (LIGHT)                        │
│  - Reads results from workers                                   │
│  - Applies bone.rotation_quaternion (fast, pre-computed)        │
│  - NO matrix math, NO IK solving, NO heavy computation          │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ results
┌─────────────────────────────────────────────────────────────────┐
│                    ENGINE WORKERS (HEAVY)                        │
│  - IK solving (law of cosines, quaternion math)                 │
│  - Pole computation (character-relative directions)             │
│  - Animation blending (numpy vectorized)                        │
│  - All expensive calculations                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Current IK System State

### What Works
- **Two-bone IK solver** (`engine/animations/ik.py`) - numpy, worker-safe
- **Leg IK** - bones follow target correctly after rotation fix
- **Character-relative poles** - knees bend forward relative to pelvis
- **Visualizer** - cyan/magenta lines show computed bone directions
- **BlendSystem integration** - `set_ik_target()` and `set_ik_target_object()`
- **Multi-chain support** - can have IK on multiple limbs simultaneously

### What's Broken
- **Arm IK bone rotation** - solver computes correct positions (visualizer shows it) but `rotation_quaternion` application fails for arms
- **Parent chain transforms** - leg bones have simple parent (pelvis), arm bones have complex parent (shoulder/clavicle) with different rest orientations

### The Core Problem

```python
# This works for legs but NOT for arms:
def _apply_ik_to_bone(bone, target_direction, influence, armature):
    # Transform target to bone-local space
    # Compute rotation from Y-axis to target
    # Apply rotation_quaternion

# WHY IT FAILS FOR ARMS:
# - Arm parent chain: Spine → Chest → Shoulder → UpperArm → ForeArm
# - Each parent has rotation that affects child
# - Our transform math doesn't correctly account for shoulder orientation
```

### What We Need
1. **Understand rig hierarchy** - map out exact parent chain for arms vs legs
2. **Fix bone-local transform** - correct quaternion math for arm parent chain
3. **Better visualization** - show what rotation is being computed vs applied
4. **More logging** - trace through transform chain to find where it breaks

---

## Logging System

**CRITICAL**: Use the Fast Buffer Logger for all IK debugging. See `Exp_Game/developer/CLAUDE_LOGGER.md`

### Rules
1. **NEVER print() in game loop** - destroys performance (~1000μs per print)
2. **USE log_game()** - fast buffer write (~1μs per log)
3. **Frequency gating** - Master Hz controls log rate (1-30 Hz)
4. **Export after test** - logs go to `diagnostics_latest.txt`

### IK Log Category
```python
from Exp_Game.developer.dev_logger import log_game

# Log IK events:
log_game("IK", f"SOLVE chain={chain} target=({x:.2f},{y:.2f},{z:.2f}) dist={dist:.3f}m")
log_game("IK", f"APPLY bone={bone.name} rot=({q.w:.3f},{q.x:.3f},{q.y:.3f},{q.z:.3f})")
```

### Reading Logs
```
C:\Users\spenc\Desktop\engine_output_files\diagnostics_latest.txt
```

Format: `[CATEGORY F#### T##.###s] message`

---

## Visualizer System

The Rig Visualizer (`developer/rig_visualizer.py`) draws debug overlays:

### Current IK Visualization
- **Cyan line**: Upper bone direction (shoulder→elbow or hip→knee)
- **Magenta line**: Lower bone direction (elbow→hand or knee→foot)
- **Green circle**: IK target position
- **Yellow lines**: Pole position indicator

### What Visualizer Shows
The visualizer draws the **computed** bone directions from the IK solver. If lines point correctly but mesh doesn't follow, the problem is in `_apply_ik_to_bone()`.

### Enable Visualizer
Developer Tools panel → IK Visual Debug checkbox

---

## Pose Layer System (PLANNED)

To handle conflicts between animation, IK, ragdoll:

```
┌─────────────────────────────────────────────────────┐
│                   POSE RESOLVER                      │
│                                                      │
│  Layer 0: BASE (locomotion/idle)                    │
│     └── influence: {all: 1.0}                       │
│                                                      │
│  Layer 1: ACTION (jump, attack)                     │
│     └── influence: {upper_body: 1.0}                │
│                                                      │
│  Layer 2: IK (reaching, foot plant)                 │
│     └── influence: {arm_R: 1.0, leg_L: 0.8}        │
│                                                      │
│  Layer 3: RAGDOLL (physics takeover)                │
│     └── influence: {all: 1.0}                       │
└─────────────────────────────────────────────────────┘
```

### Key Concept
- **Bone masks are STRUCTURAL** (defined once per layer)
- **Influence is DYNAMIC** (changes at runtime 0-1)
- **Priority resolves conflicts** (higher layer wins on same bones)

---

## Key Files

| File | Purpose |
|------|---------|
| `engine/animations/ik.py` | Two-bone IK solver, pole computation (WORKER-SAFE) |
| `animations/runtime_ik.py` | Applies IK results to bones (main thread, LIGHT) |
| `animations/blend_system.py` | IK targets, animation layers |
| `developer/rig_visualizer.py` | Debug visualization |
| `developer/dev_logger.py` | Fast buffer logging |
| `developer/CLAUDE_LOGGER.md` | Logging rules |
| `animations/rig.md` | Rig bone definitions, chain lengths |

---

## IK Chain Definitions

From `engine/animations/ik.py`:

```python
LEG_IK = {
    "leg_L": {
        "root": "LeftThigh",      # Parent: Pelvis (simple)
        "mid": "LeftShin",
        "tip": "LeftFoot",
        "len_upper": 0.4947,
        "len_lower": 0.4784,
        "reach": 0.9731,
    },
}

ARM_IK = {
    "arm_L": {
        "root": "LeftArm",        # Parent: LeftShoulder (complex!)
        "mid": "LeftForeArm",
        "tip": "LeftHand",
        "len_upper": 0.2782,
        "len_lower": 0.2863,
        "reach": 0.5645,
    },
}
```

---

## Next Steps

### Immediate (Fix Arm IK)
1. Add more logging in `_apply_ik_to_bone()` to trace transform chain
2. Log bone parent names and rest orientations
3. Compare leg bone transforms vs arm bone transforms
4. Identify where the math diverges

### Short Term
- [ ] Fix arm IK rotation application
- [ ] Test IK with character rotation (validate poles)
- [ ] Add "Reach To" reaction node

### Medium Term
- [ ] Implement pose layer system
- [ ] Add influence masks per bone group
- [ ] Foot planting during locomotion

### Long Term
- [ ] Procedural stumble/trip reactions
- [ ] Dynamic grab with IK + physics
- [ ] Ragdoll blend in/out

---

## Design Rules

1. **MAIN THREAD FREE** - Only read results, apply pre-computed values
2. **WORKERS DO MATH** - All IK solving, blending, transforms in engine
3. **LOG EVERYTHING** - Can't debug what you can't see
4. **VISUALIZE FIRST** - If visualizer is wrong, solver is wrong. If visualizer is right, application is wrong.
5. **HUMAN RIG BEHAVIOR** - Knees always forward, elbows back/down, anatomically correct

---

**Last Updated**: 2025-12-22
