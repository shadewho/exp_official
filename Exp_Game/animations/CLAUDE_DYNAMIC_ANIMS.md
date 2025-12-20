# Dynamic Animation System

**Status**: Foundation phase - IK solver built, testing tools in progress

---

## The Vision

The current locomotion system (walk, run, jump, idle) is **effective but rigid**. Animations play exactly as authored regardless of the environment. Character feet float over slopes, hands clip through walls, there's no reaction to impacts.

The Dynamic Animation System is a **reactive layer on top of base locomotion** that adapts the character to the world around them. It bridges the gap from rigid to fluid.

**Core Principle**: The world informs the animation, not the other way around.

---

## What This System Does

### Receives Input From:
- **Physics system** - ground height, slope angle, wall collisions, platform motion
- **Interaction system** - objects to grab, buttons to press, ledges to climb
- **Reaction system** - impacts, trips, stumbles, environmental hazards
- **Game state** - carrying objects, injury state, fatigue

### Outputs:
- **Bone corrections** via IK (feet plant on actual ground, hands reach actual targets)
- **Additive animations** (stumble when hit, brace when falling)
- **Blend weight adjustments** (override part or all of base locomotion)
- **Procedural motion** (breathing, weight shifting, look-at)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FINAL POSE                              │
│              (What gets applied to the armature)                │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ Blend
┌─────────────────────────────────────────────────────────────────┐
│                    DYNAMIC LAYER (Optional)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Foot IK   │  │   Arm IK    │  │  Look-At    │             │
│  │ (grounding) │  │ (reaching)  │  │  (head)     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Reactions  │  │  Additive   │  │ Procedural  │             │
│  │ (stumble)   │  │  (breathe)  │  │  (balance)  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ Base pose
┌─────────────────────────────────────────────────────────────────┐
│                    BASE LOCOMOTION LAYER                        │
│         Walk, Run, Jump, Idle, Crouch (existing system)         │
│                   Plays regardless of world                     │
└─────────────────────────────────────────────────────────────────┘
```

**Key**: Dynamic layer is **100% optional**. Users can enable/disable individual features. Base locomotion always works.

---

## The Standard Rig

All dynamic features target one armature: **the Exploratory standard rig**.

See: `rig.md` (same directory)

- 53 bones, Mixamo-style naming
- Fixed bone lengths and hierarchy
- Pre-defined IK chains (legs, arms, spine, look-at)
- This rig never changes - all characters conform to it

**Why one rig?**
- IK chain lengths are known constants
- Animation data is transferable between characters
- Procedural systems don't need per-character configuration
- Users just weight paint their mesh, everything else works

---

## IK System (Foundation)

IK is the core mechanism for world-reactive animation.

### What We Have:
- **Two-bone analytical solver** (`engine/animations/ik.py`)
  - Law of cosines for joint angles
  - Works for legs (thigh→shin→foot) and arms (upper→fore→hand)
  - Returns world positions for joints (knee, elbow)

- **Bone rotation conversion** (`animations/test_panel.py`)
  - `point_bone_at_target()` - rotates bone to face a world position
  - Handles parent chains and bone roll correctly
  - Works for both legs and arms

- **Test panel** (Developer Tools > IK Test)
  - Chain selector (left/right leg/arm)
  - Target sliders (foot height, arm reach)
  - Apply/Reset buttons

### What We Need:
- **Full XYZ target control** - not just height/forward
- **3D cursor as target** - place cursor, arm reaches for it
- **Pole vector control** - knee/elbow bend direction
- **Reach limit visualization** - show max extent spheres
- **Live update mode** - IK updates as you drag sliders
- **Edge case testing** - unreachable targets, hyperextension

### Integration Points:
- Physics ground raycast → foot IK target Z
- Interaction object position → hand IK target XYZ
- Character velocity → procedural lean/balance

---

## Testing Philosophy

**Test thoroughly before gameplay integration.**

The N panel test suite must verify:

| Test | What It Validates |
|------|-------------------|
| Leg reaches target Z | Basic IK math works |
| Knee bends forward | Pole vector correct |
| Arm reaches XYZ | 3D targeting works |
| Elbow bends backward | Arm pole vector correct |
| Target at max reach | Extension behavior |
| Target beyond reach | Clamping/failure handling |
| Target too close | Hyperflexion behavior |
| Both legs simultaneously | Multi-chain IK |
| IK + animation blend | Layering works |wa

**Visualizer integration:**
- IK targets (spheres)
- Bone chains (lines)
- Reach limits (transparent spheres)
- Pole directions (arrows)

This can live alongside or within the KCC Visual Debug system.

---

## User Experience Goal

**Make dynamic animation hands-free for world builders.**

Current pain point: Users want characters that react to the world, but creating those animations requires:
- Animation software knowledge
- Understanding of blend trees
- Manual setup per interaction

**Target experience:**
1. User places a weapon in the world
2. User marks it as "grabbable"
3. Character walks up, arm automatically reaches and grabs
4. No animation authoring required

The system uses:
- Pre-made reaction animations (stumble, grab, brace)
- IK to adapt those animations to actual world positions
- Physics data to trigger appropriate reactions

---

## What We Might Include

### Animation Library (bundled or downloadable):
- Stumble/trip variations
- Grab/reach poses
- Impact reactions (front, back, sides)
- Throw/drop motions
- Climb/vault transitions
- Fall/land sequences

These serve as **templates** that IK and blending adapt to specific situations.

### Procedural Layers:
- Breathing (chest expansion cycle)
- Weight shifting (idle micro-movements)
- Look-at (head tracks points of interest)
- Balance adjustment (lean into velocity)

### Physics Integration:
- Foot grounding (IK feet to actual ground)
- Wall bracing (hand reaches toward nearby wall)
- Platform riding (body adjusts to moving surface)
- Impact reactions (stumble direction based on collision normal)

---

## Performance Requirements

**Must not impact frame rate.**

- All IK computation runs in **worker threads** (not main thread)
- Bone transforms are **numpy arrays** (vectorized math)
- Raycasts for foot grounding reuse **existing physics raycasts** where possible
- Features are **individually toggleable** (disable what you don't need)
- Logging uses **fast buffer system** (no console prints during gameplay)

Log output: `C:\Users\spenc\Desktop\engine_output_files\diagnostics_latest.txt`

---

## Development Phases

### Phase 1: IK Foundation (Current)
- [x] Two-bone IK solver
- [x] Bone rotation conversion
- [x] Basic test panel
- [ ] Expanded test panel (XYZ, 3D cursor, pole control)
- [ ] IK visualizer (targets, chains, limits)
- [ ] Edge case handling

### Phase 2: Foot Grounding
- [ ] Ground raycast per foot (from animation foot position)
- [ ] IK target from raycast hit
- [ ] Pelvis height adjustment
- [ ] Blend with walk/run animation
- [ ] Slope adaptation

### Phase 3: Arm Reaching
- [ ] Interaction system integration
- [ ] Hand IK to object position
- [ ] Grab pose blending
- [ ] Two-handed grabs

### Phase 4: Reactions
- [ ] Impact detection from physics
- [ ] Reaction animation selection
- [ ] Procedural stumble direction
- [ ] Recovery blending

### Phase 5: Procedural Polish
- [ ] Look-at system
- [ ] Breathing/idle variation
- [ ] Balance/lean from velocity
- [ ] Ragdoll transition

---

## Key Files

| File | Purpose |
|------|---------|
| `rig.md` | Standard rig specification (bone names, lengths, chains) |
| `engine/animations/ik.py` | Two-bone IK solver (worker-safe, numpy) |
| `animations/test_panel.py` | IK test operators and properties |
| `developer/dev_panel.py` | IK Test UI section |
| `developer/dev_properties.py` | Debug properties (add IK visualizer here) |

---

## Remember

1. **Optional** - Every dynamic feature can be disabled
2. **Layered** - Dynamic sits on top of base locomotion, doesn't replace it
3. **Reactive** - World informs animation, not the reverse
4. **Tested** - Validate in N panel before gameplay integration
5. **Fast** - Worker threads, numpy, no main thread physics
6. **Simple for users** - Complexity hidden, results automatic

---

**Last Updated**: 2025-12-20
