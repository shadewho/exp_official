# Dynamic Animation System

**Status**: Rigid foundation built - needs procedural layer and node integration

**Backend Reference**: See `Exp_Game/engine/animations/CLAUDE_ANIM.md` for low-level animation architecture (numpy arrays, worker threading, blend math).

---

## Current State

We have a **functional but rigid** animation system:

### What Works Now
- **Locomotion state machine** - walk, run, jump, idle, fall, land
- **BlendSystem overlays** - additive and override layers on top of locomotion
- **Body group masking** - target specific bones (UPPER_BODY, LEG_L, ARMS, etc.)
- **Character Action node** - play animations via reactions with blend time, speed, force lock
- **Vectorized blending** - numpy operations, ~10-20us per layer (fast)

### What's Missing
- **No procedural motion** - everything requires pre-made animation data
- **No variance** - same animation plays identically every time
- **Weak node integration** - reactions trigger animations but no smart selection
- **No built-in library** - users must create all animations from scratch
- **No world reactivity** - character doesn't respond to physics/environment

---

## The Problem

**Current workflow is too manual:**

1. User wants "flinch when hit"
2. User must create flinch animation in Blender
3. User must bake animation
4. User must set up Collision Trigger + Character Action reaction
5. Animation plays identically regardless of hit direction, force, etc.

**Target workflow:**

1. User wants "flinch when hit"
2. User adds Collision Trigger + "Dynamic Flinch" reaction
3. System selects appropriate flinch variant based on impact direction
4. Procedural adjustments add variance and world-responsiveness

---

## Architecture Goal

```
┌─────────────────────────────────────────────────────────────────┐
│                         FINAL POSE                              │
│              (Applied to armature each frame)                   │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    PROCEDURAL LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Variance   │  │   Noise     │  │  Physics    │             │
│  │ (per-play)  │  │  (micro)    │  │  Response   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│                    REACTION LAYER (BlendSystem)                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Override   │  │  Additive   │  │    IK       │             │
│  │ (flinch)    │  │  (breathe)  │  │  (reach)    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│         Triggered by: Nodes (Interactions/Reactions)            │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│                    BASE LOCOMOTION                              │
│         Walk, Run, Jump, Idle (AnimationController)             │
│                   Worker-computed, always running               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Node System Integration

The animation system connects to gameplay through **Exp_Nodes**:

### Current Connection
```
Trigger Node (Collision, Proximity, Action, etc.)
       │
       ▼
Reaction Node (Character Action)
       │
       ▼
BlendSystem.play_override() / play_additive()
       │
       ▼
Animation plays on masked bones
```

### Problem: Too Manual
- User must specify exact animation
- No automatic selection based on context
- No variance between plays
- No smart bone group selection

### Target Connection
```
Trigger Node (Collision, Proximity, Action, etc.)
       │
       ├─── Context data (direction, force, object type)
       ▼
Dynamic Reaction Node (Smart selection)
       │
       ├─── Selects from animation pool based on context
       ├─── Applies variance (speed, intensity, timing)
       ├─── Chooses appropriate bone group
       ▼
BlendSystem with procedural modifiers
       │
       ▼
Unique, context-appropriate animation
```

---

## Built-In Animation Library

**Concept**: Ship a library of pre-made animations for the standard rig that users can use out of the box.

### Location
```
Exp_Game/
  animations/
    library/                    # Built-in animation data
      reactions/
        flinch_front.blend
        flinch_back.blend
        flinch_left.blend
        flinch_right.blend
        stumble_forward.blend
        stumble_backward.blend
      interactions/
        grab_high.blend
        grab_low.blend
        push.blend
        pull.blend
      locomotion/
        swim_idle.blend
        swim_forward.blend
        climb_up.blend
        climb_down.blend
      emotes/
        wave.blend
        point.blend
        shrug.blend
```

### How It Works
1. Animations are baked at game start (like user animations)
2. Dynamic nodes reference library animations by name
3. Users can override with custom animations
4. Library grows over time with common actions

### Benefits
- Users get working reactions immediately
- No animation software required for basic games
- Consistent quality baseline
- Can be extended/customized

---

## Variance System

**Goal**: Same trigger produces different results each time.

### Types of Variance

**1. Animation Selection Variance**
```python
# Instead of playing one animation:
play_animation("flinch")

# Select from pool:
pool = ["flinch_01", "flinch_02", "flinch_03"]
play_animation(random.choice(pool))
```

**2. Playback Variance**
```python
# Speed variation: 0.85x to 1.15x
speed = base_speed * random.uniform(0.85, 1.15)

# Intensity variation (blend weight)
weight = base_weight * random.uniform(0.8, 1.0)

# Timing offset
delay = random.uniform(0.0, 0.05)  # 0-50ms random delay
```

**3. Bone Group Variance**
```python
# Sometimes full upper body, sometimes just arms
groups = ["UPPER_BODY", "ARMS", "ARM_L", "ARM_R"]
weights = [0.5, 0.3, 0.1, 0.1]  # Probability distribution
selected = random.choices(groups, weights)[0]
```

**4. Procedural Noise**
- Micro-movements on idle (weight shifting)
- Breathing cycle (chest additive)
- Head micro-sway
- Hand tremor (fatigue/injury state)

### Node Exposure
```
[Character Action (Dynamic)]
├── Animation Pool: [flinch_01, flinch_02, flinch_03]
├── Speed Variance: 0.15 (±15%)
├── Weight Variance: 0.2 (±20%)
├── Delay Variance: 0.05s
└── Bone Group: Auto / Specific
```

---

## Context-Aware Selection

**Goal**: System chooses appropriate animation based on game state.

### Impact Direction
```python
# Collision provides impact normal
impact_direction = collision.normal

# Select flinch based on direction relative to character
if facing_dot(impact_direction) > 0.5:
    animation = "flinch_front"
elif facing_dot(impact_direction) < -0.5:
    animation = "flinch_back"
elif right_dot(impact_direction) > 0:
    animation = "flinch_right"
else:
    animation = "flinch_left"
```

### Object Height (for grabs/interactions)
```python
# Interaction target height relative to character
relative_height = target.z - character.z

if relative_height > 1.5:
    animation = "grab_high"
elif relative_height > 0.5:
    animation = "grab_mid"
else:
    animation = "grab_low"
```

### Character State
```python
# Modify based on current state
if character.is_injured:
    speed *= 0.7
    weight *= 1.2  # More dramatic

if character.is_exhausted:
    add_procedural_sway()
```

---

## What Needs Building

### Phase 1: Animation Library Infrastructure
- [ ] Create library folder structure
- [ ] Build library loader (bakes library anims at startup)
- [ ] Library animation browser UI
- [ ] Fallback system (use library if user anim missing)

### Phase 2: Variance System
- [ ] Animation pool data structure
- [ ] Playback variance parameters on layer
- [ ] Random selection with weighting
- [ ] Seed control for reproducibility

### Phase 3: Dynamic Reaction Nodes
- [ ] "Dynamic Character Action" node type
- [ ] Context inputs (direction, force, height)
- [ ] Auto bone group selection
- [ ] Pool-based animation selection UI

### Phase 4: Procedural Modifiers
- [ ] Noise generator (Perlin/simplex)
- [ ] Breathing additive layer
- [ ] Micro-movement system
- [ ] State-based modification (injury, fatigue)

### Phase 5: Context Integration
- [ ] Collision normal → flinch direction
- [ ] Interaction height → reach animation
- [ ] Velocity → lean/balance
- [ ] Physics impact → stumble intensity

---

## Key Files

| File | Purpose |
|------|---------|
| `blend_system.py` | Layer management, vectorized blending |
| `controller.py` | Base locomotion, worker integration |
| `bone_groups.py` | Body part masks (UPPER_BODY, LEGS, etc.) |
| `Exp_Nodes/reaction_nodes.py` | Character Action node UI |
| `reactions/exp_reactions.py` | Reaction executors |
| `engine/animations/blend.py` | Vectorized blend math |
| `engine/animations/CLAUDE_ANIM.md` | Backend architecture docs |

---

## Design Principles

1. **Rigid by default, fluid when enabled** - Base system always works, procedural is opt-in
2. **Node-driven** - All dynamic behavior configured through visual nodes
3. **Library-first** - Ship animations users can use immediately
4. **Variance everywhere** - Nothing should look identical twice
5. **Context-aware** - System uses available data to make smart choices
6. **Performance-safe** - All heavy math in workers, main thread stays light

---

## Success Criteria

**A user should be able to:**

1. Add a "flinch on hit" reaction in under 30 seconds
2. See different flinch variations each time character is hit
3. Have flinch direction match impact direction automatically
4. Never need to open animation software for basic reactions
5. Optionally override with custom animations when desired

**The system should feel:**

- Alive (subtle constant motion)
- Responsive (reactions match context)
- Varied (no two plays identical)
- Effortless (minimal setup required)

---

**Last Updated**: 2025-12-21
