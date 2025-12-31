# Ragdoll System (Rig Agnostic)

**Status:** Simple pendulum collapse - works on any armature

## Overview

Simple ragdoll that makes any armature collapse like a marionette with cut strings.
- Every bone is a pendulum that falls toward gravity
- No complex physics - just collapse
- Works on ANY armature without hardcoded bone names

---

## How It Works

### The Simple Model

Each bone is treated as a pendulum hanging from its parent:
1. Gravity pulls the bone's tip downward
2. Bone rotates around its joint toward gravity
3. Damping prevents wild swinging
4. Joint limits prevent unnatural poses

That's it. No spring-to-rest fighting the collapse, no parent alignment, no complex torque calculations.

### Two-Part System

**Main Thread (exp_ragdoll.py):**
- Position drop: armature falls to ground with gravity
- Submits bone data to worker
- Applies rotation results to pose bones

**Worker (ragdoll.py):**
- Computes bone rotations based on gravity
- Simple: `torque = gravity_direction * GRAVITY_STRENGTH`
- Returns new rotations each frame

---

## Physics Constants

### Position Drop (Main Thread)
```python
DROP_GRAVITY = -20.0    # Fast drop
DROP_DAMPING = 0.3      # Low bounce
DROP_DURATION = 1.5     # Full collapse time
```

### Bone Rotation (Worker)
```python
GRAVITY_STRENGTH = 4.0  # How hard gravity pulls
DAMPING = 0.85          # Smooth settle
STIFFNESS = 0.02        # Near-zero (no fighting)
MAX_ANG_VEL = 8.0       # Velocity limit

# Joint limits by role (radians)
LIMITS = {
    "core": 0.6,   # ~35° - spine/hips (limited)
    "limb": 1.8,   # ~100° - arms/legs (loose)
    "head": 1.0,   # ~57° - head/neck
    "hand": 2.2,   # ~125° - hands/feet (very loose)
}
```

---

## Role Detection (Automatic)

Bones are classified by name keywords:

| Role | Keywords | Behavior |
|------|----------|----------|
| core | spine, hip, pelvis, torso, chest, root | Limited rotation - holds body together |
| limb | arm, leg, thigh, shin, forearm, shoulder | Loose - collapses freely |
| head | head, neck, skull | Medium limits |
| hand | hand, finger, thumb, foot, toe, ankle | Very loose - dangles |

Fallback: bones deeper than 5 levels → "hand", otherwise "limb"

---

## Data Flow

```
Trigger RAGDOLL reaction
        │
        ▼
┌─────────────────────────────────────────┐
│  CAPTURE (once at start)                │
│  - Per-bone rest matrix                 │
│  - Role detection                       │
│  - Initial rotations                    │
│  - Activate PHYSICS channel             │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  EACH FRAME (30Hz)                      │
│                                         │
│  Main Thread:                           │
│  1. Update position drop (gravity)      │
│  2. Submit bone data to worker          │
│                                         │
│  Worker:                                │
│  3. For each bone:                      │
│     - Project gravity → bone local      │
│     - Compute torque from gravity       │
│     - Integrate velocity + damping      │
│     - Clamp to joint limits             │
│  4. Return new rotations                │
│                                         │
│  Main Thread:                           │
│  5. Apply rotations to pose bones       │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  FINISH (duration ends)                 │
│  - Restore pose                         │
│  - Deactivate PHYSICS channel           │
│  - Resume normal animations             │
└─────────────────────────────────────────┘
```

---

## Files

| File | Purpose |
|------|---------|
| `reactions/exp_ragdoll.py` | Main thread: capture, drop, apply |
| `engine/worker/reactions/ragdoll.py` | Worker: bone physics |
| `animations/layer_manager.py` | PHYSICS channel |

---

## Debug

Enable: Developer Tools → Game Systems → ✅ Ragdoll
Export: Developer Tools → ✅ Export Session Log

Logs show:
- `DROP z=X vel=Y` - Position drop progress
- `BONE name: phys=(X,Y,Z) vel=(X,Y,Z)` - Applied rotations
- `WORKER: N ragdolls, M bones, Xus` - Worker timing

---

## Tuning

If collapse is too slow:
- Increase `GRAVITY_STRENGTH` (worker)
- Increase `DROP_GRAVITY` (main thread)

If collapse is too chaotic:
- Increase `DAMPING` (worker)
- Decrease `MAX_ANG_VEL` (worker)

If limbs don't droop enough:
- Increase `LIMITS["limb"]` and `LIMITS["hand"]`

If spine bends too much:
- Decrease `LIMITS["core"]`
