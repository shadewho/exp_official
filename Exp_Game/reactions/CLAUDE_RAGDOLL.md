# Ragdoll System (Rig Agnostic)

**Status:** Simple pendulum collapse - works on ANY armature

## Overview

Simple ragdoll that makes any armature collapse like a marionette with cut strings.
- Every bone is a pendulum that falls toward gravity
- No complex physics - just collapse
- Works on ANY armature - no hardcoded bone names, no role detection

---

## How It Works

### The Simple Model

Each bone is treated as a pendulum hanging from its parent:
1. Gravity pulls the bone's tip downward
2. Bone rotates around its joint toward gravity
3. Damping prevents wild swinging
4. Joint limits prevent unnatural poses

That's it. No spring-to-rest fighting the collapse, no parent alignment.

### Two-Part System

**Main Thread (exp_ragdoll.py):**
- Position drop: armature falls to ground with gravity
- Submits bone data to worker
- Applies rotation results to pose bones

**Worker (ragdoll.py):**
- Computes bone rotations based on gravity
- Simple: `torque = bone_Y x gravity` (pendulum physics)
- Returns new rotations each frame

---

## Physics Constants

### Position Drop (Main Thread)
```python
DROP_GRAVITY = -20.0    # Fast drop (m/s^2)
DROP_DAMPING = 0.3      # Low bounce
DROP_DURATION = 1.5     # Full collapse time
```

### Bone Rotation (Worker)
```python
GRAVITY_STRENGTH = 12.0  # How hard gravity pulls
BONE_DAMPING = 0.85      # Velocity decay (smooth settle)
BONE_LIMIT = 2.5         # Max rotation radians (~143 degrees)
MAX_ANG_VEL = 15.0       # Velocity cap
```

All bones use the SAME physics constants - no role detection, no special cases.

---

## Data Flow

```
Trigger RAGDOLL reaction
        |
        v
+---------------------------------------+
|  CAPTURE (once at start)              |
|  - Per-bone rest matrix               |
|  - Initial rotations                  |
|  - Activate PHYSICS channel           |
+---------------------------------------+
        |
        v
+---------------------------------------+
|  EACH FRAME (30Hz)                    |
|                                       |
|  Main Thread:                         |
|  1. Update position drop (gravity)    |
|  2. Submit bone data to worker        |
|                                       |
|  Worker:                              |
|  3. For each bone:                    |
|     - Project gravity -> bone local   |
|     - Compute torque from gravity     |
|     - Integrate velocity + damping    |
|     - Clamp to joint limits           |
|  4. Return new rotations              |
|                                       |
|  Main Thread:                         |
|  5. Apply rotations to pose bones     |
+---------------------------------------+
        |
        v
+---------------------------------------+
|  FINISH (duration ends)               |
|  - Restore pose                       |
|  - Deactivate PHYSICS channel         |
|  - Resume normal animations           |
+---------------------------------------+
```

---

## Files

| File | Purpose |
|------|---------|
| `reactions/exp_ragdoll.py` | Main thread: capture, drop, apply |
| `engine/worker/reactions/ragdoll.py` | Worker: bone physics |
| `animations/layer_manager.py` | PHYSICS channel |
| `developer/ragdoll_test.py` | Standalone test UI (calls worker directly) |

---

## Dev Testing

Standalone test in Developer Tools panel:
1. Select armature via `target_armature`
2. Developer 2.0 > Ragdoll Test section
3. Start Ragdoll Test button

Uses the same physics code as runtime - just calls worker function directly on main thread.

---

## Debug

Enable: Developer Tools > Game Systems > Ragdoll
Export: Developer Tools > Export Session Log

Logs show:
- `DROP z=X vel=Y` - Position drop progress
- `BONE name: T=(X,Z) R=(X,Y,Z)` - Torque and rotation
- `WORKER: N ragdolls, M bones, Xus` - Worker timing

---

## Tuning

If collapse is too slow:
- Increase `GRAVITY_STRENGTH` (worker)
- Increase `DROP_GRAVITY` (main thread)

If collapse is too chaotic:
- Increase `BONE_DAMPING` (worker)
- Decrease `MAX_ANG_VEL` (worker)

If bones don't droop enough:
- Increase `BONE_LIMIT`

If bones curl too aggressively:
- Decrease `GRAVITY_STRENGTH`
- Increase `BONE_DAMPING`
