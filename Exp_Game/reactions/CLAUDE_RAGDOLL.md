# Ragdoll System (Rig Agnostic)

**Status:** 6/10 - Functional but needs improvements

## Overview

Articulated-rod ragdoll physics that works on **ANY armature**.
- Gravity torque projected into bone local space
- Parent-alignment torque prevents curling into a ball
- Position drop physics (character falls to ground)
- Per-bone stiffness/damping/limits based on role detection
- KCC handles normal character position - ragdoll drops armature independently

---

## Current Implementation

### What Works
1. **Bone rotation physics** - Bones respond to gravity with per-bone parameters
2. **Position drop** - Character falls to ground during ragdoll (0.5s drop phase)
3. **Role detection** - Automatically detects core/limb/head/hand from bone names
4. **Layer Manager integration** - PHYSICS channel blocks blend_system
5. **Ragdoll finish** - Returns to locomotion after duration ends

### Known Issues / Improvements Needed
1. **Bones still tend to curl** - Parent-alignment helps but not enough
2. **Drop feels disconnected** - Armature drops but bones don't respond naturally to the fall
3. **No collision with environment** - Only simple floor plane check
4. **KCC may fight ragdoll** - Need to fully disable KCC during ragdoll
5. **Ground contact is primitive** - Just a Z-plane check, no actual collision

---

## Architecture

### Two-Part Physics System

**Main Thread (exp_ragdoll.py):**
- Position drop physics (gravity + floor collision)
- Simple and cheap - runs every frame
- Moves armature.location.z down with gravity

**Worker (ragdoll.py):**
- Bone rotation physics (articulated-rod model)
- Per-bone gravity torque projection
- Parent-alignment torque
- Angular velocity clamping

### Data Flow

```
Main Thread                              Worker
-----------                              ------
1. Capture bone data (once at start)
2. Each frame:
   - Update position drop (gravity)
   - Submit bone data to worker    -->   3. Compute bone physics
   - Poll results                  <--   4. Return bone rotations
   - Apply bone rotations
```

---

## Physics Model

### Position Drop (Main Thread)
```python
DROP_GRAVITY = -12.0        # m/s^2 (softer for game feel)
DROP_DAMPING = 0.95         # Bounce damping
DROP_DURATION = 0.5         # Drop phase length

# Each frame:
velocity += gravity * dt
position += velocity * dt
if position < floor:
    position = floor
    velocity *= -damping  # Bounce
```

### Bone Rotation (Worker)
```python
# Per-bone state: rot (euler), ang_vel (rad/s)

# 1. Project gravity into bone local space
local_gravity = bone_rest_inv @ armature_rot_inv @ world_gravity

# 2. Gravity torque on primary axes
torque_x = -local_gravity.z * 0.6 * cos(rot.x)  # Forward/back
torque_z = local_gravity.x * 0.6 * cos(rot.z)   # Side tilt
torque_y = local_gravity.y * 0.05 * cos(rot.y)  # Twist (tiny)

# 3. Parent alignment (prevents curl)
torque_x += -0.4 * sin(rot.x)
torque_z += -0.4 * sin(rot.z)

# 4. Spring-to-rest (weak)
torque -= stiffness * rot

# 5. Integrate with clamping
ang_vel += (torque / inertia) * dt
ang_vel *= damping
ang_vel = clamp(ang_vel, -6.0, 6.0)  # Prevent runaway
rot += ang_vel * dt
rot = clamp(rot, limits)
```

---

## Role-Based Parameters

| Role | Keywords | Stiffness | Damping | Inertia |
|------|----------|-----------|---------|---------|
| core | spine, hip, pelvis, torso, chest, root | 0.8 | 0.90 | 1.2 |
| limb | arm, leg, thigh, shin, forearm, shoulder | 0.3 | 0.88 | 1.0 |
| head | head, neck, skull | 0.4 | 0.90 | 0.7 |
| hand | hand, finger, thumb, foot, toe, ankle | 0.15 | 0.85 | 0.5 |

Fallback: bones deeper than 5 levels = "hand", otherwise "limb"

---

## Physics Constants (Worker)

```python
WORLD_GRAVITY = (0.0, 0.0, -9.8)

# Gravity torque multipliers
GRAVITY_TORQUE_PRIMARY = 0.6   # X/Z axes
GRAVITY_TORQUE_TWIST = 0.05    # Y axis (keep tiny)

# Limits
DEFAULT_LIMIT = 1.4            # ~80 degrees
DEFAULT_SECONDARY_LIMIT = 0.7  # ~40 degrees

# Velocity
MAX_ANGULAR_VELOCITY = 6.0     # rad/s

# Parent alignment
PARENT_ALIGN_STRENGTH = 0.4
```

---

## Lessons Learned

### Why Character Was Curling (Not Falling)
1. **KCC holds character up** - KCC physics keeps character standing
2. **Spring-to-rest was too strong** - Bones snapped back instead of flopping
3. **No position drop** - Only rotating bones in place = curl, not fall
4. **Gravity wasn't projected correctly** - Torque wasn't based on actual bone orientation

### Solutions Applied
1. Added position drop physics (main thread moves armature down)
2. Reduced stiffness dramatically (5.0 -> 0.3)
3. Added parent-alignment torque to prevent curling
4. Increased gravity torque multiplier (0.3 -> 0.6)
5. Added angular velocity clamping (prevents runaway)
6. Reduced twist influence (Y axis -> 0.05)

### Future Improvements Needed
1. **Disable KCC during ragdoll** - KCC may still be fighting the drop
2. **Better parent coupling** - Use actual parent direction, not just sin(angle)
3. **Environment collision** - Ray/sphere casts for walls, not just floor
4. **Impact direction** - Initial impulse based on damage direction
5. **Settling animation** - Procedural "get up" transition back to locomotion

---

## Files

| File | Purpose |
|------|---------|
| `reactions/exp_ragdoll.py` | Main thread - position drop, submit jobs, apply rotations |
| `engine/worker/reactions/ragdoll.py` | Worker - bone rotation physics |
| `animations/layer_manager.py` | PHYSICS channel blocks animations |
| `animations/blend_system.py` | Checks PHYSICS channel before apply |

---

## Debug

Enable: Developer Tools > Game Systems > Ragdoll checkbox
Logs: `diagnostics_latest.txt` with `[RAGDOLL ...]` prefix

Key log messages:
- `Starting ragdoll on X, duration=Xs` - Ragdoll triggered
- `PHYSICS channel activated` - Layer manager blocking animations
- `WORKER: N ragdolls, M bones, Xus` - Worker processing
- `Ragdoll X finished - restoring locomotion` - Ragdoll ended
