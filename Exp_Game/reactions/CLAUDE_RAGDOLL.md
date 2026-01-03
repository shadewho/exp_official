# Verlet Particle Ragdoll System

**Status:** Full physics ragdoll with mesh collision

## Overview

Ragdoll that works on ANY armature using Verlet particle physics:
- Each bone joint is a free-moving particle
- Bones are distance constraints (like strings)
- Particles collide with static and dynamic proxy meshes
- True floppy collapse - each segment independent

---

## How It Works

### The Verlet Model

```
Traditional (BAD)                Verlet (GOOD)
─────────────────                ────────────────
Bones rotate on joints           Each joint is a particle
Parent constrains child          Particles connected by springs

    ●──────●──────●                  ●......●......●
    │      │      │                  .      .      .
    │ rot  │ rot  │                  . dist . dist .
    ▼      ▼      ▼                  ▼      ▼      ▼
Whole thing tilts              Each segment falls independently
```

### Physics Steps (Worker)

1. **Verlet Integration**: Apply gravity, compute velocity from position history
2. **Distance Constraints**: Iterate to maintain bone lengths (8 iterations)
3. **Mesh Collision**: Push particles out of static/dynamic meshes
4. **Re-constrain**: Satisfy distances again after collision
5. **Floor Clamp**: Ensure nothing goes below floor

### Data Flow

```
Trigger RAGDOLL reaction
        │
        ▼
┌───────────────────────────────────────┐
│  CAPTURE (once at start)              │
│  - Build particles from bone joints   │
│  - Create distance constraints        │
│  - Map bones to particle indices      │
│  - Activate PHYSICS channel           │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  EACH FRAME (30Hz)                    │
│                                       │
│  Main Thread:                         │
│  1. Submit particle data to worker    │
│                                       │
│  Worker:                              │
│  2. Verlet integration (gravity)      │
│  3. Satisfy distance constraints      │
│  4. Collide with cached meshes        │
│  5. Return new particle positions     │
│                                       │
│  Main Thread:                         │
│  6. Convert positions to bone rots    │
│  7. Apply to armature                 │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  FINISH (duration ends)               │
│  - Restore initial pose               │
│  - Deactivate PHYSICS channel         │
│  - Resume normal animations           │
└───────────────────────────────────────┘
```

---

## Physics Constants

### Worker (ragdoll.py)

```python
GRAVITY = (0.0, 0.0, -9.8)
GRAVITY_SCALE = 1.0           # Multiplier

CONSTRAINT_ITERATIONS = 8     # More = stiffer bones
CONSTRAINT_STIFFNESS = 0.9    # 0-1, bone rigidity

DAMPING = 0.98                # Velocity retention
GROUND_FRICTION = 0.7         # Friction on ground

COLLISION_RADIUS = 0.05       # Particle sphere size
```

### Main Thread (exp_ragdoll.py)

```python
FIX_ROOT = False              # If True, root stays in place
```

---

## Mesh Collision

The worker has access to:
- `cached_grid`: Static mesh triangles in spatial grid
- `cached_dynamic_meshes`: Dynamic mesh triangle data
- `cached_dynamic_transforms`: Current transforms of dynamic meshes

Each particle checks nearby triangles and is pushed out if inside:
1. Find closest point on triangle
2. If distance < COLLISION_RADIUS, push along triangle normal
3. Reflect velocity for bounce

---

## Particle Building

From armature bones, we build:

1. **Particles**: One per unique joint position
   - Bones sharing a joint share a particle
   - Root bones can optionally be fixed

2. **Constraints**: One per bone (head → tail)
   - Rest length = original bone length
   - Maintains skeleton structure

3. **Bone Map**: bone_name → (head_idx, tail_idx)
   - Used to convert particles back to rotations

---

## Converting Particles to Rotations

For each bone:
1. Get head and tail particle world positions
2. Calculate target direction: `(tail - head).normalized()`
3. Get bone's rest direction in world space
4. Calculate rotation difference
5. Convert to local rotation (relative to parent)
6. Apply as quaternion

---

## Files

| File | Purpose |
|------|---------|
| `reactions/exp_ragdoll.py` | Main thread: capture, submit, apply |
| `engine/worker/reactions/ragdoll.py` | Worker: Verlet physics + collision |
| `animations/layer_manager.py` | PHYSICS channel |

---

## Debug Logging

Enable: `bpy.data.scenes["Scene"].dev_debug_ragdoll = True`
Or: Developer Tools > Game Systems > Ragdoll toggle

Logs show:
- `START Verlet ragdoll on X, duration=Ys`
- `Built N particles, M constraints from K bones`
- `WORKER: grid_cells=X dynamic_meshes=Y`
- `UPDATE ragdoll N: X particles, Y constraints`
- `COLLISION: N particles hit mesh`
- `P0=(x,y,z) P1=(x,y,z) P2=(x,y,z)` - particle positions
- `FINISHED ragdoll N`

---

## Tuning Guide

### Ragdoll too stiff / doesn't flop
- Decrease `CONSTRAINT_STIFFNESS` (0.5-0.9)
- Decrease `CONSTRAINT_ITERATIONS` (4-6)

### Ragdoll too floppy / explodes
- Increase `CONSTRAINT_STIFFNESS` (0.9-1.0)
- Increase `CONSTRAINT_ITERATIONS` (10-15)

### Falls too slow
- Increase `GRAVITY_SCALE` (1.5-2.0)

### Bounces too much
- Decrease `DAMPING` (0.9-0.95)
- Increase `GROUND_FRICTION` (0.8-0.9)

### Clips through meshes
- Increase `COLLISION_RADIUS` (0.1-0.2)
- Ensure meshes are cached (check grid_cells in logs)

### Root flies away
- Set `FIX_ROOT = True` in exp_ragdoll.py

---

## Architecture: Why Verlet?

Traditional approaches:
1. **Rotation-based**: Each bone rotates. Parent constrains child. Result: whole thing tilts together.
2. **Spring physics**: Bones fight each other. Complex tuning. Often unstable.

Verlet advantages:
1. **Position-based**: Particles move freely, constraints resolve after
2. **Unconditionally stable**: Can't explode (unlike Euler integration)
3. **Simple collision**: Just push particles out of meshes
4. **Natural damping**: Built into position history
5. **Ignores hierarchy**: Each particle independent, skeleton emerges from constraints
