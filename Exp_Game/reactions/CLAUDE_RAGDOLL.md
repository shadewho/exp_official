# Ragdoll System Documentation

## Overview

The ragdoll system is a REACTION type that makes armature bones go limp/loose for a duration.
It runs through the engine worker system to offload physics computation from the main thread.

---

## Architecture

```
Node Editor (Design Time)         Main Thread                    Worker Process
┌─────────────────────┐          ┌─────────────────────┐        ┌─────────────────────┐
│ ReactionRagdollNode │          │ exp_ragdoll.py      │        │ worker/reactions/   │
│ - Duration          │ ──────►  │ - execute_ragdoll   │ ─JOB─► │   ragdoll.py        │
│ - Gravity Mult      │          │ - submit_update     │        │ - physics calc      │
│ - Impulse Strength  │          │ - process_results   │ ◄RESULT│ - rotation offsets  │
│ - Impulse Direction │          │ - apply to armature │        │ - position offsets  │
└─────────────────────┘          └─────────────────────┘        └─────────────────────┘
```

---

## File Connections

### Node System (Design Time)
- `Exp_Nodes/reaction_nodes.py` - `ReactionRagdollNode` class
- `Exp_Nodes/utility_nodes.py` - `ExpFloatSocket`, `ExpVectorSocket` for inputs
- Properties stored in `ReactionDefinition.ragdoll_*` fields

### Main Thread (Runtime)
- `Exp_Game/reactions/exp_ragdoll.py` - Executor and state management
- `Exp_Game/reactions/exp_interactions.py` - Dispatcher calls `execute_ragdoll_reaction()`
- `Exp_Game/reactions/exp_reaction_definition.py` - RAGDOLL enum and properties
- `Exp_Game/modal/exp_loop.py` - Submits jobs, processes results, has animation skip logic
- `Exp_Game/animations/blend_system.py` - `start_ragdoll_lock()`, `is_ragdoll_active()`

### Engine Worker (Computation)
- `Exp_Game/engine/worker/reactions/ragdoll.py` - `handle_ragdoll_update_batch()`
- `Exp_Game/engine/worker/reactions/__init__.py` - Exports handler
- `Exp_Game/engine/worker/entry.py` - Job dispatch for RAGDOLL_UPDATE_BATCH

---

## Data Flow

### Reaction Trigger
1. Interaction fires RAGDOLL reaction
2. `exp_interactions.py` dispatches to `execute_ragdoll_reaction()`
3. Creates `RagdollInstance` with bone order, parents, lengths, states
4. Calls `blend_system.start_ragdoll_lock()` to block animations

### Per-Frame Update
1. `exp_loop.py` checks `has_active_ragdolls()`
2. Calls `submit_ragdoll_update(engine, dt)`
3. Worker computes physics:
   - Root bone: position offset (gravity + floor collision)
   - Other bones: rotation offset (droop with angular velocity)
4. Results processed via `process_ragdoll_results()`
5. `_apply_ragdoll()` sets `pb.location` and `pb.rotation_quaternion`

### Animation Blocking (Attempted)
- `exp_loop.py` checks `blend_system.is_ragdoll_active()`
- Skips `poll_animation_results_with_timeout()` if ragdoll active
- Skips `blend_system.apply_to_armature()` if ragdoll active

---

## Current Physics Model

### Root Bone
- Tracks position OFFSET from rest pose (starts at 0,0,0)
- Applies 30% of scene gravity
- Damping: 0.95
- Floor constraint at Z=0

### Non-Root Bones
- Tracks rotation OFFSET in euler angles
- Droop rate: -0.5 rad/s^2 (bones tend to fall forward)
- Damping: 0.92
- Clamped to +/- 0.8 radians (~45 degrees)

---

## Current Problems

### Problem 1: Blender Action System Override

The ragdoll sets pose bone transforms via Python:
```python
pb.location = Vector(pos_offset)
pb.rotation_quaternion = euler.to_quaternion()
```

But Blender has an active action on `armature.animation_data.action`. Blender's
internal animation evaluator runs every frame and overwrites pose bone transforms
from the action - completely bypassing Python code.

Result: Visual flickering as ragdoll and locomotion action fight for control.

The blocking code only stops our custom animation systems. It does NOT stop
Blender's internal action evaluation which happens at the C level.

### Problem 2: Weak Visual Effect

Current physics produces minimal visible change:
- Root moves up (from impulse) but gravity is weak (30%)
- Bone rotations are tiny: 0.00 to -0.03 radians over several seconds
- Droop rate with damping converges too slowly
- No real "jelly" or "loose body" feel

### Problem 3: No Collision

Bones only have floor constraint at Z=0. No collision with:
- Static geometry (spatial grid)
- Dynamic meshes
- Self-collision between bones

---

## Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `ragdoll_target_use_character` | Bool | True | Use scene character armature |
| `ragdoll_target_armature` | Pointer | None | Custom armature if not character |
| `ragdoll_duration` | Float | 2.0 | How long ragdoll lasts (seconds) |
| `ragdoll_gravity_multiplier` | Float | 1.0 | Scale gravity effect |
| `ragdoll_impulse_strength` | Float | 5.0 | Initial velocity magnitude |
| `ragdoll_impulse_direction` | Vector | (0,0,1) | Initial velocity direction |

---

## Debug

Enable logging: Developer Tools > Game Systems > Ragdoll checkbox
Logs output to `diagnostics_latest.txt` with `[RAGDOLL ...]` prefix
