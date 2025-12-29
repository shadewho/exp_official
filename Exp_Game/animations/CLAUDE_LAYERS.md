# Animation Layer System

## Overview

Per-object priority system for coordinating animation sources.
Every object with actions gets its own LayerManager.

---

## Channels

```python
class AnimChannel(IntEnum):
    BASE = 0       # Default animation (locomotion, custom action)
    OVERRIDE = 1   # Triggered animations (reactions, one-shots)
    PHYSICS = 2    # Ragdoll, rigid body, cloth
    # Future: IK = 3, PROCEDURAL = 4
```

Higher priority wins. When PHYSICS is active, BASE is blocked.

---

## How PHYSICS Channel Blocks Animations

When `AnimChannel.PHYSICS` is active, `blend_system.apply_to_armature()` returns early without applying any transforms. This allows ragdoll (or other physics) to have exclusive control.

```python
# In blend_system.py:apply_to_armature()
layer_mgr = get_layer_manager_for(armature)
if layer_mgr and layer_mgr.is_channel_active(AnimChannel.PHYSICS):
    return 0  # Skip - physics has control
```

---

## Usage

```python
from ..animations.layer_manager import get_layer_manager_for, AnimChannel

# Get manager for any object
mgr = get_layer_manager_for(my_object)

# Activate a channel (takes priority)
mgr.activate_channel(AnimChannel.PHYSICS, influence=1.0, fade_in=0.0)

# Deactivate with fade
mgr.deactivate_channel(AnimChannel.PHYSICS, fade_out=0.3)

# Check what's active
if mgr.is_channel_active(AnimChannel.PHYSICS):
    # Physics has control
    pass

# Get highest priority active channel
active = mgr.get_active_channel()
```

---

## Lifecycle

| When | What Happens |
|------|--------------|
| Game Start | `init_layer_managers()` creates manager for every animated object |
| Each Frame | `update_all_managers(dt)` handles fade transitions |
| Game End | `shutdown_layer_managers()` restores all native actions |

---

## Current Users

| System | Channel | Purpose |
|--------|---------|---------|
| Ragdoll | PHYSICS | Blocks blend_system, allows ragdoll transforms |

---

## Future Integration Points

| System | Channel | When Ready |
|--------|---------|------------|
| Custom Actions | OVERRIDE | Wire up when needed |
| Rigid Body | PHYSICS | Add when implemented |
| Cloth Sim | PHYSICS | Add when implemented |
| IK | (new channel) | Add when implemented |

---

## Files

| File | Purpose |
|------|---------|
| `layer_manager.py` | Core system |
| `blend_system.py` | Checks PHYSICS channel in apply_to_armature() |
| `exp_engine_bridge.py` | Init/shutdown calls |
| `exp_loop.py` | Per-frame update |
| `exp_ragdoll.py` | PHYSICS channel user |
