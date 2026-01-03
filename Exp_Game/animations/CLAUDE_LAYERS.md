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

Higher priority wins. When PHYSICS is active, BASE and OVERRIDE are blocked.

---

## How PHYSICS Channel Blocks Animations

When `AnimChannel.PHYSICS` is active, the animation pipeline is **completely skipped**:

### In `_update_character_animation()` (exp_loop.py)

```python
# 1. Layer managers ALWAYS updated (for fade transitions)
update_all_managers(agg_dt)

# 2. Check if physics has control
physics_active = layer_manager.is_channel_active(AnimChannel.PHYSICS)

if physics_active:
    return  # Skip ALL animation processing:
            # - No animation state updates
            # - No animation jobs submitted
            # - No animation results polled
            # - No blend system applied
```

**Why skip job submission?** If we submit animation jobs during ragdoll but skip applying them, the results become orphaned. Next frame, a new job overwrites `_pending_anim_batch_job`, and the old result is silently dropped. This caused animations to never resume after ragdoll.

### Safety Check in Fallback Handler

```python
elif result.job_type == "ANIMATION_COMPUTE_BATCH":
    if char_physics_active:
        # Discard orphaned result from before physics activated
        continue
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
| Each Frame | `update_all_managers(dt)` handles fade transitions (ALWAYS runs) |
| Game End | `shutdown_layer_managers()` restores all native actions |

---

## Debug Logging

Enable **"Layer Logs"** in Developer Tools > Animation 2.0 section.

Log messages include:
- `INIT` - Layer managers created for each object
- `ACTIVATE` - Channel activated with influence and fade
- `DEACTIVATE` - Channel deactivating with fade
- `SKIP animation jobs` - Animation pipeline skipped (PHYSICS active)
- `DISCARD orphaned ANIMATION result` - Stale result from pre-physics discarded
- Status lines showing all active channels per object

---

## Ragdoll Integration

When ragdoll starts:
1. `exp_ragdoll.py` calls `layer_manager.activate_channel(AnimChannel.PHYSICS)`
2. `_update_character_animation()` detects physics active, skips animation jobs
3. Ragdoll system has full control of bone transforms

When ragdoll ends:
1. `exp_ragdoll.py` calls `layer_manager.deactivate_channel(AnimChannel.PHYSICS, fade_out=0.3)`
2. `update_all_managers()` processes the fade (runs every frame even during physics)
3. Once influence reaches 0, channel becomes inactive
4. Animation jobs resume automatically

---

## Files

| File | Purpose |
|------|---------|
| `layer_manager.py` | Core system |
| `blend_system.py` | Checks PHYSICS channel in apply_to_armature() |
| `exp_engine_bridge.py` | Init/shutdown calls |
| `exp_loop.py` | Per-frame update + animation pipeline skip when PHYSICS active |
| `exp_ragdoll.py` | PHYSICS channel user |
| `dev_properties.py` | `dev_debug_layers` toggle |
| `dev_logger.py` | LAYERS log category |
