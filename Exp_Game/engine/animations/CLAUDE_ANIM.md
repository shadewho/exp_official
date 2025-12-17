# Animation System Vision

## Overview

This animation system is a ground-up rework designed to replace the old NLA/timeline-based approach. The goal is to build something that works like real game engines do - fast, flexible, and user-friendly.

The old system is confusing and not user-friendly. It relies on Blender's NLA system which was designed for film/linear workflows, not real-time game systems. We want direct control.

---

## Core Priorities

### 1. Performance
- Main thread must stay as thin as possible
- All heavy computation (sampling, blending, interpolation) happens in the worker
- Main thread only manages state and applies final pose to bones

### 2. Blending is the Default
- We should ALWAYS be blending animations from one to the next
- Crossfade is the normal transition, not an exception
- Even a single animation is just a blend with weight=1.0

### 3. Scalability
- The old system has no way to blend animations
- The old system has no way to incorporate automatic/smart animation systems
- When developing, always consider the possibility for overrides (e.g., "pick item up" interrupting locomotion)
- Architecture must support future features without major refactors

### 4. Fast Startup
- This connects to a web app where users download and play worlds on the fly
- All caching and baking must be contained in the SCENE
- Must be reliable and fast - no 60 second baking startup sessions
- Lazy baking on first use is acceptable if it's fast enough

### 5. Compatibility
- Must work for all Blender 5.0+ users
- No exotic dependencies or requirements

---

## Key Design Decisions

### Use Physics Data
The animation system should use data from the physics system as much as possible. Don't duplicate context detection - leverage what physics already knows (grounded state, velocity, surface info, etc.).

### Worker-Based Computation
Animation computation needs its own job type in the worker system. The pattern:
1. Main thread manages what's playing (instances, weights, times)
2. Worker samples animations and computes blended pose
3. Main thread applies result to bones

### No Smart System Yet
Smart/automatic animation selection will come much later. For now, focus on:
- Solid foundation
- Clean blending
- Good performance
- Extensible architecture

The smart system (context-aware animation selection, auto-reactions, etc.) can be built on top once the foundation is solid.

---

## Future Vision

Eventually, users should be able to:
- Drop in an armature with animations and have basic locomotion "just work"
- Override any animation with context-specific actions (grab, duck, wall reactions)
- Not worry about animation if they just want to design scenes
- Have full control if they want to customize everything

The system should feel like Unity/Unreal animation systems - powerful but approachable.

---

## What We're Moving Away From

- NLA tracks and strips
- Timeline/playhead-based playback
- Manual frame scrubbing
- Global animation state
- One-animation-at-a-time limitations

## What We're Moving Toward

- Direct bone manipulation via computed poses
- Worker-offloaded blending math
- Multiple simultaneous animations with weights
- Layer-based architecture (base + overrides + additives)
- Context-aware animation selection (future)

---

## Context: 2025-12-17

**Blender 5.0+ Only** - This system targets Blender 5.0 and later. The new layered action API (`action.layers → strips → channelbags → fcurves`) is used exclusively. No legacy API support.

**Tests Completed:**
- Baking: Converts Blender Actions to BakedAnimation (tuple-based, worker-safe). 6 actions baked in ~140ms.
- Apply: Direct bone manipulation works. Setting `pose_bone.rotation_quaternion`, `location`, `scale` from baked data.
- Play: Time-based playback with looping. 60fps timer, frame index from elapsed time.
- Blend: Two-animation blending with slerp (quaternion) and lerp (location/scale). Mouse-controlled weight. Works smoothly.

**Confirmed:**
- Lighter than NLA system (no dependency graph evaluation, no timeline)
- Baked data format: `(qw, qx, qy, qz, lx, ly, lz, sx, sy, sz)` per bone per frame
- Blending math is correct and produces smooth results

**Next:** Remove test scaffolding, integrate into game loop using physics data to drive animation state.

---

## Context: 2025-12-17 (Phase 2 Complete)

### Current State

**Phase 2 integration is complete.** The old NLA-based animation system has been fully replaced with the new unified system. This is still an early foundation - much work remains, but the core architecture is in place and functional.

### What We Had (Old System - DELETED)

- `animations/exp_animations.py` - Character locomotion via NLA tracks
- `animations/exp_custom_animations.py` - Object animations via NLA tracks
- Required NLA track management, strip creation/deletion, timeline scrubbing
- Complex cleanup with "NLA guards" to prevent race conditions
- Global animation manager with state scattered across multiple systems

### What We Have Now (New System)

**Worker-Safe Components (`engine/animations/`):**
| File | Purpose |
|------|---------|
| `data.py` | `BakedAnimation` - numpy arrays of transforms per bone/object per frame |
| `cache.py` | `AnimationCache` - stores baked animations by name |
| `blend.py` | Transform math (quaternion slerp, vector lerp) |
| `baker.py` | `bake_action()` - converts Blender Action → BakedAnimation |

**Main Thread Components (`animations/`):**
| File | Purpose |
|------|---------|
| `controller.py` | `AnimationController` - playback orchestration, blending, fading |
| `apply.py` | `apply_pose()` - writes transforms directly to bones/objects |
| `state_machine.py` | `CharacterStateMachine` - locomotion states (IDLE, WALK, RUN, JUMP, FALL, LAND) |
| `test_panel.py` | UI panel for testing bake/play/blend |

**Integration Points:**
- `modal/exp_engine_bridge.py` - `init_animations()`, `shutdown_animations()`, `update_animations()`
- `modal/exp_loop.py` - `_update_character_animation()` calls state machine + controller each frame
- `reactions/exp_reactions.py` - `execute_char_action_reaction()`, `execute_custom_action_reaction()`

### Critical Architecture Principle

**The main thread must stay lean and thin.**

All heavy computation belongs in `/engine`. The main thread should only:
1. Manage state (what's playing, what weights, what times)
2. Call into engine for computation
3. Apply final results to Blender objects

Currently `apply_pose()` runs on main thread because Blender requires it, but sampling and blending math should move to workers as the system matures.

### How Physics Drives Animation

The state machine uses physics data directly:
```python
new_state, changed = state_machine.update(
    keys_pressed=op.keys_pressed,
    delta_time=dt,
    is_grounded=op.is_grounded,      # From KCC
    vertical_velocity=op.z_velocity,  # From KCC
    game_time=game_time
)
```

No duplicate detection - physics is the source of truth for grounded state, velocity, etc.

### Lifecycle

1. **Game Start:** `init_animations()` bakes ALL actions in `bpy.data.actions` into cache
2. **Each Frame:** State machine evaluates → controller plays/blends → apply transforms
3. **Game End:** `shutdown_animations()` clears all cached data

### Future Vision: Nodes Integration

The long-term goal is a **unified animation engine** that integrates with the nodes system. The nodes system would:
- Select target object or armature
- Select action(s) to play
- Configure playback behavior (loop forever, loop N times, play once, ping-pong, etc.)
- Handle transitions and blending rules

This would replace hardcoded locomotion states with user-configurable animation graphs. The current state machine is a stepping stone - it proves the architecture works. The nodes system will build on top of `AnimationController` and `BakedAnimation` without changing the core.

### What Still Needs Work

- [ ] Worker offloading for animation sampling/blending (currently all main thread)
- [ ] Additive animation layers (overlay actions on top of base)
- [ ] Animation events/notifies (trigger sounds/effects at specific frames)
- [ ] Bone masks (apply animation to subset of skeleton)
- [ ] IK integration (procedural adjustments after animation)
- [ ] Nodes system integration
- [ ] Smart/context-aware animation selection

This is foundation work. The core is solid but there's significant development ahead.

---

