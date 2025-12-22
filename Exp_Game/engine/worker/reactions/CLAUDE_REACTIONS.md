# Reactions Worker System - Projectiles & Hitscans

## Overview

The projectile and hitscan systems have been offloaded from the main thread to the `/engine` worker process. This keeps the main Blender thread lean and responsive while heavy physics calculations (raycasting, collision detection, gravity integration) run in a separate process.

## Architecture

### Main Thread (Thin) - `Exp_Game/reactions/exp_projectiles.py`
- Queues projectile spawns and hitscan requests
- Submits batch jobs to worker (`PROJECTILE_UPDATE_BATCH`, `HITSCAN_BATCH`)
- Processes results and applies visuals (spawning clones, parenting, positioning)
- Handles impact events and node graph outputs
- Local visual interpolation for smooth motion between worker updates

### Worker Process (Heavy) - `Exp_Game/engine/worker/reactions/`
- `projectiles.py` - Gravity integration, sweep raycasting, collision detection
- `hitscan.py` - Instant raycast batching against static + dynamic geometry
- Uses `unified_raycast()` from `../raycast.py` for consistent physics

### Game Loop Integration - `Exp_Game/modal/exp_loop.py`
- `update_dynamic_meshes()` runs FIRST to cache mesh data and populate transforms
- `submit_hitscan_batch()` and `submit_projectile_update()` submit jobs to worker
- `interpolate_projectile_visuals()` runs every frame for smooth visual motion
- `update_hitscan_tasks()` handles visual cleanup/despawning

## Node System

**IMPORTANT**: The Exploratory node trees are VISUAL REPRESENTATION ONLY. Nodes are NEVER called during game runtime. They configure reactions at edit time, storing settings in `scene.reactions` PropertyGroups. The runtime system reads these properties directly.

### Relevant Node Outputs (need testing):
- **Impact Event** (Bool) - Fires when projectile/hitscan hits something
- **Impact Location** (Vector) - World position of impact point
- These outputs connect to other nodes (triggers, reactions) via `_flush_impact_events_to_graph()`

## Code Locations

| Component | Location |
|-----------|----------|
| Main thread projectile/hitscan | `Exp_Game/reactions/exp_projectiles.py` |
| Worker projectile physics | `Exp_Game/engine/worker/reactions/projectiles.py` |
| Worker hitscan physics | `Exp_Game/engine/worker/reactions/hitscan.py` |
| Unified raycast | `Exp_Game/engine/worker/raycast.py` |
| Dynamic mesh caching | `Exp_Game/physics/exp_dynamic.py` |
| Game loop orchestration | `Exp_Game/modal/exp_loop.py` |
| Reaction definitions | `Exp_Game/reactions/exp_reactions.py` |

## Logger System

The `dev_logger` is CRITICAL for debugging this system. Use:
```python
from ..developer.dev_logger import log_game
log_game("CATEGORY", "message")
```

Categories in use:
- `PROJECTILE` - Spawn, update, impact events
- `HITSCAN` - Queue, batch submit, results, dynamic mesh hits
- `DYN-CACHE` - Dynamic mesh caching to worker

Output goes to `C:\Users\spenc\Desktop\engine_output_files\diagnostics_latest.txt`

## Dynamic Mesh Integration

Both systems support collision with dynamic (moving) meshes:
1. `update_dynamic_meshes()` caches mesh triangles via `CACHE_DYNAMIC_MESH` jobs
2. `get_dynamic_transforms()` sends current world matrices with each batch job
3. Worker maintains `cached_dynamic_meshes` and `cached_dynamic_transforms`
4. `unified_raycast()` tests both static grid AND dynamic meshes

## Current Issues (2024-12-22)

### Hitscan + Dynamic Mesh NOT WORKING
- Hitscans work fine for static meshes
- Hitscans never hit dynamic meshes - visual spawns at wrong location
- Suspected issues:
  1. **Parenting bug**: After creating hitscan visual, we read `matrix_world.translation` before Blender updates it (should use `impact` vector directly)
  2. **Cache timing**: Need to verify dynamic mesh cache is populated before hitscan batch processes

### Needs Testing
- [ ] Impact Location output to connected nodes
- [ ] Impact Event triggering downstream reactions
- [ ] Projectile lifetime auto-despawn (fixed but needs verification)
- [ ] Multiple dynamic meshes simultaneously
- [ ] Dynamic mesh parenting (visual stays attached when mesh moves)

## Key Fixes Made Today

1. **Game loop ordering**: Moved `update_dynamic_meshes()` BEFORE batch submissions to ensure caches are ready

2. **Projectile lifetime despawn**: Worker now includes `client_id` in expired entries for reliable matching

3. **Job queue protection**: Added guards to prevent job backup when worker is slow

4. **Per-physics-step caching**: `get_dynamic_transforms()` caches per game time to avoid redundant computation

5. **Inflight hitscan separation**: Added `_inflight_hitscans` list to prevent data loss between submit and result processing

6. **Direction resolution fix**: When aiming at dynamic meshes (no static hit), use `cam_dir` directly instead of computing direction to far point

## Debug Logging Added

Submit logging:
```
HITSCAN BATCH_SUBMIT job=X rays=Y dyn_transforms=Z
```

Worker logging:
```
HITSCAN DYNAMIC_MESHES transforms_received=X cached_meshes=Y matched=Z available=W
```

Result logging:
```
HITSCAN RESULT hit=True/False source=static/dynamic dist=X.XXX pos=(x,y,z)
```
