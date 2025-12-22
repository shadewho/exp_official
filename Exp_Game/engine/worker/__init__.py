# Exp_Game/engine/worker/__init__.py
"""
Worker process modules - ISOLATED from Blender imports.
These modules run in separate worker processes and do NOT import bpy.

Structure:
    worker/
    ├── math.py          # Geometric algorithms
    ├── raycast.py       # Unified raycast system
    ├── physics.py       # KCC physics step
    ├── jobs.py          # Camera occlusion, etc.
    ├── entry.py         # Job dispatcher + worker loop
    ├── interactions/    # Trigger evaluation
    │   ├── triggers.py  # PROXIMITY, COLLISION checks
    │   └── trackers.py  # Condition tree evaluation
    └── reactions/       # Reaction execution
        ├── projectiles.py  # Physics simulation
        └── hitscan.py      # Instant raycasting
"""

# Math utilities available for import
from .math import (
    ray_triangle_intersect,
    compute_triangle_normal,
    compute_facing_normal,
    ray_sphere_intersect,
    compute_bounding_sphere,
    compute_aabb,
    transform_aabb_by_matrix,
    ray_aabb_intersect,
    invert_matrix_4x4,
    transform_ray_to_local,
    transform_point,
    transform_triangle,
    get_adaptive_grid_resolution,
    build_triangle_grid,
    ray_grid_traverse,
)

# Raycast functions
from .raycast import (
    unified_raycast,
    cast_ray,
    test_dynamic_meshes_ray,
)

# Physics handlers
from .physics import handle_kcc_physics_step

# Job handlers
from .jobs import handle_camera_occlusion

# Interaction handlers
from .interactions import (
    handle_interaction_check_batch,
    handle_cache_trackers,
    handle_evaluate_trackers,
    reset_tracker_state,
)

# Reaction handlers
from .reactions import (
    handle_projectile_update_batch,
    reset_projectile_state,
    handle_hitscan_batch,
)

# Entry point (worker_loop and process_job)
from .entry import worker_loop, process_job

__all__ = [
    # Math utilities
    'ray_triangle_intersect',
    'compute_triangle_normal',
    'compute_facing_normal',
    'ray_sphere_intersect',
    'compute_bounding_sphere',
    'compute_aabb',
    'transform_aabb_by_matrix',
    'ray_aabb_intersect',
    'invert_matrix_4x4',
    'transform_ray_to_local',
    'transform_point',
    'transform_triangle',
    'get_adaptive_grid_resolution',
    'build_triangle_grid',
    'ray_grid_traverse',
    # Raycast functions
    'unified_raycast',
    'cast_ray',
    'test_dynamic_meshes_ray',
    # Physics handlers
    'handle_kcc_physics_step',
    # Job handlers
    'handle_camera_occlusion',
    # Interaction handlers
    'handle_interaction_check_batch',
    'handle_cache_trackers',
    'handle_evaluate_trackers',
    'reset_tracker_state',
    # Reaction handlers
    'handle_projectile_update_batch',
    'reset_projectile_state',
    'handle_hitscan_batch',
    # Entry point
    'worker_loop',
    'process_job',
]
