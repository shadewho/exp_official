# Exp_Game/developer/dev_properties.py
"""
Developer debug properties - toggleable console output by category.

All properties are stored on bpy.types.Scene for easy access from operators.
Default: All debug flags are FALSE (silent console).

UNIFIED PHYSICS ARCHITECTURE:
Static and dynamic meshes use identical physics code paths.
All physics logs show source (static/dynamic) in the same format.
There is ONE physics system, not two separate ones.
"""

import bpy


def register_properties():
    """Register all developer debug properties on Scene."""

    # ══════════════════════════════════════════════════════════════════════════
    # MASTER FREQUENCY CONTROL
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_master_hz = bpy.props.IntProperty(
        name="Master Hz",
        description=(
            "Master debug output frequency for ALL categories:\n"
            "• 30 = every frame (verbose, use for short tests)\n"
            "• 5 = every 5th frame (~0.17s intervals)\n"
            "• 1 = once per second (recommended for most debugging)"
        ),
        default=1,
        min=1,
        max=30
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ENGINE HEALTH
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_engine = bpy.props.BoolProperty(
        name="Engine Health",
        description=(
            "Core multiprocessing engine diagnostics:\n"
            "• Worker startup/shutdown status\n"
            "• Jobs/sec throughput\n"
            "• Queue depth and saturation warnings\n"
            "• Result processing latency"
        ),
        default=False
    )

    bpy.types.Scene.dev_startup_logs = bpy.props.BoolProperty(
        name="Startup Logs",
        description=(
            "Show detailed engine and modal startup sequence:\n"
            "• Engine worker spawning\n"
            "• Grid cache confirmation\n"
            "• Health checks\n"
            "• READY confirmation"
        ),
        default=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SPATIAL GRID SYSTEM
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_grid = bpy.props.BoolProperty(
        name="Spatial Grid",
        description=(
            "Spatial grid diagnostics:\n"
            "• Cell size (adaptive or fixed)\n"
            "• Triangle density (tris/m³)\n"
            "• Average triangles per cell\n"
            "• Raycast cell traversal stats"
        ),
        default=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # OFFLOAD SYSTEMS
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_kcc_physics = bpy.props.BoolProperty(
        name="KCC Physics",
        description=(
            "Full physics step offload (KCC_PHYSICS_STEP):\n"
            "• SUBMIT: pos, vel, input, jump state\n"
            "• RESULT: new pos, ground state, collision flags\n"
            "• Timing: worker calc time (µs)\n"
            "• Stats: rays cast, triangles tested"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_frame_numbers = bpy.props.BoolProperty(
        name="Frame Numbers",
        description=(
            "Print frame numbers with timestamps:\n"
            "Shows: [FRAME 0042] t=1.400s"
        ),
        default=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # UNIFIED CAMERA (Static + Dynamic use identical unified_raycast)
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_camera = bpy.props.BoolProperty(
        name="Camera Raycast",
        description=(
            "Unified camera raycast (static + dynamic):\n"
            "• Uses same unified_raycast as KCC physics\n"
            "• HIT source: STATIC or DYNAMIC\n"
            "• Hit distance and allowed camera distance\n"
            "• Worker timing (µs)\n"
            "• Dynamic meshes tested count\n"
            "\n"
            "UNIFIED: Static and dynamic use IDENTICAL raycast code"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_culling = bpy.props.BoolProperty(
        name="Performance Culling",
        description=(
            "Distance-based object culling:\n"
            "• Hide/show operations per frame\n"
            "• Culling radius thresholds"
        ),
        default=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # UNIFIED PHYSICS (Static + Dynamic use identical code paths)
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_physics = bpy.props.BoolProperty(
        name="Physics Summary",
        description=(
            "Unified physics summary per frame:\n"
            "• Total computation time (µs)\n"
            "• Ray count and triangle count\n"
            "• Ground source (static / dynamic_ObjectID)\n"
            "• Static grid + dynamic mesh count\n"
            "\n"
            "UNIFIED: Static and dynamic use IDENTICAL physics code"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_ground = bpy.props.BoolProperty(
        name="Ground Detection",
        description=(
            "Ground raycast (downward from feet):\n"
            "• Ground Z position and normal\n"
            "• Source: static or dynamic_ObjectID\n"
            "• Snap distance and grounded state\n"
            "• Walkable vs steep detection"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_horizontal = bpy.props.BoolProperty(
        name="Horizontal Collision",
        description=(
            "Horizontal collision (walls/obstacles):\n"
            "• Ray hits at 3 heights (feet, mid, head)\n"
            "• Source: static or dynamic_ObjectID\n"
            "• Width rays (narrow gap detection)\n"
            "• Slope rays (angled detection)\n"
            "• Blocked distance and normal"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_body = bpy.props.BoolProperty(
        name="Body Integrity",
        description=(
            "Body integrity ray (embedding detection):\n"
            "• Vertical ray from feet to head\n"
            "• Source: static or dynamic_ObjectID\n"
            "• EMBEDDED status if ray blocked\n"
            "• Penetration depth and correction"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_ceiling = bpy.props.BoolProperty(
        name="Ceiling Check",
        description=(
            "Ceiling raycast (upward from head):\n"
            "• Source: static or dynamic_ObjectID\n"
            "• Hit distance and position correction\n"
            "• Velocity zeroing on ceiling hit"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_step = bpy.props.BoolProperty(
        name="Step-Up",
        description=(
            "Step-up (stair climbing):\n"
            "• Headroom check (source: static/dynamic)\n"
            "• Ground detection at drop position\n"
            "• Success/failure and final position"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_slide = bpy.props.BoolProperty(
        name="Wall Slide",
        description=(
            "Wall slide along surfaces:\n"
            "• Slide direction calculation\n"
            "• Collision check (source: static/dynamic)\n"
            "• Slide distance allowed"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_slopes = bpy.props.BoolProperty(
        name="Slopes",
        description=(
            "Slope handling:\n"
            "• Slope angle (degrees)\n"
            "• Uphill velocity blocking\n"
            "• Gravity slide on steep slopes\n"
            "• Z-clamp corrections"
        ),
        default=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # DYNAMIC MESH SYSTEM (Unified with static - same physics code path)
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_dynamic_cache = bpy.props.BoolProperty(
        name="Dynamic Mesh System",
        description=(
            "Dynamic mesh system diagnostics:\n"
            "• Triangle caching (one-time per mesh)\n"
            "• Transform updates sent to worker\n"
            "• Worker transform cache state\n"
            "\n"
            "UNIFIED: Dynamic meshes use same physics as static"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_dynamic_opt = bpy.props.BoolProperty(
        name="Dynamic Optimization Stats",
        description=(
            "Dynamic mesh collision optimization stats:\n"
            "• Meshes in proximity vs skipped by AABB\n"
            "• Rays cast vs rays skipped (AABB cull)\n"
            "• Triangles tested vs triangles skipped\n"
            "• Early-out hits (stopped testing other meshes)\n"
            "• Per-frame timing breakdown"
        ),
        default=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # KCC VISUAL DEBUG (3D Viewport Overlay)
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_kcc_visual = bpy.props.BoolProperty(
        name="Enable Visual Debug",
        description=(
            "Real-time 3D visualization of KCC physics:\n"
            "• Capsule collision shape\n"
            "• Hit normals (cyan = static, orange = dynamic)\n"
            "• Ground ray (magenta = hit, purple = miss)\n"
            "• Movement vectors (green = intended, red = actual)\n"
            "\n"
            "Capsule colors:\n"
            "• Green = grounded\n"
            "• Yellow = colliding\n"
            "• Red = stuck\n"
            "• Blue = airborne"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_kcc_visual_capsule = bpy.props.BoolProperty(
        name="Capsule Shape",
        description="Draw capsule collision shape (feet + mid + head spheres)",
        default=True
    )

    bpy.types.Scene.dev_debug_kcc_visual_normals = bpy.props.BoolProperty(
        name="Hit Normals",
        description="Draw collision normals (cyan = static, orange = dynamic)",
        default=True
    )

    bpy.types.Scene.dev_debug_kcc_visual_ground = bpy.props.BoolProperty(
        name="Ground Ray",
        description="Draw ground detection raycast (cyan = static hit, orange = dynamic hit, purple = miss)",
        default=True
    )

    bpy.types.Scene.dev_debug_kcc_visual_movement = bpy.props.BoolProperty(
        name="Movement Vectors",
        description="Draw movement vectors (green = intended, red = actual)",
        default=True
    )

    bpy.types.Scene.dev_debug_kcc_visual_line_width = bpy.props.FloatProperty(
        name="Line Width",
        description="Visual debug line thickness (1.0-10.0)",
        default=2.5,
        min=1.0,
        max=10.0
    )

    bpy.types.Scene.dev_debug_kcc_visual_vector_scale = bpy.props.FloatProperty(
        name="Vector Scale",
        description="Movement vector length multiplier",
        default=3.0,
        min=0.5,
        max=10.0
    )

    # ══════════════════════════════════════════════════════════════════════════
    # GAME SYSTEMS
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_interactions = bpy.props.BoolProperty(
        name="Interactions",
        description="Interaction and reaction system",
        default=False
    )

    bpy.types.Scene.dev_debug_audio = bpy.props.BoolProperty(
        name="Audio",
        description="Audio playback and state",
        default=False
    )

    bpy.types.Scene.dev_debug_animations = bpy.props.BoolProperty(
        name="Animations",
        description="Animation and NLA system",
        default=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SESSION LOG EXPORT
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_export_session_log = bpy.props.BoolProperty(
        name="Export Diagnostics Log",
        description=(
            "Export game diagnostics to text file when game ends:\n"
            "• Location: Desktop/engine_output_files/\n"
            "• File: diagnostics_latest.txt"
        ),
        default=False
    )


def unregister_properties():
    """Unregister all developer debug properties."""

    # All current properties
    props_to_remove = [
        # Master control
        'dev_debug_master_hz',

        # Engine
        'dev_debug_engine',
        'dev_startup_logs',

        # Spatial grid
        'dev_debug_grid',

        # Offload systems
        'dev_debug_kcc_physics',
        'dev_debug_frame_numbers',
        'dev_debug_camera',
        'dev_debug_culling',

        # Unified physics
        'dev_debug_physics',
        'dev_debug_physics_ground',
        'dev_debug_physics_horizontal',
        'dev_debug_physics_body',
        'dev_debug_physics_ceiling',
        'dev_debug_physics_step',
        'dev_debug_physics_slide',
        'dev_debug_physics_slopes',

        # Dynamic mesh system
        'dev_debug_dynamic_cache',
        'dev_debug_dynamic_opt',

        # Visual debug
        'dev_debug_kcc_visual',
        'dev_debug_kcc_visual_capsule',
        'dev_debug_kcc_visual_normals',
        'dev_debug_kcc_visual_ground',
        'dev_debug_kcc_visual_movement',
        'dev_debug_kcc_visual_line_width',
        'dev_debug_kcc_visual_vector_scale',

        # Game systems
        'dev_debug_interactions',
        'dev_debug_audio',
        'dev_debug_animations',

        # Session export
        'dev_export_session_log',
    ]

    for prop in props_to_remove:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    # Clean up OLD properties from previous versions (migration)
    old_props = [
        # Old offload naming
        'dev_debug_kcc_offload',
        'dev_debug_camera_offload',
        'dev_debug_performance',

        # Old dynamic mesh properties (now unified)
        'dev_debug_dynamic_mesh',
        'dev_debug_dynamic_collision',
        'dev_debug_dynamic_body_ray',
        'dev_debug_dynamic_horizontal',

        # Old physics properties
        'dev_debug_unified_physics',
        'dev_debug_physics_capsule',
        'dev_debug_physics_body_integrity',
        'dev_debug_physics_step_up',

        # Old Hz properties
        'dev_debug_engine_hz',
        'dev_debug_kcc_offload_hz',
        'dev_debug_frame_numbers_hz',
        'dev_debug_camera_offload_hz',
        'dev_debug_performance_hz',
        'dev_debug_dynamic_offload_hz',
        'dev_debug_physics_capsule_hz',
        'dev_debug_physics_ground_hz',
        'dev_debug_physics_step_up_hz',
        'dev_debug_physics_slopes_hz',
        'dev_debug_physics_slide_hz',
        'dev_debug_interactions_hz',
        'dev_debug_audio_hz',
        'dev_debug_animations_hz',

        # Other old properties
        'dev_debug_forward_sweep',
        'dev_debug_forward_sweep_hz',
        'dev_debug_raycast_offload',
        'dev_debug_raycast_offload_hz',
        'dev_debug_physics',
        'dev_debug_physics_hz',
        'dev_run_sync_test',
    ]

    for prop in old_props:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)