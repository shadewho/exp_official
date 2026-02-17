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

# ══════════════════════════════════════════════════════════════════════════════
# SHIP FLAG — flip to False before distributing the addon
# ══════════════════════════════════════════════════════════════════════════════
# When False:
#   - Developer Tools panel is completely hidden (poll returns False)
#   - All logging disabled (log_game calls become no-ops)
#   - All debug visualizations disabled
#   - Zero performance overhead
#   - No dev UI visible to end users
# ══════════════════════════════════════════════════════════════════════════════
SHOW_DEV_PANEL = True

# DEV_MODE mirrors SHOW_DEV_PANEL — single source of truth.
# Flip SHOW_DEV_PANEL to False before shipping; everything else follows.
DEV_MODE = SHOW_DEV_PANEL


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
    # PANEL SECTION COLLAPSE STATES
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_section_engine = bpy.props.BoolProperty(
        name="Engine Health", default=False)
    bpy.types.Scene.dev_section_grid = bpy.props.BoolProperty(
        name="Spatial Grid", default=False)
    bpy.types.Scene.dev_section_offload = bpy.props.BoolProperty(
        name="Offload Systems", default=False)
    bpy.types.Scene.dev_section_physics = bpy.props.BoolProperty(
        name="Unified Physics", default=False)
    bpy.types.Scene.dev_section_camera = bpy.props.BoolProperty(
        name="Unified Camera", default=False)
    bpy.types.Scene.dev_section_dynamic = bpy.props.BoolProperty(
        name="Dynamic Mesh System", default=False)
    bpy.types.Scene.dev_section_kcc_visual = bpy.props.BoolProperty(
        name="KCC Visual Debug", default=False)
    bpy.types.Scene.dev_section_game = bpy.props.BoolProperty(
        name="Game Systems", default=False)
    bpy.types.Scene.dev_section_anim = bpy.props.BoolProperty(
        name="Animation 2.0", default=False)

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
    # RIG VISUALIZER (Comprehensive animation system debug)
    # ══════════════════════════════════════════════════════════════════════════

    def _rig_vis_update(self, context):
        """Toggle callback for rig visualizer."""
        from .rig_visualizer import enable_rig_visualizer, disable_rig_visualizer
        if self.dev_rig_visualizer_enabled:
            enable_rig_visualizer()
        else:
            disable_rig_visualizer()

    bpy.types.Scene.dev_rig_visualizer_enabled = bpy.props.BoolProperty(
        name="Enable Rig Visualizer",
        description=(
            "Comprehensive rig and animation system visualization:\n"
            "• Bone groups (colored by membership)\n"
            "• Active blend masks\n"
            "• Bone local axes\n"
            "\n"
            "Shows in 3D viewport when enabled, regardless of panel selection"
        ),
        default=False,
        update=_rig_vis_update
    )

    bpy.types.Scene.dev_rig_vis_bone_groups = bpy.props.BoolProperty(
        name="Bone Groups",
        description="Color bones by their group membership (UPPER_BODY, ARM_L, etc.)",
        default=True
    )

    bpy.types.Scene.dev_rig_vis_selected_group = bpy.props.EnumProperty(
        name="Highlight Group",
        description="Which bone group to highlight (or ALL to show all groups)",
        items=[
            ("ALL", "All Groups", "Show all bones with group-based colors"),
            ("UPPER_BODY", "Upper Body", "Highlight upper body bones"),
            ("LOWER_BODY", "Lower Body", "Highlight lower body bones"),
            ("SPINE", "Spine", "Highlight spine bones"),
            ("HEAD_NECK", "Head & Neck", "Highlight head and neck bones"),
            ("ARM_L", "Left Arm", "Highlight left arm bones"),
            ("ARM_R", "Right Arm", "Highlight right arm bones"),
            ("LEG_L", "Left Leg", "Highlight left leg bones"),
            ("LEG_R", "Right Leg", "Highlight right leg bones"),
            ("FINGERS", "Fingers", "Highlight finger bones"),
            ("ROOT", "Root (Hips)", "Highlight root/hips bone"),
        ],
        default="ALL"
    )

    # NOTE: IK visualization properties removed (2025) - IK system not used

    bpy.types.Scene.dev_rig_vis_bone_axes = bpy.props.BoolProperty(
        name="Bone Axes",
        description="Draw local X/Y/Z axes for each bone (shows orientation)",
        default=False
    )

    bpy.types.Scene.dev_rig_vis_active_mask = bpy.props.BoolProperty(
        name="Active Mask",
        description="Color bones by active blend layer mask weight (blue = low, red = high)",
        default=False
    )

    # NOTE: dev_rig_vis_full_body_ik removed (2025) - IK system not used

    bpy.types.Scene.dev_rig_vis_line_width = bpy.props.FloatProperty(
        name="Line Width",
        description="Visualizer line thickness",
        default=2.0,
        min=1.0,
        max=10.0
    )

    bpy.types.Scene.dev_rig_vis_axis_length = bpy.props.FloatProperty(
        name="Axis Length",
        description="Length of bone axis lines",
        default=0.05,
        min=0.01,
        max=0.2,
        unit='LENGTH'
    )

    # Text overlay properties
    bpy.types.Scene.dev_rig_vis_text_overlay = bpy.props.BoolProperty(
        name="Text Overlay",
        description="Show animation state text above character",
        default=True
    )

    bpy.types.Scene.dev_rig_vis_text_size = bpy.props.IntProperty(
        name="Text Size",
        description="Font size for text overlay",
        default=14,
        min=10,
        max=24
    )

    bpy.types.Scene.dev_rig_vis_text_background = bpy.props.BoolProperty(
        name="Text Background",
        description="Draw semi-transparent background behind text for readability",
        default=True
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

    bpy.types.Scene.dev_debug_trackers = bpy.props.BoolProperty(
        name="Trackers",
        description=(
            "Tracker system diagnostics (worker-offloaded):\n"
            "• Condition evaluations and results\n"
            "• Fire events with tracker UID\n"
            "• Edge detection transitions\n"
            "• Cooldown and rate limiting"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_world_state = bpy.props.BoolProperty(
        name="World State Filter",
        description=(
            "World state collection optimization (Phase 1.1):\n"
            "• Objects collected vs total scene objects\n"
            "• Tracked object names list\n"
            "• Confirms filtering is active\n"
            "\n"
            "Shows: 'COLLECT filtered=5 total=127 (96% reduction)'"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_aabb_cache = bpy.props.BoolProperty(
        name="AABB Cache",
        description=(
            "AABB cache optimization (Phase 1.2):\n"
            "• Cache hits vs misses per frame\n"
            "• Static objects cached at game start\n"
            "• Dynamic objects recalculated each frame\n"
            "\n"
            "Shows: 'FRAME hits=4 misses=2 rate=67%'"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_animations = bpy.props.BoolProperty(
        name="Animations",
        description=(
            "Animation system diagnostics (worker-offloaded):\n"
            "• Per-frame compute jobs submitted\n"
            "• Blending: animations playing, weights, times\n"
            "• Worker compute time (µs)\n"
            "• Bones updated per frame"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_anim_cache = bpy.props.BoolProperty(
        name="Animation Cache",
        description=(
            "Animation cache diagnostics:\n"
            "• Animations baked and cached in workers\n"
            "• Cache timing and bone channel counts\n"
            "• Worker cache confirmation"
        ),
        default=False
    )

    # NOTE: dev_debug_anim_worker removed (2025-01) - animations computed locally, no worker
    # NOTE: dev_debug_layers removed (2025) - layer system not used

    bpy.types.Scene.dev_debug_startup_diag = bpy.props.BoolProperty(
        name="Startup Diagnostics",
        description=(
            "Cold start diagnostics:\n"
            "• Frame interval tracking (first 100 frames)\n"
            "• Physics steps histogram\n"
            "• Hiccup detection (frames > 50ms)\n"
            "• Good/bad start classification"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_projectiles = bpy.props.BoolProperty(
        name="Projectiles",
        description=(
            "Projectile system diagnostics (worker-offloaded):\n"
            "• Spawn events with velocity and origin\n"
            "• Physics updates (gravity, sweep raycasts)\n"
            "• Impacts and collisions\n"
            "• Lifetime expiry"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_hitscans = bpy.props.BoolProperty(
        name="Hitscans",
        description=(
            "Hitscan system diagnostics (worker-offloaded):\n"
            "• Fire events with origin and direction\n"
            "• Ray results (hit/miss, distance, target)\n"
            "• Batch processing stats"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_transforms = bpy.props.BoolProperty(
        name="Transforms",
        description=(
            "Transform reaction diagnostics (worker-offloaded):\n"
            "• Batch submission (count, objects)\n"
            "• Worker calc time (µs)\n"
            "• Rotation modes (euler, quat, local_delta)\n"
            "• Apply results timing"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_tracking = bpy.props.BoolProperty(
        name="Tracking",
        description=(
            "Track To reaction diagnostics (worker-offloaded):\n"
            "• Active tracks (character vs object)\n"
            "• Batch submission (positions, goals)\n"
            "• Worker raycasts (sweep/slide/gravity)\n"
            "• Apply results (arrived, positions)"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_health = bpy.props.BoolProperty(
        name="Health",
        description=(
            "Health system diagnostics:\n"
            "• Object health enabled/disabled\n"
            "• Health value changes (current, delta)\n"
            "• Per-object health status\n"
            "• Health UI overlay enable/position\n"
            "• Reset/clear events"
        ),
        default=False
    )

    # NOTE: Pose Library System removed (2025) - not used
    # dev_debug_poses moved to old_props for cleanup

    # ══════════════════════════════════════════════════════════════════════════
    # SESSION LOG EXPORT
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_export_session_log = bpy.props.BoolProperty(
        name="Export Diagnostics Log",
        description=(
            "Export game diagnostics to text file when game ends:\n"
            "• Location: ~/Desktop/engine_output_files/diagnostics_latest.txt"
        ),
        default=False
    )


def unregister_properties():
    """Unregister all developer debug properties."""

    # All current properties
    props_to_remove = [
        # Master control
        'dev_debug_master_hz',

        # Section collapse states
        'dev_section_engine',
        'dev_section_grid',
        'dev_section_offload',
        'dev_section_physics',
        'dev_section_camera',
        'dev_section_dynamic',
        'dev_section_kcc_visual',
        'dev_section_game',
        'dev_section_anim',

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

        # Rig visualizer
        'dev_rig_visualizer_enabled',
        'dev_rig_vis_bone_groups',
        'dev_rig_vis_selected_group',
        'dev_rig_vis_bone_axes',
        'dev_rig_vis_active_mask',
        'dev_rig_vis_line_width',
        'dev_rig_vis_axis_length',
        'dev_rig_vis_text_overlay',
        'dev_rig_vis_text_size',
        'dev_rig_vis_text_background',

        # Game systems
        'dev_debug_interactions',
        'dev_debug_audio',
        'dev_debug_trackers',
        'dev_debug_world_state',
        'dev_debug_aabb_cache',
        'dev_debug_animations',
        'dev_debug_anim_cache',
        'dev_debug_startup_diag',
        'dev_debug_projectiles',
        'dev_debug_hitscans',
        'dev_debug_transforms',
        'dev_debug_tracking',
        'dev_debug_health',

        # Session export
        'dev_export_session_log',
    ]

    for prop in props_to_remove:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    # Clean up OLD properties from previous versions (migration)
    # These properties are no longer defined but may exist in saved scenes
    old_props = [
        # Replaced by unified test_bone_group
        'pose_test_bone_group',
        # Removed - IK Test section merged into unified Test Suite
        'dev_section_ik',
        # Old IK properties (replaced by Neural IK)
        'runtime_ik_chain',
        'runtime_ik_influence',
        'runtime_ik_target',
        'dev_debug_full_body_ik',
        'dev_debug_runtime_ik',
        # IK Visual Debug (removed - old limb-based IK)
        'dev_debug_ik_visual',
        'dev_debug_ik_visual_targets',
        'dev_debug_ik_visual_chains',
        'dev_debug_ik_visual_reach',
        'dev_debug_ik_visual_poles',
        'dev_debug_ik_visual_joints',
        'dev_debug_ik_line_width',
        # Rig Visualizer IK properties (removed)
        'dev_rig_vis_ik_chains',
        'dev_rig_vis_ik_targets',
        'dev_rig_vis_ik_poles',
        'dev_rig_vis_ik_reach',
        'dev_rig_vis_full_body_ik',
        # Unified IK System (removed - replaced by Neural IK)
        'dev_debug_ik',
        'dev_debug_ik_solve',
        'dev_debug_rig_state',
        'runtime_ik_enabled',
        'runtime_ik_use_blend_system',
        # Test IK properties (removed)
        'test_mode',
        'test_bone_group',
        'test_blend_enabled',
        'test_ik_chain',
        'test_ik_target',
        'test_ik_influence',
        'test_ik_pole',
        'test_ik_advanced',
        'test_ik_target_xyz',
        'test_ik_pole_offset',
        # Pose Blend Mode (removed)
        'pose_blend_mode',
        'pose_blend_a',
        'pose_blend_b',
        'pose_blend_weight',
        'pose_blend_ik_influence',
        'pose_blend_auto_play',
        'pose_blend_duration',
        'pose_blend_hold_start',
        'pose_blend_hold_end',
        # Multi-Chain IK (removed)
        'pose_blend_ik_arm_L',
        'pose_blend_ik_arm_L_target',
        'pose_blend_ik_arm_R',
        'pose_blend_ik_arm_R_target',
        'pose_blend_ik_leg_L',
        'pose_blend_ik_leg_L_target',
        'pose_blend_ik_leg_R',
        'pose_blend_ik_leg_R_target',
        # Rig Analyzer (removed 2025 - unused)
        'rig_analysis_report',
        # Pose Library debug (removed 2025 - unused)
        'dev_debug_poses',
        # Animation Layers (removed 2025 - layer system not used)
        'dev_debug_layers',
        # Pose blend/test (removed 2025 - IK system not used)
        'dev_debug_pose_blend',
        'pose_test_name',
        'pose_test_blend_time',
        'pose_blend_enabled',
    ]

    for prop in old_props:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)