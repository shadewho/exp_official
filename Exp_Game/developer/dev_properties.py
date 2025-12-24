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
    # IK VISUAL DEBUG (3D Viewport Overlay)
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_ik_visual = bpy.props.BoolProperty(
        name="Enable IK Visual Debug",
        description=(
            "Real-time 3D visualization of IK system:\n"
            "• IK targets (spheres at goal positions)\n"
            "• Bone chains (lines from root to tip)\n"
            "• Reach limits (transparent spheres)\n"
            "• Pole vectors (arrows showing bend direction)\n"
            "• Joint positions (knee/elbow markers)"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_ik_visual_targets = bpy.props.BoolProperty(
        name="IK Targets",
        description="Draw IK target positions as spheres (green = reachable, red = out of reach)",
        default=True
    )

    bpy.types.Scene.dev_debug_ik_visual_chains = bpy.props.BoolProperty(
        name="Bone Chains",
        description="Draw bone chains as lines (cyan = upper bone, magenta = lower bone)",
        default=True
    )

    bpy.types.Scene.dev_debug_ik_visual_reach = bpy.props.BoolProperty(
        name="Reach Limits",
        description="Draw maximum reach sphere from root joint (transparent yellow)",
        default=True
    )

    bpy.types.Scene.dev_debug_ik_visual_poles = bpy.props.BoolProperty(
        name="Pole Vectors",
        description="Draw pole direction arrows (orange = bend direction hint)",
        default=True
    )

    bpy.types.Scene.dev_debug_ik_visual_joints = bpy.props.BoolProperty(
        name="Joint Markers",
        description="Draw markers at joint positions (root, mid, tip)",
        default=True
    )

    bpy.types.Scene.dev_debug_ik_line_width = bpy.props.FloatProperty(
        name="Line Width",
        description="IK visual debug line thickness (1.0-10.0)",
        default=2.5,
        min=1.0,
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
            "• IK chains, targets, poles, reach\n"
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
            ("ARM_L_IK", "Left Arm IK", "Highlight left arm IK chain"),
            ("ARM_R_IK", "Right Arm IK", "Highlight right arm IK chain"),
            ("LEG_L", "Left Leg", "Highlight left leg bones"),
            ("LEG_R", "Right Leg", "Highlight right leg bones"),
            ("LEG_L_IK", "Left Leg IK", "Highlight left leg IK chain"),
            ("LEG_R_IK", "Right Leg IK", "Highlight right leg IK chain"),
            ("FINGERS", "Fingers", "Highlight finger bones"),
            ("ROOT", "Root (Hips)", "Highlight root/hips bone"),
        ],
        default="ALL"
    )

    bpy.types.Scene.dev_rig_vis_ik_chains = bpy.props.BoolProperty(
        name="IK Chains",
        description="Draw IK chain bones (cyan = upper, magenta = lower)",
        default=True
    )

    bpy.types.Scene.dev_rig_vis_ik_targets = bpy.props.BoolProperty(
        name="IK Targets",
        description="Draw IK target spheres (green = reachable, red = out of reach)",
        default=True
    )

    bpy.types.Scene.dev_rig_vis_ik_poles = bpy.props.BoolProperty(
        name="IK Poles",
        description="Draw pole vector arrows (orange = bend direction)",
        default=True
    )

    bpy.types.Scene.dev_rig_vis_ik_reach = bpy.props.BoolProperty(
        name="IK Reach",
        description="Draw maximum reach spheres from root joints",
        default=False
    )

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
        description="Show animation state text above character (IK, layers, etc.)",
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

    bpy.types.Scene.dev_debug_anim_worker = bpy.props.BoolProperty(
        name="Animation Worker",
        description=(
            "Animation worker diagnostics:\n"
            "• Shows designated animation worker ID\n"
            "• Cache transfer to single worker (saves memory)\n"
            "• Job routing to animation worker\n"
            "• Compute times on animation worker"
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

    # ══════════════════════════════════════════════════════════════════════════
    # RUNTIME IK SYSTEM
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_runtime_ik = bpy.props.BoolProperty(
        name="Runtime IK",
        description=(
            "Runtime IK system diagnostics:\n"
            "• IK job submission (target, chain, influence)\n"
            "• Worker solve time (µs)\n"
            "• Joint positions computed\n"
            "• Blend with animation"
        ),
        default=False
    )

    bpy.types.Scene.runtime_ik_enabled = bpy.props.BoolProperty(
        name="Enable Runtime IK",
        description="Enable real-time IK solving during gameplay",
        default=False
    )

    bpy.types.Scene.runtime_ik_use_blend_system = bpy.props.BoolProperty(
        name="Use BlendSystem",
        description=(
            "Route IK through BlendSystem (production path).\n"
            "When enabled, sets IK target via blend_sys.set_ik_target_object().\n"
            "Logs will show 'src=BlendSystem' instead of 'src=SceneProps'.\n"
            "Use this to test the same code path that reactions will use."
        ),
        default=False
    )

    # NOTE: runtime_ik_chain, runtime_ik_influence, runtime_ik_target removed
    # Runtime IK now uses unified test_ik_* properties from Test Suite

    # ══════════════════════════════════════════════════════════════════════════
    # POSE LIBRARY SYSTEM
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_poses = bpy.props.BoolProperty(
        name="Pose Library",
        description=(
            "Pose library system diagnostics:\n"
            "• Pose capture events\n"
            "• Pose cache transfer to workers\n"
            "• Pose playback and blending\n"
            "• Worker cache confirmation"
        ),
        default=False
    )

    # --- Pose Test Properties (Animation 2.0 Dev Panel) ---

    def _get_pose_items(self, context):
        """Dynamic enum callback for pose selection dropdown."""
        items = []
        if context and hasattr(context, 'scene') and hasattr(context.scene, 'pose_library'):
            for i, pose in enumerate(context.scene.pose_library):
                items.append((pose.name, pose.name, pose.description or "", 'ARMATURE_DATA', i))
        # Always need at least one item for Blender enums
        if not items:
            items.append(("NONE", "No Poses", "Capture a pose first", 'ERROR', 0))
        return items

    bpy.types.Scene.pose_test_name = bpy.props.EnumProperty(
        name="Test Pose",
        description="Pose to play for testing",
        items=_get_pose_items
    )

    bpy.types.Scene.pose_test_blend_time = bpy.props.FloatProperty(
        name="Blend Time",
        description="Time to blend into/out of the pose",
        default=0.25,
        min=0.0,
        max=2.0,
        unit='TIME'
    )

    # NOTE: pose_test_bone_group removed - now using unified test_bone_group

    # ══════════════════════════════════════════════════════════════════════════
    # UNIFIED ANIMATION TEST SUITE
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.test_mode = bpy.props.EnumProperty(
        name="Test Mode",
        description="Select which animation system to test",
        items=[
            ("ANIMATION", "Animation", "Test animation playback and blending", 'PLAY', 0),
            ("POSE", "Pose", "Test pose library application", 'ARMATURE_DATA', 1),
            ("IK", "IK", "Test IK chain solving", 'CON_KINEMATIC', 2),
        ],
        default="ANIMATION"
    )

    bpy.types.Scene.test_bone_group = bpy.props.EnumProperty(
        name="Bone Group",
        description="Which bones to affect (used by Pose and IK modes)",
        items=[
            ("ALL", "Full Body", "Apply to all bones"),
            ("UPPER_BODY", "Upper Body", "Apply to spine, arms, head"),
            ("LOWER_BODY", "Lower Body", "Apply to hips and legs"),
            ("ARM_L", "Left Arm", "Apply to left arm only"),
            ("ARM_R", "Right Arm", "Apply to right arm only"),
            ("ARMS", "Both Arms", "Apply to both arms"),
            ("LEG_L", "Left Leg", "Apply to left leg only"),
            ("LEG_R", "Right Leg", "Apply to right leg only"),
            ("LEGS", "Both Legs", "Apply to both legs"),
            ("SPINE_HEAD", "Spine & Head", "Apply to spine, neck, and head"),
        ],
        default="ALL"
    )

    # --- Animation Mode Options ---
    # Note: Uses existing props from ANIM2_TestProperties in test_panel.py:
    # - props.selected_animation, props.blend_animation, props.blend_weight
    # - props.play_speed, props.fade_time, props.loop_playback

    bpy.types.Scene.test_blend_enabled = bpy.props.BoolProperty(
        name="Blend Secondary",
        description="Enable blending a second animation",
        default=False
    )

    # --- Pose Mode Options ---
    # Note: Uses existing pose_test_name, pose_test_blend_time from above

    # --- IK Mode Options ---
    bpy.types.Scene.test_ik_chain = bpy.props.EnumProperty(
        name="IK Chain",
        description="Which limb chain to solve IK for",
        items=[
            ("leg_L", "Left Leg", "Left leg IK (thigh → shin → foot)"),
            ("leg_R", "Right Leg", "Right leg IK (thigh → shin → foot)"),
            ("arm_L", "Left Arm", "Left arm IK (upper → forearm → hand)"),
            ("arm_R", "Right Arm", "Right arm IK (upper → forearm → hand)"),
        ],
        default="leg_L"
    )

    bpy.types.Scene.test_ik_target = bpy.props.PointerProperty(
        name="IK Target",
        description="Object to use as IK target (Empty, etc.)",
        type=bpy.types.Object
    )

    bpy.types.Scene.test_ik_influence = bpy.props.FloatProperty(
        name="IK Influence",
        description="How much IK affects the final pose (0 = none, 1 = full)",
        default=1.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR'
    )

    bpy.types.Scene.test_ik_pole = bpy.props.EnumProperty(
        name="Pole Direction",
        description="Hint for knee/elbow bend direction",
        items=[
            ("AUTO", "Auto", "Automatically determine from rest pose"),
            ("FORWARD", "Forward (+Y)", "Bend forward"),
            ("BACK", "Back (-Y)", "Bend backward"),
            ("LEFT", "Left (-X)", "Bend left"),
            ("RIGHT", "Right (+X)", "Bend right"),
            ("UP", "Up (+Z)", "Bend upward"),
            ("DOWN", "Down (-Z)", "Bend downward"),
        ],
        default="AUTO"
    )

    bpy.types.Scene.test_ik_advanced = bpy.props.BoolProperty(
        name="Advanced Mode",
        description="Show advanced IK controls (XYZ offset, pole offset)",
        default=False
    )

    bpy.types.Scene.test_ik_target_xyz = bpy.props.FloatVectorProperty(
        name="Target Offset",
        description="Manual XYZ offset from rest position (advanced mode)",
        default=(0.0, 0.0, 0.0),
        subtype='XYZ',
        unit='LENGTH'
    )

    bpy.types.Scene.test_ik_pole_offset = bpy.props.FloatProperty(
        name="Pole Offset",
        description="Distance of pole from joint midpoint",
        default=0.5,
        min=0.1,
        max=2.0,
        unit='LENGTH'
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

        # IK visual debug
        'dev_debug_ik_visual',
        'dev_debug_ik_visual_targets',
        'dev_debug_ik_visual_chains',
        'dev_debug_ik_visual_reach',
        'dev_debug_ik_visual_poles',
        'dev_debug_ik_visual_joints',
        'dev_debug_ik_line_width',

        # Rig visualizer
        'dev_rig_visualizer_enabled',
        'dev_rig_vis_bone_groups',
        'dev_rig_vis_selected_group',
        'dev_rig_vis_ik_chains',
        'dev_rig_vis_ik_targets',
        'dev_rig_vis_ik_poles',
        'dev_rig_vis_ik_reach',
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
        'dev_debug_anim_worker',
        'dev_debug_projectiles',
        'dev_debug_hitscans',

        # Runtime IK
        'dev_debug_runtime_ik',
        'runtime_ik_enabled',
        'runtime_ik_use_blend_system',
        # runtime_ik_chain, runtime_ik_influence, runtime_ik_target moved to old_props

        # Pose Library
        'dev_debug_poses',
        'pose_test_name',
        'pose_test_blend_time',
        # pose_test_bone_group moved to old_props (replaced by test_bone_group)

        # Unified Test Suite
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

        # Session export
        'dev_export_session_log',
    ]

    for prop in props_to_remove:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    # Clean up OLD properties from previous versions (migration)
    old_props = [
        # Replaced by unified test_bone_group
        'pose_test_bone_group',
        # Removed - IK Test section merged into unified Test Suite
        'dev_section_ik',
        # Replaced by unified test_ik_* properties
        'runtime_ik_chain',
        'runtime_ik_influence',
        'runtime_ik_target',
    ]

    for prop in old_props:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)