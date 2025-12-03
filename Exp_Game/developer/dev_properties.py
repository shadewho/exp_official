# Exp_Game/developer/dev_properties.py
"""
Developer debug properties - toggleable console output by category.

All properties are stored on bpy.types.Scene for easy access from operators.
Default: All debug flags are FALSE (silent console).

Categories:
  - Engine Health: Core multiprocessing engine status
  - Offload Systems: Worker-bound computation (KCC, camera, culling, dynamic)
  - Game Systems: Main thread systems (interactions, audio, animations)
"""

import bpy


def register_properties():
    """Register all developer debug properties on Scene."""

    # ══════════════════════════════════════════════════════════════════════════
    # ENGINE HEALTH
    # Core multiprocessing engine diagnostics
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_engine = bpy.props.BoolProperty(
        name="Engine Health",
        description=(
            "Core multiprocessing engine diagnostics:\n"
            "• Worker startup/shutdown status\n"
            "• Jobs/sec throughput (at 1Hz: rolling average)\n"
            "• Queue depth and saturation warnings\n"
            "• Heartbeat monitoring\n"
            "• Result processing latency"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_engine_hz = bpy.props.IntProperty(
        name="Engine Hz",
        description=(
            "Debug output frequency.\n"
            "30 = every frame (verbose)\n"
            "1 = once per second (summary stats)"
        ),
        default=1,  # Default to summary mode
        min=1,
        max=30
    )

    # ══════════════════════════════════════════════════════════════════════════
    # OFFLOAD SYSTEMS
    # Worker-bound computation offloaded from main thread
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_kcc_offload = bpy.props.BoolProperty(
        name="KCC Physics",
        description=(
            "Full physics step offload (KCC_PHYSICS_STEP):\n"
            "• SUBMIT: pos, vel, input, jump state\n"
            "• RESULT: new pos, ground state, collision flags\n"
            "• Timing: worker calc time (µs)\n"
            "• Stats: rays cast, triangles tested\n"
            "• Events: BLOCKED, STEP, SLIDE, CEILING\n"
            "\n"
            "At 1Hz: Summary with avg timing + event counts"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_kcc_offload_hz = bpy.props.IntProperty(
        name="KCC Hz",
        description=(
            "Debug output frequency.\n"
            "30 = every frame (per-step details)\n"
            "1 = once per second (summary stats)"
        ),
        default=1,  # Default to summary mode
        min=1,
        max=30
    )

    # ──────────────────────────────────────────────────────────────────────────
    # PHYSICS DIAGNOSTICS (Multiple toggles for deep physics debugging)
    # ──────────────────────────────────────────────────────────────────────────

    bpy.types.Scene.dev_debug_physics_timing = bpy.props.BoolProperty(
        name="Physics Timing",
        description=(
            "Detailed physics timing breakdown:\n"
            "• Total step time (ms)\n"
            "• Worker computation time (µs)\n"
            "• Poll time (µs)\n"
            "• Timeout occurrences\n"
            "• Poll attempt count\n"
            "• Same-frame vs timeout ratio\n"
            "\n"
            "Helps identify if worker is slow or polling is taking too long"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_timing_hz = bpy.props.IntProperty(
        name="Timing Hz",
        description=(
            "Output frequency:\n"
            "30 = every physics step\n"
            "1 = summary per second"
        ),
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_physics_catchup = bpy.props.BoolProperty(
        name="Physics Catchup",
        description=(
            "Track frame catchup (multiple physics steps per frame):\n"
            "• When 2+ physics steps run in one timer tick\n"
            "• Catchup frequency\n"
            "• Max catchup steps observed\n"
            "• Impact on frame timing\n"
            "\n"
            "Frequent catchup (>10%) indicates modal timer drift"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_catchup_hz = bpy.props.IntProperty(
        name="Catchup Hz",
        description="Output frequency (1-30 Hz)",
        default=1,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_physics_platform = bpy.props.BoolProperty(
        name="Platform Carry",
        description=(
            "Platform carry application timing:\n"
            "• When platform motion applied\n"
            "• Platform velocity magnitude\n"
            "• Position delta from platform\n"
            "• Rotation carry (Z-axis)\n"
            "\n"
            "Shows if platform carry is causing stutter"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_platform_hz = bpy.props.IntProperty(
        name="Platform Hz",
        description="Output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_physics_consistency = bpy.props.BoolProperty(
        name="Physics Consistency",
        description=(
            "Frame-to-frame consistency tracking:\n"
            "• Step time variance (ms)\n"
            "• Position delta consistency\n"
            "• Velocity consistency\n"
            "• Gravity application timing\n"
            "• Ground state flipping\n"
            "\n"
            "Detects stuttering and timing irregularities"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_consistency_hz = bpy.props.IntProperty(
        name="Consistency Hz",
        description="Output frequency (1-30 Hz)",
        default=1,  # Summary mode by default
        min=1,
        max=30
    )

    # ──────────────────────────────────────────────────────────────────────────
    # GRANULAR PHYSICS DIAGNOSTICS (Subsystem-specific debugging)
    # ──────────────────────────────────────────────────────────────────────────

    bpy.types.Scene.dev_debug_physics_capsule = bpy.props.BoolProperty(
        name="Capsule Collision",
        description=(
            "Capsule sweep and collision testing:\n"
            "• Horizontal/vertical sweep distances\n"
            "• Hit detection (feet/head sphere)\n"
            "• Collision normals and hit points\n"
            "• Triangles tested per sweep\n"
            "• Early vs full sweep results\n"
            "\n"
            "Use to diagnose: Phasing through walls, collision not working"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_capsule_hz = bpy.props.IntProperty(
        name="Capsule Hz",
        description="Output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_physics_depenetration = bpy.props.BoolProperty(
        name="Depenetration",
        description=(
            "Stuck/overlap detection and push-out:\n"
            "• Overlap detection (feet/head spheres)\n"
            "• Push-out vectors and distances\n"
            "• Iterations needed to escape\n"
            "• Was stuck flag\n"
            "• Final corrected position\n"
            "\n"
            "Use to diagnose: Getting stuck in geometry, jittery movement"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_depenetration_hz = bpy.props.IntProperty(
        name="Depenetration Hz",
        description="Output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_physics_ground = bpy.props.BoolProperty(
        name="Ground Detection",
        description=(
            "Ground sphere cast and snap logic:\n"
            "• Raycast down distance and results\n"
            "• Ground Z position and normal\n"
            "• Snap distance from character to ground\n"
            "• On ground vs airborne state\n"
            "• Walkable vs too steep detection\n"
            "\n"
            "Use to diagnose: Floating, not landing, falling through floor"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_ground_hz = bpy.props.IntProperty(
        name="Ground Hz",
        description="Output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_physics_step_up = bpy.props.BoolProperty(
        name="Step-Up",
        description=(
            "Step-up sequence and stair climbing:\n"
            "• Step-up attempts (triggered vs skipped)\n"
            "• Headroom check results\n"
            "• Forward sweep at raised height\n"
            "• Drop-down landing validation\n"
            "• Final step success/failure\n"
            "\n"
            "Use to diagnose: Can't climb stairs, stuck on small obstacles"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_step_up_hz = bpy.props.IntProperty(
        name="Step-Up Hz",
        description="Output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_physics_slopes = bpy.props.BoolProperty(
        name="Slopes",
        description=(
            "Slope handling (uphill/downhill):\n"
            "• Slope steepness (degrees)\n"
            "• Uphill velocity blocking\n"
            "• Downhill slide acceleration\n"
            "• Walkable vs too steep determination\n"
            "• Ground normal orientation\n"
            "\n"
            "Use to diagnose: Sliding down slopes, can't walk uphill"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_slopes_hz = bpy.props.IntProperty(
        name="Slopes Hz",
        description="Output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_physics_slide = bpy.props.BoolProperty(
        name="Wall Slide",
        description=(
            "Multi-plane slide algorithm:\n"
            "• Hit normals (1st, 2nd plane)\n"
            "• Slide direction calculation\n"
            "• Edge/corner detection (3+ planes)\n"
            "• Remaining movement after slide\n"
            "• Blocked flag\n"
            "\n"
            "Use to diagnose: Stuck in corners, weird sliding behavior"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_slide_hz = bpy.props.IntProperty(
        name="Slide Hz",
        description="Output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_physics_vertical = bpy.props.BoolProperty(
        name="Vertical Movement",
        description=(
            "Jumping, gravity, and ceiling hits:\n"
            "• Gravity accumulation per frame\n"
            "• Jump execution (buffered vs instant)\n"
            "• Ceiling collision detection\n"
            "• Vertical velocity changes\n"
            "• Coyote time remaining\n"
            "\n"
            "Use to diagnose: Jump not working, bonking head on ceiling"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_vertical_hz = bpy.props.IntProperty(
        name="Vertical Hz",
        description="Output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    # ──────────────────────────────────────────────────────────────────────────
    # ENHANCED PHYSICS DIAGNOSTICS (Detailed failure analysis & geometry info)
    # ──────────────────────────────────────────────────────────────────────────

    bpy.types.Scene.dev_debug_physics_enhanced = bpy.props.BoolProperty(
        name="Enhanced Diagnostics",
        description=(
            "Detailed physics diagnostics for deep debugging:\n"
            "• Step-up failure reasons (no_landing, forward_blocked, too_steep)\n"
            "• Hit geometry details (triangle indices, normals, distances)\n"
            "• Movement delta (requested vs actual movement)\n"
            "• Blocked percentage (how much movement was stopped)\n"
            "• Proximity to nearest geometry\n"
            "• Collision quality metrics\n"
            "\n"
            "Use when: Debugging specific physics bugs, need exact failure reasons"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_enhanced_hz = bpy.props.IntProperty(
        name="Enhanced Hz",
        description="Output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    # ──────────────────────────────────────────────────────────────────────────
    # KCC VISUAL DEBUG (3D Viewport Overlay Visualization)
    # ──────────────────────────────────────────────────────────────────────────

    bpy.types.Scene.dev_debug_kcc_visual = bpy.props.BoolProperty(
        name="KCC Visual Debug",
        description=(
            "Real-time 3D visualization of KCC physics:\n"
            "• Capsule collision shape (two spheres)\n"
            "• Hit normals (cyan arrows showing collision surfaces)\n"
            "• Ground detection ray (magenta = hit, purple = miss)\n"
            "• Movement vectors (green = intended, red = actual)\n"
            "\n"
            "Colors:\n"
            "• Green capsule = grounded\n"
            "• Yellow capsule = colliding\n"
            "• Red capsule = stuck (depenetrating)\n"
            "• Blue capsule = airborne\n"
            "\n"
            "Use individual toggles below to show/hide specific elements"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_kcc_visual_capsule = bpy.props.BoolProperty(
        name="Show Capsule",
        description="Draw capsule collision shape (feet + head spheres)",
        default=True
    )

    bpy.types.Scene.dev_debug_kcc_visual_normals = bpy.props.BoolProperty(
        name="Show Hit Normals",
        description="Draw collision surface normals as cyan arrows",
        default=True
    )

    bpy.types.Scene.dev_debug_kcc_visual_ground = bpy.props.BoolProperty(
        name="Show Ground Ray",
        description="Draw ground detection raycast (magenta = hit, purple = miss)",
        default=True
    )

    bpy.types.Scene.dev_debug_kcc_visual_movement = bpy.props.BoolProperty(
        name="Show Movement Vectors",
        description="Draw movement vectors (green = intended, red = actual)",
        default=True
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SESSION LOG EXPORT
    # ──────────────────────────────────────────────────────────────────────────

    bpy.types.Scene.dev_export_session_log = bpy.props.BoolProperty(
        name="Export Session Log",
        description=(
            "Export console output to text file when game ends:\n"
            "• Location: Desktop/engine_output_files/\n"
            "• Files: kcc_latest.txt (current), kcc_previous.txt (backup)\n"
            "• Contains: All console debug output from session\n"
            "• Safe: Atomic writes, no data loss on failure\n"
            "\n"
            "Use this to share debug logs with Claude for analysis"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_camera_offload = bpy.props.BoolProperty(
        name="Camera Occlusion",
        description=(
            "Camera occlusion raycast offload:\n"
            "• Ray submission (origin, target)\n"
            "• HIT/MISS results with distance\n"
            "• Static vs dynamic geometry hits\n"
            "• Worker timing (µs)"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_camera_offload_hz = bpy.props.IntProperty(
        name="Camera Hz",
        description="Debug output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_performance = bpy.props.BoolProperty(
        name="Performance Culling",
        description=(
            "Distance-based object culling (CULL_BATCH):\n"
            "• Batch submissions with object counts\n"
            "• Hide/show operations per frame\n"
            "• Placeholder mesh swapping\n"
            "• Culling radius thresholds"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_performance_hz = bpy.props.IntProperty(
        name="Culling Hz",
        description="Debug output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_dynamic_offload = bpy.props.BoolProperty(
        name="Dynamic Mesh",
        description=(
            "Dynamic mesh activation offload:\n"
            "• Distance-based activation checks\n"
            "• BVH rebuild triggers\n"
            "• Platform velocity tracking\n"
            "• Activation state transitions"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_dynamic_offload_hz = bpy.props.IntProperty(
        name="Dynamic Hz",
        description="Debug output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    # ══════════════════════════════════════════════════════════════════════════
    # GAME SYSTEMS
    # Main thread game logic (not offloaded)
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_interactions = bpy.props.BoolProperty(
        name="Interactions",
        description=(
            "Interaction and reaction system:\n"
            "• Trigger activations (proximity, collision, etc.)\n"
            "• Reaction execution\n"
            "• Task queue updates\n"
            "• Objective progress"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_interactions_hz = bpy.props.IntProperty(
        name="Interactions Hz",
        description="Debug output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_audio = bpy.props.BoolProperty(
        name="Audio",
        description=(
            "Audio system:\n"
            "• Sound playback start/stop\n"
            "• Audio state transitions\n"
            "• Volume adjustments\n"
            "• Cleanup operations"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_audio_hz = bpy.props.IntProperty(
        name="Audio Hz",
        description="Debug output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_animations = bpy.props.BoolProperty(
        name="Animations",
        description=(
            "Animation and NLA system:\n"
            "• State machine transitions\n"
            "• NLA strip playback\n"
            "• Custom animation updates\n"
            "• Blend timing"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_animations_hz = bpy.props.IntProperty(
        name="Animations Hz",
        description="Debug output frequency (1-30 Hz)",
        default=5,
        min=1,
        max=30
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ENGINE STRESS TESTS - REMOVED (Property no longer needed)
    # ══════════════════════════════════════════════════════════════════════════
    # Stress tests are now standalone operators (Quick Test / Full Stress Test)
    # in the Developer Tools panel under "Engine Health" → "Manual Stress Tests".
    #
    # PURPOSE: Test the ENGINE CORE in isolation - worker performance, job
    # throughput, queue handling, and readiness checks.
    #
    # NOT for testing engine+modal integration - that's a separate concern.
    # Engine stress tests power up the engine and measure its raw capabilities.
    #
    # Old dev_run_sync_test property has been removed - use operator buttons instead.

    # ══════════════════════════════════════════════════════════════════════════
    # STARTUP LOGS
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_startup_logs = bpy.props.BoolProperty(
        name="Startup Logs",
        description=(
            "Show detailed engine and modal startup sequence.\n\n"
            "Logs:\n"
            "• Engine worker spawning\n"
            "• PING verification\n"
            "• Grid cache confirmation\n"
            "• Health checks\n"
            "• Modal initialization steps\n"
            "• Lock-step synchronization\n"
            "• READY confirmation or failure point\n\n"
            "Use this to debug startup issues and verify engine readiness"
        ),
        default=False
    )


def unregister_properties():
    """Unregister all developer debug properties."""

    # Startup logs
    if hasattr(bpy.types.Scene, 'dev_startup_logs'):
        del bpy.types.Scene.dev_startup_logs

    # Stress tests (property removed - now use operator buttons instead)

    # Engine Health
    for prop in ('dev_debug_engine', 'dev_debug_engine_hz'):
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    # Offload Systems
    for prop in ('dev_debug_kcc_offload', 'dev_debug_kcc_offload_hz',
                 'dev_debug_camera_offload', 'dev_debug_camera_offload_hz',
                 'dev_debug_performance', 'dev_debug_performance_hz',
                 'dev_debug_dynamic_offload', 'dev_debug_dynamic_offload_hz'):
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    # Physics Diagnostics
    for prop in ('dev_debug_physics_timing', 'dev_debug_physics_timing_hz',
                 'dev_debug_physics_catchup', 'dev_debug_physics_catchup_hz',
                 'dev_debug_physics_platform', 'dev_debug_physics_platform_hz',
                 'dev_debug_physics_consistency', 'dev_debug_physics_consistency_hz'):
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    # Granular Physics Diagnostics
    for prop in ('dev_debug_physics_capsule', 'dev_debug_physics_capsule_hz',
                 'dev_debug_physics_depenetration', 'dev_debug_physics_depenetration_hz',
                 'dev_debug_physics_ground', 'dev_debug_physics_ground_hz',
                 'dev_debug_physics_step_up', 'dev_debug_physics_step_up_hz',
                 'dev_debug_physics_slopes', 'dev_debug_physics_slopes_hz',
                 'dev_debug_physics_slide', 'dev_debug_physics_slide_hz',
                 'dev_debug_physics_vertical', 'dev_debug_physics_vertical_hz'):
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    # KCC Visual Debug
    for prop in ('dev_debug_kcc_visual', 'dev_debug_kcc_visual_capsule',
                 'dev_debug_kcc_visual_normals', 'dev_debug_kcc_visual_ground',
                 'dev_debug_kcc_visual_movement'):
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    # Session Log Export
    if hasattr(bpy.types.Scene, 'dev_export_session_log'):
        delattr(bpy.types.Scene, 'dev_export_session_log')

    # Game Systems
    for prop in ('dev_debug_interactions', 'dev_debug_interactions_hz',
                 'dev_debug_audio', 'dev_debug_audio_hz',
                 'dev_debug_animations', 'dev_debug_animations_hz'):
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    # Clean up removed properties from old versions
    for old_prop in ('dev_debug_forward_sweep', 'dev_debug_forward_sweep_hz',
                     'dev_debug_raycast_offload', 'dev_debug_raycast_offload_hz',
                     'dev_debug_physics', 'dev_debug_physics_hz'):
        if hasattr(bpy.types.Scene, old_prop):
            delattr(bpy.types.Scene, old_prop)
