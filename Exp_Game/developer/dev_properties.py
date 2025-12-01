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
    # ENGINE STRESS TESTS
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_run_sync_test = bpy.props.BoolProperty(
        name="Run Engine Sync Tests",
        description=(
            "Run comprehensive engine stress tests on game start.\n\n"
            "Tests:\n"
            "• BASELINE - Zero-latency capability\n"
            "• LIGHT_LOAD - 5 jobs/frame (normal)\n"
            "• MEDIUM_LOAD - 15 jobs/frame (busy)\n"
            "• HEAVY_LOAD - 30 jobs/frame (worst case)\n"
            "• BURST_TEST - 50-job bursts\n"
            "• CATCHUP_STRESS - Modal inconsistency\n\n"
            "Duration: ~18 seconds\n"
            "Results: Console output on game end"
        ),
        default=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # MASTER TOGGLE
    # ══════════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_all = bpy.props.BoolProperty(
        name="Enable All Debug Output",
        description="Master toggle - enables ALL debug categories at once",
        default=False,
        update=lambda self, context: _update_all_debug_flags(context)
    )


def _update_all_debug_flags(context):
    """When master toggle changes, update all individual flags."""
    scene = context.scene
    enabled = scene.dev_debug_all

    # Engine Health
    scene.dev_debug_engine = enabled

    # Offload Systems
    scene.dev_debug_kcc_offload = enabled
    scene.dev_debug_camera_offload = enabled
    scene.dev_debug_performance = enabled
    scene.dev_debug_dynamic_offload = enabled

    # Game Systems
    scene.dev_debug_interactions = enabled
    scene.dev_debug_audio = enabled
    scene.dev_debug_animations = enabled


def unregister_properties():
    """Unregister all developer debug properties."""

    # Master toggle
    if hasattr(bpy.types.Scene, 'dev_debug_all'):
        del bpy.types.Scene.dev_debug_all

    # Stress tests
    if hasattr(bpy.types.Scene, 'dev_run_sync_test'):
        del bpy.types.Scene.dev_run_sync_test

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
