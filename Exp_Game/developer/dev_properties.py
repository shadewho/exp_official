# Exp_Game/developer/dev_properties.py
"""
Developer debug properties - toggleable console output by category.

All properties are stored on bpy.types.Scene for easy access from operators.
Default: All debug flags are FALSE (silent console).
"""

import bpy


def register_properties():
    """Register all developer debug properties on Scene."""

    # ══════════════════════════════════════════════════════════════════════
    # Console Debug Categories
    # ══════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_debug_engine = bpy.props.BoolProperty(
        name="Engine Debug",
        description=(
            "Print multiprocessing engine debug messages:\n"
            "• Worker startup/shutdown\n"
            "• Job submissions and completions\n"
            "• Heartbeat monitoring\n"
            "• Queue status"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_engine_hz = bpy.props.IntProperty(
        name="Engine Debug Hz",
        description="Debug output frequency (1-30 Hz). 30 = every frame, 1 = once per second",
        default=30,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_performance = bpy.props.BoolProperty(
        name="Performance Debug",
        description=(
            "Print performance culling debug messages:\n"
            "• Distance calculations\n"
            "• Object hide/show operations\n"
            "• Culling batch processing\n"
            "• Cache updates"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_performance_hz = bpy.props.IntProperty(
        name="Performance Debug Hz",
        description="Debug output frequency (1-30 Hz). 30 = every frame, 1 = once per second",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_physics = bpy.props.BoolProperty(
        name="Physics Debug",
        description=(
            "Print physics and character controller debug:\n"
            "• Character movement\n"
            "• Ground detection\n"
            "• Collision responses\n"
            "• Platform riding"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_physics_hz = bpy.props.IntProperty(
        name="Physics Debug Hz",
        description="Debug output frequency (1-30 Hz). 30 = every frame, 1 = once per second",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_interactions = bpy.props.BoolProperty(
        name="Interactions Debug",
        description=(
            "Print interaction system debug:\n"
            "• Trigger activations\n"
            "• Reaction execution\n"
            "• Task updates\n"
            "• Objective progress"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_interactions_hz = bpy.props.IntProperty(
        name="Interactions Debug Hz",
        description="Debug output frequency (1-30 Hz). 30 = every frame, 1 = once per second",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_audio = bpy.props.BoolProperty(
        name="Audio Debug",
        description=(
            "Print audio system debug:\n"
            "• Sound playback\n"
            "• Audio state changes\n"
            "• Volume adjustments\n"
            "• Audio cleanup"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_audio_hz = bpy.props.IntProperty(
        name="Audio Debug Hz",
        description="Debug output frequency (1-30 Hz). 30 = every frame, 1 = once per second",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_animations = bpy.props.BoolProperty(
        name="Animation Debug",
        description=(
            "Print animation system debug:\n"
            "• State transitions\n"
            "• NLA strip playback\n"
            "• Custom animation updates\n"
            "• Blend timing"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_animations_hz = bpy.props.IntProperty(
        name="Animation Debug Hz",
        description="Debug output frequency (1-30 Hz). 30 = every frame, 1 = once per second",
        default=5,
        min=1,
        max=30
    )

    bpy.types.Scene.dev_debug_dynamic_offload = bpy.props.BoolProperty(
        name="Dynamic Mesh Offload",
        description=(
            "Print dynamic mesh activation offload debug:\n"
            "• Job submissions with mesh count\n"
            "• Worker activation decisions\n"
            "• Activation state transitions\n"
            "• Performance timing"
        ),
        default=False
    )

    bpy.types.Scene.dev_debug_dynamic_offload_hz = bpy.props.IntProperty(
        name="Dynamic Mesh Offload Hz",
        description="Debug output frequency (1-30 Hz). 30 = every frame, 1 = once per second",
        default=5,
        min=1,
        max=30
    )

    # ══════════════════════════════════════════════════════════════════════
    # Engine Stress Tests
    # ══════════════════════════════════════════════════════════════════════

    bpy.types.Scene.dev_run_sync_test = bpy.props.BoolProperty(
        name="Run Engine Sync Tests",
        description=(
            "Run comprehensive engine synchronization stress tests on game start.\n\n"
            "Tests:\n"
            "• BASELINE - Zero-latency capability\n"
            "• LIGHT_LOAD - 5 jobs/frame (normal gameplay)\n"
            "• MEDIUM_LOAD - 15 jobs/frame (busy gameplay)\n"
            "• HEAVY_LOAD - 30 jobs/frame (worst case)\n"
            "• BURST_TEST - 50-job bursts every second\n"
            "• CATCHUP_STRESS - Modal inconsistency handling\n\n"
            "Duration: ~18 seconds\n"
            "Results: Printed to console on game end"
        ),
        default=False
    )

    # ══════════════════════════════════════════════════════════════════════
    # Master Toggle
    # ══════════════════════════════════════════════════════════════════════

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

    scene.dev_debug_engine = enabled
    scene.dev_debug_performance = enabled
    scene.dev_debug_physics = enabled
    scene.dev_debug_interactions = enabled
    scene.dev_debug_audio = enabled
    scene.dev_debug_animations = enabled
    scene.dev_debug_dynamic_offload = enabled


def unregister_properties():
    """Unregister all developer debug properties."""

    # Remove properties in reverse order
    if hasattr(bpy.types.Scene, 'dev_debug_all'):
        del bpy.types.Scene.dev_debug_all

    if hasattr(bpy.types.Scene, 'dev_run_sync_test'):
        del bpy.types.Scene.dev_run_sync_test

    if hasattr(bpy.types.Scene, 'dev_debug_dynamic_offload_hz'):
        del bpy.types.Scene.dev_debug_dynamic_offload_hz

    if hasattr(bpy.types.Scene, 'dev_debug_dynamic_offload'):
        del bpy.types.Scene.dev_debug_dynamic_offload

    if hasattr(bpy.types.Scene, 'dev_debug_animations_hz'):
        del bpy.types.Scene.dev_debug_animations_hz

    if hasattr(bpy.types.Scene, 'dev_debug_animations'):
        del bpy.types.Scene.dev_debug_animations

    if hasattr(bpy.types.Scene, 'dev_debug_audio_hz'):
        del bpy.types.Scene.dev_debug_audio_hz

    if hasattr(bpy.types.Scene, 'dev_debug_audio'):
        del bpy.types.Scene.dev_debug_audio

    if hasattr(bpy.types.Scene, 'dev_debug_interactions_hz'):
        del bpy.types.Scene.dev_debug_interactions_hz

    if hasattr(bpy.types.Scene, 'dev_debug_interactions'):
        del bpy.types.Scene.dev_debug_interactions

    if hasattr(bpy.types.Scene, 'dev_debug_physics_hz'):
        del bpy.types.Scene.dev_debug_physics_hz

    if hasattr(bpy.types.Scene, 'dev_debug_physics'):
        del bpy.types.Scene.dev_debug_physics

    if hasattr(bpy.types.Scene, 'dev_debug_performance_hz'):
        del bpy.types.Scene.dev_debug_performance_hz

    if hasattr(bpy.types.Scene, 'dev_debug_performance'):
        del bpy.types.Scene.dev_debug_performance

    if hasattr(bpy.types.Scene, 'dev_debug_engine_hz'):
        del bpy.types.Scene.dev_debug_engine_hz

    if hasattr(bpy.types.Scene, 'dev_debug_engine'):
        del bpy.types.Scene.dev_debug_engine
