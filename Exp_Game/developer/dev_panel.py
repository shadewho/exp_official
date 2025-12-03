# Exp_Game/developer/dev_panel.py
"""
Developer Tools UI Panel

Provides a clean interface for toggling debug output categories
and running diagnostic tests.

Categories:
  - Engine Health: Core multiprocessing engine diagnostics
  - Offload Systems: Worker-bound computation
  - Game Systems: Main thread game logic
"""

import bpy


def _is_create_panel_enabled(scene, key: str) -> bool:
    """Check if a specific Create panel is enabled in the filter."""
    flags = getattr(scene, "create_panels_filter", None)
    if flags is None:
        return True
    if hasattr(flags, "__len__") and len(flags) == 0:
        return False
    return (key in flags)


class DEV_PT_DeveloperTools(bpy.types.Panel):
    """Developer Tools panel in Create tab."""
    bl_label = "Developer Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_idname = "DEV_PT_developer_tools"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (context.scene.main_category == 'CREATE'
                and _is_create_panel_enabled(context.scene, 'DEV'))

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # ═══════════════════════════════════════════════════════════════
        # Dev Refresh Button (Development Mode)
        # ═══════════════════════════════════════════════════════════════
        row = layout.row()
        row.scale_y = 1.5
        row.operator("exp.dev_refresh", text="Refresh Addon", icon='FILE_REFRESH')

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Engine Health
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Engine Health", icon='MEMORY')

        col = box.column(align=True)

        # Startup logs
        col.prop(scene, "dev_startup_logs", text="Startup Logs")

        col.separator()

        row = col.row(align=True)
        row.prop(scene, "dev_debug_engine", text="Engine Diagnostics")
        if scene.dev_debug_engine:
            row.prop(scene, "dev_debug_engine_hz", text="Hz")

        # Manual stress test operators
        col.separator()
        col.label(text="Manual Stress Tests:", icon='PLAY')
        row = col.row(align=True)
        row.operator("exp_engine.quick_test", text="Quick Test", icon='CHECKMARK')
        row.operator("exp_engine.stress_test", text="Full Stress Test", icon='COMMUNITY')

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Offload Systems (Worker-bound)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Offload Systems", icon='FORCE_CHARGE')

        col = box.column(align=True)

        # KCC Physics (the big one)
        row = col.row(align=True)
        row.prop(scene, "dev_debug_kcc_offload", text="KCC Physics")
        if scene.dev_debug_kcc_offload:
            row.prop(scene, "dev_debug_kcc_offload_hz", text="Hz")

        # Camera Occlusion
        row = col.row(align=True)
        row.prop(scene, "dev_debug_camera_offload", text="Camera Occlusion")
        if scene.dev_debug_camera_offload:
            row.prop(scene, "dev_debug_camera_offload_hz", text="Hz")

        # Performance Culling
        row = col.row(align=True)
        row.prop(scene, "dev_debug_performance", text="Performance Culling")
        if scene.dev_debug_performance:
            row.prop(scene, "dev_debug_performance_hz", text="Hz")

        # Dynamic Mesh
        row = col.row(align=True)
        row.prop(scene, "dev_debug_dynamic_offload", text="Dynamic Mesh")
        if scene.dev_debug_dynamic_offload:
            row.prop(scene, "dev_debug_dynamic_offload_hz", text="Hz")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Physics Diagnostics (Deep debugging for physics smoothness)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Physics Diagnostics", icon='PHYSICS')

        col = box.column(align=True)

        # Physics Timing
        row = col.row(align=True)
        row.prop(scene, "dev_debug_physics_timing", text="Physics Timing")
        if scene.dev_debug_physics_timing:
            row.prop(scene, "dev_debug_physics_timing_hz", text="Hz")

        # Physics Catchup
        row = col.row(align=True)
        row.prop(scene, "dev_debug_physics_catchup", text="Physics Catchup")
        if scene.dev_debug_physics_catchup:
            row.prop(scene, "dev_debug_physics_catchup_hz", text="Hz")

        # Platform Carry
        row = col.row(align=True)
        row.prop(scene, "dev_debug_physics_platform", text="Platform Carry")
        if scene.dev_debug_physics_platform:
            row.prop(scene, "dev_debug_physics_platform_hz", text="Hz")

        # Physics Consistency
        row = col.row(align=True)
        row.prop(scene, "dev_debug_physics_consistency", text="Physics Consistency")
        if scene.dev_debug_physics_consistency:
            row.prop(scene, "dev_debug_physics_consistency_hz", text="Hz")

        col.separator()
        col.label(text="Granular Physics:", icon='ALIGN_JUSTIFY')

        # 1. Capsule Collision
        row = col.row(align=True)
        row.prop(scene, "dev_debug_physics_capsule", text="Capsule Collision")
        if scene.dev_debug_physics_capsule:
            row.prop(scene, "dev_debug_physics_capsule_hz", text="Hz")

        # 2. Depenetration
        row = col.row(align=True)
        row.prop(scene, "dev_debug_physics_depenetration", text="Depenetration")
        if scene.dev_debug_physics_depenetration:
            row.prop(scene, "dev_debug_physics_depenetration_hz", text="Hz")

        # 3. Ground Detection
        row = col.row(align=True)
        row.prop(scene, "dev_debug_physics_ground", text="Ground Detection")
        if scene.dev_debug_physics_ground:
            row.prop(scene, "dev_debug_physics_ground_hz", text="Hz")

        # 4. Step-Up
        row = col.row(align=True)
        row.prop(scene, "dev_debug_physics_step_up", text="Step-Up")
        if scene.dev_debug_physics_step_up:
            row.prop(scene, "dev_debug_physics_step_up_hz", text="Hz")

        # 5. Slopes
        row = col.row(align=True)
        row.prop(scene, "dev_debug_physics_slopes", text="Slopes")
        if scene.dev_debug_physics_slopes:
            row.prop(scene, "dev_debug_physics_slopes_hz", text="Hz")

        # 6. Wall Slide
        row = col.row(align=True)
        row.prop(scene, "dev_debug_physics_slide", text="Wall Slide")
        if scene.dev_debug_physics_slide:
            row.prop(scene, "dev_debug_physics_slide_hz", text="Hz")

        # 7. Vertical Movement
        row = col.row(align=True)
        row.prop(scene, "dev_debug_physics_vertical", text="Vertical Movement")
        if scene.dev_debug_physics_vertical:
            row.prop(scene, "dev_debug_physics_vertical_hz", text="Hz")

        # 8. Enhanced Diagnostics
        row = col.row(align=True)
        row.prop(scene, "dev_debug_physics_enhanced", text="Enhanced Diagnostics")
        if scene.dev_debug_physics_enhanced:
            row.prop(scene, "dev_debug_physics_enhanced_hz", text="Hz")

        # ═══════════════════════════════════════════════════════════════
        # KCC VISUAL DEBUG (3D Viewport Overlay)
        # ═══════════════════════════════════════════════════════════════

        layout.separator()
        box = layout.box()
        box.label(text="KCC Visual Debug (3D Viewport)", icon='SHADING_WIRE')
        col = box.column(align=True)

        # Master toggle
        row = col.row(align=True)
        row.prop(scene, "dev_debug_kcc_visual", text="Enable Visual Debug")

        # Individual toggles (only show if master is enabled)
        if scene.dev_debug_kcc_visual:
            col.separator(factor=0.5)
            sub = col.column(align=True)
            sub.prop(scene, "dev_debug_kcc_visual_capsule", text="Capsule Shape")
            sub.prop(scene, "dev_debug_kcc_visual_normals", text="Hit Normals")
            sub.prop(scene, "dev_debug_kcc_visual_ground", text="Ground Ray")
            sub.prop(scene, "dev_debug_kcc_visual_movement", text="Movement Vectors")

        # Session Log Export
        col.separator()
        row = col.row(align=True)
        row.prop(scene, "dev_export_session_log", text="Export Session Log to File")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Game Systems (Main thread)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Game Systems", icon='PLAY')

        col = box.column(align=True)

        # Interactions
        row = col.row(align=True)
        row.prop(scene, "dev_debug_interactions", text="Interactions")
        if scene.dev_debug_interactions:
            row.prop(scene, "dev_debug_interactions_hz", text="Hz")

        # Audio
        row = col.row(align=True)
        row.prop(scene, "dev_debug_audio", text="Audio")
        if scene.dev_debug_audio:
            row.prop(scene, "dev_debug_audio_hz", text="Hz")

        # Animations
        row = col.row(align=True)
        row.prop(scene, "dev_debug_animations", text="Animations")
        if scene.dev_debug_animations:
            row.prop(scene, "dev_debug_animations_hz", text="Hz")
