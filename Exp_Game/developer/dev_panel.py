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
        # Master Frequency Control
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Master Frequency Control", icon='TIME')
        col = box.column(align=True)
        col.prop(scene, "dev_debug_master_hz", text="Output Frequency (Hz)")
        col.label(text="Controls ALL debug output frequency:", icon='INFO')
        col.label(text="  • 30 = Every frame (verbose)")
        col.label(text="  • 5 = Every 5th frame (~0.17s)")
        col.label(text="  • 1 = Once per second (recommended)")

        # Export Session Log toggle
        col.separator()
        col.prop(scene, "dev_export_session_log", text="Export Diagnostics Log to File")

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

        col.prop(scene, "dev_debug_engine", text="Engine Diagnostics")

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
        col.prop(scene, "dev_debug_kcc_offload", text="KCC Physics")

        # Frame Numbers (separate from KCC logs)
        col.prop(scene, "dev_debug_frame_numbers", text="Frame Numbers")

        # Camera Occlusion
        col.prop(scene, "dev_debug_camera_offload", text="Camera Occlusion")

        # Performance Culling
        col.prop(scene, "dev_debug_performance", text="Performance Culling")

        # Dynamic Mesh
        col.prop(scene, "dev_debug_dynamic_offload", text="Dynamic Mesh")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Physics Diagnostics (Deep debugging for physics smoothness)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Physics Diagnostics", icon='PHYSICS')

        col = box.column(align=True)

        # Physics Timing
        col.prop(scene, "dev_debug_physics_timing", text="Physics Timing")

        # Physics Catchup
        col.prop(scene, "dev_debug_physics_catchup", text="Physics Catchup")

        # Platform Carry
        col.prop(scene, "dev_debug_physics_platform", text="Platform Carry")

        # Physics Consistency
        col.prop(scene, "dev_debug_physics_consistency", text="Physics Consistency")

        col.separator()
        col.label(text="Granular Physics:", icon='ALIGN_JUSTIFY')

        # 1. Capsule Collision
        col.prop(scene, "dev_debug_physics_capsule", text="Capsule Collision")

        # 2. Depenetration
        col.prop(scene, "dev_debug_physics_depenetration", text="Depenetration")

        # 3. Ground Detection
        col.prop(scene, "dev_debug_physics_ground", text="Ground Detection")

        # 4. Step-Up
        col.prop(scene, "dev_debug_physics_step_up", text="Step-Up")

        # 5. Slopes
        col.prop(scene, "dev_debug_physics_slopes", text="Slopes")

        # 6. Wall Slide
        col.prop(scene, "dev_debug_physics_slide", text="Wall Slide")

        # 7. Vertical Movement
        col.prop(scene, "dev_debug_physics_vertical", text="Vertical Movement")

        # 8. Enhanced Diagnostics
        col.prop(scene, "dev_debug_physics_enhanced", text="Enhanced Diagnostics")

        # 9. Body Integrity Ray
        col.prop(scene, "dev_debug_physics_body_integrity", text="Body Integrity Ray")

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

            # Line width and vector scale controls
            col.separator(factor=0.5)
            col.prop(scene, "dev_debug_kcc_visual_line_width", text="Line Width")
            col.prop(scene, "dev_debug_kcc_visual_vector_scale", text="Vector Scale")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Game Systems (Main thread)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Game Systems", icon='PLAY')

        col = box.column(align=True)

        # Interactions
        col.prop(scene, "dev_debug_interactions", text="Interactions")

        # Audio
        col.prop(scene, "dev_debug_audio", text="Audio")

        # Animations
        col.prop(scene, "dev_debug_animations", text="Animations")