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
        # Master Toggle
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        row = box.row()
        row.scale_y = 1.2
        row.prop(scene, "dev_debug_all", text="Enable All Debug Output", toggle=True)

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Engine Health
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Engine Health", icon='MEMORY')

        col = box.column(align=True)

        row = col.row(align=True)
        row.prop(scene, "dev_debug_engine", text="Engine Diagnostics")
        if scene.dev_debug_engine:
            row.prop(scene, "dev_debug_engine_hz", text="Hz")

        # Stress test toggle
        col.separator()
        col.prop(scene, "dev_run_sync_test", text="Run Stress Tests on Start")

        if scene.dev_run_sync_test:
            info_box = box.box()
            info_col = info_box.column(align=True)
            info_col.scale_y = 0.8
            info_col.label(text="Tests engine-modal sync", icon='INFO')
            info_col.label(text="Duration: ~18 seconds")

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
