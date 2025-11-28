# Exp_Game/developer/dev_panel.py
"""
Developer Tools UI Panel

Provides a clean interface for toggling debug output categories
and running diagnostic tests.
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
        # Console Debug Categories
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Console Debug Output", icon='CONSOLE')

        col = box.column(align=True)

        # Engine debug with Hz control
        row = col.row(align=True)
        row.prop(scene, "dev_debug_engine", text="Engine (Multiprocessing)")
        if scene.dev_debug_engine:
            row.prop(scene, "dev_debug_engine_hz", text="Hz")

        # Performance debug with Hz control
        row = col.row(align=True)
        row.prop(scene, "dev_debug_performance", text="Performance Culling")
        if scene.dev_debug_performance:
            row.prop(scene, "dev_debug_performance_hz", text="Hz")

        # Dynamic mesh offload debug with Hz control
        row = col.row(align=True)
        row.prop(scene, "dev_debug_dynamic_offload", text="Dynamic Mesh Offload")
        if scene.dev_debug_dynamic_offload:
            row.prop(scene, "dev_debug_dynamic_offload_hz", text="Hz")

        # KCC offload debug with Hz control
        row = col.row(align=True)
        row.prop(scene, "dev_debug_kcc_offload", text="KCC Physics Offload")
        if scene.dev_debug_kcc_offload:
            row.prop(scene, "dev_debug_kcc_offload_hz", text="Hz")

        # Raycast offload debug with Hz control
        row = col.row(align=True)
        row.prop(scene, "dev_debug_raycast_offload", text="Raycast Offload")
        if scene.dev_debug_raycast_offload:
            row.prop(scene, "dev_debug_raycast_offload_hz", text="Hz")

        # Physics debug with Hz control
        row = col.row(align=True)
        row.prop(scene, "dev_debug_physics", text="Physics & Character")
        if scene.dev_debug_physics:
            row.prop(scene, "dev_debug_physics_hz", text="Hz")

        # Interactions debug with Hz control
        row = col.row(align=True)
        row.prop(scene, "dev_debug_interactions", text="Interactions & Reactions")
        if scene.dev_debug_interactions:
            row.prop(scene, "dev_debug_interactions_hz", text="Hz")

        # Audio debug with Hz control
        row = col.row(align=True)
        row.prop(scene, "dev_debug_audio", text="Audio System")
        if scene.dev_debug_audio:
            row.prop(scene, "dev_debug_audio_hz", text="Hz")

        # Animations debug with Hz control
        row = col.row(align=True)
        row.prop(scene, "dev_debug_animations", text="Animations & NLA")
        if scene.dev_debug_animations:
            row.prop(scene, "dev_debug_animations_hz", text="Hz")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Engine Stress Tests
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Engine Diagnostics", icon='EXPERIMENTAL')

        col = box.column(align=True)
        col.prop(scene, "dev_run_sync_test", text="Run Sync Stress Tests on Start")

        # Info label
        if scene.dev_run_sync_test:
            info_box = box.box()
            info_col = info_box.column(align=True)
            info_col.scale_y = 0.8
            info_col.label(text="Tests engine-modal synchronization", icon='INFO')
            info_col.label(text="Duration: ~18 seconds")
            info_col.label(text="Results: Printed to console on game end")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Future: Performance Rendering (placeholder for later)
        # ═══════════════════════════════════════════════════════════════
        # Uncomment when ready to add render diagnostics:
        #
        # box = layout.box()
        # box.label(text="Performance Rendering", icon='SHADING_RENDERED')
        # col = box.column(align=True)
        # col.enabled = False  # Disabled until implemented
        # col.label(text="Coming Soon:", icon='TIME')
        # col.label(text="• Frame timing overlay")
        # col.label(text="• Physics step counter")
        # col.label(text="• Memory usage graph")
