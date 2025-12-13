# Exp_Game/developer/dev_panel.py
"""
Developer Tools UI Panel

Provides a clean interface for toggling debug output categories.

UNIFIED PHYSICS: Static and dynamic meshes use identical physics code.
All physics logs show source (static/dynamic) - there is ONE system.
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
        # Dev Refresh Button
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
        col.separator()
        col.prop(scene, "dev_export_session_log", text="Export Diagnostics Log")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Engine Health
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Engine Health", icon='MEMORY')
        col = box.column(align=True)
        col.prop(scene, "dev_startup_logs", text="Startup Logs")
        col.prop(scene, "dev_debug_engine", text="Engine Diagnostics")

        col.separator()
        col.label(text="Manual Stress Tests:", icon='PLAY')
        row = col.row(align=True)
        row.operator("exp_engine.quick_test", text="Quick Test", icon='CHECKMARK')
        row.operator("exp_engine.stress_test", text="Full Stress Test", icon='COMMUNITY')

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Offload Systems
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Offload Systems", icon='FORCE_CHARGE')
        col = box.column(align=True)
        col.prop(scene, "dev_debug_kcc_physics", text="KCC Physics")
        col.prop(scene, "dev_debug_frame_numbers", text="Frame Numbers")
        col.prop(scene, "dev_debug_culling", text="Performance Culling")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # UNIFIED PHYSICS (Single section - static + dynamic identical)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Unified Physics", icon='PHYSICS')

        col = box.column(align=True)
        col.prop(scene, "dev_debug_physics", text="Physics Summary")

        col.separator()
        col.label(text="Granular (shows source: static/dynamic):", icon='ALIGN_JUSTIFY')

        col.prop(scene, "dev_debug_physics_ground", text="Ground Detection")
        col.prop(scene, "dev_debug_physics_horizontal", text="Horizontal Collision")
        col.prop(scene, "dev_debug_physics_body", text="Body Integrity")
        col.prop(scene, "dev_debug_physics_ceiling", text="Ceiling Check")
        col.prop(scene, "dev_debug_physics_step", text="Step-Up")
        col.prop(scene, "dev_debug_physics_slide", text="Wall Slide")
        col.prop(scene, "dev_debug_physics_slopes", text="Slopes")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Unified Camera (Static + Dynamic use same raycast)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Unified Camera", icon='VIEW_CAMERA')
        col = box.column(align=True)
        col.prop(scene, "dev_debug_camera", text="Camera Raycast")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Dynamic Mesh System (Unified with static)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Dynamic Mesh System", icon='MESH_DATA')
        col = box.column(align=True)
        col.prop(scene, "dev_debug_dynamic_cache", text="Cache & Transforms")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # KCC Visual Debug (3D Viewport)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="KCC Visual Debug (3D Viewport)", icon='SHADING_WIRE')
        col = box.column(align=True)

        row = col.row(align=True)
        row.prop(scene, "dev_debug_kcc_visual", text="Enable Visual Debug")

        if scene.dev_debug_kcc_visual:
            col.separator(factor=0.5)
            sub = col.column(align=True)
            sub.prop(scene, "dev_debug_kcc_visual_capsule", text="Capsule Shape")
            sub.prop(scene, "dev_debug_kcc_visual_normals", text="Hit Normals")
            sub.prop(scene, "dev_debug_kcc_visual_ground", text="Ground Ray")
            sub.prop(scene, "dev_debug_kcc_visual_movement", text="Movement Vectors")

            col.separator(factor=0.5)
            col.prop(scene, "dev_debug_kcc_visual_line_width", text="Line Width")
            col.prop(scene, "dev_debug_kcc_visual_vector_scale", text="Vector Scale")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Game Systems
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Game Systems", icon='PLAY')
        col = box.column(align=True)
        col.prop(scene, "dev_debug_interactions", text="Interactions")
        col.prop(scene, "dev_debug_audio", text="Audio")
        col.prop(scene, "dev_debug_animations", text="Animations")


def register():
    bpy.utils.register_class(DEV_PT_DeveloperTools)


def unregister():
    bpy.utils.unregister_class(DEV_PT_DeveloperTools)
