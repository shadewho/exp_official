from __future__ import annotations
import bpy
from bpy.types import Panel
from .dev_props import ensure_scene_props

class EXP_DEV_PT_HUD(Panel):
    bl_label = "Developer HUD"
    bl_idname = "EXP_DEV_PT_HUD"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    def draw(self, context):
        ensure_scene_props()
        s = context.scene
        col = self.layout.column(align=True)
        col.prop(s, "dev_hud_enable")
        col.prop(s, "dev_hud_graphs")
        col.prop(s, "dev_hud_position")
        col.prop(s, "dev_hud_scale")
        col.separator()
        col.label(text="Sections:")
        row = col.row(align=True)
        row.prop(s, "dev_hud_show_xr"); row.prop(s, "dev_hud_show_world")
        row = col.row(align=True)
        row.prop(s, "dev_hud_show_physics"); row.prop(s, "dev_hud_show_camera")
        col.prop(s, "dev_hud_show_view")
        col.prop(s, "dev_hud_show_custom")
        col.separator()
        col.prop(s, "dev_hud_log_console")
        col.prop(s, "dev_hud_max_samples")
        col.separator()
        col.label(text="Console logging (channels):")
        row = col.row(align=True)
        row.prop(s, "dev_log_view_console"); row.prop(s, "dev_log_xr_console")
        row = col.row(align=True)
        row.prop(s, "dev_log_view_hz"); row.prop(s, "dev_log_xr_hz")

_CLASSES = (EXP_DEV_PT_HUD,)

def register():
    ensure_scene_props()
    for C in _CLASSES:
        try: bpy.utils.register_class(C)
        except Exception: pass

def unregister():
    for C in reversed(_CLASSES):
        try: bpy.utils.unregister_class(C)
        except Exception: pass
