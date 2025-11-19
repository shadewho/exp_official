from __future__ import annotations
import bpy
from bpy.types import Panel
from .dev_props import ensure_scene_props
def _is_create_panel_enabled(scene, key: str) -> bool:
    flags = getattr(scene, "create_panels_filter", None)
    # If the property doesn't exist yet, default to visible.
    if flags is None:
        return True
    # If the property exists but is an empty set, hide all.
    if hasattr(flags, "__len__") and len(flags) == 0:
        return False
    return (key in flags)


class EXP_DEV_PT_HUD(Panel):
    bl_label = "Developer HUD"
    bl_idname = "EXP_DEV_PT_HUD"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    @classmethod
    def poll(cls, context):
        return (context.scene.main_category == 'CREATE'
                and _is_create_panel_enabled(context.scene, 'DEV'))
    
    def draw(self, context):
        ensure_scene_props()
        s = context.scene
        col = self.layout.column(align=True)

        # Core
        col.prop(s, "dev_hud_enable")
        r = col.row(align=True); r.prop(s, "dev_hud_position"); r.prop(s, "dev_hud_scale")
        col.prop(s, "dev_hud_graphs")
        col.prop(s, "dev_hud_max_samples")

        # Categories
        col.separator(); col.label(text="Categories:")
        g = col.grid_flow(row_major=True, columns=3, even_columns=True, even_rows=True, align=True)
        g.prop(s, "dev_hud_show_xr",      text="XR (core)")
        g.prop(s, "dev_hud_show_world",   text="World")
        g.prop(s, "dev_hud_show_physics", text="Physics")
        g.prop(s, "dev_hud_show_camera",  text="Camera")
        g.prop(s, "dev_hud_show_view",    text="View")
        g.prop(s, "dev_hud_show_geom",    text="XR.Geom")

        # XR consolidated
        box = col.box(); box.label(text="General XR and Modal (LEFT):")
        box.prop(s, "dev_xr_modal_general", text="Enable")

        # XR.Geom
        box = col.box(); box.label(text="XR.Geom (sub-sections):")
        gg = box.grid_flow(row_major=True, columns=3, even_columns=True, even_rows=True, align=True)
        gg.prop(s, "dev_xr_geom_mode_auth", text="(1) Mode/Auth")
        gg.prop(s, "dev_xr_geom_static",    text="(2) Static")
        gg.prop(s, "dev_xr_geom_dynamic",   text="(3) Dynamic")
        gg.prop(s, "dev_xr_geom_xforms",    text="(4) Xforms")
        gg.prop(s, "dev_xr_geom_down_dyn",  text="(5) DownDyn")
        gg.prop(s, "dev_xr_geom_authority", text="(6) Authority")
        gg.prop(s, "dev_xr_geom_parity",    text="(7) Parity")
        gg.prop(s, "dev_xr_geom_rates",     text="(8) Rates")
        gg.prop(s, "dev_xr_geom_dump",      text="(9) Raw dump")

        # Console logging
        col.separator(); col.label(text="Console logging (master + channels):")
        r = col.row(align=True); r.prop(s, "dev_hud_log_console", text="Master Console On/Off")
        r = col.row(align=True); r.prop(s, "dev_log_xr_console");   r.prop(s, "dev_log_xr_hz")
        r = col.row(align=True); r.prop(s, "dev_log_view_console"); r.prop(s, "dev_log_view_hz")
        r = col.row(align=True); r.prop(s, "dev_log_kcc_console");  r.prop(s, "dev_log_kcc_hz")
        r = col.row(align=True); r.prop(s, "dev_log_physics_console"); r.prop(s, "dev_log_physics_hz")
        r = col.row(align=True); r.prop(s, "dev_log_geom_console"); r.prop(s, "dev_log_geom_hz")

        # DEV parity / authority
        col.separator()
        b = col.box(); b.label(text="Parity (DEV):")
        r = b.row(align=True); r.prop(s, "dev_geom_parity_enable", text="Enable XR.Geom Parity"); r.prop(s, "dev_geom_parity_samples", text="Rays / step")
        b = col.box(); b.label(text="Authority (DEV):"); b.prop(s, "dev_geom_down_dyn_auth", text="DOWN: XR dynamic authority")



_CLASSES = (EXP_DEV_PT_HUD,)

def register():
    ensure_scene_props()
    for C in _CLASSES:
        try:
            bpy.utils.register_class(C)
        except Exception:
            pass

def unregister():
    for C in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(C)
        except Exception:
            pass
