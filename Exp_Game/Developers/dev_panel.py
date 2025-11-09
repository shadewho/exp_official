# Exp_Game/Developers/dev_panel.py
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

        # ===== Core =====
        col.prop(s, "dev_hud_enable")
        row = col.row(align=True)
        row.prop(s, "dev_hud_position")
        row.prop(s, "dev_hud_scale")  # 1..5 mapped internally to pixel factor
        col.prop(s, "dev_hud_graphs")
        col.prop(s, "dev_hud_max_samples")

        # ===== Sections (categories) =====
        col.separator()
        col.label(text="Categories:")
        g = col.grid_flow(row_major=True, columns=3, even_columns=True, even_rows=True, align=True)
        g.prop(s, "dev_hud_show_xr",      text="XR (core)")
        g.prop(s, "dev_hud_show_world",   text="World")
        g.prop(s, "dev_hud_show_physics", text="Physics")
        g.prop(s, "dev_hud_show_camera",  text="Camera")   # kept for future split if needed
        g.prop(s, "dev_hud_show_view",    text="View")
        g.prop(s, "dev_hud_show_geom",    text="XR.Geom")

        # ===== XR Core breakdown =====
        box = col.box()
        box.label(text="XR Core (sub-sections):")
        r = box.row(align=True)
        r.prop(s, "dev_xr_core_status", text="(1) Status (rtt/phase)")
        r.prop(s, "dev_xr_core_flow",   text="(2) Flow (req/ok/fail)")

        # ===== XR.Geom breakdown =====
        box = col.box()
        box.label(text="XR.Geom (sub-sections):")
        g = box.grid_flow(row_major=True, columns=3, even_columns=True, even_rows=True, align=True)
        g.prop(s, "dev_xr_geom_mode_auth", text="(1) Mode/Auth")
        g.prop(s, "dev_xr_geom_static",    text="(2) Static summary")
        g.prop(s, "dev_xr_geom_dynamic",   text="(3) Dynamic summary")
        g.prop(s, "dev_xr_geom_xforms",    text="(4) Xforms")
        g.prop(s, "dev_xr_geom_down_dyn",  text="(5) DownDyn")
        g.prop(s, "dev_xr_geom_authority", text="(6) Authority proof")
        g.prop(s, "dev_xr_geom_parity",    text="(7) Parity")
        g.prop(s, "dev_xr_geom_rates",     text="(8) Query rates")
        g.prop(s, "dev_xr_geom_dump",      text="(9) Raw XR.* dump")  # <— brings back the detailed wall when needed


        # ===== Console logging =====
        col.separator()
        col.label(text="Console logging (master + channels):")
        row = col.row(align=True)
        row.prop(s, "dev_hud_log_console", text="Master Console On/Off")
        row = col.row(align=True)
        row.prop(s, "dev_log_xr_console");   row.prop(s, "dev_log_xr_hz")
        row = col.row(align=True)
        row.prop(s, "dev_log_view_console"); row.prop(s, "dev_log_view_hz")
        row = col.row(align=True)
        row.prop(s, "dev_log_kcc_console");  row.prop(s, "dev_log_kcc_hz")
        row = col.row(align=True)
        row.prop(s, "dev_log_physics_console"); row.prop(s, "dev_log_physics_hz")
        row = col.row(align=True)
        row.prop(s, "dev_log_geom_console"); row.prop(s, "dev_log_geom_hz")

        # ===== DEV parity / authority =====
        col.separator()
        box = col.box()
        box.label(text="Parity (DEV):")
        r = box.row(align=True)
        r.prop(s, "dev_geom_parity_enable", text="Enable XR.Geom Parity")
        r.prop(s, "dev_geom_parity_samples", text="Rays / step")
        box = col.box()
        box.label(text="Authority (DEV):")
        box.prop(s, "dev_geom_down_dyn_auth", text="DOWN: XR dynamic authority")

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
