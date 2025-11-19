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
        return (
            context.scene.main_category == 'CREATE'
            and _is_create_panel_enabled(context.scene, 'DEV')
        )

    def draw(self, context):
        ensure_scene_props()
        s = context.scene
        layout = self.layout

        # --------------------------------------------------
        # HUD CORE
        # --------------------------------------------------
        col = layout.column(align=True)
        col.label(text="HUD:")
        col.prop(s, "dev_hud_enable", text="Enable HUD")

        row = col.row(align=True)
        row.prop(s, "dev_hud_position", text="Position")
        row.prop(s, "dev_hud_scale", text="Scale")

        col.prop(s, "dev_hud_graphs", text="Show Graphs")
        col.prop(s, "dev_hud_max_samples", text="Samples")

        # --------------------------------------------------
        # CATEGORIES (TOP-LEVEL ON/OFF)
        # --------------------------------------------------
        col.separator()
        col.label(text="Categories:")

        g = col.grid_flow(
            row_major=True,
            columns=3,
            even_columns=True,
            even_rows=True,
            align=True,
        )
        g.prop(s, "dev_hud_show_xr",      text="XR (core)")
        g.prop(s, "dev_hud_show_world",   text="World")
        g.prop(s, "dev_hud_show_physics", text="Physics")
        g.prop(s, "dev_hud_show_camera",  text="Camera")
        g.prop(s, "dev_hud_show_view",    text="View")
        g.prop(s, "dev_hud_show_geom",    text="XR.Geom")

        # --------------------------------------------------
        # XR SECTIONS
        # --------------------------------------------------
        box = col.box()
        box.label(text="General XR and Modal (LEFT):")
        box.prop(s, "dev_xr_modal_general", text="Enable")

        box = col.box()
        box.label(text="XR.Geom (sub-sections):")
        gg = box.grid_flow(
            row_major=True,
            columns=3,
            even_columns=True,
            even_rows=True,
            align=True,
        )
        gg.prop(s, "dev_xr_geom_mode_auth", text="(1) Mode/Auth")
        gg.prop(s, "dev_xr_geom_static",    text="(2) Static")
        gg.prop(s, "dev_xr_geom_dynamic",   text="(3) Dynamic")
        gg.prop(s, "dev_xr_geom_xforms",    text="(4) Xforms")
        gg.prop(s, "dev_xr_geom_down_dyn",  text="(5) DownDyn")
        gg.prop(s, "dev_xr_geom_authority", text="(6) Authority")
        gg.prop(s, "dev_xr_geom_parity",    text="(7) Parity")
        gg.prop(s, "dev_xr_geom_rates",     text="(8) Rates")
        gg.prop(s, "dev_xr_geom_dump",      text="(9) Raw dump")

        # --------------------------------------------------
        # CONSOLE LOGGING
        # --------------------------------------------------
        col.separator()
        col.label(text="Console logging:")

        row = col.row(align=True)
        row.prop(s, "dev_hud_log_console", text="Master Console On/Off")

        row = col.row(align=True)
        row.prop(s, "dev_log_xr_console", text="XR Core")
        row.prop(s, "dev_log_xr_hz", text="Hz")

        row = col.row(align=True)
        row.prop(s, "dev_log_view_console", text="View")
        row.prop(s, "dev_log_view_hz", text="Hz")

        row = col.row(align=True)
        row.prop(s, "dev_log_kcc_console", text="KCC")
        row.prop(s, "dev_log_kcc_hz", text="Hz")

        row = col.row(align=True)
        row.prop(s, "dev_log_physics_console", text="Physics")
        row.prop(s, "dev_log_physics_hz", text="Hz")

        row = col.row(align=True)
        row.prop(s, "dev_log_geom_console", text="XR.Geom")
        row.prop(s, "dev_log_geom_hz", text="Hz")

        row = col.row(align=True)
        row.prop(s, "dev_log_forward_sweep_min3_console", text="forward_sweep_min3")
        row.prop(s, "dev_log_forward_sweep_min3_hz", text="Hz")

        # XR health channel (kept, because it's already wired to console)
        row = col.row(align=True)
        row.prop(s, "dev_log_xr_health_console", text="XR Health")
        row.prop(s, "dev_log_xr_health_hz", text="Hz")
        col.prop(s, "dev_log_xr_health_oneline", text="XR Health One-line")


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
