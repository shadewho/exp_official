from __future__ import annotations
import bpy

_PROP_NAMES = [
    # HUD core
    "dev_hud_enable", "dev_hud_graphs", "dev_hud_position", "dev_hud_scale",
    "dev_hud_log_console", "dev_hud_max_samples",
    # Sections (categories)
    "dev_hud_show_xr", "dev_hud_show_world", "dev_hud_show_physics",
    "dev_hud_show_camera", "dev_hud_show_view", "dev_hud_show_geom",
    # Console channels
    "dev_log_view_console","dev_log_view_hz",
    "dev_log_xr_console","dev_log_xr_hz",
    "dev_log_kcc_console","dev_log_kcc_hz",
    "dev_log_physics_console","dev_log_physics_hz",
    "dev_log_geom_console","dev_log_geom_hz",
    "dev_log_xr_health_console","dev_log_xr_health_hz","dev_log_xr_health_oneline",
    "dev_log_forward_sweep_min3_console",
    # XR consolidated + legacy sub-sections
    "dev_xr_modal_general",
    # XR.Geom sub-sections
    "dev_xr_geom_mode_auth","dev_xr_geom_static","dev_xr_geom_dynamic",
    "dev_xr_geom_xforms","dev_xr_geom_down_dyn","dev_xr_geom_authority",
    "dev_xr_geom_parity","dev_xr_geom_rates","dev_xr_geom_dump",
    # DEV toggles
    "dev_geom_parity_enable","dev_geom_parity_samples","dev_geom_down_dyn_auth",
]



def ensure_scene_props():
    S = bpy.types.Scene
    def add(name, prop):
        if not hasattr(S, name):
            setattr(S, name, prop)

    # HUD core
    add("dev_hud_enable", bpy.props.BoolProperty(name="Enable Dev HUD", default=False))
    add("dev_hud_graphs", bpy.props.BoolProperty(name="Show Graphs", default=True))
    add("dev_hud_position", bpy.props.EnumProperty(
        name="HUD Position", default='TR',
        items=[('TR','Top Right',''),('TL','Top Left',''),
               ('BR','Bottom Right',''),('BL','Bottom Left','')]))
    add("dev_hud_scale", bpy.props.IntProperty(name="Scale", default=3, min=1, max=5))
    add("dev_hud_log_console", bpy.props.BoolProperty(name="Console Logs (master)", default=True))
    add("dev_hud_max_samples", bpy.props.IntProperty(name="Samples", default=300, min=60, max=1200))

    # Categories
    add("dev_hud_show_xr",      bpy.props.BoolProperty(name="Show XR", default=True))
    add("dev_hud_show_world",   bpy.props.BoolProperty(name="Show World", default=True))
    add("dev_hud_show_physics", bpy.props.BoolProperty(name="Show Physics", default=True))
    add("dev_hud_show_camera",  bpy.props.BoolProperty(name="Show Camera", default=True))
    add("dev_hud_show_view",    bpy.props.BoolProperty(name="Show View", default=True))
    add("dev_hud_show_geom",    bpy.props.BoolProperty(name="Show XR.Geom", default=True))

    # Console channels
    add("dev_log_view_console",   bpy.props.BoolProperty(name="Log View", default=False))
    add("dev_log_view_hz",        bpy.props.FloatProperty(name="View Log Hz", default=4.0, min=0.1, max=60.0))
    add("dev_log_xr_console",     bpy.props.BoolProperty(name="Log XR Core", default=False))
    add("dev_log_xr_hz",          bpy.props.FloatProperty(name="XR Core Log Hz", default=2.0, min=0.1, max=60.0))
    add("dev_log_kcc_console",    bpy.props.BoolProperty(name="Log KCC", default=False))
    add("dev_log_kcc_hz",         bpy.props.FloatProperty(name="KCC Log Hz", default=4.0, min=0.1, max=60.0))
    add("dev_log_physics_console",bpy.props.BoolProperty(name="Log Physics", default=False))
    add("dev_log_physics_hz",     bpy.props.FloatProperty(name="Physics Log Hz", default=3.0, min=0.1, max=60.0))
    add("dev_log_geom_console",   bpy.props.BoolProperty(name="Log XR.Geom", default=False))
    add("dev_log_geom_hz",        bpy.props.FloatProperty(name="XR.Geom Log Hz", default=1.0, min=0.1, max=60.0))

    add("dev_log_forward_sweep_min3_console", bpy.props.BoolProperty(name="Log forward_sweep_min3", default=False))
    add("dev_log_forward_sweep_min3_hz", bpy.props.FloatProperty(name="forward_sweep_min3 Log Hz", default=1.0, min=0.1, max=60.0))

    "dev_xr_sync_kmax_frames","dev_xr_overwhelm_backlog_max","dev_xr_overwhelm_rttp95_ms",
    "dev_xr_softsync_enable","dev_xr_softsync_budget_ms",



    # XR Health dedicated channel
    add("dev_log_xr_health_console", bpy.props.BoolProperty(name="Log XR Health", default=True))
    add("dev_log_xr_health_hz",      bpy.props.FloatProperty(name="XR Health Log Hz", default=2.0, min=0.1, max=60.0))
    add("dev_log_xr_health_oneline", bpy.props.BoolProperty(name="XR Health One-line", default=True))

    # Consolidated XR section toggle
    add("dev_xr_modal_general", bpy.props.BoolProperty(name="General XR and Modal (LEFT)", default=True))

    # XR.Geom
    add("dev_xr_geom_mode_auth", bpy.props.BoolProperty(name="Mode/Auth", default=True))
    add("dev_xr_geom_static",    bpy.props.BoolProperty(name="Static summary", default=True))
    add("dev_xr_geom_dynamic",   bpy.props.BoolProperty(name="Dynamic summary", default=True))
    add("dev_xr_geom_xforms",    bpy.props.BoolProperty(name="Xforms", default=True))
    add("dev_xr_geom_down_dyn",  bpy.props.BoolProperty(name="DownDyn", default=False))
    add("dev_xr_geom_authority", bpy.props.BoolProperty(name="Authority proof", default=True))
    add("dev_xr_geom_parity",    bpy.props.BoolProperty(name="Parity", default=True))
    add("dev_xr_geom_rates",     bpy.props.BoolProperty(name="Query rates", default=False))
    add("dev_xr_geom_dump",      bpy.props.BoolProperty(name="Raw XR.* dump", default=False))
    


def register(): ensure_scene_props()
def unregister():
    S = bpy.types.Scene
    for n in _PROP_NAMES:
        try:
            if hasattr(S, n): delattr(S, n)
        except Exception: pass

__all__ = ["ensure_scene_props", "register", "unregister"]
