from __future__ import annotations
import bpy

def ensure_scene_props():
    S = bpy.types.Scene

    def add(name, prop):
        if not hasattr(S, name):
            setattr(S, name, prop)

    add("dev_hud_enable", bpy.props.BoolProperty(
        name="Enable Dev HUD", default=False,
        description="Draw Developer HUD overlay with XR & modal stats"))

    add("dev_hud_graphs", bpy.props.BoolProperty(
        name="Show Graphs", default=True,
        description="Draw mini line graphs under the text"))

    add("dev_hud_position", bpy.props.EnumProperty(
        name="HUD Position", default='TR',
        items=[('TR','Top Right',''),('TL','Top Left',''),
               ('BR','Bottom Right',''),('BL','Bottom Left','')],
        description="Screen corner for HUD"))

    add("dev_hud_scale", bpy.props.IntProperty(
        name="Scale", default=1, min=1, max=3,
        description="HUD scale multiplier (fonts, padding)"))

    add("dev_hud_log_console", bpy.props.BoolProperty(
        name="Console Logs (summary)", default=True,
        description="Print a short HUD summary to the system console ~1/s"))

    add("dev_hud_max_samples", bpy.props.IntProperty(
        name="Samples", default=300, min=60, max=1200,
        description="How many points to keep in each graph"))

    # Section toggles
    add("dev_hud_show_xr",      bpy.props.BoolProperty(name="Show XR", default=True))
    add("dev_hud_show_world",   bpy.props.BoolProperty(name="Show World", default=True))
    add("dev_hud_show_physics", bpy.props.BoolProperty(name="Show Physics", default=True))
    add("dev_hud_show_camera",  bpy.props.BoolProperty(name="Show Camera", default=True))
    add("dev_hud_show_view",    bpy.props.BoolProperty(
        name="Show View", default=True,
        description="Show detailed View (boom) stats & frequencies"))
    add("dev_hud_show_custom",  bpy.props.BoolProperty(name="Show Custom", default=True))

    # Channelled console logging (toggles + Hz)
    add("dev_log_view_console", bpy.props.BoolProperty(
        name="Log ViewXR", default=False,
        description="Console prints for XR view jobs (rate-limited)"))
    add("dev_log_view_hz", bpy.props.FloatProperty(
        name="View Log Hz", default=4.0, min=0.1, max=60.0,
        description="Max ViewXR prints per second"))
    add("dev_log_xr_console", bpy.props.BoolProperty(
        name="Log XR Core", default=False,
        description="Console prints for XR core (req/ok/fail, phase)"))
    add("dev_log_xr_hz", bpy.props.FloatProperty(
        name="XR Core Log Hz", default=2.0, min=0.1, max=60.0,
        description="Max XR core prints per second"))
