# Master aggregator for the Developer HUD.
# Public API is preserved and re-exported from focused modules.

from __future__ import annotations
import bpy

# Local pieces (single responsibility)
from .dev_props import ensure_scene_props
from .dev_bus import BUS, devhud_set, devhud_post, devhud_series_push  # re-exported
from .dev_state import frame_begin, frame_end
from .dev_draw import draw_2d
from .dev_panel import register as _register_panel, unregister as _unregister_panel

# NEW: bring in prop class registration and section imports explicitly
from .dev_props import register as _register_props, unregister as _unregister_props
from .dev_registry import REGISTRY

__all__ = [
    "devhud_set", "devhud_post", "devhud_series_push",
    "devhud_frame_begin", "devhud_frame_end",
    "register", "unregister"
]

# -------- draw handler wiring --------

_HANDLER = None
_ENABLED = False

def _draw_cb():
    # Context-free draw callback; draw_2d pulls bpy.context internally.
    draw_2d(BUS)

def _enable_draw_handler():
    global _HANDLER, _ENABLED
    if _HANDLER is None:
        _HANDLER = bpy.types.SpaceView3D.draw_handler_add(_draw_cb, (), 'WINDOW', 'POST_PIXEL')
    _ENABLED = True

def _disable_draw_handler():
    global _HANDLER, _ENABLED
    if _HANDLER is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_HANDLER, 'WINDOW')
        except Exception:
            pass
        _HANDLER = None
    _ENABLED = False

def _import_sections():
    """
    Force-load all section modules so they can self-register into REGISTRY.
    Safe to call multiple times.
    """
    # Prevent duplicate instances on reload by clearing the registry first.
    try:
        REGISTRY.clear()
    except Exception:
        pass

    # Import modules for side-effect registration (order matters only for defaults)
    try:
        from .dev_sections import section_core as _s0           # noqa: F401
        from .dev_sections import sections_xr as _s1            # XR core (status/flow)
        from .dev_sections import section_xr_geom as _s1g       # XR.Geom (the sub-sections you toggle)   # noqa: F401
        from .dev_sections import section_world as _s2           # noqa: F401
        from .dev_sections import section_physics as _s3         # noqa: F401
        from .dev_sections import section_camera_view as _s4     # noqa: F401
        from .dev_sections import section_graphs as _s5          # noqa: F401
        # Removed section_custom import by design (no “Show Custom” path anymore)
    except Exception as e:
        print(f"[DEVHUD] section import error: {e}")

# -------- public frame hooks (unchanged names) --------

def devhud_frame_begin(modal):
    """Call near the top of your TIMER frame."""
    scn = bpy.context.scene if bpy.context is not None else None
    if not (scn and getattr(scn, "dev_hud_enable", False)):
        _disable_draw_handler()
        return
    ensure_scene_props()
    if not _ENABLED:
        _enable_draw_handler()
    # begin-of-frame state update + per-frame BUS.temp clear happens inside frame_begin
    frame_begin(modal, BUS)

def devhud_frame_end(modal, context):
    """Call at the very end of your TIMER frame."""
    scn = context.scene if context else None
    if not (scn and getattr(scn, "dev_hud_enable", False)):
        return
    frame_end(modal, context, BUS)

# -------- Blender registration --------

def register():
    # 1) Types/props (DevHudChannel, Scene props)
    _register_props()
    ensure_scene_props()
    # 2) Sections (side-effect registration into REGISTRY)
    _import_sections()
    # 3) Panel/UI
    _register_panel()

def unregister():
    _disable_draw_handler()
    _unregister_panel()
    _unregister_props()
