from __future__ import annotations
import bpy

def pyget(obj, name, default=None):
    """Python-level getattr that bypasses Blender RNA. Safe for freed operators."""
    try:
        return object.__getattribute__(obj, name)
    except Exception:
        return default

def safe_modal():
    """Return ACTIVE_MODAL_OP if its Python wrapper is still valid; else None."""
    try:
        from ..audio import exp_globals as _g
        m = getattr(_g, "ACTIVE_MODAL_OP", None)
    except Exception:
        return None
    if m is None:
        return None
    try:
        _ = object.__getattribute__(m, "__dict__")
    except Exception:
        return None
    return m
