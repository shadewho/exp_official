# ────────── Exploratory/Exp_UI/interface/drawing/fonts.py ──────────
import os
import blf
from .config import CUSTOM_FONT_PATH

_FONT_ID: int | None = None             # cache once per *file*

def _is_font_id_alive(fid: int) -> bool:
    """
    Quick probe: try to measure the width of a single space.
    If the underlying datablock was freed, blf.dimensions raises
    a RuntimeError, telling us the ID is toast.
    """
    try:
        blf.dimensions(fid, " ")
        return True
    except RuntimeError:
        return False

def reset_font():
    """Call this when a new .blend is loaded."""
    global _FONT_ID
    _FONT_ID = None                      # forces reload on next access

def get_font_id() -> int:
    """
    Returns a valid blf font‑ID at all times.
    Reloads automatically after a file change.
    """
    global _FONT_ID

    if _FONT_ID is not None and _is_font_id_alive(_FONT_ID):
        return _FONT_ID

    # Either first use *or* cached ID became invalid  (re‑)load
    if os.path.exists(CUSTOM_FONT_PATH):
        _FONT_ID = blf.load(CUSTOM_FONT_PATH)
    else:
        _FONT_ID = 0                     # Blender’s default

    return _FONT_ID
