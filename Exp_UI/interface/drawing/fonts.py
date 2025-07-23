"""
fonts.py – minimal, cross‑platform font loader for the UI.

• Loads the single Sono‑Medium.ttf shipped with the add‑on.
• If the file is missing, uses Blender’s default font (ID 0).
• No platform‑specific logic, no hard‑coded slashes.

Import get_font_id() wherever you call blf.size / blf.dimensions / blf.draw.
"""

import os
import blf
from .config import CUSTOM_FONT_PATH   # path is already built with os.path.join

# Cached ID so we load only once
_FONT_ID: int | None = None


def get_font_id() -> int:
    """
    Return the blf font‑ID that every UI draw call should use.
    Loads Sono‑Medium.ttf once per Blender session.
    """
    global _FONT_ID
    if _FONT_ID is not None:
        return _FONT_ID

    if os.path.exists(CUSTOM_FONT_PATH):
        _FONT_ID = blf.load(CUSTOM_FONT_PATH)
    else:
        # Bundled file missing – fall back to Blender’s default
        _FONT_ID = 0

    return _FONT_ID
