#Exploratory/Exp_Game/reactions/exp_fonts.py

"""
exp_fonts.py – central registry for all “reaction” fonts.

• Any *.ttf placed in Exp_Game/reactions/reaction_fonts/ is auto‑discovered.
• discover_fonts()  → list of Enum items for a Blender EnumProperty.
• get_font_id_by_name(name) → blf font‑id (0 = Blender’s default).
• register() / unregister() attach Scene.custom_reaction_font automatically.
"""

from __future__ import annotations

import os
import blf
import bpy

# ──────────────────────────────────────────────────────────────────────────────
#  Constants & caches
# ──────────────────────────────────────────────────────────────────────────────
FONT_DIR = os.path.join(os.path.dirname(__file__), "reaction_fonts")
# Cache so each font loads only once per session
_font_cache: dict[str, tuple[int, str]] = {}           # {name: (font_id, path)}

# ──────────────────────────────────────────────────────────────────────────────
#  Public helpers
# ──────────────────────────────────────────────────────────────────────────────
def discover_fonts() -> list[tuple[str, str, str]]:
    """
    Scan FONT_DIR for *.ttf → return Enum items:
        (identifier, name_in_dropdown, description)
    The identifier is the file name without extension.
    """
    items: list[tuple[str, str, str]] = [
        ("DEFAULT", "Default", "Blender’s built‑in font")
    ]

    if not os.path.isdir(FONT_DIR):
        return items

    for fname in sorted(os.listdir(FONT_DIR)):
        if fname.lower().endswith(".ttf"):
            ident = os.path.splitext(fname)[0]          # “AstaSans‑Bold”
            items.append((ident, ident, f"Font file {fname}"))
    return items


def get_font_id_by_name(name: str) -> int:
    """
    • "DEFAULT" / empty → 0  (internal font)
    • otherwise → loads <name>.ttf once, returns its blf font‑id
    """
    if not name or name == "DEFAULT":
        return 0

    if name in _font_cache:
        return _font_cache[name][0]

    path = os.path.join(FONT_DIR, f"{name}.ttf")
    if not os.path.isfile(path):
        return 0                                         # graceful fallback

    font_id = blf.load(path)
    _font_cache[name] = (font_id, path)
    return font_id


def clear_font_cache():
    """
    Clear the font cache. Call on game end to release loaded fonts.
    Note: blf doesn't have an unload function, but clearing the cache
    allows fonts to be reloaded fresh on next game start.
    """
    _font_cache.clear()