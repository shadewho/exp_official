# Exp_Game/Developers/dev_draw.py
from __future__ import annotations
import bpy
from .dev_registry import REGISTRY
from .dev_state import STATE
from .dev_draw_prims import draw_text, draw_rect

# Semi-opaque grey like the original global panel
_PANEL_BG = (0.10, 0.12, 0.15, 0.85)

# UI scale mapping (scene.dev_hud_scale = 1..5 -> visual factor)
# 1 is deliberately much smaller than before.
_SCALE_MAP = {1: 0.60, 2: 0.80, 3: 1.00, 4: 1.25, 5: 1.60}

def _scale_factor(sval: int) -> float:
    try:
        v = int(sval)
    except Exception:
        v = 3
    return float(_SCALE_MAP.get(max(1, min(5, v)), 1.0))

def _region_size():
    r = getattr(bpy.context, "region", None)
    w = int(getattr(r, "width", 0) or 1920)
    h = int(getattr(r, "height", 0) or 1080)
    return w, h

def _measure_and_pack(scene, BUS, sf: float, lh: int, col_w: int, cols: int):
    """
    Returns (col_lists, heights):
      col_lists: list per column with [(section, measured_h), ...]
      heights:   accumulated height per column
    """
    # Gather active sections, keep declared order
    secs = list(REGISTRY.active(scene, "LEFT")) + list(REGISTRY.active(scene, "RIGHT"))
    secs.sort(key=lambda s: int(getattr(s, "order", 0)))

    measured = []
    for s in secs:
        try:
            h = int(max(0, int(s.measure(scene, STATE, BUS, sf, lh, col_w))))
            if h > 0:
                measured.append((s, h))
        except Exception:
            continue

    # Pinned sections (e.g. graphs) go in column 0
    pinned = [(s, h) for (s, h) in measured if bool(getattr(s, "sticky_left", False))]
    rest   = [(s, h) for (s, h) in measured if not bool(getattr(s, "sticky_left", False))]

    col_lists = [[] for _ in range(cols)]
    heights   = [0 for _ in range(cols)]

    for s, h in pinned:
        col_lists[0].append((s, h))
        heights[0] += h

    # Greedy height balancing across remaining columns
    for s, h in rest:
        idx = min(range(cols), key=lambda i: heights[i])
        col_lists[idx].append((s, h))
        heights[idx] += h

    return col_lists, heights

def draw_2d(BUS):
    scn = bpy.context.scene
    if not (scn and getattr(scn, "dev_hud_enable", False)):
        return

    # --- Layout parameters (fixed 3 columns, dynamic packing) ---
    sf   = _scale_factor(getattr(scn, "dev_hud_scale", 3))
    lh   = int(round(14 * sf))
    cols = 3  # force up to 3 columns (no property anymore)

    gap      = int(round(8  * sf))
    margin   = int(round(10 * sf))
    header_h = int(round(16 * sf))
    pad_top  = int(round(6  * sf))
    pad_bot  = int(round(8  * sf))

    W, H = _region_size()

    # Column width fills available width with sane bounds
    avail_w = max(1, W - 2 * margin - (cols - 1) * gap)
    col_w   = max(int(round(220 * sf)), min(int(round(480 * sf)), avail_w // cols))
    total_w = cols * col_w + (cols - 1) * gap

    # Measure + pack (graphs are pinned-left via section.sticky_left=True)
    col_lists, heights = _measure_and_pack(scn, BUS, sf, lh, col_w, cols)
    tallest = max(heights) if heights else 0

    # Overall panel height (title line + tallest column + interior padding)
    panel_inner_h = header_h + tallest
    panel_h       = panel_inner_h + pad_top + pad_bot

    # Corner anchoring (single panel; grey background restored)
    pos = str(getattr(scn, "dev_hud_position", "TR"))
    left_x = margin if pos in ("TL", "BL") else max(0, W - margin - total_w)
    top_y  = (H - margin) if pos in ("TR", "TL") else (margin + panel_h)
    bottom_y = top_y - panel_h

    # Draw unified background
    draw_rect(left_x, bottom_y, total_w, panel_h, _PANEL_BG)

    # Title row
    y_title = top_y - pad_top
    draw_text(left_x, y_title, "DEVELOPER HUD", max(12, int(round(14 * sf))))
    y_after_header = y_title - header_h

    # Columns
    for ci in range(cols):
        x = left_x + ci * (col_w + gap)
        y = y_after_header
        for (section, _h) in col_lists[ci]:
            try:
                y = section.draw(x, y, scn, STATE, BUS, sf, lh, col_w)
            except Exception:
                continue
