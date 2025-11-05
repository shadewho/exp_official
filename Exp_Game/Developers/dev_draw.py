from __future__ import annotations
import bpy
from .dev_registry import REGISTRY
from .dev_draw_prims import corner_xy, draw_rect
from .dev_state import STATE

def draw_2d(BUS):
    scene = bpy.context.scene
    if not (scene and getattr(scene, "dev_hud_enable", False)):
        return

    try:
        area = next((a for a in bpy.context.screen.areas if a.type == 'VIEW_3D'), None)
        if not area: return
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if not region: return
    except Exception:
        return

    scale = max(1, int(scene.dev_hud_scale))
    lh    = 16 * scale
    pad   = 12 * scale
    gap   = 22 * scale
    col_w_left  = 460 * scale
    col_w_right = 460 * scale
    box_w = col_w_left + gap + col_w_right

    hL = 0
    for sec in REGISTRY.active(scene, "LEFT"):
        hL += int(sec.measure(scene, STATE, BUS, scale, lh, col_w_left))

    hR = 0
    for sec in REGISTRY.active(scene, "RIGHT"):
        hR += int(sec.measure(scene, STATE, BUS, scale, lh, col_w_right))

    box_h = max(hL, hR) + 2*pad
    x0, y0 = corner_xy(region.width, region.height, box_w, box_h, scene.dev_hud_position, pad)

    draw_rect(x0-4, y0-4, box_w+8, box_h+8, (0.07, 0.08, 0.10, 0.92))
    draw_rect(x0-4, y0-4, box_w+8, 1, (0.22,0.26,0.32,1.0))
    draw_rect(x0-4, y0-4+box_h+8-1, box_w+8, 1, (0.22,0.26,0.32,1.0))

    xL = x0 + 8*scale
    xR = x0 + col_w_left + gap + 8*scale
    y_top = y0 + box_h - lh

    yL = y_top + int(0.3*lh)
    yR = y_top

    for sec in REGISTRY.active(scene, "LEFT"):
        yL = sec.draw(xL, yL, scene, STATE, BUS, scale, lh, col_w_left)

    for sec in REGISTRY.active(scene, "RIGHT"):
        yR = sec.draw(xR, yR, scene, STATE, BUS, scale, lh, col_w_right)
