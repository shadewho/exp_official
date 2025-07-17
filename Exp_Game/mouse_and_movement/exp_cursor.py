# Exploratory/mouse_and_movement/exp_cursor.py
# Optimised for Blender 4.5+ raw-input; safe fallback for ≤ 4.4

# Exploratory/mouse_and_movement/exp_cursor.py
# Blender 4.5+ only – uses relative-pointer events (no fallback paths)

import bpy
import math


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


# ─────────────────────────────────────────────
# Setup – call from ExpModal.invoke()
# ─────────────────────────────────────────────
def setup_cursor_region(ctx, op):
    """
    • Locates the VIEW_3D window region.
    • Records its centre (op._cx / op._cy).
    • Hides the OS cursor.
    """
    ctx.window.cursor_modal_set('HAND')  # invisible cursor; change to 'HAND' if desired

    for area in ctx.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        for region in area.regions:
            if region.type == 'WINDOW':
                op._reg = region
                op._cx  = region.x + region.width  // 2
                op._cy  = region.y + region.height // 2
                return
    raise RuntimeError("VIEW_3D WINDOW region not found")


# ─────────────────────────────────────────────
# Mouse-move handler – call from ExpModal.modal()
# ─────────────────────────────────────────────
def handle_mouse_move(op, ctx, evt):
    """
    Raw relative-pointer events (Blender 4.5+):
      • Apply deltas to yaw/pitch.
      • Clamp pitch.
      • Update camera immediately.
      • Warp OS cursor back to centre so the arrow never drifts.
    """
    if evt.is_mouse_absolute:
        # Raw input disabled – nothing to do (addon targets 4.5+ raw mode only).
        return

    # Relative motion (pixels since previous event)
    dx = evt.mouse_x - evt.mouse_prev_x
    dy = evt.mouse_y - evt.mouse_prev_y

    # Apply to camera orientation
    op.yaw   -= dx * op.sensitivity
    op.pitch -= dy * op.sensitivity
    op.pitch  = _clamp(op.pitch, -math.pi / 2 + 0.1,  math.pi / 2 - 0.1)

    # Keep OS pointer fixed in the window centre
    ctx.window.cursor_warp(op._cx, op._cy)
