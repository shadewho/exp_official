# File: Exploratory/Exp_Game/mouse_and_movement/exp_cursor.py

import bpy
import math
import sys

# ─── Windows-only cursor confinement ────────────────────────────────────────────
if sys.platform == 'win32':
    import ctypes
    import ctypes.wintypes

    def confine_cursor_to_window():
        hwnd = ctypes.windll.user32.GetActiveWindow()
        rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
        # client→screen top-left
        pt = ctypes.wintypes.POINT(rect.left, rect.top)
        ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
        rect.left, rect.top = pt.x, pt.y
        # client→screen bottom-right
        pt = ctypes.wintypes.POINT(rect.right, rect.bottom)
        ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
        rect.right, rect.bottom = pt.x, pt.y
        ctypes.windll.user32.ClipCursor(ctypes.byref(rect))

    def release_cursor_clip():
        ctypes.windll.user32.ClipCursor(None)

else:
    def confine_cursor_to_window(): pass
    def release_cursor_clip():    pass


def setup_cursor_region(context, operator):
    """
    Hide the system cursor, lock it in modal, and on Windows
    confine it to the Blender window. Initialize raw-delta state.
    """
    context.window.cursor_modal_set('NONE')
    confine_cursor_to_window()
    operator.last_mouse_x = None
    operator.last_mouse_y = None


def handle_mouse_move(operator, context, event):
    """
    Record the first MOUSEMOVE, then compute dx/dy on each move,
    apply yaw/pitch, clamp pitch, and update last_mouse_x/_y.
    """
    if operator.last_mouse_x is None or operator.last_mouse_y is None:
        operator.last_mouse_x = event.mouse_x
        operator.last_mouse_y = event.mouse_y
        return

    dx = event.mouse_x - operator.last_mouse_x
    dy = event.mouse_y - operator.last_mouse_y

    operator.yaw   -= dx * operator.sensitivity
    operator.pitch -= dy * operator.sensitivity

    operator.pitch = max(-math.pi/2 + 0.1,
                         min(math.pi/2 - 0.1, operator.pitch))

    operator.last_mouse_x = event.mouse_x
    operator.last_mouse_y = event.mouse_y
