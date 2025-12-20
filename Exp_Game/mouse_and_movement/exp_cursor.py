# File: Exploratory/Exp_Game/mouse_and_movement/exp_cursor.py

import bpy
import math
import sys
import atexit

_IS_MAC = (sys.platform == 'darwin')

# ─── Failsafe cursor restoration ───────────────────────────────────────────────
# Track whether cursor is currently captured so we can restore on crash
_cursor_captured = False

def _atexit_restore_cursor():
    """Last-resort cursor restoration when Python exits."""
    global _cursor_captured
    if _cursor_captured:
        try:
            release_cursor_clip()
            # Try to restore cursor in any available window
            if bpy.context and bpy.context.window:
                bpy.context.window.cursor_modal_restore()
        except:
            pass  # Best effort - might fail if Blender is shutting down
        _cursor_captured = False

# Register atexit handler
atexit.register(_atexit_restore_cursor)


def force_restore_cursor():
    """
    Force restore cursor visibility - call this on any failure path.
    Safe to call even if cursor wasn't captured.
    """
    global _cursor_captured
    release_cursor_clip()

    # Try to restore cursor in current context
    try:
        if bpy.context and bpy.context.window:
            bpy.context.window.cursor_modal_restore()
    except:
        pass

    # Also try all windows as fallback
    try:
        for window in bpy.context.window_manager.windows:
            try:
                window.cursor_modal_restore()
            except:
                pass
    except:
        pass

    _cursor_captured = False

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
        global _cursor_captured
        ctypes.windll.user32.ClipCursor(None)
        _cursor_captured = False

    # ─── Focus Detection (Windows) ──────────────────────────────────────────────
    _cached_blender_hwnd = None

    def cache_blender_hwnd():
        """
        Cache Blender's window handle at modal start.
        Call this once in invoke() before any focus checks.
        """
        global _cached_blender_hwnd

        # Try GetActiveWindow first (works when called from direct user action)
        _cached_blender_hwnd = ctypes.windll.user32.GetActiveWindow()

        # Fallback: If GetActiveWindow returns 0 (timer/callback context),
        # use GetForegroundWindow - Blender should be foreground during game start
        if _cached_blender_hwnd == 0:
            _cached_blender_hwnd = ctypes.windll.user32.GetForegroundWindow()
            if _cached_blender_hwnd != 0:
                print(f"[CURSOR_DEBUG] Cached Blender HWND via fallback: {_cached_blender_hwnd}")
            else:
                print("[CURSOR_DEBUG] WARNING: Could not get Blender HWND (both methods failed)")
        else:
            print(f"[CURSOR_DEBUG] Cached Blender HWND: {_cached_blender_hwnd}")

    def is_blender_focused() -> bool:
        """
        Check if Blender is the foreground window.
        Returns True if focused or if we can't determine (fail-safe).
        """
        global _cached_blender_hwnd
        if _cached_blender_hwnd is None or _cached_blender_hwnd == 0:
            return True  # Can't check, assume OK
        foreground = ctypes.windll.user32.GetForegroundWindow()
        return foreground == _cached_blender_hwnd

else:
    def confine_cursor_to_window(): pass
    def release_cursor_clip():
        global _cursor_captured
        _cursor_captured = False

    # ─── Focus Detection (Mac/Linux stubs) ──────────────────────────────────────
    def cache_blender_hwnd():
        """No-op on non-Windows platforms."""
        pass

    def is_blender_focused() -> bool:
        """Can't check on non-Windows, assume focused (fallbacks will catch issues)."""
        return True


# ─── Cross-Platform Delta Sanity Check ──────────────────────────────────────────

def is_delta_sane(last_x, last_y, current_x, current_y, threshold=300) -> bool:
    """
    Check if mouse movement delta is reasonable.
    Returns False if cursor jumped impossibly far (likely escaped and re-entered).

    Args:
        last_x, last_y: Previous mouse position
        current_x, current_y: Current mouse position
        threshold: Max reasonable delta in pixels (default 300)

    Returns:
        True if delta is sane, False if anomaly detected
    """
    if last_x is None or last_y is None:
        return True  # First move, can't check

    dx = abs(current_x - last_x)
    dy = abs(current_y - last_y)
    return dx < threshold and dy < threshold


def ensure_cursor_hidden_if_mac(context):
    """On macOS, defensively keep the cursor hidden while the modal is running."""
    if _IS_MAC:
        try:
            context.window.cursor_modal_set('NONE')
        except Exception:
            pass


def setup_cursor_region(context, operator):
    """
    Hide the system cursor, lock it in modal, and on Windows
    confine it to the Blender window. Initialize raw-delta state.
    """
    global _cursor_captured
    context.window.cursor_modal_set('NONE')
    confine_cursor_to_window()
    operator.last_mouse_x = None
    operator.last_mouse_y = None
    _cursor_captured = True  # Track that cursor is now hidden


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

    ensure_cursor_hidden_if_mac(context)
