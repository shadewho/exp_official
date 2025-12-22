# Cursor & Window Focus System - Architecture & Issues

## Overview

This document describes the cursor capture, window focus, and input handling system for the Exploratory game modal. The system has critical gaps that cause users to lose cursor control, requiring Blender restarts.

---

## Startup Flows

There are TWO entry points that both ultimately call `EXP_GAME_OT_StartGame`:

### Flow A: Local Game Start (N-Panel)

User clicks "Start Game" in the Blender N-panel sidebar.

```
User clicks "Start Game" button
    │
    ▼
EXP_GAME_OT_StartGame.execute()          [exp_startup.py:425]
    │
    ├── launched_from_ui = False
    ├── enter_fullscreen_once()           [exp_fullscreen.py:122]
    │       └── Toggles fullscreen OR just hides UI if already fullscreen
    │
    └── timer(0.2s) → invoke_modal_in_current_view3d()
                            │
                            ▼
                      ExpModal.invoke()   [exp_modal.py:305]
                            │
                            ├── setup_cursor_region()      [exp_cursor.py:99]
                            │       ├── cursor_modal_set('NONE')
                            │       └── confine_cursor_to_window() [Windows only]
                            │
                            ├── center_cursor_in_3d_view() [exp_startup.py:14]
                            ├── _bind_view3d_once()        [exp_view_helpers.py:94]
                            ├── modal_handler_add()
                            └── timer(1/30s) starts game loop
```

### Flow B: UI Download & Explore (Download Code)

User enters a download code in the Exploratory UI panel, downloads a world, then plays it.

```
User enters download code → clicks Search
    │
    ▼
WEBAPP_OT_ShowDetailByCode.execute()     [Exp_UI/interface/operators/display.py:242]
    │
    ├── Fetches world metadata from server
    ├── Downloads thumbnail
    └── Opens detail overlay (PACKAGE_OT_Display)
            │
            ▼
      User clicks "Explore" button in overlay
            │
            ▼
      explore_icon_handler()              [Exp_UI/download_and_explore/explore_main.py]
            │
            ├── Stores original scene name: wm['original_scene']
            ├── Downloads .blend file from server (async thread)
            ├── timer_finish_download() polls until complete
            │       │
            │       ├── validate_blend_header()        [Security: magic bytes]
            │       ├── scan_blend_for_scripts()       [Security: text datablocks]
            │       ├── bpy.data.libraries.load()      [Verify valid .blend]
            │       └── append_scene_from_blend()      [Load scene into Blender]
            │
            └── On success: bpy.ops.exploratory.start_game(launched_from_ui=True)
                            │
                            ▼
                      EXP_GAME_OT_StartGame.execute()  [exp_startup.py:425]
                            │
                            ├── launched_from_ui = True
                            ├── disable_live_perf_overlay_next_tick()
                            ├── enter_fullscreen_once()
                            │
                            └── timer(0.4s) → invoke_modal_in_current_view3d()
                                                │
                                                ▼
                                          ExpModal.invoke()
                                          [Same as Flow A from here]
```

**Key Difference:** `launched_from_ui=True` triggers:
- Longer delay (0.4s vs 0.2s) before modal starts
- Scene cleanup on game exit (reverts to original scene)
- Downloaded .blend file deletion on exit

---

## Current Cursor/Input Architecture

### Files Involved

| File | Purpose |
|------|---------|
| `Exp_Game/mouse_and_movement/exp_cursor.py` | OS-level cursor control |
| `Exp_Game/startup_and_reset/exp_startup.py` | Game startup, cursor centering |
| `Exp_Game/startup_and_reset/exp_fullscreen.py` | Fullscreen toggle, UI hiding |
| `Exp_Game/modal/exp_modal.py` | Main game loop, event handling |
| `Exp_Game/modal/exp_view_helpers.py` | View3D binding and validation |
| `Exp_Game/modal/exp_loop.py` | Per-frame game logic orchestration |

### exp_cursor.py - Core Functions

```python
# Windows-only cursor confinement
def confine_cursor_to_window():
    """Uses ClipCursor Win32 API to trap mouse in Blender window."""
    hwnd = ctypes.windll.user32.GetActiveWindow()
    rect = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
    # Convert client coords to screen coords...
    ctypes.windll.user32.ClipCursor(ctypes.byref(rect))

def release_cursor_clip():
    """Releases ClipCursor confinement."""
    ctypes.windll.user32.ClipCursor(None)

# Cross-platform
def setup_cursor_region(context, operator):
    """Called on modal invoke - hides cursor, sets up tracking."""
    context.window.cursor_modal_set('NONE')
    confine_cursor_to_window()  # No-op on Mac/Linux
    operator.last_mouse_x = None
    operator.last_mouse_y = None
    _cursor_captured = True

def handle_mouse_move(operator, context, event):
    """Processes raw mouse deltas for camera yaw/pitch."""
    # First move: just record position
    if operator.last_mouse_x is None:
        operator.last_mouse_x = event.mouse_x
        operator.last_mouse_y = event.mouse_y
        return

    # Compute delta from last position
    dx = event.mouse_x - operator.last_mouse_x
    dy = event.mouse_y - operator.last_mouse_y

    operator.yaw -= dx * operator.sensitivity
    operator.pitch -= dy * operator.sensitivity
    # Clamp pitch...

    operator.last_mouse_x = event.mouse_x
    operator.last_mouse_y = event.mouse_y

def force_restore_cursor():
    """Emergency cursor restore - call on any failure path."""
    release_cursor_clip()
    # Try to restore cursor in all windows...
```

### ExpModal - Event Loop

```python
class ExpModal(bpy.types.Operator):
    bl_options = {'BLOCKING', 'GRAB_CURSOR'}  # Blender's built-in cursor grab

    def modal(self, context, event):
        if event.type == 'TIMER':
            self._loop.on_timer(context)  # Game logic
            return {'RUNNING_MODAL'}

        elif event.type == 'MOUSEMOVE':
            handle_mouse_move(self, context, event)
            return {'RUNNING_MODAL'}

        elif event.type in {key_bindings...}:
            self.handle_key_input(event)

        return {'RUNNING_MODAL'}
```

---

## The Problem

### Symptoms

1. **Cursor escapes window** - User alt-tabs or clicks outside Blender, cursor is no longer confined
2. **Infinite mouse breaks** - Raw delta calculation produces huge jumps when cursor re-enters
3. **Stuck keys** - Keys held when focus lost stay in `keys_pressed` set forever
4. **Input stops working** - Mouse/keyboard events stop being processed
5. **Can't click anything** - Cursor is hidden but not captured, user is blind
6. **Must restart Blender** - No recovery path, only hard restart fixes it

### Root Causes

#### 1. No Window Focus Detection

Blender's Python API does NOT provide window focus events. The system has no way to know when:
- User alt-tabs away
- User clicks on another application
- Another window covers Blender
- System dialog appears (e.g., Windows Update)

#### 2. ClipCursor Only Works When Active (Windows)

```python
hwnd = ctypes.windll.user32.GetActiveWindow()
```

`GetActiveWindow()` returns the active window in the CURRENT THREAD. If Blender loses focus, `ClipCursor` silently fails or is ignored by Windows.

#### 3. No Pause State

The game loop continues running physics, animations, etc. even when window is unfocused. There's no concept of "paused" state.

#### 4. No Cursor Recovery

Once the cursor escapes:
- `last_mouse_x/y` are stale (last known position)
- Next MOUSEMOVE could have huge delta (cursor moved across screen)
- No re-centering mechanism during gameplay

#### 5. Mac Has No Confinement

Mac only hides the cursor (`cursor_modal_set('NONE')`), it doesn't confine it. The cursor can physically leave the window bounds.

#### 6. Linux Not Handled

No Linux-specific cursor confinement. Falls through to no-op.

---

## OS-Specific Considerations

### Windows

**Available APIs:**
```python
import ctypes
from ctypes import wintypes

# Check if Blender is the foreground window
def is_blender_focused():
    foreground = ctypes.windll.user32.GetForegroundWindow()
    blender = ctypes.windll.user32.GetActiveWindow()
    return foreground == blender

# Re-confine cursor (call when focus regained)
def reconfine_cursor():
    confine_cursor_to_window()

# Warp cursor to specific screen position
def warp_cursor_to(x, y):
    ctypes.windll.user32.SetCursorPos(x, y)
```

**Caveats:**
- `GetActiveWindow()` returns window active in current thread, not necessarily focused
- `GetForegroundWindow()` returns the actual focused window across all processes
- `ClipCursor` is advisory - other apps can override it
- High-DPI scaling can cause coordinate mismatches

### macOS

**Available APIs:**
```python
# Requires pyobjc or ctypes to Cocoa
# Option 1: pyobjc (if available)
from AppKit import NSApplication
def is_blender_focused():
    return NSApplication.sharedApplication().isActive()

# Option 2: CGEvent for cursor warping
from Quartz import CGWarpMouseCursorPosition, CGPoint
def warp_cursor_to(x, y):
    CGWarpMouseCursorPosition(CGPoint(x, y))
```

**Caveats:**
- pyobjc may not be bundled with Blender's Python
- CGEventPost requires accessibility permissions
- macOS 10.15+ has stricter input control restrictions

### Linux (X11/Wayland)

**X11:**
```python
# Requires python-xlib
from Xlib import X, display
# XGrabPointer for cursor grab
# XWarpPointer for cursor warp
```

**Wayland:**
- Explicit cursor confinement protocol exists but client support varies
- No direct cursor warping (security feature)

**Caveats:**
- Wayland is increasingly common (Ubuntu 22.04+, Fedora default)
- X11 APIs don't work on Wayland
- May need to detect display server type