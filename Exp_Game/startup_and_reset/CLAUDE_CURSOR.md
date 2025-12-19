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

---

## Proposed Solutions

### Solution A: Event-Gap Detection (Minimal, Cross-Platform)

**Concept:** Track time since last input event. If gap exceeds threshold, assume unfocused.

**Implementation:**
```python
# In ExpModal
_last_event_time: float = 0.0
_is_paused: bool = False
FOCUS_TIMEOUT = 0.5  # seconds

def modal(self, context, event):
    now = time.perf_counter()

    # Any event resets the timeout
    if event.type != 'TIMER':
        if self._is_paused:
            # Resuming from pause
            self._resume_from_pause(context)
        self._last_event_time = now
        self._is_paused = False

    # Check for focus timeout on TIMER
    if event.type == 'TIMER':
        if now - self._last_event_time > FOCUS_TIMEOUT:
            if not self._is_paused:
                self._pause_game(context)
            return {'RUNNING_MODAL'}  # Skip game logic while paused

def _pause_game(self, context):
    self._is_paused = True
    self.keys_pressed.clear()  # Clear stuck keys
    release_cursor_clip()
    # Optionally show "PAUSED" overlay

def _resume_from_pause(self, context):
    self._is_paused = False
    center_cursor_in_3d_view(context)
    setup_cursor_region(context, self)
```

**Pros:**
- Works on all platforms
- No OS-specific code needed
- Simple to implement

**Cons:**
- Can't distinguish "no movement" from "unfocused"
- 0.5s delay before detecting unfocus
- May pause unexpectedly if user holds still

---

### Solution B: OS Focus Check (Windows First)

**Concept:** Actively poll OS to check if Blender is focused.

**Implementation (exp_cursor.py):**
```python
import sys

if sys.platform == 'win32':
    import ctypes

    def is_window_focused() -> bool:
        """Check if Blender is the foreground window."""
        foreground = ctypes.windll.user32.GetForegroundWindow()
        # Get Blender's main window handle
        blender_hwnd = ctypes.windll.user32.GetActiveWindow()
        return foreground == blender_hwnd

    def get_blender_window_rect():
        """Get Blender window client rect in screen coords."""
        hwnd = ctypes.windll.user32.GetActiveWindow()
        rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
        pt = ctypes.wintypes.POINT(rect.left, rect.top)
        ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
        return (pt.x, pt.y, pt.x + rect.right, pt.y + rect.bottom)

elif sys.platform == 'darwin':
    def is_window_focused() -> bool:
        """Check if Blender is active on Mac."""
        try:
            from AppKit import NSApplication
            return NSApplication.sharedApplication().isActive()
        except ImportError:
            return True  # Assume focused if can't check

else:  # Linux
    def is_window_focused() -> bool:
        return True  # TODO: X11/Wayland implementation
```

**In ExpModal:**
```python
def modal(self, context, event):
    # Check focus every TIMER tick
    if event.type == 'TIMER':
        if not is_window_focused():
            if not self._is_paused:
                self._pause_game(context)
            return {'RUNNING_MODAL'}
        else:
            if self._is_paused:
                self._resume_from_pause(context)
```

**Pros:**
- Immediate detection on Windows
- No false positives from user holding still

**Cons:**
- OS-specific code paths
- Mac requires pyobjc (may not be available)
- Linux not implemented

---

### Solution C: Hybrid (Recommended)

**Concept:** Combine event-gap detection with OS focus check where available.

```python
def _check_focus_state(self, context) -> bool:
    """Returns True if game should be running, False if paused."""

    # Primary: OS-level check (fast, accurate)
    if sys.platform == 'win32':
        if not is_window_focused():
            return False

    # Secondary: Event gap detection (fallback for Mac/Linux)
    if time.perf_counter() - self._last_event_time > FOCUS_TIMEOUT:
        return False

    # Tertiary: Mouse bounds check
    if self._last_mouse_region_valid is False:
        return False

    return True
```

---

### Solution D: Blender GRAB_CURSOR Reliance

**Concept:** Trust Blender's built-in `bl_options = {'GRAB_CURSOR'}` and add recovery hooks.

The modal already has `'GRAB_CURSOR'` which tells Blender to handle cursor grabbing. The issue is Blender's implementation may release grab on focus loss without notification.

**Implementation:**
```python
def modal(self, context, event):
    # Detect if cursor grab was lost (Blender-specific)
    if event.type == 'MOUSEMOVE':
        region = context.region
        if region:
            # Check if mouse is way outside expected region
            margin = 100
            if (event.mouse_region_x < -margin or
                event.mouse_region_x > region.width + margin or
                event.mouse_region_y < -margin or
                event.mouse_region_y > region.height + margin):
                # Cursor escaped! Attempt recovery
                self._attempt_cursor_recovery(context)
```

---

## Files to Modify

When implementing a solution, these files will need changes:

| File | Changes Needed |
|------|----------------|
| `exp_cursor.py` | Add `is_window_focused()`, focus recovery functions |
| `exp_modal.py` | Add pause/resume state, focus check in modal loop |
| `exp_loop.py` | Skip game logic when paused |
| `exp_startup.py` | Possibly adjust cursor centering for recovery |

---

## Testing Checklist

When implementing, test these scenarios:

- [ ] Alt-tab away and back (Windows)
- [ ] Click on another window and back
- [ ] System notification appears
- [ ] Drag window to another monitor
- [ ] Windows lock screen and unlock
- [ ] Blender goes to background briefly
- [ ] Hold still for 10+ seconds (shouldn't false-pause)
- [ ] Rapid alt-tab spam
- [ ] Start game from N-panel (local)
- [ ] Start game from download code (UI)
- [ ] Mac: Same scenarios
- [ ] Linux: Same scenarios (if supported)

---

## Current Status

**NOT IMPLEMENTED** - This document describes the problem and proposed solutions. Implementation is pending.

Last updated: 2024
