# Exploratory/Exp_Game/reactions/exp_crosshairs.py
import bpy, time, math
import gpu
from gpu_extras.batch import batch_for_shader

_draw_handle = None
_params = None
_end_time = None

def enable_crosshairs(
    length: int = 12,
    gap: int = 6,
    thickness: int = 2,
    dot_radius: int = 0,
    color=(1.0, 1.0, 1.0, 0.85),
    style: str = "PLUS",            # "PLUS" | "PLUS_DOT" | "X" | "X_DOT"
    duration: float | None = None   # None => indefinite
):
    """Create/refresh an always-centered, pixel-based crosshair overlay."""
    global _params, _end_time, _draw_handle
    _params = {
        "length":      max(0, int(length)),
        "gap":         max(0, int(gap)),
        "thickness":   max(1, int(thickness)),
        "dot_radius":  max(0, int(dot_radius)),
        "color":       tuple(color),
        "style":       style if style in {"PLUS", "PLUS_DOT", "X", "X_DOT"} else "PLUS",
    }
    _end_time = None if (duration is None or duration <= 0.0) else (time.monotonic() + float(duration))

    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_PIXEL')

    _tag_all_view3d_for_redraw()


def disable_crosshairs():
    """Remove the overlay and cleanup."""
    global _draw_handle, _params, _end_time
    if _draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        except Exception:
            pass
    _draw_handle = None
    _params = None
    _end_time = None
    _tag_all_view3d_for_redraw()


def is_enabled() -> bool:
    return _draw_handle is not None


# ---------------- Internals ----------------

def _tag_all_view3d_for_redraw():
    """ONLY tag redraws. Do NOT call bpy.ops.* here (it can un-hide the cursor)."""
    wm = getattr(bpy.context, "window_manager", None)
    if not wm:
        return
    for win in wm.windows:
        scr = win.screen
        if not scr:
            continue
        for area in scr.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
                for reg in area.regions:
                    if reg.type == 'WINDOW':
                        reg.tag_redraw()


def _find_window_region():
    """Return (region, area) for the first VIEW_3D WINDOW region in the active window."""
    win = bpy.context.window
    if not win or not win.screen:
        return None, None
    for area in win.screen.areas:
        if area.type == 'VIEW_3D':
            for reg in area.regions:
                if reg.type == 'WINDOW':
                    return reg, area
    return None, None


def _quad_for_segment(ax, ay, bx, by, half_t):
    """
    Build a 4-vertex quad for a thick line segment from A to B with half thickness 'half_t'.
    Returns list of 4 (x,y) tuples in TRI_FAN order. If degenerate, returns None.
    """
    dx = bx - ax
    dy = by - ay
    length = (dx * dx + dy * dy) ** 0.5
    if length <= 1e-6:
        return None
    # Perpendicular normal (pixel space)
    nx = -dy / length
    ny =  dx / length

    ox = nx * half_t
    oy = ny * half_t

    # Quad corners (fan order)
    return [
        (ax - ox, ay - oy),
        (ax + ox, ay + oy),
        (bx + ox, by + oy),
        (bx - ox, by - oy),
    ]


def _draw():
    global _params, _end_time

    if not _params:
        return

    # Auto-expire
    if _end_time is not None and time.monotonic() >= _end_time:
        disable_crosshairs()
        return

    region, _area = _find_window_region()
    if not region:
        return

    w = int(region.width)
    h = int(region.height)
    if w <= 0 or h <= 0:
        return

    # Center in pixels
    cx = float(w // 2)
    cy = float(h // 2)

    L  = float(int(_params["length"]))
    G  = float(int(_params["gap"]))
    T  = float(max(1, int(_params["thickness"])))
    D  = int(_params["dot_radius"])
    col = tuple(_params["color"])
    style = _params["style"]

    # Build segments
    segs = []
    if style in {"PLUS", "PLUS_DOT"}:
        segs += [((cx, cy + G),     (cx, cy + G + L))]
        segs += [((cx, cy - G),     (cx, cy - G - L))]
        segs += [((cx + G, cy),     (cx + G + L, cy))]
        segs += [((cx - G, cy),     (cx - G - L, cy))]
    if style in {"X", "X_DOT"}:
        segs += [((cx + G, cy + G), (cx + G + L, cy + G + L))]
        segs += [((cx - G, cy - G), (cx - G - L, cy - G - L))]
        segs += [((cx + G, cy - G), (cx + G + L, cy - G - L))]
        segs += [((cx - G, cy + G), (cx - G - L, cy + G + L))]

    try:
        gpu.state.blend_set('ALPHA')
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')

        # Segments as quads
        half_t = T * 0.5
        for (ax, ay), (bx, by) in segs:
            quad = _quad_for_segment(float(ax), float(ay), float(bx), float(by), half_t)
            if not quad:
                continue
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": quad})
            shader.bind()
            shader.uniform_float('color', col)
            batch.draw(shader)

        # Optional center dot
        if D > 0 or "DOT" in style:
            R = float(max(D, 2))
            n = 28
            circle = [(cx + R * math.cos(i * 2*math.pi / n),
                       cy + R * math.sin(i * 2*math.pi / n)) for i in range(n)]
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": circle})
            shader.bind()
            shader.uniform_float('color', col)
            batch.draw(shader)

    except Exception as e:
        # Fail safe instead of unwinding the modal
        try:
            print(f"[Crosshairs] draw error: {e}")
        except Exception:
            pass
        disable_crosshairs()
    finally:
        try:
            gpu.state.blend_set('NONE')
        except Exception:
            pass


def execute_crosshairs_reaction(r):
    """
    Enable a pixel-perfect crosshair overlay via reaction parameters.
    """
    duration = None if getattr(r, "crosshair_indefinite", True) else float(getattr(r, "crosshair_duration", 0.0))
    enable_crosshairs(
        length     = getattr(r, "crosshair_length_px",    12),
        gap        = getattr(r, "crosshair_gap_px",       6),
        thickness  = getattr(r, "crosshair_thickness_px", 2),
        dot_radius = getattr(r, "crosshair_dot_radius_px", 0),
        color      = tuple(getattr(r, "crosshair_color", (1.0, 1.0, 1.0, 0.85))),
        style      = getattr(r, "crosshair_style", "PLUS"),
        duration   = duration,
    )