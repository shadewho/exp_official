#Exp_UI/interface/drawing/utilities.py

import bpy
import blf
import gpu
from .config import LAST_VIEWPORT_SIZE, TEMPLATE_ASPECT_RATIO, OFFSET_FACTOR
from datetime import datetime, timezone
from gpu_extras.batch import batch_for_shader
from .fonts import get_font_id 
import time                        # <-- NEW
from . import animated_sequence    # <-- NEW (sits beside explore_loading.py)
# ------------------------------------------------------------------------
# Text Drawing Utilities
# ------------------------------------------------------------------------

CUSTOM_FONT_ID = get_font_id()

def build_text_item(
    text, 
    x, 
    y, 
    size=16, 
    color=(1.0, 1.0, 1.0, 1.0), 
    alignment='LEFT',
    multiline=True
):
    """
    Build a dictionary that describes how and where to draw text.

    :param text:       The string to draw (can include \n for multiple lines).
    :param x, y:       Position in the 3D View region coordinates.
    :param size:       Font size in points.
    :param color:      RGBA tuple (0.0 - 1.0).
    :param alignment:  'LEFT', 'CENTER', or 'RIGHT'.
    :param multiline:  If True, split on \n and draw multiple lines.
    :return:           A dict that the draw callback can interpret.
    """
    return {
        "type": "text",
        "content": text,
        "position": (x, y),
        "size": size,
        "color": color,
        "alignment": alignment,
        "multiline": multiline,
        "font_id":   get_font_id(),
    }

# ----------------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------------

def calculate_free_space():
    """
    Calculate the available space in the current 3D View.
    We use the current region’s dimensions directly.
    (Assuming context.region is the main drawing area with (0,0) in its lower left.)
    Also subtract the widths of any side panels (TOOLS and UI regions) from the horizontal space.
    """
    context = bpy.context
    # Check that context.area and context.region exist.
    if not context.area or not context.region:
        return None

    # Use the current region (the drawing area)
    region = context.region
    free_space = {
        "x": 0,
        "y": 0,
        "width": region.width,
        "height": region.height
    }
    
    # Subtract side panels from horizontal space (if they exist)
    for reg in context.area.regions:
        if reg.type in {'TOOLS', 'UI'}:
            # If the panel is on the left (x == 0), shift x and reduce width.
            if reg.x == 0:
                free_space["x"] += reg.width
                free_space["width"] -= reg.width
            # If on the right, simply reduce the width.
            elif reg.x > 0:
                free_space["width"] -= reg.width
    return free_space


def calculate_template_position(free_space, aspect_ratio=TEMPLATE_ASPECT_RATIO, offset_factor=OFFSET_FACTOR, buffer_ratio_x=0.05, buffer_ratio_y=0.05):
    """
    Given a free_space dictionary with keys "x", "y", "width", and "height",
    this function calculates the largest possible template rectangle that:
      1. Preserves the desired aspect ratio.
      2. Is centered in an "effective" space that is the original free space
         with a buffer on all sides (specified by buffer_ratio_x and buffer_ratio_y).
      3. Then applies a vertical offset (by offset_factor * total height) and
         clamps the template so it remains entirely onscreen.
    
    The parameters buffer_ratio_x and buffer_ratio_y are the percentages of the
    total width and height (respectively) to leave as margins on both sides.
    """
    # Unpack the original available space.
    base_x = free_space["x"]
    base_y = free_space["y"]
    total_w = free_space["width"]
    total_h = free_space["height"]

    # Compute buffer amounts.
    buffer_x = total_w * buffer_ratio_x  # margin on left and right
    buffer_y = total_h * buffer_ratio_y  # margin on top and bottom

    # Define the effective available space for the template.
    effective_w = total_w - 2 * buffer_x
    effective_h = total_h - 2 * buffer_y

    # Compute the maximum template size that preserves the aspect ratio and fits in the effective space.
    desired_w = effective_w
    desired_h = effective_w / aspect_ratio
    if desired_h > effective_h:
        desired_h = effective_h
        desired_w = effective_h * aspect_ratio

    # Center the template within the effective space.
    x = base_x + buffer_x + (effective_w - desired_w) / 2
    y = base_y + buffer_y + (effective_h - desired_h) / 2

    # Apply vertical offset: subtract (offset_factor * total_h) from y.
    proposed_y = y - (total_h * offset_factor)
    # Clamp y so that the template remains fully within the original free space (with buffers).
    y = max(base_y + buffer_y, min(proposed_y, base_y + total_h - buffer_y - desired_h))

    return {
        "x1": x,
        "y1": y,
        "x2": x + desired_w,
        "y2": y + desired_h,
    }



def viewport_changed():
    """
    Return True if the VIEW_3D area size changed since last check.
    If no valid viewport is available, returns False.
    """
    global LAST_VIEWPORT_SIZE
    free_space = calculate_free_space()
    if free_space is None:
        # No valid viewport information available.
        return False

    w, h = free_space["width"], free_space["height"]

    if (w, h) != (LAST_VIEWPORT_SIZE["width"], LAST_VIEWPORT_SIZE["height"]):
        LAST_VIEWPORT_SIZE["width"] = w
        LAST_VIEWPORT_SIZE["height"] = h
        return True
    return False


# -------------------------------------------------------------------
# Time Utils
# -------------------------------------------------------------------

def format_relative_time(upload_date_str):
    """
    Converts an upload date string to a relative format.
    Example outputs:
      - "50s" for 50 seconds ago
      - "2m" for 2 minutes ago
      - "3h" for 3 hours ago
      - "5d" for 5 days ago
      - "1y" for 1 year ago

    This function assumes the upload_date_str is in ISO 8601 format.
    Adjust the parsing logic if your format differs.
    """
    try:
        # Try ISO format first
        dt = datetime.fromisoformat(upload_date_str)
    except Exception:
        # Fallback: try a common format (adjust as needed)
        try:
            dt = datetime.strptime(upload_date_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            # If parsing fails, return the original string
            return upload_date_str

    # Ensure dt is timezone aware (assume UTC if not provided)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    diff = now - dt
    seconds = diff.total_seconds()

    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h"
    elif seconds < 31536000:  # less than a year
        days = int(seconds / 86400)
        return f"{days}d"
    else:
        years = int(seconds / 31536000)
        return f"{years}y"
    
### the render loop for your custom UI. Without it registered, nothing ever gets drawn in the viewport.
def draw_image_buttons_callback():
    data = bpy.types.Scene.gpu_image_buttons_data
    if not data:
        return

    # ────────────────────────────────────────────────────────────────────
    # 1) Rectangles (solid‑colour quads)
    # ────────────────────────────────────────────────────────────────────
    for item in data:
        if item.get("type") == "rect":
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            shader.bind()
            shader.uniform_float("color", item.get("color", (1, 1, 1, 1)))
            x1, y1, x2, y2 = item.get("pos")
            verts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
            batch.draw(shader)

    # ────────────────────────────────────────────────────────────────────
    # 2) Images:  static PNG/JPG       and       animated PNG sequence
    # ────────────────────────────────────────────────────────────────────
    for item in data:
        t = item.get("type")

        # Skip rects (handled) and text (handled later)
        if t in {"rect", "text"}:
            continue

        # ---- animated frame sequence ----------------------------------
        if t == "frame_seq":
            n_frames = item["_n_frames"]
            fps      = item["_fps"]
            frame_ix = int(time.time() * fps) % n_frames
            tex      = animated_sequence._texture_for(frame_ix)

            shader   = item["shader"]
            batch    = item["batch"]

            shader.bind()
            shader.uniform_sampler("image", tex)
            batch.draw(shader)
            continue  # handled, go to next item

        # ---- existing static image path -------------------------------
        shader = item.get("shader")
        batch  = item.get("batch")
        tex    = item.get("texture")
        if shader and batch and tex:
            shader.bind()
            shader.uniform_sampler("image", tex)
            batch.draw(shader)

    # ────────────────────────────────────────────────────────────────────
    # 3) Text labels
    # ────────────────────────────────────────────────────────────────────
    for item in data:
        if item.get("type") == "text":
            draw_text_item(item)

# ------------------------------------------------------------------------
###this function is invoked by draw_image_buttons_callback for every 
# item where item["type"] == "text", it’s the piece that turns text-item 
#dictionaries into actual visible labels in the Blender viewport.

def draw_text_item(item):
    """
    Given a text item dictionary, draw it in the current region.
    Supports multiline, color, alignment, etc.
    """

    text = item["content"]
    x, y = item["position"]
    size = item["size"]
    r, g, b, a = item["color"]
    alignment_mode = item["alignment"]
    multiline = item["multiline"]

    font_id = item.get("font_id", CUSTOM_FONT_ID)
    blf.color(font_id, r, g, b, a)
    blf.size(font_id, int(size))


    # We’ll handle alignment by measuring text width and offsetting x.
    # For multiline, we’ll also handle multiple lines and line spacing.
    # Example line spacing factor:
    line_spacing = size * 1.2

    if multiline and ("\n" in text):
        lines = text.split("\n")
    else:
        lines = [text]

    # We’ll draw from top line to bottom line
    # so each subsequent line goes below the previous one:
    total_height = line_spacing * len(lines)

    # Start from top Y if you prefer to anchor from the top. 
    # Or remain consistent with how your existing code handles text anchors.
    current_y = y

    for i, line in enumerate(lines):
        # Measure the line width:
        line_width, _ = blf.dimensions(font_id, line)

        # Adjust x offset for alignment:
        if alignment_mode == 'CENTER':
            draw_x = x - (line_width / 2.0)
        elif alignment_mode == 'RIGHT':
            draw_x = x - line_width
        else:
            # 'LEFT' or default
            draw_x = x

        # If you'd like to anchor the first line at y, 
        # we can do one of these two approaches:
        # (A) Top-down approach:
        line_index_from_top = i  # 0 for top line
        line_offset_y = - (line_index_from_top * line_spacing)
        draw_y = current_y + line_offset_y

        # (B) Bottom-up approach:
        # line_index_from_bottom = (len(lines) - 1) - i
        # line_offset_y = line_index_from_bottom * line_spacing
        # draw_y = current_y + line_offset_y

        blf.position(font_id, draw_x, draw_y, 0)
        blf.draw(font_id, line)