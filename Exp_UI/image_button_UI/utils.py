# image_button_UI/utils.py

import bpy
from .config import LAST_VIEWPORT_SIZE, TEMPLATE_ASPECT_RATIO, OFFSET_FACTOR

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