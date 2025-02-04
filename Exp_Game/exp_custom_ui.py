########################################
# File: exp_custom_ui.py
# Purpose: Draw ephemeral or indefinite text in 3D Viewport,
#          using ratio-based margin & scale for resolution independence.
########################################

import bpy
import blf
import time
from .exp_time import get_game_time

_ui_draw_handler = None

_reaction_texts = []  # store ratio-based data
_ui_messages = []     # optional legacy debug messages

def add_text_reaction(
    text_str,
    anchor,
    margin_x,
    margin_y,
    scale,
    end_time=None,
    color=(1,1,1,1)
):
    item = {
        "text": text_str,
        "anchor": anchor,
        "margin_x": margin_x,
        "margin_y": margin_y,
        "scale": scale,
        "end_time": end_time,
        "color": color,
    }
    _reaction_texts.append(item)
    return item  # <--- return it so we can modify it further


def update_text_reactions():
    now = get_game_time()
    keep_list = []

    for it in _reaction_texts:
        sub = it.get("subtype", "STATIC")

        # ===============================
        #  A) OBJECTIVE_TIMER_DISPLAY
        # ===============================
        if sub == "OBJECTIVE_TIMER_DISPLAY":
            idx_str = it.get("objective_index", "")
            if idx_str.isdigit():
                idx = int(idx_str)
                scene = bpy.context.scene
                if 0 <= idx < len(scene.objectives):
                    objv = scene.objectives[idx]

                    # Either way, we can format it as “seconds”:
                    current_time_val = max(0.0, objv.timer_value)
                    it["text"] = format_hms(current_time_val)

            # Check indefinite vs ephemeral
            e_time = it.get("end_time")
            if e_time is None:
                keep_list.append(it)  # keep forever
            else:
                if now < e_time:
                    keep_list.append(it)  # still within time

        # ===============================
        #  B) OBJECTIVE (Simple Counter)
        # ===============================
        elif sub == "OBJECTIVE":
            idx_str = it.get("objective_index", "")
            if idx_str.isdigit():
                idx = int(idx_str)
                scene = bpy.context.scene
                if 0 <= idx < len(scene.objectives):
                    objv = scene.objectives[idx]
                    fmt = it.get("format", "{value}")
                    it["text"] = fmt.format(value=objv.current_value)

            # Indefinite vs ephemeral
            e_time = it.get("end_time")
            if e_time is None:
                keep_list.append(it)
            else:
                if now < e_time:
                    keep_list.append(it)

        # ===============================
        #  C) STATIC
        # ===============================
        elif sub == "STATIC":
            e_time = it.get("end_time")
            if e_time is None:
                keep_list.append(it)
            else:
                if now < e_time:
                    keep_list.append(it)

    _reaction_texts[:] = keep_list






def add_ui_message(text, duration=3.0, indefinite=False):
    if indefinite:
        end_time = None
    else:
        end_time = time.time() + duration
    _ui_messages.append((text, end_time, indefinite))

def update_ui_messages():
    now = time.time()
    keep = []
    for (txt, et, indef) in _ui_messages:
        if indef or (et and et > now):
            keep.append((txt, et, indef))
    _ui_messages[:] = keep

def clear_ui_messages():
    _ui_messages.clear()

def clear_all_text():
    clear_text_reactions()
    clear_ui_messages()

def draw_ui_callback():
    # Instead of using bpy.context.region, manually find
    # the 3D View "WINDOW" region in the current window/screen.
    region_3d = None
    window = bpy.context.window
    if not window or not window.screen:
        return

    for area in window.screen.areas:
        if area.type == 'VIEW_3D':
            for reg in area.regions:
                if reg.type == 'WINDOW':
                    region_3d = reg
                    break
            break

    if not region_3d:
        # Can't find a 3D 'WINDOW' region => bail
        return

    width  = region_3d.width
    height = region_3d.height

    # ------------------------------------------------
    # Now your existing text drawing code is unchanged:
    # ------------------------------------------------
    # 1) Reaction-based text
    for it in _reaction_texts:
        text_str = it["text"]
        anchor   = it["anchor"]

        # Convert margin from ratio => pixels
        mx_px = it["margin_x"] * width
        my_px = it["margin_y"] * height

        # Convert scale => font size in pixels
        scale_ratio = it["scale"]
        font_size_px = int(scale_ratio * height)
        if font_size_px < 10:
            font_size_px = 10  # minimum size safety

        color_rgba = it["color"]
        font_id = 0  # default Blender font
        blf.size(font_id, font_size_px)

        tw, th = blf.dimensions(font_id, text_str)

        # anchor logic => pass pixel-based margins
        draw_x, draw_y = compute_anchored_position(
            width, height, tw, th, anchor, mx_px, my_px
        )

        blf.position(font_id, draw_x, draw_y, 0)
        r, g, b, a = color_rgba
        blf.color(font_id, r, g, b, a)
        blf.draw(font_id, text_str)

    # 2) Optional legacy debug messages
    font_id = 0
    blf.size(font_id, 20)
    line_gap = 24
    center_x = width // 2
    center_y = height // 2
    reversed_msgs = list(reversed(_ui_messages))

    for i, (msg_txt, msg_et, msg_indef) in enumerate(reversed_msgs):
        tw, th = blf.dimensions(font_id, msg_txt)
        x = center_x - (tw / 2)
        y = center_y + (i * line_gap)
        blf.position(font_id, x, y, 0)
        blf.color(font_id, 1, 1, 1, 1)
        blf.draw(font_id, msg_txt)


def compute_anchored_position(width, height, text_w, text_h, anchor, margin_x_px, margin_y_px):
    """
    Normal anchor logic, except margin_x_px, margin_y_px are already in pixels.
    """
    if anchor == 'TOP_LEFT':
        draw_x = margin_x_px
        draw_y = height - margin_y_px - text_h
    elif anchor == 'TOP_CENTER':
        draw_x = (width // 2) - (text_w // 2)
        draw_y = height - margin_y_px - text_h
    elif anchor == 'TOP_RIGHT':
        draw_x = width - margin_x_px - text_w
        draw_y = height - margin_y_px - text_h
    elif anchor == 'MIDDLE_LEFT':
        draw_x = margin_x_px
        draw_y = (height // 2) - (text_h // 2)
    elif anchor == 'MIDDLE_CENTER':
        draw_x = (width // 2) - (text_w // 2)
        draw_y = (height // 2) - (text_h // 2)
    elif anchor == 'MIDDLE_RIGHT':
        draw_x = width - margin_x_px - text_w
        draw_y = (height // 2) - (text_h // 2)
    elif anchor == 'BOTTOM_LEFT':
        draw_x = margin_x_px
        draw_y = margin_y_px
    elif anchor == 'BOTTOM_CENTER':
        draw_x = (width // 2) - (text_w // 2)
        draw_y = margin_y_px
    elif anchor == 'BOTTOM_RIGHT':
        draw_x = width - margin_x_px - text_w
        draw_y = margin_y_px
    else:
        draw_x = margin_x_px
        draw_y = height - margin_y_px - text_h

    return draw_x, draw_y

def register_ui_draw():
    global _ui_draw_handler
    if _ui_draw_handler is None:
        _ui_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_ui_callback, (), 'WINDOW', 'POST_PIXEL'
        )

def unregister_ui_draw():
    global _ui_draw_handler
    if _ui_draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_ui_draw_handler, 'WINDOW')
        _ui_draw_handler = None
    clear_all_text()


def format_hms(seconds_left: float) -> str:
    """
    Convert total seconds => conditional 'Xh Xm X.Xs' format.
    Only shows hours if hours>0, only shows minutes if minutes>0, etc.
    e.g. '10.0s', '5m 2.3s', '1h 15.2s', or '1h 30m 12.0s'.
    If all are zero => '0.0s'.
    """
    if seconds_left < 0:
        seconds_left = 0.0

    hours = int(seconds_left // 3600)
    remain = seconds_left % 3600
    minutes = int(remain // 60)
    secs = remain % 60  # float

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    # For seconds, you might want to show it even if 0.0,
    # or only if > 0.  Here we show it only if > 0:
    if secs > 0:
        parts.append(f"{secs:.1f}s")

    if not parts:
        # Means hours=0, minutes=0, secs=0 => show "0.0s"
        return "0.0s"

    return " ".join(parts)

def clear_text_reactions():
    """
    Removes all text items from the _reaction_texts list.
    """
    _reaction_texts.clear()