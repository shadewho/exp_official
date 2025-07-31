#Exploratory/Exp_UI/interface/drawing/explore_loading.py
from .utilities import build_text_item
from ...cache_system.manager import cache_manager
from .animated_sequence import build_loading_frames
def build_loading_progress(template_item, progress):
    """
    Draws a semi-transparent overlay, a centered dark box,
    a single progress bar with its percentage, and a short message:
      "Your game will begin momentarily..."
    """
    data_list = []

    # The template's bounding box
    x1, y1, x2, y2 = template_item["pos"]
    template_w = x2 - x1
    template_h = y2 - y1

    # -------------------------------------------------------
    # 1) Semi-Transparent Overlay
    # -------------------------------------------------------
    data_list.append({
        "type": "rect",
        "pos": (x1, y1, x2, y2),
        "color": (0.0, 0.0, 0.0, 0.6)  # Black, 60% opacity
    })

    # -------------------------------------------------------
    # 2) Centered Box for Progress UI
    # -------------------------------------------------------
    # Make it ~40% of the template width, 25% of the template height
    box_w = template_w * 0.7
    box_h = template_h * 0.5

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    box_x1 = center_x - (box_w / 2)
    box_x2 = center_x + (box_w / 2)
    box_y1 = center_y - (box_h / 2)
    box_y2 = center_y + (box_h / 2)


    # ────────── Loading animation (20 PNG frames) ──────────
    spinner_side = box_w * 0.28               # 35 % of the box width
    data_list.append(
        build_loading_frames(
            center_x,                         # centre of the box
            center_y - box_h * 0.05,          # slight upward nudge
            spinner_side                      # size in pixels
        )
    )
    spinner_cy        = center_y - box_h * 0.05          # same centre you gave the spinner
    spinner_top_y     = spinner_cy + spinner_side / 2
    spinner_bottom_y  = spinner_cy - spinner_side / 2
    # ───────────────────────────────────────────────────────

    # Dark background box
    data_list.append({
        "type": "rect",
        "pos": (box_x1, box_y1, box_x2, box_y2),
        "color": (0.1, 0.1, 0.1, 0.9)
    })


    # -------------------------------------------------------
    # 3) "Loading..." Title Text
    # -------------------------------------------------------
    title_font_size = min(box_w, box_h) * 0.15
    title_x = (box_x1 + box_x2) / 2
    # Slightly below the top edge of the box
    title_y = spinner_top_y + title_font_size * 0.6   # 0.6 × font‑size gap

    data_list.append(build_text_item(
        text="Loading...",
        x=title_x,
        y=title_y,
        size=title_font_size,
        color=(1.0, 1.0, 1.0, 1.0),
        alignment='CENTER',
        multiline=False
    ))

    # -------------------------------------------------------
    # 4) Progress Bar
    # -------------------------------------------------------
    bar_width = box_w * 0.8
    bar_height = box_h * 0.15
    bar_x1 = (box_x1 + box_x2) / 2 - (bar_width / 2)
    bar_x2 = bar_x1 + bar_width
    # Position it near the bottom quarter of the box
    bar_y1 = box_y1 + (box_h * 0.25)
    bar_y2 = bar_y1 + bar_height

    # Bar background
    data_list.append({
        "type": "rect",
        "pos": (bar_x1, bar_y1, bar_x2, bar_y2),
        "color": (0.2, 0.2, 0.2, 1.0)  # Dark gray
    })

    # Filled portion
    fill_w = bar_width * progress
    data_list.append({
        "type": "rect",
        "pos": (bar_x1, bar_y1, bar_x1 + fill_w, bar_y2),
        "color": (0.0, 0.7, 0.0, 1.0)  # Green
    })

    # -------------------------------------------------------
    # 5) Percentage Text Inside the Bar
    # -------------------------------------------------------
    percent_font_size = bar_height * 0.7
    percent_x = (bar_x1 + bar_x2) / 2
    percent_y = spinner_bottom_y - percent_font_size * 0.6

    data_list.append(build_text_item(
        text=f"{int(progress * 100)}%",
        x=percent_x,
        y=percent_y,
        size=percent_font_size,
        color=(1.0, 1.0, 1.0, 1.0),
        alignment='CENTER',
        multiline=False
    ))

    # -------------------------------------------------------
    # 6) "Your game will begin momentarily..." Message
    # -------------------------------------------------------
    message_text = "Your game will begin momentarily..."
    message_font_size = box_h * 0.10  # or tweak to taste
    message_x = (box_x1 + box_x2) / 2
    message_y = percent_y - message_font_size * 1.6

    data_list.append(build_text_item(
        text=message_text,
        x=message_x,
        y=message_y,
        size=message_font_size,
        color=(0.8, 0.8, 0.8, 1.0),
        alignment='CENTER',
        multiline=False
    ))

    return data_list