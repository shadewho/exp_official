#Exploratory/Exp_UI/interface/drawing/detail_content.py
import blf
import os
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from .utilities import format_relative_time, build_text_item
from .config import (
    BACK_BUTTON_PATH,
    EXPLORE_BUTTON_PATH,
    MISSING_THUMB,
    VISIT_SHOP_BUTTON_PATH,
    SUBMIT_WORLD_BUTTON_PATH,
)
from ...image_button_UI.cache import get_or_load_image, get_or_create_texture
from ...cache_manager import cache_manager
def build_detail_content(template_item):
    """
    Build the detail view content, including:
      - A back button
      - An Explore button (or Visit Shop for shop items)
      - An enlarged thumbnail (1:1), falling back to a placeholder if missing
      - Detailed description text
    """
    if not template_item:
        return []

    data_list = []
    x1, y1, x2, y2 = template_item["pos"]

    # --- Back Button ---
    try:
        back_img = get_or_load_image(BACK_BUTTON_PATH)
        if not back_img:
            raise RuntimeError(f"Failed to load back image from {BACK_BUTTON_PATH}")
        back_tex = get_or_create_texture(back_img)
        if not back_tex:
            raise RuntimeError("Failed to create texture for back image")

        back_shader = gpu.shader.from_builtin('IMAGE')

        native_w, native_h = back_img.size
        native_aspect = native_w / native_h if native_h else 1.0

        back_w = 0.1 * (x2 - x1)
        back_h = back_w / native_aspect

        back_margin_top = 0.05 * (y2 - y1)
        back_margin_left = 0.02 * (x2 - x1)

        back_x1 = x1 + back_margin_left
        back_x2 = back_x1 + back_w
        back_y2 = y2 - back_margin_top
        back_y1 = back_y2 - back_h

        verts = [
            (back_x1, back_y1),
            (back_x2, back_y1),
            (back_x2, back_y2),
            (back_x1, back_y2),
        ]
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        back_batch = batch_for_shader(back_shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})

        data_list.append({
            "shader":  back_shader,
            "batch":   back_batch,
            "texture": back_tex,
            "name":    "Back_Icon",
            "pos":     (back_x1, back_y1, back_x2, back_y2),
        })
    except Exception as e:
        print(f"[ERROR] build_detail_content (Back button): {e}")

    # --- Explore / Visit Shop / Submit World Button ---
    try:
        # pick which image + name
        if bpy.context.scene.package_item_type == 'shop_item':
            btn_img  = get_or_load_image(VISIT_SHOP_BUTTON_PATH)
            btn_name = "Visit_Shop_Icon"
        elif (bpy.context.scene.package_item_type == 'event'
              and bpy.context.scene.event_stage == 'submission'):
            btn_img  = get_or_load_image(SUBMIT_WORLD_BUTTON_PATH)
            btn_name = "Submit_World_Icon"
        else:
            btn_img  = get_or_load_image(EXPLORE_BUTTON_PATH)
            btn_name = "Explore_Icon"

        if not btn_img:
            # determine which path we tried
            source = {
                "Visit_Shop_Icon": VISIT_SHOP_BUTTON_PATH,
                "Submit_World_Icon": SUBMIT_WORLD_BUTTON_PATH,
                "Explore_Icon": EXPLORE_BUTTON_PATH
            }[btn_name]
            raise RuntimeError(f"Failed to load button image from {source}")

        btn_tex = get_or_create_texture(btn_img)
        if not btn_tex:
            raise RuntimeError("Failed to create texture for button image")

        btn_shader = gpu.shader.from_builtin('IMAGE')

        native_w, native_h = btn_img.size
        aspect = (native_w / native_h) if native_h else 1.0

        btn_w = 0.3 * (x2 - x1)
        btn_h = btn_w / aspect

        offset_x = 0.25 * (x2 - x1)
        offset_y = -0.4 * (y2 - y1)

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        bx1 = center_x + offset_x - (btn_w / 2)
        by1 = center_y + offset_y - (btn_h / 2)
        bx2 = bx1 + btn_w
        by2 = by1 + btn_h

        verts = [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        batch = batch_for_shader(btn_shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})

        data_list.append({
            "shader":  btn_shader,
            "batch":   batch,
            "texture": btn_tex,
            "name":    btn_name,
            "pos":     (bx1, by1, bx2, by2),
        })
    except Exception as e:
        print(f"[ERROR] build_detail_content (Action button): {e}")

    # --- Enlarged Thumbnail (1:1 square, with placeholder fallback) ---
    try:
        sel_path = bpy.context.scene.selected_thumbnail or ""
        if not os.path.exists(sel_path):
            sel_path = MISSING_THUMB

        img = get_or_load_image(sel_path)
        if not img:
            raise RuntimeError(f"Failed to load thumbnail image from {sel_path}")
        tex = get_or_create_texture(img)
        if not tex:
            raise RuntimeError("Failed to create texture for thumbnail image")

        shader = gpu.shader.from_builtin('IMAGE')

        # Force square enlargement
        thumb_width = 0.05 * (x2 - x1)
        enlarge_factor = 8
        enlarged_w = thumb_width * enlarge_factor
        enlarged_h = enlarged_w

        offset_x = -0.25 * (x2 - x1)
        offset_y = -0.075 * (y2 - y1)

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        enlarged_x1 = center_x - (enlarged_w / 2) + offset_x
        enlarged_y1 = center_y - (enlarged_h / 2) + offset_y
        enlarged_x2 = enlarged_x1 + enlarged_w
        enlarged_y2 = enlarged_y1 + enlarged_h

        verts = [
            (enlarged_x1, enlarged_y1),
            (enlarged_x2, enlarged_y1),
            (enlarged_x2, enlarged_y2),
            (enlarged_x1, enlarged_y2),
        ]
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        enlarged_batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})

        data_list.append({
            "shader":  shader,
            "batch":   enlarged_batch,
            "texture": tex,
            "name":    "Enlarged_Thumbnail",
            "pos":     (enlarged_x1, enlarged_y1, enlarged_x2, enlarged_y2),
        })
    except Exception as e:
        print(f"[ERROR] build_detail_content (Enlarged thumbnail): {e}")

    
    # --- Detailed description text ---
    build_item_detail_text(data_list, template_item)

    return data_list


def build_item_detail_text(data_list, template_item):
    """
    Displays detail info with a bit more spacing between items,
    including a heart for 'likes', and the existing word-wrap approach.
    """
    scene = bpy.context.scene
    addon_data = scene.my_addon_data

    if addon_data.file_id <= 0:
        print("NO DATA TO DISPLAY")
        return

    # Pull fields
    title = addon_data.package_name
    author = addon_data.author
    description = addon_data.description
    likes = addon_data.likes
    upload_date = addon_data.upload_date
    download_count = addon_data.download_count  

    # Heart symbol
    heart_char = "♥"

    upload_date_formatted = format_relative_time(upload_date)

    # Add blank lines between items for spacing
    lines = [
        f"Title: {title}",
        "",  # blank line
        f"Author: {author}",
        "",  # blank line
        f"Description: {description}",
        "",  # blank line
        f"Likes: {heart_char} {likes}",
        "",  # blank line
        f"Downloads: {download_count}",  # <--- Insert the new line
        "",  # blank line
        f"Uploaded: {upload_date_formatted} ago",
        ""

    ]

    x1, y1, x2, y2 = template_item["pos"]
    template_width  = (x2 - x1)
    template_height = (y2 - y1)

    # Bounding box margins
    TEXT_LEFT_MARGIN_RATIO   = 0.50
    TEXT_RIGHT_MARGIN_RATIO  = 0.20
    TEXT_TOP_MARGIN_RATIO    = 0.25
    TEXT_BOTTOM_MARGIN_RATIO = 0.15

    box_x1 = x1 + (TEXT_LEFT_MARGIN_RATIO  * template_width)
    box_x2 = x2 - (TEXT_RIGHT_MARGIN_RATIO * template_width)
    box_y2 = y2 - (TEXT_TOP_MARGIN_RATIO   * template_height)
    box_y1 = y1 + (TEXT_BOTTOM_MARGIN_RATIO* template_height)

    box_width  = box_x2 - box_x1
    box_height = box_y2 - box_y1

    # Initial font size attempt
    initial_font_size_ratio = 0.05
    font_size = template_height * initial_font_size_ratio

    # Increase this if you want even more vertical space between each line
    def line_spacing_for(size):
        return size * 1.3

    font_id = 0

    def wrap_paragraph(paragraph, max_width, test_font_size):
        blf.size(font_id, int(test_font_size))
        words = paragraph.split(" ")
        wrapped_lines = []
        current_line = ""

        for w in words:
            candidate = (current_line + " " + w).strip()
            line_width, _ = blf.dimensions(font_id, candidate)
            if line_width > max_width:
                wrapped_lines.append(current_line)
                current_line = w
            else:
                current_line = candidate

        if current_line:
            wrapped_lines.append(current_line)
        return wrapped_lines

    def wrap_all(lines_list, max_width, test_font_size):
        result = []
        for line in lines_list:
            # If it's a blank line, keep it as-is (no wrapping needed)
            if not line.strip():
                result.append("")
                continue

            sub_lines = wrap_paragraph(line, max_width, test_font_size)
            result.extend(sub_lines)
        return result

    # Iterative approach to ensure text fits in the bounding box
    for _attempt in range(10):
        wrapped_lines = wrap_all(lines, box_width, font_size)
        needed_height = len(wrapped_lines) * line_spacing_for(font_size)
        if needed_height <= box_height:
            break
        else:
            ratio = box_height / needed_height
            font_size *= ratio * 0.95

    current_y = box_y2
    line_h = line_spacing_for(font_size)

    blf.size(font_id, int(font_size))

    text_color = (1.0, 1.0, 1.0, 1.0)

    # Draw each wrapped line
    for line_text in wrapped_lines:
        text_item = build_text_item(
            text=line_text,
            x=box_x1,
            y=current_y,
            size=font_size,
            color=text_color,
            alignment='LEFT',
            multiline=False
        )
        data_list.append(text_item)

        current_y -= line_h
        if current_y < box_y1:
            break

