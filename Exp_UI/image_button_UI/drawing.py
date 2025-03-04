# image_button_UI/drawing.py
import blf
import os
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from ..helper_functions import download_thumbnail, format_relative_time

from .config import (
    GRID_COLUMNS, GRID_ROWS, TEMPLATE_ASPECT_RATIO,
    OFFSET_FACTOR, THUMBNAIL_SPACING_FACTOR,
    TEMPLATE_MARGIN_HORIZONTAL, TEMPLATE_MARGIN_TOP, TEMPLATE_MARGIN_BOTTOM, TEMPLATE_IMAGE_PATH,
    ARROW_LEFT_PATH, ARROW_RIGHT_PATH, CLOSE_WINDOW_PATH, BACK_BUTTON_PATH,
    THUMBNAIL_TEXT_SIZE, THUMBNAIL_TEXT_COLOR, THUMBNAIL_TEXT_ALIGNMENT, THUMBNAIL_TEXT_OFFSET_RATIO, THUMBNAIL_TEXT_SIZE_RATIO, 
    THUMBNAILS_PER_PAGE, EXPLORE_BUTTON_PATH,
    LOADING_IMAGE_OFFSET_X, LOADING_IMAGE_OFFSET_Y, LOADING_IMAGE_PATH, LOADING_IMAGE_SCALE
)
from .cache import get_or_load_image, get_or_create_texture
from .utils import calculate_free_space, calculate_template_position
from ..cache_manager import cache_manager  # Ensure the cache_manager is imported

def build_template_and_close():
    """
    Create draw data for:
      - The background template (2:1 ratio)
      - The Close button
    Return a list of dicts, each describing a draw item (batch, texture, etc.).
    """
    data_list = []
    
    # 1) Calculate template position
    free_space = calculate_free_space()
    template_pos = calculate_template_position(
        free_space,
        TEMPLATE_ASPECT_RATIO,
        OFFSET_FACTOR
    )
    x1, y1, x2, y2 = template_pos["x1"], template_pos["y1"], template_pos["x2"], template_pos["y2"]

    # 2) Template background
    template_img = get_or_load_image(TEMPLATE_IMAGE_PATH)
    if template_img:
        shader = gpu.shader.from_builtin('IMAGE')
        tex = get_or_create_texture(template_img)
        if tex:
            verts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            coords = [(0,0), (1,0), (1,1), (0,1)]
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})
            data_list.append({
                "shader": shader,
                "batch": batch,
                "texture": tex,
                "name": "Template",
                "pos": (x1, y1, x2, y2),
            })
    
    # 3) Close button
    close_img = get_or_load_image(CLOSE_WINDOW_PATH)
    if close_img:
        close_tex = get_or_create_texture(close_img)
        if close_tex:
            close_shader = gpu.shader.from_builtin('IMAGE')

            native_w, native_h = close_img.size
            if native_h == 0:
                native_aspect = 1.0
            else:
                native_aspect = native_w / native_h

            # We'll size the close button relative to template width, e.g. 4%
            close_w = 0.05 * (x2 - x1)
            close_h = close_w / native_aspect

            # % of template width/height:
            margin_right = 0.025 * (x2 - x1)
            margin_top   = 0.035 * (y2 - y1)


            close_x2 = x2 - margin_right
            close_x1 = close_x2 - close_w
            close_y2 = y2 - margin_top
            close_y1 = close_y2 - close_h

            verts = [
                (close_x1, close_y1),
                (close_x2, close_y1),
                (close_x2, close_y2),
                (close_x1, close_y2)
            ]
            coords = [(0,0), (1,0), (1,1), (0,1)]
            close_batch = batch_for_shader(
                close_shader, 'TRI_FAN',
                {"pos": verts, "texCoord": coords}
            )
            data_list.append({
                "shader": close_shader,
                "batch": close_batch,
                "texture": close_tex,
                "name": "Close_Icon",
                "pos": (close_x1, close_y1, close_x2, close_y2),
            })

    return data_list


def build_browse_content(template_item):
    """
    Draws the BROWSE view: a grid of thumbnails for the packages on the current page,
    plus optional arrows if there are multiple pages, ensuring each thumbnail is a square (1:1).
    Includes a "Page X / Y" label at the bottom center.
    """
    if not template_item:
        return []

    data_list = []
    x1, y1, x2, y2 = template_item["pos"]

    template_width = x2 - x1
    template_height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # --- 1) Fetch packages for the current page — REMOVED SLICING ---
    packages_data = getattr(bpy.types.Scene, 'fetched_packages_data', [])
    scene = bpy.context.scene
    page = scene.current_thumbnail_page
    total_pages = scene.total_thumbnail_pages

    # We simply do:
    visible_packages = packages_data

    # --- 2) Grid positioning ---
    margin_x = template_width * TEMPLATE_MARGIN_HORIZONTAL
    margin_y_top = template_height * TEMPLATE_MARGIN_TOP
    margin_y_bottom = template_height * TEMPLATE_MARGIN_BOTTOM

    thumbs_x1 = x1 + margin_x
    thumbs_x2 = x2 - margin_x
    thumbs_y1 = y1 + margin_y_bottom
    thumbs_y2 = y2 - margin_y_top

    usable_width = thumbs_x2 - thumbs_x1
    usable_height = thumbs_y2 - thumbs_y1

    thumbnail_width = usable_width / GRID_COLUMNS * (1 - THUMBNAIL_SPACING_FACTOR)
    thumbnail_height = usable_height / GRID_ROWS * (1 - THUMBNAIL_SPACING_FACTOR)

    # Enforce square thumbnails
    thumb_size = min(thumbnail_width, thumbnail_height)

    spacing_x = 0
    spacing_y = 0
    if GRID_COLUMNS > 1:
        spacing_x = (usable_width - GRID_COLUMNS * thumb_size) / (GRID_COLUMNS - 1)
    if GRID_ROWS > 1:
        spacing_y = (usable_height - GRID_ROWS * thumb_size) / (GRID_ROWS - 1)

    # --- 3) Build thumbnail items ---
    for index, pkg in enumerate(visible_packages):
        thumb_url = pkg.get("thumbnail_url")
        local_thumb_path = pkg.get("local_thumb_path")
        
        # If no valid local thumbnail exists, try to download it if a URL is provided.
        if (not local_thumb_path or not os.path.exists(local_thumb_path)) and thumb_url:
            local_thumb_path = download_thumbnail(thumb_url)
            pkg["local_thumb_path"] = local_thumb_path

        # If still missing, skip this package.
        if not local_thumb_path or not os.path.exists(local_thumb_path):
            continue

        img = get_or_load_image(local_thumb_path)
        if not img:
            continue
        tex = get_or_create_texture(img)
        if not tex:
            continue

        row = index // GRID_COLUMNS
        col = index % GRID_COLUMNS

        # X position is as before
        thumb_x1 = thumbs_x1 + col * (thumb_size + spacing_x)
        thumb_x2 = thumb_x1 + thumb_size

        # Y position from the top row downward:
        thumb_y2 = thumbs_y2 - row * (thumb_size + spacing_y)
        thumb_y1 = thumb_y2 - thumb_size

        shader = gpu.shader.from_builtin('IMAGE')
        verts = [
            (thumb_x1, thumb_y1),
            (thumb_x2, thumb_y1),
            (thumb_x2, thumb_y2),
            (thumb_x1, thumb_y2),
        ]
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        batch_obj = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})

        data_list.append({
            "shader": shader,
            "batch": batch_obj,
            "texture": tex,
            "name": pkg.get("package_name", "unknown"),
            "pos": (thumb_x1, thumb_y1, thumb_x2, thumb_y2),
        })

        package_name = pkg.get("package_name", "NoName")
        add_thumbnail_title(
            data_list,
            (thumb_x1, thumb_y1, thumb_x2, thumb_y2),
            package_name
        )

    # --- 4) If there's more than 1 page, draw arrows ---
    if total_pages > 1:
        arrow_scale = 0.03
        arrow_offset_x = 0.48
        arrow_offset_y = -0.05

        left_arrow_img = get_or_load_image(ARROW_LEFT_PATH)
        if left_arrow_img:
            left_arrow_tex = get_or_create_texture(left_arrow_img)
            if left_arrow_tex:
                img_width, img_height = left_arrow_img.size
                aspect_ratio = img_width / img_height if img_height else 1.0
                arrow_width = arrow_scale * template_width
                arrow_height = arrow_width / aspect_ratio

                left_x1 = center_x - (arrow_offset_x * template_width) - (arrow_width / 2)
                left_y1 = center_y + (arrow_offset_y * template_height) - (arrow_height / 2)
                left_x2 = left_x1 + arrow_width
                left_y2 = left_y1 + arrow_height

                shader = gpu.shader.from_builtin('IMAGE')
                left_arrow_batch = batch_for_shader(
                    shader, 'TRI_FAN',
                    {
                        "pos": [(left_x1, left_y1), (left_x2, left_y1),
                                (left_x2, left_y2), (left_x1, left_y2)],
                        "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)]
                    }
                )

                data_list.append({
                    "shader": shader,
                    "batch": left_arrow_batch,
                    "texture": left_arrow_tex,
                    "name": "Left_Arrow",
                    "pos": (left_x1, left_y1, left_x2, left_y2),
                })

        right_arrow_img = get_or_load_image(ARROW_RIGHT_PATH)
        if right_arrow_img:
            right_arrow_tex = get_or_create_texture(right_arrow_img)
            if right_arrow_tex:
                img_width, img_height = right_arrow_img.size
                aspect_ratio = img_width / img_height if img_height else 1.0
                arrow_width = arrow_scale * template_width
                arrow_height = arrow_width / aspect_ratio

                right_x1 = center_x + (arrow_offset_x * template_width) - (arrow_width / 2)
                right_y1 = center_y + (arrow_offset_y * template_height) - (arrow_height / 2)
                right_x2 = right_x1 + arrow_width
                right_y2 = right_y1 + arrow_height

                shader = gpu.shader.from_builtin('IMAGE')
                right_arrow_batch = batch_for_shader(
                    shader, 'TRI_FAN',
                    {
                        "pos": [(right_x1, right_y1), (right_x2, right_y1),
                                (right_x2, right_y2), (right_x1, right_y2)],
                        "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)]
                    }
                )

                data_list.append({
                    "shader": shader,
                    "batch": right_arrow_batch,
                    "texture": right_arrow_tex,
                    "name": "Right_Arrow",
                    "pos": (right_x1, right_y1, right_x2, right_y2),
                })

    # --- 5) Show "Page X / Y" text at the bottom center (if desired) ---
    page_label = f"Page {page} / {total_pages}"
    text_size = template_height * 0.03
    text_y = y1 + 0.01 * template_height  # e.g. 1% above the bottom
    data_list.append({
        "type": "text",
        "content": page_label,
        "position": ((x1 + x2) / 2, text_y),
        "size": text_size,
        "color": (1.0, 1.0, 1.0, 1.0),
        "alignment": 'CENTER',
        "multiline": False
    })

    return data_list


def build_detail_content(template_item):
    """
    Build the detail view content, including a back button and an enlarged thumbnail.
    Also adds an Explore button positioned relative to the center.
    """
    if not template_item:
        return []
    
    data_list = []
    x1, y1, x2, y2 = template_item["pos"]

    # --- Back Button ---
    back_img = get_or_load_image(BACK_BUTTON_PATH)
    if back_img:
        back_tex = get_or_create_texture(back_img)
        if back_tex:
            back_shader = gpu.shader.from_builtin('IMAGE')

            native_w, native_h = back_img.size
            native_aspect = native_w / native_h if native_h != 0 else 1.0

            back_w_ratio = 0.1
            back_w = back_w_ratio * (x2 - x1)
            back_h = back_w / native_aspect

            back_margin_top_ratio = 0.05
            back_margin_left_ratio = 0.02

            back_margin_top = back_margin_top_ratio * (y2 - y1)
            back_margin_left = back_margin_left_ratio * (x2 - x1)

            back_x1 = x1 + back_margin_left
            back_x2 = back_x1 + back_w
            back_y2 = y2 - back_margin_top
            back_y1 = back_y2 - back_h

            verts = [
                (back_x1, back_y1),
                (back_x2, back_y1),
                (back_x2, back_y2),
                (back_x1, back_y2)
            ]
            coords = [(0,0), (1,0), (1,1), (0,1)]
            back_batch = batch_for_shader(
                back_shader, 'TRI_FAN',
                {"pos": verts, "texCoord": coords}
            )

            data_list.append({
                "shader": back_shader,
                "batch": back_batch,
                "texture": back_tex,
                "name": "Back_Icon",
                "pos": (back_x1, back_y1, back_x2, back_y2),
            })

    # --- Explore Button ---
    explore_img = get_or_load_image(EXPLORE_BUTTON_PATH)
    if explore_img:
        explore_tex = get_or_create_texture(explore_img)
        if explore_tex:
            explore_shader = gpu.shader.from_builtin('IMAGE')

            native_w, native_h = explore_img.size
            explore_aspect = native_w / native_h if native_h != 0 else 1.0

            explore_w_ratio = 0.3
            explore_w = explore_w_ratio * (x2 - x1)
            explore_h = explore_w / explore_aspect

            explore_offset_x_ratio = 0.25
            explore_offset_y_ratio = -0.4

            explore_offset_x = explore_offset_x_ratio * (x2 - x1)
            explore_offset_y = explore_offset_y_ratio * (y2 - y1)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            explore_x1 = center_x + explore_offset_x - (explore_w / 2)
            explore_y1 = center_y + explore_offset_y - (explore_h / 2)
            explore_x2 = explore_x1 + explore_w
            explore_y2 = explore_y1 + explore_h

            verts = [
                (explore_x1, explore_y1),
                (explore_x2, explore_y1),
                (explore_x2, explore_y2),
                (explore_x1, explore_y2)
            ]
            coords = [(0,0), (1,0), (1,1), (0,1)]
            explore_batch = batch_for_shader(
                explore_shader, 'TRI_FAN',
                {"pos": verts, "texCoord": coords}
            )

            data_list.append({
                "shader": explore_shader,
                "batch": explore_batch,
                "texture": explore_tex,
                "name": "Explore_Icon",
                "pos": (explore_x1, explore_y1, explore_x2, explore_y2),
            })
    
    # --- Enlarged Thumbnail (forcing 1:1 square) ---
    selected_image_path = str(bpy.context.scene.selected_thumbnail)
    if selected_image_path and os.path.exists(selected_image_path):
        enlarged_img = get_or_load_image(selected_image_path)
        if enlarged_img:
            enlarged_tex = get_or_create_texture(enlarged_img)
            if enlarged_tex:
                enlarged_shader = gpu.shader.from_builtin('IMAGE')

                # Ignoring the actual aspect ratio:
                # We'll keep the same enlarge_factor approach, but
                # force enlarged height = enlarged width for a 1:1 square.

                enlarge_factor = 8
                thumb_width_ratio = 0.05  # originally 5% of template width
                thumb_width = thumb_width_ratio * (x2 - x1)
                enlarged_w = thumb_width * enlarge_factor
                enlarged_h = enlarged_w  # <--- Force 1:1

                offset_x_ratio = -0.25
                offset_y_ratio = -0.075

                offset_x = offset_x_ratio * (x2 - x1)
                offset_y = offset_y_ratio * (y2 - y1)

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
                    (enlarged_x1, enlarged_y2)
                ]
                coords = [(0,0), (1,0), (1,1), (0,1)]
                enlarged_batch = batch_for_shader(
                    enlarged_shader, 'TRI_FAN',
                    {"pos": verts, "texCoord": coords}
                )

                data_list.append({
                    "shader": enlarged_shader,
                    "batch": enlarged_batch,
                    "texture": enlarged_tex,
                    "name": "Enlarged_Thumbnail",
                    "pos": (enlarged_x1, enlarged_y1, enlarged_x2, enlarged_y2),
                })

    # Detailed description text
    build_item_detail_text(data_list, template_item)

    return data_list

def load_image_buttons():
    """
    Build the entire UI data list (images + text items) for the current mode (BROWSE or DETAIL).
    1) Grab the page's data from cache_manager, copy it into fetched_packages_data.
    2) Build the "template" background, close icon, then either the thumbnail grid or detail view.
    3) Return a list of draw items (for the draw callback).
    """
    scene = bpy.context.scene

    # (A) Which page are we on?
    current_page = scene.current_thumbnail_page

    # (C) Start building the draw items
    data_list = []

    # 1) Build the background template + close button
    template_and_close = build_template_and_close()  # some function returning the main background + close icon
    data_list.extend(template_and_close)

    # 2) Find the template item in data_list so we know where to place content
    template_item = next((b for b in data_list if b.get("name") == "Template"), None)

    if scene.ui_current_mode == "BROWSE":
        # Build the grid of thumbnails
        browse_content = build_browse_content(template_item)
        data_list.extend(browse_content)
    elif scene.ui_current_mode == "DETAIL":
        # Build a single large thumbnail + detail text + back button
        detail_content = build_detail_content(template_item)
        data_list.extend(detail_content)

    # 3) If scene.show_loading_image is True, add the spinner
    if scene.show_loading_image and template_item:
        loading_indicator = build_loading_indicator(template_item)
        data_list.extend(loading_indicator)

    return data_list



# ------------------------------------------------------------------------
# Text Drawing Utilities
# ------------------------------------------------------------------------
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
        "multiline": multiline
    }


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

    font_id = 2  # or any other channel if you prefer
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


def add_thumbnail_title(data_list, thumb_box, title_text):
    x1, y1, x2, y2 = thumb_box
    thumb_height = (y2 - y1)

    # 1) Truncate at 20 chars, adding "..."
    max_chars = 20
    if len(title_text) > max_chars:
        title_text = title_text[:(max_chars - 3)] + "..."

    center_x = (x1 + x2) / 2.0
    
    # 2) The vertical offset from the bottom of the thumbnail
    offset_pixels = THUMBNAIL_TEXT_OFFSET_RATIO * thumb_height
    text_y = y1 - offset_pixels

    # 3) Font size
    dynamic_font_size = THUMBNAIL_TEXT_SIZE_RATIO * thumb_height
    
    # 4) Build the text item
    text_label = build_text_item(
        text=title_text,
        x=center_x,
        y=text_y,
        size=dynamic_font_size,
        color=THUMBNAIL_TEXT_COLOR,
        alignment=THUMBNAIL_TEXT_ALIGNMENT,
        multiline=False
    )
    data_list.append(text_label)

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
        f"Uploaded: {upload_date_formatted} ago"
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


def build_loading_indicator(template_item):
    """
    Builds a draw dictionary for the 'loading' image.
    This is similar to how back or explore icons are created,
    except it's not clickable – just an indicator.
    We'll place it relative to the template area.
    """
    data_list = []
    
    x1, y1, x2, y2 = template_item["pos"]
    template_w = x2 - x1
    template_h = y2 - y1

    loading_img = get_or_load_image(LOADING_IMAGE_PATH)
    if not loading_img:
        return data_list  # nothing to draw

    loading_tex = get_or_create_texture(loading_img)
    if not loading_tex:
        return data_list

    shader = gpu.shader.from_builtin('IMAGE')

    # We'll scale the loading image by LOADING_IMAGE_SCALE * template_w
    native_w, native_h = loading_img.size
    aspect = native_w / native_h if native_h != 0 else 1.0

    icon_width = template_w * LOADING_IMAGE_SCALE
    icon_height = icon_width / aspect

    # Position it. For example, center it in the template:
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Optionally apply offsets in terms of template width/height
    offset_x = LOADING_IMAGE_OFFSET_X * template_w
    offset_y = LOADING_IMAGE_OFFSET_Y * template_h

    # Now compute corners
    icon_x1 = center_x - (icon_width / 2) + offset_x
    icon_y1 = center_y - (icon_height / 2) + offset_y
    icon_x2 = icon_x1 + icon_width
    icon_y2 = icon_y1 + icon_height

    verts = [
        (icon_x1, icon_y1),
        (icon_x2, icon_y1),
        (icon_x2, icon_y2),
        (icon_x1, icon_y2)
    ]
    coords = [(0,0), (1,0), (1,1), (0,1)]
    batch_obj = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})

    data_list.append({
        "shader": shader,
        "batch": batch_obj,
        "texture": loading_tex,
        "name": "Loading_Indicator",
        "pos": (icon_x1, icon_y1, icon_x2, icon_y2),
    })

    return data_list

def draw_image_buttons_callback():
    data = bpy.types.Scene.gpu_image_buttons_data
    if not data:
        return

    # Draw non-text items first.
    for item in data:
        if item.get("type") != "text":
            shader = item.get("shader")
            batch_obj = item.get("batch")
            gpu_texture = item.get("texture")
            if shader and batch_obj and gpu_texture:
                shader.bind()
                shader.uniform_sampler("image", gpu_texture)
                batch_obj.draw(shader)

    # Then draw text items.
    for item in data:
        if item.get("type") == "text":
            draw_text_item(item)
