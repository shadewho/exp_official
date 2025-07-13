# File: Exploratory/Exp_UI/interface/drawing/browse_content.py

import os
import bpy
import gpu
import threading
from gpu_extras.batch import batch_for_shader
from ...cache_system.download_helpers import download_thumbnail
from .config import (
    GRID_COLUMNS,
    GRID_ROWS,
    THUMBNAIL_SPACING_FACTOR,
    TEMPLATE_MARGIN_HORIZONTAL,
    TEMPLATE_MARGIN_TOP,
    TEMPLATE_MARGIN_BOTTOM,
    ARROW_LEFT_PATH,
    ARROW_RIGHT_PATH,
    THUMBNAIL_TEXT_COLOR,
    THUMBNAIL_TEXT_ALIGNMENT,
    THUMBNAIL_TEXT_OFFSET_RATIO,
    THUMBNAIL_TEXT_SIZE_RATIO,
    MISSING_THUMB,
)
from ...cache_system.persistence import get_or_load_image, get_or_create_texture
from ...cache_system.manager import cache_manager
from .utilities import build_text_item

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

    # --- 1) Fetch packages for the current page ---
    packages_data = getattr(bpy.types.Scene, 'fetched_packages_data', [])
    scene = bpy.context.scene
    page = scene.current_thumbnail_page
    total_pages = scene.total_thumbnail_pages
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

    thumb_size = min(thumbnail_width, thumbnail_height)

    spacing_x = (usable_width - GRID_COLUMNS * thumb_size) / (GRID_COLUMNS - 1) if GRID_COLUMNS > 1 else 0
    spacing_y = (usable_height - GRID_ROWS * thumb_size) / (GRID_ROWS - 1) if GRID_ROWS > 1 else 0

    # --- 3) Build thumbnail items ---
    for index, pkg in enumerate(visible_packages):
        try:
            pkg_id   = pkg.get("file_id")
            thumb_url = pkg.get("thumbnail_url")

            # 1) check SQLite cache first
            record = cache_manager.get_thumbnail(pkg_id) if pkg_id is not None else None
            if record and os.path.exists(record["file_path"]):
                local_thumb = record["file_path"]
            else:
                local_thumb = pkg.get("local_thumb_path")

            # If it's not on disk, queue a background download with the integer pkg_id, then use placeholder
            if not local_thumb or not os.path.exists(local_thumb):
                if thumb_url and pkg_id is not None:
                    threading.Thread(
                        target=lambda url=thumb_url, fid=pkg_id: download_thumbnail(url, file_id=fid),
                        daemon=True
                    ).start()
                    bpy.app.timers.register(lambda: setattr(bpy.types.Scene, "package_ui_dirty", True), first_interval=0.1)

                local_thumb = MISSING_THUMB
                pkg["local_thumb_path"] = local_thumb

            img = get_or_load_image(local_thumb)
            if not img:
                raise RuntimeError(f"Could not load image: {local_thumb}")
            tex = get_or_create_texture(img)
            if not tex:
                raise RuntimeError(f"Could not create texture: {local_thumb}")

            # Grid cell
            row = index // GRID_COLUMNS
            col = index % GRID_COLUMNS
            thumb_x1 = thumbs_x1 + col * (thumb_size + spacing_x)
            thumb_x2 = thumb_x1 + thumb_size
            thumb_y2 = thumbs_y2 - row * (thumb_size + spacing_y)
            thumb_y1 = thumb_y2 - thumb_size

            # Draw the quad
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
                "shader":  shader,
                "batch":   batch_obj,
                "texture": tex,
                "name":    pkg.get("package_name", "unknown"),
                "pos":     (thumb_x1, thumb_y1, thumb_x2, thumb_y2),
            })

            # Draw the title + stats
            add_thumbnail_title(
                data_list,
                (thumb_x1, thumb_y1, thumb_x2, thumb_y2),
                pkg.get("package_name", "NoName"),
                pkg.get("likes", 0),
                pkg.get("download_count", 0),
            )

        except Exception as e:
            continue

    # --- 4) If there's more than 1 page, draw arrows ---
    if total_pages > 1:
        arrow_scale = 0.03
        arrow_offset_x = 0.48
        arrow_offset_y = -0.05

        # Left arrow
        try:
            left_img = get_or_load_image(ARROW_LEFT_PATH)
            if not left_img:
                raise RuntimeError(f"Missing arrow image: {ARROW_LEFT_PATH}")
            left_tex = get_or_create_texture(left_img)
            if not left_tex:
                raise RuntimeError("Failed to create left arrow texture")

            w, h = left_img.size
            aspect = w / h if h else 1.0
            arrow_w = arrow_scale * template_width
            arrow_h = arrow_w / aspect

            left_x1 = center_x - arrow_offset_x * template_width - arrow_w / 2
            left_y1 = center_y + arrow_offset_y * template_height - arrow_h / 2
            left_x2 = left_x1 + arrow_w
            left_y2 = left_y1 + arrow_h

            shader = gpu.shader.from_builtin('IMAGE')
            verts = [(left_x1, left_y1), (left_x2, left_y1), (left_x2, left_y2), (left_x1, left_y2)]
            coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})

            data_list.append({
                "shader":  shader,
                "batch":   batch,
                "texture": left_tex,
                "name":    "Left_Arrow",
                "pos":     (left_x1, left_y1, left_x2, left_y2),
            })
        except Exception as e:
            print(f"[ERROR] build_browse_content Left arrow skipped: {e}")

        # Right arrow
        try:
            right_img = get_or_load_image(ARROW_RIGHT_PATH)
            if not right_img:
                raise RuntimeError(f"Missing arrow image: {ARROW_RIGHT_PATH}")
            right_tex = get_or_create_texture(right_img)
            if not right_tex:
                raise RuntimeError("Failed to create right arrow texture")

            w, h = right_img.size
            aspect = w / h if h else 1.0
            arrow_w = arrow_scale * template_width
            arrow_h = arrow_w / aspect

            right_x1 = center_x + arrow_offset_x * template_width - arrow_w / 2
            right_y1 = center_y + arrow_offset_y * template_height - arrow_h / 2
            right_x2 = right_x1 + arrow_w
            right_y2 = right_y1 + arrow_h

            shader = gpu.shader.from_builtin('IMAGE')
            verts = [(right_x1, right_y1), (right_x2, right_y1), (right_x2, right_y2), (right_x1, right_y2)]
            coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})

            data_list.append({
                "shader":  shader,
                "batch":   batch,
                "texture": right_tex,
                "name":    "Right_Arrow",
                "pos":     (right_x1, right_y1, right_x2, right_y2),
            })
        except Exception as e:
            print(f"[ERROR] build_browse_content Right arrow skipped: {e}")

    # --- 5) Page label ---
    try:
        page_label = f"Page {page} / {total_pages}"
        text_size = template_height * 0.03
        text_y = y1 + 0.01 * template_height
        data_list.append({
            "type":     "text",
            "content":  page_label,
            "position": ((x1 + x2) / 2, text_y),
            "size":     text_size,
            "color":    (1.0, 1.0, 1.0, 1.0),
            "alignment": 'CENTER',
            "multiline": False
        })
    except Exception as e:
        print(f"[ERROR] build_browse_content page label skipped: {e}")

    return data_list


def add_thumbnail_title(
    data_list,
    thumb_box,
    title_text,
    likes=0,
    download_count=0
):
    """
    Draws two lines of text under the thumbnail:
      1) Title (truncated to 20 chars)
      2) "↓ <download_count> | ♥ <likes>"
    """
    x1, y1, x2, y2 = thumb_box
    thumb_height = (y2 - y1)

    # 1) Truncate title at 20 chars, adding "..."
    max_chars = 20
    if len(title_text) > max_chars:
        title_text = title_text[:(max_chars - 3)] + "..."

    center_x = (x1 + x2) / 2.0

    # 2) Vertical offset from bottom of thumbnail for the title line
    offset_pixels = THUMBNAIL_TEXT_OFFSET_RATIO * thumb_height
    title_y = y1 - offset_pixels

    # 3) Font size
    dynamic_font_size = THUMBNAIL_TEXT_SIZE_RATIO * thumb_height

    # ---- First line: Title ----
    text_label = build_text_item(
        text=title_text,
        x=center_x,
        y=title_y,
        size=dynamic_font_size,
        color=THUMBNAIL_TEXT_COLOR,
        alignment=THUMBNAIL_TEXT_ALIGNMENT,
        multiline=False
    )
    data_list.append(text_label)

    # ---- Second line: Downloads + Likes ----
    meta_y = title_y - (dynamic_font_size * 1.2)
    meta_text = f"↓ {download_count} | ♥ {likes}"

    meta_label = build_text_item(
        text=meta_text,
        x=center_x,
        y=meta_y,
        size=dynamic_font_size * 0.9,
        color=THUMBNAIL_TEXT_COLOR,
        alignment=THUMBNAIL_TEXT_ALIGNMENT,
        multiline=False
    )
    data_list.append(meta_label)
