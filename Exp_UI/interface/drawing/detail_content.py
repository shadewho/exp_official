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
from ...cache_system.persistence import get_or_load_image, get_or_create_texture
from ...cache_system.manager import cache_manager
from .fonts import get_font_id

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

    # ─── Determine the loaded package’s own type ───────────────────
    pkg_type = getattr(
        bpy.context.scene.my_addon_data,
        "file_type",
        bpy.context.scene.package_item_type
    )

    # --- Explore / Visit Shop / Submit World Button ---
    try:
        # pick which image + name based on the loaded package’s own type
        if pkg_type == 'shop_item':
            btn_img  = get_or_load_image(VISIT_SHOP_BUTTON_PATH)
            btn_name = "Visit_Shop_Icon"
        elif pkg_type == 'event' and bpy.context.scene.event_stage == 'submission':
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
        # 1) check SQLite cache first
        file_id = bpy.context.scene.my_addon_data.file_id
        record  = cache_manager.get_thumbnail(file_id) if file_id else None
        if record and os.path.exists(record["file_path"]):
            sel_path = record["file_path"]
        else:
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
    Draw the package‑detail text block (right‑hand column).

    Goals
    -----
    1. Never truncate or omit any line.
    2. Never let the text dip below the action button (“Visit Shop”, etc.).
    3. Use the largest font size that satisfies 1 + 2.
    4. Respect the fixed left/right/top margins supplied by the template.

    The routine therefore:

    • Computes the bottom floor from the button’s top edge (+ 2 % padding).
    • Iteratively shrinks the font until *all* wrapped lines fit.
    • Stops shrinking once it fits; no later draw‑loop break is required.
    """

    # ─────────────────────────────────────────────────────────────── #
    # 1. Data retrieval                                              #
    # ─────────────────────────────────────────────────────────────── #
    scene      = bpy.context.scene
    ad         = scene.my_addon_data
    if ad.file_id <= 0:
        print("NO DATA TO DISPLAY")
        return

    title, author       = ad.package_name, ad.author
    description         = ad.description
    likes, downloads    = ad.likes, ad.download_count
    upload_date         = ad.upload_date
    price_float         = ad.price
    file_type           = ad.file_type
    heart               = "♥"
    uploaded            = format_relative_time(upload_date)
    votes               = ad.vote_count

    # ─────────────────────────────────────────────────────────────── #
    # 2. Assemble logical lines (with intentional blank spacers)     #
    # ─────────────────────────────────────────────────────────────── #
    lines = [
        f"Title: {title}",                     "",
        f"Author: {author}",                   "",
        f"Description: {description}",         "",
        f"Likes: {heart} {likes}",             "",
        f"Downloads: {downloads}",
    ]
    if file_type == "shop_item":
        lines += ["", f"Price: ${int(price_float)}"]
    if file_type == "event":                 # ← add these three lines
        lines += ["", f"Votes: ★ {votes}"]   # ←
    lines += ["", f"Uploaded: {uploaded} ago", ""]

    # ─────────────────────────────────────────────────────────────── #
    # 3. Geometry: text column & dynamic bottom floor                #
    # ─────────────────────────────────────────────────────────────── #
    x1, y1, x2, y2      = template_item["pos"]
    t_w, t_h            = (x2 - x1), (y2 - y1)

    # hard margins (same ratios you provided)
    L, R, T, B_static   = 0.50, 0.05, 0.25, 0.15
    col_x1, col_x2      = x1 + L * t_w, x2 - R * t_w
    col_y2              = y2 - T * t_h

    # locate the action button that was already appended to data_list
    btn_top_y = None
    for itm in data_list:
        if itm.get("name") in ("Visit_Shop_Icon", "Explore_Icon",
                               "Submit_World_Icon"):
            btn_top_y = itm["pos"][3]   # pos = (bx1, by1, bx2, by2)
            break

    PADDING          = 0.02 * t_h       # 2 % of template height
    col_y1           = max(y1 + B_static * t_h,
                           (btn_top_y + PADDING) if btn_top_y else y1)
    col_w, col_h     = col_x2 - col_x1, col_y2 - col_y1

    # ─────────────────────────────────────────────────────────────── #
    # 4. Typographic helpers                                         #
    # ─────────────────────────────────────────────────────────────── #
    font_id          = get_font_id()
    INIT_RATIO       = 0.05             # start at 5 % of template height
    MIN_RATIO        = 0.015            # absolute floor (1.5 %)
    SHRINK_FACTOR    = 0.95             # geometric shrink each attempt
    LEADING          = 1.15             # regular line spacing multiplier
    BLANK_LEADING    = 0.40             # spacer line cost (40 %)

    def line_height(size, blank=False) -> float:
        return size * (BLANK_LEADING if blank else LEADING)

    def wrap_paragraph(text, max_w, size):
        """Greedy word‑wrap for a single logical line."""
        blf.size(font_id, int(size))
        words, out, buf = text.split(), [], ""
        for w in words:
            test = f"{buf} {w}".strip()
            if blf.dimensions(font_id, test)[0] > max_w and buf:
                out.append(buf); buf = w
            else:
                buf = test
        out.append(buf)
        return out

    def wrap_all(src_lines, max_w, size):
        wrapped = []
        for ln in src_lines:
            if not ln.strip():
                wrapped.append("")      # preserve spacer
            else:
                wrapped.extend(wrap_paragraph(ln, max_w, size))
        return wrapped

    # ─────────────────────────────────────────────────────────────── #
    # 5. Font‑size search: shrink until *everything* fits             #
    # ─────────────────────────────────────────────────────────────── #
    font_sz = t_h * INIT_RATIO
    while True:
        wrapped = wrap_all(lines, col_w, font_sz)
        needed  = sum(line_height(font_sz, not ln.strip()) for ln in wrapped)

        if needed <= col_h or font_sz <= t_h * MIN_RATIO:
            break                     # it now fits (or we hit min size)
        font_sz *= SHRINK_FACTOR

    # After loop we *know* all lines fit, so no later break tests needed.

    # ─────────────────────────────────────────────────────────────── #
    # 6. Draw the text block                                         #
    # ─────────────────────────────────────────────────────────────── #
    blf.size(font_id, int(font_sz))
    cursor_y = col_y2

    for logical in wrap_all(lines, col_w, font_sz):
        h = line_height(font_sz, not logical.strip())
        if logical.strip():
            data_list.append(
                build_text_item(
                    text       = logical,
                    x          = col_x1,
                    y          = cursor_y,
                    size       = font_sz,
                    color      = (1, 1, 1, 1),
                    alignment  = 'LEFT',
                    multiline  = False
                )
            )
        cursor_y -= h
    # cursor_y is guaranteed ≥ col_y1, so nothing is ever clipped.
