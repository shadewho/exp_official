# Exploratory/Exp_UI/interface/drawing/build_template.py

import bpy, gpu
from gpu_extras.batch import batch_for_shader
from .config import TEMPLATE_ASPECT_RATIO, OFFSET_FACTOR, TEMPLATE_IMAGE_PATH, CLOSE_WINDOW_PATH
from .utilities import calculate_free_space, calculate_template_position

# minimal in-memory loader
_LOADED_IMAGES, _LOADED_TEX = {}, {}

def _get_image(path):
    if not path:
        return None
    img = _LOADED_IMAGES.get(path)
    try:
        _ = img.name
    except Exception:
        img = None

    if img is None:
        try:
            img = bpy.data.images.load(path, check_existing=True)
        except Exception:
            return None

    # Ensure UI-safe settings even for cached images
    try:
        cs = getattr(img, "colorspace_settings", None)
        if cs and getattr(cs, "name", None) != "Non-Color":
            cs.name = "Non-Color"   # critical: bypass color management for UI art
    except Exception:
        pass
    try:
        img.alpha_mode = 'STRAIGHT'  # harmless for JPG (no alpha), prevents fringes for PNGs
    except Exception:
        pass

    _LOADED_IMAGES[path] = img
    return img


def _get_texture(img):
    if not img:
        return None
    key = img.name
    tex = _LOADED_TEX.get(key)
    if tex:
        return tex
    try:
        tex = gpu.texture.from_image(img)
        _LOADED_TEX[key] = tex
        return tex
    except Exception:
        return None

def build_template_and_close():
    data_list = []

    free_space = calculate_free_space()
    if not free_space:
        return data_list

    template_pos = calculate_template_position(free_space, TEMPLATE_ASPECT_RATIO, OFFSET_FACTOR)
    x1, y1, x2, y2 = template_pos["x1"], template_pos["y1"], template_pos["x2"], template_pos["y2"]

    # Template
    template_img = _get_image(TEMPLATE_IMAGE_PATH)
    if template_img:
        shader = gpu.shader.from_builtin('IMAGE')
        tex    = _get_texture(template_img)
        if tex:
            verts  = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            coords = [(0,0), (1,0), (1,1), (0,1)]
            batch  = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})
            data_list.append({"shader": shader, "batch": batch, "texture": tex, "name": "Template", "pos": (x1,y1,x2,y2)})

    # Close button (top-right)
    close_img = _get_image(CLOSE_WINDOW_PATH)
    if close_img:
        tex = _get_texture(close_img)
        if tex:
            shader = gpu.shader.from_builtin('IMAGE')
            w, h   = close_img.size
            aspect = (w / h) if h else 1.0
            close_w = 0.05 * (x2 - x1)
            close_h = close_w / aspect
            margin_right = 0.025 * (x2 - x1)
            margin_top   = 0.035 * (y2 - y1)
            cx2 = x2 - margin_right
            cx1 = cx2 - close_w
            cy2 = y2 - margin_top
            cy1 = cy2 - close_h
            verts  = [(cx1,cy1),(cx2,cy1),(cx2,cy2),(cx1,cy2)]
            coords = [(0,0),(1,0),(1,1),(0,1)]
            batch  = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})
            data_list.append({"shader": shader, "batch": batch, "texture": tex, "name": "Close_Icon", "pos": (cx1,cy1,cx2,cy2)})

    return data_list
