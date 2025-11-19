# Exploratory/Exp_UI/interface/drawing/loading_template.py

import bpy, gpu
from gpu_extras.batch import batch_for_shader
from .config import (
    LOADING_IMAGE_OFFSET_X, LOADING_IMAGE_OFFSET_Y, LOADING_IMAGE_PATH, LOADING_IMAGE_SCALE
)

# local image/tex cache
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

def build_loading_indicator(template_item):
    data_list = []
    x1, y1, x2, y2 = template_item["pos"]
    template_w = x2 - x1
    template_h = y2 - y1

    img = _get_image(LOADING_IMAGE_PATH)
    tex = _get_texture(img) if img else None
    if not tex:
        return data_list

    shader = gpu.shader.from_builtin('IMAGE')
    w, h = img.size
    aspect = (w / h) if h else 1.0

    icon_w = template_w * LOADING_IMAGE_SCALE
    icon_h = icon_w / aspect
    cx, cy = (x1+x2)/2, (y1+y2)/2
    ox, oy = LOADING_IMAGE_OFFSET_X * template_w, LOADING_IMAGE_OFFSET_Y * template_h

    ix1 = cx - icon_w/2 + ox
    iy1 = cy - icon_h/2 + oy
    ix2 = ix1 + icon_w
    iy2 = iy1 + icon_h

    verts  = [(ix1,iy1),(ix2,iy1),(ix2,iy2),(ix1,iy2)]
    coords = [(0,0),(1,0),(1,1),(0,1)]
    batch  = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})

    data_list.append({"shader": shader, "batch": batch, "texture": tex, "name": "Loading_Indicator", "pos": (ix1,iy1,ix2,iy2)})
    return data_list
