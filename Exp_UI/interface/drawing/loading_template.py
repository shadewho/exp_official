#Exploratory/Exp_UI/interface/drawing/explore_loading.py
import gpu
from gpu_extras.batch import batch_for_shader
from .config import (
    LOADING_IMAGE_OFFSET_X,
    LOADING_IMAGE_OFFSET_Y,
    LOADING_IMAGE_PATH,
    LOADING_IMAGE_SCALE,
)
from ...cache_system.persistence import get_or_load_image, get_or_create_texture
from ...cache_system.manager import cache_manager

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