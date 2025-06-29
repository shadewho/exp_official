#Exploratory/Exp_UI/interface/drawing/build_template.py
import gpu
from gpu_extras.batch import batch_for_shader
from .config import (
    TEMPLATE_ASPECT_RATIO,
    OFFSET_FACTOR,
    TEMPLATE_IMAGE_PATH,
    CLOSE_WINDOW_PATH,
)
from ...cache_system.persistence import get_or_load_image, get_or_create_texture
from .utilities import calculate_free_space, calculate_template_position
from ...cache_system.manager import cache_manager

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