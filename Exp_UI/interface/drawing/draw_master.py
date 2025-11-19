# Exploratory/Exp_UI/interface/drawing/draw_master.py
import bpy
from .build_template import build_template_and_close
from .detail_content import build_detail_content
from .loading_template import build_loading_indicator
from .explore_loading import build_loading_progress

def load_image_buttons():
    """
    Detail-only UI builder. No browse grid, no pagination, no cache.
    """
    scene = bpy.context.scene
    data  = []

    # template
    data.extend(build_template_and_close())
    template_item = next((b for b in data if b.get("name") == "Template"), None)
    if not template_item:
        return data

    mode = scene.ui_current_mode
    if mode == "DETAIL":
        data.extend(build_detail_content(template_item))
    elif mode == "LOADING":
        data.extend(build_loading_progress(template_item, scene.download_progress))
    elif mode == "GAME":
        return []  # nothing to draw

    if scene.show_loading_image and template_item:
        data.extend(build_loading_indicator(template_item))

    return data
