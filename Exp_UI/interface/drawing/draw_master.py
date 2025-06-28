#Exploratory/Exp_UI/interface/drawing/draw_master.py
import bpy
# the global cache manager at Exp_UI/cache_manager.py
from ...cache_manager import cache_manager
from .build_template import build_template_and_close
from .browse_content import build_browse_content
from .detail_content import build_detail_content
from .loading_template import build_loading_indicator
from .explore_loading import build_loading_progress

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

    elif scene.ui_current_mode == "LOADING":
        loading_items = build_loading_progress(template_item, scene.download_progress)
        data_list.extend(loading_items)


    elif scene.ui_current_mode == "GAME":
        return []  # Do not build any UI when in game mode.


    # 3) If scene.show_loading_image is True, add the spinner
    if scene.show_loading_image and template_item:
        loading_indicator = build_loading_indicator(template_item)
        data_list.extend(loading_indicator)

    return data_list