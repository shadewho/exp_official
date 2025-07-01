#Exploratory/Exp_UI/interface/drawing/draw_master.py
import bpy
# the global cache manager at Exp_UI/cache_manager.py
from ...cache_system.manager import cache_manager
from .build_template import build_template_and_close
from .browse_content import build_browse_content
from .detail_content import build_detail_content
from .loading_template import build_loading_indicator
from .explore_loading import build_loading_progress
from ...cache_system.db import get_thumbnail_path

def load_image_buttons():
    """
    Build the entire UI data list (images + text items) for the current mode (BROWSE or DETAIL).
    1) If we haven’t yet populated fetched_packages_data, pull it directly from cache_manager + SQLite.
    2) Draw the background template + close button.
    3) Draw either the thumbnail grid, the detail view, or loading indicators.
    """
    scene = bpy.context.scene

    # ─── 1) PRIME fetched_packages_data from disk/SQLite if empty
    if not getattr(bpy.types.Scene, "fetched_packages_data", None):
        # grab whatever’s in the package_list table for this type
        pkgs = cache_manager.get_package_data().get(scene.package_item_type, [])
        # annotate each dict with its on-disk thumbnail path
        for pkg in pkgs:
            fid = pkg.get("file_id")
            if fid is not None:
                pkg["local_thumb_path"] = get_thumbnail_path(fid)
        # stash into the Scene so build_browse_content sees it
        bpy.types.Scene.fetched_packages_data = pkgs

    # (A) Which page are we on?
    current_page = scene.current_thumbnail_page

    # (C) Start building the draw items
    data_list = []

    # 2) Build the background template + close button
    template_and_close = build_template_and_close()
    data_list.extend(template_and_close)

    # 3) Find the template item in data_list
    template_item = next((b for b in data_list if b.get("name") == "Template"), None)

    # 4) Depending on mode, draw thumbnails, detail, or loading
    if scene.ui_current_mode == "BROWSE":
        browse_items = build_browse_content(template_item)
        data_list.extend(browse_items)

    elif scene.ui_current_mode == "DETAIL":
        detail_items = build_detail_content(template_item)
        data_list.extend(detail_items)

    elif scene.ui_current_mode == "LOADING":
        loading_items = build_loading_progress(template_item, scene.download_progress)
        data_list.extend(loading_items)

    elif scene.ui_current_mode == "GAME":
        return []  # nothing to draw in game mode

    # 5) If we’re showing the spinner overlay
    if scene.show_loading_image and template_item:
        spinner = build_loading_indicator(template_item)
        data_list.extend(spinner)

    return data_list
