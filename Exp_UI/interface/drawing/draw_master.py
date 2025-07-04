#Exploratory/Exp_UI/interface/drawing/draw_master.py
import bpy
# the global cache manager at Exp_UI/cache_manager.py
from ...cache_system.manager import cache_manager, filter_cached_data
from .build_template import build_template_and_close
from .browse_content import build_browse_content
from .detail_content import build_detail_content
from .loading_template import build_loading_indicator
from .explore_loading import build_loading_progress
from ...cache_system.db import get_thumbnail_path

def load_image_buttons():
    """
    Build the entire UI data list (images + text items) for the current mode.
    1) If fetched_packages_data is empty, run it through filter_cached_data()
       (which now also applies event-stage + event-selection filters).
    2) Draw the template + close button.
    3) Depending on mode, draw thumbnails, detail, or loading.
    4) If a spinner is requested, overlay it.
    """
    scene = bpy.context.scene

    # ─── 1) PRIME fetched_packages_data if empty ───
    if not getattr(bpy.types.Scene, "fetched_packages_data", None):
        file_type    = scene.package_item_type
        search_query = scene.package_search_query.strip()
        pkgs = filter_cached_data(file_type, search_query) or []

        # annotate each pkg with its on-disk thumbnail
        for pkg in pkgs:
            fid = pkg.get("file_id")
            pkg["local_thumb_path"] = get_thumbnail_path(fid) if fid is not None else None

        # stash for the rest of the UI code to consume
        bpy.types.Scene.fetched_packages_data = pkgs

    data_list = []

    # ─── 2) template + close button ───
    data_list.extend(build_template_and_close())
    template_item = next((b for b in data_list if b.get("name") == "Template"), None)

    # ─── 3) draw content for current mode ───
    mode = scene.ui_current_mode
    if mode == "BROWSE":
        data_list.extend(build_browse_content(template_item))
    elif mode == "DETAIL":
        data_list.extend(build_detail_content(template_item))
    elif mode == "LOADING":
        data_list.extend(build_loading_progress(template_item, scene.download_progress))
    elif mode == "GAME":
        return []  # nothing to draw

    # ─── 4) optional spinner overlay ───
    if scene.show_loading_image and template_item:
        data_list.extend(build_loading_indicator(template_item))

    return data_list
