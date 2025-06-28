#Exploratory/Exp_UI/interface/operators/fetch.py

import bpy
import os
import threading
import queue

from ...auth.helpers import load_token
from ...internet.helpers import ensure_internet_connection, is_internet_available
from ...helper_functions import download_thumbnail, build_filter_signature
from ...exp_api import fetch_packages
from ...cache_manager import cache_manager, filter_cached_data

from ..drawing.draw_master import load_image_buttons
from ..drawing.utilities import viewport_changed
from ..drawing.config import THUMBNAILS_PER_PAGE

fetch_page_queue = queue.Queue()
load_page_queue = queue.Queue()

# ------------------------------------------------------------------------
# 1) FETCH_PACKAGES_OT_WebApp
#    - Retrieves data from the server
#    - Downloads images into the cached folder
#    - Populates `fetched_packages_data`
# ------------------------------------------------------------------------
class FETCH_PAGE_THREADED_OT_WebApp(bpy.types.Operator):
    """
    Fetches items for the user’s current filter/search or uses cached data,
    always paginating the final data in 8-item pages.
    
    Steps:
      1) Try to retrieve cached data.
      2) If no cache, spawn a background thread to fetch from the server.
      3) As each package’s thumbnail download finishes, update the master list
         (stored on the scene) and immediately re-paginate the UI.
      4) When the full fetch is complete, finalize pagination.
    """
    bl_idname = "webapp.fetch_page"
    bl_label = "Fetch Page (Threaded, Unified Cache+Lazy)"
    bl_options = {'REGISTER'}

    page_number: bpy.props.IntProperty(
        name="Page Number",
        default=1,
        description="Which page to load"
    )

    _timer = None

    def execute(self, context):
        scene = context.scene

        # Force BROWSE mode on every fetch
        scene.ui_current_mode        = 'BROWSE'
        scene.current_thumbnail_page = self.page_number
        scene.show_loading_image     = True
        
        if not ensure_internet_connection(context):
            self.report({'ERROR'}, "No internet connection detected. Cannot fetch packages.")
            return {'CANCELLED'}
        
        file_type = scene.package_item_type  # "world" or "shop_item"
        sort_by = scene.package_sort_by        # "newest", etc.
        search_query = scene.package_search_query.strip()
        requested_page = self.page_number

        if sort_by == 'featured':
            file_type = 'featured'
            
        # Use the new helper to filter cached data
        cached_filtered = filter_cached_data(file_type, search_query)

        if cached_filtered:
            self.report({'INFO'}, f"Using cached data: {len(cached_filtered)} items found for filter '{sort_by}'.")
            # Immediately finalize UI with available cached data
            self._finalize_data(context, cached_filtered, requested_page, sort_by)

            # If cached data is insufficient for a full page, start a lazy load to fetch missing items
            if len(cached_filtered) < THUMBNAILS_PER_PAGE:
                self.lazy_load_missing_data(file_type, sort_by, search_query, len(cached_filtered))
            return {'FINISHED'}
        else:
            self.report({'INFO'}, "No cached data found; fetching full data from server.")
            # Proceed with the full server fetch in a background thread
            threading.Thread(target=self._fetch_worker, args=(file_type, sort_by, search_query), daemon=True).start()
            scene.show_loading_image = True
            wm = context.window_manager
            self._timer = wm.event_timer_add(0.2, window=context.window)
            wm.modal_handler_add(self)
            return {'RUNNING_MODAL'}


    def modal(self, context, event):
        if event.type == 'TIMER':
            # Process partial or final results from the background thread.
            while True:
                try:
                    result_type, payload = fetch_page_queue.get_nowait()

                    if result_type == "PACKAGE_LIST":
                        # The server returned the complete list but we haven't processed thumbnails.
                        pass

                    elif result_type == "PACKAGE_READY":
                        # Process each package as it arrives.
                        self._handle_partial_thumb(context, payload)

                    elif result_type == "FETCH_DONE":
                        # Full fetch complete: finalize pagination.
                        final_list = payload["packages"]
                        sort_by = payload["sort_by"]
                        self.report({'INFO'}, f"Server fetch done: {len(final_list)} items total.")

                        context.scene.show_loading_image = False
                        self._finalize_data(context, final_list, self.page_number, sort_by)

                        wm = context.window_manager
                        wm.event_timer_remove(self._timer)
                        return {'FINISHED'}

                    elif result_type == "FETCH_ERROR":
                        err_msg = payload["error"]
                        self.report({'ERROR'}, f"Server fetch failed: {err_msg}")
                        context.scene.show_loading_image = False
                        wm = context.window_manager
                        wm.event_timer_remove(self._timer)
                        return {'CANCELLED'}

                except queue.Empty:
                    break

            if viewport_changed() and hasattr(bpy.types.Scene, "gpu_image_buttons_data"):
                bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
                if context.area:
                    context.area.tag_redraw()

        if event.type == 'ESC':
            wm = context.window_manager
            wm.event_timer_remove(self._timer)
            context.scene.show_loading_image = False
            self.report({'WARNING'}, "Fetch canceled by user.")
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def _fetch_worker(self, file_type, sort_by, search_query):
        token = load_token()
        if not token:
            fetch_page_queue.put(("FETCH_ERROR", {"error": "Not logged in. Fetch cancelled."}))
            return

        if not is_internet_available():
            fetch_page_queue.put(("FETCH_ERROR", {"error": "No internet connection detected."}))
            return

        try:
            params = {
                "file_type": file_type,
                "sort_by":   sort_by,
                "offset":    0,
                "limit":     9999
            }
            if search_query:
                params["search_query"] = search_query

            # For event files, add filters for event stage and the selected event.
            if file_type == 'event':
                scene = bpy.context.scene
                params["event_stage"] = scene.event_stage
                if hasattr(scene, "selected_event") and scene.selected_event and scene.selected_event != "0":
                    params["selected_event"] = scene.selected_event

            data = fetch_packages(params)
            if not data.get("success"):
                fetch_page_queue.put(("FETCH_ERROR", {"error": data.get("message", "Unknown error")}))
                return

            packages = data.get("packages", [])

            # ——— PRIME THE CACHE FOR THIS file_type ———
            cache_manager.set_package_data({ file_type: packages })

            # Tell the UI we have the full list
            fetch_page_queue.put(("PACKAGE_LIST", {"packages": packages}))

            # Process each package progressively.
            for pkg in packages:
                thumb_url = pkg.get("thumbnail_url", "")
                if thumb_url:
                    package_id = pkg.get("file_id")
                    cached = cache_manager.get_thumbnail(package_id)
                    if cached and os.path.exists(cached["file_path"]):
                        pkg["local_thumb_path"] = cached["file_path"]
                    else:
                        local_path = download_thumbnail(thumb_url, file_id=package_id)
                        pkg["local_thumb_path"] = local_path
                else:
                    pkg["local_thumb_path"] = None

                fetch_page_queue.put(("PACKAGE_READY", {"package": pkg, "sort_by": sort_by}))

            fetch_page_queue.put(("FETCH_DONE", {"packages": packages, "sort_by": sort_by}))

        except Exception as e:
            fetch_page_queue.put(("FETCH_ERROR", {"error": str(e)}))



    def update_pagination(self, context, master_list, current_page, items_per_page=THUMBNAILS_PER_PAGE):
        """
        Sort the master list, chunk it into pages, and update scene properties so that
        the UI displays only the current page.
        """
        sorted_list = self._sort_packages(master_list, context.scene.package_sort_by)
        pages = [sorted_list[i:i + items_per_page] for i in range(0, len(sorted_list), items_per_page)]
        total_pages = len(pages)
        context.scene.total_thumbnail_pages = total_pages

        if not hasattr(bpy.types.Scene, "all_pages_data"):
            bpy.types.Scene.all_pages_data = {}
        bpy.types.Scene.all_pages_data.clear()
        for i, chunk in enumerate(pages, start=1):
            bpy.types.Scene.all_pages_data[i] = chunk

        # Clamp current page.
        if current_page < 1:
            current_page = 1
        if current_page > total_pages:
            current_page = total_pages
        context.scene.current_thumbnail_page = current_page

        if pages:
            bpy.types.Scene.fetched_packages_data = pages[current_page - 1]
        else:
            bpy.types.Scene.fetched_packages_data = []
        if context.area:
            context.area.tag_redraw()

    def _handle_partial_thumb(self, context, payload):
        """
        Called for each PACKAGE_READY event. Adds or updates the package in a master list
        stored on the scene, then calls update_pagination so the UI reflects the current page.
        """
        pkg = payload["package"]
        master_list = getattr(bpy.types.Scene, "master_package_list", None)
        if master_list is None:
            bpy.types.Scene.master_package_list = []
            master_list = bpy.types.Scene.master_package_list

        pkg_id = pkg.get("file_id")
        replaced = False
        for i, existing in enumerate(master_list):
            if existing.get("file_id") == pkg_id:
                master_list[i] = pkg
                replaced = True
                break
        if not replaced:
            master_list.append(pkg)

        self._dirty = True

        current_page = context.scene.current_thumbnail_page
        self.update_pagination(context, master_list, current_page)

    def _finalize_data(self, context, all_packages, requested_page, sort_by):
        """
        After the full fetch is complete, sort, chunk, and store the data into
        scene.all_pages_data and scene.fetched_packages_data.
        """
        scene = context.scene
        sorted_list = self._sort_packages(all_packages, sort_by)
        page_chunks = []
        for i in range(0, len(sorted_list), THUMBNAILS_PER_PAGE):
            page_chunks.append(sorted_list[i:i + THUMBNAILS_PER_PAGE])
        if not page_chunks:
            page_chunks = [[]]

        total_pages = len(page_chunks)
        scene.total_thumbnail_pages = total_pages

        if not hasattr(bpy.types.Scene, "all_pages_data"):
            bpy.types.Scene.all_pages_data = {}
        bpy.types.Scene.all_pages_data.clear()
        for i, chunk in enumerate(page_chunks, start=1):
            bpy.types.Scene.all_pages_data[i] = chunk

        if requested_page < 1:
            requested_page = 1
        if requested_page > total_pages:
            requested_page = total_pages
        scene.current_thumbnail_page = requested_page

        bpy.types.Scene.fetched_packages_data = page_chunks[requested_page - 1]

        scene.last_filter_signature = build_filter_signature(scene)

        self._dirty = True
        
        if hasattr(bpy.types.Scene, "gpu_image_buttons_data"):
            bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
        if context.area:
            context.area.tag_redraw()

        self.report({'INFO'}, f"Pagination done. {len(sorted_list)} items total, {total_pages} pages.")

    def _sort_packages(self, pkgs, sort_by):
        """
        Basic sorting logic. Adjust as needed.
        """
        if sort_by == "newest":
            return sorted(pkgs, key=lambda p: p.get("upload_date", ""), reverse=True)
        elif sort_by == "oldest":
            return sorted(pkgs, key=lambda p: p.get("upload_date", ""))
        elif sort_by == "popular":
            return sorted(pkgs, key=lambda p: p.get("likes", 0), reverse=True)
        elif sort_by == "random":
            import random
            new_list = list(pkgs)
            random.shuffle(new_list)
            return new_list
        else:
            return pkgs

    def _get_all_cached_filtered(self, file_type, sort_by, search_query):
        """
        Retrieve cached data from cache_manager and filter it based on file_type and search_query.
        """
        all_data = cache_manager.get_package_data()  # e.g., {1: [pkg1, pkg2...], 2: [...]}
        big_list = all_data.get(1, [])
        def matches(pkg):
            if pkg.get("file_type") != file_type:
                return False
            name = pkg.get("package_name", "").lower()
            auth = pkg.get("uploader", "").lower()
            sq = search_query.lower()
            if sq and (sq not in name and sq not in auth):
                return False
            return True
        filtered = [p for p in big_list if matches(p)]
        return filtered if filtered else None
    
    def lazy_load_missing_data(self, file_type, sort_by, search_query, current_count):
        additional_needed = THUMBNAILS_PER_PAGE - current_count
        threading.Thread(
            target=self._lazy_load_worker,
            args=(file_type, sort_by, search_query, additional_needed),
            daemon=True
        ).start()

    def _lazy_load_worker(self, file_type, sort_by, search_query, additional_needed):
        # Prepare parameters to fetch only the additional needed items.
        params = {
            "file_type": file_type,
            "sort_by":   sort_by,
            "offset":    0,
            "limit":     additional_needed
        }
        if search_query:
            params["search_query"] = search_query

        try:
            data = fetch_packages(params)
            if data.get("success"):
                new_items = data.get("packages", [])

                # Merge new items with the existing cache for this type
                current_cache = cache_manager.get_package_data().get(file_type, [])
                updated_cache = current_cache + new_items
                cache_manager.set_package_data({file_type: updated_cache})

                # Schedule a re-fetch of the current page so UI repaginates with the new cache
                bpy.app.timers.register(
                    lambda: bpy.ops.webapp.fetch_page('EXEC_DEFAULT', page_number=self.page_number),
                    first_interval=0.1
                )
            else:
                print("Lazy load failed: " + data.get("message", "Unknown error"))
        except Exception as e:
            print("Error in lazy load: ", e)
