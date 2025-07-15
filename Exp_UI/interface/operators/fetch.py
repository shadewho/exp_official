#Exploratory/Exp_UI/interface/operators/fetch.py

import bpy
import os
import threading
import queue
import hashlib
import random
from ...auth.helpers import load_token
from ...internet.helpers import ensure_internet_connection, is_internet_available
from ...cache_system.download_helpers import download_thumbnail
from .utilities import build_filter_signature, fetch_packages
from ...cache_system.manager import cache_manager, filter_cached_data

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

        # ─── reset pagination state ────────────────────────
        bpy.types.Scene.master_package_list = []
        bpy.types.Scene.all_pages_data     = {}

        # ── 0) Compute filter parameters ────────────────────────────────
        file_type = scene.package_item_type

        # Always force events to only ever sort by newest
        if file_type == 'event':
            sort_by = 'newest'
            # keep the UI button in sync
            scene.package_sort_by = 'newest'
        else:
            sort_by = scene.package_sort_by

        search_query   = scene.package_search_query.strip()
        requested_page = self.page_number

        # ── 0.5) Compute event filters ───────────────────────────────────
        event_stage    = scene.event_stage    if file_type == 'event' else ""
        selected_event = scene.selected_event if file_type == 'event' else ""

        # ── 1) Store for modal later ─────────────────────────────────────
        self._file_type      = file_type
        self._sort_by        = sort_by
        self._search_query   = search_query
        self._page_number    = requested_page
        self._event_stage    = event_stage
        self._selected_event = selected_event

        # ── 2) Ensure connectivity ───────────────────────────────────────
        if not ensure_internet_connection(context):
            self.report({'ERROR'}, "No internet connection detected. Cannot fetch packages.")
            return {'CANCELLED'}

        # ── 3) Try loading from SQLite cache first ───────────────────────
        from ...cache_system.db import load_package_list
        pkg_list = load_package_list(file_type, event_stage, selected_event)
        if pkg_list:
            for pkg in pkg_list:
                pkg.update({
                    'file_type':      file_type,
                    'event_stage':    event_stage,
                    'selected_event': selected_event,
                })
            cache_manager.set_package_data({file_type: pkg_list})

        # ── 4) Filter against in-memory cache ────────────────────────────
        cached_filtered = filter_cached_data(file_type, search_query)
        if cached_filtered:
            # only keep featured items when requested
            if scene.package_sort_by == 'featured':
                cached_filtered = [pkg for pkg in cached_filtered if pkg.get("is_featured", False)]

            # display immediately
            self._finalize_data(context, cached_filtered, requested_page, sort_by)
            # if we need more than the cache has, lazy-load the rest
            if len(cached_filtered) < THUMBNAILS_PER_PAGE:
                self.lazy_load_missing_data(
                    file_type, sort_by, search_query, len(cached_filtered)
                )
            return {'FINISHED'}

        # ── 5) Fallback: no cache → fetch from server in background ───────
        threading.Thread(
            target=self._fetch_worker,
            args=(file_type, sort_by, search_query),
            daemon=True
        ).start()

        # Show loading indicator
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
                        sort_by = payload["sort_by"]

                        # stop spinner
                        context.scene.show_loading_image = False

                        # **always** take the freshly-cached data and run it through filter_cached_data()
                        filtered = filter_cached_data(self._file_type, self._search_query)
                        self._finalize_data(context, filtered, self.page_number, sort_by)

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
        """
        Background thread to fetch packages (with unified caching + lazy thumbnail download),
        now tagging each package with file_type, event_stage and selected_event so that
        filter_cached_data() can correctly display event‐stage‐specific items.
        """
        token = load_token()
        if not token:
            fetch_page_queue.put(("FETCH_ERROR", {"error": "Not logged in. Fetch cancelled."}))
            return

        if not is_internet_available():
            fetch_page_queue.put(("FETCH_ERROR", {"error": "No internet connection detected."}))
            return

        try:
            # ─── Build request parameters ───────────────────────────────────────────
            params = {
                "file_type": file_type,
                "sort_by":   sort_by,
                "offset":    0,
                "limit":     9999,
            }
            if search_query:
                params["search_query"] = search_query

            # ─── Apply event‐specific filters ─────────────────────────────────────
            if file_type == 'event':
                # use the values captured in execute()
                params["event_stage"]    = self._event_stage
                if self._selected_event and self._selected_event != "0":
                    params["selected_event"] = self._selected_event

            # ─── Fetch from server ───────────────────────────────────────────────
            data = fetch_packages(params)
            if not data.get("success"):
                fetch_page_queue.put(("FETCH_ERROR", {"error": data.get("message", "Unknown error")}))
                return

            packages = data.get("packages", [])

            # ─── Annotate each package for unified cache filtering ───────────────
            for pkg in packages:
                pkg["file_type"]      = file_type
                pkg["event_stage"]    = self._event_stage
                pkg["selected_event"] = self._selected_event

            # ─── Prime the in‐memory cache and notify UI ─────────────────────────
            cache_manager.set_package_data({file_type: packages})
            fetch_page_queue.put(("PACKAGE_LIST", {"packages": packages}))

            # ─── Download (or load) each thumbnail and stream updates to UI ──────
            for pkg in packages:
                pkg_id    = pkg.get("file_id")
                thumb_url = pkg.get("thumbnail_url", "")

                if thumb_url and pkg_id is not None:
                    record = cache_manager.get_thumbnail(pkg_id)
                    if record and os.path.exists(record["file_path"]):
                        pkg["local_thumb_path"] = record["file_path"]
                    else:
                        pkg["local_thumb_path"] = download_thumbnail(thumb_url, file_id=pkg_id)
                else:
                    pkg["local_thumb_path"] = None

                fetch_page_queue.put(("PACKAGE_READY", {"package": pkg, "sort_by": sort_by}))

            # ─── Signal completion ───────────────────────────────────────────────
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

    def _finalize_data(self, context, packages_to_page, requested_page, sort_by):
        """
        Paginate & display exactly the list you pass in:
        - `packages_to_page` must already be run through filter_cached_data().
        - No filtering or event logic here.
        - If sort_by=='featured' and there are no items, show nothing.
        """
        scene = context.scene

        # 1) Sort the incoming list
        sorted_list = self._sort_packages(packages_to_page, sort_by)

        # 2) Build pages—but if we're in "featured" mode and got zero items, keep pages empty
        if sort_by == 'featured' and not sorted_list:
            pages = []
        else:
            pages = [
                sorted_list[i : i + THUMBNAILS_PER_PAGE]
                for i in range(0, len(sorted_list), THUMBNAILS_PER_PAGE)
            ]

        # 3) Update total pages
        total = len(pages)
        scene.total_thumbnail_pages = total

        # 4) Clamp & write out the current page + data
        if total == 0:
            scene.current_thumbnail_page = 0
            bpy.types.Scene.fetched_packages_data = []
        else:
            page = max(1, min(requested_page, total))
            scene.current_thumbnail_page = page
            bpy.types.Scene.fetched_packages_data = pages[page - 1]

        # 5) Clear spinner, rebuild UI
        scene.show_loading_image = False
        self._dirty = True
        if hasattr(bpy.types.Scene, "gpu_image_buttons_data"):
            bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
        if context.area:
            context.area.tag_redraw()

    def _sort_packages(self, pkgs, sort_by):
        scene = bpy.context.scene

        if sort_by == "newest":
            return sorted(pkgs, key=lambda p: p.get("upload_date", ""), reverse=True)
        elif sort_by == "oldest":
            return sorted(pkgs, key=lambda p: p.get("upload_date", ""))
        elif sort_by == "popular":
            return sorted(pkgs, key=lambda p: p.get("likes", 0), reverse=True)
        elif sort_by == "random":
            # Build a stable seed from the current filter signature
            sig = getattr(scene, "last_filter_signature", "")
            seed = int(hashlib.sha256(sig.encode('utf-8')).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            shuffled = list(pkgs)
            rng.shuffle(shuffled)
            return shuffled
        else:
            # for any custom sort modes, just return a copy
            return list(pkgs)

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
        params = {
            "file_type": file_type,
            "sort_by":   sort_by,
            "offset":    0,
            "limit":     additional_needed
        }
        if search_query:
            params["search_query"] = search_query

        try:
            resp = fetch_packages(params)
            if resp.get("not_modified"):
                # nothing changed → skip
                return
            if not resp.get("success"):
                return

            new_items = resp["packages"]
            current_cache = cache_manager.get_package_data().get(file_type, [])
            cache_manager.set_package_data({file_type: current_cache + new_items})

            # re-fire your page fetch
            page = self.page_number
            bpy.app.timers.register(
                lambda page=page: bpy.ops.webapp.fetch_page('EXEC_DEFAULT', page_number=page),
                first_interval=0.1
            )
        except Exception as e:
            print("Error in lazy load:", e)
