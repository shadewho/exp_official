# operators.py

import bpy
import os
from ..auth import load_token, ensure_internet_connection, is_internet_available
from .drawing import load_image_buttons, draw_image_buttons_callback
from .utils import viewport_changed
from .config import THUMBNAILS_PER_PAGE
from ..helper_functions import download_thumbnail
from ..main_config import PACKAGES_ENDPOINT, EVENTS_URL, SHOP_URL
from ..exp_api import fetch_packages, fetch_detail_for_file
import threading
import queue
from ..cache_manager import cache_manager, filter_cached_data
from .explore_downloads import explore_icon_handler
# A global queue for background fetch results
fetch_page_queue = queue.Queue()
load_page_queue = queue.Queue()

LOADED_PAGE_DATA = {}
THUMBNAILS_PER_PAGE = 8

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
        if not ensure_internet_connection(context):
            self.report({'ERROR'}, "No internet connection detected. Cannot fetch packages.")
            return {'CANCELLED'}
        scene = context.scene
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
                "sort_by": sort_by,
                "offset": 0,
                "limit": 9999
            }
            if search_query:
                params["search_query"] = search_query

            # For event files, add filters for event stage and the selected event.
            if file_type == 'event':
                scene = bpy.context.scene
                params["event_stage"] = scene.event_stage
                # If the user has selected a specific event (and it's not the default "0")
                if hasattr(scene, "selected_event") and scene.selected_event and scene.selected_event != "0":
                    params["selected_event"] = scene.selected_event

            data = fetch_packages(params)
            if not data.get("success"):
                fetch_page_queue.put(("FETCH_ERROR", {"error": data.get("message", "Unknown error")}))
                return

            packages = data.get("packages", [])
            fetch_page_queue.put(("PACKAGE_LIST", {"packages": packages}))

            # Process each package progressively.
            for pkg in packages:
                thumb_url = pkg.get("thumbnail_url", "")
                if thumb_url:
                    # Check cache using file_id (assumed to be unique for each package)
                    package_id = pkg.get("file_id")
                    cached_thumb = cache_manager.get_thumbnail(package_id)
                    if cached_thumb and os.path.exists(cached_thumb["file_path"]):
                        pkg["local_thumb_path"] = cached_thumb["file_path"]
                    else:
                        # Download and cache the thumbnail; pass file_id for consistent caching.
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
            "sort_by": sort_by,
            "offset": 0,  # You might adjust the offset if you want to start after cached items
            "limit": additional_needed
        }
        if search_query:
            params["search_query"] = search_query
        try:
            data = fetch_packages(params)
            if data.get("success"):
                new_items = data.get("packages", [])
                # Merge new items with the existing cache for page 1
                current_cache = cache_manager.get_package_data().get(1, [])
                updated_cache = current_cache + new_items
                cache_manager.set_package_data({1: updated_cache})
                # Trigger a UI refresh on the main thread
                bpy.app.timers.register(lambda: self._finalize_data(bpy.context, 
                                                                    filter_cached_data(file_type, search_query), 
                                                                    self.page_number, sort_by), first_interval=0.1)
            else:
                print("Lazy load failed: " + data.get("message", "Unknown error"))
        except Exception as e:
            print("Error in lazy load: ", e)

# ------------------------------------------------------------------------
# 2) PACKAGE_OT_Display
#    - Displays images found in `bpy.types.Scene.fetched_packages_data`
#      (assuming their thumbnails are in `CACHED_IMAGE_FOLDER`)
#    - Minimal 2:1 template, 4x2 grid for images
# -----------------------------------------------------------------------

class PACKAGE_OT_Display(bpy.types.Operator):
    """
    Displays the UI in the 3D Viewport using load_image_buttons() from drawing.py.
    This modal operator continuously monitors filter properties and updates its data
    when they change, while keeping the UI visible.
    """
    bl_idname = "view3d.add_package_display"
    bl_label = "Show Package Thumbnails"
    bl_options = {'REGISTER'}

    keep_mode: bpy.props.BoolProperty(
        name="Keep Mode",
        default=False,
        description="Do not override the current UI mode"
    )

    # Internal modal state
    _handler = None
    _timer = None
    _dirty   = True  # start dirty so first draw builds

    # Saved filter values (for monitoring changes)
    last_item_type: str = ""
    last_sort_by: str = ""
    last_search: str = ""
    last_event_stage: str = ""  # New: store the last event stage
    last_selected_event: str = ""  # New: store the last selected event

    # Loading state (for when a page/detail refresh is in progress)
    _do_loading = False
    loading_step = 0
    _active_task = ""           # "page" or "detail"
    _target_page = 0            # target page number when refreshing page data
    _detail_file_id = 0         # file_id to fetch when loading detail view

    _original_area_type = None  # to ensure we remain in the same area type

    def invoke(self, context, event):
        scene = context.scene
        self._original_area_type = context.area.type
        self._dirty = True  # force initial build
        # If not keeping mode, force the UI mode to BROWSE.
        if not self.keep_mode:
            scene.ui_current_mode = "BROWSE"

        # Save initial filter property values.
        self.last_item_type = scene.package_item_type
        self.last_sort_by = scene.package_sort_by
        self.last_search = scene.package_search_query
        self.last_event_stage = scene.event_stage  # Save the initial event stage
        self.last_selected_event = scene.selected_event  # Save the initial selected event

        # Initialize UI draw data.
        bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()

        # Set up the draw handler and a timer for the modal operator.
        self._handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_image_buttons_callback, (), 'WINDOW', 'POST_PIXEL'
        )
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        if context.area:
            context.area.tag_redraw()
        self.report({'INFO'}, "Package UI displayed.")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if not is_internet_available():
            bpy.ops.webapp.logout()
            self.report({'ERROR'}, "No internet connection. Logging out and closing UI.")
            return self.cancel(context)
        scene = context.scene

        if event.type == 'TIMER':
            # If the 3D View area has changed type, cancel the modal.
            if (not context.area) or (context.area.type != self._original_area_type):
                self.report({'INFO'}, "3D View context changed – closing UI.")
                self.cancel(context)
                return {'CANCELLED'}
            
            if self._dirty:
                bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
                if context.area:
                    context.area.tag_redraw()
                self._dirty = False

            # Check for filter changes by comparing current scene properties with stored ones.
            current_item = scene.package_item_type
            current_sort = scene.package_sort_by
            current_search = scene.package_search_query
            current_event_stage = scene.event_stage  # current event stage
            current_selected_event = scene.selected_event  # current selected event

            # If the current type is "event", also monitor event_stage and selected_event.
            if (current_item != self.last_item_type or
                current_sort != self.last_sort_by or
                current_search != self.last_search or
                (current_item == 'event' and (current_event_stage != self.last_event_stage or current_selected_event != self.last_selected_event))):
                
                # Update the saved values.
                self.last_item_type = current_item
                self.last_sort_by = current_sort
                self.last_search = current_search
                self.last_event_stage = current_event_stage
                self.last_selected_event = current_selected_event

                # Reset to page one and show a loading indicator.
                scene.current_thumbnail_page = 1
                scene.show_loading_image = True

                # Call the threaded fetch operator to refresh the package list.
                bpy.ops.webapp.fetch_page('EXEC_DEFAULT', page_number=1)
                self._dirty = True   # mark UI dirty on filter change
                self.report({'INFO'}, "Filter change detected. Refreshing data.")

            # Handle any ongoing loading sequence.
            if self._do_loading:
                if self.loading_step == 0:
                    self.loading_step = 1
                    return {'RUNNING_MODAL'}
                elif self.loading_step == 1:
                    if self._active_task == "page":
                        result = bpy.ops.webapp.fetch_page('EXEC_DEFAULT', page_number=self._target_page)
                        if result not in ({'FINISHED'}, {'RUNNING_MODAL'}):
                            self.report({'ERROR'}, "Page load failed or cancelled.")
                    elif self._active_task == "detail":
                        detail_data = fetch_detail_for_file(file_id=self._detail_file_id)
                        if detail_data and detail_data.get("success"):
                            addon_data = scene.my_addon_data
                            addon_data.init_from_package(detail_data)
                            addon_data.comments.clear()
                            for cdict in detail_data.get("comments", []):
                                c_item = addon_data.comments.add()
                                c_item.author = cdict.get("author", "")
                                c_item.text = cdict.get("content", "")
                                c_item.timestamp = cdict.get("timestamp", "")
                            scene.ui_current_mode = "DETAIL"
                        else:
                            self.report({'ERROR'}, "Failed to fetch detail data.")
                    self.loading_step = 2
                    return {'RUNNING_MODAL'}
                elif self.loading_step == 2:
                    scene.show_loading_image = False
                    bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
                    if context.area:
                        context.area.tag_redraw()
                    self._do_loading = False
                    self.loading_step = 0
                    self._active_task = ""
                    self._target_page = 0
                    self._detail_file_id = 0
                    return {'RUNNING_MODAL'}

        # Handle mouse movement for hover effects.
        if event.type == 'MOUSEMOVE':
            mouse_x = event.mouse_region_x
            mouse_y = event.mouse_region_y
            hover_found = False
            gpu_data = bpy.types.Scene.gpu_image_buttons_data
            if gpu_data:
                for button in gpu_data:
                    # Define clickable elements.
                    if button.get("name") in {"Close_Icon", "Right_Arrow", "Left_Arrow", "Back_Icon", "Explore_Icon"} or \
                       (button.get("name") not in {"Template"}):
                        x1, y1, x2, y2 = button.get("pos", (0, 0, 0, 0))
                        if (x1 <= mouse_x <= x2) and (y1 <= mouse_y <= y2):
                            hover_found = True
                            break
            context.window.cursor_modal_set('HAND' if hover_found else 'DEFAULT')

        # Process left-mouse clicks.
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            mouse_x = event.mouse_region_x
            mouse_y = event.mouse_region_y
            gpu_data = bpy.types.Scene.gpu_image_buttons_data
            packages_list = getattr(bpy.types.Scene, 'fetched_packages_data', [])
            for button in gpu_data:
                if button.get("name") == "handler":
                    continue  # skip any handler
                x1, y1, x2, y2 = button.get("pos", (0, 0, 0, 0))
                if (x1 <= mouse_x <= x2) and (y1 <= mouse_y <= y2):
                    # Handle specific button actions.
                    if button["name"] == "Close_Icon":
                        from ..image_button_UI.explore_downloads import current_download_task
                        if current_download_task is not None:
                            current_download_task.cancel()
                        self.report({'INFO'}, "Close button clicked! Cancelling download and removing UI...")
                        return self.cancel(context)

                    elif button["name"] == "Template":
                        continue
                    elif button["name"] == "Right_Arrow":
                        page = scene.current_thumbnail_page
                        total_pages = scene.total_thumbnail_pages
                        if page < total_pages:
                            self.begin_loading_for_page(context, page + 1)
                            self._dirty = True 
                        return {'RUNNING_MODAL'}
                    elif button["name"] == "Left_Arrow":
                        page = scene.current_thumbnail_page
                        if page > 1:
                            self.begin_loading_for_page(context, page - 1)
                            self._dirty = True 
                        return {'RUNNING_MODAL'}
                    elif button["name"] == "Back_Icon":
                        if scene.ui_current_mode == "DETAIL":
                            self.report({'INFO'}, "Back button clicked -> BROWSE.")
                            scene.ui_current_mode = "BROWSE"
                            scene.selected_thumbnail = ""
                            bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
                            if context.area:
                                context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                    elif button["name"] == "Explore_Icon":
                        download_code = scene.download_code
                        if scene.my_addon_data.file_id > 0 and download_code:
                            token = load_token()
                            if not token:
                                self.report({'ERROR'}, "You must be logged in to explore a package.")
                                return {'CANCELLED'}
                            # Set a loading flag and reset progress (for the UI drawing)
                            scene.download_progress = 0.0
                            scene.ui_current_mode = "LOADING"  # This is only for drawing, not for process control.
                            # Immediately start the download process regardless of UI mode.
                            explore_icon_handler(context, download_code)
                        else:
                            self.report({'ERROR'}, "No package selected or missing download code.")
                        return {'RUNNING_MODAL'}
                    elif button["name"] == "Submit_World_Icon":
                        # jump out to your website’s events page
                        bpy.ops.webapp.open_url(
                            'INVOKE_DEFAULT',
                            url=EVENTS_URL
                        )
                        return {'RUNNING_MODAL'}
                    elif button["name"] == "Visit_Shop_Icon":
                        bpy.ops.webapp.open_url('INVOKE_DEFAULT', url=SHOP_URL)
                        return {'RUNNING_MODAL'}
                    
                    else:
                        # Handle clicking a thumbnail in BROWSE mode.
                        if scene.ui_current_mode == "BROWSE":
                            self.report({'INFO'}, f"Thumbnail '{button['name']}' clicked in BROWSE!")
                            selected_pkg = next((p for p in packages_list if p.get("package_name") == button["name"]), None)
                            if selected_pkg:
                                file_id = selected_pkg.get("file_id", 0)
                                if file_id > 0:
                                    scene.download_code = selected_pkg.get("download_code", "")
                                    scene.my_addon_data.file_id = file_id
                                    scene.selected_thumbnail = selected_pkg.get("local_thumb_path", "")
                                    self.begin_loading_for_detail(context, file_id)
                                else:
                                    self.report({'ERROR'}, "Selected package has invalid file_id.")
                            else:
                                self.report({'ERROR'}, "Package not found in data.")
                            return {'RUNNING_MODAL'}

        if event.type == 'ESC':
            return self.cancel(context)

        return {'PASS_THROUGH'}

    def cancel(self, context):
        if self._handler:
            bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
            self._handler = None
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if hasattr(bpy.types.Scene, "gpu_image_buttons_data"):
            bpy.types.Scene.gpu_image_buttons_data.clear()
        # Set mode to GAME to prevent any further UI drawing.
        context.scene.ui_current_mode = "GAME"
        context.window.cursor_modal_restore()
        if context.area:
            context.area.tag_redraw()
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        self.report({'INFO'}, "Package UI closed.")
        return {'CANCELLED'}

    def begin_loading_for_page(self, context, new_page):
        self._do_loading = True
        self.loading_step = 0
        self._active_task = "page"
        self._target_page = new_page
        context.scene.show_loading_image = True
        bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
        if context.area:
            context.area.tag_redraw()

    def begin_loading_for_detail(self, context, file_id):
        self._do_loading = True
        self.loading_step = 0
        self._active_task = "detail"
        self._detail_file_id = file_id
        context.scene.show_loading_image = True
        bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
        if context.area:
            context.area.tag_redraw()



# ------------------------------------------------------------------------
# 3) REMOVE_PACKAGE_OT_Display
#    - Removes the UI (if still active)
# ------------------------------------------------------------------------
class REMOVE_PACKAGE_OT_Display(bpy.types.Operator):
    """
    Explicitly removes the UI if it's active.
    """
    bl_idname = "view3d.remove_package_display"
    bl_label = "Hide Package Thumbnails"
    bl_options = {'REGISTER'}

    def execute(self, context):
        data = getattr(bpy.types.Scene, "gpu_image_buttons_data", None)
        if not data:
            self.report({'INFO'}, "No active package thumbnails to remove.")
            return {'CANCELLED'}

        # Remove the draw handler if stored
        for item in data:
            handler = item.get("handler")
            if handler:
                bpy.types.SpaceView3D.draw_handler_remove(handler, 'WINDOW')

        # Clear
        bpy.types.Scene.gpu_image_buttons_data.clear()
        self.report({'INFO'}, "Package thumbnails removed.")
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# APPLY_FILTERS_SHOWUI_OT
#   - An operator that displays the UI first, then fetches packages if needed
# ----------------------------------------------------------------------------

class APPLY_FILTERS_SHOWUI_OT(bpy.types.Operator):
    """
    1) Opens the UI if it's not already active.
    2) Enables the scene.show_loading_image flag to display a 'loading' indicator.
    3) Waits briefly so Blender can update the interface.
    4) Fetches packages (only if they're not already in memory).
    5) Disables the loading indicator and refreshes the UI with final thumbnails.
    """

    bl_idname = "webapp.apply_filters_showui"
    bl_label = "Apply Filters + Show UI"
    bl_options = {'REGISTER'}

    page_number: bpy.props.IntProperty(
        name="Page Number",
        default=1,
        description="Which page to load when applying filters"
    )

    _timer = None
    _step = 0

    def invoke(self, context, event):
        scene = context.scene

        # 1) Ensure the thumbnail UI is visible.
        result = bpy.ops.view3d.add_package_display('INVOKE_DEFAULT')
        if result not in ({'FINISHED'}, {'RUNNING_MODAL'}):
            self.report({'ERROR'}, "Could not open Package Display UI.")
            return {'CANCELLED'}

        # 2) Show the loading image immediately.
        scene.show_loading_image = True

        # 3) Trigger a redraw so the loading indicator is drawn right now.
        if hasattr(bpy.types.Scene, "gpu_image_buttons_data"):
            bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
        if context.area:
            context.area.tag_redraw()

        # 4) Start a short modal timer; once it ticks, we'll run the fetch logic.
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        self._step = 0

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            # We step through two phases:
            #   _step == 0 => just let the UI finish drawing the loading image
            #   _step == 1 => do the fetch (if needed), finalize the UI, then exit

            if self._step == 0:
                self._step = 1
                return {'RUNNING_MODAL'}

            elif self._step == 1:
                scene = context.scene

                # Check if the current page is already loaded
                fetched = getattr(bpy.types.Scene, "fetched_packages_data", [])
                cache_is_valid = (
                    fetched and scene.current_thumbnail_page == self.page_number
                )

                # If we have cached data for this page, skip fetching
                if cache_is_valid:
                    self.report({'INFO'}, f"Page {self.page_number} is already in memory; no fetch needed.")
                else:
                    self.report({'INFO'}, f"Fetching page {self.page_number} now...")
                    scene.current_thumbnail_page = self.page_number

                    # Actually call the fetch operator
                    result = bpy.ops.webapp.fetch_page('EXEC_DEFAULT', page_number=self.page_number)
                    if result in ({'FINISHED'}, {'RUNNING_MODAL'}):
                        self.report({'INFO'}, "Fetch operator started or completed.")
                    else:
                        self.report({'ERROR'}, "Unable to start fetch operator.")
                        self._cleanup(context)
                        return {'CANCELLED'}

                # Turn off loading once we're done (fetched or not)
                scene.show_loading_image = False

                # Rebuild the UI with final data
                if hasattr(bpy.types.Scene, "gpu_image_buttons_data"):
                    bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
                if context.area:
                    context.area.tag_redraw()

                self.report({'INFO'}, f"Page {self.page_number} loaded. UI updated.")
                self._cleanup(context)
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def _cleanup(self, context):
        # Remove the timer if it exists
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def cancel(self, context):
        self._cleanup(context)