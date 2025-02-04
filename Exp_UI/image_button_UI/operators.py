# operators.py

import bpy
import math
import requests
from gpu_extras.batch import batch_for_shader
from ..auth import load_token
from .drawing import load_image_buttons, draw_image_buttons_callback
from .utils import calculate_free_space, calculate_template_position, viewport_changed
from .cache import get_or_load_image, get_or_create_texture, clear_image_datablocks
from .config import THUMBNAILS_PER_PAGE
from ..helper_functions import download_thumbnail
from ..main_config import PACKAGES_ENDPOINT
from ..exp_api import fetch_packages, explore_package, fetch_detail_for_file
import threading
import queue
from ..exp_api import download_blend_file, append_scene_from_blend

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
    A single operator that fetches the specified page (including page 1) in a background thread,
    does progressive thumbnail loading, and caches the results so we don't re-fetch next time.
    """
    bl_idname = "webapp.fetch_page"
    bl_label = "Fetch Page (Threaded, Progressive)"
    bl_options = {'REGISTER'}

    page_number: bpy.props.IntProperty(
        name="Page Number",
        default=1,
        description="Which page to load"
    )

    _timer = None

    def execute(self, context):
        """
        1) If page is cached, just use that data and return FINISHED.
        2) Otherwise, spawn a background thread:
           - fetches the package list for this page (offset/limit),
           - sends "PACKAGE_LIST" to the queue,
           - downloads thumbnails, sending "PACKAGE_READY" for each,
           - finally sends "SUCCESS" or "ERROR".
        3) Set up a modal timer so we can handle progressive updates in modal().
        """

        # --- A) Create the all_pages_data cache if not present
        if not hasattr(bpy.types.Scene, "all_pages_data"):
            bpy.types.Scene.all_pages_data = {}

        scene = context.scene
        page_num = self.page_number

        # --- B) Check cache
        cached_page = bpy.types.Scene.all_pages_data.get(page_num)
        if cached_page:
            # Already cached -> just set fetched_packages_data
            bpy.types.Scene.fetched_packages_data = cached_page
            scene.current_thumbnail_page = page_num
            self.report({'INFO'}, f"Page {page_num} is already cached. Using local data.")
            # Optionally force UI redraw
            if context.area:
                context.area.tag_redraw()
            return {'FINISHED'}

        # Otherwise, we do a new fetch in background

        # Check login
        token = load_token()
        if not token:
            self.report({'ERROR'}, "You must log in first.")
            return {'CANCELLED'}

        # Build offset/limit
        offset = (page_num - 1) * 8
        limit  = 8

        # Gather extra filter params from scene
        file_type = scene.package_item_type
        sort_by   = scene.package_sort_by
        search_query = scene.package_search_query.strip()

        params = {
            "file_type": file_type,
            "sort_by":   sort_by,
            "offset":    offset,
            "limit":     limit
        }
        if search_query:
            params["search_query"] = search_query

        # Clear any old results from the queue
        with fetch_page_queue.mutex:
            fetch_page_queue.queue.clear()

        # Thread function
        def _fetch_worker():
            try:
                data = fetch_packages(params)
                if not data.get("success"):
                    fetch_page_queue.put(("ERROR", data.get("message", "Unknown error")))
                    return

                packages = data.get("packages", [])
                total_count = data.get("total_count", 0)

                # Step 1: Send the entire list (no thumbs yet)
                fetch_page_queue.put((
                    "PACKAGE_LIST",
                    {
                        "page_num":   page_num,
                        "packages":   packages,
                        "total_count": total_count
                    }
                ))

                # Step 2: Download each thumbnail in the background
                for pkg in packages:
                    thumb_url = pkg.get("thumbnail_url")
                    if thumb_url:
                        local_path = download_thumbnail(thumb_url)
                        pkg["local_thumb_path"] = local_path
                    else:
                        pkg["local_thumb_path"] = None

                    # Send partial update
                    fetch_page_queue.put((
                        "PACKAGE_READY",
                        {
                            "page_num": page_num,
                            "package":  pkg
                        }
                    ))

                # Step 3: Done
                fetch_page_queue.put((
                    "SUCCESS",
                    {
                        "page_num":   page_num,
                        "packages":   packages,
                        "total_count": total_count
                    }
                ))

            except Exception as e:
                fetch_page_queue.put(("ERROR", str(e)))

        # Start the thread
        worker = threading.Thread(target=_fetch_worker)
        worker.start()

        # Turn on a spinner if you want
        scene.show_loading_image = True

        # Start modal timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)

        self.report({'INFO'}, f"Fetching page {page_num} in background (progressive).")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        """
        Called repeatedly to handle partial updates from the queue:
          - "PACKAGE_LIST": init or clear scene data for this page
          - "PACKAGE_READY": add/update one package in Scene.fetched_packages_data
          - "SUCCESS": store final packages in cache & scene, remove timer
          - "ERROR": show error, remove timer
        """

        if event.type == 'TIMER':
            while True:
                try:
                    result_type, payload = fetch_page_queue.get_nowait()

                    if result_type == "PACKAGE_LIST":
                        page_num    = payload["page_num"]
                        packages    = payload["packages"]
                        total_count = payload["total_count"]

                        # Create an empty list in the cache
                        bpy.types.Scene.all_pages_data[page_num] = []

                        # Also clear Scene.fetched_packages_data
                        bpy.types.Scene.fetched_packages_data = []
                        context.scene.current_thumbnail_page = page_num
                        # e.g., compute total pages
                        if total_count > 0:
                            context.scene.total_thumbnail_pages = max(
                                1, math.ceil(total_count / 8)
                            )

                        self.report({'INFO'}, f"Page {page_num}: got {len(packages)} packages (no thumbs yet).")

                    elif result_type == "PACKAGE_READY":
                        page_num = payload["page_num"]
                        pkg      = payload["package"]
                        # Insert/update in cache and scene
                        self._insert_or_update_cache(page_num, pkg)
                        self._insert_or_update_scene(pkg)
                        # Rebuild UI
                        bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
                        if context.area:
                            context.area.tag_redraw()

                    elif result_type == "SUCCESS":
                        page_num    = payload["page_num"]
                        packages    = payload["packages"]
                        total_count = payload["total_count"]

                        # Store final packages in cache & scene
                        bpy.types.Scene.all_pages_data[page_num] = packages
                        bpy.types.Scene.fetched_packages_data   = packages

                        # Turn off spinner
                        context.scene.show_loading_image = False

                        # Remove timer
                        wm = context.window_manager
                        wm.event_timer_remove(self._timer)

                        self.report({'INFO'}, f"Page {page_num} => all thumbs loaded, total {total_count}.")
                        return {'FINISHED'}

                    elif result_type == "ERROR":
                        error_msg = payload
                        self.report({'ERROR'}, f"Fetch page error: {error_msg}")
                        context.scene.show_loading_image = False
                        wm = context.window_manager
                        wm.event_timer_remove(self._timer)
                        return {'CANCELLED'}

                except queue.Empty:
                    break  # no more items right now

        # If user presses ESC, we cancel
        if event.type == 'ESC':
            wm = context.window_manager
            wm.event_timer_remove(self._timer)
            context.scene.show_loading_image = False
            self.report({'WARNING'}, "Page fetch canceled by user.")
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def _insert_or_update_cache(self, page_num, pkg):
        cache_list = bpy.types.Scene.all_pages_data.get(page_num, [])
        pkg_id = pkg.get("file_id")
        for i, existing in enumerate(cache_list):
            if existing.get("file_id") == pkg_id:
                cache_list[i] = pkg
                return
        cache_list.append(pkg)

    def _insert_or_update_scene(self, pkg):
        scene_data = bpy.types.Scene.fetched_packages_data
        pkg_id = pkg.get("file_id")
        for i, existing in enumerate(scene_data):
            if existing.get("file_id") == pkg_id:
                scene_data[i] = pkg
                return
        scene_data.append(pkg)


# ------------------------------------------------------------------------
# 2) PACKAGE_OT_Display
#    - Displays images found in `bpy.types.Scene.fetched_packages_data`
#      (assuming their thumbnails are in `CACHED_IMAGE_FOLDER`)
#    - Minimal 2:1 template, 4x2 grid for images
# -----------------------------------------------------------------------

class PACKAGE_OT_Display(bpy.types.Operator):
    """
    Displays the UI in the 3D Viewport using `load_image_buttons()` from drawing.py.
    This operator handles the modal logic (timer, mouse, ESC, etc.).
    """
    bl_idname = "view3d.add_package_display"
    bl_label = "Show Package Thumbnails"
    bl_options = {'REGISTER'}

    _handler = None
    _timer = None

    # "Loading" state fields
    _do_loading = False         # Are we currently showing the spinner/loading?
    loading_step = 0            # Sub-step for the loading sequence
    _active_task = ""           # "page" or "detail"
    _target_page = 0            # Which page we want to load if _active_task=="page"
    _detail_file_id = 0         # Which file_id we want to load if _active_task=="detail"

    _original_area_type = None  # Store the original area type

    def invoke(self, context, event):
        self._original_area_type = context.area.type  # record where we were invoked
        context.scene.ui_current_mode = "BROWSE"
        bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()

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
        # --- First, check if the context is still valid
        if event.type == 'TIMER':
            if (not context.area) or (context.area.type != self._original_area_type):
                self.report({'INFO'}, "3D View context changed – closing UI.")
                self.cancel(context)
                return {'CANCELLED'}

            if viewport_changed():
                bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
                if context.area:
                    context.area.tag_redraw()

            # (Your existing loading-step logic follows here...)
            if self._do_loading:
                if self.loading_step == 0:
                    self.loading_step = 1
                    return {'RUNNING_MODAL'}
                elif self.loading_step == 1:
                    if self._active_task == "page":
                        result = bpy.ops.webapp.fetch_page('EXEC_DEFAULT', page_number=self._target_page)
                        if result != {'FINISHED'}:
                            self.report({'ERROR'}, "Page load failed or cancelled.")
                    elif self._active_task == "detail":
                        detail_data = fetch_detail_for_file(file_id=self._detail_file_id)
                        if detail_data and detail_data.get("success"):
                            addon_data = context.scene.my_addon_data
                            addon_data.init_from_package(detail_data)
                            addon_data.comments.clear()
                            for cdict in detail_data.get("comments", []):
                                c_item = addon_data.comments.add()
                                c_item.author = cdict.get("author", "")
                                c_item.text = cdict.get("text", "")
                                c_item.timestamp = cdict.get("timestamp", "")
                            context.scene.ui_current_mode = "DETAIL"
                        else:
                            self.report({'ERROR'}, "Failed to fetch detail data.")
                    self.loading_step = 2
                    return {'RUNNING_MODAL'}
                elif self.loading_step == 2:
                    context.scene.show_loading_image = False
                    bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
                    if context.area:
                        context.area.tag_redraw()
                    self._do_loading = False
                    self.loading_step = 0
                    self._active_task = ""
                    self._target_page = 0
                    self._detail_file_id = 0
                    return {'RUNNING_MODAL'}

        # --- Now, add a hover effect on mouse move ---
        if event.type == 'MOUSEMOVE':
            mouse_x, mouse_y = event.mouse_region_x, event.mouse_region_y
            hover_found = False
            gpu_data = bpy.types.Scene.gpu_image_buttons_data
            if gpu_data:
                for button in gpu_data:
                    # Decide which buttons are “clickable”
                    if button.get("name") in {"Close_Icon", "Right_Arrow", "Left_Arrow", "Back_Icon", "Explore_Icon"} or \
                       (button.get("name") not in {"Template"}):  # adjust condition as needed
                        x1, y1, x2, y2 = button.get("pos", (0, 0, 0, 0))
                        if (x1 <= mouse_x <= x2) and (y1 <= mouse_y <= y2):
                            hover_found = True
                            break
            if hover_found:
                context.window.cursor_modal_set('HAND')
            else:
                context.window.cursor_modal_set('DEFAULT')

        # --- Process left-mouse clicks as before ---
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            mouse_x, mouse_y = event.mouse_region_x, event.mouse_region_y
            gpu_data = bpy.types.Scene.gpu_image_buttons_data
            packages_list = getattr(bpy.types.Scene, 'fetched_packages_data', [])
            for button in gpu_data:
                if button.get("name") == "handler":
                    continue  # skip any handler item
                x1, y1, x2, y2 = button.get("pos", (0, 0, 0, 0))
                if (x1 <= mouse_x <= x2) and (y1 <= mouse_y <= y2):

                    if button["name"] == "Close_Icon":
                        self.report({'INFO'}, "Close button clicked! Removing UI...")
                        self.cancel(context)
                        return {'FINISHED'}
                    
                    elif button["name"] == "Template":
                        continue

                    elif button["name"] == "Right_Arrow":
                        scene = context.scene
                        page = scene.current_thumbnail_page
                        total_pages = scene.total_thumbnail_pages
                        if page < total_pages:
                            self.begin_loading_for_page(context, page + 1)
                        return {'RUNNING_MODAL'}
                    
                    elif button["name"] == "Left_Arrow":
                        scene = context.scene
                        page = scene.current_thumbnail_page
                        if page > 1:
                            self.begin_loading_for_page(context, page - 1)
                        return {'RUNNING_MODAL'}
                    
                    elif button["name"] == "Back_Icon":
                        if context.scene.ui_current_mode == "DETAIL":
                            self.report({'INFO'}, "Back button clicked -> BROWSE.")
                            context.scene.ui_current_mode = "BROWSE"
                            context.scene.selected_thumbnail = ""
                            bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
                            if context.area:
                                context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                    
                    elif button["name"] == "Explore_Icon":
                        # Use the scene-level download_code property
                        download_code = context.scene.download_code
                        if context.scene.my_addon_data.file_id > 0 and download_code:
                            token = load_token()
                            if not token:
                                self.report({'ERROR'}, "You must be logged in to explore a package.")
                                return {'CANCELLED'}
                            headers = {
                                "Authorization": f"Bearer {token}",
                                "Content-Type": "application/json"
                            }
                            explore_url = PACKAGES_ENDPOINT.replace("/packages", "/explore")
                            payload = {
                                "download_code": download_code,
                                "file_type": "world"
                            }
                            try:
                                response = requests.post(explore_url, json=payload, headers=headers)
                                if response.status_code == 200:
                                    data = response.json()
                                    if data.get("success"):
                                        download_url = data.get("download_url")
                                        local_blend_path = download_blend_file(download_url)
                                        if local_blend_path:
                                            result = append_scene_from_blend(local_blend_path, new_scene_name="Appended_Scene")
                                            if result == {'FINISHED'}:
                                                self.report({'INFO'}, "Scene appended successfully!")
                                                # *** Now call the game modal operator ***
                                                bpy.ops.view3d.exp_modal('INVOKE_DEFAULT')
                                                # And cancel/close the UI modal so only one modal remains:
                                                return self.cancel(context)
                                            else:
                                                self.report({'ERROR'}, "Failed to append scene.")
                                                return {'CANCELLED'}
                                        else:
                                            self.report({'ERROR'}, "Failed to download .blend file.")
                                            return {'CANCELLED'}
                                    else:
                                        self.report({'ERROR'}, f"Explore failed: {data.get('message')}")
                                        return {'CANCELLED'}
                                else:
                                    self.report({'ERROR'}, f"API Error {response.status_code}: {response.text}")
                                    return {'CANCELLED'}
                            except Exception as e:
                                self.report({'ERROR'}, f"Error exploring package: {e}")
                                return {'CANCELLED'}
                        else:
                            self.report({'ERROR'}, "No package selected to explore or missing download code.")
                        return {'RUNNING_MODAL'}

                    else:
                        if context.scene.ui_current_mode == "BROWSE":
                            self.report({'INFO'}, f"Thumbnail '{button['name']}' clicked in BROWSE!")
                            selected_pkg = next((p for p in packages_list
                                                if p.get("package_name") == button["name"]), None)
                            if selected_pkg:
                                file_id = selected_pkg.get("file_id", 0)
                                if file_id > 0:
                                    # Update the scene-level property for the download code
                                    context.scene.download_code = selected_pkg.get("download_code", "")
                                    context.scene.my_addon_data.file_id = file_id
                                    context.scene.selected_thumbnail = selected_pkg.get("local_thumb_path", "")
                                    # Trigger the detail view by calling begin_loading_for_detail:
                                    self.begin_loading_for_detail(context, file_id)
                                else:
                                    self.report({'ERROR'}, "Selected package has invalid file_id.")
                            else:
                                self.report({'ERROR'}, "Package not found in data.")
                            return {'RUNNING_MODAL'}


        if event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}

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
        context.scene.ui_current_mode = "BROWSE"
        # Restore the cursor to default when canceling
        context.window.cursor_modal_restore()
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


# ------------------------------------------------------------------------
# ) APPLY_FILTERS_SHOWUI_OT
#    - A new operator that ensures the UI is drawn BEFORE fetching packages
# ------------------------------------------------------------------------
class APPLY_FILTERS_SHOWUI_OT(bpy.types.Operator):
    """
    This operator:
      1) If UI isn't active, opens it
      2) Turn ON scene.show_loading_image so the 'loading' image draws
      3) Wait a short time so Blender can redraw
      4) Fetch packages (unless already loaded)
      5) Turn OFF scene.show_loading_image, rebuild UI with final thumbnails
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
        # Step A) If the UI isn't active, open it
        if not hasattr(bpy.types.Scene, "gpu_image_buttons_data") or not bpy.types.Scene.gpu_image_buttons_data:
            result = bpy.ops.view3d.add_package_display('INVOKE_DEFAULT')
            if result not in ({'FINISHED'}, {'RUNNING_MODAL'}):
                self.report({'ERROR'}, "Failed to open Package Display UI.")
                return {'CANCELLED'}

        # Step B) Turn on loading so user sees the 'loading' image right away
        scene.show_loading_image = True

        # Rebuild the UI data so the template + loading indicator appear
        if hasattr(bpy.types.Scene, "gpu_image_buttons_data"):
            bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
        if context.area:
            context.area.tag_redraw()

        # Start a short modal so we can fetch data on the next TIMER
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        self._step = 0

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            if self._step == 0:
                # Step 0 => Just finished drawing the loading image
                self._step = 1
                return {'RUNNING_MODAL'}

            elif self._step == 1:
                # Step 1 => Actually do the fetch (unless it's already loaded)
                scene = context.scene
                fetched = getattr(bpy.types.Scene, "fetched_packages_data", [])

                already_loaded_this_page = (
                    fetched and scene.current_thumbnail_page == self.page_number
                )

                if already_loaded_this_page:
                    self.report({'INFO'}, f"Page {self.page_number} is already in memory; no fetch needed.")
                else:
                    self.report({'INFO'}, f"Fetching page {self.page_number} now...")
                    scene.current_thumbnail_page = self.page_number
                    # Call the existing fetch operator
                    result = bpy.ops.webapp.fetch_page('EXEC_DEFAULT', page_number=self.page_number)

                    if result in ({'FINISHED'}, {'RUNNING_MODAL'}):
                        self.report({'INFO'}, "Fetch operator started or finished.")
                    else:
                        self.report({'ERROR'}, "Failed to start fetch operator.")
                        self._cleanup(context)
                        return {'CANCELLED'}


                # Now turn OFF loading
                scene.show_loading_image = False

                # Rebuild UI with final thumbnails
                if hasattr(bpy.types.Scene, "gpu_image_buttons_data"):
                    bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
                if context.area:
                    context.area.tag_redraw()

                self.report({'INFO'}, f"Page {self.page_number} loaded. UI updated.")

                self._cleanup(context)
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def _cleanup(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def cancel(self, context):
        self._cleanup(context)