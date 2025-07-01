#Exploratory/Exp_UI/interface/operators/display.py

import bpy
import queue
from ...auth.helpers import load_token
from ...internet.helpers import is_internet_available
from ...main_config import EVENTS_URL, SHOP_URL
from .utilities import fetch_detail_for_file
from ..drawing.draw_master import load_image_buttons
from ..drawing.utilities import draw_image_buttons_callback
from ...download_and_explore.explore_main import explore_icon_handler
# A global queue for background fetch results
fetch_page_queue = queue.Queue()
load_page_queue = queue.Queue()

# ------------------------------------------------------------------------
# PACKAGE_OT_Display
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
        self._last_progress = -1.0

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
            # -------- NEW: rebuild when progress OR flag changes --------
            cur = scene.download_progress
            if abs(cur - self._last_progress) > 1e-4:   # progress moved?
                self._last_progress = cur
                self._dirty = True                      # force UI rebuild

            if getattr(bpy.types.Scene, "package_ui_dirty", False):
                bpy.types.Scene.package_ui_dirty = False
                self._dirty = True
            # -----------------------------------------------------------------

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

                # --- wipe the old grid immediately ---
                bpy.types.Scene.fetched_packages_data = []     # no thumbnails to draw
                scene.total_thumbnail_pages = 1
                scene.selected_thumbnail = ""
                self._dirty = True                             # force UI rebuild right now
                # --------------------------------------

                # Reset to page one and show a loading indicator
                scene.current_thumbnail_page = 1
                scene.show_loading_image = True
                bpy.ops.webapp.fetch_page('EXEC_DEFAULT', page_number=1)
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
                        from ...download_and_explore.explore_main import current_download_task
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
                            self._dirty = True
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