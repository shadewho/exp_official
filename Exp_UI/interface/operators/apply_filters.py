#Exploratory/Exp_UI/interface/operators/apply_filters.py

import bpy
from .utilities import  build_filter_signature
from ..drawing.draw_master import load_image_buttons
import time
# ----------------------------------------------------------------------------
# APPLY_FILTERS_SHOWUI_OT
#   - An operator that displays the UI first, then fetches packages if needed
# ----------------------------------------------------------------------------

class APPLY_FILTERS_SHOWUI_OT(bpy.types.Operator):
    """
    Open the Exploratory interface.
    Browse by item type and/or filters.
    Updates every 10 minutes if logged in.
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
                
                # turn panel blank while we fetch
                bpy.types.Scene.fetched_packages_data = []
                scene.total_thumbnail_pages = 1
                scene.selected_thumbnail = ""
                self._dirty = True

                # Check if the current page is already loaded
                fetched = getattr(bpy.types.Scene, "fetched_packages_data", [])
                sig_now = build_filter_signature(scene)
                #if we're in Random mode, bump it so it always differs
                if scene.package_sort_by == 'random':
                    sig_now = f"{sig_now}:{int(time.time())}"

                cache_is_valid = (
                    fetched
                    and scene.current_thumbnail_page == self.page_number
                    and scene.last_filter_signature == sig_now
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