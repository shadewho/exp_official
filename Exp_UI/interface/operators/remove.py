#Exploratory/Exp_UI/interface/operators/remove.py

import bpy
# ------------------------------------------------------------------------
# REMOVE_PACKAGE_OT_Display
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
