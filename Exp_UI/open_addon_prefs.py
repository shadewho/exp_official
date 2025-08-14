# Exploratory/Exp_UI/open_addon_prefs.py
import bpy

class OPEN_ADDON_PREFS_OT(bpy.types.Operator):
    """Open Blender Preferences in a new window → Add-ons tab"""
    bl_idname = "wm.open_addon_prefs"
    bl_label = "Open Add-on Preferences"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        try:
            # Open Preferences in a new window
            bpy.ops.screen.userpref_show('INVOKE_DEFAULT')
        except Exception as e:
            self.report({'ERROR'}, f"Could not open Preferences: {e}")
            return {'CANCELLED'}

        # Use a timer to wait until the Preferences area exists
        def _focus_addons():
            for win in bpy.context.window_manager.windows:
                for area in win.screen.areas:
                    if area.type == 'PREFERENCES':
                        try:
                            # Switch to the Add-ons tab
                            area.spaces.active.context = 'ADDONS'
                        except Exception:
                            pass
                        return None  # stop timer
            return 0.1  # keep polling until Preferences is found

        bpy.app.timers.register(_focus_addons, first_interval=0.1)
        return {'FINISHED'}


# Registration helpers
classes = (OPEN_ADDON_PREFS_OT,)

def register():
    for c in classes:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
