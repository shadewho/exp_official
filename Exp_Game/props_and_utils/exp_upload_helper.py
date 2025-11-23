#Exploratory/Exp_Game/exp_upload_helper.py

import bpy
import os

def get_blend_file_size_mib():
    """
    Returns the size of the current .blend file in mebibytes (float),
    or None if the file is unsaved or missing.
    """
    path = bpy.data.filepath
    if not path or not os.path.exists(path):
        return None
    try:
        size_bytes = os.path.getsize(path)
        return size_bytes / (1024.0 ** 2)  # divide by 2^20 for MiB
    except OSError:
        return None

class EXP_UPLOADHELPER_OT_RefreshSize(bpy.types.Operator):
    """Recompute and store the current .blend file size in MiB."""
    bl_idname = "exploratory.uploadhelper_refresh_size"
    bl_label = "Refresh File Size"
    bl_options = {'REGISTER'}

    def execute(self, context):
        size_mib = get_blend_file_size_mib()
        context.scene.upload_helper_file_size = size_mib or 0.0
        if size_mib is None:
            self.report({'WARNING'}, "Blend file not saved; size unavailable.")
        else:
            self.report(
                {'INFO'},
                f"File size updated: {size_mib:.2f} MiB / 500 MiB"
            )
        return {'FINISHED'}

def register():
    # scene prop holds MiB
    bpy.types.Scene.upload_helper_file_size = bpy.props.FloatProperty(
        name="Blend File Size (MiB)",
        description="Cached size of the current .blend file (in MiB)",
        default=0.0
    )
    bpy.utils.register_class(EXP_UPLOADHELPER_OT_RefreshSize)

def unregister():
    bpy.utils.unregister_class(EXP_UPLOADHELPER_OT_RefreshSize)
    if hasattr(bpy.types.Scene, 'upload_helper_file_size'):
        del bpy.types.Scene.upload_helper_file_size