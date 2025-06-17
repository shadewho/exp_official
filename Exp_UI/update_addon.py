#Exp_UI/update_addon.py
import bpy
import requests
import tempfile
import os
import urllib.request

from .main_config import BASE_URL
from .version_info import CURRENT_VERSION
from .exp_api import update_latest_version_cache, get_cached_latest_version, fetch_latest_version

class WEBAPP_OT_UpdateAddon(bpy.types.Operator):
    """Download and install the latest Exploratory add-on"""
    bl_idname = "webapp.update_addon"
    bl_label = "Update Exploratory Add-on"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            # 1) fetch the active version info
            resp = requests.get(f"{BASE_URL}/api/addon_version", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("success"):
                self.report({'ERROR'}, "Could not fetch update info")
                return {'CANCELLED'}

            latest = data["version_string"]
            download_url = data["download_url"]

            # 2) check if we're already up-to-date
            if latest == CURRENT_VERSION:
                self.report({'INFO'}, f"Already on {CURRENT_VERSION}")
                return {'FINISHED'}

            # 3) download the zip to a temp file
            tmp_dir = tempfile.gettempdir()
            zip_path = os.path.join(tmp_dir, "exploratory_update.zip")
            urllib.request.urlretrieve(download_url, zip_path)

            # 4) uninstall current add-on
            bpy.ops.preferences.addon_remove(module="Exploratory")

            # 5) install & enable the new one
            bpy.ops.preferences.addon_install(filepath=zip_path, overwrite=True)
            bpy.ops.preferences.addon_enable(module="Exploratory")

            # 6) persist preferences
            bpy.ops.wm.save_userpref()

            self.report({'INFO'}, f"Updated to {latest}")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Update failed: {e}")
            return {'CANCELLED'}


class WEBAPP_OT_RefreshVersion(bpy.types.Operator):
    bl_idname = "webapp.refresh_version"
    bl_label = "Check for Update"
    bl_description = "Fetch the latest add-on version from the server"

    def execute(self, context):
        # 1) refresh the in-memory cache
        update_latest_version_cache()

        # 2) force Blender to repaint the Settings panel
        if context.area:
            context.area.tag_redraw()

        # 3) read back and report
        latest = get_cached_latest_version()
        if latest is None:
            self.report({'ERROR'}, "Failed to fetch latest version.")
        elif latest == CURRENT_VERSION:
            self.report({'INFO'}, f"Exploratory is up to date ({CURRENT_VERSION}).")
        else:
            self.report({'INFO'}, f"New version available: {latest}")
        return {'FINISHED'}