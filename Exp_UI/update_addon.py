#Exp_UI/update_addon.py
import bpy
import requests
import tempfile
import os
import urllib.request

from .main_config import BASE_URL
from .version_info import CURRENT_VERSION
from .exp_api import update_latest_version_cache, get_cached_latest_version

class WEBAPP_OT_UpdateAddon(bpy.types.Operator):
    """
    Download and install the latest Exploratory add-on with confirmation popup
    """
    bl_idname = "webapp.update_addon"
    bl_label = "Update Add-on"
    bl_options = {'REGISTER', 'UNDO'}

    version: bpy.props.StringProperty()
    download_url: bpy.props.StringProperty()
    update_type: bpy.props.StringProperty()
    changelog: bpy.props.StringProperty()

    def invoke(self, context, event):
        try:
            resp = requests.get(f"{BASE_URL}/api/addon_version", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data.get('success'):
                self.report({'ERROR'}, data.get('message', 'Failed to fetch update info'))
                return {'CANCELLED'}

            self.version = data['version_string']
            self.download_url = data['download_url']
            self.update_type = data.get('update_type', 'full')
            self.changelog = data.get('changelog', '')

            if self.version == CURRENT_VERSION:
                self.report({'INFO'}, f"Already on {CURRENT_VERSION}")
                return {'CANCELLED'}

            return context.window_manager.invoke_props_dialog(self, width=420)

        except Exception as e:
            self.report({'ERROR'}, f"Error fetching update info: {e}")
            return {'CANCELLED'}

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text=f"{self.update_type.capitalize()} Update Available", icon='FILE_TICK')

        row = box.row(align=True)
        row.label(text="Version:", icon='SORTTIME')
        row.label(text=self.version)

        if self.changelog:
            box.separator()
            box.label(text="Changelog", icon='BOOKMARKS')
            for line in self.changelog.splitlines():
                bullet_row = box.row(align=True)
                bullet_row.label(text=f"• {line}")

        layout.separator()
        layout.label(text="⏳ This may take a moment…", icon='TIME')
        layout.label(text="✨ We'll have you back up and running shortly.", icon='INFO')

    def execute(self, context):
        tmp_dir = tempfile.gettempdir()
        zip_path = os.path.join(tmp_dir, "exploratory_update.zip")

        try:
            urllib.request.urlretrieve(self.download_url, zip_path)
            bpy.ops.preferences.addon_remove(module="Exploratory")
            bpy.ops.preferences.addon_install(filepath=zip_path, overwrite=True)
            bpy.ops.preferences.addon_enable(module="Exploratory")
            bpy.ops.wm.save_userpref()

            self.report({'INFO'}, f"Updated to {self.version}")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Update failed: {e}")
            return {'CANCELLED'}

class WEBAPP_OT_RefreshVersion(bpy.types.Operator):
    bl_idname = "webapp.refresh_version"
    bl_label = "Check for Update"
    bl_description = "Fetch the latest add-on version from the server"

    def execute(self, context):
        update_latest_version_cache()
        latest = get_cached_latest_version()
        if latest is None:
            self.report({'ERROR'}, "Failed to fetch latest version.")
        elif latest == CURRENT_VERSION:
            self.report({'INFO'}, f"Exploratory is up to date ({CURRENT_VERSION}).")
        else:
            self.report({'INFO'}, f"New version available: {latest}")
        return {'FINISHED'}