# backend.py
import os
import bpy
import requests  # Added missing import
import shutil
from .helper_functions import (
 download_blend_file, append_scene_from_blend
)
from .auth import load_token, save_token, clear_token
import traceback
from .main_config import (LOGIN_ENDPOINT, DOWNLOAD_ENDPOINT, THUMBNAIL_CACHE_FOLDER)
from .exp_api import login, logout
from .helper_functions import download_thumbnail


# ----------------------------------------------------------------------------
# LOGIN/LOGOUT
# ----------------------------------------------------------------------------
class LOGIN_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.login"
    bl_label = "Login to Web App"
    bl_options = {'REGISTER'}

    def execute(self, context):
        username = context.scene.username
        password = context.scene.password

        try:
            data = login(username, password)
            if data.get("success"):
                token = data.get("token")
                save_token(token)
                self.report({'INFO'}, "Login successful!")
            else:
                self.report({'ERROR'}, "Login failed: " + data.get("message", "Unknown error"))
        except Exception as e:
            self.report({'ERROR'}, f"Connection error: {str(e)}")

        return {'FINISHED'}


class LOGOUT_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.logout"
    bl_label = "Logout from Web App"
    bl_options = {'REGISTER'}

    def execute(self, context):
        clear_token()
        # Removed cache clearing code so that persistent cached data is preserved.
        self.report({'INFO'}, "Logged out successfully. Cached data preserved.")
        return {'FINISHED'}

# ----------------------------------------------------------------------------
# DOWNLOAD CODE
# ----------------------------------------------------------------------------
class DOWNLOAD_CODE_OT_File(bpy.types.Operator):
    bl_idname = "webapp.download_code"
    bl_label = "Show World Details"
    bl_options = {'REGISTER'}

    def execute(self, context):
        token = load_token()
        if not token:
            self.report({'ERROR'}, "You must log in first.")
            return {'CANCELLED'}

        download_code = context.scene.download_code.strip()
        if not download_code:
            self.report({'ERROR'}, "Please enter a download code first.")
            return {'CANCELLED'}

        url = DOWNLOAD_ENDPOINT
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {"download_code": download_code}

        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    # Inspect the package details
                    package_details = data.get("package", data)
                    print("Package Details:", package_details)

                    # Initialize the scene property group with these details.
                    context.scene.my_addon_data.init_from_package(package_details)

                    # Set UI mode to DETAIL.
                    context.scene.ui_current_mode = "DETAIL"
                    context.scene.download_code = download_code

                    # Download thumbnail if available.
                    thumbnail_url = package_details.get("thumbnail_url")
                    if thumbnail_url:
                        thumb_path = download_thumbnail(thumbnail_url)
                        context.scene.selected_thumbnail = thumb_path
                    else:
                        context.scene.selected_thumbnail = ""

                    bpy.ops.view3d.add_package_display('INVOKE_DEFAULT', keep_mode=True)
                    self.report({'INFO'}, "Showing package details for the world.")
                else:
                    self.report({'ERROR'}, data.get("message", "Download code failed."))
                    return {'CANCELLED'}
            else:
                self.report({'ERROR'}, f"API Error {response.status_code}: {response.text}")
                return {'CANCELLED'}
        except Exception as e:
            traceback.print_exc()
            self.report({'ERROR'}, f"Error: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
