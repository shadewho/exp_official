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
        # Clear cached thumbnails
        if os.path.exists(THUMBNAIL_CACHE_FOLDER):
            shutil.rmtree(THUMBNAIL_CACHE_FOLDER)
        os.makedirs(THUMBNAIL_CACHE_FOLDER, exist_ok=True)

        self.report({'INFO'}, "Logged out successfully, cache cleared.")
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# APPEND SCENE
# ----------------------------------------------------------------------------
class APPEND_SCENE_OT_File(bpy.types.Operator):
    bl_idname = "webapp.append_scene"
    bl_label = "Append Scene"
    bl_options = {'REGISTER'}

    def execute(self, context):
        token = load_token()
        if not token:
            self.report({'ERROR'}, "You must log in first.")
            return {'CANCELLED'}

        # *** Now we read from context.scene.download_code
        download_code = context.scene.download_code
        if not download_code.strip():
            self.report({'ERROR'}, "Please enter a download code first.")
            return {'CANCELLED'}

        # The rest is unchanged:
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
                    download_url = data["download_url"]

                    # Download the .blend file
                    local_blend_path = download_blend_file(download_url)
                    if not local_blend_path:
                        self.report({'ERROR'}, "Failed to download .blend file.")
                        return {'CANCELLED'}

                    # Append the scene
                    result = append_scene_from_blend(local_blend_path, new_scene_name="Appended_Scene")
                    if result == {'FINISHED'}:
                        self.report({'INFO'}, "Scene appended successfully!")
                    else:
                        self.report({'ERROR'}, "Failed to append scene.")
                        return {'CANCELLED'}

                else:
                    self.report({'ERROR'}, data.get("message", "Download failed"))
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