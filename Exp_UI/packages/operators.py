#Exploratory/Exp_UI/packages/operators.py

import bpy
import requests
from ..auth.helpers import load_token
from ..internet.helpers import ensure_internet_connection
import traceback
from ..main_config import DOWNLOAD_ENDPOINT
from ..cache_system.download_helpers import download_thumbnail

# ----------------------------------------------------------------------------
# DOWNLOAD CODE - this opertator handles downloading a world using a download code 
#                 and displaying its details in the custom UI.
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
        
        if not ensure_internet_connection(context):
            self.report({'ERROR'}, "No internet connection detected. Logging out.")
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
