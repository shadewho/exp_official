# File: Exploratory/Exp_UI/packages/operators.py

import bpy
import requests
from ..auth.helpers import load_token
from ..internet.helpers import ensure_internet_connection
from ..main_config import DOWNLOAD_ENDPOINT
from ..cache_system.download_helpers import download_thumbnail

# ----------------------------------------------------------------------------
# DOWNLOAD CODE - this operator handles downloading a world using a download code 
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
            response.raise_for_status()
            data = response.json()

            if not data.get("success"):
                self.report({'ERROR'}, data.get("message", "Download code failed."))
                return {'CANCELLED'}

            # Inspect the package details
            package_details = data.get("package", data)
            print("Package Details:", package_details)

            # Initialize the scene property group with these details.
            context.scene.my_addon_data.init_from_package(package_details)
            # store the real package type so detail_content.py can pick it up
            context.scene.my_addon_data.file_type = package_details.get(
                "file_type", context.scene.package_item_type
            )

            # Set UI mode to DETAIL.
            context.scene.ui_current_mode = "DETAIL"
            context.scene.download_code = download_code

            # Download thumbnail if available, using integer file_id as cache key
            thumbnail_url = package_details.get("thumbnail_url")
            if thumbnail_url:
                pkg_id = package_details.get("file_id", 0)
                thumb_path = download_thumbnail(thumbnail_url, file_id=pkg_id)
                context.scene.selected_thumbnail = thumb_path or ""
            else:
                context.scene.selected_thumbnail = ""

            # Open the package display UI in DETAIL mode
            bpy.ops.view3d.add_package_display('INVOKE_DEFAULT', keep_mode=True)
            self.report({'INFO'}, "Showing package details for the world.")
            return {'FINISHED'}

        except requests.RequestException as e:
            self.report({'ERROR'}, f"API Error: {e}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Unexpected error: {e}")
            return {'CANCELLED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
