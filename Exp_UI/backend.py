# backend.py

import bpy
from bpy.props import StringProperty
import requests
import webbrowser
from .auth import load_token, clear_token, initiate_login, start_local_server, ensure_internet_connection

import traceback
import threading
from .main_config import (DOWNLOAD_ENDPOINT, DOCS_URL)
from .version_info import CURRENT_VERSION
from .helper_functions import download_thumbnail, auto_refresh_usage

# ----------------------------------------------------------------------------
# LOGIN/LOGOUT
# ----------------------------------------------------------------------------

class LOGIN_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.login"
    bl_label = "Login to Web App"
    bl_options = {'REGISTER'}

    def execute(self, context):
        # 1) Make sure we're online (will clear token & disable UI if offline)
        if not ensure_internet_connection(context):
            self.report({'ERROR'}, "No internet connection detected. Cannot login.")
            return {'CANCELLED'}

        # 2) Refresh the cached "latest version" value
        from .exp_api import update_latest_version_cache, get_cached_latest_version
        update_latest_version_cache()
        latest = get_cached_latest_version()

        # 3) If there *is* a newer version, block login with a warning
        if latest and latest != CURRENT_VERSION:
            self.report(
                {'WARNING'},
                f"New Exploratory version {latest} available. Please update before logging in."
            )
            return {'CANCELLED'}

        # 4) All good → proceed with your normal login flow
        threading.Thread(target=start_local_server, args=(8000,), daemon=True).start()
        initiate_login()
        self.report({'INFO'}, "Login page opened. Complete login in your browser.")
        bpy.app.timers.register(auto_refresh_usage, first_interval=3.0)
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

#------------------------------------
#Documentation Link
#------------------------------------

class OPEN_DOCS_OT(bpy.types.Operator):
    bl_idname = "webapp.open_docs"
    bl_label = "Open Documentation"
    bl_description = "Open the online documentation"

    url: StringProperty(
        name="URL",
        default=DOCS_URL,
        description="Documentation page URL"
    )

    def execute(self, context):
        webbrowser.open(self.url)
        return {'FINISHED'}