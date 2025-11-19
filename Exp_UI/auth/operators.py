#Exploratory/Exp_UI/auth/operators.py

import bpy
import webbrowser
from bpy.props import StringProperty
import threading
from ..version_info import CURRENT_VERSION
from ..internet.helpers import ensure_internet_connection, start_local_server, initiate_login
from ..auth.helpers import clear_token
from ..main_config import DOCS_URL

# ----------------------------------------------------------------------------
# LOGIN/LOGOUT
# ----------------------------------------------------------------------------

class LOGIN_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.login"
    bl_label = "Login to Web App"
    bl_options = {'REGISTER'}

    def execute(self, context):
        # 1) Ensure we're online
        if not ensure_internet_connection(context):
            self.report({'ERROR'}, "No internet connection detected. Cannot login.")
            return {'CANCELLED'}

        # 2) Refresh the cached "latest version" value
        from ..version_info import update_latest_version_cache, get_cached_latest_version
        update_latest_version_cache()
        latest = get_cached_latest_version()

        # 3) If there *is* a newer version, block login
        if latest and latest != CURRENT_VERSION:
            self.report(
                {'WARNING'},
                f"New Exploratory version {latest} available. Please update before logging in."
            )
            return {'CANCELLED'}

        # 4) Kick off OAuth flow
        threading.Thread(target=start_local_server, args=(8000,), daemon=True).start()
        initiate_login()
        self.report({'INFO'}, "Login page opened. Complete login in your browser.")


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