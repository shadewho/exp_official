#Exploratory/Exp_UI/auth/operators.py

import bpy
from bpy.props import StringProperty
import threading
from ..version_info import CURRENT_VERSION
from ..helper_functions import auto_refresh_usage
from ..internet.helpers import ensure_internet_connection, start_local_server, initiate_login
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
        from ..exp_api import update_latest_version_cache, get_cached_latest_version
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