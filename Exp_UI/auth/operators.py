#Exploratory/Exp_UI/auth/operators.py

import bpy
import webbrowser
from bpy.props import StringProperty
import threading
import requests
from ..version_info import CURRENT_VERSION
from .helpers import auto_refresh_usage
from ..internet.helpers import ensure_internet_connection, start_local_server, initiate_login
from ..auth.helpers import clear_token
from ..auth.helpers import load_token
from ..main_config import USAGE_ENDPOINT, DOCS_URL
from ..cache_system.preload import stop_cache_worker, start_cache_worker

# ----------------------------------------------------------------------------
# LOGIN/LOGOUT
# ----------------------------------------------------------------------------
def _restart_cache_worker():
    # this will be called every second until load_token() returns truthy
    if load_token():
        stop_cache_worker()
        start_cache_worker()
        return None   # stop the timer
    return 1.0        # retry in 1s

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
        
        # Force UI to browse mode on login
        context.scene.ui_current_mode = 'BROWSE'
        # 5) Refresh usage meter after login
        bpy.app.timers.register(auto_refresh_usage, first_interval=3.0)

        # 6) Restart cache worker *after* token is actually present
        bpy.app.timers.register(_restart_cache_worker, first_interval=1.0)

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
#refresh subscription usage
# ----------------------------------------------------------------------------
class REFRESH_USAGE_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.refresh_usage"
    bl_label = "Refresh Subscription Usage"

    def execute(self, context):
        token = load_token()
        if not token:
            self.report({'ERROR'}, "Not logged in")
            return {'CANCELLED'}

        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = requests.get(USAGE_ENDPOINT, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            self.report({'ERROR'}, f"Error fetching usage: {e}")
            return {'CANCELLED'}

        if not data.get("success"):
            self.report({'ERROR'}, data.get("message", "Usage data error"))
            return {'CANCELLED'}

        # Update scene properties with the data from the backend.
        scene = context.scene
        addon_data = scene.my_addon_data

        addon_data.subscription_tier = data.get("subscription_tier", "Free")
        addon_data.downloads_used    = int(data.get("downloads_used", 0))
        addon_data.downloads_limit   = int(data.get("downloads_limit", 0))
        addon_data.downloads_scope   = data.get("downloads_scope", "daily")  # ← NEW
        addon_data.uploads_used      = int(data.get("uploads_used", 0))
        addon_data.username          = data.get("username", "")
        addon_data.profile_url       = data.get("profile_url", "")

        self.report({'INFO'}, "Subscription usage refreshed")
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