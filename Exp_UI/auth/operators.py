#Exploratory/Exp_UI/auth/operators.py

import bpy
import webbrowser
from bpy.props import StringProperty
import threading
from ..version_info import CURRENT_VERSION
from ..internet.helpers import ensure_internet_connection, start_local_server, initiate_login
from ..auth.helpers import clear_token
from ..main_config import DOCS_URL, WORLD_URL, SHOP_URL, EVENTS_URL

def clear_login_error():
    """Timer callback to clear the login error message after 5 seconds"""
    if hasattr(bpy.context.scene, 'login_error_message'):
        bpy.context.scene.login_error_message = ""
    return None  # Don't repeat timer

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
            context.scene.login_error_message = "Login Failed\nNo Internet Connection"
            bpy.app.timers.register(clear_login_error, first_interval=5.0)
            self.report({'ERROR'}, "No internet connection detected. Cannot login.")
            return {'CANCELLED'}

        # 2) Refresh the cached "latest version" value
        from ..version_info import update_latest_version_cache, get_cached_latest_version
        update_latest_version_cache()
        latest = get_cached_latest_version()

        # 3) Block login if we couldn't fetch version (rate limited, server error, etc.)
        if latest is None:
            context.scene.login_error_message = "Login Failed\nCouldn't verify addon version\nPlease try again in a moment"
            bpy.app.timers.register(clear_login_error, first_interval=5.0)
            self.report(
                {'ERROR'},
                "Failed to check addon version. You may be rate limited. Please try again."
            )
            return {'CANCELLED'}

        # 4) If there *is* a newer version, block login
        if latest != CURRENT_VERSION:
            context.scene.login_error_message = f"Login Failed\nNew Update Available: {latest}"
            bpy.app.timers.register(clear_login_error, first_interval=5.0)
            self.report(
                {'WARNING'},
                f"New Exploratory version {latest} available. Please update before logging in."
            )
            return {'CANCELLED'}

        # 5) Clear any previous error messages on successful login
        context.scene.login_error_message = ""

        # 6) Kick off OAuth flow
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


class REFRESH_USAGE_OT(bpy.types.Operator):
    bl_idname = "webapp.refresh_usage"
    bl_label = "Refresh Usage Data"
    bl_description = "Fetch current usage data from the server"
    bl_options = {'REGISTER'}

    def execute(self, context):
        from ..auth.helpers import get_usage_data

        try:
            data = get_usage_data()

            # Update scene properties with usage data
            addon_data = context.scene.my_addon_data
            addon_data.subscription_tier = data.get("subscription_tier", "Unknown")
            addon_data.downloads_used = data.get("downloads_used", 0)
            addon_data.downloads_limit = data.get("downloads_limit", 0)
            addon_data.downloads_scope = data.get("downloads_scope", "daily")
            addon_data.uploads_used = data.get("uploads_used", 0)
            addon_data.uploads_limit = data.get("uploads_limit", 0)
            addon_data.username = data.get("username", "")

            self.report({'INFO'}, "Usage data refreshed successfully")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed to fetch usage data: {str(e)}")
            return {'CANCELLED'}


# ─────────────────────────────────────────────────────────────
# Browse Operators
# ─────────────────────────────────────────────────────────────

class BROWSE_WORLD_OT(bpy.types.Operator):
    bl_idname = "webapp.browse_world"
    bl_label = "Browse Worlds"
    bl_description = "Browse worlds on Exploratory"

    def execute(self, context):
        webbrowser.open(WORLD_URL)
        return {'FINISHED'}


class BROWSE_SHOP_OT(bpy.types.Operator):
    bl_idname = "webapp.browse_shop"
    bl_label = "Browse Shop"
    bl_description = "Browse shop on Exploratory"

    def execute(self, context):
        webbrowser.open(SHOP_URL)
        return {'FINISHED'}


class BROWSE_EVENTS_OT(bpy.types.Operator):
    bl_idname = "webapp.browse_events"
    bl_label = "Browse Events"
    bl_description = "Browse events on Exploratory"

    def execute(self, context):
        webbrowser.open(EVENTS_URL)
        return {'FINISHED'}