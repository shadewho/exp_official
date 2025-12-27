import bpy
from .auth.helpers import load_token
from .version_info import CURRENT_VERSION, get_cached_latest_version
from .main_config import PROFILE_URL, WORLD_URL, SHOP_URL, EVENTS_URL


# ─────────────────────────────────────────────────────────────
# Exploratory Main Panel - Unified Explore + Account
# ─────────────────────────────────────────────────────────────
class VIEW3D_PT_ExploreByCode(bpy.types.Panel):
    bl_label = "Exploratory"
    bl_idname = "VIEW3D_PT_explore_by_code"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Explore and Search"

    @classmethod
    def poll(cls, context):
        return getattr(context.scene, "main_category", 'EXPLORE') == 'EXPLORE'

    def draw(self, context):
        layout = self.layout
        token = load_token()

        # ─── Not logged in ───
        if not token:
            error_msg = context.scene.login_error_message
            if error_msg:
                error_box = layout.box()
                for line in error_msg.split('\n'):
                    error_box.label(text=line, icon='INFO')
                layout.separator()

            layout.operator("webapp.login", text="Log In", icon='URL')
            layout.separator()
            layout.label(text="Log in to explore user creations!", icon='INFO')
            return

        # ─── Logged in ───
        addon_data = context.scene.my_addon_data

        # Log Out button
        layout.operator("webapp.logout", text="Log Out", icon='URL')
        layout.separator()

        # ─── Play Section ───
        code_box = layout.box()

        # Header
        code_box.label(text="Play Worlds & Games", icon='PLAY')

        # Code field (full width)
        code_box.label(text="Enter a download code to play:")
        code_box.prop(context.scene, "download_code", text="")

        # Paste and Search buttons side by side
        row = code_box.row(align=True)
        row.operator("webapp.paste_code", text="Paste", icon='PASTEDOWN')
        row.operator("webapp.show_detail_by_code", text="Search", icon='VIEWZOOM')

        layout.separator()

        # ─── Browse Section ───
        browse_box = layout.box()
        browse_box.label(text="Browse", icon='WORLD')
        row = browse_box.row(align=True)
        op_world = row.operator("webapp.browse_world", text="World", icon='WORLD')
        op_shop = row.operator("webapp.browse_shop", text="Shop", icon='TAG')
        op_events = row.operator("webapp.browse_events", text="Events", icon='TIME')

        layout.separator()

        # ─── Account Section ───
        account_box = layout.box()
        account_box.label(text="Account", icon='USER')

        # Username / Profile link
        username = addon_data.username or "Profile"
        row = account_box.row(align=True)
        op = row.operator("webapp.open_url", text=username, icon='URL')
        op.url = PROFILE_URL

        account_box.separator()

        # Subscription tier
        account_box.label(text=f"Tier: {addon_data.subscription_tier or 'Unknown'}")

        # Downloads
        downloads_scope_label = addon_data.downloads_scope.capitalize() if addon_data.downloads_scope else "Daily"
        account_box.label(text=f"Downloads ({downloads_scope_label}): {addon_data.downloads_used} / {addon_data.downloads_limit}")

        # Uploads
        account_box.label(text=f"Uploads: {addon_data.uploads_used} / {addon_data.uploads_limit}")

        account_box.separator()

        # Refresh button
        account_box.operator("webapp.refresh_usage", text="Refresh", icon='FILE_REFRESH')


# Keep this for backwards compatibility but it's now empty/hidden
class VIEW3D_PT_ProfileAccount(bpy.types.Panel):
    bl_label = "Profile / Account"
    bl_idname = "VIEW3D_PT_profile_account"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'HIDE_HEADER'}

    @classmethod
    def poll(cls, context):
        # Always return False to hide this panel
        return False

    def draw(self, context):
        pass



# ─────────────────────────────────────────────────────────────
# Settings & Update (unchanged except for imports)
# ─────────────────────────────────────────────────────────────
class VIEW3D_PT_SettingsAndUpdate(bpy.types.Panel):
    bl_label = "Settings and Update"
    bl_idname = "VIEW3D_PT_settings_update"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return getattr(context.scene, "main_category", 'EXPLORE') == 'EXPLORE'

    def draw(self, context):
        layout = self.layout

        # Docs link
        layout.operator("webapp.open_docs", text="Documentation", icon='HELP')
        layout.separator()

        # Version info
        ver_box = layout.box()
        ver_box.label(text="Add-on Version", icon='FILE_BLEND')
        ver_box.label(text=f"Current Exploratory Version: {CURRENT_VERSION}")

        latest = get_cached_latest_version()
        if latest is None:
            ver_box.label(text="Update status unknown", icon='QUESTION')
            ver_box.operator("webapp.refresh_version", icon='FILE_REFRESH')
        elif latest == CURRENT_VERSION:
            ver_box.label(text="Exploratory is up to date!", icon='CHECKMARK')
            ver_box.operator("webapp.refresh_version", text="Re-check", icon='FILE_REFRESH')
        else:
            ver_box.label(text=f"Update Available: {latest}", icon='ERROR')
            ver_box.operator_context = 'INVOKE_DEFAULT'
            ver_box.operator("webapp.update_addon", text="Update Add-on", icon='IMPORT')

        layout.separator()

        # Audio controls (unchanged)
        box = layout.box()
        box.label(text="Audio", icon='SOUND')
        prefs = context.preferences.addons["Exploratory"].preferences
        split = box.split(factor=0.5, align=True)
        col = split.column(align=True)
        icon = 'RADIOBUT_ON' if prefs.enable_audio else 'RADIOBUT_OFF'
        col.prop(prefs, "enable_audio", text="Master Volume", icon=icon)
        split.column(align=True).prop(prefs, "audio_level", text="Volume", slider=True)

        layout.separator()

        # Persistent Settings
        prefs_box = layout.box()
        prefs_box.label(text="Adjust Persistent Settings", icon='PREFERENCES')
        prefs_box.operator("wm.open_addon_prefs", text="Open Add-on Preferences")
        prefs_box.label(text="• Change skins and audio")
        prefs_box.label(text="• Change performance settings")
        prefs_box.label(text="• Change key binds and sensitivity")
        prefs_box.label(text="• Find \"Exploratory\" and adjust settings")