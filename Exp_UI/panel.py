import bpy
from .auth.helpers import load_token
from .version_info import CURRENT_VERSION, get_cached_latest_version
from .main_config import PROFILE_URL
# ─────────────────────────────────────────────────────────────
# Explore / Log in/ Logout
# ─────────────────────────────────────────────────────────────
class VIEW3D_PT_ExploreByCode(bpy.types.Panel):
    bl_label = "Explore / Log In"
    bl_idname = "VIEW3D_PT_explore_by_code"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    @classmethod
    def poll(cls, context):
        # Only show when the EXPLORE tab is active
        return getattr(context.scene, "main_category", 'EXPLORE') == 'EXPLORE'

    def draw(self, context):
        layout = self.layout
        token = load_token()

        # — Top button (normal full-width, no box) —
        if not token:
            layout.operator("webapp.login", text="Log In", icon='URL')
            layout.separator()
            layout.label(text="Log in to explore user creations!", icon='INFO')
            return
        else:
            layout.operator("webapp.logout", text="Log Out", icon='URL')
            layout.separator()

        layout.separator()

        # Download Code (boxed) — REPLACE THIS WHOLE BLOCK
        code_box = layout.box()
        code_box.label(icon="INFO", text="Explore user creations below!")
        code_box.label(icon="INFO", text="Download code required.")
        code_box.separator()
        code_box.label(text="Download Code:")

        # Row 1: code field + Search (icon button)
        row = code_box.split(factor=0.9, align=True)  # tweak factor to give the button more space
        row.prop(context.scene, "download_code", text="")
        btns = row.row(align=True)
        btns.operator("webapp.show_detail_by_code", text="", icon='VIEWZOOM')  # Search

        code_box.label(text="Or:")
        # Row 2: full-width Paste & Search button (separate line, same box)
        paste = code_box.column(align=True)
        paste.operator("webapp.paste_and_search", text="Paste & Search", icon='PASTEDOWN')

        code_box.separator()


# ─────────────────────────────────────────────────────────────
# Profile / Account — link only (no usage)
# ─────────────────────────────────────────────────────────────
class VIEW3D_PT_ProfileAccount(bpy.types.Panel):
    bl_label = "Profile / Account"
    bl_idname = "VIEW3D_PT_profile_account"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        # Keep this category gate if you use it elsewhere
        return getattr(context.scene, "main_category", 'EXPLORE') == 'EXPLORE'

    def draw(self, context):
        layout = self.layout
        token = load_token()
        if not token:
            layout.label(text="Please log in to access your account", icon='ERROR')
            return

        username = getattr(context.scene.my_addon_data, "username", "") or "Profile"
        row = layout.row(align=True)
        op = row.operator("webapp.open_url", text=username, icon='URL')
        op.url = PROFILE_URL



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