# panel.py

import bpy
from .auth.helpers import load_token
from .interface.drawing.utilities import format_relative_time
from .version_info import CURRENT_VERSION
from .exp_api import get_cached_latest_version
from .main_config import PROFILE_URL

class VIEW3D_PT_ProfileAccount(bpy.types.Panel):
    bl_label = "Profile / Account"
    bl_idname = "VIEW3D_PT_profile_account"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    @classmethod
    def poll(cls, context):
        return context.scene.main_category == 'EXPLORE'

    def draw(self, context):
        layout = self.layout
        scene  = context.scene

        # 1) must be logged in
        token = load_token()
        if not token:
            layout.label(text="Please log in to access your account", icon='ERROR')
            return

        # 2) safe to show profile link + usage
        addon_data = scene.my_addon_data
        username = getattr(addon_data, "username", "User")

        # ——— Profile link ———
        row = layout.row(align=True)
        op = row.operator("webapp.open_url", text=username, icon='URL')
        op.url = PROFILE_URL

        layout.separator()

        # ——— Subscription Usage ———
        layout.label(text=f"Plan: {addon_data.subscription_tier}")
        layout.label(text=f"Downloads: {addon_data.downloads_used} / {addon_data.downloads_limit}")
        layout.label(text=f"Uploads: {addon_data.uploads_used}")
        remaining = addon_data.downloads_limit - addon_data.downloads_used
        layout.label(text=f"Remaining Downloads: {remaining}")

        # optional “refresh” button
        layout.operator("webapp.refresh_usage", text="Refresh Usage", icon='FILE_REFRESH')


class VIEW3D_PT_PackageDisplay_Login(bpy.types.Panel):
    bl_label = "Login / Logout"
    bl_idname = "VIEW3D_PT_package_display_login"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    @classmethod
    def poll(cls, context):
        return context.scene.main_category == 'EXPLORE'
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        token = load_token()

        if not token:
            layout.label(text="Please log in:")
            layout.operator("webapp.login", text="Login", icon='URL')
        else:
            layout.label(text="You are logged in!")
            layout.operator("webapp.logout", text="Logout")


class VIEW3D_PT_PackageDisplay_FilterAndScene(bpy.types.Panel):
    bl_label = "Explore and Search"
    bl_idname = "VIEW3D_PT_package_display_filter_scene"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    @classmethod
    def poll(cls, context):
        return context.scene.main_category == 'EXPLORE'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        token = load_token()

        if not token:
            layout.label(text="(Log in to access filters and scenes.)", icon='ERROR')
            return

        # Master buttons
        row = layout.row()
        op = row.operator("webapp.apply_filters_showui", text="Exploratory Interface", icon='RESTRICT_VIEW_OFF')
        op.page_number = 1

        row = layout.row()
        row.operator("webapp.refresh_filters", text="Refresh Interface", icon='FILE_REFRESH')

        layout.separator()
        layout.separator()

        # Filter UI header
        row = layout.row()
        row.alignment = 'CENTER'
        row.label(text="-Filters-")

        # 1) Item Type toggle buttons
        row = layout.row(align=True)
        row.prop(scene, "package_item_type", text="Item Type", expand=True)

        # If the package type is 'event', display event-specific filters.
        if scene.package_item_type == 'event':
            layout.separator()
            layout.prop(scene, "event_stage", text="Event Stage", expand=True)
            layout.prop(scene, "selected_event", text="Event")
        else:
            # For non-event types, show sort and search controls.
            layout.prop(scene, "package_sort_by", text="Sort By")
            layout.prop(scene, "package_search_query", text="Search")

        layout.separator()

        # Scene "Find Item" UI
        layout.label(text="Download Code:")
        split = layout.split(factor=0.85)
        split.prop(scene, "download_code", text="")  # Text field with no label.
        split.operator("webapp.download_code", text="", icon='VIEWZOOM')





class VIEW3D_PT_PackageDisplay_CurrentItem(bpy.types.Panel):
    bl_label = "Current Item"
    bl_idname = "VIEW3D_PT_package_display_current_item"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    @classmethod
    def poll(cls, context):
        return context.scene.main_category == 'EXPLORE'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        token = load_token()

        if not token:
            layout.label(text="(Log in to see item details.)", icon='ERROR')
            return

        addon_data = scene.my_addon_data

        # Only show details if in DETAIL mode and we have a valid file_id
        if scene.ui_current_mode == "DETAIL" and addon_data.file_id > 0:
            # 1) Title
            layout.label(text=f"Title: {addon_data.package_name}")

            # 2) Author
            row = layout.row(align=True)
            row.label(text="Author:")
            if addon_data.profile_url:
                op = row.operator("webapp.open_url", text=addon_data.author, icon='URL')
                op.url = addon_data.profile_url
            else:
                row.label(text=addon_data.author)

            # 3) Description
            layout.label(text=f"Description: {addon_data.description}")

            # 4) Upload Date
            layout.label(text=f"Uploaded: {format_relative_time(addon_data.upload_date)} ago")
            
            # 4.5) Download Count
            layout.label(text=f"Downloads: {addon_data.download_count}")

            # 5) Either show the vote/favorite button or the like button:
            row = layout.row(align=True)
            if scene.package_item_type == 'event' and scene.event_stage == 'voting':
                # Show "Favorite" button with a star icon (★)
                vote_op = row.operator("webapp.vote_map", text="★ Vote")
                vote_op.skip_popup = True
            else:
                like_op = row.operator("webapp.like_package", text=f"♥ {addon_data.likes}")
                like_op.skip_popup = True

            # 6) Comments
            layout.label(text="Comments:")
            row = layout.row()
            row.template_list(
                "EXPLORATORY_UL_Comments", "",
                addon_data, "comments",
                addon_data, "active_comment_index",
                rows=5
            )
            layout.operator("webapp.comment_package", text="Add Comment", icon='ADD').skip_popup = False

        else:
            layout.label(text="No active item to display.", icon='INFO')


class VIEW3D_PT_SettingsAndUpdate(bpy.types.Panel):
    bl_label = "Settings and Update"
    bl_idname = "VIEW3D_PT_settings_update"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    def draw(self, context):
        layout = self.layout

        # Docs link
        layout.operator("webapp.open_docs", text="Documentation", icon='HELP')
        layout.separator()

        # Current version
        layout.label(text=f"Current Exploratory Version: {CURRENT_VERSION}")
        layout.separator()

        # —— read from cache, no network here! ——
        latest = get_cached_latest_version()
        if latest is None:
            # never fetched (or fetch failed)
            layout.label(text="Update status unknown", icon='QUESTION')
            layout.operator("webapp.refresh_version", icon='FILE_REFRESH')
        elif latest == CURRENT_VERSION:
            layout.label(text="Exploratory is up to date!", icon='CHECKMARK')
            layout.operator("webapp.refresh_version", text="Re-check", icon='FILE_REFRESH')
        else:
            layout.label(text=f"Update Available: {latest}", icon='ERROR')
            # force invoke so invoke_props_dialog() is called instead of execute()
            layout.operator_context = 'INVOKE_DEFAULT'
            layout.operator("webapp.update_addon", text="Update Add-on", icon='IMPORT')

        layout.separator()

        # Audio controls (unchanged)…
        box = layout.box()
        box.label(text="Audio", icon='SOUND')
        prefs = context.preferences.addons["Exploratory"].preferences
        split = box.split(factor=0.5, align=True)
        col = split.column(align=True)
        icon = 'RADIOBUT_ON' if prefs.enable_audio else 'RADIOBUT_OFF'
        col.prop(prefs, "enable_audio", text="Master Volume", icon=icon)
        split.column(align=True).prop(prefs, "audio_level", text="Volume", slider=True)