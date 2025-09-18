#Exploratory/Exp_UI/panel.py

import bpy
from .auth.helpers import load_token
from .interface.drawing.utilities import format_relative_time
from .version_info import CURRENT_VERSION
from .version_info import get_cached_latest_version
from .main_config import PROFILE_URL

class VIEW3D_PT_ProfileAccount(bpy.types.Panel):
    bl_label = "Profile / Account"
    bl_idname = "VIEW3D_PT_profile_account"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}


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
        tier = getattr(addon_data, "subscription_tier", "Tier 1")
        used = int(getattr(addon_data, "downloads_used", 0))
        limit = int(getattr(addon_data, "downloads_limit", 0))
        scope = getattr(addon_data, "downloads_scope", "daily")  # "lifetime" or "daily"
        uploads_used = int(getattr(addon_data, "uploads_used", 0))

        layout.label(text=f"Plan: {tier}")

        # Label text depends on scope
        if scope == "lifetime":
            layout.label(text=f"Lifetime Downloads: {used} / {limit}")
        else:
            layout.label(text=f"Daily Downloads: {used} / {limit}")

        layout.label(text=f"Uploads: {uploads_used}")

        remaining = max(0, limit - used)
        if scope == "lifetime":
            layout.label(text=f"Remaining Lifetime Downloads: {remaining}")
        else:
            layout.label(text=f"Remaining Daily Downloads: {remaining}")

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
    bl_options = {'DEFAULT_CLOSED'}

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
            # — Main Info Box (same order/feel as popup) —
            box = layout.box()

            # Title (more “title-like” icon)
            box.label(text=f"Title: {addon_data.package_name}")
            box.label(text=f"Description: {addon_data.description}")
            box.label(text=f"Uploaded: {format_relative_time(addon_data.upload_date)} ago")
            box.label(text=f"Downloads: {addon_data.download_count}", icon='IMPORT')

            # Author (no "Author:" label; clickable with web icon if URL exists)
            author_row = box.row(align=True)
            if addon_data.profile_url:
                op = author_row.operator("webapp.open_url", text=addon_data.author, icon='URL')
                op.url = addon_data.profile_url
            else:
                author_row.label(text=addon_data.author)

            # Action buttons — inside the box, left-aligned, widened
            btn_row = box.row(align=True)
            btn_row.alignment = 'LEFT'

            like_group = btn_row.row(align=True)
            like_group.scale_x = 1.8  # widen but not full-width
            like_op = like_group.operator("webapp.like_package", text=f"♥ {addon_data.likes}")
            like_op.skip_popup = True
            # panel is persistent, so keep counts in sync just like popup
            like_op.launched_from_persistent = True

            if scene.package_item_type == 'event' and scene.event_stage == 'voting':
                vote_group = btn_row.row(align=True)
                vote_group.scale_x = 1.8
                vote_op = vote_group.operator("webapp.vote_map", text="★ Vote")
                vote_op.skip_popup = True

            # Comments
            box.separator()
            box.label(text="Comments:", icon='COMMUNITY')

            list_row = box.row()
            list_row.template_list(
                "EXPLORATORY_UL_Comments", "",
                addon_data, "comments",
                addon_data, "active_comment_index",
                rows=5
            )

            entry = box.row(align=True)
            entry.prop(scene, "comment_text", text="", emboss=True)
            entry.operator("webapp.comment_package_inline", text="", icon='ADD')

        else:
            layout.label(text="No active item to display.", icon='INFO')


class VIEW3D_PT_SettingsAndUpdate(bpy.types.Panel):
    bl_label = "Settings and Update"
    bl_idname = "VIEW3D_PT_settings_update"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        return context.scene.main_category == 'EXPLORE'

    def draw(self, context):
        layout = self.layout

        # Docs link
        layout.operator("webapp.open_docs", text="Documentation", icon='HELP')
        layout.separator()

        # Version info in its own box
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

        # Audio controls (unchanged)…
        box = layout.box()
        box.label(text="Audio", icon='SOUND')
        prefs = context.preferences.addons["Exploratory"].preferences
        split = box.split(factor=0.5, align=True)
        col = split.column(align=True)
        icon = 'RADIOBUT_ON' if prefs.enable_audio else 'RADIOBUT_OFF'
        col.prop(prefs, "enable_audio", text="Master Volume", icon=icon)
        split.column(align=True).prop(prefs, "audio_level", text="Volume", slider=True)

        layout.separator()

        # Adjust Persistent Settings box
        prefs_box = layout.box()
        prefs_box.label(text="Adjust Persistent Settings", icon='PREFERENCES')
        prefs_box.operator("wm.open_addon_prefs", text="Open Add-on Preferences")

        # Bullet list (using label with indentation for a bullet-like effect)
        prefs_box.label(text="• Change skins and audio")
        prefs_box.label(text="• Change performance settings")
        prefs_box.label(text="• Change key binds and sensitivity")
        prefs_box.label(text="• Find \"Exploratory\" and adjust settings")
