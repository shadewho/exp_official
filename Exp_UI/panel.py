# panel.py

import bpy
from .auth import load_token
from .helper_functions import format_relative_time

class VIEW3D_PT_SubscriptionUsage(bpy.types.Panel):
    bl_label = "Subscription Usage"
    bl_idname = "VIEW3D_PT_subscription_usage"
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
            layout.label(text="Please log in to see usage data", icon='ERROR')
            return

        addon_data = scene.my_addon_data
        layout.label(text=f"Plan: {addon_data.subscription_tier}")
        layout.label(text=f"Downloads: {addon_data.downloads_used} / {addon_data.downloads_limit}")
        layout.label(text=f"Uploads: {addon_data.uploads_used}")
        remaining = addon_data.downloads_limit - addon_data.downloads_used
        layout.label(text=f"Remaining Downloads: {remaining}")

        layout.operator("webapp.refresh_usage", text="Refresh Usage")


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

        # Master buttons in their own rows (stacked vertically)
        row = layout.row()
        op = row.operator("webapp.apply_filters_showui", text="Exploratory Interface", icon='RESTRICT_VIEW_OFF')
        op.page_number = 1

        row = layout.row()
        row.operator("webapp.refresh_filters", text="Refresh Interface", icon='FILE_REFRESH')

        layout.separator()
        layout.separator()

        # Filter UI:
        row = layout.row()
        row.alignment = 'CENTER'
        row.label(text="-Filters-")
        layout.prop(scene, "package_item_type", text="Item Type")
        layout.prop(scene, "package_sort_by", text="Sort By")
        layout.prop(scene, "package_search_query", text="Search")

        layout.separator()

        # Scene "Find Item" UI:
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

            # 3) Description (one line)
            layout.label(text=f"Description: {addon_data.description}")

            # 4) Upload Date
            layout.label(text=f"Uploaded: {format_relative_time(addon_data.upload_date)} ago")

            # 5) Likes (heart + count)
            row = layout.row(align=True)
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

