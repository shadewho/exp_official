# panel.py

import bpy
from .auth import load_token

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
            layout.prop(scene, "username")
            layout.prop(scene, "password")
            layout.operator("webapp.login", text="Login")
        else:
            layout.label(text="You are logged in!")
            layout.operator("webapp.logout", text="Logout")


class VIEW3D_PT_PackageDisplay_FilterAndScene(bpy.types.Panel):
    bl_label = "Filter + Scene"
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
        # Two buttons: one to display the UI, one to refresh in-place
        row = layout.row(align=True)
        op = row.operator("webapp.apply_filters_showui", text="Display UI", icon='RESTRICT_VIEW_OFF')
        op.page_number = 1

        row.operator("webapp.refresh_filters", text="Refresh UI", icon='FILE_REFRESH')

        layout.separator()
        
        # Filter UI
        layout.label(text="Filters:")
        layout.prop(scene, "package_item_type", text="Item Type")
        layout.prop(scene, "package_sort_by", text="Sort By")
        layout.prop(scene, "package_search_query", text="Search")

        layout.separator()

        # Scene "Find Item" UI
        layout.label(text="Find Item by Download Code:")
        layout.prop(scene, "download_code", text="Code")
        layout.operator("webapp.download_code", text="Find Item")



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

        # Only show if we have a valid item
        if (scene.ui_current_mode == "DETAIL" or addon_data.is_from_webapp) and addon_data.file_id > 0:
            # 1) Author row
            row = layout.row()
            row.label(text=f"By {addon_data.author}")
            if addon_data.profile_url:
                row.operator("webapp.open_url", text="(profile)", icon='URL').url = addon_data.profile_url

            # 2) Item Name
            layout.label(text=f"Item Name: {addon_data.package_name}")

            # 3) Description
            layout.label(text=f"Description: {addon_data.description}")

            # 4) Upload Date
            layout.label(text=f"Upload Date: {addon_data.upload_date}")

            # 5) Likes
            row_likes = layout.row()
            row_likes.label(text=str(addon_data.likes), icon='FUND')  # Heart icon
            row_likes.operator("webapp.like_package", text="Like")

            # -- Comments Section --
            layout.label(text="Comments:")

            # A) UIList for comments:
            row = layout.row()
            row.template_list(
                "EXPLORATORY_UL_Comments",  # The UIList class
                "",                         # list_id (not needed)
                addon_data,                # The owner of the collection
                "comments",                # The CollectionProperty name
                addon_data,                # The active index owner
                "active_comment_index",    # The name of the active index
                rows=5
            )

            # B) “Add Comment” operator with no text, plus icon
            layout.operator("webapp.comment_package", text="", icon='ADD')

        else:
            layout.label(text="No active item to display.", icon='INFO')




