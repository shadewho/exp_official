# Exploratory/Exp_UI/packages/social_operators.py

from .utilities import like_package, comment_package
import bpy
import webbrowser
from bpy.props import StringProperty
from ..interface.drawing.utilities import format_relative_time

class LIKE_PACKAGE_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.like_package"
    bl_label = "Like Package"

    skip_popup: bpy.props.BoolProperty(default=True)
    launched_from_persistent: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        file_id = context.scene.my_addon_data.file_id
        try:
            data = like_package(file_id)
            if data.get("success"):
                new_likes = data.get("likes")
                if new_likes is not None:
                    context.scene.my_addon_data.likes = new_likes
                self.report({'INFO'}, f"Liked item! (Total likes: {new_likes})")
            else:
                msg = data.get("message", "Failed to like package.")
                if "Already liked" in msg:
                    self.report({'INFO'}, "You already liked this item.")
                    return {'CANCELLED'}
                self.report({'ERROR'}, msg)
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Like error: {e}")
            return {'CANCELLED'}

        context.area.tag_redraw()
        return {'FINISHED'}

    def invoke(self, context, event):
        return self.execute(context)


class COMMENT_PACKAGE_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.comment_package"
    bl_label = "Comment on Item"

    comment_text: bpy.props.StringProperty(name="Comment", default="")
    skip_popup: bpy.props.BoolProperty(default=False)
    launched_from_persistent: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        if self.launched_from_persistent and not self.comment_text:
            self.comment_text = context.scene.comment_text

        if not self.comment_text.strip():
            self.report({'ERROR'}, "Comment cannot be empty")
            return {'CANCELLED'}

        file_id = context.scene.my_addon_data.file_id
        try:
            data = comment_package(file_id, self.comment_text)
            if not data.get("success"):
                self.report({'ERROR'}, data.get("message", "Failed to add comment."))
                return {'CANCELLED'}
            c = data.get("comment", {})
            new = context.scene.my_addon_data.comments.add()
            new.author = c.get("author", "Anonymous")
            new.text = c.get("content", self.comment_text)
            new.timestamp = c.get("timestamp", "")
            self.report({'INFO'}, "Comment added!")
            if self.launched_from_persistent:
                context.scene.comment_text = ""
            else:
                self.comment_text = ""
        except Exception as e:
            self.report({'ERROR'}, f"Connection error: {e}")
            return {'CANCELLED'}

        context.area.tag_redraw()
        return {'FINISHED'}

    def invoke(self, context, event):
        if self.launched_from_persistent or self.skip_popup:
            return self.execute(context)
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        if not self.launched_from_persistent:
            self.layout.prop(self, "comment_text")


class COMMENT_PACKAGE_INLINE_OT_WebApp(bpy.types.Operator):
    """Inline comment operator: no popup, for use in the social-details panel."""
    bl_idname = "webapp.comment_package_inline"
    bl_label = "Comment on Item (Inline)"

    comment_text: bpy.props.StringProperty(name="Comment", default="")

    def execute(self, context):
        text = context.scene.comment_text.strip()
        if not text:
            self.report({'ERROR'}, "Comment cannot be empty")
            return {'CANCELLED'}

        file_id = context.scene.my_addon_data.file_id
        try:
            data = comment_package(file_id, text)
        except Exception as e:
            self.report({'ERROR'}, f"Connection error: {e}")
            return {'CANCELLED'}

        if not data.get("success"):
            self.report({'ERROR'}, data.get("message", "Failed to add comment."))
            return {'CANCELLED'}

        c = data.get("comment", {})
        new = context.scene.my_addon_data.comments.add()
        new.author = c.get("author", "Anonymous")
        new.text = c.get("content", text)
        new.timestamp = c.get("timestamp", "")

        self.report({'INFO'}, "Comment added!")
        context.scene.comment_text = ""
        context.area.tag_redraw()
        return {'FINISHED'}

    def invoke(self, context, event):
        return self.execute(context)


class OPEN_URL_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.open_url"
    bl_label = "Open Profile"
    bl_description = "Open the author's profile in your web browser"

    url: StringProperty(
        name="URL",
        description="The URL to open",
        default=""
    )

    def execute(self, context):
        if not self.url:
            self.report({'ERROR'}, "No URL provided")
            return {'CANCELLED'}
        try:
            webbrowser.open(self.url)
            self.report({'INFO'}, f"Opening URL: {self.url}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to open URL: {e}")
            return {'CANCELLED'}


class EXPLORATORY_UL_Comments(bpy.types.UIList):
    """UIList for displaying MyAddonComment items."""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text=f"{item.author}: {item.text[:30]}")
        elif self.layout_type == 'GRID':
            layout.label(text=item.author)


class POPUP_SOCIAL_DETAILS_OT(bpy.types.Operator):
    bl_idname = "view3d.popup_social_details"
    bl_label = "World Social Details"
    bl_options = {'REGISTER'}

    def execute(self, context):
        return {'FINISHED'}

    def invoke(self, context, event):
        # Use a props-dialog so it’s centered in the 3D View and stays until OK/Cancel
        return context.window_manager.invoke_props_dialog(self, width=600)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        addon_data = scene.my_addon_data

        layout.label(text="World Social Details", icon='WORLD')
        layout.separator()

        if addon_data.file_id > 0:
            # — Info Box —
            box = layout.box()
            box.label(text=f"Title: {addon_data.package_name}", icon='BOOKMARKS')
            box.label(text=f"Description: {addon_data.description}")
            box.label(
                text=f"Uploaded: {format_relative_time(addon_data.upload_date)} ago"
            )
            box.label(
                text=f"Downloads: {addon_data.download_count}", icon='IMPORT'
            )
            row = box.row(align=True)
            row.label(text=f"Author: {addon_data.author}")
            if addon_data.profile_url:
                op = row.operator("webapp.open_url", text="Profile")
                op.url = addon_data.profile_url

            layout.separator()

            # Vote vs. Like
            if scene.package_item_type == 'event' and scene.event_stage == 'voting':
                vop = layout.row().operator("webapp.vote_map", text="★ Vote")
                vop.skip_popup = True
            else:
                lop = layout.row().operator(
                    "webapp.like_package", text=f"♥ {addon_data.likes}"
                )
                lop.skip_popup = True
                lop.launched_from_persistent = True

            layout.separator()
            layout.label(text="Comments:", icon='COMMUNITY')

            # Traditional UIList with scrollbar, showing up to 5 rows
            row = layout.row()
            row.template_list(
                "EXPLORATORY_UL_Comments", "",
                addon_data, "comments",
                addon_data, "active_comment_index",
                rows=5
            )

            layout.separator()
            # Inline entry for new comment
            row = layout.row(align=True)
            row.prop(scene, "comment_text", text="", emboss=True)
            row.operator("webapp.comment_package_inline", text="", icon='ADD')

        else:
            layout.label(text="No social data available.", icon='INFO')
