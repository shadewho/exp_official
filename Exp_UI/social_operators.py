# social_operators.py

from .auth import load_token
from .main_config import LIKE_PACKAGE_ENDPOINT, COMMENT_PACKAGE_ENDPOINT
from .exp_api import like_package, comment_package
import bpy
import bpy
import webbrowser
from bpy.props import StringProperty

class LIKE_PACKAGE_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.like_package"
    bl_label = "Like Package"

    def execute(self, context):
        file_id = context.scene.my_addon_data.file_id
        try:
            data = like_package(file_id)
            # Now we won't get an exception for HTTP 400 "Already liked"
            # Instead, data might be { "success": false, "message": "Already liked" }

            if data.get("success"):
                new_likes = data.get("likes")
                if new_likes is not None:
                    context.scene.my_addon_data.likes = new_likes
                self.report({'INFO'}, f"Liked item! (Total likes: {new_likes})")
                return {'FINISHED'}

            else:
                # success=False
                msg = data.get("message", "Failed to like package.")
                # Check if it's "Already liked"
                if "Already liked" in msg:
                    self.report({'INFO'}, "You already liked this item.")
                    # No red error popup. Let's just return FINISHED or CANCELLED at your preference.
                    return {'CANCELLED'}

                # Some other failure scenario
                self.report({'ERROR'}, msg)
                return {'CANCELLED'}

        except Exception as e:
            # This catches network issues, 401, etc.
            self.report({'ERROR'}, f"Like error: {e}")
            return {'CANCELLED'}



class COMMENT_PACKAGE_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.comment_package"
    bl_label = "Comment on Item"

    comment_text: bpy.props.StringProperty(name="Comment")

    def execute(self, context):
        file_id = context.scene.my_addon_data.file_id
        try:
            data = comment_package(file_id, self.comment_text)
            if data.get("success"):
                # The server returns {"success": True, "comment": {...}} on success
                comment_data = data.get("comment", {})
                new_comment = context.scene.my_addon_data.comments.add()
                new_comment.author = comment_data.get("author", "Anonymous")
                new_comment.text = comment_data.get("text", self.comment_text)
                new_comment.timestamp = comment_data.get("timestamp", "")
                self.report({'INFO'}, "Comment added!")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, data.get("message", "Failed to add comment."))
        except Exception as e:
            self.report({'ERROR'}, f"Connection error: {e}")
        return {'CANCELLED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "comment_text")



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
        if self.url:
            try:
                webbrowser.open(self.url)
                self.report({'INFO'}, f"Opening URL: {self.url}")
                return {'FINISHED'}
            except Exception as e:
                self.report({'ERROR'}, f"Failed to open URL: {e}")
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "No URL provided")
            return {'CANCELLED'}


class EXPLORATORY_UL_Comments(bpy.types.UIList):
    """
    A custom UIList to show the MyAddonSceneProps.comments collection in a scrollable list.
    """
    # The draw_item method is where you define how each row is displayed.
    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        """
        data:        Usually the 'owner' of the collection (scene.my_addon_data).
        item:        The element in the collection (a MyAddonComment).
        index:       The index of the item in the collection.
        layout:      The UI layout for this row.
        """
        # item is a MyAddonComment
        # We'll show a small label with author and text snippet

        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text=f"{item.author}: {item.text[:30]}")  # e.g. show 30 chars
        elif self.layout_type in {'GRID'}:
            # For GRID layout, just show something minimal
            layout.label(text=item.author)

