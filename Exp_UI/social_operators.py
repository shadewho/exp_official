# social_operators.py

from .auth import load_token
from .main_config import LIKE_PACKAGE_ENDPOINT, COMMENT_PACKAGE_ENDPOINT, USAGE_ENDPOINT, EVENTS_URL
from .exp_api import like_package, comment_package
import bpy
import webbrowser
from bpy.props import StringProperty
import requests
from .interface.drawing.utilities import format_relative_time

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

        # Simply refresh the UI so the persistent popup remains visible.
        context.area.tag_redraw()
        return {'FINISHED'}

    def invoke(self, context, event):
        # Since we want the operator to run immediately, no popup is needed.
        return self.execute(context)



class COMMENT_PACKAGE_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.comment_package"
    bl_label = "Comment on Item"

    # Local property used when not in persistent mode
    comment_text: bpy.props.StringProperty(name="Comment", default="")
    skip_popup: bpy.props.BoolProperty(default=False)
    launched_from_persistent: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        # If launched from the persistent popup, use scene.comment_text if needed
        if self.launched_from_persistent:
            if not self.comment_text:
                self.comment_text = context.scene.comment_text

        if not self.comment_text.strip():
            self.report({'ERROR'}, "Comment cannot be empty")
            return {'CANCELLED'}

        file_id = context.scene.my_addon_data.file_id
        try:
            data = comment_package(file_id, self.comment_text)
            if data.get("success"):
                comment_data = data.get("comment", {})
                new_comment = context.scene.my_addon_data.comments.add()
                new_comment.author = comment_data.get("author", "Anonymous")
                new_comment.text = comment_data.get("content", self.comment_text)
                new_comment.timestamp = comment_data.get("timestamp", "")
                self.report({'INFO'}, "Comment added!")
                # If in persistent mode, clear the scene property so the text field resets.
                if self.launched_from_persistent:
                    context.scene.comment_text = ""
                else:
                    self.comment_text = ""
            else:
                self.report({'ERROR'}, data.get("message", "Failed to add comment."))
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Connection error: {e}")
            return {'CANCELLED'}

        context.area.tag_redraw()
        return {'FINISHED'}

    def invoke(self, context, event):
        if self.launched_from_persistent or self.skip_popup:
            # In persistent mode, execute immediately (using the scene property)
            return self.execute(context)
        else:
            # In other contexts, bring up the dialog for text input.
            return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        # Only draw the text field if NOT launched from persistent mode.
        if not self.launched_from_persistent:
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
        addon_data.downloads_used = data.get("downloads_used", 0)
        addon_data.downloads_limit = data.get("downloads_limit", 0)
        addon_data.uploads_used = data.get("uploads_used", 0)
        addon_data.username    = data.get("username", "")
        addon_data.profile_url = data.get("profile_url", "")

        self.report({'INFO'}, "Subscription usage refreshed")
        return {'FINISHED'}

class POPUP_SOCIAL_DETAILS_OT(bpy.types.Operator):
    bl_idname = "view3d.popup_social_details"
    bl_label = "World Social Details"

    def execute(self, context):
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=400)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        addon_data = scene.my_addon_data

        layout.label(text="World Review", icon='WORLD')

        if addon_data.file_id > 0:
            box_info = layout.box()
            box_info.label(text=f"Title: {addon_data.package_name}", icon='BOOKMARKS')
            box_info.label(text=f"Description: {addon_data.description}")
            box_info.label(text=f"Uploaded: {format_relative_time(addon_data.upload_date)} ago")
            box_info.label(text=f"Downloads: {addon_data.download_count}", icon='IMPORT') 
            row_author = box_info.row(align=True)
            row_author.label(text=f"Author: {addon_data.author}")
            if addon_data.profile_url:
                op = row_author.operator("webapp.open_url", text="Profile")
                op.url = addon_data.profile_url

            # Check if the package is an event and the event stage is 'voting'
            if scene.package_item_type == 'event' and scene.event_stage == 'voting':
                row_vote = layout.row(align=True)
                # Display the vote operator with a star icon; adjust text as needed.
                vote_op = row_vote.operator("webapp.vote_map", text="★ Vote")
                # You can set properties on vote_op if necessary:
                vote_op.skip_popup = True
            else:
                row_likes = layout.row(align=True)
                row_likes.label(text=f"{addon_data.likes} Likes", icon='FUND')
                op_like = row_likes.operator("webapp.like_package", text="Like")
                op_like.skip_popup = True
                op_like.launched_from_persistent = True

            layout.separator()
            layout.label(text="Comments:", icon='COMMUNITY')

            row_comments = layout.row()
            row_comments.template_list(
                "EXPLORATORY_UL_Comments", "", addon_data,
                "comments", addon_data, "active_comment_index", rows=5
            )

            layout.separator()
            # Inline text field for new comments.
            row = layout.row(align=True)
            row.prop(scene, "comment_text", text="", emboss=True)
            op_comment = row.operator("webapp.comment_package", text="", icon='ADD')
            op_comment.launched_from_persistent = True
            op_comment.skip_popup = True

        else:
            layout.label(text="No social data available.", icon='INFO')


class VOTE_MAP_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.vote_map"
    bl_label = "Vote for Map"
    bl_options = {'REGISTER'}
    
    skip_popup: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        scene = context.scene
        addon_data = scene.my_addon_data

        if scene.package_item_type != 'event' or scene.event_stage != 'voting':
            self.report({'ERROR'}, "Not in voting stage for an event map.")
            return {'CANCELLED'}

        submission_id = addon_data.event_submission_id
        if submission_id <= 0:
            self.report({'ERROR'}, "No valid submission ID found.")
            return {'CANCELLED'}

        token = load_token()
        if not token:
            self.report({'ERROR'}, "You must be logged in to vote.")
            return {'CANCELLED'}
        
        # Ensure header format is correct
        headers = {"Authorization": f"Bearer {token}"}
        
        vote_url = f"{EVENTS_URL}/vote/{submission_id}"
        
        try:
            response = requests.post(vote_url, headers=headers, timeout=5)

            response.raise_for_status()
            data = response.json()
            if not data.get("success"):
                self.report({'ERROR'}, data.get("message", "Vote failed."))
                return {'CANCELLED'}
            self.report({'INFO'}, data.get("message", "Vote cast successfully!"))
            return {'FINISHED'}
        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:
                try:
                    data = response.json()
                    self.report({'INFO'}, data.get("message", "Already voted."))
                except Exception:
                    self.report({'ERROR'}, "Bad Request")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, f"Error during vote: {e}")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error during vote: {e}")
            return {'CANCELLED'}
