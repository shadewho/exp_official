#Exploratory/Exp_UI/events/operators.py

from ..auth.helpers import load_token
from ..main_config import EVENTS_URL
import bpy
import requests


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