#Exploratory/Exp_UI/events/utilities.py

import requests
import bpy

from ..main_config import (
    EVENTS_ENDPOINT
)

# -------------------------------------------------------------------
# Filter Events
# -------------------------------------------------------------------
def fetch_events_by_stage():
    # Build the URL. (Adjust BASE_URL if necessary so that it points to your website's API.)
    url = EVENTS_ENDPOINT
    print(url)

    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("success"):
            return data  # Expected to have keys: 'submission', 'vote', 'winners'
        else:
            print("Event fetch error:", data.get("message"))
            return {}
    except Exception as e:
        print("Error fetching events by stage:", e)
        return {}
    
def update_event_stage(self, context):
    events = fetch_events_by_stage()
    context.scene["fetched_events_data"] = events

    stage = context.scene.event_stage  # will be 'submission', 'voting', or 'winners'
    stage_events = events.get(stage, [])
    if stage_events:
        context.scene.selected_event = str(stage_events[0]["id"])
    else:
        context.scene.selected_event = "0"
    
    if context.area:
        context.area.tag_redraw()




# Define the selected_event property that uses a dynamic items callback.
def get_event_items(self, context):
    events_data = context.scene.get("fetched_events_data", {})
    stage = context.scene.event_stage  # already 'submission', 'voting', or 'winners'
    items = []
    stage_events = events_data.get(stage, [])
    for event in stage_events:
        items.append((str(event["id"]), event["title"], event.get("description", "")))
    if not items:
        items = [("0", "No events", "No active event in this stage")]
    return items



bpy.types.Scene.selected_event = bpy.props.EnumProperty(
    name="Event",
    description="Select an event to filter packages",
    items=get_event_items
)