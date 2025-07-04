#Exploratory/Exp_UI/events/properties.py

import bpy
from bpy.props import EnumProperty, PointerProperty
from .utilities import get_event_items, update_event_stage


def register():
    # 1) Event-stage dropdown (must match API keys)
    bpy.types.Scene.event_stage = EnumProperty(
        name="Event Stage",
        description="Select the current stage for events",
        items=[
            ('submission', 'Submit',   'Submission phase'),
            ('voting',     'Vote',     'Voting phase'),
            ('winners',    'Winners',  'Completed events'),
        ],
        default='submission',
        update=update_event_stage
    )

    # 2) Fetched-events storage
    bpy.types.Scene.fetched_events = PointerProperty(
        name="Fetched Events",
        type=bpy.types.PropertyGroup
    )

    # 3) Selected-event dropdown
    bpy.types.Scene.selected_event = EnumProperty(
        name="Event",
        description="Select an event to filter packages",
        items=get_event_items
    )

def unregister():
    # reverse order
    del bpy.types.Scene.selected_event
    del bpy.types.Scene.fetched_events
    del bpy.types.Scene.event_stage