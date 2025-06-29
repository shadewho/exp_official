# Exploratory/Exp_UI/interface/properties.py

import bpy
from bpy.props import EnumProperty, BoolProperty, StringProperty

def register():
    # Re-register the five UI-related scene properties directly on Scene

    bpy.types.Scene.ui_current_mode = EnumProperty(
        name="UI Mode",
        description="Which UI mode is shown",
        items=[
            ("BROWSE", "Browse", ""),
            ("DETAIL", "Detail", ""),
            ("LOADING", "Loading", ""),
            ("GAME",   "Game",    ""),
        ],
        default="BROWSE",
    )

    bpy.types.Scene.show_image_buttons = BoolProperty(
        name="Show Thumbnails",
        description="Toggle thumbnail display",
        default=False,
    )

    bpy.types.Scene.selected_thumbnail = StringProperty(
        name="Selected Thumbnail",
        description="Path to the current thumbnail image",
        default="",
    )

    bpy.types.Scene.show_loading = BoolProperty(
        name="Show Loading",
        description="Display loading spinner",
        default=False,
    )

    bpy.types.Scene.show_loading_image = BoolProperty(
        name="Show Loading Image",
        description="Display loading indicator",
        default=False,
    )

def unregister():
    # Remove those five properties in reverse order

    del bpy.types.Scene.show_loading_image
    del bpy.types.Scene.show_loading
    del bpy.types.Scene.selected_thumbnail
    del bpy.types.Scene.show_image_buttons
    del bpy.types.Scene.ui_current_mode
