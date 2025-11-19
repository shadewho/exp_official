# Exploratory/Exp_UI/interface/properties.py

import bpy
from bpy.props import EnumProperty, BoolProperty, StringProperty

def register():
    bpy.types.Scene.ui_current_mode = EnumProperty(
        name="UI Mode",
        description="Which UI mode is shown",
        items=[
            ("DETAIL", "Detail", ""),
            ("LOADING", "Loading", ""),
            ("GAME",    "Game",    ""),
        ],
        default="GAME",
    )

    # Path to currently loaded thumbnail (downloaded once per code)
    bpy.types.Scene.selected_thumbnail = StringProperty(
        name="Selected Thumbnail",
        description="Path to the current thumbnail image",
        default="",
    )

    # Simple loading flags for overlay/animation
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

    # The only “search” left: a download code from your web app
    bpy.types.Scene.download_code = StringProperty(
        name="Download Code",
        description="Paste a code from the web app",
        default="",
    )

def unregister():
    del bpy.types.Scene.download_code
    del bpy.types.Scene.show_loading_image
    del bpy.types.Scene.show_loading
    del bpy.types.Scene.selected_thumbnail
    del bpy.types.Scene.ui_current_mode
