bl_info = {
    "name": "Exploratory",
    "blender": (4, 2, 0),
    "author": "Spencer Shade",
    "doc_url": "https://exploratory.online/",
    "tracker_url": "https://discord.gg/GJpGAAvMBe",
    "description": "Explore and Create within Blender.",
    "version": (1, 0, 2),
}

import bpy

# Import the submodules
from . import Exp_Game
from . import Exp_UI
# from . import Exp_Nodes
from .exp_preferences import (
    ExploratoryAddonPreferences,
    EXPLORATORY_OT_SetKeybind,
    EXPLORATORY_OT_BuildCharacter
)

def register():
    Exp_Game.register()
    Exp_UI.register()
    # Exp_Nodes.register()
    bpy.utils.register_class(ExploratoryAddonPreferences)
    bpy.utils.register_class(EXPLORATORY_OT_SetKeybind)
    bpy.utils.register_class(EXPLORATORY_OT_BuildCharacter)
    
    print("Exploratory Addon registered.")

def unregister():
    bpy.utils.unregister_class(EXPLORATORY_OT_BuildCharacter)
    bpy.utils.unregister_class(EXPLORATORY_OT_SetKeybind)
    bpy.utils.unregister_class(ExploratoryAddonPreferences)
    # Exp_Nodes.unregister()
    Exp_UI.unregister()
    Exp_Game.unregister()
    
    print("Exploratory Addon unregistered.")

if __name__ == "__main__":
    register()
