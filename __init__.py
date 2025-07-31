bl_info = {
    "name": "Exploratory",
    "blender": (4, 5, 0),
    "author": "Spencer Shade",
    "doc_url": "https://exploratory.online/",
    "tracker_url": "https://discord.gg/GJpGAAvMBe",
    "description": "Explore and Create within Blender.",
    "version": (1, 0, 1),
}

import bpy
from .Exp_UI.internet.helpers import is_internet_available
from .Exp_UI.version_info import update_latest_version_cache
from .update_addon import WEBAPP_OT_UpdateAddon, WEBAPP_OT_RefreshVersion

# persistence handlers
from .Exp_UI.prefs_persistence import (
    apply_prefs,
    register_prefs_handlers,
    unregister_prefs_handlers,
)

# your prefs + operators
from .exp_preferences import (
    ExploratoryAddonPreferences,
    EXPLORATORY_OT_SetKeybind,
)
from .build_character import EXPLORATORY_OT_BuildCharacter

# submodule APIs
from . import Exp_Game, Exp_UI

def version_check_timer():
    if is_internet_available():
        update_latest_version_cache()
    return 900.0  # run again in 15m

def register():
    # 1) register core classes
    for cls in (
        EXPLORATORY_OT_SetKeybind,
        ExploratoryAddonPreferences,
        EXPLORATORY_OT_BuildCharacter,
        WEBAPP_OT_UpdateAddon,
        WEBAPP_OT_RefreshVersion,
    ):
        bpy.utils.register_class(cls)

    # 2) restore saved prefs right away
    apply_prefs()
    #    and keep it hooked for hot-reloads
    register_prefs_handlers()

    # 3) version check
    if is_internet_available():
        update_latest_version_cache()
    bpy.app.timers.register(version_check_timer, first_interval=0.0)

    # 4) register your submodules (panels, operators, etc.)
    Exp_Game.register()
    Exp_UI.register()

def unregister():
    # 1) save & teardown prefs persistence
    unregister_prefs_handlers()

    # 2) unregister submodules first
    Exp_UI.unregister()
    Exp_Game.unregister()

    # 3) unregister core classes (reverse order)
    for cls in reversed((
        WEBAPP_OT_RefreshVersion,
        WEBAPP_OT_UpdateAddon,
        EXPLORATORY_OT_BuildCharacter,
        EXPLORATORY_OT_SetKeybind,
        ExploratoryAddonPreferences,
    )):
        bpy.utils.unregister_class(cls)

    # 4) stop version timer
    try:
        bpy.app.timers.unregister(version_check_timer)
    except Exception:
        pass

if __name__ == "__main__":
    register()
