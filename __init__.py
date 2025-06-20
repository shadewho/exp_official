#Exploratory/__init__.py
bl_info = {
    "name": "Exploratory",
    "blender": (4, 4, 0),
    "author": "Spencer Shade",
    "doc_url": "https://exploratory.online/",
    "tracker_url": "https://discord.gg/GJpGAAvMBe",
    "description": "Explore and Create within Blender.",
    "version": (1, 0, 1),
}
import bpy

from .Exp_UI.prefs_persistence import (
    on_pref_update,
    on_keep_preferences_update,
    load_prefs_from_json
)

from .Exp_UI.auth import is_internet_available
from .Exp_UI.exp_api import update_latest_version_cache
from .update_addon import WEBAPP_OT_UpdateAddon, WEBAPP_OT_RefreshVersion
# preferences class + operators
from .exp_preferences import (
    ExploratoryAddonPreferences,
    EXPLORATORY_OT_SetKeybind,
    EXPLORATORY_OT_BuildCharacter
)

# submodule APIs
from . import Exp_Game, Exp_UI

def version_check_timer():
    # only hit the network if we have connectivity
    if is_internet_available():
        update_latest_version_cache()
    # return seconds until next call (900s = 15min)
    return 900.0

def register():
    # 0) inject JSON‐save callbacks onto ExploratoryAddonPreferences
    for name, rna_prop in ExploratoryAddonPreferences.bl_rna.properties.items():
        if name == "rna_type":
            continue
        cb = on_keep_preferences_update if name == "keep_preferences" else on_pref_update
        try:
            rna_prop.update = cb
        except (AttributeError, TypeError):
            pass

    # 1) register preferences & keybind/build operators FIRST
    bpy.utils.register_class(ExploratoryAddonPreferences)
    bpy.utils.register_class(EXPLORATORY_OT_SetKeybind)
    bpy.utils.register_class(EXPLORATORY_OT_BuildCharacter)
    bpy.utils.register_class(WEBAPP_OT_UpdateAddon)
    bpy.utils.register_class(WEBAPP_OT_RefreshVersion)

    # initial version check on startup
    if is_internet_available():
        update_latest_version_cache()
    bpy.app.timers.register(version_check_timer, first_interval=0.0)

    # 2) now register submodules (which include your panels)
    Exp_Game.register()
    Exp_UI.register()

    # 3) load whatever prefs the user last saved
    load_prefs_from_json()

    print("Exploratory Addon registered.")


def unregister():
    # save prefs before teardown
    from .Exp_UI.prefs_persistence import save_prefs_to_json
    save_prefs_to_json()

    # 1) unregister submodules first
    Exp_UI.unregister()
    Exp_Game.unregister()

    # 2) then unregister our own classes (in reverse registration order)
    bpy.utils.unregister_class(WEBAPP_OT_RefreshVersion)
    bpy.utils.unregister_class(WEBAPP_OT_UpdateAddon)
    bpy.utils.unregister_class(EXPLORATORY_OT_BuildCharacter)
    bpy.utils.unregister_class(EXPLORATORY_OT_SetKeybind)
    bpy.utils.unregister_class(ExploratoryAddonPreferences)

    # 3) finally cancel our version‐check timer
    try:
        bpy.app.timers.unregister(version_check_timer)
    except Exception:
        pass

    print("Exploratory Addon unregistered.")


if __name__ == "__main__":
    register()
