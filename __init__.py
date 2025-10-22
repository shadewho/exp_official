# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "Exploratory",
    "blender": (4, 5, 0),
    "author": "Spencer Shade",
    "doc_url": "https://exploratory.online/",
    "tracker_url": "https://discord.gg/GJpGAAvMBe",
    "description": "Explore and Create within Blender. Location: 3D Viewport N-panel",
    "version": (1, 0, 0),
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
from .build_character import EXPLORATORY_OT_BuildCharacter, EXPLORATORY_OT_BuildArmature

# submodule APIs
from . import Exp_Game, Exp_UI, Exp_Nodes

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
        EXPLORATORY_OT_BuildArmature,
        WEBAPP_OT_UpdateAddon,
        WEBAPP_OT_RefreshVersion,
    ):
        bpy.utils.register_class(cls)

    # 2) restore saved prefs right away + hot-reload persistence
    apply_prefs()
    register_prefs_handlers()

    # 3) version check + background timer
    if is_internet_available():
        update_latest_version_cache()
    bpy.app.timers.register(version_check_timer, first_interval=0.0)

    # 4) register submodules
    #    Order matters a bit: Game defines properties/types that node sockets may reference.
    Exp_Game.register()
    Exp_UI.register()
    Exp_Nodes.register()   # <-- your Nodes system is now part of main init


def unregister():
    import bpy

    # 1) save & teardown prefs persistence
    unregister_prefs_handlers()

    # 2) unregister submodules (reverse-ish order; remove Nodes before UI/Game to
    #    avoid dangling menu/categories in the Node Editor)
    try:
        # Remove load_post helper if present
        for h in list(bpy.app.handlers.load_post):
            # We registered this as a local function; remove by name match if needed
            if getattr(h, "__name__", "") == "_ensure_exploratory_nodes_tree":
                bpy.app.handlers.load_post.remove(h)
    except Exception:
        pass

    Exp_Nodes.unregister()
    Exp_UI.unregister()
    Exp_Game.unregister()

    # 3) unregister core classes (reverse order)
    for cls in reversed((
        WEBAPP_OT_RefreshVersion,
        WEBAPP_OT_UpdateAddon,
        EXPLORATORY_OT_BuildArmature,
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
