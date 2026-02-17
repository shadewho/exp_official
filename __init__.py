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
    "blender": (5, 0, 0),
    "author": "Spencer Shade",
    "doc_url": "https://exploratory.online/",
    "tracker_url": "https://discord.gg/GJpGAAvMBe",
    "description": "Explore and Create within Blender. Location: 3D Viewport N-panel",
    "version": (1, 0, 0),
}

# ============================================================================
# BPY IMPORT GUARD — worker processes don't have bpy, so try/except is the
# most reliable detection regardless of what else is in sys.modules.
# ============================================================================
try:
    import bpy
    _HAS_BPY = True
except ImportError:
    _HAS_BPY = False

if _HAS_BPY:
    # ── Blender hot-reload support ─────────────────────────────────────
    # importlib.reload() on __init__.py does NOT reload submodules.
    # Without this block, Python keeps the OLD cached submodules and
    # imports from them — causing ImportError when names have changed.
    if "exp_preferences" in locals():
        import importlib
        from . import prefs_persistence;   importlib.reload(prefs_persistence)
        from . import exp_preferences;     importlib.reload(exp_preferences)
        from . import build_character;     importlib.reload(build_character)
        from . import Exp_Game;            importlib.reload(Exp_Game)
        from . import Exp_Nodes;           importlib.reload(Exp_Nodes)
        from . import dev_refresh;         importlib.reload(dev_refresh)

    # persistence handlers
    from .prefs_persistence import (
        apply_prefs,
        register_prefs_handlers,
        unregister_prefs_handlers,
    )

    # prefs + operators
    from .exp_preferences import (
        AssetPackEntry,
        EXPLORATORY_UL_AssetPackList,
        EXPLORATORY_OT_AddAssetPack,
        EXPLORATORY_OT_RemoveAssetPack,
        EXPLORATORY_OT_SetKeybind,
        ExploratoryAddonPreferences,
    )
    from .build_character import EXPLORATORY_OT_BuildCharacter, EXPLORATORY_OT_BuildArmature

    # submodule APIs
    from . import Exp_Game, Exp_Nodes, dev_refresh

    def register():
        # 1) register core classes (AssetPackEntry must come before
        #    ExploratoryAddonPreferences because it's used as a CollectionProperty type)
        for cls in (
            AssetPackEntry,
            EXPLORATORY_UL_AssetPackList,
            EXPLORATORY_OT_AddAssetPack,
            EXPLORATORY_OT_RemoveAssetPack,
            EXPLORATORY_OT_SetKeybind,
            ExploratoryAddonPreferences,
            EXPLORATORY_OT_BuildCharacter,
            EXPLORATORY_OT_BuildArmature,
        ):
            bpy.utils.register_class(cls)

        # 2) restore saved prefs right away + hot-reload persistence
        apply_prefs()
        register_prefs_handlers()

        # 3) register submodules
        #    Order matters a bit: Game defines properties/types that node sockets may reference.
        Exp_Game.register()
        Exp_Nodes.register()   # <-- your Nodes system is now part of main init
        dev_refresh.register()  # <-- development refresh panel (ENABLED flag controls visibility)


    def unregister():
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

        dev_refresh.unregister()
        Exp_Nodes.unregister()
        Exp_Game.unregister()

        # 3) unregister core classes (reverse order)
        for cls in reversed((
            AssetPackEntry,
            EXPLORATORY_UL_AssetPackList,
            EXPLORATORY_OT_AddAssetPack,
            EXPLORATORY_OT_RemoveAssetPack,
            EXPLORATORY_OT_SetKeybind,
            ExploratoryAddonPreferences,
            EXPLORATORY_OT_BuildCharacter,
            EXPLORATORY_OT_BuildArmature,
        )):
            bpy.utils.unregister_class(cls)

else:
    # Worker process — bpy not available, provide stubs
    def register():
        pass

    def unregister():
        pass

if __name__ == "__main__":
    register()
