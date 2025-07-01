# Exploratory/Exp_UI/cache_system/operators/clear.py

import os
import shutil
import bpy

from ...main_config import THUMBNAIL_CACHE_FOLDER
from ..persistence import clear_image_datablocks
from ..db import drop_all_tables, init_db, evict_all_thumbnails


class CLEAR_ALL_DATA_OT_WebApp(bpy.types.Operator):
    """
    Completely resets your add-on data and UI:
      - Cancels background fetch (if any)
      - Clears Scene-based caches (fetched_packages_data, etc.)
      - Removes appended Scenes/Objects if desired
      - Clears the entire SQLite cache (thumbnails & metadata)
      - Clears on-disk thumbnails + Blender GPU images
      - Removes your custom UI operator so it no longer draws
    """
    bl_idname = "webapp.clear_all_data"
    bl_label = "Clear All Loaded Data"
    bl_options = {'REGISTER'}

    def execute(self, context):
        # 1) Stop or signal any background threads/queues if you have them
        #    (not shown here—keep your existing logic)

        # 2) Clear Python-side Scene data
        if hasattr(bpy.types.Scene, "fetched_packages_data"):
            bpy.types.Scene.fetched_packages_data.clear()
        if hasattr(bpy.types.Scene, "all_pages_data"):
            bpy.types.Scene.all_pages_data.clear()

        scene = context.scene
        scene.current_thumbnail_page = 1
        scene.total_thumbnail_pages   = 1
        scene.package_search_query    = ""

        if hasattr(scene, "my_addon_data"):
            scene.my_addon_data.is_from_webapp = False
            scene.my_addon_data.file_id        = 0
            scene.my_addon_data.comments.clear()

        # 3) Remove any appended scenes/collections/etc.
        appended = bpy.data.scenes.get("Appended_Scene")
        if appended:
            bpy.data.scenes.remove(appended)

        # 4) Wipe the entire SQLite cache and re-create tables
        drop_all_tables()
        init_db()

        # 5) Also clear out any on-disk thumbnails
        if os.path.exists(THUMBNAIL_CACHE_FOLDER):
            shutil.rmtree(THUMBNAIL_CACHE_FOLDER)
        os.makedirs(THUMBNAIL_CACHE_FOLDER, exist_ok=True)

        # 6) Remove GPU images/textures from Blender
        clear_image_datablocks()

        # 7) Tear down the UI if it’s still running
        bpy.ops.view3d.remove_package_display('EXEC_DEFAULT')

        self.report({'INFO'}, "All data and UI cleared successfully.")
        return {'FINISHED'}


class CLEAR_THUMBNAILS_ONLY_OT_WebApp(bpy.types.Operator):
    """
    Partial clear: remove just thumbnail images (DB entries + files) and GPU memory.
    Leaves metadata and package_data untouched.
    """
    bl_idname = "webapp.clear_thumbnails_only"
    bl_label = "Clear Thumbnails Only"
    bl_options = {'REGISTER'}

    def execute(self, context):
        # 1) Remove thumbnail records from SQLite
        evict_all_thumbnails()

        # 2) Wipe on-disk thumbnail files
        if os.path.exists(THUMBNAIL_CACHE_FOLDER):
            shutil.rmtree(THUMBNAIL_CACHE_FOLDER)
        os.makedirs(THUMBNAIL_CACHE_FOLDER, exist_ok=True)

        # 3) Clear GPU images/textures
        clear_image_datablocks()

        self.report({'INFO'}, "Thumbnails cleared from disk, DB, and memory.")
        return {'FINISHED'}
