# cache_memory_operators.py

import bpy
import os
import shutil
from .image_button_UI.cache import save_thumbnail_index
from .exp_api import fetch_packages
from .image_button_UI.drawing import load_image_buttons
from .main_config import THUMBNAIL_CACHE_FOLDER
from .image_button_UI.cache import (
    clear_image_datablocks,
    save_thumbnail_index
)
from .helper_functions import download_thumbnail

class CLEAR_ALL_DATA_OT_WebApp(bpy.types.Operator):
    """
    Completely resets your add-on data and UI:
      - Cancels background fetch (if any)
      - Clears Scene-based caches (fetched_packages_data, etc.)
      - Removes appended Scenes/Objects if desired
      - Clears thumbnail cache on disk & in memory
      - Removes your custom UI operator, so it no longer draws
    """
    bl_idname = "webapp.clear_all_data"
    bl_label = "Clear All Loaded Data"
    bl_options = {'REGISTER'}

    def execute(self, context):
        # --------------------------------------------------------------
        # 1) Stop or signal any background threads/queues if you have them
        #    e.g. set a cancel flag, or join threads. Pseudocode:
        #
        # if my_fetch_thread and my_fetch_thread.is_alive():
        #     my_fetch_thread_cancel_flag = True
        #     my_fetch_thread.join()
        #
        # Also clear the queue if you’re using one:
        # with fetch_page_queue.mutex:
        #     fetch_page_queue.queue.clear()

        # --------------------------------------------------------------
        # 2) Clear Python-side data in Scene
        #    (names may vary in your add-on)
        # --------------------------------------------------------------
        if hasattr(bpy.types.Scene, "fetched_packages_data"):
            bpy.types.Scene.fetched_packages_data.clear()

        if hasattr(bpy.types.Scene, "all_pages_data"):
            bpy.types.Scene.all_pages_data.clear()

        # Reset page counters, filters, search queries, etc.
        if hasattr(context.scene, "current_thumbnail_page"):
            context.scene.current_thumbnail_page = 1
        if hasattr(context.scene, "total_thumbnail_pages"):
            context.scene.total_thumbnail_pages = 1
        if hasattr(context.scene, "package_search_query"):
            context.scene.package_search_query = ""

        # If you’re using a PropertyGroup (my_addon_data) for extra fields:
        if hasattr(context.scene, "my_addon_data"):
            context.scene.my_addon_data.is_from_webapp = False
            context.scene.my_addon_data.file_id = 0
            context.scene.my_addon_data.comments.clear()
            # ... any other resets as needed

        # --------------------------------------------------------------
        # 3) Remove appended scenes or objects if you only wanted them temporarily
        # --------------------------------------------------------------
        appended_scene = bpy.data.scenes.get("Appended_Scene")
        if appended_scene:
            bpy.data.scenes.remove(appended_scene)

        # Similarly remove any appended collections, objects, materials, etc.
        # appended_coll = bpy.data.collections.get("MyAppendedCollection")
        # if appended_coll:
        #     bpy.data.collections.remove(appended_coll)

        # --------------------------------------------------------------
        # 4) Clear thumbnail cache folder on disk + JSON index
        # --------------------------------------------------------------
        if os.path.exists(THUMBNAIL_CACHE_FOLDER):
            shutil.rmtree(THUMBNAIL_CACHE_FOLDER)
        os.makedirs(THUMBNAIL_CACHE_FOLDER, exist_ok=True)

        # Reset your thumbnail index so it doesn’t try to reuse stale paths
        save_thumbnail_index({})

        # --------------------------------------------------------------
        # 5) Clear loaded GPU images from Blender’s memory
        #    (Removes from bpy.data.images, LOADED_IMAGES, LOADED_TEXTURES, etc.)
        # --------------------------------------------------------------
        clear_image_datablocks()

        # --------------------------------------------------------------
        # 6) Remove / refresh your custom UI
        #    If your "thumbnail UI" operator is still running, remove it.
        # --------------------------------------------------------------
        result = bpy.ops.view3d.remove_package_display('EXEC_DEFAULT')
        # If result == {'FINISHED'}, your UI is now removed.

        # If you want to immediately show a *new* blank UI, you could do:
        # bpy.ops.view3d.add_package_display('INVOKE_DEFAULT')

        self.report({'INFO'}, "All data and UI cleared successfully.")
        return {'FINISHED'}


class CLEAR_THUMBNAILS_ONLY_OT_WebApp(bpy.types.Operator):
    """
    Example partial clear: remove just thumbnail images from memory and disk.
    Useful if you want to keep other data around (like appended scenes).
    """
    bl_idname = "webapp.clear_thumbnails_only"
    bl_label = "Clear Thumbnails Only"
    bl_options = {'REGISTER'}

    def execute(self, context):
        # Delete thumbnail folder
        if os.path.exists(THUMBNAIL_CACHE_FOLDER):
            shutil.rmtree(THUMBNAIL_CACHE_FOLDER)
        os.makedirs(THUMBNAIL_CACHE_FOLDER, exist_ok=True)
        save_thumbnail_index({})

        # Remove images from bpy.data
        clear_image_datablocks()

        # Possibly also rebuild or remove UI to reflect the missing images:
        # data = getattr(bpy.types.Scene, "gpu_image_buttons_data", None)
        # if data:
        #     data.clear()
        #     # Optionally rebuild with load_image_buttons()

        self.report({'INFO'}, "Thumbnails cleared from disk and memory.")
        return {'FINISHED'}


# cache_memory_operators.py (or a new file, or any place you like)

class REFRESH_FILTERS_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.refresh_filters"
    bl_label = "Refresh Filters (Threaded)"
    
    def execute(self, context):
        scene = context.scene
        
        # 1) Clear old data/caches, remove old thumbnails
        if hasattr(bpy.types.Scene, "fetched_packages_data"):
            bpy.types.Scene.fetched_packages_data.clear()
        if hasattr(bpy.types.Scene, "all_pages_data"):
            bpy.types.Scene.all_pages_data.clear()
        # clear on-disk thumbnails and GPU images if you like…

        # 2) Start fresh on page 1, show spinner
        scene.current_thumbnail_page = 1
        scene.show_loading_image = True
        
        # 3) Use your threaded fetch operator 
        #    so we get progressive loading, page count, arrow logic, etc.
        bpy.ops.webapp.fetch_page('EXEC_DEFAULT', page_number=1)
        
        self.report({'INFO'}, "Threaded refresh started.")
        return {'FINISHED'}
