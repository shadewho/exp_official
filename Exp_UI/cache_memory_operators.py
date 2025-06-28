# cache_memory_operators.py

import bpy
import os
import shutil
from .image_button_UI.cache import save_thumbnail_index
from .main_config import THUMBNAIL_CACHE_FOLDER
from .image_button_UI.cache import (
    clear_image_datablocks,
    save_thumbnail_index,
    get_cached_metadata
)
from .helper_functions import background_fetch_metadata
from .cache_manager import cache_manager, ensure_package_data
import time

_last_validation_time = time.time()


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


class PRELOAD_METADATA_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.preload_metadata"
    bl_label = "Preload Metadata"
    bl_description = (
        "Starts background threads to preload metadata (e.g. JSON and images) "
        "for all packages that aren’t already cached."
    )

    def execute(self, context):
        scene = context.scene

        # If there's no package data at all, prime the cache for the current type
        if not cache_manager.get_package_data():
            # PASS the file_type (and you can pass a second arg for limit if you like)
            if not ensure_package_data(scene.package_item_type):
                self.report(
                    {'WARNING'},
                    f"No package data available to preload for “{scene.package_item_type}”."
                )
                return {'CANCELLED'}

        total_preloaded = 0
        for page, packages in cache_manager.get_package_data().items():
            for pkg in packages:
                pkg_id = pkg.get("file_id")
                if pkg_id is None:
                    continue
                # Only spawn a background fetch if we don’t already have metadata
                if cache_manager.get_metadata(pkg_id) is None:
                    background_fetch_metadata(pkg_id)
                    total_preloaded += 1

        self.report({'INFO'}, f"Started preloading metadata for {total_preloaded} packages.")
        return {'FINISHED'}


def preload_metadata_timer():
    """
    Timer callback that calls the PRELOAD_METADATA_OT_WebApp operator to preload metadata.
    In addition, every hour, it validates and refreshes the persistent cache (thumbnails and metadata).
    """
    import time

    scene = bpy.context.scene
    # If game modal is active, skip preloading to avoid hitches.
    if scene.ui_current_mode == "GAME":
        return 10.0  # Delay next check, but do nothing

    try:
        bpy.ops.webapp.preload_metadata('INVOKE_DEFAULT')
    except Exception as e:
        print(f"[ERROR] Metadata preload timer: {e}")

    # --- Begin merged cache validation logic ---
    global _last_validation_time
    current_time = time.time()
    # Run full validation every hour (3600 seconds)
    if current_time - _last_validation_time > 3600:
        # Validate the persistent thumbnail JSON cache.
        from .image_button_UI.cache import load_thumbnail_index, save_thumbnail_index
        import os
        index_data = load_thumbnail_index()
        keys_to_remove = []
        for key, entry in index_data.items():
            file_path = entry.get("file_path")
            last_access = entry.get("last_access", 0)
            # Example rule: if the file is missing or hasn't been accessed in 7 days.
            if not file_path or not os.path.exists(file_path) or (current_time - last_access > 7 * 24 * 3600):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            index_data.pop(key, None)
        save_thumbnail_index(index_data)
        
        # Validate the in-memory metadata cache.
        from .cache_manager import cache_manager
        for package_id, metadata in list(cache_manager.metadata_cache.items()):
            metadata_time = metadata.get("last_access", 0)
            # If metadata is older than 1 day, refresh it.
            if current_time - metadata_time > 24 * 3600:
                from .helper_functions import background_fetch_metadata
                background_fetch_metadata(package_id)
        
        _last_validation_time = current_time

    # --- End merged cache validation logic ---
    
    # Return the interval (in seconds) for the next call.
    return 30.0  # This timer will run every 10 seconds.

