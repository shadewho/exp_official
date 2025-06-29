#Exploratory/Exp_UI/cache_system/preload.py
import os
import time
import bpy
from ..main_config import THUMBNAIL_CACHE_FOLDER
from .persistence import load_thumbnail_index, get_or_load_image, get_or_create_texture
# Define the JSON index file for thumbnail caching.
THUMBNAIL_INDEX_FILE = os.path.join(THUMBNAIL_CACHE_FOLDER, "thumbnail_index.json")

# A dictionary in Python that tracks loaded images in Blender.
LOADED_IMAGES = {}
LOADED_TEXTURES = {}

# Define a JSON index file for metadata caching.
METADATA_INDEX_FILE = os.path.join(THUMBNAIL_CACHE_FOLDER, "metadata_index.json")

# In-memory cache for package metadata.
METADATA_CACHE = {}

import time

_last_validation_time = time.time()

def preload_in_memory_thumbnails():
    """
    Preloads thumbnail images from the persistent disk cache (JSON index)
    into the in-memory cache (LOADED_IMAGES and LOADED_TEXTURES).
    This function uses bpy.app.timers to load one image per tick, so the UI remains responsive.
    """
    index_data = load_thumbnail_index()
    keys = list(index_data.keys())
    total = len(keys)

    def load_next_thumbnail():
        if not keys:
            return None  # Returning None stops the timer.
        key = keys.pop(0)
        entry = index_data.get(key)
        file_path = entry.get("file_path") if entry else None
        if file_path and os.path.exists(file_path):
            # Load the image and create its GPU texture.
            img = get_or_load_image(file_path)
            if img:
                get_or_create_texture(img)
        # Schedule next tick in 0.1 seconds (adjust as needed).
        return 0.1

    bpy.app.timers.register(load_next_thumbnail)


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
        from .persistence import load_thumbnail_index, save_thumbnail_index
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
        from .manager import cache_manager
        for package_id, metadata in list(cache_manager.metadata_cache.items()):
            metadata_time = metadata.get("last_access", 0)
            # If metadata is older than 1 day, refresh it.
            if current_time - metadata_time > 24 * 3600:
                from .download_helpers import background_fetch_metadata
                background_fetch_metadata(package_id)
        
        _last_validation_time = current_time

    # --- End merged cache validation logic ---
    
    # Return the interval (in seconds) for the next call.
    return 30.0  # This timer will run every 10 seconds.
