# image_button_UI/cache.py

import os
import json
import time
import bpy
import gpu
from ..main_config import THUMBNAIL_CACHE_FOLDER

# Define the JSON index file for thumbnail caching.
THUMBNAIL_INDEX_FILE = os.path.join(THUMBNAIL_CACHE_FOLDER, "thumbnail_index.json")

# A dictionary in Python that tracks loaded images in Blender.
LOADED_IMAGES = {}
LOADED_TEXTURES = {}

# Define a JSON index file for metadata caching.
METADATA_INDEX_FILE = os.path.join(THUMBNAIL_CACHE_FOLDER, "metadata_index.json")

# In-memory cache for package metadata.
METADATA_CACHE = {}

# ------------------------------------------------------------------------------
# 1) JSON Index Logic (Thumbnail Cache using file_id as key)
# ------------------------------------------------------------------------------

def load_thumbnail_index():
    """Load the thumbnail index from a JSON file."""
    if not os.path.exists(THUMBNAIL_INDEX_FILE):
        return {}
    try:
        with open(THUMBNAIL_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load thumbnail index: {e}")
        return {}

def save_thumbnail_index(index_data):
    """Save the thumbnail index to a JSON file."""
    try:
        with open(THUMBNAIL_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Could not save thumbnail index: {e}")

def register_thumbnail_in_index(key, local_path, thumbnail_url=None):
    """
    Update the JSON index with the given key, local thumbnail path,
    and optionally the thumbnail_url.
    """
    index_data = load_thumbnail_index()
    index_data[str(key)] = {
        "file_path": local_path,
        "last_access": time.time(),
        "thumbnail_url": thumbnail_url
    }
    save_thumbnail_index(index_data)

def update_thumbnail_access(file_id):
    """Bump 'last_access' for the given file_id in the index."""
    index_data = load_thumbnail_index()
    key = str(file_id)
    if key in index_data:
        index_data[key]["last_access"] = time.time()
        save_thumbnail_index(index_data)

def get_cached_path_if_exists(key):
    """
    Return the cached thumbnail file path if it exists in the index and on disk.
    Otherwise, return None.
    """
    index_data = load_thumbnail_index()
    key = str(key)
    if key in index_data:
        local_path = index_data[key]["file_path"]
        if os.path.exists(local_path):
            update_thumbnail_access(key)
            return local_path
    return None

# ------------------------------------------------------------------------------
# 2) Blender GPU Loading
# ------------------------------------------------------------------------------

def get_or_load_image(image_path):
    """
    Returns a bpy.types.Image corresponding to the given file path.
    If the image is cached and still valid, returns it.
    Otherwise, loads it from disk and caches it.
    """
    if image_path in LOADED_IMAGES:
        cached_img = LOADED_IMAGES[image_path]
        try:
            # Try to access a property (like name) to ensure the image is still valid.
            _ = cached_img.name
        except ReferenceError:
            # The cached image reference is stale. Remove it from our cache.
            print(f"[INFO] Cached image for {image_path} is stale. Reloading from disk.")
            del LOADED_IMAGES[image_path]
        else:
            return cached_img

    if not os.path.exists(image_path):
        print(f"[ERROR] get_or_load_image: File not found: {image_path}")
        return None

    try:
        img = bpy.data.images.load(image_path)
        LOADED_IMAGES[image_path] = img
        return img
    except RuntimeError:
        print(f"[ERROR] Failed to load image: {image_path}")
        return None


def get_or_create_texture(img):
    """
    Returns a GPU texture for the given image.
    If the image is not valid, returns None.
    """
    if not img:
        return None
    try:
        key = img.name 
    except ReferenceError:
        print("[ERROR] Tried to access img.name but the image has been removed.")
        return None

    if key in LOADED_TEXTURES:
        return LOADED_TEXTURES[key]

    try:
        tex = gpu.texture.from_image(img)
    except ReferenceError:
        print("[ERROR] Failed to create texture because the image is no longer valid.")
        return None

    LOADED_TEXTURES[key] = tex
    return tex

def clear_image_datablocks():
    """
    Clears out our cached images and GPU textures.
    Iterates over LOADED_IMAGES, removes each image from bpy.data.images (if still present),
    and then clears both caches.
    This function should be called when a new blend file is loaded or when you need to refresh the UI.
    """
    global LOADED_IMAGES, LOADED_TEXTURES
    # Iterate over a copy of the items to avoid modifying the dictionary during iteration.
    for path, img in list(LOADED_IMAGES.items()):
        if img and img.name in bpy.data.images:
            try:
                bpy.data.images.remove(bpy.data.images[img.name], do_unlink=True)
                print(f"[INFO] Removed image: {img.name}")
            except Exception as e:
                print(f"[ERROR] Could not remove image {img.name}: {e}")
    # Clear the caches.
    LOADED_IMAGES.clear()
    LOADED_TEXTURES.clear()

# ------------------------------------------------------------------------------
# 3) Metadata Cache Setup
# ------------------------------------------------------------------------------

def load_metadata_index():
    """Load the metadata cache index from disk."""
    if not os.path.exists(METADATA_INDEX_FILE):
        return {}
    try:
        with open(METADATA_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load metadata index: {e}")
        return {}

def save_metadata_index(index):
    """Save the metadata cache index to disk."""
    try:
        with open(METADATA_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Could not save metadata index: {e}")

def register_metadata_in_index(package_id, metadata):
    """
    Save the metadata for a given package ID into the in-memory cache
    and persist it on disk via the JSON index.
    """
    METADATA_CACHE[package_id] = metadata
    index = load_metadata_index()
    index[str(package_id)] = {
        "metadata": metadata,
        "last_access": time.time()
    }
    save_metadata_index(index)

def update_metadata_access(package_id):
    """Update the last access time for the given package in the metadata index."""
    index = load_metadata_index()
    key = str(package_id)
    if key in index:
        index[key]["last_access"] = time.time()
        save_metadata_index(index)

def get_cached_metadata(package_id):
    """
    Check first in the in-memory cache, then on disk.
    Returns the metadata if found; otherwise, returns None.
    """
    if package_id in METADATA_CACHE:
        update_metadata_access(package_id)
        return METADATA_CACHE[package_id]
    
    index = load_metadata_index()
    entry = index.get(str(package_id))
    if entry:
        update_metadata_access(package_id)
        # Optionally, load it into memory:
        METADATA_CACHE[package_id] = entry.get("metadata")
        return entry.get("metadata")
    return None

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