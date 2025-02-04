# image_button_UI/cache.py

import os
import json
import time
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from ..main_config import THUMBNAIL_CACHE_FOLDER

THUMBNAIL_INDEX_FILE = os.path.join(THUMBNAIL_CACHE_FOLDER, "thumbnail_index.json")

# A dictionary in Python that tracks loaded images in Blender
LOADED_IMAGES = {}
LOADED_TEXTURES = {}


# ------------------------------------------------------------------------------
# 1) JSON Index Logic
# ------------------------------------------------------------------------------

def load_thumbnail_index():
    """Load the thumbnail index from a JSON file."""
    if not os.path.exists(THUMBNAIL_INDEX_FILE):
        return {}
    try:
        with open(THUMBNAIL_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def save_thumbnail_index(index_data):
    """Save the thumbnail index to a JSON file."""
    try:
        with open(THUMBNAIL_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Could not save thumbnail index: {e}")


def register_thumbnail_in_index(url, local_path):
    """Add or update an entry in the thumbnail index for the given URL."""
    index_data = load_thumbnail_index()
    index_data[url] = {
        "file_path": local_path,
        "last_access": time.time()
    }
    save_thumbnail_index(index_data)


def update_thumbnail_access(url):
    """Bump 'last_access' for the given URL in the index."""
    index_data = load_thumbnail_index()
    if url in index_data:
        index_data[url]["last_access"] = time.time()
        save_thumbnail_index(index_data)


def get_cached_path_if_exists(url):
    """
    Return the local file_path if it exists in index & on disk.
    Otherwise, return None.
    """
    index_data = load_thumbnail_index()
    if url in index_data:
        local_path = index_data[url]["file_path"]
        if os.path.exists(local_path):
            # update last access
            update_thumbnail_access(url)
            return local_path
    return None


# ------------------------------------------------------------------------------
# 2) Blender GPU Loading
# ------------------------------------------------------------------------------

def get_or_load_image(image_path):
    """
    Return a bpy.types.Image if it's in LOADED_IMAGES, else load from disk once.
    This avoids repeated creation of image data blocks for the same file.
    """
    if image_path in LOADED_IMAGES:
        return LOADED_IMAGES[image_path]
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
    """Return a gpu.types.GPUTexture if it's in LOADED_TEXTURES, else create it once."""
    if not img:
        return None
    key = img.name
    if key in LOADED_TEXTURES:
        return LOADED_TEXTURES[key]

    tex = gpu.texture.from_image(img)
    LOADED_TEXTURES[key] = tex
    return tex


def clear_image_datablocks():
    """
    Remove references to all loaded images so Blender won't save them
    in the .blend file. Clears LOADED_IMAGES and LOADED_TEXTURES.
    Call this when you close your thumbnail UI.
    """
    for path, img in LOADED_IMAGES.items():
        if img and img.name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[img.name], do_unlink=True)

    LOADED_IMAGES.clear()
    LOADED_TEXTURES.clear()
