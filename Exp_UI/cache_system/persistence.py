# File: Exploratory/Exp_UI/cache_system/persistence.py

import os
import json
import time
import threading
import bpy
import gpu

from .db import (
    register_thumbnail,
    get_thumbnail_path,
    bump_thumbnail_access,
    register_metadata,
    get_metadata,
    bump_metadata_access,
)
from ..main_config import THUMBNAIL_CACHE_FOLDER
import logging
logger = logging.getLogger("Exploratory.cache")

# In-Blender caches for GPU assets and metadata objects
LOADED_IMAGES   = {}
LOADED_TEXTURES = {}
METADATA_CACHE  = {}

_lock = threading.Lock()

# ------------------------------------------------------------------
# 1) Thumbnail Index Logic (via SQLite)
# ------------------------------------------------------------------

def register_thumbnail_in_index(file_id: int, local_path: str, thumbnail_url: str=None):
    """
    Store or update a thumbnail record in SQLite.
    """
    register_thumbnail(file_id, local_path, thumbnail_url)


def get_cached_path_if_exists(file_id: int) -> str | None:
    """
    Return the local thumbnail path if it exists on disk and in the DB,
    bump its access time, or return None.
    """
    path = get_thumbnail_path(file_id)
    if path and os.path.exists(path):
        bump_thumbnail_access(file_id)
        return path
    return None


def update_thumbnail_access(file_id: int):
    """
    Alias for bump_thumbnail_access, so imports work.
    """
    bump_thumbnail_access(file_id)


# ------------------------------------------------------------------
# 2) Blender GPU Loading (unchanged)
# ------------------------------------------------------------------

def get_or_load_image(image_path):
    """
    Returns a bpy.types.Image for image_path, caching it in LOADED_IMAGES.
    """
    if image_path in LOADED_IMAGES:
        img = LOADED_IMAGES[image_path]
        try:
            _ = img.name
        except ReferenceError:
            del LOADED_IMAGES[image_path]
        else:
            return img

    if not os.path.exists(image_path):
        return None

    try:
        img = bpy.data.images.load(image_path)
        LOADED_IMAGES[image_path] = img
        return img
    except RuntimeError:
        return None


def get_or_create_texture(img):
    """
    Returns a GPU texture for img, caching it in LOADED_TEXTURES.
    """
    if not img:
        return None
    try:
        key = img.name
    except ReferenceError:
        return None

    if key in LOADED_TEXTURES:
        return LOADED_TEXTURES[key]

    try:
        tex = gpu.texture.from_image(img)
        LOADED_TEXTURES[key] = tex
        return tex
    except Exception:
        return None


def clear_image_datablocks():
    """
    Remove every image GPU resource we ever loaded and
    empty the in‑memory lookup tables – **without** raising
    ReferenceError even if the datablock vanished when the
    new .blend was opened.
    """
    global LOADED_IMAGES, LOADED_TEXTURES

    for path, img in list(LOADED_IMAGES.items()):
        try:
            # Accessing img.name fails if the datablock is already gone.
            img_name = img.name
        except ReferenceError:
            # Datablock died during the file switch – just forget it.
            pass
        else:
            # Datablock is still alive → remove it cleanly
            if img_name in bpy.data.images:
                try:
                    bpy.data.images.remove(bpy.data.images[img_name],
                                           do_unlink=True)
                except Exception:
                    pass

    LOADED_IMAGES.clear()
    LOADED_TEXTURES.clear()


# ------------------------------------------------------------------
# 3) Metadata Index Logic (via SQLite)
# ------------------------------------------------------------------

def register_metadata_in_index(package_id: int, metadata: dict):
    """
    Cache metadata in memory and persist as JSON in SQLite.
    """
    METADATA_CACHE[package_id] = metadata
    register_metadata(package_id, json.dumps(metadata))


def update_metadata_access(package_id: int):
    """
    Bump the metadata last_access in SQLite.
    """
    bump_metadata_access(package_id)


def get_cached_metadata(package_id: int) -> dict | None:
    """
    Return the metadata dict if in-memory or in SQLite, else None.
    """
    # In-memory first
    if package_id in METADATA_CACHE:
        bump_metadata_access(package_id)
        return METADATA_CACHE[package_id]

    # Then DB
    data_json = get_metadata(package_id)
    if data_json:
        metadata = json.loads(data_json)
        METADATA_CACHE[package_id] = metadata
        bump_metadata_access(package_id)
        return metadata

    return None

# Initialize the DB immediately when this module is imported
from .db import init_db
init_db()
