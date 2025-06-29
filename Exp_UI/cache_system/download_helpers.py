#Exploratory/Exp_UI/cache_system/download_helpers.py

import os
import requests
import threading
from urllib.parse import urlparse

from ..main_config import ( THUMBNAIL_CACHE_FOLDER, PACKAGE_DETAILS_ENDPOINT)

from .persistence import (
    get_cached_path_if_exists,
    register_thumbnail_in_index,
    register_metadata_in_index, get_cached_metadata
)
from ..auth.helpers import load_token
from bpy.app.handlers import persistent


# ----------------------------------------------------------------------------------
# Thumbnail Download Helper
# ----------------------------------------------------------------------------------

def download_thumbnail(url, file_id=None):
    """
    Download the thumbnail on the main thread, store it in THUMBNAIL_CACHE_FOLDER,
    register it in the JSON index, and return the local path.
    If file_id is provided, it will be used as the cache key.
    """
    # Use file_id as key if provided; otherwise fallback to URL.
    key = file_id if file_id is not None else url
    cached_path = get_cached_path_if_exists(key)
    if cached_path:
        return cached_path

    parsed = urlparse(url)
    base_name = os.path.basename(parsed.path) or "unknown_thumbnail.png"
    _, ext = os.path.splitext(base_name)
    if ext.lower() not in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
        ext = ".png"
    local_filename = base_name
    local_path = os.path.join(THUMBNAIL_CACHE_FOLDER, local_filename)

    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            return None

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        # Register the thumbnail in the JSON cache using the chosen key.
        register_thumbnail_in_index(key, local_path, thumbnail_url=url)
        return local_path
    except Exception as e:
        return None

# -------------------------------------------------------------------
# Background Threading for Metadata Fetching
# -------------------------------------------------------------------

def background_fetch_metadata(package_id):
    if get_cached_metadata(package_id) is not None:
        print(f"[INFO] Metadata for package {package_id} already cached.")
        return  # Already cached, so skip lazy load.
    # Otherwise, start background loading.
    thread = threading.Thread(target=_fetch_metadata_worker, args=(package_id,), daemon=True)
    thread.start()

def _fetch_metadata_worker(package_id):
    token = load_token()
    if not token:
        print("[ERROR] Cannot fetch metadata; not logged in.")
        return

    headers = {"Authorization": f"Bearer {token}"}
    url = f"{PACKAGE_DETAILS_ENDPOINT}/{package_id}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            metadata = data  # Adjust this if your JSON structure differs.
            # --- Integrate image download ---
            thumb_url = metadata.get("thumbnail_url")
            if thumb_url:
                local_thumb_path = download_thumbnail(thumb_url)
                metadata["local_thumb_path"] = local_thumb_path
            else:
                metadata["local_thumb_path"] = None

            register_metadata_in_index(package_id, metadata)
        else:
            print(f"[ERROR] Failed to fetch metadata for package {package_id}: {data.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"[ERROR] Exception while fetching metadata for package {package_id}: {e}")