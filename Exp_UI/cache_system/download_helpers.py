import os
import json
import requests
import threading
from urllib.parse import urlparse

from bpy.app.handlers import persistent
from ..main_config import THUMBNAIL_CACHE_FOLDER, PACKAGE_DETAILS_ENDPOINT
from ..auth.helpers import load_token

# ---- bring in your SQLite-backed cache API ----
from .db import (
    get_thumbnail_path,
    bump_thumbnail_access,
    register_thumbnail,
    get_metadata,
    bump_metadata_access,
    register_metadata,
)

# ----------------------------------------------------------------------------------
# Thumbnail Download Helper
# ----------------------------------------------------------------------------------

def download_thumbnail(url, file_id=None):
    """
    Download (or return cached) thumbnail, persisting to SQLite instead of JSON.
    """
    # Key by file_id when present, else fallback to URL string.
    key = file_id if file_id is not None else url

    # 1) Try to load from DB
    cached = get_thumbnail_path(key)
    if cached and os.path.exists(cached):
        bump_thumbnail_access(key)
        return cached

    # 2) Not in cache, fetch from the web
    parsed = urlparse(url)
    base_name = os.path.basename(parsed.path) or "thumbnail.png"
    _, ext = os.path.splitext(base_name)
    if ext.lower() not in (".png", ".jpg", ".jpeg", ".gif", ".bmp"):
        ext = ".png"
    local_name = base_name if base_name.endswith(ext) else base_name + ext
    local_path = os.path.join(THUMBNAIL_CACHE_FOLDER, local_name)

    try:
        resp = requests.get(url, stream=True, timeout=10)
        resp.raise_for_status()
        os.makedirs(THUMBNAIL_CACHE_FOLDER, exist_ok=True)
        with open(local_path, "wb") as out:
            for chunk in resp.iter_content(8192):
                if chunk:
                    out.write(chunk)
    except Exception:
        return None

    # 3) Persist into DB
    register_thumbnail(key, local_path, thumbnail_url=url)
    return local_path

# -------------------------------------------------------------------
# Background Threading for Metadata Fetching
# -------------------------------------------------------------------

def background_fetch_metadata(package_id):
    """
    Kick off a thread to fetch metadata only if it's not already in the DB.
    """
    existing = get_metadata(package_id)
    if existing is not None:
        # Already have it in SQLite
        bump_metadata_access(package_id)
        print(f"[INFO] Metadata for {package_id} from DB, skipping fetch.")
        return

    thread = threading.Thread(
        target=_fetch_metadata_worker,
        args=(package_id,),
        daemon=True,
    )
    thread.start()


def _fetch_metadata_worker(package_id):
    token = load_token()
    if not token:
        print("[ERROR] Cannot fetch metadata; not logged in.")
        return

    headers = {"Authorization": f"Bearer {token}"}
    url = f"{PACKAGE_DETAILS_ENDPOINT}/{package_id}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ERROR] Metadata request failed: {e}")
        return

    if not data.get("success"):
        print(f"[ERROR] Server error fetching metadata: {data.get('message')}")
        return

    # Attach local thumbnail if present
    thumb_url = data.get("thumbnail_url")
    if thumb_url:
        local = download_thumbnail(thumb_url, file_id=package_id)
        data["local_thumb_path"] = local
    else:
        data["local_thumb_path"] = None

    # Serialize and register into SQLite
    metadata_json = json.dumps(data)
    register_metadata(package_id, metadata_json)
