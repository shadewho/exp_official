# helper_functions.py

import os
import requests
import bpy
import uuid
import traceback
import threading
from urllib.parse import urlparse

from .main_config import (
    DOWNLOAD_ENDPOINT, VALIDATE_TOKEN_ENDPOINT, ADDON_FOLDER,
    WORLD_DOWNLOADS_FOLDER, SHOP_DOWNLOADS_FOLDER,
    TOKEN_FILE, THUMBNAIL_CACHE_FOLDER, PACKAGE_DETAILS_ENDPOINT
)

from .image_button_UI.cache import (
    get_cached_path_if_exists,
    register_thumbnail_in_index, load_token,
    register_metadata_in_index, get_cached_metadata
)
from bpy.app.handlers import persistent
from datetime import datetime, timezone
from .cache_manager import filter_cached_data


# ----------------------------------------------------------------------------------
# Download Helper
# ----------------------------------------------------------------------------------

def download_blend_file(url):
    """
    Downloads the .blend file from the given `url` into the addon directory's
    "World Downloads" folder. Returns the local file path if successful.
    Ensures the file is fully written before returning.
    """
    # Extract the filename from the URL, removing query parameters
    base_filename = os.path.basename(url.split('?')[0])
    if not base_filename.endswith('.blend'):
        base_filename += '.blend'
    
    # Create a unique filename to prevent collisions
    unique_id = uuid.uuid4().hex  # Generate a unique identifier
    unique_filename = f"{os.path.splitext(base_filename)[0]}_{unique_id}.blend"
    local_path = os.path.join(WORLD_DOWNLOADS_FOLDER, unique_filename)
    
    print(f"[INFO] Downloading .blend file to: {local_path}")

    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"[ERROR] Failed to download file. HTTP {response.status_code}")
            return None

        # Write the file to disk
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    file.write(chunk)
            file.flush()  # Ensure data is written
            os.fsync(file.fileno())  # Physically ensure it's on disk
        print("[INFO] Downloaded .blend file successfully.")

        # Verify file size
        if os.path.getsize(local_path) == 0:
            print(f"[ERROR] Downloaded file is empty: {local_path}")
            os.remove(local_path)
            return None

        return local_path

    except Exception as e:
        print(f"[ERROR] Exception during download: {e}")
        traceback.print_exc()
        return None


# ----------------------------------------------------------------------------------
# Scene Setup Helper
# ----------------------------------------------------------------------------------

def append_scene_from_blend(local_blend_path):
    """
    Appends the first scene from the given `.blend` file into the current Blender project.
    Does NOT delete the `.blend` file.
    
    Returns:
       A tuple: (result, appended_scene_name)
       result is {'FINISHED'} on success, {'CANCELLED'} on failure.
       appended_scene_name is the actual name of the scene appended, or None if failed.
    """
    print(f"[INFO] Appending scene from: {local_blend_path}")
    try:
        # Load the list of scenes from the .blend file
        with bpy.data.libraries.load(local_blend_path, link=False) as (data_from, data_to):
            scene_names = data_from.scenes

        if not scene_names:
            print(f"[ERROR] No scenes found in {local_blend_path}")
            return {'CANCELLED'}, None

        # Choose the first scene from the file
        scene_to_append = scene_names[0]
        print(f"[INFO] Appending scene: {scene_to_append}")

        bpy.ops.wm.append(
            filepath=os.path.join(local_blend_path, "Scene", scene_to_append),
            directory=os.path.join(local_blend_path, "Scene"),
            filename=scene_to_append
        )
        print(f"[INFO] Scene '{scene_to_append}' appended successfully.")

        # Retrieve the appended scene by its name (as it comes from the file)
        appended_scene = bpy.data.scenes.get(scene_to_append)
        if appended_scene is None:
            print(f"[ERROR] Appended scene '{scene_to_append}' not found in bpy.data.scenes")
            return {'CANCELLED'}, None

        # Set the appended scene as the active scene
        bpy.context.window.scene = appended_scene
        print(f"[INFO] Scene set as active: {appended_scene.name}")

    except Exception as e:
        print(f"[ERROR] Failed to append scene: {e}")
        traceback.print_exc()
        return {'CANCELLED'}, None

    return {'FINISHED'}, appended_scene.name




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
        print(f"[INFO] Thumbnail already cached: {cached_path}")
        return cached_path

    parsed = urlparse(url)
    base_name = os.path.basename(parsed.path) or "unknown_thumbnail.png"
    _, ext = os.path.splitext(base_name)
    if ext.lower() not in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
        ext = ".png"
    local_filename = base_name
    local_path = os.path.join(THUMBNAIL_CACHE_FOLDER, local_filename)
    print(f"[INFO] Downloading thumbnail to: {local_path}")

    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"[ERROR] Failed to download thumbnail. HTTP {response.status_code}")
            return None

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        # Register the thumbnail in the JSON cache using the chosen key.
        register_thumbnail_in_index(key, local_path, thumbnail_url=url)
        return local_path
    except Exception as e:
        print(f"[ERROR] Exception during thumbnail download: {e}")
        return None


def on_filter_changed(self, context):
    # Call the new operator
    bpy.ops.webapp.refresh_filters()


#remove the world temp file and appended scene - called in game modal cancel()
def cleanup_downloaded_worlds():

    # Try to get the active scene and the stored appended scene name.
    try:
        scene = bpy.context.scene
        appended_scene_name = scene.get("appended_scene_name")
    except ReferenceError:
        scene = None
        appended_scene_name = None

    if appended_scene_name and appended_scene_name in bpy.data.scenes:
        appended_scene = bpy.data.scenes[appended_scene_name]
        print(f"Found appended scene: {appended_scene_name}")

        # Iterate over a copy of the appended scene's objects.
        for obj in list(appended_scene.objects):
            # Save the object's name before removal.
            try:
                obj_name = obj.name
            except ReferenceError:
                obj_name = "<unknown>"
            
            # If the object has data (e.g. mesh), store it and its name.
            mesh_data = None
            mesh_data_name = None
            if hasattr(obj, "data") and obj.data is not None:
                try:
                    mesh_data = obj.data
                    mesh_data_name = mesh_data.name
                except ReferenceError:
                    mesh_data = None
                    mesh_data_name = "<unknown>"

            # Remove the object.
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
                print(f"Removed object: {obj_name}")
            except Exception as e:
                print(f"Error removing object {obj_name}: {e}")

            # If the object had mesh data and no one is using it, remove it.
            if mesh_data is not None:
                try:
                    if mesh_data.users == 0:
                        bpy.data.meshes.remove(mesh_data)
                        print(f"Removed mesh data: {mesh_data_name}")
                except Exception as e:
                    # If the data block is already removed, we skip.
                    print(f"Error removing mesh data {mesh_data_name}: {e}")

        # Now remove the appended scene.
        try:
            bpy.data.scenes.remove(appended_scene)
            print(f"Removed appended scene: {appended_scene_name}")
        except Exception as e:
            print(f"Error removing scene {appended_scene_name}: {e}")
    else:
        print("No appended scene found to remove.")

    # Delete all files in the WORLD_DOWNLOADS_FOLDER.
    if os.path.isdir(WORLD_DOWNLOADS_FOLDER):
        for filename in os.listdir(WORLD_DOWNLOADS_FOLDER):
            file_path = os.path.join(WORLD_DOWNLOADS_FOLDER, filename)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        print("WORLD_DOWNLOADS_FOLDER not found.")

    # Remove the custom properties from the scene if it is still valid.
    if scene is not None:
        try:
            for key in ("appended_scene_name", "world_blend_path"):
                if key in scene:
                    del scene[key]
        except ReferenceError:
            pass



#-------------------------------------
#clear the World Download temp folder
#-------------------------------------
@persistent
def cleanup_world_downloads(dummy=None):
    """
    Removes all files within the World Downloads folder.
    Called on certain Blender events (load_post, save_pre, etc.).
    """
    if not os.path.isdir(WORLD_DOWNLOADS_FOLDER):
        return  # Folder doesn't even exist; nothing to do.

    for filename in os.listdir(WORLD_DOWNLOADS_FOLDER):
        file_path = os.path.join(WORLD_DOWNLOADS_FOLDER, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"[INFO] Removed leftover file: {file_path}")
            except OSError as e:
                print(f"[WARNING] Could not remove {file_path}: {e}")
        # If you expect subdirectories, handle them here too


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
    print(f"[INFO] Started background thread to fetch metadata for package {package_id}.")


    thread = threading.Thread(target=_fetch_metadata_worker, args=(package_id,), daemon=True)
    thread.start()
    print(f"[INFO] Started background thread to fetch metadata for package {package_id}.")

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
                print(f"[INFO] Downloaded thumbnail for package {package_id} to {local_thumb_path}")
            else:
                metadata["local_thumb_path"] = None

            register_metadata_in_index(package_id, metadata)
            print(f"[INFO] Successfully fetched and cached metadata for package {package_id}.")
        else:
            print(f"[ERROR] Failed to fetch metadata for package {package_id}: {data.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"[ERROR] Exception while fetching metadata for package {package_id}: {e}")




# -------------------------------------------------------------------
# Time Utils
# -------------------------------------------------------------------

def format_relative_time(upload_date_str):
    """
    Converts an upload date string to a relative format.
    Example outputs:
      - "50s" for 50 seconds ago
      - "2m" for 2 minutes ago
      - "3h" for 3 hours ago
      - "5d" for 5 days ago
      - "1y" for 1 year ago

    This function assumes the upload_date_str is in ISO 8601 format.
    Adjust the parsing logic if your format differs.
    """
    try:
        # Try ISO format first
        dt = datetime.fromisoformat(upload_date_str)
    except Exception:
        # Fallback: try a common format (adjust as needed)
        try:
            dt = datetime.strptime(upload_date_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            # If parsing fails, return the original string
            return upload_date_str

    # Ensure dt is timezone aware (assume UTC if not provided)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    diff = now - dt
    seconds = diff.total_seconds()

    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h"
    elif seconds < 31536000:  # less than a year
        days = int(seconds / 86400)
        return f"{days}d"
    else:
        years = int(seconds / 31536000)
        return f"{years}y"