# helper_functions.py

import os
import requests
import bpy
import uuid
import traceback
from urllib.parse import urlparse

from .main_config import (
    DOWNLOAD_ENDPOINT, VALIDATE_TOKEN_ENDPOINT, ADDON_FOLDER,
    WORLD_DOWNLOADS_FOLDER, SHOP_DOWNLOADS_FOLDER,
    TOKEN_FILE, THUMBNAIL_CACHE_FOLDER
)

from .image_button_UI.cache import (
    get_cached_path_if_exists,
    register_thumbnail_in_index
)

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

def append_scene_from_blend(local_blend_path, new_scene_name="Appended_Scene"):
    """
    Appends the first scene from the given `.blend` file into the current Blender project.
    Does NOT delete the `.blend` file.
    
    After appending, renames the appended scene to `new_scene_name` and sets it as the active scene.
    
    Returns {'FINISHED'} on success, {'CANCELLED'} on failure.
    """

    print(f"[INFO] Appending scene from: {local_blend_path}")

    try:
        # Load the list of scenes from the .blend file
        with bpy.data.libraries.load(local_blend_path, link=False) as (data_from, data_to):
            scene_names = data_from.scenes

        if not scene_names:
            print(f"[ERROR] No scenes found in {local_blend_path}")
            return {'CANCELLED'}

        # Choose the first scene from the file
        scene_to_append = scene_names[0]
        print(f"[INFO] Appending scene: {scene_to_append}")

        # Append the scene using bpy.ops.wm.append
        bpy.ops.wm.append(
            filepath=os.path.join(local_blend_path, "Scene", scene_to_append),
            directory=os.path.join(local_blend_path, "Scene"),
            filename=scene_to_append
        )
        print(f"[INFO] Scene '{scene_to_append}' appended successfully.")

        # Retrieve the appended scene from bpy.data.scenes.
        # It will have the same name as in the source blend file.
        appended_scene = bpy.data.scenes.get(scene_to_append)
        if appended_scene is None:
            print(f"[ERROR] Appended scene '{scene_to_append}' not found in bpy.data.scenes")
            return {'CANCELLED'}

        # Set the newly appended (and renamed) scene as the active scene in the current window.
        bpy.context.window.scene = appended_scene
        print(f"[INFO] Scene set as active: {appended_scene.name}")
        

    except Exception as e:
        print(f"[ERROR] Failed to append scene: {e}")
        traceback.print_exc()
        return {'CANCELLED'}

    return {'FINISHED'}



# ----------------------------------------------------------------------------------
# Thumbnail Download Helper
# ----------------------------------------------------------------------------------

def download_thumbnail(url):
    """
    Download the thumbnail *immediately* on the main thread, 
    storing it in THUMBNAIL_CACHE_FOLDER, and return local_path or None.
    """
    cached_path = get_cached_path_if_exists(url)
    if cached_path:
        print(f"[INFO] Thumbnail is already cached: {cached_path}")
        return cached_path

    # Build a local_path from the URL (like before)
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

        # Write the thumbnail to disk immediately
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Return the local_path so calling code knows we have it
        return local_path
    except Exception as e:
        print(f"[ERROR] Exception during thumbnail download: {e}")
        return None


def on_filter_changed(self, context):
    # Call the new operator
    bpy.ops.webapp.refresh_filters()