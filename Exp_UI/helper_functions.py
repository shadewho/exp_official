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
    TOKEN_FILE, THUMBNAIL_CACHE_FOLDER, PACKAGE_DETAILS_ENDPOINT,
    EVENTS_ENDPOINT
)

from .image_button_UI.cache import (
    get_cached_path_if_exists,
    register_thumbnail_in_index,
    register_metadata_in_index, get_cached_metadata
)
from .auth import load_token
from bpy.app.handlers import persistent
from datetime import datetime, timezone
from .cache_manager import filter_cached_data


# ----------------------------------------------------------------------------------
# Download Helper
# ----------------------------------------------------------------------------------

def download_blend_file(url, progress_callback=None):
    """
    Downloads the .blend file from the given `url` into the addon directory's
    "World Downloads" folder. Returns the local file path if successful.
    If progress_callback is provided, it is called with a float between 0.0 and 1.0.
    """
    import uuid, os, traceback

    base_filename = os.path.basename(url.split('?')[0])
    if not base_filename.endswith('.blend'):
        base_filename += '.blend'
    
    unique_id = uuid.uuid4().hex
    unique_filename = f"{os.path.splitext(base_filename)[0]}_{unique_id}.blend"
    local_path = os.path.join(WORLD_DOWNLOADS_FOLDER, unique_filename)

    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            return None

        total = response.headers.get("Content-Length")
        if total is not None:
            total = int(total)
        else:
            total = 0

        downloaded = 0
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and progress_callback:
                        progress = downloaded / total
                        progress_callback(progress)
            file.flush()
            os.fsync(file.fileno())

        if os.path.getsize(local_path) == 0:
            os.remove(local_path)
            return None

        # Ensure progress is set to 100%
        if progress_callback:
            progress_callback(1.0)
        return local_path

    except Exception as e:
        traceback.print_exc()
        return None


# ----------------------------------------------------------------------------------
# Scene Setup Helper
# ----------------------------------------------------------------------------------

def append_scene_from_blend(local_blend_path, scene_name=None):
    """
    Attempts to append a scene from a .blend file. If a scene_name is provided and exists
    in the file, that scene is appended; otherwise, if a text datablock called "GAME_WORLD"
    exists in the file, its content is used as the scene name. If neither is provided or valid,
    the first available scene is appended.

    This version also handles naming collisions by detecting which new scene
    was actually appended (e.g. "Scene.001") and then renaming it (if desired)
    so it never conflicts with the user's original scenes.

    Returns:
        (result, appended_scene_name)

        result: {'FINISHED'} if the scene was successfully appended, otherwise {'CANCELLED'}.

        appended_scene_name: The final name of the appended scene if successful;
                             otherwise None.
    """

    print("[LOG] Starting append_scene_from_blend with file:", local_blend_path)

    # Step 1: Load available scenes and texts from the blend file.
    try:
        print("[LOG] Attempting to load library from file...")
        with bpy.data.libraries.load(local_blend_path, link=False) as (data_from, data_to):
            scene_names = data_from.scenes
            text_names = data_from.texts
            print("[LOG] Library loaded. Found scenes:", scene_names)
            print("[LOG] Library loaded. Found texts:", text_names)
    except Exception as e:
        print("[EXCEPTION] Failed to load library from file:", local_blend_path)
        traceback.print_exc()
        return {'CANCELLED'}, None

    # Step 2: Check if any scenes are available.
    if not scene_names:
        print("[ERROR] No scenes found in the file:", local_blend_path)
        return {'CANCELLED'}, None

    # Step 3: Determine which scene to append.
    chosen_scene_name = None
    if scene_name:
        if scene_name in scene_names:
            chosen_scene_name = scene_name
            print("[LOG] Using user-specified game world scene:", chosen_scene_name)
        else:
            print("[ERROR] The specified scene", scene_name, "was not found in the blend file.")
            return {'CANCELLED'}, None
    else:
        # If no specific scene is specified, check for a text datablock called "GAME_WORLD".
        if "GAME_WORLD" in text_names:
            try:
                # Load the GAME_WORLD text datablock.
                with bpy.data.libraries.load(local_blend_path, link=False) as (data_from2, data_to2):
                    data_to2.texts = ["GAME_WORLD"]

                game_world_text = bpy.data.texts.get("GAME_WORLD")
                if game_world_text:
                    inferred_scene_name = game_world_text.as_string().strip()
                    print("[LOG] GAME_WORLD text found. Specified scene name:", inferred_scene_name)
                    if inferred_scene_name in scene_names:
                        chosen_scene_name = inferred_scene_name
                    else:
                        print("[ERROR] Inferred scene name from GAME_WORLD not found among available scenes.")
                        return {'CANCELLED'}, None
                else:
                    print("[ERROR] GAME_WORLD text block could not be loaded properly.")
                    chosen_scene_name = scene_names[0]
            except Exception as e:
                print("[EXCEPTION] Failed to load GAME_WORLD text block:")
                traceback.print_exc()
                chosen_scene_name = scene_names[0]
        else:
            # Default to the first available scene
            chosen_scene_name = scene_names[0]
            print("[LOG] No GAME_WORLD text block found. Defaulting to first scene:", chosen_scene_name)

    # Step 4: Build file paths for the append operator.
    append_filepath = os.path.join(local_blend_path, "Scene", chosen_scene_name)
    append_directory = os.path.join(local_blend_path, "Scene")
    print("[LOG] Prepared append parameters:")
    print("      filepath =", append_filepath)
    print("      directory =", append_directory)
    print("      filename =", chosen_scene_name)

    # ## NEW: keep track of existing scenes before appending
    before_scenes = set(bpy.data.scenes.keys())

    # Step 5: Call the append operator.
    try:
        result = bpy.ops.wm.append(
            filepath=append_filepath,
            directory=append_directory,
            filename=chosen_scene_name
        )
        print("[LOG] bpy.ops.wm.append returned:", result)
    except Exception as e:
        print("[EXCEPTION] Exception during bpy.ops.wm.append:")
        traceback.print_exc()
        return {'CANCELLED'}, None

    # Step 6: Figure out what new scene(s) arrived and pick the relevant one
    after_scenes = set(bpy.data.scenes.keys())
    new_scenes = after_scenes - before_scenes

    if not new_scenes:
        print("[ERROR] No new scene was appended, naming conflict or unknown error.")
        return {'CANCELLED'}, None
    elif len(new_scenes) > 1:
        # In rare cases, multiple new scenes might appear if you appended a group. 
        # You might pick the one that either ends with chosen_scene_name or handle differently.
        appended_scene_name = None
        for nm in new_scenes:
            # If Blender appended it with a suffix, e.g. "Scene" -> "Scene.001", we check startswith
            if nm.startswith(chosen_scene_name):
                appended_scene_name = nm
                break
        if appended_scene_name is None:
            # fallback: just pick one
            appended_scene_name = new_scenes.pop()
    else:
        appended_scene_name = new_scenes.pop()

    appended_scene = bpy.data.scenes.get(appended_scene_name)
    if not appended_scene:
        print("[ERROR] Could not find appended scene data-block after append.")
        return {'CANCELLED'}, None

    # ## OPTIONAL: rename the appended scene to a guaranteed-unique name
    unique_name = f"Appended_{uuid.uuid4().hex[:8]}"
    appended_scene.name = unique_name
    appended_scene_name = unique_name
    print(f"[LOG] Renamed appended scene to: {unique_name}")

    # Step 7: Set the appended scene as active.
    try:
        bpy.context.window.scene = appended_scene
        print("[LOG] Appended scene set as active.")
    except Exception as e:
        print("[EXCEPTION] Failed to set the appended scene as active:")
        traceback.print_exc()
        return {'CANCELLED'}, appended_scene_name

    print("[LOG] Successfully appended scene with final name:", appended_scene.name)
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
            except Exception as e:
                print(f"Error removing object {obj_name}: {e}")

            # If the object had mesh data and no one is using it, remove it.
            if mesh_data is not None:
                try:
                    if mesh_data.users == 0:
                        bpy.data.meshes.remove(mesh_data)
                except Exception as e:
                    # If the data block is already removed, we skip.
                    print(f"Error removing mesh data {mesh_data_name}: {e}")

        # Now remove the appended scene.
        try:
            bpy.data.scenes.remove(appended_scene)
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
    

def on_filter_changed(self, context):
    # Attempt to remove the active modal UI (if any)
    try:
        bpy.ops.view3d.remove_package_display('EXEC_DEFAULT')
    except Exception as e:
        pass
    # Now, call the operator to apply the filters and show a fresh UI.
    bpy.ops.webapp.apply_filters_showui('EXEC_DEFAULT', page_number=1)


# -------------------------------------------------------------------
# Filter Events
# -------------------------------------------------------------------
def fetch_events_by_stage():
    # Build the URL. (Adjust BASE_URL if necessary so that it points to your website's API.)
    url = EVENTS_ENDPOINT
    print(url)

    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("success"):
            return data  # Expected to have keys: 'submission', 'vote', 'winners'
        else:
            print("Event fetch error:", data.get("message"))
            return {}
    except Exception as e:
        print("Error fetching events by stage:", e)
        return {}
    
def update_event_stage(self, context):
    events = fetch_events_by_stage()
    context.scene["fetched_events_data"] = events

    stage = context.scene.event_stage  # will be 'submission', 'voting', or 'winners'
    stage_events = events.get(stage, [])
    if stage_events:
        context.scene.selected_event = str(stage_events[0]["id"])
    else:
        context.scene.selected_event = "0"
    
    if context.area:
        context.area.tag_redraw()




# Define the selected_event property that uses a dynamic items callback.
def get_event_items(self, context):
    events_data = context.scene.get("fetched_events_data", {})
    stage = context.scene.event_stage  # already 'submission', 'voting', or 'winners'
    items = []
    stage_events = events_data.get(stage, [])
    for event in stage_events:
        items.append((str(event["id"]), event["title"], event.get("description", "")))
    if not items:
        items = [("0", "No events", "No active event in this stage")]
    return items



bpy.types.Scene.selected_event = bpy.props.EnumProperty(
    name="Event",
    description="Select an event to filter packages",
    items=get_event_items
)

# -------------------------------------------------------------------
#refresh the usage data for a users subscription
# -------------------------------------------------------------------
def auto_refresh_usage():
    try:
        bpy.ops.webapp.refresh_usage('INVOKE_DEFAULT')
        print("Auto refresh usage operator called.")
    except Exception as e:
        print("Error auto refreshing usage:", e)
    return None  # Returning None stops the timer