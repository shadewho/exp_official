#Exploratory/Exp_UI/download_and_explore/explore_main.py

import bpy
import requests
from ..auth.helpers import load_token
from ..main_config import PACKAGES_ENDPOINT, WORLD_DOWNLOADS_FOLDER
from .append_scene_helper import append_scene_from_blend
import threading, os, uuid, traceback

# Global download task reference.
current_download_task = None
download_progress_value = 0.0


#the main download task class that handles the download process
class DownloadTask:
    def __init__(self, download_code):
        self.download_code = download_code
        self.download_url = None
        self.local_blend_path = None
        self.error = None
        self.done = False
        self.progress = 0.0  # A float from 0.0 to 1.0
        self.cancelled = False  # Cancellation flag

    def cancel(self):
        self.cancelled = True
        print("[INFO] DownloadTask cancelled.")

    def run(self):
        try:
            # Step 1: POST request to get the download URL.
            if self.cancelled:
                self.done = True
                return
            token = load_token()
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            explore_url = PACKAGES_ENDPOINT.replace("/packages", "/explore")
            payload = {"download_code": self.download_code, "file_type": "world"}
            response = requests.post(explore_url, json=payload, headers=headers)
            if self.cancelled:
                self.done = True
                return
            if response.status_code != 200 or not response.json().get("success"):
                self.error = "Explore API call failed."
                self.done = True
                return
            self.download_url = response.json().get("download_url")
            # Step 2: Download the .blend file with streaming.
            self.local_blend_path = async_download_blend_file(
                self.download_url, 
                progress_callback=self.update_progress, 
                task=self
            )
            self.done = True
        except Exception as e:
            self.error = str(e)
            self.done = True

    def update_progress(self, progress):
        global download_progress_value
        self.progress = progress
        download_progress_value = progress

def async_download_blend_file(url, progress_callback, task):
    """
    Downloads the file from `url` in chunks and calls progress_callback(progress)
    where progress is a float from 0.0 to 1.0. This version checks for cancellation.
    Returns the local file path when done or None if cancelled.
    """
    try:
        # Create a unique filename in the WORLD_DOWNLOADS_FOLDER.
        base_filename = os.path.basename(url.split('?')[0])
        if not base_filename.endswith('.blend'):
            base_filename += '.blend'
        unique_id = uuid.uuid4().hex
        unique_filename = f"{os.path.splitext(base_filename)[0]}_{unique_id}.blend"
        local_path = os.path.join(WORLD_DOWNLOADS_FOLDER, unique_filename)
        
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            return None

        total = response.headers.get("Content-Length")
        total = int(total) if total else 0
        downloaded = 0
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if task.cancelled:
                    file.close()
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    return None
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        progress_callback(downloaded / total)
        progress_callback(1.0)
        return local_path
    except Exception as e:
        traceback.print_exc()
        return None

def clear_world_downloads_folder():
    """
    Clears all files in the WORLD_DOWNLOADS_FOLDER.
    """
    if os.path.isdir(WORLD_DOWNLOADS_FOLDER):
        for filename in os.listdir(WORLD_DOWNLOADS_FOLDER):
            file_path = os.path.join(WORLD_DOWNLOADS_FOLDER, filename)
            try:
                os.remove(file_path)
                print(f"[INFO] Removed file: {file_path}")
            except Exception as e:
                print(f"[WARNING] Could not remove {file_path}: {e}")

def timer_finish_download():
    global current_download_task, download_progress_value

    scene = bpy.context.scene
    # Update progress bar
    scene.download_progress = download_progress_value

    # 1) Wait for a valid task
    if current_download_task is None:
        return None      # nothing to do
    if not current_download_task.done:
        return 0.1       # poll again shortly
    if current_download_task.error:
        current_download_task = None
        return None      # stop on error

    # 2) Handle cancellation or missing file
    if current_download_task.cancelled or not current_download_task.local_blend_path:
        clear_world_downloads_folder()
        current_download_task = None
        return None

    # 3) Snapshot existing datablocks just *once*, before we append
    scene["initial_datablocks"] = {
        "actions": list(bpy.data.actions.keys()),
        "images":  list(bpy.data.images.keys()),
        "sounds":  list(bpy.data.sounds.keys()),
        "meshes":  list(bpy.data.meshes.keys()),   # ← new
        "objects": list(bpy.data.objects.keys()),  # ← optional, if you want object-level diffs
    }

    # 4) Append the downloaded blend
    result, appended_scene_name = append_scene_from_blend(current_download_task.local_blend_path)
    if result != {'FINISHED'} or not appended_scene_name:
        print(f"[ERROR] append_scene_from_blend failed: {result}")
        clear_world_downloads_folder()
        current_download_task = None
        return None

    # 5) Record scene metadata
    scene["appended_scene_name"] = appended_scene_name
    scene["world_blend_path"]     = current_download_task.local_blend_path

    # 6) Compute which datablocks arrived with that append
    init = scene["initial_datablocks"]
    scene["appended_datablocks"] = {
        "actions": [a for a in bpy.data.actions.keys() if a not in init["actions"]],
        "images":  [i for i in bpy.data.images.keys()  if i not in init["images"]],
        "sounds":  [s for s in bpy.data.sounds.keys()  if s not in init["sounds"]],
        "meshes":  [m for m in bpy.data.meshes.keys()  if m not in init["meshes"]],
        "objects": [o for o in bpy.data.objects.keys() if o not in init["objects"]],
    }

    # 7) Clean up the temporary UI
    bpy.ops.view3d.remove_package_display('EXEC_DEFAULT')
    if hasattr(bpy.types.Scene, "gpu_image_buttons_data"):
        bpy.types.Scene.gpu_image_buttons_data.clear()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    # 8) Wait until the new scene really exists, then launch the game
    def check_scene():
        scene_obj = bpy.data.scenes.get(appended_scene_name)
        if not scene_obj:
            print("Waiting for appended scene…")
            return 1.0

        # pull and clear original scene/workspace
        wm = bpy.context.window_manager
        orig_scene = wm.pop('original_scene', None)
        orig_ws    = bpy.context.window.workspace.name

        # switch into it
        bpy.context.window.scene = scene_obj
        scene_obj.ui_current_mode = "GAME"

        # find a VIEW_3D override
        view3d = next((a for a in bpy.context.window.screen.areas if a.type == 'VIEW_3D'), None)
        if view3d:
            region = next((r for r in view3d.regions if r.type == 'WINDOW'), None)
            if region:
                override = {
                    'window': bpy.context.window,
                    'screen': bpy.context.window.screen,
                    'area':   view3d,
                    'region': region,
                }
                with bpy.context.temp_override(**override):
                    bpy.ops.exploratory.start_game(
                        'INVOKE_DEFAULT',
                        launched_from_ui=True,
                        original_workspace_name=orig_ws,
                        original_scene_name=orig_scene
                    )
        return None

    bpy.app.timers.register(check_scene)

    # 9) Done—clear the task so we don’t re-enter
    current_download_task = None
    return None



def explore_icon_handler(context, download_code):
    """
    Called when the Explore Icon is clicked.
    It stores the original scene on the window_manager, then
    kicks off the download task exactly as before.
    """
    global current_download_task

    # 1) Store the name of the scene you were on
    context.window_manager['original_scene'] = context.window.scene.name

    # 2) Kick off the download + timer as before
    if download_code:
        current_download_task = DownloadTask(download_code)
        threading.Thread(target=current_download_task.run, daemon=True).start()
        bpy.app.timers.register(timer_finish_download, first_interval=0.1)
    else:
        print("No download code provided.")