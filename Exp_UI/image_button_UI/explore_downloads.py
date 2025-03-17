import bpy
import requests
from ..auth import load_token
from ..main_config import PACKAGES_ENDPOINT, WORLD_DOWNLOADS_FOLDER
from ..exp_api import download_blend_file, append_scene_from_blend
import threading, os, uuid, traceback

# Global download task reference.
current_download_task = None
download_progress_value = 0.0

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
                    print("[INFO] Download cancelled. Removing incomplete file.")
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
    bpy.context.scene.download_progress = download_progress_value

    if current_download_task is None:
        return None  # No task; stop timer.
    if not current_download_task.done:
        return 0.1  # Poll again in 0.1 seconds.
    if current_download_task.error:
        current_download_task = None
        return None  # Stop timer.

    # If the task was cancelled or no file was downloaded, clear downloads.
    if current_download_task.cancelled or not current_download_task.local_blend_path:
        print("[INFO] DownloadTask was cancelled or no file was downloaded. Clearing downloads folder.")
        clear_world_downloads_folder()
        current_download_task = None
        return None

    print("Download finished. Attempting to append scene from:", current_download_task.local_blend_path)
    result, appended_scene_name = append_scene_from_blend(current_download_task.local_blend_path)
    if result == {'FINISHED'} and appended_scene_name:
        bpy.context.scene["appended_scene_name"] = appended_scene_name
        bpy.context.scene["world_blend_path"] = current_download_task.local_blend_path
        print("Scene appended; now waiting for it to be available...")
        
        # Remove the custom UI.
        bpy.ops.view3d.remove_package_display('EXEC_DEFAULT')
        if hasattr(bpy.types.Scene, "gpu_image_buttons_data"):
            bpy.types.Scene.gpu_image_buttons_data.clear()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        
        # Wait until the appended scene is available before launching the game modal.
        def check_scene():
            scene_obj = bpy.data.scenes.get(appended_scene_name)
            if scene_obj:
                bpy.context.window.scene = scene_obj  # Set as active scene.
                print("Appended scene detected. Launching game modal.")
                bpy.context.scene.ui_current_mode = "GAME"
                view3d_area = next((area for area in bpy.context.window.screen.areas if area.type == 'VIEW_3D'), None)
                if view3d_area:
                    view3d_region = next((region for region in view3d_area.regions if region.type == 'WINDOW'), None)
                    if view3d_region:
                        override = bpy.context.copy()
                        override['area'] = view3d_area
                        override['region'] = view3d_region
                        with bpy.context.temp_override(**override):
                            bpy.ops.view3d.exp_modal('INVOKE_DEFAULT', launched_from_ui=True)
                    else:
                        print("No valid VIEW_3D region found.")
                else:
                    print("No VIEW_3D area found for context override.")
                return None  # Stop the timer.
            else:
                print("Waiting for appended scene...")
                return 1.0  # Check again in 1 second.
        bpy.app.timers.register(check_scene)
    else:
        print("Failed to append scene.")
    current_download_task = None
    return None

def explore_icon_handler(context, download_code):
    """
    Called when the Explore Icon is clicked.
    It starts a background download task and registers a timer callback to poll
    the task until completion.
    """
    global current_download_task
    if download_code:
        current_download_task = DownloadTask(download_code)
        threading.Thread(target=current_download_task.run, daemon=True).start()
        bpy.app.timers.register(timer_finish_download, first_interval=0.1)
    else:
        print("No download code provided.")
