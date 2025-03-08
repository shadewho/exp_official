import bpy
import requests
from ..auth import load_token
from ..main_config import PACKAGES_ENDPOINT
from ..exp_api import download_blend_file, append_scene_from_blend
import threading

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

    def run(self):
        try:
            # Step 1: POST request to get the download URL.
            token = load_token()
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            explore_url = PACKAGES_ENDPOINT.replace("/packages", "/explore")
            payload = {"download_code": self.download_code, "file_type": "world"}
            response = requests.post(explore_url, json=payload, headers=headers)
            if response.status_code != 200 or not response.json().get("success"):
                self.error = "Explore API call failed."
                self.done = True
                print("Explore API call failed:", response.text)
                return
            self.download_url = response.json().get("download_url")
            print("Download URL received:", self.download_url)
            # Step 2: Download the .blend file with streaming.
            self.local_blend_path = async_download_blend_file(self.download_url, progress_callback=self.update_progress)
            self.done = True
        except Exception as e:
            self.error = str(e)
            self.done = True
            print("Exception in download task:", e)

    def update_progress(self, progress):
        global download_progress_value
        download_progress_value = progress



def async_download_blend_file(url, progress_callback):
    """
    Download the file from `url` in chunks and call progress_callback(progress)
    where progress is a float from 0.0 to 1.0.
    Return the local file path when done.
    
    For now, this stub simply calls your synchronous download_blend_file.
    Replace this with streaming code if needed.
    """
    local_path = download_blend_file(url, progress_callback=progress_callback)
    progress_callback(1.0)
    return local_path

def timer_finish_download():
    global current_download_task, download_progress_value
    bpy.context.scene.download_progress = download_progress_value

    if current_download_task is None:
        return None  # No task; stop timer.
    if not current_download_task.done:
        print("Polling download progress:", current_download_task.progress)
        return 0.1  # Poll again in 0.1 seconds.
    if current_download_task.error:
        print("Download error:", current_download_task.error)
        current_download_task = None
        return None  # Stop timer.
    print("Download finished. Attempting to append scene from:", current_download_task.local_blend_path)
    result, appended_scene_name = append_scene_from_blend(current_download_task.local_blend_path)
    if result == {'FINISHED'} and appended_scene_name:
        bpy.context.scene["appended_scene_name"] = appended_scene_name
        bpy.context.scene["world_blend_path"] = current_download_task.local_blend_path
        print("Scene appended successfully, launching game modal.")
        
        # Remove the custom UI.
        bpy.ops.view3d.remove_package_display('EXEC_DEFAULT')
        if hasattr(bpy.types.Scene, "gpu_image_buttons_data"):
            bpy.types.Scene.gpu_image_buttons_data.clear()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        
        # Set UI mode to GAME so load_image_buttons returns an empty list.
        bpy.context.scene.ui_current_mode = "GAME"
        
        # Launch the game modal.
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
    else:
        print("Failed to append scene.")
    current_download_task = None
    return None


def explore_icon_handler(context, download_code):
    """
    This function is called when the Explore Icon is clicked.
    It starts a background download task and registers a timer callback
    to poll the task until it completes.
    """
    global current_download_task
    if download_code:
        current_download_task = DownloadTask(download_code)
        threading.Thread(target=current_download_task.run, daemon=True).start()
        bpy.app.timers.register(timer_finish_download, first_interval=0.1)
    else:
        print("No download code provided.")
