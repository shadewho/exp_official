#Exploratory/Exp_UI/download_and_explore/explore_main.py

import bpy
import time
import requests
from ..auth.helpers import load_token
from ..main_config import PACKAGES_ENDPOINT, WORLD_DOWNLOADS_FOLDER
from .append_scene_helper import append_scene_from_blend
import threading, os, uuid, traceback

# Global download task reference.
current_download_task = None
download_progress_value = 0.0

def reset_download_progress():
    """
    Hard reset of all progress counters before a new download begins.
    Called from UI click handler just before starting a DownloadTask.
    """
    global download_progress_value
    download_progress_value = 0.0

# --- Central bail path -----------------------------------------------
def _bail_ui(reason: str):
    """
    Runs on Blender's main thread. Resets UI back to Browse,
    clears progress, cleans temp files, and forces a redraw.
    """
    try:
        scene = bpy.context.scene
        scene.ui_current_mode = "BROWSE"
        scene.show_loading_image = False
        scene.download_progress = 0.0
        reset_download_progress()                 # <- from previous fix
    except Exception:
        pass

    # Nuke temp downloads and forget the running task
    try:
        clear_world_downloads_folder()
    except Exception:
        pass

    global current_download_task
    current_download_task = None

    # Nudges the draw loop to rebuild
    bpy.types.Scene.package_ui_dirty = True
    print(f"[LOAD-BAIL] {reason}")
    try:
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    except Exception:
        pass

def bail_to_browse(reason: str):
    """
    Schedules the UI bail on the main thread (safe to call from worker threads).
    """
    def _runner():
        _bail_ui(reason)
        return None
    bpy.app.timers.register(_runner, first_interval=0.0)
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
            if self.cancelled:
                self.done = True
                return

            token = load_token()
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            explore_url = PACKAGES_ENDPOINT.replace("/packages", "/explore")
            payload = {"download_code": self.download_code, "file_type": "world"}
            response = requests.post(explore_url, json=payload, headers=headers, timeout=15)
            if self.cancelled:
                self.done = True
                return
            if response.status_code != 200:
                self.error = f"Explore API {response.status_code}"
                self.done = True
                return
            j = response.json()
            if not j.get("success"):
                self.error = j.get("message", "Explore API call failed.")
                self.done = True
                return

            self.download_url = j.get("download_url")
            if not self.download_url:
                self.error = "No download_url provided by server."
                self.done = True
                return

            self.local_blend_path = async_download_blend_file(
                self.download_url,
                progress_callback=self.update_progress,
                task=self
            )
            # async_download_blend_file already bails on failure; still set error for the timer
            if not self.local_blend_path:
                self.error = "Download failed."
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
    Downloads with basic stall/timeout detection.
    Returns the local path or None on failure/cancel.
    """
    try:
        base_filename = os.path.basename(url.split('?')[0]) or "world.blend"
        if not base_filename.endswith('.blend'):
            base_filename += '.blend'
        unique_id = uuid.uuid4().hex
        unique_filename = f"{os.path.splitext(base_filename)[0]}_{unique_id}.blend"
        local_path = os.path.join(WORLD_DOWNLOADS_FOLDER, unique_filename)

        # Short connect timeout + bounded read timeout
        response = requests.get(url, stream=True, timeout=(5, None))
        if response.status_code != 200:
            bail_to_browse(f"HTTP {response.status_code} while downloading.")
            return None

        total = int(response.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        last_progress_time = time.time()
        STALL_SECS = 60        # no bytes for this long => bail
        MIN_OK_RATIO = 0.95    # if Content-Length present, require at least this much

        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if task.cancelled:
                    file.close()
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    bail_to_browse("Download cancelled.")
                    return None

                if not chunk:
                    # No chunk received—check for stall
                    if time.time() - last_progress_time > STALL_SECS:
                        file.close()
                        try:
                            os.remove(local_path)
                        except Exception:
                            pass
                        bail_to_browse("Network stalled while downloading.")
                        return None
                    continue

                file.write(chunk)
                downloaded += len(chunk)
                last_progress_time = time.time()
                if total > 0:
                    progress_callback(downloaded / total)

        if total > 0 and downloaded < int(total * MIN_OK_RATIO):
            # Probably truncated
            try:
                os.remove(local_path)
            except Exception:
                pass
            bail_to_browse("Download incomplete / truncated.")
            return None

        progress_callback(1.0)
        return local_path

    except requests.RequestException as e:
        bail_to_browse(f"Network error: {e!s}")
        return None
    except Exception as e:
        traceback.print_exc()
        bail_to_browse("Unexpected error during download.")
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
    from .explore_main import clear_world_downloads_folder  # local import ok
    global current_download_task, download_progress_value

    scene = bpy.context.scene
    # Keep the loading UI in sync
    scene.download_progress = download_progress_value

    # 1) Wait for a valid task
    if current_download_task is None:
        return None  # nothing to do

    if not current_download_task.done:
        return 0.1   # poll again shortly

    # If the worker reported an error, bail to Browse
    if current_download_task.error:
        bail_to_browse(f"Load failed: {current_download_task.error}")
        current_download_task = None
        return None

    # 2) Handle cancellation or missing file
    if current_download_task.cancelled or not current_download_task.local_blend_path:
        bail_to_browse("Download cancelled or missing file.")
        clear_world_downloads_folder()
        current_download_task = None
        return None

    # 2.5) Quick sanity check that the downloaded file is a valid .blend
    try:
        with bpy.data.libraries.load(current_download_task.local_blend_path, link=False) as (df, dt):
            _ = df.scenes  # touch to force header read
    except Exception:
        bail_to_browse("Downloaded file is not a valid .blend.")
        clear_world_downloads_folder()
        current_download_task = None
        return None

    # 3) Snapshot existing datablocks just *once*, before we append
    scene["initial_datablocks"] = {
        "actions": list(bpy.data.actions.keys()),
        "images":  list(bpy.data.images.keys()),
        "sounds":  list(bpy.data.sounds.keys()),
        "meshes":  list(bpy.data.meshes.keys()),
        "objects": list(bpy.data.objects.keys()),
        "materials": list(bpy.data.materials.keys()),
        "armatures":  list(bpy.data.armatures.keys()),
        "curves":    list(bpy.data.curves.keys()),
        "lights":    list(bpy.data.lights.keys()),
    }

    # 4) Append the downloaded blend
    result, appended_scene_name = append_scene_from_blend(current_download_task.local_blend_path)
    if result != {'FINISHED'} or not appended_scene_name:
        bail_to_browse("Append failed or produced no scene.")
        clear_world_downloads_folder()
        current_download_task = None
        return None

    # 5) Record scene metadata
    scene["appended_scene_name"] = appended_scene_name
    scene["world_blend_path"]    = current_download_task.local_blend_path

    # 6) Compute which datablocks arrived with that append
    init = scene["initial_datablocks"]
    scene["appended_datablocks"] = {
        "actions": [a for a in bpy.data.actions.keys() if a not in init["actions"]],
        "images":  [i for i in bpy.data.images.keys()  if i not in init["images"]],
        "sounds":  [s for s in bpy.data.sounds.keys()  if s not in init["sounds"]],
        "meshes":  [m for m in bpy.data.meshes.keys()  if m not in init["meshes"]],
        "objects": [o for o in bpy.data.objects.keys() if o not in init["objects"]],
        "materials": [t for t in bpy.data.materials.keys() if t not in init["materials"]],
        "armatures":  [r for r in bpy.data.armatures.keys()  if r not in init["armatures"]],
        "curves":    [c for c in bpy.data.curves.keys()    if c not in init["curves"]],
        "lights":    [l for l in bpy.data.lights.keys()    if l not in init["lights"]],
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

        # Do NOT pop original_scene here; cancel() will pop and revert later.
        # Just switch into the appended scene and mark UI mode.
        bpy.context.window.scene = scene_obj
        scene_obj.ui_current_mode = "GAME"

        # Find a VIEW_3D override and launch the game (fullscreen path in your operator)
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
                    bpy.ops.exploratory.start_game('INVOKE_DEFAULT', launched_from_ui=True)
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

    reset_download_progress()
    context.scene.download_progress = 0.0

    # 2) Kick off the download + timer as before
    if download_code:
        current_download_task = DownloadTask(download_code)
        threading.Thread(target=current_download_task.run, daemon=True).start()
        bpy.app.timers.register(timer_finish_download, first_interval=0.1)
    else:
        print("No download code provided.")