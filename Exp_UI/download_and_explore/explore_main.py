import bpy
import time
import requests
import threading, os, uuid, traceback

from ..auth.helpers import load_token
from ..main_config import PACKAGES_ENDPOINT, WORLD_DOWNLOADS_FOLDER
from .append_scene_helper import append_scene_from_blend
# Global download task reference.
current_download_task = None
download_progress_value = 0.0


def reset_download_progress():
    """
    Reset progress counters before a new download begins.
    """
    global download_progress_value
    download_progress_value = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Central bail/cleanup (detail-only UI; no browse mode)
# ──────────────────────────────────────────────────────────────────────────────
def _bail_ui(reason: str):
    """
    Runs on Blender's main thread. Closes the overlay, resets progress,
    clears temp files, and returns UI to GAME (no overlay).
    """
    try:
        # Close the overlay if it's up (safe if not running)
        try:
            bpy.ops.view3d.remove_package_display('EXEC_DEFAULT')
        except Exception:
            pass

        scene = bpy.context.scene
        scene.ui_current_mode = "GAME"
        scene.show_loading_image = False
        scene.download_progress = 0.0
        reset_download_progress()
    except Exception:
        pass

    # Nuke temp downloads and forget the running task
    try:
        clear_world_downloads_folder()
    except Exception:
        pass

    global current_download_task
    current_download_task = None

    print(f"[LOAD-BAIL] {reason}")
    try:
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    except Exception:
        pass


def bail_to_idle(reason: str):
    """
    Schedule the UI bail on the main thread (safe to call from worker threads).
    """
    def _runner():
        _bail_ui(reason)
        return None
    bpy.app.timers.register(_runner, first_interval=0.0)


# Backward-compat shim (if any old code still calls this)
def bail_to_browse(reason: str):
    return bail_to_idle(reason)


# ──────────────────────────────────────────────────────────────────────────────
# Download task
# ──────────────────────────────────────────────────────────────────────────────
class DownloadTask:
    def __init__(self, download_code):
        self.download_code = download_code
        self.download_url = None
        self.local_blend_path = None
        self.error = None
        self.done = False
        self.progress = 0.0
        self.cancelled = False

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
    Downloads with stall/timeout detection.
    Returns the local path or None on failure/cancel.
    """
    try:
        base_filename = os.path.basename(url.split('?')[0]) or "world.blend"
        if not base_filename.endswith('.blend'):
            base_filename += '.blend'
        unique_id = uuid.uuid4().hex
        unique_filename = f"{os.path.splitext(base_filename)[0]}_{unique_id}.blend"
        local_path = os.path.join(WORLD_DOWNLOADS_FOLDER, unique_filename)

        response = requests.get(url, stream=True, timeout=(5, None))
        if response.status_code != 200:
            bail_to_idle(f"HTTP {response.status_code} while downloading.")
            return None

        total = int(response.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        last_progress_time = time.time()
        STALL_SECS = 60
        MIN_OK_RATIO = 0.95

        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if task.cancelled:
                    file.close()
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    bail_to_idle("Download cancelled.")
                    return None

                if not chunk:
                    if time.time() - last_progress_time > STALL_SECS:
                        file.close()
                        try:
                            os.remove(local_path)
                        except Exception:
                            pass
                        bail_to_idle("Network stalled while downloading.")
                        return None
                    continue

                file.write(chunk)
                downloaded += len(chunk)
                last_progress_time = time.time()
                if total > 0:
                    progress_callback(downloaded / total)

        if total > 0 and downloaded < int(total * MIN_OK_RATIO):
            try:
                os.remove(local_path)
            except Exception:
                pass
            bail_to_idle("Download incomplete / truncated.")
            return None

        progress_callback(1.0)
        return local_path

    except requests.RequestException as e:
        bail_to_idle(f"Network error: {e!s}")
        return None
    except Exception:
        traceback.print_exc()
        bail_to_idle("Unexpected error during download.")
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


# ──────────────────────────────────────────────────────────────────────────────
# Timer: finalize download, append, launch game
# ──────────────────────────────────────────────────────────────────────────────
def timer_finish_download():
    from .explore_main import clear_world_downloads_folder  # local import ok
    global current_download_task, download_progress_value

    scene = bpy.context.scene
    scene.download_progress = download_progress_value


    # 1) Wait for a valid task
    if current_download_task is None:
        return None

    if not current_download_task.done:
        return 0.1

    # 2) Error / cancellation cases
    if current_download_task.error:
        bail_to_idle(f"Load failed: {current_download_task.error}")
        current_download_task = None
        return None

    if current_download_task.cancelled or not current_download_task.local_blend_path:
        bail_to_idle("Download cancelled or missing file.")
        clear_world_downloads_folder()
        current_download_task = None
        return None

    # 3) Validate .blend header
    try:
        with bpy.data.libraries.load(current_download_task.local_blend_path, link=False) as (df, dt):
            _ = df.scenes
    except Exception:
        bail_to_idle("Downloaded file is not a valid .blend.")
        clear_world_downloads_folder()
        current_download_task = None
        return None

    # 4) Snapshot existing datablocks (before append)
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

    # 5) Append the downloaded blend
    result, appended_scene_name = append_scene_from_blend(current_download_task.local_blend_path)
    if result != {'FINISHED'} or not appended_scene_name:
        bail_to_idle("Append failed or produced no scene.")
        clear_world_downloads_folder()
        current_download_task = None
        return None

    # 6) Record scene metadata
    scene["appended_scene_name"] = appended_scene_name
    scene["world_blend_path"]    = current_download_task.local_blend_path

    # 7) Compute which datablocks arrived with that append
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

    
    try:
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    except Exception:
        pass

    # 9) Switch into the appended scene and launch the game
    def check_scene():
        scene_obj = bpy.data.scenes.get(appended_scene_name)
        if not scene_obj:
            print("Waiting for appended scene…")
            return 1.0

        bpy.context.window.scene = scene_obj
        scene_obj.ui_current_mode = "GAME"

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

    # 10) Done—clear the task so we don’t re-enter
    current_download_task = None
    return None


def explore_icon_handler(context, download_code):
    """
    Called when the Explore Icon is clicked in the detail overlay.
    Kicks off the download task and the polling timer.
    """
    global current_download_task

    # Keep original scene name if you need it elsewhere
    try:
        context.window_manager['original_scene'] = context.window.scene.name
    except Exception:
        pass

    reset_download_progress()
    context.scene.download_progress = 0.0

    if download_code:
        current_download_task = DownloadTask(download_code)
        threading.Thread(target=current_download_task.run, daemon=True).start()
        bpy.app.timers.register(timer_finish_download, first_interval=0.1)
    else:
        print("No download code provided.")
