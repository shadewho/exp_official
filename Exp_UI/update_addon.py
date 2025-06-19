# Exploratory/Exp_UI/update_addon.py
"""
Operator to fetch, download, and install a new version of the add-on in-place.
"""

import bpy
import addon_utils
import threading
import requests
import os
import shutil
import zipfile
import tempfile
from bpy.props import StringProperty, FloatProperty
from .main_config import BASE_URL, ADDON_FOLDER
from .version_info import CURRENT_VERSION
from .prefs_persistence import save_prefs_to_json, load_prefs_from_json

# --- Temp and staging paths ---
TMP_DIR    = tempfile.gettempdir()
ZIP_PATH   = os.path.join(TMP_DIR, "{ADDON_FOLDER}_update.zip")
STAGE_DIR  = os.path.join(TMP_DIR, "{ADDON_FOLDER}_stage")
BACKUP_DIR = os.path.join(TMP_DIR, "{ADDON_FOLDER}_backup")

class WEBAPP_OT_UpdateAddon(bpy.types.Operator):
    bl_idname = "webapp.update_addon"
    bl_label = "Update Add-on"
    bl_options = {'REGISTER', 'UNDO'}

    version:      StringProperty()
    download_url: StringProperty()
    update_type:  StringProperty()
    changelog:    StringProperty()

    progress:     FloatProperty(default=0.0)
    _thread       = None
    _timer        = None
    _stage_done   = False
    _error        = None

    def invoke(self, context, event):
        try:
            resp = requests.get(f"{BASE_URL}/api/addon_version", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data.get('success'):
                self.report({'ERROR'}, data.get('message', 'Failed to fetch update info'))
                return {'CANCELLED'}

            # Capture into locals so thread doesn't rely on RNA props
            self.version      = data['version_string']
            self.download_url = data['download_url']
            self.update_type  = data.get('update_type', 'full')
            self.changelog    = data.get('changelog', '')

            if self.version == CURRENT_VERSION:
                self.report({'INFO'}, f"Already on {CURRENT_VERSION}")
                return {'CANCELLED'}

            return context.window_manager.invoke_props_dialog(self, width=400)
        except Exception as e:
            self.report({'ERROR'}, f"Error fetching update info: {e}")
            return {'CANCELLED'}

    def draw(self, context):
        box = self.layout.box()
        box.label(text=f"{self.update_type.capitalize()} → {self.version}", icon='FILE_TICK')
        if self.changelog:
            box.separator()
            box.label(text="Changelog:")
            for line in self.changelog.splitlines():
                box.label(text=f"• {line}")
        box.separator()
        box.label(text="⏳ Please wait…", icon='TIME')

    def execute(self, context):
        # Save prefs then disable add-on
        save_prefs_to_json()
        bpy.ops.preferences.addon_disable(module=os.path.basename(ADDON_FOLDER))

        # Start staging thread
        self._error = None
        self._stage_done = False
        self._thread = threading.Thread(
            target=self._stage_update,
            args=(self.download_url,),
            daemon=True
        )
        self._thread.start()

        # Modal timer to track progress and perform swap
        wm = context.window_manager
        wm.progress_begin(0, 100)
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            # update progress
            context.window_manager.progress_update(int(self.progress * 100))

            # wait for staging to complete
            if not self._thread.is_alive() and not self._stage_done:
                # thread finished staging
                self._stage_done = True

            if self._stage_done:
                # atomic swap on main thread
                try:
                    shutil.rmtree(BACKUP_DIR, ignore_errors=True)
                    shutil.move(ADDON_FOLDER, BACKUP_DIR)
                    shutil.move(STAGE_DIR, ADDON_FOLDER)
                except Exception as e:
                    self._error = str(e)

                # reload and re-enable
                addon_utils.modules(refresh=True)
                bpy.utils.refresh_script_paths()
                if self._error:
                    self.report({'ERROR'}, f"Update failed: {self._error}")
                else:
                    try:
                        bpy.ops.preferences.addon_enable(module=os.path.basename(ADDON_FOLDER))
                    except:
                        addon_utils.enable(os.path.basename(ADDON_FOLDER))
                    load_prefs_from_json()
                    self.report({'INFO'}, f"Updated to {self.version}. Restart Blender to finalize.")

                context.window_manager.progress_end()
                return {'FINISHED'} if not self._error else {'CANCELLED'}

        return {'PASS_THROUGH'}

    def _stage_update(self, download_url):
        """
        Background thread: download and unzip into STAGE_DIR, update self.progress.
        """
        try:
            # download zip
            resp = requests.get(download_url, stream=True, timeout=30)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0)) or 1
            downloaded = 0
            os.makedirs(TMP_DIR, exist_ok=True)
            with open(ZIP_PATH, 'wb') as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    self.progress = downloaded / total

            # extract to STAGE_DIR
            shutil.rmtree(STAGE_DIR, ignore_errors=True)
            os.makedirs(STAGE_DIR, exist_ok=True)
            with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
                zf.extractall(STAGE_DIR)

            # cleanup zip
            os.remove(ZIP_PATH)
        except Exception as e:
            self._error = str(e)

    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        return {'CANCELLED'}


class WEBAPP_OT_RefreshVersion(bpy.types.Operator):
    bl_idname = "webapp.refresh_version"
    bl_label = "Check for Update"
    bl_description = "Fetch the latest add-on version"

    def execute(self, context):
        from .exp_api import update_latest_version_cache, get_cached_latest_version
        update_latest_version_cache()
        latest = get_cached_latest_version()
        if not latest:
            self.report({'ERROR'}, "Failed to fetch latest version.")
        elif latest == CURRENT_VERSION:
            self.report({'INFO'}, f"Up to date ({CURRENT_VERSION}).")
        else:
            self.report({'INFO'}, f"New version available: {latest}")
        return {'FINISHED'}
