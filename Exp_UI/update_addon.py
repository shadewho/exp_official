# Exp_UI/update_addon.py
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

# --- Derive real addon root (one level up from this file) ---
THIS_DIR     = os.path.dirname(__file__)
ADDON_FOLDER = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
ADDON_NAME   = os.path.basename(ADDON_FOLDER)

# --- Temp paths ---
TMP_DIR    = tempfile.gettempdir()
ZIP_PATH   = os.path.join(TMP_DIR, f"{ADDON_NAME}_update.zip")
STAGE_DIR  = os.path.join(TMP_DIR, f"{ADDON_NAME}_stage")
BACKUP_DIR = os.path.join(TMP_DIR, f"{ADDON_NAME}_backup")


class WEBAPP_OT_UpdateAddon(bpy.types.Operator):
    bl_idname = "webapp.update_addon"
    bl_label = "Update Add-on"
    bl_options = {'REGISTER', 'UNDO'}

    # Popup props
    version:      StringProperty()
    download_url: StringProperty()
    update_type:  StringProperty()
    changelog:    StringProperty()

    # Internal tracking
    progress: FloatProperty(default=0.0)
    _thread = None
    _timer  = None
    _error  = None

    def invoke(self, context, event):
        try:
            resp = requests.get(f"{BASE_URL}/api/addon_version", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data.get('success'):
                self.report({'ERROR'}, data.get('message','Failed to fetch update info'))
                return {'CANCELLED'}

            self.version      = data['version_string']
            self.download_url = data['download_url']
            self.update_type  = data.get('update_type','full')
            self.changelog    = data.get('changelog','')

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
        wm = context.window_manager
        wm.progress_begin(0, 100)
        self._error = None
        self._thread = threading.Thread(target=self._download_and_install, daemon=True)
        self._thread.start()
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            pct = int(self.progress * 100)
            context.window_manager.progress_update(pct)
            if not self._thread.is_alive():
                context.window_manager.progress_end()
                if self._error:
                    self.report({'ERROR'}, f"Update failed: {self._error}")
                    return {'CANCELLED'}
                self.report({'INFO'}, f"Updated to {self.version}. Restart Blender to finalize.")
                return {'FINISHED'}
        return {'PASS_THROUGH'}

    def _download_and_install(self):
        try:
            # Download the ZIP
            resp = requests.get(self.download_url, stream=True, timeout=30)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0)) or None
            downloaded = 0
            with open(ZIP_PATH, 'wb') as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    self.progress = (downloaded / total) if total else 0.0

            # Prepare staging area
            shutil.rmtree(STAGE_DIR, ignore_errors=True)
            os.makedirs(STAGE_DIR, exist_ok=True)

            # Extract all
            with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
                zf.extractall(STAGE_DIR)

            # Detect single top-level folder
            entries = [e for e in os.listdir(STAGE_DIR) if not e.startswith('.')]
            if len(entries) == 1 and os.path.isdir(os.path.join(STAGE_DIR, entries[0])):
                extracted_root = os.path.join(STAGE_DIR, entries[0])
            else:
                extracted_root = STAGE_DIR

            # Disable current addon
            try:
                bpy.ops.preferences.addon_disable(module=ADDON_NAME)
            except Exception:
                addon_utils.disable(ADDON_NAME)

            # Backup old code
            shutil.rmtree(BACKUP_DIR, ignore_errors=True)
            shutil.move(ADDON_FOLDER, BACKUP_DIR)

            # Re-create addon folder and move in new files
            os.makedirs(ADDON_FOLDER, exist_ok=True)
            for item in os.listdir(extracted_root):
                src = os.path.join(extracted_root, item)
                dst = os.path.join(ADDON_FOLDER, item)
                shutil.move(src, dst)

            # Reload & re-enable
            addon_utils.modules(refresh=True)
            bpy.utils.refresh_script_paths()
            try:
                bpy.ops.preferences.addon_enable(module=ADDON_NAME)
            except Exception:
                addon_utils.enable(ADDON_NAME)

            # Clean up temp
            shutil.rmtree(STAGE_DIR, ignore_errors=True)
            os.remove(ZIP_PATH)
            shutil.rmtree(BACKUP_DIR, ignore_errors=True)

        except Exception as e:
            self._error = str(e)
            # Attempt restore
            try:
                shutil.rmtree(ADDON_FOLDER, ignore_errors=True)
                if os.path.isdir(BACKUP_DIR):
                    shutil.move(BACKUP_DIR, ADDON_FOLDER)
                addon_utils.modules(refresh=True)
                bpy.utils.refresh_script_paths()
                try:
                    bpy.ops.preferences.addon_enable(module=ADDON_NAME)
                except:
                    addon_utils.enable(ADDON_NAME)
            except:
                pass

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
        return {'FINITE'}
