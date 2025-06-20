# File: update_addon.py
"""
Operator to fetch, download, and install a new version of the add-on in-place,
replacing the entire add-on at the top level.
"""

import os
import tempfile
import threading
import zipfile
import shutil
import requests

import bpy
import addon_utils
from bpy.props import StringProperty, FloatProperty

from .Exp_UI.main_config       import BASE_URL
from .Exp_UI.version_info      import CURRENT_VERSION
from .Exp_UI.prefs_persistence import save_prefs_to_json, load_prefs_from_json

# ─── Figure out where this add-on lives ───────────────────────────────────
THIS_DIR     = os.path.dirname(__file__)
ADDON_FOLDER = THIS_DIR
ADDON_NAME   = os.path.basename(ADDON_FOLDER)

# ─── Temp + staging paths ──────────────────────────────────────────────────
TMP_DIR    = tempfile.gettempdir()
ZIP_PATH   = os.path.join(TMP_DIR, f"{ADDON_NAME}_update.zip")
STAGE_DIR  = os.path.join(TMP_DIR, f"{ADDON_NAME}_stage")
BACKUP_DIR = os.path.join(TMP_DIR, f"{ADDON_NAME}_backup")


class WEBAPP_OT_UpdateAddon(bpy.types.Operator):
    bl_idname = "webapp.update_addon"
    bl_label  = "Update Exploratory Add-on"
    bl_options = {'REGISTER', 'UNDO'}

    # props filled in invoke()
    version:      StringProperty()
    download_url: StringProperty()
    update_type:  StringProperty()
    changelog:    StringProperty()

    # internal
    progress:   FloatProperty(default=0.0)
    _thread     = None
    _timer      = None
    _stage_done = False
    _error      = None

    def invoke(self, context, event):
        try:
            resp = requests.get(f"{BASE_URL}/api/addon_version", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data.get('success'):
                self.report({'ERROR'}, data.get('message', 'Failed to fetch update info'))
                return {'CANCELLED'}

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
        layout = self.layout

        # ─── Update Type ─────────────────────────────────────────────────────
        box = layout.box()
        # map your internal update_type to a more user-friendly label
        up_type = self.update_type.lower()
        friendly = "Full Update" if up_type == "full" else "Patch / Bug Fix"
        box.label(text=f"Update Type: {friendly}")

        # ─── New Version ──────────────────────────────────────────────────────
        box = layout.box()
        box.label(text=f"New Version: {self.version}")

        # ─── Changelog ────────────────────────────────────────────────────────
        if self.changelog:
            box = layout.box()
            box.label(text="Changelog:")
            for line in self.changelog.splitlines():
                box.label(text=f"• {line}")

        # ─── Instructions ────────────────────────────────────────────────────
        box = layout.box()
        box.label(text="The add-on update may take a few minutes.")
        box.label(text="Please wait until the update is complete before closing Blender.")
        box.label(text="After the update finishes, restart Blender to initialize the new version.")

    def execute(self, context):
        # 1) persist prefs & disable
        save_prefs_to_json()
        try:
            bpy.ops.preferences.addon_disable(module=ADDON_NAME)
        except:
            addon_utils.disable(ADDON_NAME)

        # 2) background download & unzip
        self._error      = None
        self._stage_done = False
        self._thread     = threading.Thread(target=self._stage_update, daemon=True)
        self._thread.start()

        # 3) modal progress
        wm = context.window_manager
        wm.progress_begin(0, 100)
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            context.window_manager.progress_update(int(self.progress * 100))

            if not self._thread.is_alive() and not self._stage_done:
                self._stage_done = True

            if self._stage_done:
                try:
                    # backup old
                    shutil.rmtree(BACKUP_DIR, ignore_errors=True)
                    shutil.move(ADDON_FOLDER, BACKUP_DIR)

                    # detect single root
                    contents = [e for e in os.listdir(STAGE_DIR) if not e.startswith('.')]
                    if len(contents)==1 and os.path.isdir(os.path.join(STAGE_DIR, contents[0])):
                        root = os.path.join(STAGE_DIR, contents[0])
                    else:
                        root = STAGE_DIR

                    # clear & repopulate
                    shutil.rmtree(ADDON_FOLDER, ignore_errors=True)
                    os.makedirs(ADDON_FOLDER, exist_ok=True)
                    for name in os.listdir(root):
                        shutil.move(os.path.join(root, name),
                                    os.path.join(ADDON_FOLDER, name))

                    # cleanup
                    shutil.rmtree(STAGE_DIR, ignore_errors=True)
                    os.remove(ZIP_PATH)

                except Exception as e:
                    self._error = str(e)

                # reload, re-enable, restore prefs
                addon_utils.modules(refresh=True)
                bpy.utils.refresh_script_paths()
                if not self._error:
                    try:
                        bpy.ops.preferences.addon_enable(module=ADDON_NAME)
                    except:
                        addon_utils.enable(ADDON_NAME)
                    load_prefs_from_json()

                    bpy.ops.webapp.refresh_version()
                    
                    self.report({'INFO'}, f"Updated to {self.version}. Restart Blender to finalize.")
                else:
                    self.report({'ERROR'}, f"Update failed: {self._error}")

                context.window_manager.progress_end()
                return {'FINISHED'} if not self._error else {'CANCELLED'}

        return {'PASS_THROUGH'}

    def _stage_update(self):
        try:
            r = requests.get(self.download_url, stream=True, timeout=30)
            r.raise_for_status()
            total = int(r.headers.get('content-length',0)) or 1
            dl = 0
            os.makedirs(TMP_DIR, exist_ok=True)
            with open(ZIP_PATH,'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
                    dl += len(chunk)
                    self.progress = dl/total

            # unzip
            shutil.rmtree(STAGE_DIR, ignore_errors=True)
            os.makedirs(STAGE_DIR, exist_ok=True)
            with zipfile.ZipFile(ZIP_PATH,'r') as zf:
                zf.extractall(STAGE_DIR)

        except Exception as e:
            self._error = str(e)

    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        return {'CANCELLED'}


class WEBAPP_OT_RefreshVersion(bpy.types.Operator):
    bl_idname = "webapp.refresh_version"
    bl_label  = "Check for Update"
    bl_description = "Fetch the latest add-on version"

    def execute(self, context):
        from .Exp_UI.exp_api import update_latest_version_cache, get_cached_latest_version
        update_latest_version_cache()
        latest = get_cached_latest_version()
        if not latest:
            self.report({'ERROR'}, "Failed to fetch latest version.")
        elif latest == CURRENT_VERSION:
            self.report({'INFO'}, f"Up to date ({CURRENT_VERSION}).")
        else:
            self.report({'INFO'}, f"New version available: {latest}")
        return {'FINISHED'}
