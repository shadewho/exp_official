import os
import tempfile
import shutil
import zipfile
import requests

import bpy
import addon_utils
from bpy.props import StringProperty

from .Exp_UI.main_config import BASE_URL
from .Exp_UI.version_info import CURRENT_VERSION
from .Exp_UI.prefs_persistence import write_prefs
# ─── Paths ─────────────────────────────────────────────────────────────────
THIS_DIR     = os.path.dirname(__file__)
ADDON_FOLDER = THIS_DIR
ADDON_NAME   = os.path.basename(ADDON_FOLDER)

TMP_DIR    = tempfile.gettempdir()
ZIP_PATH   = os.path.join(TMP_DIR, f"{ADDON_NAME}_update.zip")
STAGE_DIR  = os.path.join(TMP_DIR, f"{ADDON_NAME}_stage")
BACKUP_DIR = os.path.join(TMP_DIR, f"{ADDON_NAME}_backup")

# ─── Update Operator ───────────────────────────────────────────────────────
class WEBAPP_OT_UpdateAddon(bpy.types.Operator):
    bl_idname = "webapp.update_addon"
    bl_label  = "Update Exploratory Add-on"
    bl_options = {'REGISTER'}

    version:      StringProperty()
    download_url: StringProperty()
    update_type:  StringProperty()
    changelog:    StringProperty()

    def invoke(self, context, event):
        try:
            # Fetch metadata
            resp = requests.get(f"{BASE_URL}/api/addon_version", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data.get('success'):
                self.report({'ERROR'}, data.get('message', 'Failed to fetch update info'))
                return {'CANCELLED'}

            # Populate properties
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
        # Update type
        box = layout.box()
        friendly = "Full Update" if self.update_type.lower() == "full" else "Patch / Bug Fix"
        box.label(text=f"Update Type: {friendly}")
        # Version
        box = layout.box()
        box.label(text=f"New Version: {self.version}")
        # Changelog
        if self.changelog:
            box = layout.box()
            box.label(text="Changelog:")
            for line in self.changelog.splitlines():
                box.label(text=f"• {line}")
        # Instructions
        box = layout.box()
        box.label(text="This may take a few minutes.")
        box.label(text="Restart Blender when update is complete.")
        box.label(text="If the update fails, go to https://exploratory.online/ for the latest version.")

    def execute(self, context):
        # 0) Save current prefs
        write_prefs()
        try:
            # Download ZIP
            os.makedirs(TMP_DIR, exist_ok=True)
            r = requests.get(self.download_url, stream=True, timeout=30)
            r.raise_for_status()
            with open(ZIP_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract to staging
            shutil.rmtree(STAGE_DIR, ignore_errors=True)
            os.makedirs(STAGE_DIR, exist_ok=True)
            with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
                members = zf.namelist()
                root = members[0].split('/')[0] + '/'
                for member in members:
                    rel = member[len(root):]
                    if not rel:
                        continue
                    dest = os.path.join(STAGE_DIR, rel)
                    if member.endswith('/'):
                        os.makedirs(dest, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        with zf.open(member) as src, open(dest, 'wb') as dst:
                            dst.write(src.read())

            # Backup current
            shutil.rmtree(BACKUP_DIR, ignore_errors=True)
            shutil.move(ADDON_FOLDER, BACKUP_DIR)
            # Deploy new version
            shutil.move(STAGE_DIR, ADDON_FOLDER)

        except Exception as e:
            # Rollback on error
            if os.path.isdir(BACKUP_DIR):
                if os.path.isdir(ADDON_FOLDER):
                    shutil.rmtree(ADDON_FOLDER)
                shutil.move(BACKUP_DIR, ADDON_FOLDER)
            self.report({'ERROR'}, f"Update failed: {e}")
            return {'CANCELLED'}

        finally:
            # Cleanup
            for path in (ZIP_PATH, STAGE_DIR):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    elif os.path.exists(path):
                        os.remove(path)
                except:
                    pass

        # Refresh Blender to pick up new code
        addon_utils.modules(refresh=True)
        bpy.utils.refresh_script_paths()
        self.report({'INFO'}, f"Updated to {self.version}. Restart Blender to complete.")
        return {'FINISHED'}


# ─── Refresh Operator ──────────────────────────────────────────────────────
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
