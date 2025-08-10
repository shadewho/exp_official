import os
import tempfile
import shutil
import zipfile
import requests
import time
import sqlite3
import contextlib
import bpy
import addon_utils
from bpy.props import StringProperty

from .Exp_UI.main_config import BASE_URL, THUMBNAIL_CACHE_FOLDER
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


# --- helpers to avoid touching locked cache files ---------------------------
def _is_under(path: str, root: str) -> bool:
    try:
        path = os.path.realpath(path)
        root = os.path.realpath(root)
        return os.path.commonpath([path, root]) == root
    except Exception:
        return False

def _delete_tree_excluding(root_dir: str, exclude_prefixes: list[str]):
    """
    Delete everything under root_dir EXCEPT any path that is inside one
    of exclude_prefixes. Works even if exclude paths don't exist.
    """
    if not os.path.isdir(root_dir):
        return
    for entry in os.scandir(root_dir):
        sp = entry.path
        if any(_is_under(sp, ex) and os.path.exists(ex) for ex in exclude_prefixes):
            continue
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(sp, ignore_errors=True)
            else:
                try:
                    os.remove(sp)
                except PermissionError:
                    # Windows sometimes needs an attribute flip
                    try:
                        os.chmod(sp, 0o666)
                        os.remove(sp)
                    except Exception:
                        pass
        except Exception:
            pass

def _copy_tree_overwrite(src_dir: str, dst_dir: str):
    """
    Recursively copy src_dir into dst_dir, overwriting files.
    Creates directories as needed. Doesn’t delete anything.
    """
    for root, dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        target_root = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        os.makedirs(target_root, exist_ok=True)
        for d in dirs:
            os.makedirs(os.path.join(target_root, d), exist_ok=True)
        for f in files:
            src_p = os.path.join(root, f)
            dst_p = os.path.join(target_root, f)
            try:
                shutil.copy2(src_p, dst_p)
            except PermissionError:
                # last-ditch Windows fix
                try:
                    if os.path.exists(dst_p):
                        os.chmod(dst_p, 0o666)
                    shutil.copy2(src_p, dst_p)
                except Exception:
                    raise


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



            # Quiesce DB activity to reduce locks
            with contextlib.suppress(Exception):
                _quiesce_db_for_update(max_wait_seconds=6.0)

            # 2) Define what we refuse to touch during update (DB/cache)
            EXCLUDES = [THUMBNAIL_CACHE_FOLDER]

            # 3) Make a quick best-effort backup ZIP (excluding cache)
            try:
                backup_zip = os.path.join(TMP_DIR, f"{ADDON_NAME}_backup.zip")
                with zipfile.ZipFile(backup_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
                    for root, dirs, files in os.walk(ADDON_FOLDER):
                        # skip excluded subtree(s)
                        if any(_is_under(root, ex) for ex in EXCLUDES):
                            continue
                        for f in files:
                            full = os.path.join(root, f)
                            if any(_is_under(full, ex) for ex in EXCLUDES):
                                continue
                            arc = os.path.relpath(full, os.path.dirname(ADDON_FOLDER))
                            z.write(full, arcname=arc)
            except Exception:
                pass  # backup is best-effort

            # 4) Nuke everything inside the add-on folder except the cache dir
            _delete_tree_excluding(ADDON_FOLDER, EXCLUDES)

            # 5) Copy staged files over the top (overwrite), cache dir remains untouched
            _copy_tree_overwrite(STAGE_DIR, ADDON_FOLDER)


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
        from .Exp_UI.version_info import update_latest_version_cache, get_cached_latest_version
        update_latest_version_cache()
        latest = get_cached_latest_version()
        if not latest:
            self.report({'ERROR'}, "Failed to fetch latest version.")
        elif latest == CURRENT_VERSION:
            self.report({'INFO'}, f"Up to date ({CURRENT_VERSION}).")
        else:
            self.report({'INFO'}, f"New version available: {latest}")
        return {'FINISHED'}


def _quiesce_db_for_update(max_wait_seconds: float = 6.0):
    """
    Stop background DB use and release SQLite locks so we can move/delete
    the add-on folder on Windows.
    Safe to call even if the worker was never started.
    """
    db_path = os.path.join(THUMBNAIL_CACHE_FOLDER, "cache.db")
    wal = db_path + "-wal"
    shm = db_path + "-shm"

    # 1) Stop the cache worker (if present) and wait for it to exit
    try:
        from .Exp_UI.cache_system import preload as _preload
        worker = getattr(_preload, "_worker", None)
        with contextlib.suppress(Exception):
            _preload.stop_cache_worker()
        if worker is not None:
            deadline = time.time() + max_wait_seconds
            while worker.is_alive() and time.time() < deadline:
                time.sleep(0.1)
            with contextlib.suppress(Exception):
                worker.join(timeout=0.2)
    except Exception:
        pass  # not fatal

    # 2) Force-close any other SQLite connections in Blender that have this path open
    try:
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, sqlite3.Connection):
                try:
                    if obj.in_transaction:
                        obj.rollback()
                    obj.close()
                except Exception:
                    pass
    except Exception:
        pass

    # 3) If the DB exists, checkpoint WAL so SQLite drops -wal/-shm
    if os.path.exists(db_path):
        for _ in range(6):
            try:
                with sqlite3.connect(db_path, timeout=0.5, check_same_thread=False) as conn:
                    conn.execute("PRAGMA busy_timeout = 500;")
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                break
            except Exception:
                time.sleep(0.25)

    # 4) Try to delete DB artifacts (retry briefly)
    targets = [wal, shm, db_path]
    deadline = time.time() + max_wait_seconds
    while targets and time.time() < deadline:
        remaining = []
        for p in targets:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except PermissionError:
                remaining.append(p)
            except Exception:
                pass
        if not remaining:
            break
        targets = remaining
        time.sleep(0.2)


