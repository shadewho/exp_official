"""
Development Refresh Panel
Allows quick reload of addon during development without restarting Blender.
Set ENABLED = False before shipping to hide this panel.
"""

import bpy
import os
import shutil
import stat
import sys
import importlib


# ============================================================================
# ENABLE/DISABLE TOGGLE - Set to False before shipping!
# ============================================================================
ENABLED = True


# ============================================================================
# PATHS CONFIGURATION
# ============================================================================
SRC = r"C:\Users\spenc\Desktop\Exploratory\addons\Exploratory"
DEST = os.path.expandvars(r"%APPDATA%\Blender Foundation\Blender\5.0\scripts\addons\Exploratory")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _rm_error(func, path, exc_info):
    """
    If rmtree fails because the file is read-only, clear the read-only bit and retry.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _copy_file_safe(src_file, dest_file):
    """
    Copy a single file, handling locked files gracefully.
    Returns (success: bool, was_locked: bool)
    """
    try:
        # Create destination directory if needed
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)

        # Try to copy/overwrite
        shutil.copy2(src_file, dest_file)
        return True, False
    except PermissionError:
        # File is locked by Blender (fonts, assets, etc.)
        return False, True
    except Exception:
        # Other error
        return False, False


def copy_addon_files():
    """
    Copy addon files from source to Blender's addon folder.
    Handles Windows file locking gracefully by skipping locked files.
    Returns (success: bool, message: str)
    """
    # Directories and files to ignore
    ignore_patterns = {
        '.git',
        '.gitignore',
        'combine.py',
        '__pycache__',
        'CLAUDE.md',
        'LICENSE.txt',
        '.vscode',
        '.idea',
        '.DS_Store',  # macOS
        '.venv312',   # Python 3.12 venv for PyTorch training
    }

    # File extensions to ignore
    ignore_extensions = {'.pyc', '.md', '.blend1', '.pyo'}

    try:
        # Try to remove old installation (clean slate)
        if os.path.isdir(DEST):
            print(f"[DEV REFRESH] Attempting clean removal of {DEST}")
            try:
                shutil.rmtree(DEST, onerror=_rm_error)
                print(f"[DEV REFRESH] ✓ Clean removal successful")
            except Exception as e:
                print(f"[DEV REFRESH] ⚠ Clean removal failed (files locked), using incremental copy")
                print(f"[DEV REFRESH]   Reason: {str(e)[:100]}")

        # Copy files (works whether DEST exists or not)
        print(f"[DEV REFRESH] Copying {SRC} → {DEST}")

        copied_files = 0
        locked_files = []
        failed_files = []

        # Walk through source directory
        for root, dirs, files in os.walk(SRC):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_patterns]

            # Calculate relative path
            rel_path = os.path.relpath(root, SRC)
            dest_dir = os.path.join(DEST, rel_path) if rel_path != '.' else DEST

            # Copy each file
            for file in files:
                # Skip ignored files and extensions
                if file in ignore_patterns:
                    continue
                if any(file.endswith(ext) for ext in ignore_extensions):
                    continue

                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)

                success, was_locked = _copy_file_safe(src_file, dest_file)

                if success:
                    copied_files += 1
                elif was_locked:
                    locked_files.append(os.path.relpath(dest_file, DEST))
                else:
                    failed_files.append(os.path.relpath(dest_file, DEST))

        # Report results
        print(f"[DEV REFRESH] ✓ Copied {copied_files} files")

        if locked_files:
            print(f"[DEV REFRESH] ⚠ Skipped {len(locked_files)} locked files (Blender is using them):")
            for f in locked_files[:5]:  # Show first 5
                print(f"[DEV REFRESH]   - {f}")
            if len(locked_files) > 5:
                print(f"[DEV REFRESH]   ... and {len(locked_files) - 5} more")

        if failed_files:
            print(f"[DEV REFRESH] ✗ Failed to copy {len(failed_files)} files")
            for f in failed_files[:3]:
                print(f"[DEV REFRESH]   - {f}")

        # Success if we copied files (locked files are okay to skip)
        if copied_files > 0:
            msg = f"Copied {copied_files} files"
            if locked_files:
                msg += f" ({len(locked_files)} locked files skipped)"
            return True, msg
        else:
            return False, "No files were copied"

    except Exception as e:
        return False, f"Copy failed: {str(e)}"


def reload_addon_modules():
    """
    Files have been copied - now just inform user to restart Blender.
    Hot reloading causes UI state issues in Blender, so a quick restart is more reliable.
    Returns (success: bool, message: str)
    """
    print(f"[DEV REFRESH] Files updated successfully!")
    print(f"[DEV REFRESH] Restart Blender to load changes (File → Quit)")
    print(f"[DEV REFRESH] Your scene will auto-save and reopen")

    return True, "Files copied - restart Blender to apply"


# ============================================================================
# OPERATOR
# ============================================================================

class EXP_OT_dev_refresh(bpy.types.Operator):
    """Refresh addon from source folder and reload all modules"""
    bl_idname = "exp.dev_refresh"
    bl_label = "Refresh Addon"
    bl_options = {'REGISTER'}

    def execute(self, context):
        print("\n" + "="*60)
        print("[DEV REFRESH] Starting addon refresh...")
        print("="*60)

        # Step 1: Copy files
        success, message = copy_addon_files()
        if not success:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        print(f"[DEV REFRESH] ✓ {message}")

        # Step 2: Reload modules
        success, message = reload_addon_modules()
        if not success:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        print(f"[DEV REFRESH] ✓ {message}")

        # Step 3: Refresh UI
        for area in context.screen.areas:
            area.tag_redraw()

        print("[DEV REFRESH] ✓ UI refreshed")
        print("="*60)
        print("[DEV REFRESH] Addon refresh complete!")
        print("="*60 + "\n")

        self.report({'INFO'}, "Addon refreshed successfully!")
        return {'FINISHED'}


# ============================================================================
# PANEL
# ============================================================================

class EXP_PT_dev_refresh(bpy.types.Panel):
    """Development refresh panel"""
    bl_label = "Dev Refresh"
    bl_idname = "EXP_PT_dev_refresh"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Exploratory'

    def draw(self, context):
        layout = self.layout

        # Warning box
        box = layout.box()
        box.label(text="Development Mode", icon='ERROR')
        box.label(text="Disable before shipping!")

        # Refresh button
        layout.separator()
        row = layout.row()
        row.scale_y = 2.0
        row.operator("exp.dev_refresh", icon='FILE_REFRESH')

        # What it does
        layout.separator()
        box = layout.box()
        box.label(text="Workflow:", icon='INFO')
        box.label(text="1. Click Refresh")
        box.label(text="2. Restart Blender")
        box.label(text="3. Changes applied!")

        layout.separator()
        box = layout.box()
        box.label(text="5x faster than old way!", icon='TIME')

        # Paths
        layout.separator()
        box = layout.box()
        box.label(text="Source:")
        box.label(text="Desktop/Exploratory", icon='FOLDER_REDIRECT')
        box.label(text="Destination:")
        box.label(text="Blender/addons", icon='BLENDER')


# ============================================================================
# REGISTRATION
# ============================================================================
# Note: Panel is now integrated into Developer Tools panel (dev_panel.py)
# Only register the operator here

classes = (
    EXP_OT_dev_refresh,
    # EXP_PT_dev_refresh,  # Panel removed - button now in Developer Tools
)


def register():
    if ENABLED:
        for cls in classes:
            bpy.utils.register_class(cls)
        print("[DEV REFRESH] Development refresh operator enabled")


def unregister():
    if ENABLED:
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)
        print("[DEV REFRESH] Development refresh operator disabled")


if __name__ == "__main__":
    register()
