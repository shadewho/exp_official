# Exploratory/Exp_UI/download_and_explore/cleanup.py

import os
import bpy

from ..main_config import (
    WORLD_DOWNLOADS_FOLDER

)
from bpy.app.handlers import persistent


#remove the world temp file and appended scene - called in game modal cancel()
#i dont think the object/mesh deltion flow is neccessary here
def cleanup_downloaded_worlds():

    # Try to get the active scene and the stored appended scene name.
    try:
        scene = bpy.context.scene
        appended_scene_name = scene.get("appended_scene_name")
    except ReferenceError:
        scene = None
        appended_scene_name = None

    if appended_scene_name and appended_scene_name in bpy.data.scenes:
        appended_scene = bpy.data.scenes[appended_scene_name]

        # Iterate over a copy of the appended scene's objects.
        for obj in list(appended_scene.objects):
            # Save the object's name before removal.
            try:
                obj_name = obj.name
            except ReferenceError:
                obj_name = "<unknown>"
            
            # If the object has data (e.g. mesh), store it and its name.
            mesh_data = None
            mesh_data_name = None
            if hasattr(obj, "data") and obj.data is not None:
                try:
                    mesh_data = obj.data
                    mesh_data_name = mesh_data.name
                except ReferenceError:
                    mesh_data = None
                    mesh_data_name = "<unknown>"

            # Remove the object.
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception as e:
                print(f"Error removing object {obj_name}: {e}")

            # If the object had mesh data and no one is using it, remove it.
            if mesh_data is not None:
                try:
                    if mesh_data.users == 0:
                        bpy.data.meshes.remove(mesh_data)
                except Exception as e:
                    # If the data block is already removed, we skip.
                    print(f"Error removing mesh data {mesh_data_name}: {e}")

        # Now remove the appended scene.
        try:
            bpy.data.scenes.remove(appended_scene)
        except Exception as e:
            print(f"Error removing scene {appended_scene_name}: {e}")
    else:
        print("No appended scene found to remove.")

    # Delete all files in the WORLD_DOWNLOADS_FOLDER.
    if os.path.isdir(WORLD_DOWNLOADS_FOLDER):
        for filename in os.listdir(WORLD_DOWNLOADS_FOLDER):
            file_path = os.path.join(WORLD_DOWNLOADS_FOLDER, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        print("WORLD_DOWNLOADS_FOLDER not found.")

    # Remove the custom properties from the scene if it is still valid.
    if scene is not None:
        try:
            for key in ("appended_scene_name", "world_blend_path"):
                if key in scene:
                    del scene[key]
        except ReferenceError:
            pass
        
def cleanup_downloaded_datablocks():
    """
    Force-remove all datablocks and geometry that were appended with the last scene,
    without regard to user count or fake users.
    Called after cleanup_downloaded_worlds() and revert_to_original_scene().
    """
    scene = bpy.context.scene
    appended = scene.get("appended_datablocks", {})
    if not appended:
        return

    # 1) Actions
    for name in appended.get("actions", []):
        act = bpy.data.actions.get(name)
        if act:
            try:
                act.user_clear()
                act.use_fake_user = False
                bpy.data.actions.remove(act)
            except Exception as e:
                print(f"Error removing action '{name}': {e}")

    # 2) Images
    for name in appended.get("images", []):
        img = bpy.data.images.get(name)
        if img:
            try:
                img.user_clear()
                img.use_fake_user = False
                if img.packed_file:
                    img.unpack(method='REMOVE')
                bpy.data.images.remove(img)
            except Exception as e:
                print(f"Error removing image '{name}': {e}")

    # 3) Sounds
    for name in appended.get("sounds", []):
        snd = bpy.data.sounds.get(name)
        if snd:
            try:
                snd.user_clear()
                snd.use_fake_user = False
                bpy.data.sounds.remove(snd)
            except Exception as e:
                print(f"Error removing sound '{name}': {e}")

    # 4) Objects
    for name in appended.get("objects", []):
        obj = bpy.data.objects.get(name)
        if obj:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception as e:
                print(f"Error removing object '{name}': {e}")

    # 5) Mesh datablocks
    for name in appended.get("meshes", []):
        mesh = bpy.data.meshes.get(name)
        if mesh:
            try:
                mesh.user_clear()
                mesh.use_fake_user = False
                bpy.data.meshes.remove(mesh)
            except Exception as e:
                print(f"Error removing mesh '{name}': {e}")

    # 6) Clean up bookkeeping
    for key in ("appended_datablocks", "initial_datablocks"):
        if key in scene:
            del scene[key]
#-------------------------------------
#clear the World Download temp folder
#-------------------------------------
@persistent
def cleanup_world_downloads(dummy=None):
    """
    Removes all files within the World Downloads folder.
    Called on certain Blender events (load_post, save_pre, etc.).
    """
    if not os.path.isdir(WORLD_DOWNLOADS_FOLDER):
        return  # Folder doesn't even exist; nothing to do.

    for filename in os.listdir(WORLD_DOWNLOADS_FOLDER):
        file_path = os.path.join(WORLD_DOWNLOADS_FOLDER, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"[WARNING] Could not remove {file_path}: {e}")
