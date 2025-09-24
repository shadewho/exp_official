# Exploratory/Exp_UI/download_and_explore/cleanup.py

import os
import bpy
from bpy.app.handlers import persistent
from ..main_config import WORLD_DOWNLOADS_FOLDER

# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────
PURGE_ORPHANS = False


# Optional: purge orphans until clean --- not used because it feels risky
def _orphans_purge_until_clean(max_passes=5):
    """
    Purge orphaned IDs repeatedly. Must run on main thread.
    Returns the total number of removed datablocks.
    """
    removed_total = 0
    for _ in range(max_passes):
        try:
            # 4.1+ supports keyword args; older uses different flags.
            # We run twice: local IDs and linked IDs.
            res1 = bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=False)
            res2 = bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True)
            # When nothing is removed, operator returns {'CANCELLED'}
            if res1 == {'CANCELLED'} and res2 == {'CANCELLED'}:
                break
            # No reliable numeric return; just keep looping until CANCELLED twice.
        except Exception:
            # Keep going; some builds raise when nothing to purge.
            break
    return removed_total


def _unlink_material_everywhere(mat: bpy.types.Material):
    """
    Defensive unlink: remove material from all object material slots.
    """
    for obj in list(bpy.data.objects):
        data = getattr(obj, "data", None)
        if not data or not hasattr(data, "materials") or not data.materials:
            continue
        for i, slot_mat in enumerate(tuple(data.materials)):
            if slot_mat is mat:
                data.materials[i] = None


def _remove_by_name(collection: str, name: str):
    """
    Remove a datablock by its name from the appropriate bpy.data collection.
    `collection` must be one of: actions, images, sounds, meshes, objects,
    materials, armatures, curves, lights.
    """
    data_map = {
        "actions":   bpy.data.actions,
        "images":    bpy.data.images,
        "sounds":    bpy.data.sounds,
        "meshes":    bpy.data.meshes,
        "objects":   bpy.data.objects,
        "materials": bpy.data.materials,
        "armatures": bpy.data.armatures,
        "curves":    bpy.data.curves,
        "lights":    bpy.data.lights,
    }
    datablocks = data_map.get(collection)
    if not datablocks:
        return

    idb = datablocks.get(name)
    if not idb:
        return

    # Special handling for materials before removal: unlink them from all slots.
    if collection == "materials":
        _unlink_material_everywhere(idb)

    # For images: unpack so Blender can free file handles if packed.
    if collection == "images":
        try:
            if getattr(idb, "packed_file", None):
                idb.unpack(method='REMOVE')
        except Exception:
            pass

    # Try straight removal (preferred). If Blender says there are still users,
    # clear them and try again.
    try:
        datablocks.remove(idb)
    except RuntimeError:
        try:
            idb.use_fake_user = False
        except Exception:
            pass
        try:
            # `user_clear()` will drop users on many ID types.
            idb.user_clear()
        except Exception:
            # Not all ID types implement user_clear()
            pass

        # One more attempt after clearing users.
        try:
            datablocks.remove(idb)
        except Exception as e:
            print(f"[cleanup] Could not remove {collection[:-1]} '{name}': {e}")


def _unlink_and_delete_scene_objects(scene: bpy.types.Scene):
    """
    Remove all objects that live in `scene` and unlink their data cleanly.
    This avoids double-decrements later.
    """
    # Work on a copy since we'll be mutating the collections.
    for obj in list(scene.objects):
        try:
            # Unlink from all collections in the scene to drop references.
            # Removing from bpy.data.objects with do_unlink=True will also unlink.
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception as e:
            print(f"[cleanup] Error removing object '{obj.name}': {e}")


# ─────────────────────────────────────────────────────────────────────
# Public entrypoints
# ─────────────────────────────────────────────────────────────────────

def cleanup_downloaded_worlds():
    """
    Remove the appended scene and delete temp files.
    (We do NOT try to manually remove mesh data here anymore; that happens
    in cleanup_downloaded_datablocks(), which understands ID types.)
    """
    # 1) Grab handles, tolerate a torn-down context.
    try:
        scene = bpy.context.scene
        appended_scene_name = scene.get("appended_scene_name")
    except ReferenceError:
        scene = None
        appended_scene_name = None

    # 2) Remove the appended scene's objects and the scene itself.
    if appended_scene_name and appended_scene_name in bpy.data.scenes:
        appended_scene = bpy.data.scenes[appended_scene_name]
        _unlink_and_delete_scene_objects(appended_scene)
        try:
            bpy.data.scenes.remove(appended_scene)
        except Exception as e:
            print(f"[cleanup] Error removing scene '{appended_scene_name}': {e}")
    else:
        # Not necessarily an error—maybe append never finished.
        pass

    # 3) Clear temp download files.
    if os.path.isdir(WORLD_DOWNLOADS_FOLDER):
        for filename in os.listdir(WORLD_DOWNLOADS_FOLDER):
            file_path = os.path.join(WORLD_DOWNLOADS_FOLDER, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"[cleanup] Error deleting {file_path}: {e}")

    # 4) Scrub scene markers if the scene is still valid.
    if scene is not None:
        try:
            for key in ("appended_scene_name", "world_blend_path"):
                if key in scene:
                    del scene[key]
        except ReferenceError:
            pass


def cleanup_downloaded_datablocks():
    """
    Force-remove ALL datablocks that were recorded as appended for the last world.
    Order matters to avoid id_us_min spam:
      1) objects (already removed with the scene, but handle stragglers)
      2) materials (unlink from slots)
      3) geometry data (meshes/armatures/curves/lights)
      4) actions (after objects/armatures are gone)
      5) sounds
      6) images (after materials are gone)
      7) orphan purge loop
    """
    scene = bpy.context.scene
    appended = scene.get("appended_datablocks", {})
    if not appended:
        return

    # 1) Objects (stragglers)
    for name in appended.get("objects", []):
        _remove_by_name("objects", name)

    # 2) Materials (unlink slots inside _remove_by_name)
    for name in appended.get("materials", []):
        _remove_by_name("materials", name)

    # 3) Geometry ID types
    for name in appended.get("meshes", []):
        _remove_by_name("meshes", name)

    for name in appended.get("armatures", []):
        _remove_by_name("armatures", name)

    for name in appended.get("curves", []):
        _remove_by_name("curves", name)

    for name in appended.get("lights", []):
        _remove_by_name("lights", name)

    # 4) Actions (after armatures/objects have gone)
    for name in appended.get("actions", []):
        _remove_by_name("actions", name)

    # 5) Sounds
    for name in appended.get("sounds", []):
        _remove_by_name("sounds", name)

    # 6) Images (last – materials that referenced them are gone)
    for name in appended.get("images", []):
        _remove_by_name("images", name)

    # 7) Final bookkeeping (no global purge by default)
    for key in ("appended_datablocks", "initial_datablocks"):
        if key in scene:
            del scene[key]


    # Optional: purge orphans until clean --- not used because it feels risky
    if PURGE_ORPHANS:
        _orphans_purge_until_clean()


# ─────────────────────────────────────────────────────────────────────
# Optional: background folder sweeper (unchanged logic)
# ─────────────────────────────────────────────────────────────────────

@persistent
def cleanup_world_downloads(dummy=None):
    """Removes all files within the World Downloads folder."""
    if not os.path.isdir(WORLD_DOWNLOADS_FOLDER):
        return
    for filename in os.listdir(WORLD_DOWNLOADS_FOLDER):
        file_path = os.path.join(WORLD_DOWNLOADS_FOLDER, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"[WARNING] Could not remove {file_path}: {e}")
