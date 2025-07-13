#Exploratory/Exp_UI/append.scene_helper.py

import os
import bpy
import uuid
import traceback
# ----------------------------------------------------------------------------------
# Scene Setup Helper
# ----------------------------------------------------------------------------------

def append_scene_from_blend(local_blend_path, scene_name=None):
    """
    Attempts to append a scene from a .blend file. If a scene_name is provided and exists
    in the file, that scene is appended; otherwise, if a text datablock called "GAME_WORLD"
    exists in the file, its content is used as the scene name. If neither is provided or valid,
    the first available scene is appended.

    This version also handles naming collisions by detecting which new scene
    was actually appended (e.g. "Scene.001") and then renaming it (if desired)
    so it never conflicts with the user's original scenes.

    Returns:
        (result, appended_scene_name)

        result: {'FINISHED'} if the scene was successfully appended, otherwise {'CANCELLED'}.

        appended_scene_name: The final name of the appended scene if successful;
                             otherwise None.
    """

    # Step 1: Load available scenes and texts from the blend file.
    try:
        with bpy.data.libraries.load(local_blend_path, link=False) as (data_from, data_to):
            scene_names = data_from.scenes
            text_names = data_from.texts
    except Exception as e:
        traceback.print_exc()
        return {'CANCELLED'}, None

    # Step 2: Check if any scenes are available.
    if not scene_names:
        return {'CANCELLED'}, None

    # Step 3: Determine which scene to append.
    chosen_scene_name = None
    if scene_name:
        if scene_name in scene_names:
            chosen_scene_name = scene_name
        else:
            return {'CANCELLED'}, None
    else:
        if "GAME_WORLD" in text_names:
            # Temporarily rename any existing “GAME_WORLD” in the current file
            old_txt = bpy.data.texts.get("GAME_WORLD")
            if old_txt:
                old_txt.name = "OLD_GAME_WORLD_TEMP"
            try:
                # Load the downloaded file’s GAME_WORLD text into a fresh datablock
                with bpy.data.libraries.load(local_blend_path, link=False) as (src, dst):
                    dst.texts = ["GAME_WORLD"]
                    loaded = list(dst.texts)  # e.g. ["GAME_WORLD"]

                loaded_name = loaded[0] if loaded else None
                new_txt = bpy.data.texts.get(loaded_name) if loaded_name else None

                if new_txt:
                    inferred = new_txt.as_string().strip()
                    # remove the loaded marker so it doesn’t stick around
                    bpy.data.texts.remove(new_txt)

                    # restore the original text’s name
                    if old_txt:
                        old_txt.name = "GAME_WORLD"

                    # only accept it if it matches one of the scenes
                    if inferred in scene_names:
                        chosen_scene_name = inferred
                    else:
                        chosen_scene_name = scene_names[0]
                else:
                    # nothing loaded → fallback
                    if old_txt:
                        old_txt.name = "GAME_WORLD"
                    chosen_scene_name = scene_names[0]

            except Exception:
                traceback.print_exc()
                # on error, restore and fallback
                if old_txt:
                    old_txt.name = "GAME_WORLD"
                chosen_scene_name = scene_names[0]

        else:
            # no marker in that file → just use the first scene
            chosen_scene_name = scene_names[0]


    # Step 4: Build file paths for the append operator.
    append_filepath = os.path.join(local_blend_path, "Scene", chosen_scene_name)
    append_directory = os.path.join(local_blend_path, "Scene")
    # ## NEW: keep track of existing scenes before appending
    before_scenes = set(bpy.data.scenes.keys())

    # Step 5: Call the append operator.
    try:
        result = bpy.ops.wm.append(
            filepath=append_filepath,
            directory=append_directory,
            filename=chosen_scene_name
        )
    except Exception as e:
        traceback.print_exc()
        return {'CANCELLED'}, None

    # Step 6: Figure out what new scene(s) arrived and pick the relevant one
    after_scenes = set(bpy.data.scenes.keys())
    new_scenes = after_scenes - before_scenes

    if not new_scenes:
        return {'CANCELLED'}, None
    elif len(new_scenes) > 1:
        # In rare cases, multiple new scenes might appear if you appended a group. 
        # You might pick the one that either ends with chosen_scene_name or handle differently.
        appended_scene_name = None
        for nm in new_scenes:
            # If Blender appended it with a suffix, e.g. "Scene" -> "Scene.001", we check startswith
            if nm.startswith(chosen_scene_name):
                appended_scene_name = nm
                break
        if appended_scene_name is None:
            # fallback: just pick one
            appended_scene_name = new_scenes.pop()
    else:
        appended_scene_name = new_scenes.pop()

    appended_scene = bpy.data.scenes.get(appended_scene_name)
    if not appended_scene:
        return {'CANCELLED'}, None

    # ## OPTIONAL: rename the appended scene to a guaranteed-unique name
    unique_name = f"Appended_{uuid.uuid4().hex[:8]}"
    appended_scene.name = unique_name
    appended_scene_name = unique_name

    # Step 7: Set the appended scene as active.
    try:
        bpy.context.window.scene = appended_scene
    except Exception as e:
        traceback.print_exc()
        return {'CANCELLED'}, appended_scene_name

    return {'FINISHED'}, appended_scene.name
