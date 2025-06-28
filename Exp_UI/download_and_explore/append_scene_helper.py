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

    print("[LOG] Starting append_scene_from_blend with file:", local_blend_path)

    # Step 1: Load available scenes and texts from the blend file.
    try:
        print("[LOG] Attempting to load library from file...")
        with bpy.data.libraries.load(local_blend_path, link=False) as (data_from, data_to):
            scene_names = data_from.scenes
            text_names = data_from.texts
            print("[LOG] Library loaded. Found scenes:", scene_names)
            print("[LOG] Library loaded. Found texts:", text_names)
    except Exception as e:
        print("[EXCEPTION] Failed to load library from file:", local_blend_path)
        traceback.print_exc()
        return {'CANCELLED'}, None

    # Step 2: Check if any scenes are available.
    if not scene_names:
        print("[ERROR] No scenes found in the file:", local_blend_path)
        return {'CANCELLED'}, None

    # Step 3: Determine which scene to append.
    chosen_scene_name = None
    if scene_name:
        if scene_name in scene_names:
            chosen_scene_name = scene_name
            print("[LOG] Using user-specified game world scene:", chosen_scene_name)
        else:
            print("[ERROR] The specified scene", scene_name, "was not found in the blend file.")
            return {'CANCELLED'}, None
    else:
        # If no specific scene is specified, check for a text datablock called "GAME_WORLD".
        if "GAME_WORLD" in text_names:
            try:
                # Load the GAME_WORLD text datablock.
                with bpy.data.libraries.load(local_blend_path, link=False) as (data_from2, data_to2):
                    data_to2.texts = ["GAME_WORLD"]

                game_world_text = bpy.data.texts.get("GAME_WORLD")
                if game_world_text:
                    inferred_scene_name = game_world_text.as_string().strip()
                    print("[LOG] GAME_WORLD text found. Specified scene name:", inferred_scene_name)
                    if inferred_scene_name in scene_names:
                        chosen_scene_name = inferred_scene_name
                    else:
                        print("[ERROR] Inferred scene name from GAME_WORLD not found among available scenes.")
                        return {'CANCELLED'}, None
                else:
                    print("[ERROR] GAME_WORLD text block could not be loaded properly.")
                    chosen_scene_name = scene_names[0]
            except Exception as e:
                print("[EXCEPTION] Failed to load GAME_WORLD text block:")
                traceback.print_exc()
                chosen_scene_name = scene_names[0]
        else:
            # Default to the first available scene
            chosen_scene_name = scene_names[0]
            print("[LOG] No GAME_WORLD text block found. Defaulting to first scene:", chosen_scene_name)

    # Step 4: Build file paths for the append operator.
    append_filepath = os.path.join(local_blend_path, "Scene", chosen_scene_name)
    append_directory = os.path.join(local_blend_path, "Scene")
    print("[LOG] Prepared append parameters:")
    print("      filepath =", append_filepath)
    print("      directory =", append_directory)
    print("      filename =", chosen_scene_name)

    # ## NEW: keep track of existing scenes before appending
    before_scenes = set(bpy.data.scenes.keys())

    # Step 5: Call the append operator.
    try:
        result = bpy.ops.wm.append(
            filepath=append_filepath,
            directory=append_directory,
            filename=chosen_scene_name
        )
        print("[LOG] bpy.ops.wm.append returned:", result)
    except Exception as e:
        print("[EXCEPTION] Exception during bpy.ops.wm.append:")
        traceback.print_exc()
        return {'CANCELLED'}, None

    # Step 6: Figure out what new scene(s) arrived and pick the relevant one
    after_scenes = set(bpy.data.scenes.keys())
    new_scenes = after_scenes - before_scenes

    if not new_scenes:
        print("[ERROR] No new scene was appended, naming conflict or unknown error.")
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
        print("[ERROR] Could not find appended scene data-block after append.")
        return {'CANCELLED'}, None

    # ## OPTIONAL: rename the appended scene to a guaranteed-unique name
    unique_name = f"Appended_{uuid.uuid4().hex[:8]}"
    appended_scene.name = unique_name
    appended_scene_name = unique_name
    print(f"[LOG] Renamed appended scene to: {unique_name}")

    # Step 7: Set the appended scene as active.
    try:
        bpy.context.window.scene = appended_scene
        print("[LOG] Appended scene set as active.")
    except Exception as e:
        print("[EXCEPTION] Failed to set the appended scene as active:")
        traceback.print_exc()
        return {'CANCELLED'}, appended_scene_name

    print("[LOG] Successfully appended scene with final name:", appended_scene.name)
    return {'FINISHED'}, appended_scene.name
