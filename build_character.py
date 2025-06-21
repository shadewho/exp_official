#Exploratory/build_character.py
from .exp_preferences import get_addon_path
import bpy
import os

# ------------------------------------------------------------------------
# 4) The operator that appends skin + actions
# ------------------------------------------------------------------------
class EXPLORATORY_OT_BuildCharacter(bpy.types.Operator):
    """
    One operator that:
      1) Appends the skin (default or custom)
      2) Appends each action (default or custom)
      3) Assigns appended actions to scene.character_actions pointers
    """
    bl_idname = "exploratory.build_character"
    bl_label = "Build Character (Skin + Actions)"

    DEFAULT_SKIN_BLEND = os.path.join(
        get_addon_path(),
        "Exp_Game", "exp_assets", "Skins", "exp_default_char.blend"
    )
    DEFAULT_ANIMS_BLEND = os.path.join(
        get_addon_path(),
        "Exp_Game", "exp_assets", "Animations", "exp_animations.blend"
    )

    DEFAULT_IDLE_NAME = "exp_idle"
    DEFAULT_WALK_NAME = "exp_walk"
    DEFAULT_RUN_NAME  = "exp_run"
    DEFAULT_JUMP_NAME = "exp_jump"
    DEFAULT_FALL_NAME = "exp_fall"
    DEFAULT_LAND_NAME = "exp_land"

    def execute(self, context):
        prefs = context.preferences.addons["Exploratory"].preferences
        scene = context.scene

        # ─── 1) Skin ───────────────────────────────────────────────────────────
        skin_blend = (
            self.DEFAULT_SKIN_BLEND
            if prefs.skin_use_default
            else prefs.skin_custom_blend
        )
        if scene.character_spawn_lock:
            self.report(
                {'INFO'},
                "Character spawn is locked; skipping skin append."
            )
        else:
            # If the armature already exists in that blend, remove it first
            try:
                with bpy.data.libraries.load(skin_blend, link=False) as (df, _):
                    lib_names = set(df.objects)
            except Exception as e:
                self.report({'WARNING'}, f"Could not read {skin_blend}: {e}")
                lib_names = set()

            existing_arm = scene.target_armature
            if existing_arm and existing_arm.name in lib_names:
                bpy.ops.exploratory.remove_character('EXEC_DEFAULT')

            # Append skin objects
            self.append_all_skin_objects(
                use_default  = prefs.skin_use_default,
                custom_blend = prefs.skin_custom_blend
            )

        # ─── 2) Actions ───────────────────────────────────────────────────────
        if scene.character_actions_lock:
            self.report(
                {'INFO'},
                "Character actions lock is ON; skipping action assignment."
            )
        else:
            char_actions = scene.character_actions

            def process_action(state_label, use_default, custom_blend,
                               chosen_name, default_name, target_attr):
                blend = (
                    self.DEFAULT_ANIMS_BLEND
                    if use_default
                    else custom_blend
                )
                action_name = default_name if use_default else chosen_name
                if not action_name:
                    print(f"[{state_label}] No action name; skipping.")
                    return

                # Append if not already loaded
                if not bpy.data.actions.get(action_name) and os.path.isfile(blend):
                    with bpy.data.libraries.load(blend, link=False) as (df, dt):
                        if action_name in df.actions:
                            dt.actions = [action_name]

                act = bpy.data.actions.get(action_name)
                if act:
                    setattr(char_actions, target_attr, act)
                    print(f"[{state_label}] => {act.name}")

            process_action(
                "Idle",
                prefs.idle_use_default_action,
                prefs.idle_custom_blend_action,
                prefs.idle_action_enum_prop,
                self.DEFAULT_IDLE_NAME,
                "idle_action"
            )
            process_action(
                "Walk",
                prefs.walk_use_default_action,
                prefs.walk_custom_blend_action,
                prefs.walk_action_enum_prop,
                self.DEFAULT_WALK_NAME,
                "walk_action"
            )
            process_action(
                "Run",
                prefs.run_use_default_action,
                prefs.run_custom_blend_action,
                prefs.run_action_enum_prop,
                self.DEFAULT_RUN_NAME,
                "run_action"
            )
            process_action(
                "Jump",
                prefs.jump_use_default_action,
                prefs.jump_custom_blend_action,
                prefs.jump_action_enum_prop,
                self.DEFAULT_JUMP_NAME,
                "jump_action"
            )
            process_action(
                "Fall",
                prefs.fall_use_default_action,
                prefs.fall_custom_blend_action,
                prefs.fall_action_enum_prop,
                self.DEFAULT_FALL_NAME,
                "fall_action"
            )
            process_action(
                "Land",
                prefs.land_use_default_action,
                prefs.land_custom_blend_action,
                prefs.land_action_enum_prop,
                self.DEFAULT_LAND_NAME,
                "land_action"
            )

        self.report({'INFO'}, "Build Character complete!")
        return {'FINISHED'}

    # ----------------------------------------------------------------
    # Exactly the same method for skin
    # ----------------------------------------------------------------
    def append_all_skin_objects(self, use_default, custom_blend):
        import os
        import bpy
        from .exp_preferences import get_addon_path

        scene = bpy.context.scene

        # Decide .blend file
        if use_default:
            blend_path = os.path.join(get_addon_path(), "Exp_Game", "exp_assets", "Skins", "exp_default_char.blend")
        else:
            blend_path = custom_blend

        if not blend_path or not os.path.isfile(blend_path):
            self.report({'WARNING'}, f"Invalid blend file: {blend_path}")
            return

        print(f"[AppendSkin] Checking objects from: {blend_path}")

        with bpy.data.libraries.load(blend_path, link=False) as (data_from, _):
            all_obj_names_in_lib = data_from.objects

        to_append = []
        existing_armature = None

        for lib_obj_name in all_obj_names_in_lib:
            if lib_obj_name in bpy.data.objects:
                print(f"Skipping '{lib_obj_name}' (already in file).")
                existing = bpy.data.objects.get(lib_obj_name)
                if existing and existing.type == 'ARMATURE':
                    existing_armature = existing
            else:
                to_append.append(lib_obj_name)

        if not to_append:
            if existing_armature:
                scene.target_armature = existing_armature
                print(f"No new objects appended. Using existing armature '{existing_armature.name}'.")
            else:
                print("No objects to append and no existing armature found.")
            return

        appended_objects = []
        with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
            data_to.objects = to_append

        for obj in data_to.objects:
            if not obj:
                continue
            bpy.context.scene.collection.objects.link(obj)
            appended_objects.append(obj)
            print(f"Appended: {obj.name}")
            if obj.type == 'ARMATURE':
                scene.target_armature = obj
                print(f"Assigned scene.target_armature => {obj.name}")

        self.ensure_skin_in_scene_collection([o.name for o in appended_objects])
        print(f"[AppendSkin] Done. Appended {len(appended_objects)} objects from {blend_path}.")

    def ensure_skin_in_scene_collection(self, object_names):
        scene = bpy.context.scene
        for name in object_names:
            if any(o.name == name for o in scene.collection.objects):
                continue
            data_obj = bpy.data.objects.get(name)
            if data_obj:
                scene.collection.objects.link(data_obj)
                print(f"Linked '{name}' to scene collection.")
            else:
                print(f"Object '{name}' not found in bpy.data.objects.")

    # ----------------------------------------------------------------
    # Helper: ensures an action is in bpy.data.actions
    # ----------------------------------------------------------------
    def ensure_action_in_file(self, blend_path, action_name, state_label):
        if not blend_path or not os.path.isfile(blend_path):
            print(f"[{state_label}] Invalid blend path: {blend_path}")
            return None

        existing = bpy.data.actions.get(action_name)
        if existing:
            print(f"[{state_label}] Action '{action_name}' already loaded.")
            return existing

        print(f"[{state_label}] Appending '{action_name}' from {blend_path}")
        with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
            if action_name in data_from.actions:
                data_to.actions = [action_name]
            else:
                print(f"[{state_label}] Action '{action_name}' not found in {blend_path}")
                return None

        return bpy.data.actions.get(action_name)
