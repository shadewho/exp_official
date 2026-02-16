#Exploratory/build_character.py
from .exp_preferences import get_addon_path
import bpy
import os
import random

# ------------------------------------------------------------------------
# The operator that appends skin + actions via asset pack scanning
# ------------------------------------------------------------------------
class EXPLORATORY_OT_BuildCharacter(bpy.types.Operator):
    """
    One operator that:
      1) Scans enabled asset packs for SKIN-marked objects
      2) Scans enabled asset packs for action-role-marked actions
      3) Falls back to defaults when no candidates are found
      4) Respects spawn / actions locks
    """
    bl_idname = "exploratory.build_character"
    bl_label = "Build Character (Skin + Actions)"

    DEFAULT_SKIN_BLEND = os.path.join(
        get_addon_path(),
        "Exp_Game", "exp_assets", "Skins", "exp_default_char.blend"
    )
    DEFAULT_ANIMS_BLEND = os.path.join(
        get_addon_path(),
        "Exp_Game", "exp_assets", "Skins", "exp_default_char.blend"
    )

    DEFAULT_ACTION_NAMES = {
        "IDLE": "exp_idle",
        "WALK": "exp_walk",
        "RUN":  "exp_run",
        "JUMP": "exp_jump",
        "FALL": "exp_fall",
        "LAND": "exp_land",
    }

    ROLE_TO_ATTR = {
        "IDLE": "idle_action",
        "WALK": "walk_action",
        "RUN":  "run_action",
        "JUMP": "jump_action",
        "FALL": "fall_action",
        "LAND": "land_action",
    }

    def execute(self, context):
        from .Exp_Game.props_and_utils.exp_asset_marking import (
            scan_packs_for_roles, resolve_candidates, _scan_packs_for_skin,
            ACTION_ROLES,
        )

        prefs = context.preferences.addons["Exploratory"].preferences
        scene = context.scene

        pack_paths = [
            e.filepath for e in prefs.asset_packs
            if e.enabled and os.path.isfile(e.filepath)
        ]

        # ─── 1) Skin ───────────────────────────────────────────────────────────
        if scene.character_spawn_lock:
            self.report(
                {'INFO'},
                "Character spawn is locked; skipping skin append."
            )
        else:
            # Remove existing character if present
            if scene.target_armature:
                bpy.ops.exploratory.remove_character('EXEC_DEFAULT')

            skin_candidates = _scan_packs_for_skin(pack_paths)

            if skin_candidates:
                # Pick one candidate (random if multiple)
                if len(skin_candidates) > 1:
                    chosen_idx = random.randrange(len(skin_candidates))
                else:
                    chosen_idx = 0

                chosen_skin, chosen_hierarchy = skin_candidates[chosen_idx]

                # Link chosen skin + hierarchy to scene
                for obj in [chosen_skin] + chosen_hierarchy:
                    if obj.name not in scene.collection.objects:
                        scene.collection.objects.link(obj)

                # Set target_armature
                if chosen_skin.type == 'ARMATURE':
                    scene.target_armature = chosen_skin
                else:
                    for obj in chosen_hierarchy:
                        if obj.type == 'ARMATURE':
                            scene.target_armature = obj
                            break

                # Remove unchosen candidates + their hierarchies
                for i, (skin, hier) in enumerate(skin_candidates):
                    if i == chosen_idx:
                        continue
                    for obj in hier + [skin]:
                        bpy.data.objects.remove(obj)
            else:
                # No pack skins found — use default
                self.append_all_skin_objects(use_default=True, custom_blend="")

        # ─── 2) Actions ───────────────────────────────────────────────────────
        if scene.character_actions_lock:
            self.report({'INFO'}, "Character actions lock is ON; skipping action assignment.")
        else:
            action_candidates = scan_packs_for_roles(
                pack_paths, ACTION_ROLES, "actions"
            )
            action_result = resolve_candidates(
                action_candidates,
                self.DEFAULT_ANIMS_BLEND,
                self.DEFAULT_ACTION_NAMES,
                "actions",
            )

            char_actions = scene.character_actions
            for role, act in action_result.items():
                attr = self.ROLE_TO_ATTR.get(role)
                if attr and act:
                    setattr(char_actions, attr, act)

        # ─── Deselect everything so the character has no outline ────────────────
        for obj in context.view_layer.objects:
            obj.select_set(False)
        context.view_layer.objects.active = None

        self.report({'INFO'}, "Build Character complete!")
        return {'FINISHED'}

    # ----------------------------------------------------------------
    # Skin helper (fallback for defaults)
    # ----------------------------------------------------------------
    def append_all_skin_objects(self, use_default, custom_blend):

        scene = bpy.context.scene

        # Decide .blend file
        if use_default:
            blend_path = os.path.join(get_addon_path(), "Exp_Game", "exp_assets", "Skins", "exp_default_char.blend")
        else:
            blend_path = custom_blend

        if not blend_path or not os.path.isfile(blend_path):
            self.report({'WARNING'}, f"Invalid blend file: {blend_path}")
            return

        with bpy.data.libraries.load(blend_path, link=False) as (data_from, _):
            all_obj_names_in_lib = data_from.objects

        to_append = []
        existing_armature = None

        for lib_obj_name in all_obj_names_in_lib:
            if lib_obj_name in bpy.data.objects:
                existing = bpy.data.objects.get(lib_obj_name)
                if existing and existing.type == 'ARMATURE':
                    existing_armature = existing
            else:
                to_append.append(lib_obj_name)

        if not to_append:
            if existing_armature:
                scene.target_armature = existing_armature
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
            if obj.type == 'ARMATURE':
                scene.target_armature = obj

        self.ensure_skin_in_scene_collection([o.name for o in appended_objects])

    def ensure_skin_in_scene_collection(self, object_names):
        scene = bpy.context.scene
        for name in object_names:
            if any(o.name == name for o in scene.collection.objects):
                continue
            data_obj = bpy.data.objects.get(name)
            if data_obj:
                scene.collection.objects.link(data_obj)
            else:
                print(f"Object '{name}' not found in bpy.data.objects.")

    # ----------------------------------------------------------------
    # Helper: ensures an action is in bpy.data.actions
    # ----------------------------------------------------------------
    def ensure_action_in_file(self, blend_path, action_name, state_label):
        if not blend_path or not os.path.isfile(blend_path):
            return None

        existing = bpy.data.actions.get(action_name)
        if existing:
            return existing

        with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
            if action_name in data_from.actions:
                data_to.actions = [action_name]
            else:
                return None

        return bpy.data.actions.get(action_name)



# ──────────────────────────────────────────────────────────────────────────────
# Operator: Build Armature (append from add-on asset)
# ──────────────────────────────────────────────────────────────────────────────
class EXPLORATORY_OT_BuildArmature(bpy.types.Operator):
    bl_idname = "exploratory.build_armature"
    bl_label = "Build Armature"
    bl_description = (
        "\u2022 Append the default armature\n"
        "\u2022 Compatible with all default actions\n"
        "\u2022 Useful for parenting a character mesh without creating or mapping new actions\n"
        "\u2022 If you want to use your own armature, you must also map or create new actions\n"
        "\u2022 You can adjust armature bone loc/rot/scale in any way (A-pose, different character size/shape etc.)\n"
        "\u2022 Adding/removing/renaming bones may cause errors. That data is what default actions depend on."
    )
    bl_options = {'REGISTER', 'UNDO'}

    ARMATURE_BLEND = os.path.join(
        get_addon_path(),
        "Exp_Game", "exp_assets", "Armature", "Armature.blend"
    )

    def _resolve_path(self) -> str | None:
        p = self.ARMATURE_BLEND
        if os.path.isfile(p):
            return p
        # fallback to .blend1 if present
        alt = os.path.splitext(p)[0] + ".blend1"
        if os.path.isfile(alt):
            return alt
        return None

    def execute(self, context):
        scene = context.scene
        blend_path = self._resolve_path()
        if not blend_path:
            self.report({'ERROR'}, "Armature .blend not found: Exp_Game/exp_assets/Armature/Armature.blend")
            return {'CANCELLED'}

        # 1) Try to append any OBJECTs whose names look like an armature; then filter by type.
        try:
            with bpy.data.libraries.load(blend_path, link=False) as (df, dt):
                names = list(df.objects)
                likely = [n for n in names if "armature" in n.lower() or "rig" in n.lower()]
                dt.objects = likely if likely else names
        except Exception as e:
            self.report({'ERROR'}, f"Failed reading {blend_path}: {e}")
            return {'CANCELLED'}

        chosen = None
        for ob in dt.objects:
            if not ob:
                continue
            if getattr(ob, "type", "") != 'ARMATURE':
                continue
            context.scene.collection.objects.link(ob)
            chosen = ob  # last one wins if multiple
        # 2) Fallback: append raw Armature datablock and create an Object for it.
        if not chosen:
            try:
                with bpy.data.libraries.load(blend_path, link=False) as (df2, dt2):
                    if df2.armatures:
                        dt2.armatures = [df2.armatures[0]]
                    else:
                        dt2.armatures = []
                if dt2.armatures:
                    arm_data = dt2.armatures[0]
                    chosen = bpy.data.objects.new(arm_data.name, arm_data)
                    context.scene.collection.objects.link(chosen)
            except Exception as e:
                self.report({'ERROR'}, f"Could not append armature from {blend_path}: {e}")
                return {'CANCELLED'}

        if not chosen:
            self.report({'ERROR'}, "No armature found in the library.")
            return {'CANCELLED'}

        # Point the panel's Target Armature to the appended armature
        try:
            scene.target_armature = chosen
        except Exception:
            pass

        self.report({'INFO'}, f"Armature appended: {chosen.name}")
        return {'FINISHED'}
