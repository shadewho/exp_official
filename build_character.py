#Exploratory/build_character.py
from .exp_preferences import get_addon_path
from .Exp_Game.props_and_utils.exp_properties import (
    ANIM_STATE_DEFAULTS,
    get_anim_slot,
    ensure_default_slots,
)
from .Exp_Game.audio.exp_audio import extract_packed_sound
import bpy
import os

# ------------------------------------------------------------------------
# 4) The operator that appends skin + actions + sounds via collection slots
# ------------------------------------------------------------------------
class EXPLORATORY_OT_BuildCharacter(bpy.types.Operator):
    """
    One operator that:
      1) Appends the skin (default or custom)
      2) Ensures animation slot collection exists with defaults
      3) For each slot: loads default action/sound if not already set
    """
    bl_idname = "exploratory.build_character"
    bl_label = "Build Character (Skin + Actions)"

    DEFAULT_SKIN_BLEND = os.path.join(
        get_addon_path(),
        "Exp_Game", "exp_assets", "Skins", "exp_default_char.blend"
    )
    DEFAULT_ASSETS_BLEND = os.path.join(
        get_addon_path(),
        "Exp_Game", "exp_assets", "Skins", "exp_default_char.blend"
    )

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
            try:
                with bpy.data.libraries.load(skin_blend, link=False) as (df, _):
                    lib_names = set(df.objects)
            except Exception as e:
                self.report({'WARNING'}, f"Could not read {skin_blend}: {e}")
                lib_names = set()

            existing_arm = scene.target_armature
            if existing_arm and existing_arm.name in lib_names:
                bpy.ops.exploratory.remove_character('EXEC_DEFAULT')

            self.append_all_skin_objects(
                use_default  = prefs.skin_use_default,
                custom_blend = prefs.skin_custom_blend
            )

        # ─── 2) Animation Slots (actions + sounds) ────────────────────────────
        if getattr(scene, "character_slots_lock", False):
            self.report({'INFO'}, "Animation slots lock is ON; skipping slot assignment.")
        else:
            ensure_default_slots(scene)
            blend_path = self.DEFAULT_ASSETS_BLEND
            addon_root = get_addon_path()
            temp_sounds_dir = os.path.join(addon_root, "exp_assets", "Sounds", "temp_sounds")

            for state_name, cfg in ANIM_STATE_DEFAULTS.items():
                slot = get_anim_slot(scene, state_name)
                if not slot:
                    continue

                # Load default action if slot has none
                action_name = cfg["action"]
                if not slot.action and action_name:
                    if not bpy.data.actions.get(action_name) and os.path.isfile(blend_path):
                        with bpy.data.libraries.load(blend_path, link=False) as (df, dt):
                            if action_name in df.actions:
                                dt.actions = [action_name]
                    act = bpy.data.actions.get(action_name)
                    if act:
                        slot.action = act

                # Load default sound if slot has none
                sound_name = cfg["sound"]
                if not slot.sound and sound_name:
                    if not bpy.data.sounds.get(sound_name) and os.path.isfile(blend_path):
                        with bpy.data.libraries.load(blend_path, link=False) as (df, dt):
                            if sound_name in df.sounds:
                                dt.sounds = [sound_name]
                    snd = bpy.data.sounds.get(sound_name)
                    if snd:
                        slot.sound = snd
                        temp_path = extract_packed_sound(snd, temp_sounds_dir)
                        if temp_path:
                            snd.filepath = temp_path

        # ─── Deselect everything so the character has no outline ────────────────
        for obj in context.view_layer.objects:
            obj.select_set(False)
        context.view_layer.objects.active = None

        self.report({'INFO'}, "Build Character complete!")
        return {'FINISHED'}

    # ----------------------------------------------------------------
    # Exactly the same method for skin
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



# ──────────────────────────────────────────────────────────────────────────────
# Operator: Build Armature (append from add-on asset)
# ──────────────────────────────────────────────────────────────────────────────
class EXPLORATORY_OT_BuildArmature(bpy.types.Operator):
    bl_idname = "exploratory.build_armature"
    bl_label = "Build Armature"
    bl_description = (
        "• Append the default armature\n"
        "• Compatible with all default actions\n"
        "• Useful for parenting a character mesh without creating or mapping new actions\n"
        "• If you want to use your own armature, you must also map or create new actions\n"
        "• You can adjust armature bone loc/rot/scale in any way (A-pose, different character size/shape etc.)\n" 
        "• Adding/removing/renaming bones may cause errors. That data is what default actions depend on."
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

        # Point the panel’s Target Armature to the appended armature
        try:
            scene.target_armature = chosen
        except Exception:
            pass

        self.report({'INFO'}, f"Armature appended: {chosen.name}")
        return {'FINISHED'}
    