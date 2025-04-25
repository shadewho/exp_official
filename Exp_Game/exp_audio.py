import bpy
import os
import aud
from bpy.types import PropertyGroup
from .exp_animations import AnimState
from ..exp_preferences import get_addon_path
import shutil
import time
from .exp_time import get_game_time
from bpy.props import StringProperty, IntProperty
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, IntProperty
from bpy_extras.io_utils import ImportHelper
##############################################################################
# MAP FROM AnimState => The property name in scene.character_audio
##############################################################################
AUDIO_PROPS_MAP = {
    AnimState.WALK: "walk_sound",
    AnimState.RUN:  "run_sound",
    AnimState.JUMP: "jump_sound",
    AnimState.FALL: "fall_sound",
    AnimState.LAND: "land_sound",
}



def extract_packed_sound(sound_data: bpy.types.Sound, temp_dir: str) -> str:
    """
    Writes the packed bytes of 'sound_data' to a file in 'temp_dir'
    and returns the resulting absolute filepath.
    Returns None if something fails (no packed_file, etc.).
    """
    if not sound_data.packed_file:
        return None

    raw_bytes = sound_data.packed_file.data
    if not raw_bytes:
        return None

    # Try to guess the extension from the original filepath
    ext = os.path.splitext(sound_data.filepath)[1]  # e.g. ".mp3" or ".wav"
    if not ext:
        ext = ".dat"

    # Create a file name using the datablock name (remove spaces/special chars)
    file_name = sound_data.name.replace(" ", "_") + ext

    os.makedirs(temp_dir, exist_ok=True)  # ensure folder
    temp_path = os.path.join(temp_dir, file_name)

    with open(temp_path, "wb") as f:
        f.write(raw_bytes)

    return temp_path


def clear_temp_sounds(temp_sounds_dir: str):
    """
    Deletes the entire temp_sounds folder (if it exists),
    then re-creates an empty folder at that path.
    """
    if os.path.isdir(temp_sounds_dir):
        shutil.rmtree(temp_sounds_dir)  # Removes folder + all contents
    os.makedirs(temp_sounds_dir, exist_ok=True)


##############################################################################
# OPTIONAL: If you no longer want a "play_sound(filepath)" approach, remove this
##############################################################################
class ExpAudioManager:
    """
    Minimal manager if you ever want to manually play an in-memory `bpy.types.Sound`.
    """
    def __init__(self):
        self.device = aud.Device()

    def play_sound_datablock(self, sound_data: bpy.types.Sound):
        """Plays a sound datablock's factory (if it has one)."""
        prefs = bpy.context.preferences.addons["Exploratory"].preferences
        if not prefs.enable_audio or not sound_data or not sound_data.factory:
            return None

        handle = self.device.play(sound_data.factory)
        handle.volume = prefs.audio_level
        return handle

    def stop_all(self):
        self.device.stopAll()

_global_audio_manager = None
def get_global_audio_manager():
    global _global_audio_manager
    if _global_audio_manager is None:
        _global_audio_manager = ExpAudioManager()
    return _global_audio_manager

##############################################################################
# CharacterAudioStateManager
##############################################################################
class CharacterAudioStateManager:
    def __init__(self):
        self.last_state = None
        self.current_handle = None
        self.current_sound_start_time = None
        self.current_sound_duration = None

    def update_audio_state(self, new_state):
        prefs = bpy.context.preferences.addons["Exploratory"].preferences
        if not prefs.enable_audio:
            self.stop_current_sound()
            self.last_state = None
            return

        current_time = get_game_time()

        if new_state != self.last_state:
            self.stop_current_sound()
            self.last_state = new_state
            self.play_sound_for_state(new_state)
            return

        if new_state in (AnimState.WALK, AnimState.RUN, AnimState.FALL):
            if self.current_handle:
                elapsed = current_time - self.current_sound_start_time
                if elapsed >= self.current_sound_duration:
                    self.stop_current_sound()
                    self.play_sound_for_state(new_state)
            else:
                self.play_sound_for_state(new_state)

    def stop_current_sound(self):
        if self.current_handle:
            self.current_handle.stop()
        self.current_handle = None
        self.current_sound_start_time = None
        self.current_sound_duration = None

    def play_sound_for_state(self, state):
        prefs = bpy.context.preferences.addons["Exploratory"].preferences
        if not prefs.enable_audio:
            return

        prop_name = AUDIO_PROPS_MAP.get(state)
        if not prop_name:
            return

        audio_pg = getattr(bpy.context.scene, "character_audio", None)
        if not audio_pg:
            return

        sound_data = getattr(audio_pg, prop_name, None)
        if not sound_data:
            return

        device = aud.Device()
        if sound_data.factory:
            handle = device.play(sound_data.factory)
        else:
            abs_path = bpy.path.abspath(sound_data.filepath or "")
            if not os.path.isfile(abs_path):
                return
            handle = device.play(aud.Sound(abs_path))

        duration = getattr(sound_data, "sound_duration", None)
        if duration is None:
            try:
                duration = handle.length
            except Exception:
                duration = 2.0

        self.current_sound_start_time = get_game_time()
        self.current_sound_duration = duration

        handle.loop_count = 0
        handle.volume = prefs.audio_level
        handle.pitch = sound_data.sound_speed
        self.current_handle = handle


##############################################################################
# GET/SET SINGLETON
##############################################################################
_global_char_audio_mgr = None
def get_global_audio_state_manager():
    global _global_char_audio_mgr
    if _global_char_audio_mgr is None:
        _global_char_audio_mgr = CharacterAudioStateManager()
    return _global_char_audio_mgr

##############################################################################
# TEST OPERATOR (No Local Path Fallback)
##############################################################################
class AUDIO_OT_TestSoundPointer(bpy.types.Operator):
    """Plays a single pointer (walk_sound, run_sound, etc.) from memory only."""
    bl_idname = "exp_audio.test_sound_pointer"
    bl_label = "Test Sound (Pointer)"

    sound_slot: bpy.props.StringProperty(default="walk_sound")

    def execute(self, context):
        prefs = context.preferences.addons["Exploratory"].preferences
        audio_pg = getattr(context.scene, "character_audio", None)
        if not audio_pg:
            self.report({'WARNING'}, "scene.character_audio not found.")
            return {'CANCELLED'}

        sound_data = getattr(audio_pg, self.sound_slot, None)
        if not sound_data:
            self.report({'WARNING'}, f"No Sound assigned to {self.sound_slot}")
            return {'CANCELLED'}

        if not prefs.enable_audio:
            self.report({'INFO'}, "Audio is disabled in Preferences.")
            return {'CANCELLED'}

        device = aud.Device()
        if sound_data.factory:
            handle = device.play(sound_data.factory)
            handle.volume = prefs.audio_level
            handle.pitch = sound_data.sound_speed
            self.report({'INFO'}, f"Playing in-memory sound: {sound_data.name}")
            return {'FINISHED'}

        abs_path = bpy.path.abspath(sound_data.filepath or "")
        if not os.path.isfile(abs_path):
            self.report({'ERROR'}, f"File not found: {abs_path}")
            return {'CANCELLED'}

        try:
            handle = device.play(aud.Sound(abs_path))
            handle.volume = prefs.audio_level
            handle.pitch = sound_data.sound_speed
            self.report({'INFO'}, f"Playing fallback file: {abs_path}")
            return {'FINISHED'}
        except Exception as ex:
            self.report({'ERROR'}, f"aud error: {ex}")
            return {'CANCELLED'}

##############################################################################
# PROPERTY GROUP
##############################################################################
class CharacterAudioPG(PropertyGroup):
    """Holds pointer-based references to appended Sound datablocks."""
    walk_sound: bpy.props.PointerProperty(type=bpy.types.Sound)
    run_sound:  bpy.props.PointerProperty(type=bpy.types.Sound)
    jump_sound: bpy.props.PointerProperty(type=bpy.types.Sound)
    fall_sound: bpy.props.PointerProperty(type=bpy.types.Sound)
    land_sound: bpy.props.PointerProperty(type=bpy.types.Sound)

##############################################################################
# BUILD AUDIO OPERATOR (UNMODIFIED LOGIC)
##############################################################################
def list_sounds_in_blend(blend_path: str):
    if not blend_path or not os.path.isfile(blend_path):
        return []
    try:
        with bpy.data.libraries.load(blend_path, link=False) as (data_from, _):
            sound_names = data_from.sounds
    except:
        sound_names = []
    return [(s, s, f"Sound: {s}") for s in sound_names]

def ensure_sound_appended(blend_path: str, sound_name: str) -> bpy.types.Sound:
    if not blend_path or not os.path.isfile(blend_path):
        return None
    existing = bpy.data.sounds.get(sound_name)
    if existing:
        return existing
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        if sound_name in data_from.sounds:
            data_to.sounds = [sound_name]
        else:
            return None
    return bpy.data.sounds.get(sound_name)

class EXPLORATORY_OT_BuildAudio(bpy.types.Operator):
    """
    Automatically appends default or custom sounds for walk/run/jump/fall/land,
    then assigns them to scene.character_audio pointer properties.
    """
    bl_idname = "exploratory.build_audio"
    bl_label  = "Build Audio (Append Sounds)"

    DEFAULT_SOUNDS_BLEND = os.path.join(get_addon_path(), "Exp_Game", "exp_assets", "Sounds", "exp_default_sounds.blend")
    DEFAULT_WALK_SOUND   = "exp_walk_sound"
    DEFAULT_RUN_SOUND    = "exp_run_sound"
    DEFAULT_JUMP_SOUND   = "exp_jump_sound"
    DEFAULT_FALL_SOUND   = "exp_fall_sound"
    DEFAULT_LAND_SOUND   = "exp_land_sound"

    def execute(self, context):
        scene = context.scene

        # 0) Honor the audio‐lock: skip everything if it's on
        if getattr(scene, "character_audio_lock", False):
            self.report({'INFO'}, "Character audio lock is ON; skipping audio build.")
            return {'FINISHED'}

        # 1) Grab prefs and prepare temp folder
        prefs = context.preferences.addons["Exploratory"].preferences
        addon_root = get_addon_path()
        temp_sounds_dir = os.path.join(addon_root, "exp_assets", "Sounds", "temp_sounds")

        # 2) Ensure the character_audio property exists
        if not hasattr(scene, "character_audio"):
            self.report({'ERROR'}, "scene.character_audio property not found.")
            return {'CANCELLED'}
        audio_pg = scene.character_audio

        # 3) Helper to choose default vs custom, append, extract packed, assign
        def process_sound(slot_name, use_default, custom_blend, chosen_name, default_name):
            if use_default:
                blend_path = self.DEFAULT_SOUNDS_BLEND
                sound_name = default_name
            else:
                blend_path = custom_blend
                sound_name = chosen_name

            if not sound_name:
                return

            # Append if needed
            existing = bpy.data.sounds.get(sound_name)
            if not existing and os.path.isfile(blend_path):
                with bpy.data.libraries.load(blend_path, link=False) as (df, dt):
                    if sound_name in df.sounds:
                        dt.sounds = [sound_name]
                existing = bpy.data.sounds.get(sound_name)

            if existing:
                setattr(audio_pg, slot_name, existing)
                # extract packed bytes so aud.Sound(filepath) works
                temp_path = extract_packed_sound(existing, temp_sounds_dir)
                if temp_path:
                    existing.filepath = temp_path

        # 4) Process each slot
        process_sound("walk_sound",
            prefs.walk_use_default_sound,
            prefs.walk_custom_blend_sound,
            prefs.walk_sound_enum_prop,
            self.DEFAULT_WALK_SOUND
        )
        process_sound("run_sound",
            prefs.run_use_default_sound,
            prefs.run_custom_blend_sound,
            prefs.run_sound_enum_prop,
            self.DEFAULT_RUN_SOUND
        )
        process_sound("jump_sound",
            prefs.jump_use_default_sound,
            prefs.jump_custom_blend_sound,
            prefs.jump_sound_enum_prop,
            self.DEFAULT_JUMP_SOUND
        )
        process_sound("fall_sound",
            prefs.fall_use_default_sound,
            prefs.fall_custom_blend_sound,
            prefs.fall_sound_enum_prop,
            self.DEFAULT_FALL_SOUND
        )
        process_sound("land_sound",
            prefs.land_use_default_sound,
            prefs.land_custom_blend_sound,
            prefs.land_sound_enum_prop,
            self.DEFAULT_LAND_SOUND
        )

        self.report({'INFO'}, "Build Audio complete (sounds assigned).")
        return {'FINISHED'}



def clean_audio_temp():
    """
    1) Stop all currently playing sounds
    2) Unset references in scene.character_audio (walk_sound, run_sound, etc.)
       – but do NOT touch other pointer properties for custom reactions
    3) Remove 'exp_*' Sound datablocks from bpy.data.sounds
    4) Sleep a moment to let Windows close file handles
    5) Delete and recreate the temp_sounds folder
    """
    # A) Stop audio playback
    audio_mgr = get_global_audio_manager()
    audio_mgr.stop_all()

    # B) Clear only the 'scene.character_audio' pointers for walk/run/jump/fall/land
    #    (i.e. the standard slots that BuildAudio operator sets up)
    scn = bpy.context.scene
    if hasattr(scn, "character_audio"):
        audio_pg = scn.character_audio
        audio_pg.walk_sound = None
        audio_pg.run_sound  = None
        audio_pg.jump_sound = None
        audio_pg.fall_sound = None
        audio_pg.land_sound = None
    # (This way, we do NOT remove any custom reaction’s pointer
    # that might reference a Sound named 'my_cool_fx.blend' or similar.)

    # C) Remove the actual Sound datablocks in bpy.data.sounds whose name starts with "exp_"
    #    (these are typically the ones appended by build_audio).
    for snd in list(bpy.data.sounds):
        if snd.name.startswith("exp_"):
            bpy.data.sounds.remove(snd)
            # If you need to ensure in-memory data is also freed, you can do:
            # snd.filepath = ""
            # snd.factory_update()
            # (But once removed(), it’s gone anyway.)

    # D) Give Windows time to release open file handles
    time.sleep(0.2)

    # E) Wipe and recreate the temp folder
    addon_root = get_addon_path()
    temp_sounds_dir = os.path.join(addon_root, "Exp_Game", "exp_assets", "Sounds", "temp_sounds")
    if os.path.isdir(temp_sounds_dir):
        shutil.rmtree(temp_sounds_dir, ignore_errors=True)
    os.makedirs(temp_sounds_dir, exist_ok=True)



# ------------------------------------------------------------------------
#Play sound reactions audio helpers -- for SOUND reactions only
#-------------------------------------------------------------------------
class EXP_AUDIO_OT_LoadAudioFile(bpy.types.Operator, ImportHelper):
    """Load an external audio file into a Play Sound reaction"""
    bl_idname = "exp_audio.load_audio_file"
    bl_label = "Load Reaction Sound"
    filename_ext = ".wav;.mp3;.ogg"

    filter_glob: StringProperty(
        default="*.wav;*.mp3;*.ogg",
        options={'HIDDEN'},
        description="Audio file extensions"
    )

    interaction_index: IntProperty()
    reaction_index:    IntProperty()

    def execute(self, context):
        # load & pack
        sound = bpy.data.sounds.load(self.filepath, check_existing=True)
        sound.pack()
        # assign to the SOUND reaction
        inter = context.scene.custom_interactions[self.interaction_index]
        reaction = inter.reactions[self.reaction_index]
        reaction.sound_pointer = sound
        self.report({'INFO'}, f"Loaded '{sound.name}' into '{reaction.name}'")
        return {'FINISHED'}


class EXP_AUDIO_OT_TestReactionSound(bpy.types.Operator):
    """Play the audio in this Sound-Reaction slot only if it’s packed"""
    bl_idname = "exp_audio.test_reaction_sound"
    bl_label = "Test Reaction Sound"

    interaction_index: IntProperty()
    reaction_index:    IntProperty()

    def execute(self, context):
        prefs = context.preferences.addons["Exploratory"].preferences
        scene = context.scene
        inter = scene.custom_interactions[self.interaction_index]
        reaction = inter.reactions[self.reaction_index]
        sound = reaction.sound_pointer

        if not sound or not sound.packed_file:
            self.report({'WARNING'}, "No packed sound assigned.")
            return {'CANCELLED'}

        addon_root = get_addon_path()
        temp_dir = os.path.join(addon_root, "exp_assets", "Sounds", "temp_sounds")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = extract_packed_sound(sound, temp_dir)
        if not temp_path:
            self.report({'ERROR'}, "Failed to extract packed sound.")
            return {'CANCELLED'}

        try:
            handle = aud.Device().play(aud.Sound(temp_path))
        except Exception as e:
            self.report({'ERROR'}, f"aud error: {e}")
            return {'CANCELLED'}

        handle.volume = prefs.audio_level * reaction.sound_volume
        handle.pitch = getattr(sound, "sound_speed", 1.0)
        return {'FINISHED'}

    
# ------------------------------------------------------------------------
# Pack all sounds to the blend file
# ------------------------------------------------------------------------
class EXP_AUDIO_OT_PackAllSounds(bpy.types.Operator):
    """Pack every Sound datablock into this .blend file"""
    bl_idname = "exp_audio.pack_all_sounds"
    bl_label = "Pack All Sounds"

    def execute(self, context):
        packed = 0
        for snd in bpy.data.sounds:
            # only pack if it’s external and not already packed
            if snd.filepath and not snd.packed_file:
                snd.pack()
                packed += 1
        self.report({'INFO'}, f"Packed {packed} sound(s) into blend")
        return {'FINISHED'}
