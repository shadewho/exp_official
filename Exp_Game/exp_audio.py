import bpy
import os
import aud
from bpy.types import PropertyGroup
from .exp_animations import AnimState
from ..exp_preferences import get_addon_path
import shutil
import time

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
        scene = bpy.context.scene
        if not scene.enable_audio or not sound_data or not sound_data.factory:
            return None
        handle = self.device.play(sound_data.factory)
        handle.volume = scene.audio_level
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
    """
    Checks AnimState each frame. If changed, stops old sound and
    plays new pointer-based sound from scene.character_audio.
    - NO local path fallback. Must have in-memory factory or it won't play.
    """

    def __init__(self):
        self.last_state = None
        self.current_handle = None

    def update_audio_state(self, new_state):
        scene = bpy.context.scene
        if not scene.enable_audio:
            self.stop_current_sound()
            self.last_state = None
            return

        # If state didn't change, do nothing
        if new_state == self.last_state:
            return

        # Stop old and try to play new
        self.stop_current_sound()
        self.play_sound_for_state(new_state)
        self.last_state = new_state

    def stop_current_sound(self):
        if self.current_handle:
            self.current_handle.stop()
            self.current_handle = None

    def play_sound_for_state(self, state):
        scene = bpy.context.scene
        if not scene.enable_audio:
            return

        prop_name = AUDIO_PROPS_MAP.get(state)
        if not prop_name:
            return

        audio_pg = getattr(scene, "character_audio", None)
        if not audio_pg:
            return

        sound_data = getattr(audio_pg, prop_name, None)
        if not sound_data:
            return

        device = aud.Device()

        if sound_data.factory:
            # If there's an in-memory factory, great—play that.
            handle = device.play(sound_data.factory)
            print(f"[CharacterAudio] Playing in-memory factory for '{sound_data.name}'.")
        else:
            # Otherwise, fallback to a local file path (e.g. in temp_sounds).
            abs_path = bpy.path.abspath(sound_data.filepath or "")
            if not os.path.isfile(abs_path):
                print(f"[CharacterAudio] No local file found at '{abs_path}' for sound '{sound_data.name}'.")
                return
            snd = aud.Sound(abs_path)
            handle = device.play(snd)
            print(f"[CharacterAudio] Playing local file fallback for '{sound_data.name}' => {abs_path}")

        handle.volume = scene.audio_level
        handle.pitch = sound_data.sound_speed  # Apply the speed multiplier
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
        scene = context.scene
        audio_pg = getattr(scene, "character_audio", None)
        if not audio_pg:
            self.report({'WARNING'}, "scene.character_audio not found.")
            return {'CANCELLED'}

        sound_data = getattr(audio_pg, self.sound_slot, None)
        if not sound_data:
            self.report({'WARNING'}, f"No Sound assigned to {self.sound_slot}")
            return {'CANCELLED'}

        if not scene.enable_audio:
            self.report({'INFO'}, "Audio is disabled in Scene properties.")
            return {'CANCELLED'}

        device = aud.Device()

        if sound_data.factory:
            # If there's an in-memory version, use it
            handle = device.play(sound_data.factory)
            handle.volume = scene.audio_level
            handle.pitch = sound_data.sound_speed  # Apply the sound speed multiplier
            self.report({'INFO'}, f"Playing in-memory sound: {sound_data.name}")
            return {'FINISHED'}
        else:
            # Fallback: local path
            abs_path = bpy.path.abspath(sound_data.filepath or "")
            if not os.path.isfile(abs_path):
                self.report({'ERROR'}, f"File not found: {abs_path}")
                return {'CANCELLED'}

            try:
                snd = aud.Sound(abs_path)
                handle = device.play(snd)
                handle.volume = scene.audio_level
                handle.pitch = sound_data.sound_speed  # Apply the sound speed multiplier
                self.report({'INFO'}, f"Playing fallback file for '{sound_data.name}' => {abs_path}")
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
    bl_label = "Build Audio (Append Sounds)"

    DEFAULT_SOUNDS_BLEND = os.path.join(get_addon_path(), "Exp_Game", "exp_assets", "Sounds", "exp_default_sounds.blend")

    DEFAULT_WALK_SOUND = "exp_walk_sound"
    DEFAULT_RUN_SOUND  = "exp_run_sound"
    DEFAULT_JUMP_SOUND = "exp_jump_sound"
    DEFAULT_FALL_SOUND = "exp_fall_sound"
    DEFAULT_LAND_SOUND = "exp_land_sound"

    def execute(self, context):
        prefs = context.preferences.addons["Exploratory"].preferences
        scene = context.scene
        
        # Create a subfolder in system’s temp dir (or anywhere you want)
        addon_root = get_addon_path()
        temp_sounds_dir = os.path.join(addon_root, "exp_assets", "Sounds", "temp_sounds")


        if not hasattr(scene, "character_audio"):
            self.report({'ERROR'}, "scene.character_audio property not found.")
            return {'CANCELLED'}
        audio_pg = scene.character_audio

        def process_sound(use_default: bool,
                        custom_blend: str,
                        chosen_name: str,
                        default_name: str,
                        assign_attr: str):
            if use_default:
                blend_path = self.DEFAULT_SOUNDS_BLEND
                sound_name = default_name
            else:
                blend_path = custom_blend
                sound_name = chosen_name

            if not sound_name:
                return

            sound_data = ensure_sound_appended(blend_path, sound_name)
            if sound_data:
                setattr(audio_pg, assign_attr, sound_data)

                temp_path = extract_packed_sound(sound_data, temp_sounds_dir)
                if temp_path:
                    print("Using temp_sounds_dir =", temp_sounds_dir)

                    # Overwrite the .filepath so we can do `aud.Sound(sound_data.filepath)`:
                    sound_data.filepath = temp_path




        # Walk
        process_sound(
            prefs.walk_use_default_sound,
            prefs.walk_custom_blend_sound,
            prefs.walk_sound_enum_prop,
            self.DEFAULT_WALK_SOUND,
            "walk_sound"
        )
        # Run
        process_sound(
            prefs.run_use_default_sound,
            prefs.run_custom_blend_sound,
            prefs.run_sound_enum_prop,
            self.DEFAULT_RUN_SOUND,
            "run_sound"
        )
        # Jump
        process_sound(
            prefs.jump_use_default_sound,
            prefs.jump_custom_blend_sound,
            prefs.jump_sound_enum_prop,
            self.DEFAULT_JUMP_SOUND,
            "jump_sound"
        )
        # Fall
        process_sound(
            prefs.fall_use_default_sound,
            prefs.fall_custom_blend_sound,
            prefs.fall_sound_enum_prop,
            self.DEFAULT_FALL_SOUND,
            "fall_sound"
        )
        # Land
        process_sound(
            prefs.land_use_default_sound,
            prefs.land_custom_blend_sound,
            prefs.land_sound_enum_prop,
            self.DEFAULT_LAND_SOUND,
            "land_sound"
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

    print(f"[clean_audio_temp] Temp folder reset: {temp_sounds_dir}")