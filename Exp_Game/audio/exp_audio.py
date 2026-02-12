import bpy
import os
import aud
from ..animations.state_machine import AnimState
from ...exp_preferences import get_addon_path
import shutil
import time
from ..props_and_utils.exp_time import get_game_time
from bpy.props import StringProperty, IntProperty
from bpy_extras.io_utils import ImportHelper
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
        if not prefs.enable_audio or prefs.audio_level <= 0 or not sound_data or not sound_data.factory:
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
# Cached Preferences for Performance
##############################################################################
# PERF: Cache preferences to avoid O(n) dictionary lookups every frame
# Preference changes are rare; audio checks happen 30+ times/second
_cached_prefs = None
_cached_prefs_time = 0.0
_PREFS_CACHE_INTERVAL = 1.0  # Re-check preferences every 1 second

def _get_cached_audio_prefs():
    """Get cached audio preferences. Re-fetches every 1 second to catch changes."""
    global _cached_prefs, _cached_prefs_time
    import time
    now = time.perf_counter()
    if _cached_prefs is None or (now - _cached_prefs_time) > _PREFS_CACHE_INTERVAL:
        try:
            prefs = bpy.context.preferences.addons["Exploratory"].preferences
            _cached_prefs = {
                'enable_audio': prefs.enable_audio,
                'audio_level': prefs.audio_level,
            }
        except Exception:
            _cached_prefs = {'enable_audio': False, 'audio_level': 0.0}
        _cached_prefs_time = now
    return _cached_prefs

def invalidate_audio_prefs_cache():
    """Call this when preferences change to force immediate refresh."""
    global _cached_prefs
    _cached_prefs = None


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
        # PERF: Use cached preferences instead of direct Blender lookup
        prefs = _get_cached_audio_prefs()
        if not prefs['enable_audio'] or prefs['audio_level'] <= 0:
            self.stop_current_sound()
            self.last_state = None
            return

        # State changed → (re)start the appropriate sound once
        if new_state != self.last_state:
            self.stop_current_sound()
            self.last_state = new_state
            self.play_sound_for_state(new_state)
            return

        # Same state as last tick:
        # - Continuous states (WALK/RUN/FALL): do nothing; they loop on the device.
        # - One-shots (JUMP/LAND): if finished, stop/clear (do NOT restart here).
        if new_state in (AnimState.WALK, AnimState.RUN, AnimState.FALL):
            return

        if new_state in (AnimState.JUMP, AnimState.LAND):
            if self.current_handle and self.current_sound_start_time is not None:
                elapsed = get_game_time() - self.current_sound_start_time
                dur = self.current_sound_duration or 0.0
                if dur > 0.0 and elapsed >= dur:
                    self.stop_current_sound()

    def stop_current_sound(self):
        if self.current_handle:
            try:
                self.current_handle.stop()
            except Exception:
                pass
        self.current_handle = None
        self.current_sound_start_time = None
        self.current_sound_duration = None

    def play_sound_for_state(self, state):
        from ..props_and_utils.exp_properties import get_anim_slot
        # PERF: Use cached preferences instead of direct Blender lookup
        prefs = _get_cached_audio_prefs()
        if not prefs['enable_audio'] or prefs['audio_level'] <= 0:
            return

        # 1) Resolve the sound pointer from the animation slot collection
        scene = bpy.context.scene
        slot = get_anim_slot(scene, state.name)
        if not slot:
            return
        sound_data = slot.sound
        if not sound_data or not sound_data.packed_file:
            return

        # 2) Reuse/extract a temp file for aud
        addon_root = get_addon_path()
        temp_sounds_dir = os.path.join(addon_root, "exp_assets", "Sounds", "temp_sounds")
        os.makedirs(temp_sounds_dir, exist_ok=True)
        temp_path = _get_or_extract_temp_path(sound_data, temp_sounds_dir)
        if not temp_path:
            return

        # 3) If this is a continuous state and something is already playing, leave it
        is_loop = slot.looping
        if is_loop and self.current_handle:
            return

        # 4) Start playback
        device = get_global_audio_manager().device
        try:
            handle = device.play(aud.Sound(temp_path))
        except Exception:
            return

        # 5) Looping vs one-shot
        handle.loop_count = -1 if is_loop else 0
        handle.volume     = prefs['audio_level']
        handle.pitch      = slot.sound_speed

        if is_loop:
            self.current_sound_start_time = None
            self.current_sound_duration   = None
        else:
            dur = getattr(sound_data, "sound_duration", None)
            if dur is None:
                try:
                    dur = handle.length
                except Exception:
                    dur = 2.0
            self.current_sound_start_time = get_game_time()
            self.current_sound_duration   = float(dur)

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


def reset_audio_managers():
    """
    Reset audio manager singletons. Call on game end.
    Stops any playing sounds and clears singleton references.
    """
    global _global_audio_manager, _global_char_audio_mgr

    # Stop character audio first
    if _global_char_audio_mgr is not None:
        _global_char_audio_mgr.stop_current_sound()
        _global_char_audio_mgr = None

    # Stop all audio on the device
    if _global_audio_manager is not None:
        try:
            _global_audio_manager.stop_all()
        except Exception:
            pass
        _global_audio_manager = None

##############################################################################
# TEST OPERATOR (No Local Path Fallback)
##############################################################################
class AUDIO_OT_TestSoundPointer(bpy.types.Operator):
    """Plays a sound from an animation slot by state name."""
    bl_idname = "exp_audio.test_sound_pointer"
    bl_label = "Test Sound (Pointer)"

    state_name: bpy.props.StringProperty(default="WALK")

    def execute(self, context):
        from ..props_and_utils.exp_properties import get_anim_slot
        prefs = context.preferences.addons["Exploratory"].preferences

        slot = get_anim_slot(context.scene, self.state_name)
        if not slot:
            self.report({'WARNING'}, f"No animation slot for state '{self.state_name}'.")
            return {'CANCELLED'}

        sound_data = slot.sound
        if not sound_data or not sound_data.packed_file:
            self.report({'WARNING'}, f"No packed sound assigned to {self.state_name}")
            return {'CANCELLED'}

        if not prefs.enable_audio:
            self.report({'INFO'}, "Audio is disabled in Preferences.")
            return {'CANCELLED'}

        addon_root = get_addon_path()
        temp_dir   = os.path.join(addon_root, "exp_assets", "Sounds", "temp_sounds")
        os.makedirs(temp_dir, exist_ok=True)

        temp_path = extract_packed_sound(sound_data, temp_dir)
        if not temp_path:
            self.report({'ERROR'}, "Failed to extract packed sound.")
            return {'CANCELLED'}

        try:
            handle = aud.Device().play(aud.Sound(temp_path))
            handle.volume = prefs.audio_level
            handle.pitch  = slot.sound_speed
            self.report({'INFO'}, f"Playing packed sound: {sound_data.name}")
            return {'FINISHED'}
        except Exception as ex:
            self.report({'ERROR'}, f"aud error: {ex}")
            return {'CANCELLED'}

def clean_audio_temp():
    """
    1) Stop all currently playing sounds
    2) Clear sound pointers in animation slots
       - does NOT touch other pointer properties for custom reactions
    3) Remove 'exp_*' Sound datablocks from bpy.data.sounds
    4) Sleep a moment to let Windows close file handles
    5) Delete and recreate the temp_sounds folder
    """
    # A) Stop audio playback
    audio_mgr = get_global_audio_manager()
    audio_mgr.stop_all()

    # B) Clear the sound pointers in the animation slots collection
    scn = bpy.context.scene
    if hasattr(scn, "character_anim_slots"):
        for slot in scn.character_anim_slots:
            slot.sound = None

    # C) Remove the actual Sound datablocks in bpy.data.sounds whose name starts with "exp_"
    #    (these are typically the ones appended by Build Character).
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
    temp_sounds_dir = os.path.join(addon_root, "exp_assets", "Sounds", "temp_sounds")
    if os.path.isdir(temp_sounds_dir):
        shutil.rmtree(temp_sounds_dir, ignore_errors=True)
    os.makedirs(temp_sounds_dir, exist_ok=True)

# Cache the extracted temp filepath on the Sound datablock (ID property)
def _get_or_extract_temp_path(sound_data: bpy.types.Sound, temp_dir: str) -> str:
    """
    Returns a reusable temp path for this packed sound.
    Extracts once and caches path on the datablock: sound_data["exp_temp_path"].
    Re-extracts only if missing.
    """
    if not sound_data or not sound_data.packed_file:
        return None
    try:
        cached = sound_data.get("exp_temp_path", None)
    except Exception:
        cached = None

    if cached and os.path.isfile(cached):
        return cached

    path = extract_packed_sound(sound_data, temp_dir)
    if path:
        try:
            sound_data["exp_temp_path"] = path
        except Exception:
            pass
    return path

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
        # assign to the SOUND reaction (independent global reaction)
        scene = context.scene
        reaction = scene.reactions[self.reaction_index]
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
        reaction = scene.reactions[self.reaction_index]
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
    """Pack every Sound datablock into this .blend and refresh its in-memory factory"""
    bl_idname = "exp_audio.pack_all_sounds"
    bl_label = "Pack All Sounds"

    def execute(self, context):
        packed = 0
        for snd in bpy.data.sounds:
            # only pack if it’s external and not already packed
            if snd.filepath and not snd.packed_file:
                snd.pack()
                # ── NEW ── clear the old external path and rebuild the factory
                snd.filepath = ""
                snd.factory_update()
                packed += 1

        self.report(
            {'INFO'},
            f"Packed {packed} sound(s) into blend and refreshed factories"
        )
        return {'FINISHED'}
    

# ------------------------------------------------------------------------
# Load CHARACTER audio into the .blend file
# ------------------------------------------------------------------------

class EXP_AUDIO_OT_LoadCharacterAudioFile(bpy.types.Operator, ImportHelper):
    """Load an external audio file into a Character Animation Slot"""
    bl_idname = "exp_audio.load_character_audio_file"
    bl_label = "Load Character Sound"
    filename_ext = ".wav;.mp3;.ogg"

    filter_glob: StringProperty(
        default="*.wav;*.mp3;*.ogg",
        options={'HIDDEN'},
        description="Audio file extensions"
    )
    state_name: StringProperty(
        name="State",
        description="Which animation slot state to assign the sound to",
        default="WALK"
    )

    def execute(self, context):
        from ..props_and_utils.exp_properties import get_anim_slot
        sound = bpy.data.sounds.load(self.filepath, check_existing=True)
        sound.pack()
        slot = get_anim_slot(context.scene, self.state_name)
        if not slot:
            self.report({'WARNING'}, f"No animation slot for state '{self.state_name}'.")
            return {'CANCELLED'}
        slot.sound = sound
        self.report(
            {'INFO'},
            f"Loaded '{sound.name}' into slot {self.state_name}"
        )
        return {'FINISHED'}