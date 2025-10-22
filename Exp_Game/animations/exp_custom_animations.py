# File: exp_custom_animations.py

import bpy
from ..props_and_utils.exp_time import get_game_time  # <-- Import your game_time system

# A global dictionary: object_name -> CustomActionManager
_custom_managers = {}

def get_custom_manager_for_object(obj):
    """
    Returns a singleton-like CustomActionManager for the given Blender object
    (creates it if necessary).
    """
    if not obj:
        return None
    obj_name = obj.name
    if obj_name not in _custom_managers:
        _custom_managers[obj_name] = CustomActionManager(obj_name)
    return _custom_managers[obj_name]


def execute_custom_action_reaction(reaction):
    target_obj   = reaction.custom_action_target
    action_ref   = reaction.custom_action_action
    if not target_obj or not action_ref:
        return

    loop          = getattr(reaction, "custom_action_loop", False)
    loop_duration = getattr(reaction, "custom_action_loop_duration", 10.0)
    speed         = max(0.05, float(getattr(reaction, "custom_action_speed", 1.0) or 1.0))

    mgr = get_custom_manager_for_object(target_obj)
    if mgr:
        mgr.play_custom_action(
            action_ref,
            loop=loop,
            loop_duration=loop_duration,
            speed=speed,
        )

def update_all_custom_managers(delta_time):
    """
    Call this once per frame from your modal or main update.
    This will scrub all the active custom actions for all objects.
    """
    for mgr in _custom_managers.values():
        mgr.scrub_time_update(delta_time)


def _wipe_track_safe(ad, track):
    """Remove all strips from track, then remove the track. Safe no-op on errors."""
    if not ad or not track:
        return
    try:
        for s in list(track.strips):
            try:
                track.strips.remove(s)
            except Exception:
                pass
        try:
            ad.nla_tracks.remove(track)
        except Exception:
            pass
    except Exception:
        pass


def purge_all_game_custom_nla(scene=None):
    """
    Hard wipe for custom/overlay NLA on game START/RESET.

    Rules:
      • On the target armature: remove EVERY track except 'exp_character',
        and also clear all strips from 'exp_character' so base starts clean.
      • On all other objects: remove tracks named 'exp_custom' or 'exp_char_custom'
        (legacy), any track tagged exp_overlay==True, and any empty track.
      • Always clear obj.animation_data.action = None when touching NLA.
    """
    import bpy
    if scene is None:
        scene = bpy.context.scene

    base_name = "exp_character"
    arm = getattr(scene, "target_armature", None)

    for obj in list(bpy.data.objects):
        ad = getattr(obj, "animation_data", None)
        if not ad:
            continue

        # —— Armature: keep the base track only (but clear its strips) ——
        if obj == arm:
            # remove every non-base track outright
            for tr in list(ad.nla_tracks):
                if tr.name != base_name:
                    _wipe_track_safe(ad, tr)
            # clear base strips
            base = ad.nla_tracks.get(base_name)
            if base:
                for s in list(base.strips):
                    try:
                        base.strips.remove(s)
                    except Exception:
                        pass
            # ensure action slot is cleared
            try:
                ad.action = None
            except Exception:
                pass
            continue

        # —— Other objects: remove only our custom/overlay tracks ——
        for tr in list(ad.nla_tracks):
            # tag-based (new overlays)
            try:
                is_tagged_overlay = bool(int(tr.get("exp_overlay", 0)))
            except Exception:
                is_tagged_overlay = False

            is_custom_name = (
                tr.name == "exp_custom" or
                tr.name == "exp_char_custom" or               # legacy
                tr.name.startswith("exp_char_custom")         # legacy variants
            )

            if is_tagged_overlay or is_custom_name:
                _wipe_track_safe(ad, tr)
                continue

            # prune empties anywhere (keeps things tidy)
            if len(tr.strips) == 0:
                _wipe_track_safe(ad, tr)

        # clear action slot if we touched anything
        try:
            ad.action = None
        except Exception:
            pass


def stop_custom_actions_and_rewind_strips():
    """
    STOP any per-object custom playback and HARD-WIPE all custom/overlay NLA:
      • Clears managers' active queues.
      • Removes overlay tracks (tagged), 'exp_custom' and legacy 'exp_char_custom' tracks.
      • On the target armature: keeps only 'exp_character' and clears its strips.
      • Prunes any empty tracks.
    """
    # stop per-object managers
    for mgr in _custom_managers.values():
        try:
            mgr.active_strips.clear()
        except Exception:
            pass

    # full purge
    try:
        purge_all_game_custom_nla()
    except Exception:
        pass




class CustomActionManager:
    """
    Handles exactly one track named 'exp_custom' on one object,
    and scrubs any strips in that track to simulate playback.

    - NLA frames are shifted by delta_time * base_speed_factor.
    - Loop durations are measured in "game time" (exp_time.get_game_time()).
    """
    def __init__(self, object_name):
        self.object_name = object_name
        self.base_speed_factor = 30.0
        self.active_strips = []

    def play_custom_action(self, action, loop=False, loop_duration=10.0, speed=1.0):
        obj = bpy.data.objects.get(self.object_name)
        if not obj:
            return

        if not obj.animation_data:
            obj.animation_data_create()

        existing_custom_track = None
        to_delete = []
        for track in obj.animation_data.nla_tracks:
            if track.name == "exp_custom":
                existing_custom_track = track
            else:
                to_delete.append(track)
        for t in to_delete:
            obj.animation_data.nla_tracks.remove(t)

        if not existing_custom_track:
            existing_custom_track = obj.animation_data.nla_tracks.new()
            existing_custom_track.name = "exp_custom"

        obj.animation_data.action = None

        a_len = action.frame_range[1] - action.frame_range[0]
        existing_strip = None
        for s in existing_custom_track.strips:
            if s.action == action:
                existing_strip = s
                break

        now = get_game_time()
        spd = max(0.05, float(speed))

        if existing_strip:
            existing_strip.frame_start = 0
            existing_strip.frame_end   = a_len

            found_data = None
            for d in self.active_strips:
                if d["strip"] == existing_strip:
                    found_data = d
                    break
            if not found_data:
                self.active_strips.append({
                    "track":         existing_custom_track,
                    "strip":         existing_strip,
                    "action_length": a_len,
                    "loop":          loop,
                    "loop_duration": loop_duration,
                    "start_time":    now,
                    "speed":         spd,
                })
            else:
                found_data["loop"]          = loop
                found_data["loop_duration"] = loop_duration
                found_data["start_time"]    = now
                found_data["speed"]         = spd
        else:
            new_strip = existing_custom_track.strips.new(action.name, start=0, action=action)
            new_strip.frame_start = 0
            new_strip.frame_end   = a_len
            new_strip.blend_in    = 0.0
            new_strip.blend_out   = 0.0

            self.active_strips.append({
                "track":         existing_custom_track,
                "strip":         new_strip,
                "action_length": a_len,
                "loop":          loop,
                "loop_duration": loop_duration,
                "start_time":    now,
                "speed":         spd,
            })


    def scrub_time_update(self, delta_time):
        """
        1) Shift strip frames by (delta_time * base_speed_factor).
        2) Stop when loop timer expires or play-once ends.
        Skips if the global NLA write lock is active.
        """
        # Guard against concurrent wipes (reset/start)
        try:
            from ..animations.exp_animations import nla_is_locked
            if nla_is_locked():
                return
        except Exception:
            pass

        if not self.active_strips:
            return

        obj = bpy.data.objects.get(self.object_name)
        if not obj or not obj.animation_data:
            return

        to_remove = []
        now = get_game_time()

        for data in list(self.active_strips):
            strip = data["strip"]
            loop = data["loop"]
            a_len = data["action_length"]
            loop_duration = data["loop_duration"]
            start_time = data["start_time"]
            spd = max(0.05, float(data.get("speed", 1.0)))
            shift_amount = delta_time * self.base_speed_factor * spd

            if loop:
                elapsed = now - start_time
                if elapsed >= loop_duration:
                    to_remove.append(data)
                    continue

            try:
                strip.frame_start -= shift_amount
                strip.frame_end   -= shift_amount
            except Exception:
                to_remove.append(data)
                continue

            if strip.frame_end <= 0:
                if loop:
                    strip.frame_start = 0
                    strip.frame_end   = a_len
                else:
                    to_remove.append(data)

        for d in to_remove:
            strip = d["strip"]
            track = d["track"]
            try:
                if any(s is strip for s in track.strips):
                    track.strips.remove(strip)
            except Exception:
                pass
            try:
                self.active_strips.remove(d)
            except Exception:
                pass
