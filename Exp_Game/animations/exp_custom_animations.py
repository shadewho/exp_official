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
    target_obj = reaction.custom_action_target
    action_ref = reaction.custom_action_action
    if not target_obj or not action_ref:
        return

    loop = getattr(reaction, "custom_action_loop", False)
    loop_duration = getattr(reaction, "custom_action_loop_duration", 10.0)

    mgr = get_custom_manager_for_object(target_obj)
    if mgr:
        mgr.play_custom_action(action_ref, loop=loop, loop_duration=loop_duration)


def update_all_custom_managers(delta_time):
    """
    Call this once per frame from your modal or main update.
    This will scrub all the active custom actions for all objects.
    """
    for mgr in _custom_managers.values():
        mgr.scrub_time_update(delta_time)


class CustomActionManager:
    """
    Handles exactly one track named 'exp_custom' on one object,
    and scrubs any strips in that track to simulate playback.

    - NLA frames are shifted by delta_time * base_speed_factor.
    - Loop durations are measured in "game time" (exp_time.get_game_time()).
    """
    def __init__(self, object_name):
        self.object_name = object_name
        self.base_speed_factor = 10.0
        self.active_strips = []

    def play_custom_action(self, action, loop=False, loop_duration=10.0):
        obj = bpy.data.objects.get(self.object_name)
        if not obj:
            return

        # Ensure animation_data
        if not obj.animation_data:
            obj.animation_data_create()

        # Remove all tracks except 'exp_custom'
        existing_custom_track = None
        to_delete = []
        for track in obj.animation_data.nla_tracks:
            if track.name == "exp_custom":
                existing_custom_track = track
            else:
                to_delete.append(track)
        for t in to_delete:
            obj.animation_data.nla_tracks.remove(t)

        # Create 'exp_custom' if needed
        if not existing_custom_track:
            existing_custom_track = obj.animation_data.nla_tracks.new()
            existing_custom_track.name = "exp_custom"

        # Clear active action (so it doesn't appear in the Action Editor)
        obj.animation_data.action = None

        # Check for existing strip
        a_len = action.frame_range[1] - action.frame_range[0]
        existing_strip = None
        for s in existing_custom_track.strips:
            if s.action == action:
                existing_strip = s
                break

        # We'll get the current game time for loop timing
        now = get_game_time()

        if existing_strip:
            # Reset start/end
            existing_strip.frame_start = 0
            existing_strip.frame_end = a_len

            # See if it's already in self.active_strips
            found_data = None
            for d in self.active_strips:
                if d["strip"] == existing_strip:
                    found_data = d
                    break

            if not found_data:
                # Add a new dictionary entry
                self.active_strips.append({
                    "track": existing_custom_track,
                    "strip": existing_strip,
                    "action_length": a_len,
                    "loop": loop,
                    "loop_duration": loop_duration,
                    "start_time": now  # store the game time when we started
                })
            else:
                # Update existing
                found_data["loop"] = loop
                found_data["loop_duration"] = loop_duration
                found_data["start_time"] = now  # reset so it loops fresh
        else:
            # Create new strip
            new_strip = existing_custom_track.strips.new(action.name, start=0, action=action)
            new_strip.frame_start = 0
            new_strip.frame_end   = a_len
            new_strip.blend_in    = 0.0
            new_strip.blend_out   = 0.0

            self.active_strips.append({
                "track": existing_custom_track,
                "strip": new_strip,
                "action_length": a_len,
                "loop": loop,
                "loop_duration": loop_duration,
                "start_time": now
            })

    def scrub_time_update(self, delta_time):
        """
        1) We shift the strip frames by (delta_time * base_speed_factor),
           so the animation "plays" quickly or slowly.
        2) If loop=True, we measure how long we've been looping by comparing
           (game_time_now - start_time) to loop_duration.
        """
        if not self.active_strips:
            return

        obj = bpy.data.objects.get(self.object_name)
        if not obj or not obj.animation_data:
            return

        # We'll do the NLA scrubbing by shift_amount:
        shift_amount = delta_time * self.base_speed_factor
        to_remove = []

        now = get_game_time()

        for data in self.active_strips:
            strip = data["strip"]
            loop = data["loop"]
            a_len = data["action_length"]
            loop_duration = data["loop_duration"]
            start_time = data["start_time"]

            # A) If loop=True => check if time is up
            if loop:
                elapsed = now - start_time
                if elapsed >= loop_duration:
                    # done looping
                    to_remove.append(data)
                    continue

            # B) Shift frames for "playback"
            strip.frame_start -= shift_amount
            strip.frame_end   -= shift_amount

            # C) If we reached the end of the strip
            if strip.frame_end <= 0:
                if loop:
                    # Reset to [0..a_len]
                    strip.frame_start = 0
                    strip.frame_end   = a_len
                else:
                    to_remove.append(data)

        # Finally remove any strips flagged
        for d in to_remove:
            strip = d["strip"]
            track = d["track"]
            # Safely remove the strip from the track if it still exists
            if any(s is strip for s in track.strips):
                track.strips.remove(strip)
            # Remove from our active list
            if d in self.active_strips:
                self.active_strips.remove(d)
