# File: exp_globals.py

# This module just stores references or flags used by multiple files.
# It's intentionally minimal to avoid circular imports.
import bpy
from ..props_and_utils.exp_time import get_game_time



ACTIVE_MODAL_OP = None   # We'll store the active ExpModal operator instance here.

_sound_tasks = []

class SoundTask:
    def __init__(self, handle, start_time, duration, mode,
                 use_distance=False,
                 dist_object=None,
                 dist_max=30.0,
                 original_volume=1.0):
        self.handle = handle
        self.start_time = start_time
        self.duration = duration
        self.mode = mode
        self.finished = False

        # Distance-based fields
        self.use_distance = use_distance
        self.dist_object  = dist_object
        self.dist_max     = dist_max
        self.original_volume = original_volume

    def update(self, now):
        if self.finished:
            return True

        if self.mode == "DURATION":
            if (now - self.start_time) >= self.duration:
                self.handle.stop()
                self.finished = True
                return True
        else:
            pass

        if self.use_distance and self.dist_object:
            arm = bpy.context.scene.target_armature
            if arm:
                d = (arm.location - self.dist_object.location).length
                factor = 0.0 if d >= self.dist_max else 1.0 - (d / self.dist_max)
                try:
                    self.handle.volume = self.original_volume * factor
                except Exception as e:
                    self.finished = True
                    return True

        return False




def update_sound_tasks():
    """
    Called once per frame in the modal update to see if any DURATION tasks need to stop,
    or if any ONCE tasks have ended.
    """
    now = get_game_time()
    done_indices = []
    for i, task in enumerate(_sound_tasks):
        if task.update(now):
            done_indices.append(i)

    # Remove in reverse order so indices stay valid
    for i in reversed(done_indices):
        _sound_tasks.pop(i)


def stop_all_sounds():
    """Force-stop all playing sounds. Call this when the game (modal) ends."""
    for task in _sound_tasks:
        task.handle.stop()
    _sound_tasks.clear()

