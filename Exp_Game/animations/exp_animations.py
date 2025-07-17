import bpy
import enum
from ..props_and_utils.exp_time import get_game_time


_global_anim_manager = None

def set_global_animation_manager(mgr):
    """
    Store a reference to the animation manager as a global singleton.
    This is so other modules (like reactions) can get it without import loops.
    """
    global _global_anim_manager
    _global_anim_manager = mgr

def get_global_animation_manager():
    """
    Retrieve the single, global animation manager instance
    (or None if not yet created).
    """
    return _global_anim_manager


############################################
# 1) AnimState & (Optional) Transitions
############################################

class AnimState(enum.Enum):
    IDLE  = 0
    WALK  = 1
    RUN   = 2
    JUMP  = 3
    FALL  = 4
    LAND  = 5

STATE_TRANSITIONS = {
    AnimState.IDLE: {
        'PRESS_SHIFT+WASD': AnimState.RUN,
        'PRESS_WASD':       AnimState.WALK,
        'PRESS_SPACE':      AnimState.JUMP,
    },
    AnimState.WALK: {
        'NO_KEYS':          AnimState.IDLE,
        'PRESS_SHIFT+WASD': AnimState.RUN,
        'PRESS_SPACE':      AnimState.JUMP,
    },
    AnimState.RUN: {
        'NO_SHIFT+WASD':    AnimState.WALK,
        'NO_SHIFT+NO_KEYS': AnimState.IDLE,
        'PRESS_SPACE':      AnimState.JUMP,
    },
    AnimState.JUMP: {
        'FALLING':          AnimState.FALL,
        'LANDED':           AnimState.LAND,
    },
    AnimState.FALL: {
        'LANDED':           AnimState.LAND,
    },
    AnimState.LAND: {
        # We won’t do transitions here,
        # we’ll handle the ending logic in the code
        # that removes the strip.
        'DONE_LANDING':     AnimState.IDLE,
    }
}

def get_user_keymap():
    prefs = bpy.context.preferences.addons["Exploratory"].preferences
    return {
        'forward':  prefs.key_forward,
        'backward': prefs.key_backward,
        'left':     prefs.key_left,
        'right':    prefs.key_right,
        'jump':     prefs.key_jump,
        'run':      prefs.key_run,
    }


############################################
# 2) Single-Track Character Animation Manager
############################################

class AnimationStateManager:

    def __init__(self):
        scene = bpy.context.scene
        self.target_object = scene.target_armature
        self.anim_state = AnimState.IDLE

        # Jump/fall logic (unchanged)
        self._last_is_grounded   = True
        self.air_time            = 0.0
        self.fall_timer          = 0.0
        self.min_fall_time       = 2.0
        self.min_fall_for_land   = 0.25
        self.landing_in_progress = False
        self.jump_played_in_air  = False

        # If we have a one-time action playing => block new
        self.one_time_in_progress = False

        # We keep a single track
        self.track_name     = "exp_character"
        self.active_actions = []

        # Controls how quickly we shift frames
        self.base_speed_factor = 10.0

        # So we don’t re-trigger the same action repeatedly
        self.last_action_name = None

        # For jump => only trigger on press, not hold
        self.jump_key_held = False



    def update(self, keys_pressed, delta_time, is_grounded, vertical_velocity=0.0):
        """
        1) If a custom/one-time action is active => skip default picking
        2) Otherwise, evaluate the state machine => pick an action
        3) Possibly start_action
        4) Scrub
        5) If a one-time ended => stop one_time_in_progress

        Additionally: after LAND starts, wait 0.2 s, then if the player presses
        a movement key, cancel the remaining LAND strip and immediately transition.
        """
        # remember grounded state
        self._last_is_grounded = is_grounded

        # ——— DELAYED CANCEL FOR LAND ANIMATION ———
        land_name = None
        if bpy.context.scene.character_actions.land_action:
            land_name = bpy.context.scene.character_actions.land_action.name

        if (
            self.anim_state == AnimState.LAND and
            self.one_time_in_progress and
            land_name
        ):
            for rec in self.active_actions:
                if rec["action_name"] == land_name:
                    elapsed = get_game_time() - rec["start_time"]
                    if elapsed >= 0.4:  # tweak this delay as desired
                        # if any movement key is held, clear the land strip
                        k = get_user_keymap()
                        move_keys = {k['forward'], k['backward'], k['left'], k['right']}
                        if keys_pressed.intersection(move_keys):
                            obj = self.target_object
                            if obj and obj.animation_data:
                                track = obj.animation_data.nla_tracks.get(self.track_name)
                                if track:
                                    for strip in list(track.strips):
                                        track.strips.remove(strip)
                            # reset so the next frame picks walk/run
                            self.active_actions.clear()
                            self.one_time_in_progress = False
                            self.landing_in_progress   = False
                            self.last_action_name      = None
                    break

        # ——— EXTRA OVERRIDE CHECK FOR JUMP ANIMATION ———
        jump_name = None
        if bpy.context.scene.character_actions.jump_action:
            jump_name = bpy.context.scene.character_actions.jump_action.name

        new_action, loop, speed, override, play_fully = self._pick_action(
            keys_pressed, is_grounded, vertical_velocity, delta_time
        )
        if (
            not new_action and
            self.one_time_in_progress and
            (self.last_action_name == jump_name)
        ):
            for rec in self.active_actions:
                if rec["action_name"] == jump_name:
                    expected = rec["action_length"] / (self.base_speed_factor * speed)
                    if (get_game_time() - rec["start_time"]) >= expected * 0.9:
                        self.one_time_in_progress = False
                    break

        # ——— A) If any one-time action still playing, just scrub it ———
        if self.one_time_in_progress:
            self.scrub_nla_strips(delta_time)
            return None

        # ——— B) Start a new default action if changed ———
        if new_action and (new_action != self.last_action_name):
            self.start_action(
                action_name=new_action,
                action_type="DEFAULT",
                loop=loop,
                override=override,
                chain=None,
                speed=speed,
                play_fully=play_fully
            )

        # ——— C) Advance all strips ———
        self.scrub_nla_strips(delta_time)

        # ——— D) If a one-time ended without us noticing, clear the flag ———
        if self.one_time_in_progress and (not self._current_has_one_time()):
            self.one_time_in_progress = False

        return new_action


    # -------------------------------------------------
    # 1) Decide which action from state (unchanged)
    # -------------------------------------------------
    def _pick_action(self, keys_pressed, is_grounded, vertical_velocity, delta_time):
        new_state = self._update_state_machine(keys_pressed, is_grounded, vertical_velocity, delta_time)
        scene = bpy.context.scene
        char  = scene.character_actions

        def safe_speed(act):
            return act.action_speed if (act and hasattr(act, "action_speed")) else 1.0

        if new_state == AnimState.IDLE:
            act = char.idle_action
            a_name = act.name if act else None
            spd    = safe_speed(act)
            return (a_name, True, spd, False, False)

        elif new_state == AnimState.WALK:
            act = char.walk_action
            a_name = act.name if act else None
            spd    = safe_speed(act)
            return (a_name, True, spd, False, False)

        elif new_state == AnimState.RUN:
            act = char.run_action
            a_name = act.name if act else None
            spd    = safe_speed(act)
            return (a_name, True, spd, False, False)

        elif new_state == AnimState.JUMP:
            act = char.jump_action
            a_name = act.name if act else None
            spd    = safe_speed(act)
            # jump: one-time (non-looping), override and must play fully
            return (a_name, False, spd, True, True)

        elif new_state == AnimState.FALL:
            act = char.fall_action
            a_name = act.name if act else None
            spd    = safe_speed(act)
            return (a_name, True, spd, False, False)

        elif new_state == AnimState.LAND:
            act = char.land_action
            a_name = act.name if act else None
            spd    = safe_speed(act)
            return (a_name, False, spd, True, True)

        return (None, True, 1.0, False, False)

    # -------------------------------------------------
    # 2) State machine (unchanged)
    # -------------------------------------------------
    def _update_state_machine(self, keys_pressed, is_grounded, vertical_velocity, delta_time):
        # EXACT old logic
        k = get_user_keymap()
        jump_is_down       = (k['jump'] in keys_pressed)
        just_pressed_jump  = (jump_is_down and not self.jump_key_held)
        self.jump_key_held = jump_is_down

        new_state = self.anim_state

        if new_state == AnimState.LAND and self.landing_in_progress:
            return new_state

        if is_grounded:
            self.jump_played_in_air = False

            if new_state == AnimState.FALL:
                if self.fall_timer >= self.min_fall_for_land:
                    new_state = AnimState.LAND
                    self.landing_in_progress = True
                else:
                    new_state = self._handle_movement_input(keys_pressed, is_grounded, just_pressed_jump)
                self.air_time   = 0.0
                self.fall_timer = 0.0
            elif new_state == AnimState.JUMP:
                new_state = self._handle_movement_input(keys_pressed, is_grounded, just_pressed_jump)
                self.air_time   = 0.0
                self.fall_timer = 0.0
            else:
                self.air_time   = 0.0
                self.fall_timer = 0.0
                new_state = self._handle_movement_input(keys_pressed, is_grounded, just_pressed_jump)
        else:
            self.air_time += delta_time
            if new_state == AnimState.JUMP:
                if self.air_time >= self.min_fall_time:
                    new_state = AnimState.FALL
                    self.fall_timer = 0.0
            elif new_state not in (AnimState.JUMP, AnimState.FALL):
                if self.air_time >= self.min_fall_time:
                    new_state = AnimState.FALL
                    self.fall_timer = 0.0
            elif new_state == AnimState.FALL:
                self.fall_timer += delta_time

        self.anim_state = new_state
        return new_state

    def _handle_movement_input(self, keys_pressed, is_grounded, just_pressed_jump):
        # EXACT old logic
        k = get_user_keymap()

        if just_pressed_jump and is_grounded and (not self.jump_played_in_air):
            self.jump_played_in_air = True
            return AnimState.JUMP

        move_keys = {k['forward'], k['backward'], k['left'], k['right']}
        pressed_move = keys_pressed.intersection(move_keys)
        run_held = (k['run'] in keys_pressed)

        if pressed_move and run_held:
            return AnimState.RUN
        elif pressed_move:
            return AnimState.WALK
        else:
            return AnimState.IDLE


    # -------------------------------------------------
    # 3) start_action => forcibly clear track (unchanged)
    # -------------------------------------------------
    def start_action(
        self,
        action_name,
        action_type="DEFAULT",
        loop=False,
        override=False,
        chain=None,
        speed=1.0,
        play_fully=False
    ):
        if not action_name:
            return
        obj = self.target_object
        if not obj:
            return

        act = bpy.data.actions.get(action_name)
        if not act:
            return

        # If not loop => one_time_in_progress
        if not loop:
            self.one_time_in_progress = True

        # If "override" => remove any existing strips no matter what
        self._ensure_track(obj)
        track = obj.animation_data.nla_tracks.get(self.track_name)

        for s in list(track.strips):
            track.strips.remove(s)
        self.active_actions.clear()

        a_len = act.frame_range[1] - act.frame_range[0]
        new_strip = track.strips.new(act.name, start=0, action=act)
        new_strip.frame_start = 0
        new_strip.frame_end   = a_len
        new_strip.blend_in  = 0.0
        new_strip.blend_out = 0.0

        obj.animation_data.action = None

        record = {
            "action_name":   action_name,
            "type":          action_type,
            "loop":          loop,
            "override":      override,
            "chain":         chain,
            "speed":         speed,
            "strip":         new_strip,
            "action_length": a_len,
            "play_fully":    play_fully,

            # Add optional loop_duration if custom wants it; default=0 => no forced end
            "loop_duration": 0.0,
            "start_time":    get_game_time()
        }
        self.active_actions.append(record)
        self.last_action_name = action_name


    # -------------------------------------------------
    # 4) scrub => shift frames each frame (time-based cutoff for custom loops)
    # -------------------------------------------------
    def scrub_nla_strips(self, delta_time):
        """
        EXACT your code, plus an early check for "CUSTOM_CHAR" with a time-limited loop.
        If rec["loop_duration"] > 0 and the time is up, we remove it immediately
        without waiting for the strip frames to end.
        """

        obj = self.target_object
        if not obj or not obj.animation_data:
            return
        track = obj.animation_data.nla_tracks.get(self.track_name)
        if not track:
            return

        now = get_game_time()
        to_remove = []

        for rec in self.active_actions:
            s     = rec["strip"]
            loop  = rec["loop"]
            a_len = rec["action_length"]
            spd   = rec["speed"]

            # -- (A) If this is a custom-labeled record (e.g. type="CUSTOM_CHAR")
            #        and they stored some loop_duration>0.0 => forcibly remove
            #        once real time has passed, ignoring frames.
            #        (We do this check BEFORE seeing if s.frame_end<=0).
            if rec.get("type") == "CUSTOM_CHAR" and loop:
                loop_dur = rec.get("loop_duration", 0.0)
                if loop_dur > 0.0:
                    elapsed = now - rec.get("start_time", 0.0)
                    if elapsed >= loop_dur:
                        # time is up => remove it right now
                        to_remove.append(rec)
                        continue  # skip the normal frame logic below

            # -- (B) Normal shifting logic (unchanged)
            shift_amount = delta_time * self.base_speed_factor * spd
            s.frame_start -= shift_amount
            s.frame_end   -= shift_amount

            if s.frame_end <= 0:
                if loop:
                    # wrap around
                    s.frame_start = 0
                    s.frame_end   = a_len
                else:
                    to_remove.append(rec)

        # remove ended
        for r in to_remove:
            st = r["strip"]
            for existing_strip in track.strips:
                if existing_strip == st:
                    track.strips.remove(existing_strip)
                    break

            self.active_actions.remove(r)
            self.last_action_name = None

            if self.one_time_in_progress:
                self.one_time_in_progress = False

            # If this action was the LAND anim => forcibly set anim_state=IDLE
            if (self.anim_state == AnimState.LAND):
                self.anim_state = AnimState.IDLE
                self.landing_in_progress = False

            # If there's a 'chain'
            nxt = r.get("chain", None)
            if nxt:
                self.start_action(nxt, loop=False, override=False, chain=None, speed=1.0)

    # -------------------------------------------------
    # 5) track & helper (unchanged)
    # -------------------------------------------------
    def _ensure_track(self, obj):
        """
        Removes all other NLA tracks, ensures 'exp_character' track,
        clears .action so it doesn't appear in Action Editor.
        """
        if not obj.animation_data:
            obj.animation_data_create()

        existing = None
        remove_list = []
        for t in obj.animation_data.nla_tracks:
            if t.name == self.track_name:
                existing = t
            else:
                remove_list.append(t)

        for t in remove_list:
            obj.animation_data.nla_tracks.remove(t)

        if not existing:
            existing = obj.animation_data.nla_tracks.new()
            existing.name = self.track_name

        obj.animation_data.action = None

    def _current_has_one_time(self):
        # Return True if any record has loop=False
        return any(not rec["loop"] for rec in self.active_actions)
    

    def play_char_action(self, action, loop=False, loop_duration=0.0):
        if not action:
            return
        obj = self.target_object
        if not obj:
            return
        
        self._ensure_track(obj)
        track = obj.animation_data.nla_tracks.get(self.track_name)

        for s in list(track.strips):
            track.strips.remove(s)
        self.active_actions.clear()

        a_len = action.frame_range[1] - action.frame_range[0]
        new_strip = track.strips.new(action.name, start=0, action=action)
        new_strip.frame_start = 0
        new_strip.frame_end   = a_len
        new_strip.blend_in    = 0.0
        new_strip.blend_out   = 0.0

        obj.animation_data.action = None

        record = {
            "action_name":   action.name,
            # Change 'CHAR_REACTION' --> 'CUSTOM_CHAR'
            "type":          "CUSTOM_CHAR",
            "loop":          loop,
            "loop_duration": loop_duration,
            "strip":         new_strip,
            "action_length": a_len,
            "speed":         1.0,
            "start_time":    get_game_time(),
        }
        self.active_actions.append(record)

        self.one_time_in_progress = True

