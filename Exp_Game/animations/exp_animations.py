#Exploratory/Exp_Game/animations/exp_animations.py

import bpy
import enum
from ..props_and_utils.exp_time import get_game_time
import re

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

_nla_write_lock = 0

def nla_guard_enter():
    """Enter a global NLA write critical section (re-entrant)."""
    global _nla_write_lock
    _nla_write_lock += 1

def nla_guard_exit():
    """Leave the global NLA write critical section."""
    global _nla_write_lock
    _nla_write_lock = max(0, _nla_write_lock - 1)

def nla_is_locked():
    return _nla_write_lock > 0

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
        self.min_fall_time       = 0.9
        self.min_fall_for_land   = 0.20
        self.landing_in_progress = False
        self.jump_played_in_air  = False

        # If we have a one-time action playing => block new
        self.one_time_in_progress = False

        # We keep a single track
        self.track_name     = "exp_character"

        self.active_actions = []

        # Controls how quickly we shift frames
        self.base_speed_factor = 30.0

        # So we don’t re-trigger the same action repeatedly
        self.last_action_name = None

        # For jump => only trigger on press, not hold
        self.jump_key_held = False


    def _ensure_base_track(self, obj):
        """
        Ensure ONLY the base locomotion track exists.
        Do NOT touch other tracks (overlays, etc).
        """
        if not obj.animation_data:
            obj.animation_data_create()
        ad = obj.animation_data
        base = ad.nla_tracks.get(self.track_name)
        if not base:
            base = ad.nla_tracks.new()
            base.name = self.track_name
        obj.animation_data.action = None  # ensure NLA evaluation
        return base

    def _create_overlay_track(self, obj, action_name: str):
        """
        Create a NEW overlay track, named after the action.
        If a track with that exact name already exists AND has strips, we make a unique suffix (.001, .002, ...).
        We tag the track with a custom prop so resets can find it.
        """
        if not obj.animation_data:
            obj.animation_data_create()
        ad = obj.animation_data

        base = (action_name or "overlay").strip()
        # sanitize (keep it readable for you)
        safe = re.sub(r"[^A-Za-z0-9_.\- ]+", "_", base).strip() or "overlay"

        name = safe
        existing = ad.nla_tracks.get(name)
        if existing is not None and len(existing.strips) == 0:
            tr = existing
        else:
            if existing is not None:
                # find a unique suffix
                idx = 1
                while ad.nla_tracks.get(f"{safe}.{idx:03d}") is not None:
                    idx += 1
                name = f"{safe}.{idx:03d}"
            tr = ad.nla_tracks.new()
            tr.name = name

        # mark as our overlay for safe cleanup later
        try:
            tr["exp_overlay"] = True
            tr["exp_overlay_action"] = safe
        except Exception:
            pass

        # best-effort: move newly created track to TOP (evaluate last)
        try:
            tracks = ad.nla_tracks
            # find current index
            cur_idx = None
            for i, t in enumerate(tracks):
                if t == tr:
                    cur_idx = i
                    break
            if cur_idx is not None and hasattr(tracks, "move"):
                tracks.move(cur_idx, len(tracks) - 1)
        except Exception:
            pass

        obj.animation_data.action = None
        return tr

    def _prune_track_if_empty(self, obj, track):
        """
        If a non-base track has no strips, delete it immediately.
        (We don't rely on tags; any empty non-base track is pruned.)
        """
        if not obj or not getattr(obj, "animation_data", None) or not track:
            return
        try:
            if track.name != self.track_name and len(track.strips) == 0:
                obj.animation_data.nla_tracks.remove(track)
        except Exception:
            pass



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

        # For base track, one-time blocks the state machine until finished
        if not loop:
            self.one_time_in_progress = True

        base_track = self._ensure_base_track(obj)

        # Clear only the BASE track & its records (unchanged behavior)
        self._clear_track_and_records(base_track)

        a_len = act.frame_range[1] - act.frame_range[0]
        new_strip = base_track.strips.new(act.name, start=0, action=act)
        new_strip.frame_start = 0
        new_strip.frame_end   = a_len
        new_strip.blend_in    = 0.0
        new_strip.blend_out   = 0.0

        obj.animation_data.action = None

        record = {
            "action_name":   action_name,
            "type":          action_type,
            "layer":         "BASE",
            "loop":          loop,
            "override":      override,
            "chain":         chain,
            "speed":         speed,
            "strip":         new_strip,
            "track":         base_track,
            "action_length": a_len,
            "play_fully":    play_fully,
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
        Shift frames for all active records (base + overlays).
        Looping records obey loop_duration. Removes finished strips.
        Guarded by the global NLA lock. Robust strip removal with identity/name/action fallbacks.
        Also vacuums any stale empty non-base tracks.
        """
        # Skip while reset/start is wiping tracks
        try:
            if nla_is_locked():
                return
        except Exception:
            pass

        obj = self.target_object
        if not obj or not obj.animation_data:
            return

        ad = obj.animation_data
        now = get_game_time()
        to_remove = []

        # --- advance & mark finished ---
        for rec in list(self.active_actions):
            s   = rec.get("strip")
            trk = rec.get("track")
            if not s or not trk:
                to_remove.append(rec)
                continue

            loop  = bool(rec.get("loop", False))
            a_len = float(rec.get("action_length", 0.0))
            spd   = max(0.05, float(rec.get("speed", 1.0)))

            if loop:
                loop_dur = float(rec.get("loop_duration", 0.0))
                if loop_dur > 0.0:
                    if now - float(rec.get("start_time", 0.0)) >= loop_dur:
                        to_remove.append(rec)
                        continue

            shift_amount = delta_time * self.base_speed_factor * spd
            try:
                s.frame_start -= shift_amount
                s.frame_end   -= shift_amount
            except Exception:
                to_remove.append(rec)
                continue

            if s.frame_end <= 0.0:
                if loop and a_len > 0.0:
                    s.frame_start = 0.0
                    s.frame_end   = a_len
                else:
                    to_remove.append(rec)

        # --- delete strips + prune tracks ---
        for r in to_remove:
            st  = r.get("strip")
            trk = r.get("track")

            if trk:
                # Robust removal: identity OR name OR action+range match
                try:
                    removed = False
                    for s2 in list(trk.strips):
                        if s2 is st or s2.name == getattr(st, "name", None):
                            trk.strips.remove(s2)
                            removed = True
                            break
                    if (not removed) and st and getattr(st, "action", None) is not None:
                        a = st.action
                        fs, fe = float(getattr(st, "frame_start", 0.0)), float(getattr(st, "frame_end", 0.0))
                        for s2 in list(trk.strips):
                            if s2.action == a and abs(s2.frame_start - fs) < 1e-4 and abs(s2.frame_end - fe) < 1e-4:
                                trk.strips.remove(s2)
                                break
                except Exception:
                    pass

            # Drop record regardless (defensive against partial removals)
            try:
                self.active_actions.remove(r)
            except Exception:
                pass

            self.last_action_name = None

            # Unblock base when its one-shot ends
            if self.one_time_in_progress and r.get("layer") == "BASE":
                self.one_time_in_progress = False

            # LAND finishing behavior for base track stays as-is
            if (self.anim_state == AnimState.LAND) and r.get("layer") == "BASE":
                self.anim_state = AnimState.IDLE
                self.landing_in_progress = False

            # Simple chain
            nxt = r.get("chain", None)
            if nxt:
                self.start_action(nxt, loop=False, override=False, chain=None, speed=1.0)

            # Prune empty non-base track immediately
            try:
                if trk and trk.name != self.track_name and len(trk.strips) == 0:
                    ad.nla_tracks.remove(trk)
            except Exception:
                pass

        # --- vacuum pass: clean any stale non-base tracks with no active record ---
        try:
            for tr in list(ad.nla_tracks):
                if tr.name == self.track_name:
                    continue
                # skip tracks still referenced by active records
                if any(rec.get("track") is tr for rec in self.active_actions):
                    continue
                # remove expired strips if any slipped past
                for s in list(tr.strips):
                    try:
                        if s.frame_end <= 0.0:
                            tr.strips.remove(s)
                    except Exception:
                        try:
                            tr.strips.remove(s)
                        except Exception:
                            pass
                if len(tr.strips) == 0:
                    try:
                        ad.nla_tracks.remove(tr)
                    except Exception:
                        pass
        except Exception:
            pass


    # -------------------------------------------------
    # 5) track & helper (unchanged)
    # -------------------------------------------------
    def _ensure_track(self, obj):
        """
        Ensure BOTH allowed tracks exist:
          • 'exp_character'   (base locomotion)
          • 'exp_char_custom' (overlay/blend)
        Remove any other stray tracks. Clear .action.
        """
        if not obj.animation_data:
            obj.animation_data_create()

        allowed = {self.track_name, self.overlay_track_name}
        base_track = None
        overlay_track = None
        to_remove = []

        for t in obj.animation_data.nla_tracks:
            if t.name == self.track_name:
                base_track = t
            elif t.name == self.overlay_track_name:
                overlay_track = t
            else:
                to_remove.append(t)

        for t in to_remove:
            obj.animation_data.nla_tracks.remove(t)

        if not base_track:
            base_track = obj.animation_data.nla_tracks.new()
            base_track.name = self.track_name

        if not overlay_track:
            overlay_track = obj.animation_data.nla_tracks.new()
            overlay_track.name = self.overlay_track_name

        obj.animation_data.action = None
        
    def _clear_track_and_records(self, track):
        """Remove all strips from the given track and drop matching records."""
        if not track:
            return
        # snapshot current strips for filtering active_records
        removed = list(track.strips)
        for s in list(track.strips):
            track.strips.remove(s)
        # keep records not tied to the removed strips
        self.active_actions = [
            rec for rec in self.active_actions
            if rec.get("strip") not in removed and rec.get("track") is not track
        ]
    def _clear_overlay_track(self, obj, delete_track: bool = True):
        """
        Remove ALL strips from the overlay track ('exp_char_custom'), and optionally
        delete the track itself. Safe no-op if track doesn't exist.

        We call this whenever there is no active OVERLAY record so the overlay
        can never influence the base locomotion.
        """
        ad = getattr(obj, "animation_data", None)
        if not ad:
            return
        tr = ad.nla_tracks.get(self.overlay_track_name)
        if not tr:
            return

        # Remove all strips
        for s in list(tr.strips):
            try:
                tr.strips.remove(s)
            except Exception:
                pass

        # Optionally remove the empty track as well
        if delete_track:
            try:
                ad.nla_tracks.remove(tr)
            except Exception:
                pass
    def _current_has_one_time(self):
        # Return True if any record has loop=False
        return any(not rec["loop"] for rec in self.active_actions)
    

    def play_char_action(self, action, loop=False, loop_duration=0.0, speed=1.0, blend=False):
        """
        Character action trigger:
        • blend=False (default): plays on base track 'exp_character' (replaces locomotion like before)
        • blend=True:  creates a NEW overlay track named after the action, adds a single strip,
                        and lets it run independently. When finished, we delete the strip AND the track.
                        Multiple overlays can coexist.
        """
        if not action:
            return
        obj = self.target_object
        if not obj:
            return

        if not obj.animation_data:
            obj.animation_data_create()

        spd = max(0.05, float(speed))

        if not blend:
            # BASE (unchanged)
            base_track = self._ensure_base_track(obj)
            self._clear_track_and_records(base_track)

            a_len = action.frame_range[1] - action.frame_range[0]
            new_strip = base_track.strips.new(action.name, start=0, action=action)
            new_strip.frame_start = 0
            new_strip.frame_end   = a_len
            new_strip.blend_in    = 0.0
            new_strip.blend_out   = 0.0

            obj.animation_data.action = None

            record = {
                "action_name":   action.name,
                "type":          "CUSTOM_CHAR",
                "layer":         "BASE",
                "loop":          bool(loop),
                "loop_duration": float(loop_duration) if loop else 0.0,
                "strip":         new_strip,
                "track":         base_track,
                "action_length": a_len,
                "speed":         spd,
                "start_time":    get_game_time(),
                "play_fully":    not loop,
            }
            self.active_actions.append(record)
            self.one_time_in_progress = True  # preserve old behavior
            return

        # OVERLAY path: create a fresh topmost track named after the action
        tr = self._create_overlay_track(obj, action.name)
        a_len = action.frame_range[1] - action.frame_range[0]
        s = tr.strips.new(action.name, start=0, action=action)
        s.frame_start = 0
        s.frame_end   = a_len
        s.blend_in    = 0.0
        s.blend_out   = 0.0

        obj.animation_data.action = None

        rec = {
            "action_name":   action.name,
            "type":          "CHAR_BLEND",
            "layer":         "OVERLAY",
            "loop":          bool(loop),
            "loop_duration": float(loop_duration) if loop else 0.0,
            "strip":         s,
            "track":         tr,
            "action_length": a_len,
            "speed":         spd,
            "start_time":    get_game_time(),
            "play_fully":    not loop,
        }
        self.active_actions.append(rec)
        # Note: overlays never set one_time_in_progress