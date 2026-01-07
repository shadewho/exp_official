# File: exp_reactions.py
import bpy
import os
import aud
from ..audio.exp_globals import _sound_tasks, SoundTask
import mathutils
from ..props_and_utils.exp_time import get_game_time
from . import exp_custom_ui
from ..audio.exp_audio import extract_packed_sound
from ...exp_preferences import get_addon_path
from .exp_transforms import _active_transform_tasks
from .exp_tracking import clear as clear_tracking_tasks
from ..developer.dev_logger import log_game
# ------------------------------
# Property Tasks
# ------------------------------

_active_property_tasks = []

class PropertyTask:
    """
    Interpolates from old_val -> new_val over `duration`.
    Once finished, if reset_enabled => we schedule a second task from new_val -> old_val.
    """
    def __init__(self, path_str, old_val, new_val, start_time, duration,
                 reset_enabled=False, reset_delay=0.0):
        self.path_str       = path_str
        self.old_val        = old_val
        self.new_val        = new_val
        self.start_time     = start_time
        self.duration       = duration
        self.reset_enabled  = reset_enabled
        self.reset_delay    = reset_delay
        self.finished       = False
        self.end_time       = start_time + duration

    def update(self, now):
        if self.finished:
            return True

        duration = self.duration
        if duration <= 0:
            # instant
            _assign_safely(self.path_str, self.new_val)
            self.finished = True
            return True

        # A) Compute alpha, then clamp it to [0..1]
        alpha = (now - self.start_time) / duration
        clamped_alpha = max(0.0, min(alpha, 1.0))


        # If we are >= 1.0 => we’re done
        if clamped_alpha >= 1.0:
            _assign_safely(self.path_str, self.new_val)
            self.finished = True
            return True

        # partial interpolation
        cur_val = _lerp_value(self.old_val, self.new_val, clamped_alpha)
        _assign_safely(self.path_str, cur_val)
        return False




def _lerp_value(old_val, new_val, alpha, is_int=False, is_bool=False):
    """
    Interpolate from old_val to new_val by 'alpha' in [0..1],
    with optional casting if the property is int or bool.
    """
    if is_bool:
        # For booleans, we can’t do a continuous lerp.
        # Simple approach: if alpha < 0.5 => old_val, else new_val
        return new_val if (alpha >= 0.5) else old_val

    if isinstance(old_val, (list, tuple)) and isinstance(new_val, (list, tuple)):
        # Vector interpolation
        if len(old_val) != len(new_val):
            return new_val  # fallback
        out = []
        for a, b in zip(old_val, new_val):
            # float-lerp each component
            c = a + (b - a)*alpha
            out.append(int(round(c)) if is_int else c)
        return out

    if isinstance(old_val, (float, int)) and isinstance(new_val, (float, int)):
        # numeric
        c = old_val + (new_val - old_val)*alpha
        return int(round(c)) if is_int else c

    # fallback or if string => no real interpolation
    return new_val


def _set_property_value(path_str, val):
    """Assign val to path_str using exec or partial indexing for arrays."""
    try:
        old_val = eval(path_str, {"bpy":bpy, "mathutils":mathutils})
        if (isinstance(old_val, (list,tuple,mathutils.Vector)) 
            and isinstance(val, list)):
            # partial assignment
            for i in range(min(len(old_val),len(val))):
                statement = f"{path_str}[{i}] = {val[i]}"
                exec(statement, {"bpy":bpy,"mathutils":mathutils})
        else:
            # single assignment
            statement = f"{path_str} = {val}"
            exec(statement, {"bpy":bpy,"mathutils":mathutils})
    except Exception as ex:
        pass

def execute_property_reaction(r):
    path_str = r.property_data_path.strip()
    if not path_str:
        return
    # NOTE: We don't eval the current value here anymore - it was never used.
    # The default_val (user-defined) is used as the starting value instead.
    # This saves an expensive eval() per reaction execution.

    # Use the user-defined default value as the starting value.
    if r.property_type == "BOOL":
        default_val = r.default_bool_value
    elif r.property_type == "INT":
        default_val = r.default_int_value
    elif r.property_type == "FLOAT":
        default_val = r.default_float_value
    elif r.property_type == "STRING":
        default_val = r.default_string_value
    elif r.property_type == "VECTOR":
        default_val = list(r.default_vector_value[:r.vector_length])
    else:
        return

    # Determine the target new value:
    if   r.property_type == "BOOL":   new_val = r.bool_value
    elif r.property_type == "INT":    new_val = r.int_value
    elif r.property_type == "FLOAT":  new_val = r.float_value
    elif r.property_type == "STRING": new_val = r.string_value
    elif r.property_type == "VECTOR":
        new_val = list(r.vector_value[:r.vector_length])
    else:
        return

    start_time = get_game_time()
    duration = r.property_transition_duration
    reset_en = r.property_reset
    reset_dly = r.property_reset_delay

    # Use the default value (provided by the user) as the 'old value'
    old_val = default_val

    if duration <= 0:
        _set_property_value(path_str, new_val)
        if reset_en:
            revert_start = get_game_time() + reset_dly
            pt2 = PropertyTask(
                path_str=path_str,
                old_val=new_val,
                new_val=old_val,
                start_time=revert_start,
                duration=duration,  # instant transition
                reset_enabled=False,
                reset_delay=0.0
            )
            _active_property_tasks.append(pt2)
    else:
        pt = PropertyTask(
            path_str=path_str,
            old_val=old_val,
            new_val=new_val,
            start_time=start_time,
            duration=duration,
            reset_enabled=reset_en,
            reset_delay=reset_dly
        )
        _active_property_tasks.append(pt)


def update_property_tasks():
    """
    Called each frame. 
    If a forward task finishes and has reset_enabled => 
    schedule a second revert task with the same duration.
    """
    now = get_game_time()
    to_remove = []

    for i, task in enumerate(_active_property_tasks):
        done = task.update(now)
        if done:
            if task.reset_enabled:
                # schedule revert
                revert_start = task.end_time + task.reset_delay
                # second task from new_val-> old_val
                pt2 = PropertyTask(
                    path_str=task.path_str,
                    old_val=task.new_val,
                    new_val=task.old_val,
                    start_time=revert_start,
                    duration=task.duration,  # same duration
                    reset_enabled=False,
                    reset_delay=0.0
                )
                _active_property_tasks.append(pt2)
            to_remove.append(i)

    for i in reversed(to_remove):
        _active_property_tasks.pop(i)

def _assign_safely(path_str, val):
    """
    Evaluate the old property reference, see if it's int/bool/float,
    cast accordingly, then do the assignment. This ensures we
    never pass float to an int property, etc.
    """
    try:
        old_val = eval(path_str, {"bpy":bpy, "mathutils":mathutils})
    except Exception as ex:
        return

    # Check type
    if isinstance(old_val, bool):
        # set to either True or False
        val = bool(val)
    elif isinstance(old_val, int):
        # cast val to int
        # e.g. val = round(val)
        val = int(round(float(val)))
    elif isinstance(old_val, float):
        # cast val to float
        val = float(val)
    elif isinstance(old_val, (list, tuple, mathutils.Vector)):
        # Possibly do the same logic per component
        new_list = []
        for orig, x in zip(old_val, val):
            if isinstance(orig, bool):
                new_list.append(bool(x))
            elif isinstance(orig, int):
                new_list.append(int(round(float(x))))
            elif isinstance(orig, float):
                new_list.append(float(x))
            else:
                new_list.append(x)
        val = new_list

    # now do the actual assignment
    try:
        if isinstance(old_val, (list,tuple,mathutils.Vector)) and isinstance(val, list):
            for i in range(min(len(old_val), len(val))):
                statement = f"{path_str}[{i}] = {val[i]}"
                # Debug
                exec(statement, {"bpy":bpy,"mathutils":mathutils})
        else:
            statement = f"{path_str} = {val}"
            exec(statement, {"bpy":bpy,"mathutils":mathutils})
    except Exception as ex:
        pass

def reset_all_tasks():
    _active_property_tasks.clear()
    _active_transform_tasks.clear()
    try:
        from . import exp_projectiles as _proj
        _proj.clear()
    except Exception:
        pass
    try:
        clear_tracking_tasks()
    except Exception:
        pass


def execute_char_action_reaction(r):
    """
    Execute a character action reaction.

    If bone_group is ALL: uses AnimationController for full body
    If bone_group is partial: uses BlendSystem to overlay on specific body part
    """
    from ..modal.exp_modal import get_active_modal_operator
    from ..animations.blend_system import get_blend_system

    op = get_active_modal_operator()
    if not op or not hasattr(op, 'anim_controller') or not op.anim_controller:
        return

    act = getattr(r, "char_action_ref", None)
    if not act:
        return

    armature = bpy.context.scene.target_armature
    if not armature:
        return

    ctrl = op.anim_controller
    action_name = act.name

    # Check if action is baked
    if not ctrl.has_animation(action_name):
        log_game("ANIMATIONS", f"CHAR_ACTION_SKIP not_baked anim={action_name}")
        return

    # Get properties
    mode = getattr(r, "char_action_mode", "PLAY_ONCE")
    loop = (mode == "LOOP")
    speed = max(0.05, float(getattr(r, "char_action_speed", 1.0) or 1.0))
    blend_time = float(getattr(r, "char_action_blend_time", 0.15) or 0.15)
    bone_group = getattr(r, "char_action_bone_group", "ALL")
    loop_duration = float(getattr(r, "char_action_loop_duration", 10.0) or 10.0)

    # If full body, use AnimationController directly
    if bone_group == "ALL":
        ctrl.play(
            armature.name,
            action_name,
            weight=1.0,
            speed=speed,
            looping=loop,
            fade_in=blend_time,
            replace=True
        )
        log_game("ANIMATIONS", f"CHAR_ACTION_PLAY anim={action_name} body=ALL speed={speed:.2f}")
    else:
        # Partial body - use BlendSystem overlay
        # Locomotion continues normally, overlay plays on specific bones
        if not blend_sys:
            log_game("ANIMATIONS", f"CHAR_ACTION_SKIP no_blend_system anim={action_name}")
            return

        priority = 0

        # For looping, use loop_duration as the total duration
        # For play once, use -1.0 to play full animation once
        override_duration = loop_duration if loop else -1.0

        blend_sys.play_override(
            action_name,
            mask=bone_group,
            weight=1.0,
            speed=speed,
            duration=override_duration,
            fade_in=blend_time,
            fade_out=blend_time,
            looping=loop,
            priority=priority
        )
        log_game("ANIMATIONS", f"CHAR_ACTION_PLAY anim={action_name} body={bone_group} speed={speed:.2f} priority={priority} duration={override_duration:.2f}")


def execute_custom_action_reaction(r):
    """
    Execute a custom action reaction on an arbitrary object.
    Plays an action on the specified target object via the animation controller.
    """
    from ..modal.exp_modal import get_active_modal_operator

    op = get_active_modal_operator()
    if not op or not hasattr(op, 'anim_controller') or not op.anim_controller:
        return

    # Get target object from reaction (property: custom_action_target)
    target_obj = getattr(r, "custom_action_target", None)
    if not target_obj:
        return

    # Get action to play (property: custom_action_action)
    action = getattr(r, "custom_action_action", None)
    if not action:
        return

    ctrl = op.anim_controller
    action_name = action.name

    # Check if action is baked
    if not ctrl.has_animation(action_name):
        log_game("ANIMATIONS", f"CUSTOM_ACTION_SKIP not_baked anim={action_name}")
        return

    # Get playback settings (property: custom_action_loop is a bool)
    loop = bool(getattr(r, "custom_action_loop", False))
    speed = max(0.05, float(getattr(r, "custom_action_speed", 1.0) or 1.0))

    # Play the animation on the target object
    ctrl.play(
        target_obj.name,
        action_name,
        weight=1.0,
        speed=speed,
        looping=loop,
        fade_in=0.1,
        replace=True
    )


def execute_custom_ui_text_reaction(r):

    subtype = r.custom_text_subtype
    anchor  = r.custom_text_anchor
    mx      = r.custom_text_margin_x
    my      = r.custom_text_margin_y
    scale   = r.custom_text_scale
    color   = tuple(r.custom_text_color)

    if subtype == "STATIC":
        # Normal static text
        if r.custom_text_indefinite:
            e_time = None
        else:
            now = get_game_time()
            e_time = now + r.custom_text_duration

        item = exp_custom_ui.add_text_reaction(
            text_str=r.custom_text_value,
            anchor=anchor,
            margin_x=mx,
            margin_y=my,
            scale=scale,
            end_time=e_time,
            color=color,
            font_name = r.custom_text_font,
        )
        item["subtype"] = "STATIC"

    elif subtype == "COUNTER_DISPLAY":
        if r.custom_text_indefinite:
            e_time = None
        else:
            e_time = get_game_time() + r.custom_text_duration

        # Get the current counter value
        scene = bpy.context.scene
        counter_value = "?"
        if r.text_counter_index.isdigit():
            idx = int(r.text_counter_index)
            if hasattr(scene, "counters") and 0 <= idx < len(scene.counters):
                counter_value = scene.counters[idx].current_value

        # Build the display text from separate fields
        if r.custom_text_include_counter:
            display_text = f"{r.custom_text_prefix}{counter_value}{r.custom_text_suffix}"
        else:
            display_text = f"{r.custom_text_prefix}{r.custom_text_suffix}"

        # Create the reaction text item with the composed text
        item = exp_custom_ui.add_text_reaction(
            text_str=r.custom_text_value,
            anchor=r.custom_text_anchor,
            margin_x=r.custom_text_margin_x,
            margin_y=r.custom_text_margin_y,
            scale=r.custom_text_scale,
            end_time=e_time,
            color=tuple(r.custom_text_color),
            font_name=r.custom_text_font,
        )
        item["subtype"] = "COUNTER_DISPLAY"
        item["counter_index"] = r.text_counter_index
        item["custom_text_prefix"] = r.custom_text_prefix
        item["custom_text_suffix"] = r.custom_text_suffix
        item["custom_text_include_counter"] = r.custom_text_include_counter

    elif subtype == "TIMER_DISPLAY":
        # Determine expiration time (or indefinite)
        if r.custom_text_indefinite:
            e_time = None
        else:
            e_time = get_game_time() + r.custom_text_duration

        # Create a new text reaction, with an empty placeholder.
        # update_text_reactions() will fill in the real timer string each frame.
        item = exp_custom_ui.add_text_reaction(
            text_str="",  # placeholder, overwritten by update_text_reactions()
            anchor=r.custom_text_anchor,
            margin_x=r.custom_text_margin_x,
            margin_y=r.custom_text_margin_y,
            scale=r.custom_text_scale,
            end_time=e_time,
            color=tuple(r.custom_text_color),
            font_name=r.custom_text_font,
        )
        item["subtype"] = "TIMER_DISPLAY"
        item["timer_index"] = r.text_timer_index


##----Sound reaction----------------------#
def execute_sound_reaction(r):
    """
    Called when a 'SOUND' reaction fires.
    Now supports:
      - ONCE or DURATION looping
      - Per-reaction relative volume multiplier
      - Optional distance-based volume attenuation
    """
    # 1) Grab the chosen Sound data from ReactionDefinition
    sound_data = r.sound_pointer
    if not sound_data:
        return

    # 2) Must be packed to play
    if not sound_data.packed_file:
        return

    # 3) Global mute check & master volume from prefs
    prefs = bpy.context.preferences.addons["Exploratory"].preferences
    if not prefs.enable_audio or prefs.audio_level <= 0:
        return

    # 4) Prepare extraction folder (temp_sounds)
    addon_root = get_addon_path()
    temp_sounds_dir = os.path.join(addon_root, "exp_assets", "Sounds", "temp_sounds")
    os.makedirs(temp_sounds_dir, exist_ok=True)

    # 5) Extract the packed bytes -> local .wav/.ogg
    temp_path = extract_packed_sound(sound_data, temp_sounds_dir)
    if not temp_path:
        return

    # 6) Prepare the aud device
    device = aud.Device()

    # 7) Compute the base volume = (master volume) * (reaction’s relative volume)
    base_volume = prefs.audio_level * r.sound_volume

    # 8) Play
    if r.sound_play_mode == "DURATION":
        # Loop indefinitely, then schedule a stop
        factory = aud.Sound(temp_path).loop(-1)
        handle = device.play(factory)
        handle.volume = base_volume

        start_time = get_game_time()
        t = SoundTask(
            handle=handle,
            start_time=start_time,
            duration=r.sound_duration,
            mode="DURATION",
            use_distance=r.sound_use_distance,
            dist_object=r.sound_distance_object,
            dist_max=r.sound_max_distance,
            original_volume=base_volume
        )
        _sound_tasks.append(t)

    else:
        # Play once
        factory = aud.Sound(temp_path)
        handle = device.play(factory)
        handle.volume = base_volume

        start_time = get_game_time()
        t = SoundTask(
            handle=handle,
            start_time=start_time,
            duration=0.0,
            mode="ONCE",
            use_distance=r.sound_use_distance,
            dist_object=r.sound_distance_object,
            dist_max=r.sound_max_distance,
            original_volume=base_volume
        )
        _sound_tasks.append(t)


##----Counter reaction----------------------#
def execute_counter_update_reaction(r):
    scene = bpy.context.scene

    # Convert the chosen counter index to int
    if not r.counter_index.isdigit():
        return
    idx = int(r.counter_index)
    if not hasattr(scene, "counters") or idx < 0 or idx >= len(scene.counters):
        return

    counter = scene.counters[idx]

    # ─── perform the counter operation ───────────
    if r.counter_op == "ADD":
        counter.current_value += r.counter_amount

    elif r.counter_op == "SUBTRACT":
        counter.current_value -= r.counter_amount
        if counter.current_value < 0:
            counter.current_value = 0

    elif r.counter_op == "RESET":
        counter.current_value = counter.default_value
    # ───────────────────────────────────────────────

    # ─── NOW clamp to min/max if enabled ─────────
    if getattr(counter, "use_min_limit", False) and counter.current_value < counter.min_value:
        counter.current_value = counter.min_value

    if getattr(counter, "use_max_limit", False) and counter.current_value > counter.max_value:
        counter.current_value = counter.max_value
    # ───────────────────────────────────────────────


##----Timer reaction----------------------#
def execute_timer_control_reaction(r):
    scene = bpy.context.scene
    if not r.timer_index.isdigit():
        return
    idx = int(r.timer_index)
    if not hasattr(scene, "timers") or idx < 0 or idx >= len(scene.timers):
        return
    timer = scene.timers[idx]

    now = get_game_time()

    if r.timer_op == "START":
        # if not interruptible and timer is already running (and not yet finished), skip
        if not r.interruptible and timer.is_active and not timer.just_finished:
            return
        timer.start(now)

    elif r.timer_op == "STOP":
        timer.stop()
