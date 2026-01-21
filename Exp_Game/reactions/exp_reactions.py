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
# Property Path Parser (eliminates eval/exec in hot path)
# ------------------------------

import re

# Regex patterns for path parsing
_ATTR_PATTERN = re.compile(r'\.([a-zA-Z_][a-zA-Z0-9_]*)')
_ITEM_PATTERN = re.compile(r'\["([^"]+)"\]|\[\'([^\']+)\'\]')
_INDEX_PATTERN = re.compile(r'\[(\d+)\]')


def _parse_property_path(path_str):
    """
    Parse a property path string into navigation steps.

    Examples:
        'bpy.context.scene.prop' -> [('attr','context'), ('attr','scene'), ('attr','prop')]
        'bpy.data.objects["Cube"].prop' -> [('attr','data'), ('attr','objects'), ('item','Cube'), ('attr','prop')]
        'bpy.context.scene.vec[0]' -> [('attr','context'), ('attr','scene'), ('attr','vec'), ('index',0)]

    Returns:
        list of (step_type, key) tuples
    """
    # Remove leading 'bpy.' as we start from bpy module
    if path_str.startswith('bpy.'):
        path_str = path_str[4:]
    elif path_str.startswith('bpy'):
        path_str = path_str[3:]

    steps = []
    pos = 0
    length = len(path_str)

    while pos < length:
        # Try attribute access: .name
        if path_str[pos] == '.':
            match = _ATTR_PATTERN.match(path_str, pos)
            if match:
                steps.append(('attr', match.group(1)))
                pos = match.end()
                continue

        # Try item access: ["key"] or ['key']
        if path_str[pos] == '[':
            # Check for string key first
            match = _ITEM_PATTERN.match(path_str, pos)
            if match:
                key = match.group(1) or match.group(2)
                steps.append(('item', key))
                pos = match.end()
                continue

            # Check for numeric index
            match = _INDEX_PATTERN.match(path_str, pos)
            if match:
                steps.append(('index', int(match.group(1))))
                pos = match.end()
                continue

        # Try bare attribute at start (e.g., 'context' from 'context.scene')
        match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', path_str[pos:])
        if match:
            steps.append(('attr', match.group(1)))
            pos += match.end()
            continue

        # Skip unexpected characters
        pos += 1

    return steps


def _navigate_path(steps):
    """
    Navigate path steps starting from bpy module.

    Returns:
        (parent_obj, final_step_type, final_key) or (None, None, None) on error
    """
    obj = bpy

    # Navigate to parent (all steps except last)
    for step_type, key in steps[:-1]:
        try:
            if step_type == 'attr':
                obj = getattr(obj, key)
            elif step_type == 'item':
                obj = obj[key]
            elif step_type == 'index':
                obj = obj[key]
        except (AttributeError, KeyError, IndexError, TypeError):
            return None, None, None

    if not steps:
        return None, None, None

    final_type, final_key = steps[-1]
    return obj, final_type, final_key


def _get_property_value_fast(steps):
    """Get property value using pre-parsed path steps (no eval)."""
    parent, final_type, final_key = _navigate_path(steps)
    if parent is None:
        return None

    try:
        if final_type == 'attr':
            return getattr(parent, final_key)
        elif final_type == 'item':
            return parent[final_key]
        elif final_type == 'index':
            return parent[final_key]
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


def _set_property_value_fast(steps, val):
    """Set property value using pre-parsed path steps (no exec)."""
    parent, final_type, final_key = _navigate_path(steps)
    if parent is None:
        return False

    try:
        if final_type == 'attr':
            setattr(parent, final_key, val)
        elif final_type == 'item':
            parent[final_key] = val
        elif final_type == 'index':
            parent[final_key] = val
        return True
    except (AttributeError, KeyError, IndexError, TypeError):
        return False


# ------------------------------
# Property Tasks
# ------------------------------

_active_property_tasks = []

class PropertyTask:
    """
    Interpolates from old_val -> new_val over `duration`.
    Once finished, if reset_enabled => we schedule a second task from new_val -> old_val.

    PERFORMANCE: Path is pre-parsed once at creation. Updates use direct
    getattr/setattr instead of eval/exec (10-100x faster per frame).
    """
    def __init__(self, path_str, old_val, new_val, start_time, duration,
                 reset_enabled=False, reset_delay=0.0, parsed_steps=None):
        self.path_str       = path_str
        self.old_val        = old_val
        self.new_val        = new_val
        self.start_time     = start_time
        self.duration       = duration
        self.reset_enabled  = reset_enabled
        self.reset_delay    = reset_delay
        self.finished       = False
        self.end_time       = start_time + duration

        # Pre-parse path for fast assignment (avoids eval/exec in hot path)
        # Can be passed in to avoid re-parsing for reset tasks
        self._parsed_steps = parsed_steps if parsed_steps else _parse_property_path(path_str)

        # Determine value type for proper casting during interpolation
        self._is_vector = isinstance(old_val, (list, tuple))

    def _assign_value(self, val):
        """Fast assignment using pre-parsed path (no eval/exec)."""
        if self._is_vector:
            # For vectors, we need to assign component-by-component
            # because Blender vector properties don't accept list assignment directly
            base_steps = self._parsed_steps
            for i, component in enumerate(val):
                # Append index step for each component
                component_steps = base_steps + [('index', i)]
                _set_property_value_fast(component_steps, component)
        else:
            _set_property_value_fast(self._parsed_steps, val)

    def update(self, now):
        if self.finished:
            return True

        duration = self.duration
        if duration <= 0:
            # instant
            self._assign_value(self.new_val)
            self.finished = True
            return True

        # A) Compute alpha, then clamp it to [0..1]
        alpha = (now - self.start_time) / duration
        clamped_alpha = max(0.0, min(alpha, 1.0))

        # If we are >= 1.0 => we're done
        if clamped_alpha >= 1.0:
            self._assign_value(self.new_val)
            self.finished = True
            return True

        # partial interpolation
        cur_val = _lerp_value(self.old_val, self.new_val, clamped_alpha)
        self._assign_value(cur_val)
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
    """
    Assign val to path_str using fast path parsing (no eval/exec).

    PERFORMANCE: Uses pre-parsed path navigation instead of exec().
    """
    steps = _parse_property_path(path_str)
    if not steps:
        return

    # Check if it's a vector assignment
    if isinstance(val, (list, tuple)):
        # For vectors, assign component-by-component
        for i, component in enumerate(val):
            component_steps = steps + [('index', i)]
            _set_property_value_fast(component_steps, component)
    else:
        _set_property_value_fast(steps, val)

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
                # Reuse parsed_steps to avoid re-parsing the path
                pt2 = PropertyTask(
                    path_str=task.path_str,
                    old_val=task.new_val,
                    new_val=task.old_val,
                    start_time=revert_start,
                    duration=task.duration,  # same duration
                    reset_enabled=False,
                    reset_delay=0.0,
                    parsed_steps=task._parsed_steps  # Reuse parsed path
                )
                _active_property_tasks.append(pt2)
            to_remove.append(i)

    for i in reversed(to_remove):
        _active_property_tasks.pop(i)


def _assign_safely(path_str, val):
    """
    Compatibility wrapper for fast property assignment.

    Used by exp_game_reset.py to reset property reactions to defaults.
    Uses the new pre-parsed path system instead of eval/exec.
    """
    _set_property_value(path_str, val)


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
        blend_sys = get_blend_system()
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
