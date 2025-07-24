# File: exp_reactions.py
import bpy
import time
import os
import aud
from ..audio.exp_globals import _sound_tasks, SoundTask
import mathutils
from mathutils import Vector, Euler, Matrix
from ..props_and_utils.exp_time import get_game_time
from ..animations.exp_animations import get_global_animation_manager
from . import exp_custom_ui
from ..audio.exp_audio import extract_packed_sound
from ...exp_preferences import get_addon_path

# ------------------------------
# TransformTask + Manager
# ------------------------------

_active_transform_tasks = []

class TransformTask:
    def __init__(self, obj, start_loc, start_rot, start_scl,
                 end_loc, end_rot, end_scl,
                 start_time, duration):
        self.obj = obj

        # Store starting transforms:
        self.start_loc = start_loc
        self.start_rot = start_rot
        self.start_scl = start_scl

        # Target transforms:
        self.end_loc = end_loc
        self.end_rot = end_rot
        self.end_scl = end_scl

        self.start_time = start_time
        self.duration = duration

    def update(self, now):
        """
        Returns True if this task has finished;
        otherwise returns False to keep going.
        """
        # ─── Instant-complete zero-length transforms ───────────────
        if self.duration <= 0.0:
            self.obj.location       = self.end_loc
            self.obj.rotation_euler = self.end_rot
            self.obj.scale          = self.end_scl
            return True

        # ─── Normal interpolation ───────────────────────────────────
        t = (now - self.start_time) / self.duration
        if t >= 1.0:
            # Snap to final
            self.obj.location       = self.end_loc
            self.obj.rotation_euler = self.end_rot
            self.obj.scale          = self.end_scl
            return True
        else:
            # Lerp location
            self.obj.location = self.start_loc.lerp(self.end_loc, t)

            # Lerp rotation euler component-wise
            current_rot = self.start_rot.copy()
            current_rot.x = (1.0 - t) * self.start_rot.x + (t * self.end_rot.x)
            current_rot.y = (1.0 - t) * self.start_rot.y + (t * self.end_rot.y)
            current_rot.z = (1.0 - t) * self.start_rot.z + (t * self.end_rot.z)
            self.obj.rotation_euler = current_rot

            # Lerp scale
            self.obj.scale = self.start_scl.lerp(self.end_scl, t)
            return False


def schedule_transform(obj, end_loc, end_rot, end_scl, duration):
    """
    Create a transform animation from the object's *current* transforms
    to the given final transforms (end_loc, end_rot, end_scl),
    taking `duration` seconds.
    """
    if not obj:
        return

    start_loc = obj.location.copy()
    start_rot = obj.rotation_euler.copy()
    start_scl = obj.scale.copy()

    start_time = get_game_time()

    task = TransformTask(
        obj,
        start_loc, start_rot, start_scl,
        end_loc, end_rot, end_scl,
        start_time, duration
    )
    _active_transform_tasks.append(task)

def update_transform_tasks():
    """
    Called once per frame.
    Removes tasks that have finished.
    """
    now = get_game_time()
    finished_indices = []
    for i, task in enumerate(_active_transform_tasks):
        done = task.update(now)
        if done:
            finished_indices.append(i)
    # remove in reverse so indexes don't shift
    for i in reversed(finished_indices):
        _active_transform_tasks.pop(i)

        # In exp_reactions.py

def execute_transform_reaction(reaction):
    """
    Applies a transform reaction to either:
      • the scene’s target_armature (if use_character=True), or
      • the specified transform_object.
    """
    scene = bpy.context.scene

    # 1) Pick target: character vs. user-picked object
    if getattr(reaction, "use_character", False):
        target_obj = scene.target_armature
    else:
        target_obj = reaction.transform_object

    # 2) Bail if nothing to move
    if not target_obj:
        return

    # 3) Ensure Euler XYZ rotation mode
    target_obj.rotation_mode = 'XYZ'

    # 4) Clamp duration
    duration = reaction.transform_duration
    if duration < 0.0:
        duration = 0.0

    # 5) Dispatch based on transform_mode
    mode = reaction.transform_mode

    if mode == "OFFSET":
        # The old approach: interpret location/rotation/scale as global offsets
        apply_offset_transform(reaction, target_obj, duration)

    elif mode == "TO_LOCATION":
        # Interpret location/rotation/scale as absolute world transforms
        apply_to_location_transform(reaction, target_obj, duration)

    elif mode == "TO_OBJECT":
        to_obj = reaction.transform_to_object
        if not to_obj:
            return

        # capture the original transforms
        start_loc = target_obj.location.copy()
        start_rot = target_obj.rotation_euler.copy()
        start_scl = target_obj.scale.copy()

        # pick which channels to override
        end_loc = (
            to_obj.location.copy()
            if reaction.transform_use_location
            else start_loc
        )
        end_rot = (
            to_obj.rotation_euler.copy()
            if reaction.transform_use_rotation
            else start_rot
        )
        end_scl = (
            to_obj.scale.copy()
            if reaction.transform_use_scale
            else start_scl
        )
        # schedule the transform with only the selected channels
        schedule_transform(target_obj, end_loc, end_rot, end_scl, duration)

    elif mode == "LOCAL_OFFSET":
        # Interpret location/rotation/scale as offsets in local space
        apply_local_offset_transform(reaction, target_obj, duration)

def apply_offset_transform(reaction, target_obj, duration):
    """
    Rotate/translate/scale in *global* axes,
    pivoting around the object's current origin.
    """

    loc_off = Vector(reaction.transform_location)
    rot_off = Euler(reaction.transform_rotation, 'XYZ')
    scl_off = Vector(reaction.transform_scale)

    # Build the user offset in global space
    T_off = Matrix.Translation(loc_off)
    R_off = rot_off.to_matrix().to_4x4()
    S_off = Matrix.Diagonal((scl_off.x, scl_off.y, scl_off.z, 1.0))
    user_offset_mat = T_off @ R_off @ S_off

    # The pivot is the object’s current location in world coords
    pivot_world = target_obj.matrix_world.translation

    pivot_inv = Matrix.Translation(-pivot_world)
    pivot_fwd = Matrix.Translation(pivot_world)

    # final_mat = pivot_fwd * user_offset_mat * pivot_inv * current_mat
    start_mat = target_obj.matrix_world.copy()
    offset_mat = pivot_fwd @ user_offset_mat @ pivot_inv
    final_mat = offset_mat @ start_mat

    # Decompose => schedule
    end_loc, end_rot_quat, end_scl = final_mat.decompose()
    end_rot_euler = end_rot_quat.to_euler('XYZ')
    schedule_transform(target_obj, end_loc, end_rot_euler, end_scl, duration)


def apply_to_location_transform(reaction, target_obj, duration):

    # interpret transform_location, transform_rotation, transform_scale
    # as the actual final world transforms
    end_loc = Vector(reaction.transform_location)
    end_rot = Euler(reaction.transform_rotation, 'XYZ')
    end_scl = Vector(reaction.transform_scale)

    # read current
    current_loc = target_obj.location.copy()
    current_rot = target_obj.rotation_euler.copy()
    current_scl = target_obj.scale.copy()

    schedule_transform(target_obj, end_loc, end_rot, end_scl, duration)

def apply_to_object_transform(reaction, target_obj, to_obj, duration):

    # We'll read the to_obj's location, rotation_euler, scale as the final
    end_loc = to_obj.location.copy()
    end_rot = to_obj.rotation_euler.copy()
    end_scl = to_obj.scale.copy()

    # current
    start_loc = target_obj.location.copy()
    start_rot = target_obj.rotation_euler.copy()
    start_scl = target_obj.scale.copy()

    # schedule
    schedule_transform(target_obj, end_loc, end_rot, end_scl, duration)

def apply_local_offset_transform(reaction, target_obj, duration):
    """Apply a LOCAL offset in location/rotation/scale (like R in Local)."""
    loc_off = Vector(reaction.transform_location)
    rot_off = Euler(reaction.transform_rotation, 'XYZ')
    scl_off = Vector(reaction.transform_scale)

    T_off = Matrix.Translation(loc_off)
    R_off = rot_off.to_matrix().to_4x4()
    S_off = Matrix.Diagonal((scl_off.x, scl_off.y, scl_off.z, 1.0))

    offset_mat = T_off @ R_off @ S_off

    start_mat = target_obj.matrix_world.copy()

    # For a local transform, multiply AFTER the existing matrix
    final_mat = start_mat @ offset_mat

    end_loc, end_rot_quat, end_scl = final_mat.decompose()
    end_rot_euler = end_rot_quat.to_euler('XYZ')
    schedule_transform(target_obj, end_loc, end_rot_euler, end_scl, duration)




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
    try:
        current_val = eval(path_str, {"bpy": bpy, "mathutils": mathutils})
    except Exception as ex:
        return

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



def execute_char_action_reaction(r):
    from ..animations.exp_animations import get_global_animation_manager
    mgr = get_global_animation_manager()
    if not mgr or not r.char_action_ref:
        return

    # Decide loop or once from the enum:
    if r.char_action_mode == 'LOOP':
        loop = True
        loop_dur = r.char_action_loop_duration
    else:
        # PLAY_ONCE
        loop = False
        loop_dur = 0.0

    mgr.play_char_action(
        action=r.char_action_ref,
        loop=loop,
        loop_duration=loop_dur
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

    elif subtype == "OBJECTIVE":
        if r.custom_text_indefinite:
            e_time = None
        else:
            e_time = get_game_time() + r.custom_text_duration

        # Here you obtain the current objective value
        # (Assuming that elsewhere in your update loop you update the reaction item's text.)
        # For example, objective_value can be obtained from the scene's objective list.
        scene = bpy.context.scene
        if r.text_objective_index.isdigit():
            idx = int(r.text_objective_index)
            if 0 <= idx < len(scene.objectives):
                objective_value = scene.objectives[idx].current_value
            else:
                objective_value = "?"
        else:
            objective_value = "?"

        # Build the display text from separate fields.
        if r.custom_text_include_counter:
            display_text = f"{r.custom_text_prefix}{objective_value}{r.custom_text_suffix}"
        else:
            display_text = f"{r.custom_text_prefix}{r.custom_text_suffix}"

        # Create the reaction text item with the composed text.
        item = exp_custom_ui.add_text_reaction(
            text_str=r.custom_text_value,
            anchor=r.custom_text_anchor,
            margin_x=r.custom_text_margin_x,
            margin_y=r.custom_text_margin_y,
            scale=r.custom_text_scale,
            end_time=e_time,
            color=tuple(r.custom_text_color),
            font_name = r.custom_text_font,

        )
        item["subtype"] = "OBJECTIVE"
        item["objective_index"] = r.text_objective_index

        # Newly added lines:
        item["custom_text_prefix"] = r.custom_text_prefix
        item["custom_text_suffix"] = r.custom_text_suffix
        item["custom_text_include_counter"] = r.custom_text_include_counter

    elif subtype == "OBJECTIVE_TIMER_DISPLAY":
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
            font_name = r.custom_text_font,
        )
        item["subtype"] = "OBJECTIVE_TIMER_DISPLAY"
        item["objective_index"] = r.text_objective_index


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
    if not prefs.enable_audio:
        print("[execute_sound_reaction] Audio is disabled in Preferences.")
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


##----Objectives reaction----------------------#
def execute_objective_counter_reaction(r):
    scene = bpy.context.scene

    # Convert the chosen objective index to int
    if not r.objective_index.isdigit():
        return
    idx = int(r.objective_index)
    if idx < 0 or idx >= len(scene.objectives):
        return

    objv = scene.objectives[idx]

    # ─── perform the counter operation ───────────
    if r.objective_op == "ADD":
        objv.current_value += r.objective_amount

    elif r.objective_op == "SUBTRACT":
        objv.current_value -= r.objective_amount
        if objv.current_value < 0:
            objv.current_value = 0

    elif r.objective_op == "RESET":
        objv.current_value = objv.default_value
    # ───────────────────────────────────────────────

    # ─── NOW clamp to min/max if enabled ─────────
    if getattr(objv, "use_min_limit", False) and objv.current_value < objv.min_value:
        objv.current_value = objv.min_value

    if getattr(objv, "use_max_limit", False) and objv.current_value > objv.max_value:
        objv.current_value = objv.max_value
    # ───────────────────────────────────────────────


def execute_objective_timer_reaction(r):
    scene = bpy.context.scene
    if not r.objective_index.isdigit():
        return
    idx = int(r.objective_index)
    if idx < 0 or idx >= len(scene.objectives):
        return
    objv = scene.objectives[idx]

    now = get_game_time()

    if r.objective_timer_op == "START":
        # if not interruptible and timer is already running (and not yet finished), skip
        if not r.interruptible and objv.timer_active and not objv.just_finished:
            return
        objv.start_timer(now)

    elif r.objective_timer_op == "STOP":
        objv.stop_timer()