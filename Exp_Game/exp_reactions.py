# File: exp_reactions.py
import bpy
import time
import os
import aud
from .exp_globals import _sound_tasks, SoundTask
import mathutils
from mathutils import Vector, Euler, Matrix
from .exp_time import get_game_time
from .exp_animations import get_global_animation_manager
from . import exp_custom_ui
from .exp_mobility_and_game_reactions import MobilityGameReactionsPG
from .exp_audio import extract_packed_sound
from ..exp_preferences import get_addon_path

#---custom propertys--#
def update_property_data_path(self, context):
    """
    Called whenever the user changes 'property_data_path'.
    We'll eval() the string to find the property,
    detect its type, and store the current value.
    """
    # Clear type initially
    self.property_type = "NONE"

    path_str = self.property_data_path.strip()
    if not path_str:
        return  # user left it blank

    # Attempt to evaluate the data path
    try:
        prop_ref = eval(path_str)  # e.g. eval("bpy.data.materials['Mat'].node_tree...")
    except Exception as ex:
        return

    # Now 'prop_ref' is the property. Let's detect its type:
    #  - bool, int, float, str, or array (vector).
    if isinstance(prop_ref, bool):
        self.property_type = "BOOL"
        self.bool_value = prop_ref
    elif isinstance(prop_ref, int):
        self.property_type = "INT"
        self.int_value = prop_ref
    elif isinstance(prop_ref, float):
        self.property_type = "FLOAT"
        self.float_value = prop_ref
    elif isinstance(prop_ref, str):
        self.property_type = "STRING"
        self.string_value = prop_ref
    elif hasattr(prop_ref, "__getitem__") and hasattr(prop_ref, "__len__"):
        # It's likely an array property
        self.property_type = "VECTOR"
        length = len(prop_ref)
        self.vector_length = min(length, 4)
        # copy up to 4 components
        tmp = [prop_ref[i] for i in range(self.vector_length)]
        self.vector_value = tmp
    else:
        self.property_type = "NONE"

#---objective counter--#
def enum_objective_items(self, context):
    """
    Build a list of (identifier, name, description) 
    for each objective in scene.objectives.
    The identifier can be the index as a string.
    """
    if not context:
        return []

    scn = context.scene
    items = []
    for i, objv in enumerate(scn.objectives):
        # The "identifier" must be a string, so let's just use str(i).
        # The "name" is what shows up in the dropdown, so use objv.name or something user-friendly.
        items.append((str(i), objv.name, f"Objective: {objv.name}"))
    return items


class ReactionDefinition(bpy.types.PropertyGroup):
    """
    Represents one 'Reaction' item, but stored in an Interaction’s
    sub-collection. We do not store these at the Scene level.
    """
    name: bpy.props.StringProperty(
        name="Name",
        default="Reaction"
    )

    reaction_type: bpy.props.EnumProperty(
        name="Reaction Type",
        items=[
            ("CUSTOM_ACTION",     "Custom Action", ""),
            ("CHAR_ACTION",       "Character Action", ""),
            ("SOUND",             "Play Sound", ""),
            ("PROPERTY",          "Property Value", ""),
            ("TRANSFORM",         "Transform", ""),
            ("CUSTOM_UI_TEXT",    "Custom UI Text", ""),
            ("OBJECTIVE_COUNTER", "Objective Counter", ""),
            ("OBJECTIVE_TIMER",   "Objective Timer", "Start/Stop an objective's timer"),
            ("MOBILITY_GAME", "Mobility & Game", "Enable/Disable movement, jump, sprint, time, etc.")
        ],
        default="CUSTOM_ACTION"
    )


    # --------------------------------------------------
    # Character Action
    # --------------------------------------------------
    char_action_ref: bpy.props.PointerProperty(
        name="Character Action",
        type=bpy.types.Action,
        description="Which Action to play on the main character"
    )

    char_action_loop_duration: bpy.props.FloatProperty(
        name="Loop Duration (sec)",
        default=10.0,
        min=0.0,
        description="How long a looping action will continue before it stops"
    )

    char_action_mode: bpy.props.EnumProperty(
        name="Play Mode",
        items=[
            ("PLAY_ONCE", "Play Once", "Play the action once and return to default"),
            ("LOOP", "Loop", "Loop the action for up to loop_duration")
        ],
        default="PLAY_ONCE"
    )

    #--------------------------------------------------------
    #AUDIO AUDIO AUDIO
    #--------------------------------------------------------
    sound_pointer: bpy.props.PointerProperty(
        name="Sound Pointer",
        type=bpy.types.Sound,
        description="Which packed Sound to play for this reaction"
    )
    sound_play_mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ("ONCE", "Play Once", "Play the audio once then stop"),
            ("DURATION", "Duration", "Loop until the timer ends"),
        ],
        default="ONCE"
    )
    sound_duration: bpy.props.FloatProperty(
        name="Duration",
        default=5.0,
        min=0.0,
        description="If Mode=Duration, loop for these many seconds"
    )
    sound_volume: bpy.props.FloatProperty(
        name="Relative Volume",
        description="Volume multiplier for this reaction (0..1). Multiplies the global scene.audio_level.",
        default=1.0,
        min=0.0,
        max=1.0
    )

    sound_use_distance: bpy.props.BoolProperty(
        name="Distance Attenuation?",
        description="If True, volume fades out based on distance from armature to the specified object",
        default=False
    )
    sound_distance_object: bpy.props.PointerProperty(
        name="Distance Object",
        type=bpy.types.Object,
        description="Sound source location for distance-based volume"
    )
    sound_max_distance: bpy.props.FloatProperty(
        name="Max Distance",
        default=30.0,
        min=0.0,
        description="Distance at which sound volume becomes 0"
    )

    # --------------------------------------------------
    # 1) TRANSFORM REACTION FIELDS
    # --------------------------------------------------
    # (NEW) The user can pick which object to transform:

    transform_mode: bpy.props.EnumProperty(
        name="Transform Mode",
        description="How we interpret the transform (absolute location, offset, move to object, etc.)",
        items=[
            ("OFFSET",    "Global Offset",    "Add location/rotation/scale to the current transforms (existing behavior)"),
            ("LOCAL_OFFSET","Local Offset","Offset the transforms in local space, rather than world space"),
            ("TO_LOCATION", "To Location", "Teleport or animate to a specific global 3D location (and optional rotation/scale)"),
            ("TO_OBJECT",   "To Object",   "Animate or teleport to another object’s transforms"),
        ],
        default="OFFSET"
    )
    transform_object: bpy.props.PointerProperty(
        name="Transform Object",
        type=bpy.types.Object,
        description="Which object will be transformed?"
    )
    transform_to_object: bpy.props.PointerProperty(
        name="Target Object (To Object)",
        type=bpy.types.Object,
        description="If Transform Mode = 'TO_OBJECT', we use this object’s location/rotation/scale"
    )

    transform_location: bpy.props.FloatVectorProperty(
        name="Location",
        default=(0.0, 0.0, 0.0),
        subtype='TRANSLATION',
        description="Destination location"
    )
    transform_rotation: bpy.props.FloatVectorProperty(
        name="Rotation (Euler)",
        default=(0.0, 0.0, 0.0),
        subtype='EULER',
        description="Destination rotation (XYZ eulers)"
    )
    transform_scale: bpy.props.FloatVectorProperty(
        name="Scale",
        default=(1.0, 1.0, 1.0),
        subtype='XYZ',
        description="Destination scale"
    )
    transform_duration: bpy.props.FloatProperty(
        name="Duration",
        default=1.0,
        min=0.0,
        description="How long the transform should take"
    )

    # --------------------------------------------------
    # 2) CUSTOM UI TEXT REACTION FIELDS
    # --------------------------------------------------
    custom_text_value: bpy.props.StringProperty(
        name="Text Value",
        default="Hello World"
    )
    custom_text_anchor: bpy.props.EnumProperty(
        name="Anchor",
        items=[
            ('TOP_LEFT',     "Top-Left",     ""),
            ('TOP_CENTER',   "Top-Center",   ""),
            ('TOP_RIGHT',    "Top-Right",    ""),
            ('MIDDLE_LEFT',  "Middle-Left",  ""),
            ('MIDDLE_CENTER',"Middle-Center",""),
            ('MIDDLE_RIGHT', "Middle-Right", ""),
            ('BOTTOM_LEFT',  "Bottom-Left",  ""),
            ('BOTTOM_CENTER',"Bottom-Center",""),
            ('BOTTOM_RIGHT', "Bottom-Right", ""),
        ],
        default='TOP_LEFT'
    )
    custom_text_margin_x: bpy.props.IntProperty(
        name="Margin X",
        default=0,
        description="Horizontal grid offset. 0 is at the anchor; +1 shifts one grid unit right (or left, as defined)"
    )
    custom_text_margin_y: bpy.props.IntProperty(
        name="Margin Y",
        default=0,
        description="Vertical grid offset. 0 is at the anchor; +1 shifts one grid unit up (or down) relative to the anchor"
    )

    custom_text_scale: bpy.props.IntProperty(
        name="Scale",
        default=10,  # A mid-range default (scale of 10/20)
        min=0,
        max=20,
        description="Text scaling factor in grid units (0=small, 20=large)"
    )


    # NEW: Duration or Indefinite
    custom_text_duration: bpy.props.FloatProperty(
        name="Duration (sec)",
        default=3.0,
        min=0.0,
        description="How long (in seconds) this text should appear. 0 => instant fade. If 'Indefinite' is true, this is ignored."
    )
    custom_text_indefinite: bpy.props.BoolProperty(
        name="Indefinite?",
        default=False,
        description="If checked, text stays on-screen forever (until you remove or reset)."
    )
    custom_text_color: bpy.props.FloatVectorProperty(
        name="Text Color",
        size=4,
        subtype='COLOR',
        min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
        description="RGBA color for the text"
    )
    custom_text_subtype: bpy.props.EnumProperty(
        name="Text Subtype",
        items=[
            ('STATIC', "Static Text", ""),
            ('OBJECTIVE', "Objective Counter", "Displays an objective’s current_value in real time"),
            ('OBJECTIVE_TIMER_DISPLAY', "Objective Timer Display", "Show an objective’s timer countdown"),
        ],
        default='STATIC'
    )
    text_objective_index: bpy.props.EnumProperty(
        name="Objective",
        description="Which objective's current_value to display",
        items=enum_objective_items  # <--- same function you use elsewhere
    )

    # fields for more intuitive Objective Counter formatting:
    custom_text_prefix: bpy.props.StringProperty(
        name="Prefix Text",
        default="",
        description="Text displayed before the objective counter value."
    )
    custom_text_suffix: bpy.props.StringProperty(
        name="Suffix Text",
        default="",
        description="Text displayed after the objective counter value."
    )
    custom_text_include_counter: bpy.props.BoolProperty(
        name="Include Counter",
        default=True,
        description="If enabled, the numeric value of the objective is displayed."
    )


    # --------------------------------------------------
   #Pointer properties for CUSTOM object & action
    # --------------------------------------------------
    custom_action_target: bpy.props.PointerProperty(
        name="Custom Action Target",
        type=bpy.types.Object,
        description="Which object (or armature) to play the custom action on"
    )

    custom_action_action: bpy.props.PointerProperty(
        name="Custom Action",
        type=bpy.types.Action,
        description="Which Action to play (NLA track in exp_custom_actions)"
    )
    custom_action_loop: bpy.props.BoolProperty(
        name="Loop?",
        default=False,
        description="If True, the custom action loops until loop_duration is reached"
    )

    custom_action_loop_duration: bpy.props.FloatProperty(
        name="Loop Duration (sec)",
        default=10.0,
        min=0.0,
        description="How long a looping custom action will continue before it stops"
    )
    custom_action_message: bpy.props.StringProperty(
        name="Details about your custom action",
        default=""
    )

    ###########################################
    # 'PROPERTY' reaction:
    ###########################################
    property_data_path: bpy.props.StringProperty(
        name="Data Path (eval)",
        description="Paste the full Blender data path (e.g. from Right-Click -> Copy Data Path)",
        default="",
        update=update_property_data_path  # callback
    )
    property_type: bpy.props.EnumProperty(
        name="Property Type",
        items=[
            ("NONE",   "None",   ""),
            ("BOOL",   "Bool",   ""),
            ("INT",    "Int",    ""),
            ("FLOAT",  "Float",  ""),
            ("STRING", "String", ""),
            ("VECTOR", "Vector", ""),
        ],
        default="NONE"
    )
        # How long it takes to go from old => new. 0 => instant.
    property_transition_duration: bpy.props.FloatProperty(
        name="Duration",
        default=0.0,
        min=0.0,
        description="How long to interpolate the property from old value to new value (in seconds). 0 means instant."
    )

    # Whether or not we revert to the old value after some delay
    property_reset: bpy.props.BoolProperty(
        name="Reset After",
        default=False,
        description="If checked, we revert to the old property value after some delay."
    )

    # How long after finishing the new value do we wait before reverting to old
    property_reset_delay: bpy.props.FloatProperty(
        name="Reset Delay",
        default=1.0,
        min=0.0,
        description="How many seconds after finishing the new value do we wait before resetting the property?"
    )
    bool_value: bpy.props.BoolProperty(default=False)
    int_value: bpy.props.IntProperty(default=0)
    float_value: bpy.props.FloatProperty(default=0.0)
    string_value: bpy.props.StringProperty(default="")
    vector_value: bpy.props.FloatVectorProperty(size=4, default=(0,0,0,0))
    vector_length: bpy.props.IntProperty(default=3)

    # --- New default value fields (for the starting/default value) ---
    default_bool_value: bpy.props.BoolProperty(default=False)
    default_int_value: bpy.props.IntProperty(default=0)
    default_float_value: bpy.props.FloatProperty(default=0.0)
    default_string_value: bpy.props.StringProperty(default="")
    default_vector_value: bpy.props.FloatVectorProperty(size=4, default=(0.0, 0.0, 0.0, 0.0))

#############################################
##### OBJECTIVE REACTION FIELDS
#############################################
    objective_index: bpy.props.EnumProperty(
        name="Objective",
        description="Which objective to modify?",
        items=enum_objective_items,
    )
    objective_op: bpy.props.EnumProperty(
        name="Operation",
        items=[
            ("ADD",      "Add",      "Add to the current_value"),
            ("SUBTRACT", "Subtract", "Subtract from the current_value"),
            ("RESET",    "Reset",    "Set current_value back to default_value"),
        ],
        default="ADD"
    )
    objective_amount: bpy.props.IntProperty(
        name="Amount",
        default=1,
        min=0,
        description="How much to add or subtract for ADD/SUBTRACT"
    )
    objective_timer_op: bpy.props.EnumProperty(
        name="Timer Operation",
        items=[
            ("START", "Start Timer", "Begin the countdown"),
            ("STOP",  "Stop Timer",  "Stop the countdown immediately"),
        ],
        default="START"
    )
    interruptible: bpy.props.BoolProperty(
        name="Interruptible",
        default=True,
        description="If True, the timer can continuously be restarted in-game.)"
    )


#############################################
##### mobility and game reactions
#############################################
    mobility_game_settings: bpy.props.PointerProperty(
        name="Mobility & Game Settings",
        type=MobilityGameReactionsPG
    )

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
        t = (now - self.start_time) / self.duration
        if t >= 1.0:
            # Snap to final
            self.obj.location = self.end_loc
            self.obj.rotation_euler = self.end_rot
            self.obj.scale = self.end_scl
            return True
        else:
            # Lerp location
            self.obj.location = self.start_loc.lerp(self.end_loc, t)

            # Lerp rotation euler
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
    target_obj = reaction.transform_object
    if not target_obj:
        return
    
    target_obj.rotation_mode = 'XYZ'

    duration = reaction.transform_duration
    if duration < 0.0:
        duration = 0.0

    mode = reaction.transform_mode

    if mode == "OFFSET":
        # The old approach: we interpret transform_location/rotation/scale as offsets
        apply_offset_transform(reaction, target_obj, duration)

    elif mode == "TO_LOCATION":
        # We interpret transform_location/rotation/scale as absolute world transforms
        apply_to_location_transform(reaction, target_obj, duration)

    elif mode == "TO_OBJECT":
        # We move from current => the transforms of reaction.transform_to_object
        to_obj = reaction.transform_to_object
        if to_obj:
            apply_to_object_transform(reaction, target_obj, to_obj, duration)

    elif mode == "LOCAL_OFFSET":
        # We interpret transform_location/rotation/scale as offsets in local space
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
    from .exp_animations import get_global_animation_manager
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
            color=color
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
            color=tuple(r.custom_text_color)
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

    # Overwrite sound_data.filepath so aud can load it properly
    sound_data.filepath = temp_path

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