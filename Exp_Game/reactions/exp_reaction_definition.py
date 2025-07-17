#Exploratory/Exp_UI/exp_reaction_definition.py

import bpy
from .exp_mobility_and_game_reactions import MobilityGameReactionsPG

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

    use_character: bpy.props.BoolProperty(
        name="Use Character",
        default=False,
        description="If True, transform the scene’s target_armature instead of the chosen object"
    )

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

    # ─── ^TO_OBJECT per-channel toggles ────────────────────────────────
    transform_use_location: bpy.props.BoolProperty(
        name="Location",
        default=True,
        description="Copy the target object's location"
    )
    transform_use_rotation: bpy.props.BoolProperty(
        name="Rotation",
        default=True,
        description="Copy the target object's rotation"
    )
    transform_use_scale: bpy.props.BoolProperty(
        name="Scale",
        default=True,
        description="Copy the target object's scale"
    )
    ## ─── ^^end TO_OBJECT per-channel toggles ────────────────────────────────
    #----------------------------------------------------

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