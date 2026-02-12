#Exploratory/Exp_UI/exp_reaction_definition.py

import bpy
from .exp_mobility_and_game_reactions import (
    MobilityReactionsPG,
    MeshVisibilityReactionsPG,
)
from .exp_fonts import discover_fonts 
from .exp_action_keys import _update_action_key_name
#---custom propertys--#
def update_property_data_path(self, context):
    """
    Called whenever the user changes 'property_data_path'.

    Fixes:
      • Evaluate the path with proper globals so 'bpy', 'mathutils' resolve.
      • Robust type detection for BOOL / INT / FLOAT / STRING / VECTOR.
      • Seed both the target value fields and the *default_* fields from
        the current property so reset/revert behaves correctly.
    """
    import mathutils

    # Clear type initially
    self.property_type = "NONE"

    path_str = (self.property_data_path or "").strip()
    if not path_str:
        return

    # Evaluate with a safe, known environment so expressions like
    # "bpy.data.materials['Mat']..." actually work.
    env = {"bpy": bpy, "mathutils": mathutils}
    try:
        prop_ref = eval(path_str, env)
    except Exception:
        # Invalid path or not currently resolvable.
        return

    # ---- Scalar types -------------------------------------------------
    if isinstance(prop_ref, bool):
        self.property_type        = "BOOL"
        self.bool_value           = bool(prop_ref)
        self.default_bool_value   = bool(prop_ref)
        return

    if isinstance(prop_ref, int) and not isinstance(prop_ref, bool):
        self.property_type        = "INT"
        self.int_value            = int(prop_ref)
        self.default_int_value    = int(prop_ref)
        return

    if isinstance(prop_ref, float):
        self.property_type        = "FLOAT"
        self.float_value          = float(prop_ref)
        self.default_float_value  = float(prop_ref)
        return

    if isinstance(prop_ref, str):
        self.property_type        = "STRING"
        self.string_value         = str(prop_ref)
        self.default_string_value = str(prop_ref)
        return

    # ---- Vector/array-like -------------------------------------------
    # Covers Blender RNA float/int arrays and mathutils.Vector/Color/etc.
    try:
        is_seq = hasattr(prop_ref, "__len__") and hasattr(prop_ref, "__getitem__")
        length = len(prop_ref) if is_seq else 0
    except Exception:
        is_seq = False
        length = 0

    if is_seq and length > 0:
        self.property_type   = "VECTOR"
        # Clamp vector UI to 4 channels (common for colors/locations).
        self.vector_length   = min(int(length), 4)

        # Convert the current value to a list[float] up to vector_length.
        cur_vals = []
        for i in range(self.vector_length):
            try:
                v = prop_ref[i]
                # Cast bool→float/ int→float for a smooth interpolation path
                if isinstance(v, bool):
                    v = 1.0 if v else 0.0
                elif isinstance(v, (int, float)):
                    v = float(v)
                elif isinstance(v, mathutils.Vector):
                    v = float(v)  # unlikely; but keep conversion symmetry
                else:
                    # If it’s something exotic, best effort float()
                    v = float(v)
            except Exception:
                v = 0.0
            cur_vals.append(v)

        # Pad to 4 to satisfy RNA fixed-size storage
        while len(cur_vals) < 4:
            cur_vals.append(0.0)

        # Seed both the target and default vector fields
        self.vector_value         = cur_vals
        self.default_vector_value = cur_vals
        return

    # If we reach here, we don’t support the referenced type
    self.property_type = "NONE"


# --- Counter enum ---
def enum_counter_items(self, context):
    """Build a list of (identifier, name, description) for each counter."""
    if not context:
        return []
    scn = context.scene
    items = []
    if hasattr(scn, "counters"):
        for i, counter in enumerate(scn.counters):
            items.append((str(i), counter.name, f"Counter: {counter.name}"))
    if not items:
        items.append(("0", "No Counters", ""))
    return items


# --- Timer enum ---
def enum_timer_items(self, context):
    """Build a list of (identifier, name, description) for each timer."""
    if not context:
        return []
    scn = context.scene
    items = []
    if hasattr(scn, "timers"):
        for i, timer in enumerate(scn.timers):
            items.append((str(i), timer.name, f"Timer: {timer.name}"))
    if not items:
        items.append(("0", "No Timers", ""))
    return items


# ─────────────────────────────────────────────────────────
# Interactions and reactions -- Duplicate
# ─────────────────────────────────────────────────────────
def _deep_copy_pg(src, dst, skip: set[str] = frozenset()):
    from bpy.types import ID as _ID
    for prop in src.bl_rna.properties:
        ident = prop.identifier
        if ident in {"rna_type"} or ident in skip:
            continue
        if getattr(prop, "is_readonly", False):
            continue
        try:
            value = getattr(src, ident)
        except Exception:
            continue
        try:
            if prop.type == 'POINTER':
                if isinstance(value, _ID) or value is None:
                    setattr(dst, ident, value)
                else:
                    sub_dst = getattr(dst, ident)
                    _deep_copy_pg(value, sub_dst)
            elif prop.type == 'COLLECTION':
                dst_coll = getattr(dst, ident)
                try:
                    dst_coll.clear()
                except AttributeError:
                    while len(dst_coll):
                        dst_coll.remove(len(dst_coll) - 1)
                for src_item in value:
                    dst_item = dst_coll.add()
                    _deep_copy_pg(src_item, dst_item)
            else:
                setattr(dst, ident, value)
        except Exception:
            pass


class EXPLORATORY_OT_DuplicateGlobalReaction(bpy.types.Operator):
    """Duplicate the selected Reaction in the global library."""
    bl_idname = "exploratory.duplicate_global_reaction"
    bl_label = "Duplicate Reaction"
    bl_options = {'REGISTER', 'UNDO'}

    index: bpy.props.IntProperty(name="Index", default=-1)

    def execute(self, context):
        scn = context.scene
        src_idx = self.index if self.index >= 0 else scn.reactions_index
        if not (0 <= src_idx < len(scn.reactions)):
            self.report({'WARNING'}, "No valid Reaction selected.")
            return {'CANCELLED'}

        src = scn.reactions[src_idx]
        dst = scn.reactions.add()

        _deep_copy_pg(src, dst)

        try:
            dst.name = f"{src.name} (Copy)"
        except Exception:
            pass

        scn.reactions_index = len(scn.reactions) - 1
        return {'FINISHED'}


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
            ("ENABLE_CROSSHAIRS", "Enable Crosshairs", "Pixel based crosshair at screen center"),
            ("HITSCAN",           "Hitscan",   "Instant ray from origin to max range"),
            ("PROJECTILE",        "Projectile","Simulated projectile with gravity"),
            ("COUNTER_UPDATE",    "Counter Update", "Add/subtract/reset a counter"),
            ("TIMER_CONTROL",     "Timer Control", "Start/Stop a timer"),
            ("MOBILITY",          "Mobility",             "Enable/Disable movement, jump, sprint"),
            ("MESH_VISIBILITY",   "Mesh Visibility",      "Hide/Unhide/Toggle a mesh object"),
            ("RESET_GAME",        "Reset Game",           "Reset the game state"),
            ("ACTION_KEYS",       "Action Keys",        "Enable/Disable/Toggle a named Action Key"),
            ("DELAY",             "Delay (Utility)",      "Pause before continuing to next reactions"),
            ("PARENTING",         "Parent / Unparent",    "Parent to an object/armature bone, or restore original parent"),
            ("TRACK_TO",          "Track To",             "Move/chase from A to B with reroute & ground snap"),
            ("ENABLE_HEALTH",     "Enable Health",        "Attach health component to an object"),
            ("DISPLAY_HEALTH_UI", "Display Health UI",    "Show health bar on screen"),
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
    
    char_action_speed: bpy.props.FloatProperty(
        name="Speed Multiplier",
        default=1.0,
        min=0.05,
        soft_max=5.0,
        description="1.0 = normal. Multiplies playback speed for this character action."
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
    char_action_blend_time: bpy.props.FloatProperty(
        name="Blend Time",
        default=0.15,
        min=0.0,
        max=2.0,
        description="Fade/transition time in seconds when starting this animation"
    )
    char_action_bone_group: bpy.props.EnumProperty(
        name="Bone Group",
        items=[
            ("ALL", "Full Body", "Apply to entire body"),
            ("UPPER_BODY", "Upper Body", "Spine, arms, head, neck"),
            ("LOWER_BODY", "Lower Body", "Hips, legs"),
            ("ARM_L", "Left Arm", "Left arm only"),
            ("ARM_R", "Right Arm", "Right arm only"),
            ("ARMS", "Both Arms", "Both arms"),
            ("LEG_L", "Left Leg", "Left leg only"),
            ("LEG_R", "Right Leg", "Right leg only"),
            ("LEGS", "Both Legs", "Both legs"),
            ("HEAD_NECK", "Head & Neck", "Head and neck only"),
            ("SPINE", "Spine Only", "Spine bones only"),
        ],
        default="ALL",
        description="Which body part to affect with this animation"
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
    # Which object will be transformed (target to move)?
    use_character: bpy.props.BoolProperty(
        name="Use Character",
        default=False,
        description="If True, transform the scene’s target_armature instead of the chosen object"
    )
    transform_object: bpy.props.PointerProperty(
        name="Transform Object",
        type=bpy.types.Object,
        description="Object to transform when Use Character is False"
    )

    # How we interpret the transform
    transform_mode: bpy.props.EnumProperty(
        name="Transform Mode",
        description="How to compute the destination transform",
        items=[
            ("OFFSET",       "Global Offset", "Add location/rotation/scale in world space"),
            ("LOCAL_OFFSET", "Local Offset",  "Add location/rotation/scale in local space"),
            ("TO_LOCATION",  "To Location",   "Move to an explicit world-space transform"),
            ("TO_OBJECT",    "To Object",     "Copy transforms from another object"),
            ("TO_BONE",      "To Bone",       "Copy transforms from a specific bone on an armature"),
        ],
        default="OFFSET"
    )

    # ----- TO_OBJECT source -----
    transform_to_object: bpy.props.PointerProperty(
        name="Target Object (To Object)",
        type=bpy.types.Object,
        description="If Transform Mode = 'TO_OBJECT' and the toggle below is OFF, copy from this object"
    )
    transform_to_use_character: bpy.props.BoolProperty(
        name="Use Character as 'To Object'",
        default=False,
        description="If True in TO_OBJECT mode, copy transforms from scene.target_armature instead of a picked object"
    )

    # Per-channel toggles for TO_OBJECT/TO_BONE
    transform_use_location: bpy.props.BoolProperty(
        name="Location",
        default=True,
        description="Copy the source location"
    )
    transform_use_rotation: bpy.props.BoolProperty(
        name="Rotation",
        default=True,
        description="Copy the source rotation"
    )
    transform_use_scale: bpy.props.BoolProperty(
        name="Scale",
        default=True,
        description="Copy the source scale"
    )

    # ----- TO_BONE source -----
    transform_to_bone_use_character: bpy.props.BoolProperty(
        name="Use Character Armature",
        default=True,
        description="If True in TO_BONE mode, read the bone from scene.target_armature"
    )
    transform_to_armature: bpy.props.PointerProperty(
        name="Armature",
        type=bpy.types.Object,
        description="If not using the character, read the bone from this armature object"
    )
    transform_bone_name: bpy.props.StringProperty(
        name="Bone Name",
        default="",
        description="Bone name to copy from (string only; survives armature rebuilds)"
    )

    # Values for OFFSET / LOCAL_OFFSET / TO_LOCATION
    transform_location: bpy.props.FloatVectorProperty(
        name="Location",
        default=(0.0, 0.0, 0.0),
        subtype='TRANSLATION',
        description="Destination (world) location or offset"
    )
    transform_rotation: bpy.props.FloatVectorProperty(
        name="Rotation (Euler)",
        default=(0.0, 0.0, 0.0),
        subtype='EULER',
        description="Destination (world) rotation (XYZ) or offset"
    )
    transform_scale: bpy.props.FloatVectorProperty(
        name="Scale",
        default=(1.0, 1.0, 1.0),
        subtype='XYZ',
        description="Destination scale or multiplicative local scale"
    )

    transform_duration: bpy.props.FloatProperty(
        name="Duration",
        default=1.0,
        min=0.0,
        description="How long the transform should take (0 = instant)"
    )
    

    # --------------------------------------------------
    # 2) CUSTOM UI TEXT REACTION FIELDS
    # --------------------------------------------------
    custom_text_value: bpy.props.StringProperty(
        name="Text Value",
        default="Hello World"
    )

    custom_text_font: bpy.props.EnumProperty(
        name        = "Font",
        description = "Font for this text reaction (Default = Blender’s built‑in font)",
        items       = lambda _self, _ctx: discover_fonts(),
        default     = 0                    #  ←  just change this line
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
            ('COUNTER_DISPLAY', "Counter Display", "Displays a counter's current value in real time"),
            ('TIMER_DISPLAY', "Timer Display", "Show a timer's countdown/countup"),
        ],
        default='STATIC'
    )
    text_counter_index: bpy.props.EnumProperty(
        name="Counter",
        description="Which counter's current value to display",
        items=enum_counter_items
    )
    text_timer_index: bpy.props.EnumProperty(
        name="Timer",
        description="Which timer's current value to display",
        items=enum_timer_items
    )

    # fields for more intuitive Counter formatting:
    custom_text_prefix: bpy.props.StringProperty(
        name="Prefix Text",
        default="",
        description="Text displayed before the counter/timer value."
    )
    custom_text_suffix: bpy.props.StringProperty(
        name="Suffix Text",
        default="",
        description="Text displayed after the counter/timer value."
    )
    custom_text_include_counter: bpy.props.BoolProperty(
        name="Include Value",
        default=True,
        description="If enabled, the numeric value is displayed."
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
        description="Which Action to play on the target object"
    )
    custom_action_speed: bpy.props.FloatProperty(
        name="Speed Multiplier",
        default=1.0,
        min=0.05,
        soft_max=5.0,
        description="1.0 = normal. Multiplies playback speed for this custom action."
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
        description="Paste the full Blender data path (e.g. from Right-Click -> Copy Full Data Path)",
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
##### COUNTER REACTION FIELDS
#############################################
    counter_index: bpy.props.EnumProperty(
        name="Counter",
        description="Which counter to modify?",
        items=enum_counter_items,
    )
    counter_op: bpy.props.EnumProperty(
        name="Operation",
        items=[
            ("ADD",      "Add",      "Add to the current value"),
            ("SUBTRACT", "Subtract", "Subtract from the current value"),
            ("RESET",    "Reset",    "Set current value back to default"),
        ],
        default="ADD"
    )
    counter_amount: bpy.props.IntProperty(
        name="Amount",
        default=1,
        min=0,
        description="How much to add or subtract for ADD/SUBTRACT"
    )

#############################################
##### TIMER REACTION FIELDS
#############################################
    timer_index: bpy.props.EnumProperty(
        name="Timer",
        description="Which timer to control?",
        items=enum_timer_items,
    )
    timer_op: bpy.props.EnumProperty(
        name="Timer Operation",
        items=[
            ("START", "Start Timer", "Begin the countdown/countup"),
            ("STOP",  "Stop Timer",  "Stop the timer immediately"),
        ],
        default="START"
    )
    interruptible: bpy.props.BoolProperty(
        name="Interruptible",
        default=False,
        description="If True, the timer can be restarted while running"
    )


    #############################################
    ##### mobility and game reactions
    #############################################
    mobility_settings: bpy.props.PointerProperty(
        name="Mobility Settings",
        type=MobilityReactionsPG
    )

    mesh_visibility: bpy.props.PointerProperty(
        name="Mesh Visibility",
        type=MeshVisibilityReactionsPG
    )



    # --------------------------------------------------
    # CROSSHAIRS REACTION FIELDS
    # --------------------------------------------------
    crosshair_style: bpy.props.EnumProperty(
        name="Style",
        items=[
            ("PLUS",     "Plus",         "+"),
            ("PLUS_DOT", "+ with Dot",   "+ with center dot"),
            ("X",        "X",            "Diagonal cross"),
            ("X_DOT",    "X with Dot",   "Diagonal cross with dot"),
        ],
        default="PLUS"
    )
    crosshair_length_px: bpy.props.IntProperty(
        name="Arm Length (px)", default=12, min=0, max=512
    )
    crosshair_gap_px: bpy.props.IntProperty(
        name="Gap (px)", default=6, min=0, max=256
    )
    crosshair_thickness_px: bpy.props.IntProperty(
        name="Line Thickness (px)", default=2, min=1, max=16
    )
    crosshair_dot_radius_px: bpy.props.IntProperty(
        name="Dot Radius (px)", default=0, min=0, max=32
    )
    crosshair_color: bpy.props.FloatVectorProperty(
        name="Color", size=4, subtype='COLOR', min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0, 0.85)
    )
    crosshair_indefinite: bpy.props.BoolProperty(
        name="Indefinite?", default=True,
        description="If True, stays until reset or disabled"
    )
    crosshair_duration: bpy.props.FloatProperty(
        name="Duration (sec)", default=5.0, min=0.0,
        description="Only used when Indefinite is off"
    )


    # --------------------------------------------------
    # PROJECTILE / HITSCAN REACTION FIELDS
    # --------------------------------------------------

    # Where the shot starts
    proj_use_character_origin: bpy.props.BoolProperty(
        name="Use Character as Origin",
        default=True,
        description="If True, start from scene.target_armature + offset; else from the chosen origin object"
    )
    proj_origin_object: bpy.props.PointerProperty(
        name="Origin Object",
        type=bpy.types.Object,
        description="Used if Use Character Origin = False"
    )
    proj_origin_offset: bpy.props.FloatVectorProperty(
        name="Origin Offset (local)",
        subtype='TRANSLATION',
        default=(0.0, 0.2, 1.4),
        description="Local offset from origin (e.g. slightly in front/up from the chest)"
    )

    # Where we aim
    proj_aim_source: bpy.props.EnumProperty(
        name="Aim",
        description="How to aim the shot",
        items=[
            ("CROSSHAIR",    "Crosshair (Center)", "Ray from the center of the active 3D View; target point determines direction"),
            ("CHAR_FORWARD", "Character Forward",  "Use the character's facing (+Y local)"),
        ],
        default="CROSSHAIR"
    )

    # Optional visual object to manipulate
    proj_object: bpy.props.PointerProperty(
        name="Projectile Object",
        type=bpy.types.Object,
        description="Optional object to visualize the shot (moved on hitscan, animated for projectile)"
    )
    proj_align_object_to_velocity: bpy.props.BoolProperty(
        name="Align Object to Velocity",
        default=True,
        description="Rotate the object so its +Y axis points along velocity/direction"
    )

    # Hitscan tuning
    proj_max_range: bpy.props.FloatProperty(
        name="Hitscan Max Range",
        default=60.0, min=0.1, max=1000.0
    )
    proj_place_hitscan_object: bpy.props.BoolProperty(
        name="Place Object at Impact",
        default=True,
        description="If a projectile object is set, move it to the impact (or end of range on miss)"
    )

    # Projectile (sim) tuning
    proj_speed: bpy.props.FloatProperty(
        name="Initial Speed (m/s)",
        default=24.0, min=0.0, max=300.0
    )
    proj_gravity: bpy.props.FloatProperty(
        name="Gravity (m/s²)",
        default=-21.0, min=-60.0, max=0.0,
        description="Downward acceleration; usually matches Scene.char_physics.gravity"
    )
    proj_lifetime: bpy.props.FloatProperty(
        name="Lifetime (sec)",
        default=3.0, min=0.0, max=30.0,
        description="Max time before the projectile auto-despawns"
    )
    proj_on_contact_stop: bpy.props.BoolProperty(
        name="Stop on Contact",
        default=True,
        description="Stop and despawn on the first collision"
    )
    proj_radius: bpy.props.FloatProperty(
        name="Radius (m)",
        default=0.05, min=0.0, max=1.0,
        description="Optional radius hint for future expansion; current pass uses ray tests"
    )
    proj_pool_limit: bpy.props.IntProperty(
        name="Max Active Projectiles",
        default=8, min=1, max=64,
        description="Hard cap to protect the frame budget"
    )



    # --------------------------------------------------
    # ACTION KEYS REACTION FIELDS
    # --------------------------------------------------
    action_key_op: bpy.props.EnumProperty(
        name="Operation",
        items=[
            ("ENABLE",  "Enable",  "Enable the specified Action"),
            ("DISABLE", "Disable", "Disable the specified Action"),
            ("TOGGLE",  "Toggle",  "Toggle the specified Action"),
        ],
        default="ENABLE"
    )

    # New authoritative name field the user edits on the reaction node.
    action_key_name: bpy.props.StringProperty(
        name="Action Name",
        default="",
        update=_update_action_key_name  # keeps Scene registry, legacy id, and triggers in sync
    )

    # Hidden index into Scene.action_keys for robust rename/delete.
    action_key_index: bpy.props.IntProperty(
        name="Action Index",
        default=-1,
        min=-1
    )


    # --- Utility: Delay ---
    utility_delay_seconds: bpy.props.FloatProperty(
        name="Delay (sec)",
        default=0.25,
        min=0.0,
        description="Pauses the chain by this many seconds; affects subsequent reactions in the same Interaction."
    )



    #############################################
    ##### PARENTING REACTION FIELDS (NEW)
    #############################################
    parenting_op: bpy.props.EnumProperty(
        name="Operation",
        items=[
            ("PARENT_TO", "Parent To", "Parent the target to the chosen parent"),
            ("UNPARENT",  "Unparent",  "Restore the target's original parent from game start"),
        ],
        default="PARENT_TO"
    )

    # Child (the object being parented/unparented)
    parenting_target_use_character: bpy.props.BoolProperty(
        name="Use Character as Target",
        default=False,
        description="If True, the child to (un)parent is the scene.target_armature"
    )
    parenting_target_object: bpy.props.PointerProperty(
        name="Target Object",
        type=bpy.types.Object,
        description="Object to (un)parent if not using the character"
    )

    # Parent side: either armature (optionally specific bone) or any object
    parenting_parent_use_armature: bpy.props.BoolProperty(
        name="Parent to Character Armature",
        default=True,
        description="If True, parent to scene.target_armature (optionally a specific bone by name)"
    )
    parenting_parent_object: bpy.props.PointerProperty(
        name="Parent Object",
        type=bpy.types.Object,
        description="If not using armature, parent to this object"
    )
    parenting_bone_name: bpy.props.StringProperty(
        name="Bone (name)",
        default="",
        description="Optional bone name on the character armature"
    )

    # Local offset from parent origin
    parenting_local_offset: bpy.props.FloatVectorProperty(
        name="Local Offset",
        size=3,
        default=(0.0, 0.0, 0.0),
        description="Local position offset from parent origin"
    )

    # --------------------------------------------------
    # TRACK TO (chase / move toward) REACTION FIELDS
    # --------------------------------------------------
    track_mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('DIRECT', "Direct", "Move straight toward target, stop at walls"),
            ('PATHFINDING', "Pathfinding", "Smart navigation around obstacles (like zombies)"),
        ],
        default='DIRECT',
        description="How the mover navigates toward the target"
    )

    track_from_use_character: bpy.props.BoolProperty(
        name="From: Use Character",
        default=False,
        description="If True, the mover is scene.target_armature; else use 'From Object'"
    )
    track_from_object: bpy.props.PointerProperty(
        name="From Object",
        type=bpy.types.Object,
        description="Mover when 'From: Use Character' is False"
    )

    track_to_use_character: bpy.props.BoolProperty(
        name="To: Use Character",
        default=False,
        description="If True, target is scene.target_armature; else use 'To Object'"
    )
    track_to_object: bpy.props.PointerProperty(
        name="To Object",
        type=bpy.types.Object,
        description="Target to move toward when 'To: Use Character' is False"
    )

    track_speed: bpy.props.FloatProperty(
        name="Speed (m/s)",
        default=3.5, min=0.0, soft_max=15.0,
        description="Desired chase speed. Character mover maps this to walk/run; object mover uses exact m/s"
    )
    track_arrive_radius: bpy.props.FloatProperty(
        name="Arrive Radius (m)",
        default=0.30, min=0.0, soft_max=2.0,
        description="Stop when within this horizontal distance"
    )
    track_respect_proxy_meshes: bpy.props.BoolProperty(
        name="Respect Proxy Meshes",
        default=True,
        description="Collide/slide/step using scene proxy meshes; if off, mover goes in a straight line"
    )
    track_use_gravity: bpy.props.BoolProperty(
        name="Gravity",
        default=True,
        description="Apply gravity to mover while tracking (accumulates downward velocity, lands on surfaces)"
    )
    track_max_runtime: bpy.props.FloatProperty(
        name="Max Runtime (sec)",
        default=0.0, min=0.0, soft_max=120.0,
        description="0 = unlimited. Stops automatically after this many seconds"
    )


    # --------------------------------------------------
    # ENABLE_HEALTH REACTION FIELDS
    # --------------------------------------------------
    health_target_object: bpy.props.PointerProperty(
        name="Target Object",
        type=bpy.types.Object,
        description="Object to attach health tracking to"
    )
    health_start_value: bpy.props.FloatProperty(
        name="Start Value",
        default=100.0,
        min=0.0,
        description="Initial health value (also used on reset)"
    )
    health_min_value: bpy.props.FloatProperty(
        name="Min Value",
        default=0.0,
        description="Minimum health value (usually 0)"
    )
    health_max_value: bpy.props.FloatProperty(
        name="Max Value",
        default=100.0,
        min=0.0,
        description="Maximum health value"
    )
    # Health UI options (used by DISPLAY_HEALTH_UI)
    health_ui_position: bpy.props.EnumProperty(
        name="Position",
        items=[
            ("BOTTOM", "Bottom", "Horizontal bar at bottom of screen"),
            ("TOP", "Top", "Horizontal bar at top of screen"),
            ("LEFT", "Left", "Vertical bar on left side"),
            ("RIGHT", "Right", "Vertical bar on right side"),
        ],
        default="BOTTOM"
    )
    health_ui_scale: bpy.props.IntProperty(
        name="Scale",
        default=10,
        min=1,
        max=20,
        description="Size of health bar (1=smallest, 20=largest)"
    )
    health_ui_offset_x: bpy.props.IntProperty(
        name="Offset X",
        default=0,
        min=-20,
        max=20,
        description="Horizontal offset in grid units (negative=left, positive=right)"
    )
    health_ui_offset_y: bpy.props.IntProperty(
        name="Offset Y",
        default=1,
        min=-20,
        max=20,
        description="Vertical offset in grid units (negative=down, positive=up)"
    )

    # --------------------------------------------------
    # DISPLAY_HEALTH_UI REACTION FIELDS
    # --------------------------------------------------
    health_ui_target_object: bpy.props.PointerProperty(
        name="Target Object",
        type=bpy.types.Object,
        description="Object whose health to display"
    )


def register_reaction_library_properties():
    """
    Attach a top-level Reaction library to the Scene.
    Reactions are *independent* from Interactions now.
    """
    from .exp_reaction_definition import ReactionDefinition

    if not hasattr(bpy.types.Scene, "reactions"):
        bpy.types.Scene.reactions = bpy.props.CollectionProperty(type=ReactionDefinition)
    if not hasattr(bpy.types.Scene, "reactions_index"):
        bpy.types.Scene.reactions_index = bpy.props.IntProperty(default=0)


def unregister_reaction_library_properties():
    if hasattr(bpy.types.Scene, "reactions"):
        del bpy.types.Scene.reactions
    if hasattr(bpy.types.Scene, "reactions_index"):
        del bpy.types.Scene.reactions_index
    