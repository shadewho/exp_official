import bpy
from ..reactions.exp_reaction_definition import ReactionDefinition, enum_objective_items
from .exp_interactions import trigger_mode_items
from ..reactions.exp_reaction_definition import register_reaction_library_properties
from ..reactions.exp_fonts import discover_fonts

class ReactionLinkPG(bpy.types.PropertyGroup):
    """
    A simple link: stores the index into scene.reactions.
    """
    reaction_index: bpy.props.IntProperty(
        name="Reaction Index",
        default=-1,
        min=-1,
        description="Index into scene.reactions; -1 means unassigned"
    )

def update_trigger_type(self, context):
    # ensure current mode is still valid when trigger_type changes
    allowed = [item[0] for item in trigger_mode_items(self, context)]
    if self.trigger_mode not in allowed:
        # reset to first allowed mode
        self.trigger_mode = allowed[0]
        

###############################################################################
# 1) InteractionDefinition
###############################################################################
class InteractionDefinition(bpy.types.PropertyGroup):
    """
    The main Interaction: each item includes:
      1) A trigger (trigger_type, etc.)
      2) A sub-collection of ReactionDefinition
    """
    name: bpy.props.StringProperty(
        name="Name",
        default="Interaction"
    )
    description: bpy.props.StringProperty(
        name="Description",
        default="Describe what this interaction does..."
    )

    # Trigger Type
    trigger_type: bpy.props.EnumProperty(
        name="Trigger Type",
        items=[
            ("PROXIMITY",       "Proximity",        "Triggers when within distance"),
            ("COLLISION",       "Collision",        "Triggers on collision"),
            ("INTERACT",        "Interact Key",     "Triggers on user pressing Interact"),
            ("ACTION",           "Action Key",       "Triggers on user pressing Action"), 
            ("OBJECTIVE_UPDATE","Objective Update", "Triggers when an objective changes"),
            ("TIMER_COMPLETE",  "Timer Complete",   "Fires when an objective’s timer ends"),
            ("ON_GAME_START",  "On Game Start",   "Fires once right after gameplay begins or a reset"),
            ("EXTERNAL",        "Trigger",          "Fires when an external boolean input is True"),
        ],
        default="PROXIMITY",
        update=update_trigger_type,  # ← ensure mode stays valid
    )

    trigger_mode: bpy.props.EnumProperty(
        name="Trigger Mode",
        items=trigger_mode_items,  # a Python callback, not a static list
        default=0,                  # ⟵ must be an integer index into whatever trigger_mode_items() returns
    )
    
    #use character allows users to assign the scene's target_armature as Object A
    use_character: bpy.props.BoolProperty(
        name="Use Character",
        default=False,
        description="If enabled, Object A is the game characters target_armature"
    )

    # Proximity Trigger
    proximity_object_a: bpy.props.PointerProperty(
        name="Object A",
        type=bpy.types.Object,
        description="First object in proximity check"
    )
    proximity_object_b: bpy.props.PointerProperty(
        name="Object B",
        type=bpy.types.Object,
        description="Second object in proximity check"
    )
    proximity_distance: bpy.props.FloatProperty(
        name="Distance",
        default=2.0,
        description="Distance threshold"
    )

    # Collision Trigger
    collision_object_a: bpy.props.PointerProperty(
        name="Collision Object A",
        type=bpy.types.Object,
        description="Which objects must collide to trigger?"
    )
    collision_object_b: bpy.props.PointerProperty(
        name="Collision Object B",
        type=bpy.types.Object,
        description="Which objects must collide to trigger?"
    )
    collision_margin: bpy.props.FloatProperty(
        name="Collision Margin",
        default=0.0,
        min=0.0,
        description="Extra distance for collision checks. 0 means exact bounding-box overlap."
    )

    trigger_cooldown: bpy.props.FloatProperty(
        name="Trigger Cooldown",
        default=0.0,
        min=0.0,
        description="Time in seconds before we can re-fire if we remain inside (used only if trigger_mode=COOLDOWN)."
    )

    interact_object: bpy.props.PointerProperty(
        name="Interact Object",
        type=bpy.types.Object,
        description="Which object must the player be near to trigger INTERACT?"
    )
    interact_distance: bpy.props.FloatProperty(
        name="Interact Distance",
        default=2.0,
        min=0.0,
        description="How close the player must be to interact_object (in Blender units)."
    )
 
    trigger_delay: bpy.props.FloatProperty(
    name="Trigger Delay",
    default=0.0,
    min=0.0,
    description="Delay (sec) before the reaction fires after the trigger is detected."
)


    #####Objectives Trigger Properties #####
    objective_index: bpy.props.EnumProperty(
        name="Objective",
        description="Which objective do we monitor?",
        items=enum_objective_items,
        default=None
    )

    objective_condition: bpy.props.EnumProperty(
        name="Condition",
        items=[
            ("CHANGED",    "Changed",    "Fires whenever the current_value changes in any direction"),
            ("INCREASED",  "Increased",  "Fires only if the current_value goes up"),
            ("DECREASED",  "Decreased",  "Fires only if the current_value goes down"),
            ("EQUALS",     "Equals",     "Fire if current_value == Condition Value"),
            ("AT_LEAST",   "At Least",   "Fire if current_value >= Condition Value"),
        ],
        default="CHANGED"
    )

    objective_condition_value: bpy.props.IntProperty(
        name="Condition Value",
        default=5,
        min=0,
        description="Used if Condition is EQUALS or AT_LEAST"
    )

    last_known_count: bpy.props.IntProperty(
        name="Last Known Count",
        default=-1,
        description="(internal) tracks the previous count so we can detect changes"
    )
    timer_objective_index: bpy.props.EnumProperty(
        name="Timer Objective",
        items=enum_objective_items,  # same function you use for objective listing
        description="Which objective’s timer we watch for completion"
    )



##backend trigger properties
    is_in_zone: bpy.props.BoolProperty(
        name="Is In Zone",
        default=False,
        description="Are we currently in proximity or collision?"
    )

    has_fired: bpy.props.BoolProperty(
        name="Has Fired",
        default=False
    )
    last_trigger_time: bpy.props.FloatProperty(
        name="Last Trigger Time",
        default=0.0
    )

    # Action Key Trigger config
    action_key_id: bpy.props.StringProperty(
        name="Action Key ID",
        default="",
        description="Unique identifier for this Action trigger. "
                    "This must be ENABLED by the 'Action Keys' reaction to fire."
    )

    # External trigger input (set this from outside to make the EXTERNAL trigger fire)
    external_signal: bpy.props.BoolProperty(
        name="External Signal",
        default=False,
        description="External boolean gate. When True, the EXTERNAL Trigger may fire according to mode/cooldown."
    )


    # Sub-collection for Reactions
    reaction_links: bpy.props.CollectionProperty(type=ReactionLinkPG)
    reaction_links_index: bpy.props.IntProperty(default=0)



###############################################################################
# 9) Registration Helpers
###############################################################################
def register_interaction_properties():
    """
    Attaches the top-level CollectionProperty `custom_interactions` to the Scene,
    where each item is an InteractionDefinition that holds ReactionDefinition.
    """
    bpy.types.Scene.custom_interactions = bpy.props.CollectionProperty(type=InteractionDefinition)
    bpy.types.Scene.custom_interactions_index = bpy.props.IntProperty(default=0)
    register_reaction_library_properties()
    
def unregister_interaction_properties():
    if hasattr(bpy.types.Scene, 'custom_interactions'):
        del bpy.types.Scene.custom_interactions
    if hasattr(bpy.types.Scene, 'custom_interactions_index'):
        del bpy.types.Scene.custom_interactions_index