# File: exp_interactions.py

import bpy
from mathutils import Vector
from .exp_reactions import (ReactionDefinition, execute_transform_reaction,
                             schedule_transform, execute_property_reaction,
                             execute_char_action_reaction, execute_custom_ui_text_reaction,
                             execute_objective_counter_reaction, enum_objective_items,
                             execute_objective_timer_reaction, execute_sound_reaction
)

from .exp_custom_animations import execute_custom_action_reaction
from .exp_time import get_game_time
from .exp_objectives import update_all_objective_timers
from .exp_mobility_and_game_reactions import execute_mobility_reaction
from . import exp_custom_ui

# Global list to hold pending trigger tasks.
_pending_trigger_tasks = []

# -------------------------------------------------------------------
# Dynamic trigger_mode items based on trigger_type
# -------------------------------------------------------------------
def trigger_mode_items(self, context):
    # always include One‑Shot and Cooldown
    items = [
        ("ONE_SHOT", "One-Shot",    "Fire once only, never again"),
        ("COOLDOWN", "Cooldown",    "Allow repeated fires but only after cooldown"),
    ]
    # but only proximity/collision get the Enter‑Only mode
    if self.trigger_type in {"PROXIMITY", "COLLISION"}:
        items.insert(1, ("ENTER_ONLY", "Enter Only",
                         "Fire on each new enter/press; resets on release/exit"))
    return items

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
            ("OBJECTIVE_UPDATE","Objective Update", "Triggers when an objective changes"),
            ("TIMER_COMPLETE",  "Timer Complete",   "Fires when an objective’s timer ends"),
        ],
        default="PROXIMITY",
        update=update_trigger_type,  # ← ensure mode stays valid
    )

    trigger_mode: bpy.props.EnumProperty(
        name="Trigger Mode",
        items=trigger_mode_items,  # a Python callback, not a static list
        default=0,                  # ⟵ must be an integer index into whatever trigger_mode_items() returns
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
        default=2.0,
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
        items=enum_objective_items,  # a function that enumerates scene.objectives
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


    # Sub-collection for Reactions
    reactions: bpy.props.CollectionProperty(type=ReactionDefinition)
    reactions_index: bpy.props.IntProperty(default=0)


###############################################################################
# 2) Operators for Adding/Removing Interactions
###############################################################################
class EXPLORATORY_OT_AddInteraction(bpy.types.Operator):
    """Add a new InteractionDefinition to the scene custom_interactions."""
    bl_idname = "exploratory.add_interaction"
    bl_label = "Add Interaction"

    def execute(self, context):
        scene = context.scene
        new_item = scene.custom_interactions.add()
        new_item.name = f"Interaction_{len(scene.custom_interactions)}"
        scene.custom_interactions_index = len(scene.custom_interactions) - 1
        return {'FINISHED'}


class EXPLORATORY_OT_RemoveInteraction(bpy.types.Operator):
    """Remove an InteractionDefinition from scene.custom_interactions by index."""
    bl_idname = "exploratory.remove_interaction"
    bl_label = "Remove Interaction"

    index: bpy.props.IntProperty()

    def execute(self, context):
        scene = context.scene
        if 0 <= self.index < len(scene.custom_interactions):
            scene.custom_interactions.remove(self.index)
            scene.custom_interactions_index = max(
                0, 
                min(self.index, len(scene.custom_interactions) - 1)
            )
        return {'FINISHED'}


###############################################################################
# 3) Operators for Adding/Removing Reactions within an Interaction
###############################################################################
class EXPLORATORY_OT_AddReactionToInteraction(bpy.types.Operator):
    """Add a new ReactionDefinition to the currently selected Interaction."""
    bl_idname = "exploratory.add_reaction_to_interaction"
    bl_label = "Add Reaction to Interaction"

    def execute(self, context):
        scene = context.scene
        i_idx = scene.custom_interactions_index
        if i_idx < 0 or i_idx >= len(scene.custom_interactions):
            self.report({'WARNING'}, "No valid Interaction selected.")
            return {'CANCELLED'}

        interaction = scene.custom_interactions[i_idx]
        new_r = interaction.reactions.add()
        new_r.name = f"Reaction_{len(interaction.reactions)}"
        interaction.reactions_index = len(interaction.reactions) - 1
        return {'FINISHED'}


class EXPLORATORY_OT_RemoveReactionFromInteraction(bpy.types.Operator):
    """Remove a ReactionDefinition from the currently selected Interaction by index."""
    bl_idname = "exploratory.remove_reaction_from_interaction"
    bl_label = "Remove Reaction"

    index: bpy.props.IntProperty()

    def execute(self, context):
        scene = context.scene
        i_idx = scene.custom_interactions_index
        if i_idx < 0 or i_idx >= len(scene.custom_interactions):
            self.report({'WARNING'}, "No valid Interaction selected.")
            return {'CANCELLED'}

        interaction = scene.custom_interactions[i_idx]
        if 0 <= self.index < len(interaction.reactions):
            interaction.reactions.remove(self.index)
            interaction.reactions_index = max(
                0, 
                min(self.index, len(interaction.reactions) - 1)
            )
        return {'FINISHED'}


###############################################################################
# 4) The Main check_interactions() Logic
###############################################################################
def check_interactions(context):
    scene = context.scene
    current_time = get_game_time()

    # ─── CLAMP ANY INVALID trigger_mode VALUES ───
    for inter in scene.custom_interactions:
        allowed = [item[0] for item in trigger_mode_items(inter, context)]
        if inter.trigger_mode not in allowed:
            inter.trigger_mode = allowed[0]

    pressed_this_frame = was_interact_pressed()

    update_all_objective_timers(context.scene)

    for inter in scene.custom_interactions:
        t = inter.trigger_type

        if t == "PROXIMITY":
            handle_proximity_trigger(inter, current_time)
        elif t == "COLLISION":
            handle_collision_trigger(inter, current_time)
        elif t == "INTERACT":
            handle_interact_trigger(inter, current_time, pressed_this_frame, context)
        elif t == "OBJECTIVE_UPDATE":
            handle_objective_update_trigger(inter, current_time)
        elif t == "TIMER_COMPLETE":
            handle_timer_complete_trigger(inter)


    # Now process any pending triggers that have matured.
    process_pending_triggers()

    global _global_interact_pressed
    _global_interact_pressed = False


###############################################################################
# 4)trigger helpers
###############################################################################
def schedule_trigger(interaction, fire_time):
    """
    Schedule the given interaction's reactions to fire at fire_time (game time).
    """
    _pending_trigger_tasks.append({
        "interaction": interaction,
        "fire_time": fire_time
    })

def process_pending_triggers():
    """
    Checks the pending trigger tasks. For each task whose fire_time has passed,
    fires its reactions and removes the task.
    """
    current_time = get_game_time()
    # Iterate over a shallow copy so that removal does not affect the loop.
    for task in _pending_trigger_tasks[:]:
        if current_time >= task["fire_time"]:
            run_reactions(task["interaction"].reactions)
            _pending_trigger_tasks.remove(task)

# -------------------------------------------------------------------
# 4.1) Proximity Trigger: ENTER/LEAVE (updated)
# -------------------------------------------------------------------
def handle_proximity_trigger(inter, current_time):
    obj_a = inter.proximity_object_a
    obj_b = inter.proximity_object_b
    dist_thresh = inter.proximity_distance

    if not obj_a or not obj_b:
        return

    dist = (obj_a.location - obj_b.location).length
    inside_now = (dist <= dist_thresh)
    was_in_zone = inter.is_in_zone
    inter.is_in_zone = inside_now

    # 1) ONE_SHOT: never re‐fire
    if inter.trigger_mode == "ONE_SHOT" and inter.has_fired:
        return

    # 2) ENTER
    if not was_in_zone and inside_now:
        # common fire logic
        def fire():
            if inter.trigger_delay > 0.0:
                schedule_trigger(inter, current_time + inter.trigger_delay)
            else:
                run_reactions(inter.reactions)
            inter.has_fired = True
            inter.last_trigger_time = current_time

        if inter.trigger_mode in {"ONE_SHOT", "ENTER_ONLY"}:
            fire()

        elif inter.trigger_mode == "COOLDOWN":
            time_since = current_time - inter.last_trigger_time
            if (not inter.has_fired) or (time_since >= inter.trigger_cooldown):
                fire()

    # 3) LEAVE → only ENTER_ONLY resets here
    elif was_in_zone and not inside_now:
        if inter.trigger_mode == "ENTER_ONLY":
            inter.has_fired = False

    # 4) STILL INSIDE: allow COOLDOWN to retrigger when timer expires
    else:
        if inside_now and inter.trigger_mode == "COOLDOWN" and inter.has_fired:
            time_since = current_time - inter.last_trigger_time
            if time_since >= inter.trigger_cooldown:
                if inter.trigger_delay > 0.0:
                    schedule_trigger(inter, current_time + inter.trigger_delay)
                else:
                    run_reactions(inter.reactions)
                inter.has_fired = True
                inter.last_trigger_time = current_time


###############################################################################
# 4.2) Collision Trigger: ENTER/LEAVE
###############################################################################
def handle_collision_trigger(inter, current_time):
    obj_a = inter.collision_object_a
    obj_b = inter.collision_object_b
    if not obj_a or not obj_b:
        return

    margin = inter.collision_margin
    colliding_now = bounding_sphere_collision(obj_a, obj_b, margin=margin)


    # ENTER
    if (not inter.is_in_zone) and colliding_now:
        inter.is_in_zone = True
        if can_fire_trigger(inter, current_time):
            if inter.trigger_delay > 0.0:
                schedule_trigger(inter, current_time + inter.trigger_delay)
            else:
                run_reactions(inter.reactions)
            inter.has_fired = True
            inter.last_trigger_time = current_time

    # LEAVE
    elif inter.is_in_zone and (not colliding_now):
        inter.is_in_zone = False
        reset_interaction_if_needed(inter)

    # STILL COLLIDING
    elif colliding_now and inter.is_in_zone:
        if inter.trigger_mode == "COOLDOWN":
            if can_fire_trigger(inter, current_time):
                if inter.trigger_delay > 0.0:
                    schedule_trigger(inter, current_time + inter.trigger_delay)
                else:
                    run_reactions(inter.reactions)
                inter.has_fired = True
                inter.last_trigger_time = current_time


###############################################################################
# 4.3) Interact Trigger
###############################################################################
def handle_interact_trigger(inter, current_time, pressed_now, context):
    # If it's one_shot and we've already fired => do nothing
    if inter.trigger_mode == "ONE_SHOT" and inter.has_fired:
        return

    # Find the player's character object
    char_obj = context.scene.target_armature
    if not char_obj:
        return

    # Check if we should display the interact prompt
    if inter.interact_object:
        distance = (char_obj.location - inter.interact_object.location).length
        if distance <= inter.interact_distance:
            prefs = context.preferences.addons["Exploratory"].preferences
            interact_key = prefs.key_interact
            exp_custom_ui.add_text_reaction(
                text_str=f"Press [{interact_key}] to interact",
                anchor='BOTTOM_CENTER',
                margin_x=0.5,
                margin_y=18,
                scale=3,
                end_time=get_game_time() + 0.1,
                color=(1, 1, 1, 1)
            )
    # If user did NOT press interact key => reset logic.
    if not pressed_now:
        if inter.trigger_mode != "ONE_SHOT" and inter.has_fired:
            if inter.trigger_mode == 'COOLDOWN':
                time_since = current_time - inter.last_trigger_time
                if time_since >= inter.trigger_cooldown:
                    inter.has_fired = False
            else:
                inter.has_fired = False
        return

    # The key was pressed this frame => check distance again.
    if inter.interact_object:
        dist = (char_obj.location - inter.interact_object.location).length
        if dist > inter.interact_distance:
            return  # Too far; do nothing

    if can_fire_trigger(inter, current_time):
        if inter.trigger_delay > 0.0:
            schedule_trigger(inter, current_time + inter.trigger_delay)
        else:
            run_reactions(inter.reactions)
        inter.has_fired = True
        inter.last_trigger_time = current_time


###############################################################################
# 4.4) Trigger Firing / Reset Helpers
###############################################################################
def can_fire_trigger(inter, current_time):
    # If One-Shot and we already fired, disallow forever.
    if inter.trigger_mode == "ONE_SHOT" and inter.has_fired:
        return False

    # If COOLDOWN and we already fired, check time
    if inter.trigger_mode == "COOLDOWN" and inter.has_fired:
        time_since = current_time - inter.last_trigger_time
        if time_since < inter.trigger_cooldown:
            return False

    # If ENTER_ONLY => we only require that has_fired be False at press time
    # but that’s handled by resetting has_fired on release/exit. So no extra check needed.
    return True




def reset_interaction_if_needed(inter):
    """
    Only ENTER_ONLY should clear has_fired on exit.
    COOLDOWN holds has_fired=True until cooldown expires.
    ONE_SHOT never resets.
    """
    if inter.trigger_mode == "ENTER_ONLY":
        inter.has_fired = False


###############################################################################
# 5) Reaction Execution
###############################################################################
# In exp_interactions.py (inside run_reactions)

def run_reactions(reactions):
    for r in reactions:
        if r.reaction_type == "CUSTOM_ACTION":
            execute_custom_action_reaction(r)
        elif r.reaction_type == "OBJECTIVE_COUNTER":
            execute_objective_counter_reaction(r)
        elif r.reaction_type == "OBJECTIVE_TIMER":
            execute_objective_timer_reaction(r)  # <--- NEW
        elif r.reaction_type == "CHAR_ACTION":
            execute_char_action_reaction(r)
        elif r.reaction_type == "SOUND":
            execute_sound_reaction(r)
        elif r.reaction_type == "TRANSFORM":
            execute_transform_reaction(r)
        elif r.reaction_type == "PROPERTY":
            execute_property_reaction(r)
        elif r.reaction_type == "CUSTOM_UI_TEXT":
            execute_custom_ui_text_reaction(r)
        elif r.reaction_type == "MOBILITY_GAME":
            execute_mobility_reaction(r)

        



###############################################################################
# 6) Collision Helper
###############################################################################
import mathutils
from mathutils import Vector

def bounding_sphere_collision(obj_a, obj_b, margin=0.0):
    """
    Actually uses Axis-Aligned Bounding Boxes (AABB).
    This version lets you specify a 'margin' so we count collisions
    if they're within 'margin' distance of touching.
    """

    if not obj_a or not obj_b:
        return False

    # Helper to get the min/max of an object's world AABB
    def get_world_aabb(obj):
        corners = []
        for corner_local in obj.bound_box:
            corner_world = obj.matrix_world @ Vector(corner_local)
            corners.append(corner_world)

        min_x = min(pt.x for pt in corners)
        max_x = max(pt.x for pt in corners)
        min_y = min(pt.y for pt in corners)
        max_y = max(pt.y for pt in corners)
        min_z = min(pt.z for pt in corners)
        max_z = max(pt.z for pt in corners)
        return (min_x, max_x, min_y, max_y, min_z, max_z)

    # Get each object's AABB
    A_minx, A_maxx, A_miny, A_maxy, A_minz, A_maxz = get_world_aabb(obj_a)
    B_minx, B_maxx, B_miny, B_maxy, B_minz, B_maxz = get_world_aabb(obj_b)

    # Inflate each AABB by margin on all sides
    A_minx -= margin
    A_maxx += margin
    A_miny -= margin
    A_maxy += margin
    A_minz -= margin
    A_maxz += margin

    B_minx -= margin
    B_maxx += margin
    B_miny -= margin
    B_maxy += margin
    B_minz -= margin
    B_maxz += margin

    # Check overlap in X, Y, and Z
    overlap_x = (A_minx <= B_maxx) and (A_maxx >= B_minx)
    overlap_y = (A_miny <= B_maxy) and (A_maxy >= B_miny)
    overlap_z = (A_minz <= B_maxz) and (A_maxz >= B_minz)

    return (overlap_x and overlap_y and overlap_z)


def approximate_bounding_sphere_radius(obj):
    """
    Approximate bounding-sphere radius from bounding-box diagonal & local scale.
    """
    local_min = Vector(obj.bound_box[0])
    local_max = Vector(obj.bound_box[0])

    for corner in obj.bound_box:
        local_min.x = min(local_min.x, corner[0])
        local_min.y = min(local_min.y, corner[1])
        local_min.z = min(local_min.z, corner[2])

        local_max.x = max(local_max.x, corner[0])
        local_max.y = max(local_max.y, corner[1])
        local_max.z = max(local_max.z, corner[2])

    bb_size = local_max - local_min
    local_radius = 0.5 * bb_size.length

    scale_factors = obj.scale
    max_scale = max(abs(scale_factors.x), abs(scale_factors.y), abs(scale_factors.z))
    return local_radius * max_scale


###############################################################################
# 7) Interact Helper
###############################################################################

_global_interact_pressed = False  # <--- global placeholder

def set_interact_pressed(value: bool):
    """
    Called by the modal operator to tell us the Interact key was pressed or released.
    """
    global _global_interact_pressed
    _global_interact_pressed = value

def was_interact_pressed():

    """
    Return True once when the Interact key is pressed. 
    Resets to False so it doesn't continuously fire if the user holds the key.
    """
    global _global_interact_pressed
    if _global_interact_pressed:
        return True
    return False


def reset_all_interactions(scene):
    now = get_game_time()
    for inter in scene.custom_interactions:
        inter.has_fired = False
        inter.last_trigger_time = now
        inter.is_in_zone = False

        if inter.trigger_type == "OBJECTIVE_UPDATE":
            if inter.objective_index.isdigit():
                idx = int(inter.objective_index)
                if 0 <= idx < len(scene.objectives):
                    inter.last_known_count = scene.objectives[idx].current_value
                else:
                    inter.last_known_count = -1
            else:
                inter.last_known_count = -1


###############################################################################
# 8) Objective Trigger Helper
###############################################################################

def handle_objective_update_trigger(inter, current_time):
    scene = bpy.context.scene
    if not inter.objective_index.isdigit():
        return  # no valid objective chosen

    idx = int(inter.objective_index)
    if idx < 0 or idx >= len(scene.objectives):
        return  # out of range

    objv = scene.objectives[idx]
    old_val = inter.last_known_count
    new_val = objv.current_value

    if inter.trigger_mode == "ONE_SHOT" and inter.has_fired:
        return

    fired_now = False

    if inter.objective_condition == "CHANGED":
        if new_val != old_val:
            fired_now = True

    elif inter.objective_condition == "INCREASED":
        if new_val > old_val:
            fired_now = True

    elif inter.objective_condition == "DECREASED":
        if new_val < old_val and old_val >= 0:
            fired_now = True

    elif inter.objective_condition == "EQUALS":
        if new_val == inter.objective_condition_value:
            fired_now = True

    elif inter.objective_condition == "AT_LEAST":
        if new_val >= inter.objective_condition_value:
            fired_now = True

    if fired_now:
        if can_fire_trigger(inter, current_time):
            if inter.trigger_delay > 0.0:
                schedule_trigger(inter, current_time + inter.trigger_delay)
            else:
                run_reactions(inter.reactions)
            inter.has_fired = True
            inter.last_trigger_time = current_time

    inter.last_known_count = new_val


def handle_timer_complete_trigger(inter):
    scene = bpy.context.scene
    idx_str = inter.timer_objective_index
    if not idx_str.isdigit():
        return

    idx = int(idx_str)
    if idx < 0 or idx >= len(scene.objectives):
        return

    objv = scene.objectives[idx]
    if objv.just_finished:
        current_time = get_game_time()
        if can_fire_trigger(inter, current_time):
            if inter.trigger_delay > 0.0:
                schedule_trigger(inter, current_time + inter.trigger_delay)
            else:
                run_reactions(inter.reactions)
            inter.has_fired = True
            inter.last_trigger_time = current_time

            objv.timer_active = False
            objv.just_finished = False


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


def unregister_interaction_properties():
    del bpy.types.Scene.custom_interactions
    del bpy.types.Scene.custom_interactions_index




