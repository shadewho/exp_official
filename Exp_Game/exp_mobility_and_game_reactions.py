# File: exp_mobility_and_game_reactions.py

import bpy
from bpy.types import PropertyGroup
from .exp_time import get_game_time

class MobilityGameReactionsPG(PropertyGroup):
    """
    All booleans and other settings that control 
    the player's ability to move, jump, run, or alter game time.
    By default, everything is True => no locks.
    """
    allow_movement: bpy.props.BoolProperty(
        name="Allow Movement",
        default=True,
        description="If False, W/A/S/D is blocked"
    )
    allow_jump: bpy.props.BoolProperty(
        name="Allow Jump",
        default=True,
        description="If False, jump is blocked"
    )
    allow_sprint: bpy.props.BoolProperty(
        name="Allow Sprint",
        default=True,
        description="If False, SHIFT sprint is blocked"
    )

def execute_mobility_reaction(reaction):
    """
    If reaction.reaction_type == 'MOBILITY_GAME',
    we read the new booleans from reaction.mobility_game_settings
    and store them into the *same* property group in scene.mobility_game
    (or scene.input_locks, whichever pointer you use).
    """
    scene = bpy.context.scene

    # The ReactionDefinition has a pointer:
    #   reaction.mobility_game_settings : MobilityGameReactionsPG
    # We want to copy those values into the scene-level pointer so
    # that the modal can reference them each frame.
    if not hasattr(scene, "mobility_game"):
        print("Scene has no 'mobility_game' pointer! Did you define it in __init__?")
        return

    src = reaction.mobility_game_settings  # from the Reaction
    dst = scene.mobility_game             # the global/scene-level pointer

    # Overwrite all fields:
    dst.allow_movement = src.allow_movement
    dst.allow_jump     = src.allow_jump
    dst.allow_sprint   = src.allow_sprint