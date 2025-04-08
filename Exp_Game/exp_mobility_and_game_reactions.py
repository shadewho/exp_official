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
    # --- Mesh Visibility Properties ---
    mesh_object: bpy.props.PointerProperty(
        name="Mesh Object",
        type=bpy.types.Object,
        description="Mesh object to be hidden or unhidden"
    )
    mesh_action: bpy.props.EnumProperty(
        name="Mesh Action on Trigger",
        description="Select whether to hide, unhide or toggle the mesh object's visibility when the trigger condition is met",
        items=[
            ("NONE", "None", "Do not change mesh visibility"),
            ("HIDE", "Hide", "Hide the mesh object"),
            ("UNHIDE", "Unhide", "Unhide the mesh object"),
            ("TOGGLE", "Toggle", "Toggle mesh visibility based on its current state")
        ],
        default="NONE"
    )

def execute_mobility_reaction(reaction):
    """
    Reads the mobility and mesh settings from the reaction, copies them to the scene-level pointer,
    and then applies the mesh visibility change based on the chosen action.
    """
    scene = bpy.context.scene

    if not hasattr(scene, "mobility_game"):
        print("Scene has no 'mobility_game' pointer! Did you define it in __init__?")
        return

    src = reaction.mobility_game_settings  # Reaction-defined settings
    dst = scene.mobility_game              # Scene-level pointer

    # Copy mobility settings
    dst.allow_movement = src.allow_movement
    dst.allow_jump     = src.allow_jump
    dst.allow_sprint   = src.allow_sprint

    # Copy the mesh settings
    dst.mesh_object = src.mesh_object
    dst.mesh_action = src.mesh_action

    # --- Apply Mesh Visibility Change ---
    if src.mesh_object and src.mesh_action != "NONE":
        if src.mesh_action == "HIDE":
            src.mesh_object.hide_viewport = True
        elif src.mesh_action == "UNHIDE":
            src.mesh_object.hide_viewport = False
        elif src.mesh_action == "TOGGLE":
            # Toggle: if currently hidden, unhide; otherwise hide.
            src.mesh_object.hide_viewport = not src.mesh_object.hide_viewport