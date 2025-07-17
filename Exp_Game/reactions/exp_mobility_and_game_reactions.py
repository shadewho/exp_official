# File: exp_mobility_and_game_reactions.py

import bpy
from bpy.types import PropertyGroup
from ..props_and_utils.exp_time import get_game_time

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

    # ─── Reset Game Switch ────────────────────────────────
    reset_game: bpy.props.BoolProperty(
        name="Reset Game",
        default=False,
        description="If True, fires a full game reset when this reaction triggers"
    )


def execute_mobility_reaction(reaction):
    """
    Applies mobility/mesh changes. If reset_game is set, schedules
    a full game reset via a bpy.app.timers callback so that 
    handle_interact_trigger can finish updating its flags first.
    """
    scene = bpy.context.scene
    if not hasattr(scene, "mobility_game"):
        print("Scene has no 'mobility_game' pointer!")
        return

    src = reaction.mobility_game_settings
    dst = scene.mobility_game

    # Copy over movement locks
    dst.allow_movement = src.allow_movement
    dst.allow_jump     = src.allow_jump
    dst.allow_sprint   = src.allow_sprint

    # Mesh visibility
    dst.mesh_object = src.mesh_object
    dst.mesh_action = src.mesh_action
    if src.mesh_object and src.mesh_action != "NONE":
        if src.mesh_action == "HIDE":
            src.mesh_object.hide_viewport = True
        elif src.mesh_action == "UNHIDE":
            src.mesh_object.hide_viewport = False
        elif src.mesh_action == "TOGGLE":
            src.mesh_object.hide_viewport = not src.mesh_object.hide_viewport

    # Schedule game reset (on next frame)
    if src.reset_game:
        bpy.app.timers.register(
            lambda: bpy.ops.exploratory.reset_game('INVOKE_DEFAULT'),
            first_interval=0.0
        )