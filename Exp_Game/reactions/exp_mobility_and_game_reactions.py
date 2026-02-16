# Exploratory/Exp_Game/reactions/exp_mobility_and_game_reactions.py

import bpy
from bpy.types import PropertyGroup

# ─────────────────────────────────────────────────────────
# PropertyGroups for the split reactions
# ─────────────────────────────────────────────────────────

class MobilityReactionsPG(PropertyGroup):
    """Toggles for basic character mobility."""
    allow_movement: bpy.props.BoolProperty(
        name="Allow Movement",
        default=True,
        description="If False, W/A/S/D is blocked",
    )
    allow_jump: bpy.props.BoolProperty(
        name="Allow Jump",
        default=True,
        description="If False, jump is blocked",
    )
    allow_sprint: bpy.props.BoolProperty(
        name="Allow Sprint",
        default=True,
        description="If False, SHIFT sprint is blocked",
    )


class MeshVisibilityReactionsPG(PropertyGroup):
    """Fields for mesh visibility control."""
    mesh_object: bpy.props.PointerProperty(
        name="Mesh Object",
        type=bpy.types.Object,
        description="Mesh object to be hidden or unhidden",
    )
    mesh_action: bpy.props.EnumProperty(
        name="Action",
        description="Visibility action to perform on trigger",
        items=[
            ("NONE",   "None",   "Do not change mesh visibility"),
            ("HIDE",   "Hide",   "Hide the mesh object"),
            ("UNHIDE", "Unhide", "Unhide the mesh object"),
            ("TOGGLE", "Toggle", "Toggle visibility"),
        ],
        default="NONE",
    )

# ─────────────────────────────────────────────────────────
# Executions for the split reactions
# ─────────────────────────────────────────────────────────

def execute_mobility_reaction(reaction):
    """
    Apply only the mobility toggles.
    Copies fields from reaction.mobility_settings -> scene.mobility_game (existing container).
    """
    scene = bpy.context.scene
    if not hasattr(scene, "mobility_game"):
        # If your scene.mobility_game PG is not present, nothing to update.
        return

    ms = getattr(reaction, "mobility_settings", None)
    if not ms:
        return

    dst = scene.mobility_game
    dst.allow_movement = ms.allow_movement
    dst.allow_jump     = ms.allow_jump
    dst.allow_sprint   = ms.allow_sprint


def execute_mesh_visibility_reaction(reaction):
    """
    Perform only mesh visibility operations.
    """
    from .exp_bindings import resolve_object

    vs = getattr(reaction, "mesh_visibility", None)
    if not vs:
        return

    # Resolve from binding (socket connection) or fall back to nested property
    obj = resolve_object(reaction, "mesh_vis_object", vs.mesh_object)
    if not obj:
        return

    act = vs.mesh_action

    if act == "HIDE":
        obj.hide_viewport = True
    elif act == "UNHIDE":
        obj.hide_viewport = False
    elif act == "TOGGLE":
        obj.hide_viewport = not obj.hide_viewport
    # act == "NONE" → do nothing


def execute_reset_game_reaction(_reaction):
    """
    Make RESET_GAME from reactions behave exactly like the reset key:
    run the operator on the next tick so no trigger handler writes happen after it.
    """

    def _do_reset():
        try:
            bpy.ops.exploratory.reset_game('INVOKE_DEFAULT')
        except Exception:
            pass
        return None  # run once

    bpy.app.timers.register(_do_reset, first_interval=0.0)
