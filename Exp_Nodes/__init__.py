# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# File: Exp_Nodes/__init__.py
import bpy
from .node_editor import (
    # tree + socket
    ExploratoryNodesTree,
    TriggerOutputSocket,

    # operators + panel
    NODE_OT_create_exploratory_node_tree,
    NODE_OT_select_exploratory_node_tree,
    NODE_OT_delete_exploratory_node_tree,
    NODE_OT_rename_exploratory_node_tree,
    NODE_PT_exploratory_panel,

    # menus (native Add → Exploratory → …)
    NODE_MT_exploratory_add_triggers,
    NODE_MT_exploratory_add_reactions,
    NODE_MT_exploratory_add_objectives,
    NODE_MT_exploratory_add_actions,
    NODE_MT_exploratory_add_utilities,
    # hook used to append menu entry to NODE_MT_add
    _append_exploratory_entry,
)
from .action_key_nodes import CreateActionKeyNode

# ── TRIGGERS ──
from .trigger_nodes import (
    ProximityTriggerNode,
    CollisionTriggerNode,
    InteractTriggerNode,
    ObjectiveUpdateTriggerNode,
    TimerCompleteTriggerNode,
    OnGameStartTriggerNode,
    ActionTriggerNode
)

# ── REACTIONS ──
from .reaction_nodes import (
    ReactionTriggerInputSocket,
    ReactionOutputSocket,
    ReactionCustomActionNode,
    ReactionCharActionNode,
    ReactionSoundNode,
    ReactionPropertyNode,
    ReactionTransformNode,
    ReactionCustomTextNode,
    ReactionObjectiveCounterNode,
    ReactionObjectiveTimerNode,
    ReactionMobilityNode,
    ReactionMeshVisibilityNode,
    ReactionResetGameNode,
    ReactionCrosshairsNode,
    ReactionProjectileNode,
    ReactionHitscanNode,
    ReactionActionKeysNode,
    UtilityDelayNode,
)

from .trig_react_obj_lists import(
    VIEW3D_PT_Exploratory_Studio,
    EXPLORATORY_UL_CustomInteractions,
    EXPLORATORY_UL_ReactionsInInteraction,
    EXPLORATORY_UL_ReactionLibrary,
    EXPLORATORY_OT_AddGlobalReaction,
    EXPLORATORY_OT_RemoveGlobalReaction,
    VIEW3D_PT_Exploratory_Reactions,
    VIEW3D_PT_Objectives,
    EXPLORATORY_UL_Objectives,
)

# ── OBJECTIVES ──
from .objective_nodes import ObjectiveNode


classes = [
    # tree + socket
    ExploratoryNodesTree,
    TriggerOutputSocket,

    # triggers
    ProximityTriggerNode,
    CollisionTriggerNode,
    InteractTriggerNode,
    ObjectiveUpdateTriggerNode,
    TimerCompleteTriggerNode,
    OnGameStartTriggerNode,
    ActionTriggerNode,


    # utilities
    UtilityDelayNode, 

    # reactions
    ReactionTriggerInputSocket,
    ReactionOutputSocket,
    ReactionCustomActionNode,
    ReactionCharActionNode,
    ReactionSoundNode,
    ReactionPropertyNode,
    ReactionTransformNode,
    ReactionCustomTextNode,
    ReactionObjectiveCounterNode,
    ReactionObjectiveTimerNode,
    ReactionMobilityNode,
    ReactionMeshVisibilityNode,
    ReactionResetGameNode,
    ReactionCrosshairsNode,
    ReactionProjectileNode,
    ReactionHitscanNode,
    ReactionActionKeysNode,


    # objective
    ObjectiveNode,

    #actions
    CreateActionKeyNode,

    # operators + panel
    NODE_OT_create_exploratory_node_tree,
    NODE_OT_select_exploratory_node_tree,
    NODE_OT_delete_exploratory_node_tree,
    NODE_OT_rename_exploratory_node_tree,
    NODE_PT_exploratory_panel,

    # menus
    NODE_MT_exploratory_add_triggers,
    NODE_MT_exploratory_add_reactions,
    NODE_MT_exploratory_add_objectives,
    NODE_MT_exploratory_add_actions,

    VIEW3D_PT_Exploratory_Studio,
    EXPLORATORY_UL_CustomInteractions,
    EXPLORATORY_UL_ReactionsInInteraction,
    EXPLORATORY_UL_ReactionLibrary,
    EXPLORATORY_OT_AddGlobalReaction,
    EXPLORATORY_OT_RemoveGlobalReaction,
    VIEW3D_PT_Exploratory_Reactions,
    VIEW3D_PT_Objectives,
    EXPLORATORY_UL_Objectives,
    NODE_MT_exploratory_add_utilities,
]

EXPL_TREE_ID = "ExploratoryNodesTreeType"

def _sanitize_open_node_editors_before_unload():
    """Switch any open Node Editor that is currently showing our custom tree
    to a built-in tree type to avoid enum -1 warnings during unregister."""
    import bpy
    wm = getattr(bpy.context, "window_manager", None)
    if not wm:
        return

    for win in wm.windows:
        scr = getattr(win, "screen", None)
        if not scr:
            continue
        for area in scr.areas:
            if area.type != 'NODE_EDITOR':
                continue
            for space in area.spaces:
                if getattr(space, "type", None) != 'NODE_EDITOR':
                    continue
                # If the space is currently bound to our custom tree, detach it.
                if getattr(space, "tree_type", "") == EXPL_TREE_ID:
                    try:
                        # Drop the specific node tree reference first
                        space.node_tree = None
                    except Exception:
                        pass
                    try:
                        # Pick a valid built-in tree. Shader is usually safe.
                        space.tree_type = 'ShaderNodeTree'
                    except Exception:
                        # As a last resort, try blanking it.
                        try:
                            space.tree_type = ''
                        except Exception:
                            pass

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Append the scoped Shift+A entry (only shows in our editor)
    from bpy.types import NODE_MT_add
    NODE_MT_add.append(_append_exploratory_entry)


def unregister():
    # 0) Detach any UI that’s still looking at our tree to prevent enum -1 spam
    _sanitize_open_node_editors_before_unload()

    # 1) Remove the scoped Shift+A entry first
    from bpy.types import NODE_MT_add
    try:
        NODE_MT_add.remove(_append_exploratory_entry)
    except Exception:
        pass

    # 2) Unregister classes (reverse order)
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


if __name__ == "__main__":
    register()

