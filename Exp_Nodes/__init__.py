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

from ..Exp_Game.props_and_utils.exp_utility_store import (
    register_utility_store_properties,
    unregister_utility_store_properties,
)

from ..Exp_Game.props_and_utils.trackers import (
    register_tracker_properties,
    unregister_tracker_properties,
)

from .utility_nodes import (
    # Sockets - Float
    FloatInputSocket,
    FloatOutputSocket,
    # Sockets - Integer
    IntInputSocket,
    IntOutputSocket,
    # Sockets - Boolean
    BoolInputSocket,
    BoolOutputSocket,
    # Sockets - Object
    ObjectInputSocket,
    ObjectOutputSocket,
    # Sockets - Collection
    CollectionInputSocket,
    CollectionOutputSocket,
    # Sockets - Action
    ActionInputSocket,
    ActionOutputSocket,
    # Sockets - Float Vector
    FloatVectorInputSocket,
    FloatVectorOutputSocket,
    # Data Nodes
    FloatDataNode,
    IntDataNode,
    BoolDataNode,
    ObjectDataNode,
    CollectionDataNode,
    ActionDataNode,
    FloatVectorDataNode,
)

from .node_editor import (
    # tree + socket
    ExploratoryNodesTree,
    TriggerOutputSocket,
    TriggerInputSocket,
    # operators + panel
    NODE_OT_create_exploratory_node_tree,
    NODE_OT_select_exploratory_node_tree,
    NODE_OT_delete_exploratory_node_tree,
    NODE_OT_rename_exploratory_node_tree,
    NODE_OT_reassign_exploratory_node_tree,
    NODE_PT_exploratory_panel,

    # menus (native Add → Exploratory → …)
    NODE_MT_exploratory_add_triggers,
    NODE_MT_exploratory_add_reactions,
    NODE_MT_exploratory_add_objectives,
    NODE_MT_exploratory_add_actions,
    NODE_MT_exploratory_add_utilities,
    NODE_MT_exploratory_add_trackers,
    # hook used to append menu entry to NODE_MT_add
    _append_exploratory_entry,
)
from .action_key_nodes import CreateActionKeyNode

# ── TRACKERS ──
from .tracker_nodes import (
    TrackerObjectInputSocket,
    TrackerFloatInputSocket,
    TrackerBoolSocket,
    DistanceTrackerNode,
    StateTrackerNode,
    ContactTrackerNode,
    InputTrackerNode,
    GameTimeTrackerNode,
    LogicAndNode,
    LogicOrNode,
    LogicNotNode,
)

# ── TRIGGERS ──
from .trigger_nodes import (
    ProximityTriggerNode,
    CollisionTriggerNode,
    InteractTriggerNode,
    ObjectiveUpdateTriggerNode,
    TimerCompleteTriggerNode,
    OnGameStartTriggerNode,
    ActionTriggerNode,
    ExternalTriggerNode, 
)

# ── REACTIONS ──
from .reaction_nodes import (
    ReactionTriggerInputSocket,
    ReactionOutputSocket,
    ImpactEventOutputSocket,
    ImpactLocationOutputSocket,
    # Dynamic property sockets (draw inline with fields)
    DynamicObjectInputSocket,
    DynamicBoolInputSocket,
    DynamicFloatInputSocket,
    DynamicActionInputSocket,
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
    ReactionParentingNode,
    ReactionTrackingNode,
)

from .trig_react_obj_lists import(
    EXPL_OT_delete_orphaned,
    VIEW3D_PT_Exploratory_Studio,
    VIEW3D_PT_Exploratory_Reactions,
    VIEW3D_PT_Objectives,
    VIEW3D_PT_ActionKeys,
    VIEW3D_PT_Trackers,
    VIEW3D_PT_Utilities,
)

# ── OBJECTIVES ──
from .objective_nodes import ObjectiveNode

classes = [
    # tree + socket
    ExploratoryNodesTree,
    TriggerOutputSocket,
    TriggerInputSocket, 

    # triggers
    ProximityTriggerNode,
    CollisionTriggerNode,
    InteractTriggerNode,
    ObjectiveUpdateTriggerNode,
    TimerCompleteTriggerNode,
    OnGameStartTriggerNode,
    ActionTriggerNode,
    ExternalTriggerNode, 



    # utilities - sockets
    FloatInputSocket,
    FloatOutputSocket,
    IntInputSocket,
    IntOutputSocket,
    BoolInputSocket,
    BoolOutputSocket,
    ObjectInputSocket,
    ObjectOutputSocket,
    CollectionInputSocket,
    CollectionOutputSocket,
    ActionInputSocket,
    ActionOutputSocket,
    FloatVectorInputSocket,
    FloatVectorOutputSocket,
    # utilities - data nodes
    FloatDataNode,
    IntDataNode,
    BoolDataNode,
    ObjectDataNode,
    CollectionDataNode,
    ActionDataNode,
    FloatVectorDataNode,
    # utilities - delay
    UtilityDelayNode,
    
    # reaction sockets
    ReactionTriggerInputSocket,
    ReactionOutputSocket,
    ImpactEventOutputSocket,
    ImpactLocationOutputSocket,
    # dynamic property sockets (draw inline)
    DynamicObjectInputSocket,
    DynamicBoolInputSocket,
    DynamicFloatInputSocket,
    DynamicActionInputSocket,
    # reaction nodes
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
    ReactionParentingNode,
    ReactionTrackingNode,

    # objective
    ObjectiveNode,

    #actions
    CreateActionKeyNode,

    # operators + panel
    NODE_OT_create_exploratory_node_tree,
    NODE_OT_select_exploratory_node_tree,
    NODE_OT_delete_exploratory_node_tree,
    NODE_OT_rename_exploratory_node_tree,
    NODE_OT_reassign_exploratory_node_tree,
    NODE_PT_exploratory_panel,

    # menus
    NODE_MT_exploratory_add_triggers,
    NODE_MT_exploratory_add_reactions,
    NODE_MT_exploratory_add_objectives,
    NODE_MT_exploratory_add_actions,
    NODE_MT_exploratory_add_utilities,
    NODE_MT_exploratory_add_trackers,

    # tracker sockets (must register before nodes)
    TrackerObjectInputSocket,
    TrackerFloatInputSocket,
    TrackerBoolSocket,
    # tracker nodes
    DistanceTrackerNode,
    StateTrackerNode,
    ContactTrackerNode,
    InputTrackerNode,
    GameTimeTrackerNode,
    LogicAndNode,
    LogicOrNode,
    LogicNotNode,

    EXPL_OT_delete_orphaned,
    VIEW3D_PT_Exploratory_Studio,
    VIEW3D_PT_Exploratory_Reactions,
    VIEW3D_PT_Objectives,
    VIEW3D_PT_ActionKeys,
    VIEW3D_PT_Trackers,
    VIEW3D_PT_Utilities,
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
    register_utility_store_properties()
    register_tracker_properties()
    from bpy.types import NODE_MT_add
    NODE_MT_add.append(_append_exploratory_entry)

def unregister():
    _sanitize_open_node_editors_before_unload()
    from bpy.types import NODE_MT_add
    try: NODE_MT_add.remove(_append_exploratory_entry)
    except Exception: pass
    for cls in reversed(classes):
        try: bpy.utils.unregister_class(cls)
        except Exception: pass
    unregister_utility_store_properties()
    unregister_tracker_properties()




if __name__ == "__main__":
    register()
