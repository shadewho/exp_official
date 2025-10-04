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
from .node_editor import (
    ExploratoryNodesTree,
    TriggerOutputSocket,
    NODE_OT_create_exploratory_node_tree,
    NODE_PT_exploratory_panel,
    NODE_OT_delete_exploratory_node_tree,
    NODE_OT_select_exploratory_node_tree,
    node_categories,
)

# ── TRIGGERS: five separate nodes ──
from .trigger_nodes import (
    ProximityTriggerNode,
    CollisionTriggerNode,
    InteractTriggerNode,
    ObjectiveUpdateTriggerNode,
    TimerCompleteTriggerNode,
)

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
    ReactionMobilityGameNode,
    NODE_OT_add_reaction_to_node,
)

from .objective_nodes import ObjectiveNode

classes = [
    ExploratoryNodesTree,
    TriggerOutputSocket,

    # five concrete triggers
    ProximityTriggerNode,
    CollisionTriggerNode,
    InteractTriggerNode,
    ObjectiveUpdateTriggerNode,
    TimerCompleteTriggerNode,

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
    ReactionMobilityGameNode,

    # objective
    ObjectiveNode,

    # ops / panel
    NODE_OT_add_reaction_to_node,
    NODE_OT_create_exploratory_node_tree,
    NODE_OT_select_exploratory_node_tree,
    NODE_PT_exploratory_panel,
    NODE_OT_delete_exploratory_node_tree,
]

def register():
    import bpy
    for cls in classes:
        bpy.utils.register_class(cls)
    from nodeitems_utils import register_node_categories
    register_node_categories(ExploratoryNodesTree.bl_idname, node_categories)

def unregister():
    from nodeitems_utils import unregister_node_categories
    unregister_node_categories(ExploratoryNodesTree.bl_idname)
    import bpy
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()

