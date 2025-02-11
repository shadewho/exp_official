from .node_editor import (
    ExploratoryNodesTree, InteractionSocket, TriggerOutputSocket,
    NODE_OT_create_exploratory_node_tree, NODE_PT_exploratory_panel,
    NODE_OT_delete_exploratory_node_tree, node_categories, NODE_PT_character_actions_panel,
    NODE_PT_exploratory_proxy, NODE_OT_select_exploratory_node_tree
)
from .trigger_nodes import TriggerNode
from .reaction_nodes import ReactionNode, ReactionTriggerInputSocket, ReactionOutputSocket, NODE_OT_add_reaction_to_node
from .interaction_nodes import InteractionNode, NODE_OT_add_interaction_to_node, NODE_OT_remove_interaction_from_node
from .objective_nodes import ObjectiveNode

# List each class only once
classes = [
    ExploratoryNodesTree,
    InteractionSocket,
    TriggerOutputSocket,
    TriggerNode,
    InteractionNode,
    ReactionNode,
    ObjectiveNode,
    ReactionTriggerInputSocket,
    ReactionOutputSocket,
    NODE_OT_add_reaction_to_node,
    NODE_OT_create_exploratory_node_tree,
    NODE_OT_select_exploratory_node_tree,
    NODE_PT_exploratory_panel,
    NODE_PT_character_actions_panel,
    NODE_PT_exploratory_proxy,
    NODE_OT_add_interaction_to_node,
    NODE_OT_remove_interaction_from_node,
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
