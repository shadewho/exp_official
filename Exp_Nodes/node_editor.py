# Exp_Nodes/node_editor.py
import bpy
from bpy.types import NodeTree, Node, NodeSocket
from nodeitems_utils import NodeCategory, NodeItem, register_node_categories, unregister_node_categories
from .trigger_nodes import TriggerNode
from .reaction_nodes import ReactionNode, ReactionTriggerInputSocket, ReactionOutputSocket, NODE_OT_add_reaction_to_node
from .interaction_nodes import InteractionNode, NODE_OT_add_interaction_to_node, NODE_OT_remove_interaction_from_node
# -----------------------------
# Define the custom node tree
# -----------------------------
class ExploratoryNodesTree(bpy.types.NodeTree):
    """A custom node tree for the Exploratory Node Editor"""
    bl_idname = 'ExploratoryNodesTreeType'
    bl_label = 'Exploratory Node Editor Tree'
    bl_icon = 'NODETREE'

# -----------------------------
# Define a custom socket (optional)
# -----------------------------

# NEW: Custom socket for Interaction connections (blue)
class InteractionSocket(bpy.types.NodeSocket):
    bl_idname = 'InteractionSocketType'
    bl_label = 'Interaction Socket'
    
    def draw(self, context, layout, node, text):
        layout.label(text=text)
    
    def draw_color(self, context, node):
        # Blue color (you can adjust the values as you like)
        return (0.4, 0.4, 1.0, 1.0)

# NEW: Custom socket for Trigger Node outputs (purple)
class TriggerOutputSocket(bpy.types.NodeSocket):
    bl_idname = 'TriggerOutputSocketType'
    bl_label = 'Trigger Output Socket'
    
    def draw(self, context, layout, node, text):
        layout.label(text=text)
    
    def draw_color(self, context, node):
        # Purple color (for example: 0.5 red, 0.0 green, 0.5 blue)
        return (0.8, 0.3, 0.8, 1.0)


# -----------------------------
# Define Node Categories for the Shift+A menu
# -----------------------------
node_categories = [
    NodeCategory("TRIGGERS", "Triggers", items=[
        NodeItem("TriggerNodeType"),
    ]),
    NodeCategory("REACTIONS", "Reactions", items=[
        NodeItem("ReactionNodeType"),
    ]),
    NodeCategory("INTERACTIONS", "Interactions", items=[
        NodeItem("InteractionNodeType"),
    ]),
]


# -----------------------------
# Operator and Panel for the Node Editor sidebar
# -----------------------------
class NODE_OT_create_exploratory_node_tree(bpy.types.Operator):
    bl_idname = "node.create_exploratory_node_tree"
    bl_label = "Create Exploratory Node Tree"
    
    def execute(self, context):
        new_tree = bpy.data.node_groups.new("Exploratory Node Tree", 'ExploratoryNodesTreeType')
        for area in context.screen.areas:
            if area.type == 'NODE_EDITOR':
                space = area.spaces.active
                space.tree_type = 'ExploratoryNodesTreeType'
                space.node_tree = new_tree
                break
        self.report({'INFO'}, "Exploratory Node Tree created.")
        return {'FINISHED'}

class NODE_PT_exploratory_panel(bpy.types.Panel):
    bl_label = "Exploratory Node Editor"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == 'ExploratoryNodesTreeType'
    
    def draw(self, context):
        layout = self.layout
        layout.operator("node.create_exploratory_node_tree", icon='NODETREE')
        layout.separator()
        layout.label(text="Use Shift+A to add nodes by category.")

# -----------------------------
# Registration
# -----------------------------

classes = [
    ExploratoryNodesTree,
    InteractionSocket,
    TriggerOutputSocket,
    TriggerNode,
    InteractionNode,
    ReactionNode,
    ReactionTriggerInputSocket,  # Include the custom input socket class.
    ReactionOutputSocket,        # Include the custom output socket class.
    NODE_OT_add_reaction_to_node,
    NODE_OT_create_exploratory_node_tree,
    NODE_PT_exploratory_panel,
    NODE_OT_add_interaction_to_node,
    NODE_OT_remove_interaction_from_node
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    register_node_categories(ExploratoryNodesTree.bl_idname, node_categories)

def unregister():
    unregister_node_categories(ExploratoryNodesTree.bl_idname)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
