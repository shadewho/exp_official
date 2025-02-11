# Exp_Nodes/node_editor.py
import bpy
from bpy.types import NodeTree, Node, NodeSocket
from nodeitems_utils import NodeCategory, NodeItem, register_node_categories, unregister_node_categories

# -----------------------------
# Define the custom node tree
# -----------------------------
class ExploratoryNodesTree(bpy.types.NodeTree):
    """A custom node tree for the Exploratory Node Editor"""
    bl_idname = 'ExploratoryNodesTreeType'
    bl_label = 'Exploratory Node Editor Tree'
    bl_icon = 'NODETREE'

# -----------------------------
# Minimal example nodes for triggers and reactions.
# -----------------------------
class TriggerNode(bpy.types.Node):
    bl_idname = 'TriggerNodeType'
    bl_label = 'Trigger Node'
    bl_icon = 'QUESTION'
    
    def init(self, context):
        self.inputs.new('NodeSocketFloat', "Input")
        self.outputs.new('NodeSocketFloat', "Output")
    
    def draw_label(self):
        return self.bl_label

class ReactionNode(bpy.types.Node):
    bl_idname = 'ReactionNodeType'
    bl_label = 'Reaction Node'
    bl_icon = 'MODIFIER'
    
    def init(self, context):
        self.inputs.new('NodeSocketFloat', "Input")
        self.outputs.new('NodeSocketFloat', "Output")
    
    def draw_label(self):
        return self.bl_label

# -----------------------------
# Import our new Interaction Node.
# -----------------------------
from .interaction_nodes import InteractionNode

# -----------------------------
# Define a custom socket (optional)
# -----------------------------
class ExploratorySocket(bpy.types.NodeSocket):
    bl_idname = 'ExploratorySocketType'
    bl_label = 'Exploratory Socket'
    
    def draw(self, context, layout, node, text):
        layout.label(text=text)
    
    def draw_color(self, context, node):
        return (0.5, 0.5, 1.0, 1.0)

# -----------------------------
# Define Node Categories for the Shift+A menu
# -----------------------------
node_categories = [
    NodeCategory("TRIGGERS", "Triggers", items=[
        NodeItem("TriggerNodeType"),
        # Additional trigger nodes here.
    ]),
    NodeCategory("REACTIONS", "Reactions", items=[
        NodeItem("ReactionNodeType"),
        # Additional reaction nodes here.
    ]),
    NodeCategory("INTERACTIONS", "Interactions", items=[
        NodeItem("InteractionNodeType"),
        # Additional interaction nodes here.
    ]),
]

# -----------------------------
# Operator and Panel for the Node Editor sidebar.
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
    TriggerNode,
    ReactionNode,
    InteractionNode,
    ExploratorySocket,
    NODE_OT_create_exploratory_node_tree,
    NODE_PT_exploratory_panel,
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
