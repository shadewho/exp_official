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
    NodeCategory("OBJECTIVES", "Objectives", items=[
        NodeItem("ObjectiveNodeType"),
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
        # Give the new node tree a fake user to prevent it from being removed automatically.
        new_tree.use_fake_user = True
        
        for area in context.screen.areas:
            if area.type == 'NODE_EDITOR':
                space = area.spaces.active
                space.tree_type = 'ExploratoryNodesTreeType'
                space.node_tree = new_tree
                break
        self.report({'INFO'}, "Exploratory Node Tree created (fake user enabled).")
        return {'FINISHED'}

# ----------------------------------------------------------
# Operator: Delete an Exploratory Node Tree with Confirmation
# ----------------------------------------------------------
class NODE_OT_delete_exploratory_node_tree(bpy.types.Operator):
    """Delete the selected Exploratory Node Tree (with confirmation)"""
    bl_idname = "node.delete_exploratory_node_tree"
    bl_label = "Delete Exploratory Node Tree"

    tree_name: bpy.props.StringProperty()

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        node_tree = bpy.data.node_groups.get(self.tree_name)
        if node_tree:
            bpy.data.node_groups.remove(node_tree)
            self.report({'INFO'}, f"Deleted node tree: {self.tree_name}")
        else:
            self.report({'WARNING'}, "Node tree not found")
        return {'FINISHED'}

# ----------------------------------------------------------
# Panel: Exploratory Node Editor with Node Tree List
# ----------------------------------------------------------
class NODE_PT_exploratory_panel(bpy.types.Panel):
    bl_label = "Exploratory Node Editor"
    bl_idname = "NODE_PT_exploratory_panel"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    
    @classmethod
    def poll(cls, context):
        # Show this panel only when the active node tree is of our custom type.
        return context.space_data.tree_type == 'ExploratoryNodesTreeType'
    
    def draw(self, context):
        layout = self.layout

        # --- Existing UI: Create Node Tree & instructions ---
        layout.operator("node.create_exploratory_node_tree", icon='NODETREE')
        layout.separator()
        layout.label(text="Use Shift+A to add nodes by category.")
        
        # --- New Section: List all Node Trees ---
        layout.separator()
        layout.label(text="Existing Node Trees:")
        col = layout.column(align=True)
        for nt in bpy.data.node_groups:
            if nt.bl_idname == "ExploratoryNodesTreeType":
                # Wrap each entry in a box so that it has a colored background.
                box = col.box()
                row = box.row(align=True)
                row.label(text=nt.name)
                # Delete button: when pressed, a confirmation will pop up.
                del_op = row.operator("node.delete_exploratory_node_tree", text="", icon='TRASH')
                del_op.tree_name = nt.name
