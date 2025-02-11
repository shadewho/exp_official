import bpy
from bpy.types import Operator

class NODE_OT_evaluate_tree(Operator):
    """Evaluate the Exploratory Node Tree"""
    bl_idname = "node.evaluate_exploratory_tree"
    bl_label = "Evaluate Exploratory Node Tree"
    
    def execute(self, context):
        node_tree = context.space_data.node_tree
        if not node_tree:
            self.report({'WARNING'}, "No node tree found.")
            return {'CANCELLED'}
        
        # Traverse nodes and call execute methods if they exist.
        for node in node_tree.nodes:
            if hasattr(node, "execute_trigger"):
                node.execute_trigger(context)
            if hasattr(node, "execute_reaction"):
                node.execute_reaction(context)
        self.report({'INFO'}, "Node tree evaluation complete.")
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
# Panel: List all Exploratory Node Trees in the Node Editor's Sidebar
# ----------------------------------------------------------
class NODE_PT_exploratory_tree_list(bpy.types.Panel):
    bl_label = "Exploratory Node Trees"
    bl_idname = "NODE_PT_exploratory_tree_list"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.label(text="Existing Node Trees:")

        # Iterate over all node groups and list only those of our custom type.
        for nt in bpy.data.node_groups:
            if nt.bl_idname == "ExploratoryNodesTreeType":
                row = col.row(align=True)
                row.label(text=nt.name)
                # Delete button with confirmation: pass the tree name to the operator.
                del_op = row.operator("node.delete_exploratory_node_tree", text="", icon='TRASH')
                del_op.tree_name = nt.name

        layout.separator()
        # Button to add a new node tree (this operator already sets a fake user).
        layout.operator("node.create_exploratory_node_tree", text="Add New Node Tree", icon='ADD')