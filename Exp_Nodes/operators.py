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
