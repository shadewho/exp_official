# File: Exp_Nodes/node_editor.py
import bpy
from bpy.types import NodeTree, NodeSocket
from nodeitems_utils import NodeCategory, NodeItem

# ── custom node tree ──
class ExploratoryNodesTree(bpy.types.NodeTree):
    """Exploratory Node Editor"""
    bl_idname = 'ExploratoryNodesTreeType'
    bl_label  = 'Exploratory Nodes'
    bl_icon   = 'EXPERIMENTAL'

# ── shared Trigger output socket (used by Trigger nodes) ──
class TriggerOutputSocket(bpy.types.NodeSocket):
    bl_idname = 'TriggerOutputSocketType'
    bl_label  = 'Trigger Output Socket'
    def draw(self, context, layout, node, text):
        layout.label(text=text)
    def draw_color(self, context, node):
        return (0.8, 0.3, 0.8, 1.0)

# ── Shift+A categories ──
node_categories = [
    NodeCategory("TRIGGERS", "Triggers", items=[
        NodeItem("ProximityTriggerNodeType"),
        NodeItem("CollisionTriggerNodeType"),
        NodeItem("InteractTriggerNodeType"),
        NodeItem("ObjectiveUpdateTriggerNodeType"),
        NodeItem("TimerCompleteTriggerNodeType"),
    ]),
    NodeCategory("REACTIONS", "Reactions", items=[
        NodeItem("ReactionCustomActionNodeType"),
        NodeItem("ReactionCharActionNodeType"),
        NodeItem("ReactionSoundNodeType"),
        NodeItem("ReactionPropertyNodeType"),
        NodeItem("ReactionTransformNodeType"),
        NodeItem("ReactionCustomTextNodeType"),
        NodeItem("ReactionObjectiveCounterNodeType"),
        NodeItem("ReactionObjectiveTimerNodeType"),
        NodeItem("ReactionMobilityGameNodeType"),
    ]),
    NodeCategory("OBJECTIVES", "Objectives", items=[
        NodeItem("ObjectiveNodeType"),
    ]),
]

# ── minimal panel operators ──
class NODE_OT_select_exploratory_node_tree(bpy.types.Operator):
    bl_idname = "node.select_exploratory_node_tree"
    bl_label  = "Select Node Tree"
    tree_name: bpy.props.StringProperty()
    def execute(self, context):
        nt = bpy.data.node_groups.get(self.tree_name)
        if nt:
            context.space_data.node_tree = nt
            self.report({'INFO'}, f"Selected node tree: {self.tree_name}")
            return {'FINISHED'}
        self.report({'WARNING'}, "Node tree not found")
        return {'CANCELLED'}

class NODE_OT_create_exploratory_node_tree(bpy.types.Operator):
    bl_idname = "node.create_exploratory_node_tree"
    bl_label  = "Create Exploratory Node Tree"
    def execute(self, context):
        new_tree = bpy.data.node_groups.new("Exploratory Node Tree", 'ExploratoryNodesTreeType')
        new_tree.use_fake_user = True
        context.space_data.node_tree = new_tree
        self.report({'INFO'}, "Created new Exploratory Node Tree")
        return {'FINISHED'}

class NODE_OT_delete_exploratory_node_tree(bpy.types.Operator):
    bl_idname = "node.delete_exploratory_node_tree"
    bl_label  = "Delete Exploratory Node Tree"
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

class NODE_PT_exploratory_panel(bpy.types.Panel):
    bl_label = "Exploratory Node Editor"
    bl_idname = "NODE_PT_exploratory_panel"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == 'ExploratoryNodesTreeType'
    def draw(self, context):
        layout = self.layout
        active_tree = context.space_data.node_tree
        layout.operator("node.create_exploratory_node_tree", icon='NODETREE')
        layout.separator()
        layout.label(text="Existing Node Trees:")
        col = layout.column(align=True)
        for nt in bpy.data.node_groups:
            if nt.bl_idname == "ExploratoryNodesTreeType":
                row = col.box().row(align=True)
                icon = 'CON_OBJECTSOLVER' if nt == active_tree else 'NONE'
                op = row.operator("node.select_exploratory_node_tree", text=nt.name, icon=icon)
                op.tree_name = nt.name
                row.operator("node.delete_exploratory_node_tree", text="", icon='TRASH').tree_name = nt.name
