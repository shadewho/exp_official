# File: Exp_Nodes/node_editor.py
import bpy
from bpy.types import NodeTree, NodeSocket, Operator, Panel, Menu
from bpy.props import StringProperty

# ─────────────────────────────────────────────────────────
# Gate: only operate when the active editor is our tree type
# ─────────────────────────────────────────────────────────
EXPL_TREE_ID = "ExploratoryNodesTreeType"

def _in_exploratory_editor(context) -> bool:
    sd = getattr(context, "space_data", None)
    return bool(sd) and getattr(sd, "tree_type", "") == EXPL_TREE_ID


# ─────────────────────────────────────────────────────────
# Custom Node Tree & Sockets
# ─────────────────────────────────────────────────────────
class ExploratoryNodesTree(NodeTree):
    """Exploratory Node Editor"""
    bl_idname = EXPL_TREE_ID
    bl_label  = "Exploratory Nodes"
    bl_icon   = "NODE_SOCKET_OBJECT"

    def update(self):
        if not getattr(self, "use_fake_user", False):
            try:
                self.use_fake_user = True
            except Exception:
                pass

class TriggerOutputSocket(NodeSocket):
    bl_idname = "TriggerOutputSocketType"
    bl_label  = "Trigger Output Socket"

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (0.8, 0.3, 0.8, 1.0)


# ─────────────────────────────────────────────────────────
# Top-level Add menu entries (no parent, no icons)
# ─────────────────────────────────────────────────────────
class NODE_MT_exploratory_add_triggers(Menu):
    bl_idname = "NODE_MT_exploratory_add_triggers"
    bl_label  = "Triggers"

    def draw(self, context):
        layout = self.layout

        def add(lbl, idname):
            op = layout.operator("node.add_node", text=lbl, icon='NONE')
            op.type = idname
            op.use_transform = True

        add("Proximity",        "ProximityTriggerNodeType")
        add("Collision",        "CollisionTriggerNodeType")
        add("Interact Key",     "InteractTriggerNodeType")
        add("Objective Update", "ObjectiveUpdateTriggerNodeType")
        add("Timer Complete",   "TimerCompleteTriggerNodeType")


class NODE_MT_exploratory_add_reactions(Menu):
    bl_idname = "NODE_MT_exploratory_add_reactions"
    bl_label  = "Reactions"

    def draw(self, context):
        layout = self.layout
        col = layout.column_flow(columns=1, align=True)

        def add(lbl, idname):
            op = col.operator("node.add_node", text=lbl, icon='NONE')
            op.type = idname
            op.use_transform = True

        add("Custom Action",     "ReactionCustomActionNodeType")
        add("Character Action",  "ReactionCharActionNodeType")
        add("Play Sound",        "ReactionSoundNodeType")
        add("Property Value",    "ReactionPropertyNodeType")
        add("Transform",         "ReactionTransformNodeType")
        add("Custom UI Text",    "ReactionCustomTextNodeType")
        add("Objective Counter", "ReactionObjectiveCounterNodeType")
        add("Objective Timer",   "ReactionObjectiveTimerNodeType")
        add("Mobility & Game",   "ReactionMobilityGameNodeType")


class NODE_MT_exploratory_add_objectives(Menu):
    bl_idname = "NODE_MT_exploratory_add_objectives"
    bl_label  = "Objectives"

    def draw(self, context):
        layout = self.layout
        op = layout.operator("node.add_node", text="Objective", icon='NONE')
        op.type = "ObjectiveNodeType"
        op.use_transform = True


# Append hook the init module will attach to NODE_MT_add:
# it inserts our three categories directly into Shift+A (no icons).
def _append_exploratory_entry(self, context):
    if _in_exploratory_editor(context):
        self.layout.menu("NODE_MT_exploratory_add_triggers",   text="Triggers")
        self.layout.menu("NODE_MT_exploratory_add_reactions",  text="Reactions")
        self.layout.menu("NODE_MT_exploratory_add_objectives", text="Objectives")


# ─────────────────────────────────────────────────────────
# Operators: create/select/delete node trees
# ─────────────────────────────────────────────────────────
class NODE_OT_select_exploratory_node_tree(Operator):
    bl_idname = "node.select_exploratory_node_tree"
    bl_label  = "Select Node Tree"

    tree_name: StringProperty()

    def execute(self, context):
        nt = bpy.data.node_groups.get(self.tree_name)
        if nt:
            context.space_data.node_tree = nt
            self.report({'INFO'}, f"Selected node tree: {self.tree_name}")
            return {'FINISHED'}
        self.report({'WARNING'}, "Node tree not found")
        return {'CANCELLED'}


class NODE_OT_create_exploratory_node_tree(Operator):
    bl_idname = "node.create_exploratory_node_tree"
    bl_label  = "Create Exploratory Node Tree"

    def execute(self, context):
        new_tree = bpy.data.node_groups.new("Exploratory Node Tree", EXPL_TREE_ID)
        try:
            new_tree.use_fake_user = True
        except Exception:
            pass
        if _in_exploratory_editor(context):
            context.space_data.node_tree = new_tree
        self.report({'INFO'}, "Created new Exploratory Node Tree")
        return {'FINISHED'}


class NODE_OT_delete_exploratory_node_tree(Operator):
    """Delete the selected Exploratory Node Tree (with confirmation)"""
    bl_idname = "node.delete_exploratory_node_tree"
    bl_label  = "Delete Exploratory Node Tree"

    tree_name: StringProperty()

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


# ─────────────────────────────────────────────────────────
# Sidebar Panel (Node Editor → N-panel → “Exploratory”)
# ─────────────────────────────────────────────────────────
class NODE_PT_exploratory_panel(Panel):
    bl_label = "Exploratory Node Editor"
    bl_idname = "NODE_PT_exploratory_panel"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    @classmethod
    def poll(cls, context):
        return _in_exploratory_editor(context)

    def draw(self, context):
        layout = self.layout
        active_tree = getattr(context.space_data, "node_tree", None)

        layout.operator("node.create_exploratory_node_tree", icon='NODETREE')
        layout.separator()
        layout.label(text="Existing Node Trees:")

        col = layout.column(align=True)
        for nt in bpy.data.node_groups:
            if getattr(nt, "bl_idname", "") == EXPL_TREE_ID:
                row = col.box().row(align=True)
                icon = 'CON_OBJECTSOLVER' if nt == active_tree else 'NONE'
                op = row.operator("node.select_exploratory_node_tree", text=nt.name, icon=icon)
                op.tree_name = nt.name
                row.operator("node.delete_exploratory_node_tree", text="", icon='TRASH').tree_name = nt.name
