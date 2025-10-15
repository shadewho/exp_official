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
        return (0.15, 0.55, 1.0, 1.0)


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
        add("Mobility",          "ReactionMobilityNodeType")
        add("Mesh Visibility",   "ReactionMeshVisibilityNodeType")
        add("Reset Game",        "ReactionResetGameNodeType")

class NODE_MT_exploratory_add_objectives(Menu):
    bl_idname = "NODE_MT_exploratory_add_objectives"
    bl_label  = "Objectives and Timers"

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


class NODE_OT_delete_exploratory_node_tree(bpy.types.Operator):
    """Delete the selected Exploratory Node Tree and fully remove its Interactions, Reactions, and Objectives"""
    bl_idname = "node.delete_exploratory_node_tree"
    bl_label  = "Delete Exploratory Node Tree"

    tree_name: StringProperty()

    # ───────────────────────── helpers (scoped; no cross-file deps) ─────────────────────────

    @staticmethod
    def _scene() -> bpy.types.Scene | None:
        scn = getattr(bpy.context, "scene", None)
        if scn:
            return scn
        return bpy.data.scenes[0] if bpy.data.scenes else None

    # ---------- INTERACTIONS ----------
    @classmethod
    def _reindex_trigger_nodes_after_inter_remove(cls, removed_index: int) -> None:
        for ng in bpy.data.node_groups:
            if getattr(ng, "bl_idname", "") != EXPL_TREE_ID:
                continue
            for node in ng.nodes:
                if not hasattr(node, "interaction_index"):
                    continue
                idx = getattr(node, "interaction_index", -1)
                if idx < 0:
                    continue
                if idx == removed_index:
                    try:
                        node.interaction_index = -1
                    except Exception:
                        pass
                elif idx > removed_index:
                    try:
                        node.interaction_index = idx - 1
                    except Exception:
                        pass

    @classmethod
    def _delete_interaction_at(cls, index: int) -> None:
        scn = cls._scene()
        if not scn or not (0 <= index < len(getattr(scn, "custom_interactions", []))):
            return
        scn.custom_interactions.remove(index)
        cls._reindex_trigger_nodes_after_inter_remove(index)
        try:
            scn.custom_interactions_index = max(0, min(index, len(scn.custom_interactions) - 1))
        except Exception:
            pass

    # ---------- REACTIONS ----------
    @classmethod
    def _fix_interaction_reaction_indices_after_remove(cls, removed_index: int) -> None:
        scn = cls._scene()
        if not scn or not hasattr(scn, "custom_interactions"):
            return
        for inter in scn.custom_interactions:
            links = getattr(inter, "reaction_links", None)
            if not links:
                continue
            to_remove = [i for i, l in enumerate(links) if getattr(l, "reaction_index", -1) == removed_index]
            for i in reversed(to_remove):
                try:
                    links.remove(i)
                except Exception:
                    pass
            for l in links:
                ridx = getattr(l, "reaction_index", -1)
                if ridx > removed_index:
                    try:
                        l.reaction_index = ridx - 1
                    except Exception:
                        pass

    @classmethod
    def _reindex_reaction_nodes_after_remove(cls, removed_index: int) -> None:
        for ng in bpy.data.node_groups:
            if getattr(ng, "bl_idname", "") != EXPL_TREE_ID:
                continue
            for node in ng.nodes:
                if not hasattr(node, "reaction_index"):
                    continue
                idx = getattr(node, "reaction_index", -1)
                if idx < 0:
                    continue
                if idx == removed_index:
                    try:
                        node.reaction_index = -1
                    except Exception:
                        pass
                elif idx > removed_index:
                    try:
                        node.reaction_index = idx - 1
                    except Exception:
                        pass

    @classmethod
    def _delete_reaction_at(cls, index: int) -> None:
        scn = cls._scene()
        if not scn or not (0 <= index < len(getattr(scn, "reactions", []))):
            return
        scn.reactions.remove(index)
        cls._fix_interaction_reaction_indices_after_remove(index)
        cls._reindex_reaction_nodes_after_remove(index)
        try:
            scn.reactions_index = max(0, min(index, len(scn.reactions) - 1))
        except Exception:
            pass

    # ---------- OBJECTIVES ----------
    @classmethod
    def _reindex_objective_nodes_after_remove(cls, removed_index: int) -> None:
        """Repair ObjectiveNode.objective_index across all trees after a scn.objectives removal."""
        for ng in bpy.data.node_groups:
            if getattr(ng, "bl_idname", "") != EXPL_TREE_ID:
                continue
            for node in ng.nodes:
                if getattr(node, "bl_idname", "") != "ObjectiveNodeType":
                    continue
                idx = getattr(node, "objective_index", -1)
                if idx < 0:
                    continue
                if idx == removed_index:
                    try:
                        node.objective_index = -1
                    except Exception:
                        pass
                elif idx > removed_index:
                    try:
                        node.objective_index = idx - 1
                    except Exception:
                        pass

    @classmethod
    def _fix_objective_indices_in_interactions_after_remove(cls, removed_index: int) -> None:
        """Fix InteractionDefinition fields that store objective indices."""
        scn = cls._scene()
        if not scn or not hasattr(scn, "custom_interactions"):
            return
        for inter in scn.custom_interactions:
            t = getattr(inter, "trigger_type", "")
            if t == "OBJECTIVE_UPDATE":
                idx = getattr(inter, "objective_index", -1)
                if idx == removed_index:
                    try:
                        inter.objective_index = -1
                    except Exception:
                        pass
                elif idx > removed_index:
                    try:
                        inter.objective_index = idx - 1
                    except Exception:
                        pass
            elif t == "TIMER_COMPLETE":
                idx = getattr(inter, "timer_objective_index", -1)
                if idx == removed_index:
                    try:
                        inter.timer_objective_index = -1
                    except Exception:
                        pass
                elif idx > removed_index:
                    try:
                        inter.timer_objective_index = idx - 1
                    except Exception:
                        pass

    @classmethod
    def _fix_objective_indices_in_reactions_after_remove(cls, removed_index: int) -> None:
        """Fix ReactionDefinition fields that store objective indices."""
        scn = cls._scene()
        if not scn or not hasattr(scn, "reactions"):
            return
        for r in scn.reactions:
            rtype = getattr(r, "reaction_type", "")
            # OBJECTIVE_COUNTER / OBJECTIVE_TIMER store objective_index
            if rtype in {"OBJECTIVE_COUNTER", "OBJECTIVE_TIMER"}:
                idx = getattr(r, "objective_index", -1)
                if idx == removed_index:
                    try:
                        r.objective_index = -1
                    except Exception:
                        pass
                elif idx > removed_index:
                    try:
                        r.objective_index = idx - 1
                    except Exception:
                        pass
            # CUSTOM_UI_TEXT (OBJECTIVE subtype) uses text_objective_index
            if rtype == "CUSTOM_UI_TEXT":
                idx2 = getattr(r, "text_objective_index", -1)
                if idx2 == removed_index:
                    try:
                        r.text_objective_index = -1
                    except Exception:
                        pass
                elif idx2 > removed_index:
                    try:
                        r.text_objective_index = idx2 - 1
                    except Exception:
                        pass

    @classmethod
    def _delete_objective_at(cls, index: int) -> None:
        scn = cls._scene()
        if not scn or not (0 <= index < len(getattr(scn, "objectives", []))):
            return
        # Remove objective item
        scn.objectives.remove(index)
        # Repair all holders of objective indices
        cls._reindex_objective_nodes_after_remove(index)
        cls._fix_objective_indices_in_interactions_after_remove(index)
        cls._fix_objective_indices_in_reactions_after_remove(index)
        # Clamp active index
        try:
            scn.objectives_index = max(0, min(index, len(scn.objectives) - 1))
        except Exception:
            pass

    # ───────────────────────── operator flow ─────────────────────────

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        # 0) Resolve the node tree
        nt = bpy.data.node_groups.get(self.tree_name)
        if not nt or getattr(nt, "bl_idname", "") != EXPL_TREE_ID:
            self.report({'WARNING'}, "Node tree not found")
            return {'CANCELLED'}

        # 1) Collect all Interactions, Reactions, and Objectives referenced by nodes in THIS tree
        inter_indices = set()
        react_indices = set()
        objv_indices  = set()

        for node in nt.nodes:
            if hasattr(node, "interaction_index"):
                i_idx = getattr(node, "interaction_index", -1)
                if i_idx >= 0:
                    inter_indices.add(i_idx)
            if hasattr(node, "reaction_index"):
                r_idx = getattr(node, "reaction_index", -1)
                if r_idx >= 0:
                    react_indices.add(r_idx)
            # Objective nodes
            if getattr(node, "bl_idname", "") == "ObjectiveNodeType":
                o_idx = getattr(node, "objective_index", -1)
                if o_idx >= 0:
                    objv_indices.add(o_idx)

        # 2) Delete the node tree itself (nodes go away)
        try:
            bpy.data.node_groups.remove(nt)
        except Exception:
            self.report({'ERROR'}, "Failed to delete node tree.")
            return {'CANCELLED'}

        # 3) Fully delete all referenced Reactions (DESC order avoids index drift)
        for r_idx in sorted(react_indices, reverse=True):
            self._delete_reaction_at(r_idx)

        # 4) Fully delete all referenced Interactions (DESC order avoids index drift)
        for i_idx in sorted(inter_indices, reverse=True):
            self._delete_interaction_at(i_idx)

        # 5) Fully delete all referenced Objectives (DESC order avoids index drift)
        for o_idx in sorted(objv_indices, reverse=True):
            self._delete_objective_at(o_idx)

        self.report(
            {'INFO'},
            f"Deleted node tree and removed {len(inter_indices)} interaction(s), "
            f"{len(react_indices)} reaction(s), {len(objv_indices)} objective(s)."
        )
        return {'FINISHED'}

class NODE_OT_rename_exploratory_node_tree(Operator):
    bl_idname = "node.rename_exploratory_node_tree"
    bl_label  = "Rename Exploratory Node Tree"

    tree_name: StringProperty()
    new_name:  StringProperty(name="New Name")

    def invoke(self, context, event):
        nt = bpy.data.node_groups.get(self.tree_name)
        if not nt or getattr(nt, "bl_idname", "") != EXPL_TREE_ID:
            self.report({'WARNING'}, "Node tree not found")
            return {'CANCELLED'}
        self.new_name = nt.name
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "new_name", text="")

    def execute(self, context):
        nt = bpy.data.node_groups.get(self.tree_name)
        if not nt or getattr(nt, "bl_idname", "") != EXPL_TREE_ID:
            self.report({'WARNING'}, "Node tree not found")
            return {'CANCELLED'}
        name = (self.new_name or "").strip()
        if not name:
            self.report({'WARNING'}, "Name cannot be empty")
            return {'CANCELLED'}
        nt.name = name
        self.report({'INFO'}, "Node tree renamed")
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
                icon = 'NODE_SOCKET_OBJECT' if nt == active_tree else 'NONE'
                op = row.operator("node.select_exploratory_node_tree", text=nt.name, icon=icon)
                op.tree_name = nt.name

                # ← rename button (right next to trash)
                row.operator("node.rename_exploratory_node_tree", text="", icon='GREASEPENCIL').tree_name = nt.name

                # delete button
                row.operator("node.delete_exploratory_node_tree", text="", icon='TRASH').tree_name = nt.name
