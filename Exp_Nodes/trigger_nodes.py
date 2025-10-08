# File: Exp_Nodes/trigger_nodes.py
import bpy
from bpy.types import Node
from .base_nodes import TriggerNodeBase

def enum_objective_items(self, context):
    scn = getattr(context, "scene", None) or getattr(bpy.context, "scene", None)
    items = []
    if scn and hasattr(scn, "objectives"):
        for i, obj in enumerate(scn.objectives):
            # identifier must be a string for EnumProperty
            items.append((str(i), obj.name, f"Objective: {obj.name}"))
    if not items:
        items.append(("0", "No Objectives", ""))
    return items
# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _scene() -> bpy.types.Scene | None:
    scn = getattr(bpy.context, "scene", None)
    if scn:
        return scn
    return bpy.data.scenes[0] if bpy.data.scenes else None


def _ensure_interaction(kind: str) -> int:
    """
    Create a new InteractionDefinition in scn.custom_interactions and return its index.
    Sets trigger_type to kind, initializes a name.
    """
    scn = _scene()
    if not scn or not hasattr(scn, "custom_interactions"):
        return -1
    inter = scn.custom_interactions.add()
    inter.name = f"Interaction_{len(scn.custom_interactions)}"
    try:
        inter.trigger_type = kind
    except Exception:
        pass
    scn.custom_interactions_index = len(scn.custom_interactions) - 1
    return scn.custom_interactions_index


def _delete_interaction_at(index: int) -> None:
    scn = _scene()
    if not scn or not (0 <= index < len(scn.custom_interactions)):
        return

    # Remove the item at 'index' — this compacts the collection
    scn.custom_interactions.remove(index)

    # Repair all surviving trigger nodes' indices so they still point to the same logical items
    _reindex_trigger_nodes_after_inter_remove(index)

    # Clamp active index for the N-panel UX
    scn.custom_interactions_index = max(0, min(index, len(scn.custom_interactions) - 1))



def _iter_downstream_reaction_nodes_from(trigger_node: Node):
    """
    Breadth-first traversal from the trigger's 'Trigger Output' socket, yielding
    reaction nodes in a stable link-order (good enough for ordered chains).
    """
    out_sock = trigger_node.outputs.get("Trigger Output")
    if not out_sock:
        return
    visited = set()
    queue = [lk.to_node for lk in out_sock.links if lk.to_node]
    while queue:
        node = queue.pop(0)
        if node is None or node in visited:
            continue
        visited.add(node)

        blid = getattr(node, "bl_idname", "")
        if blid.startswith("Reaction") and hasattr(node, "reaction_index"):
            yield node

            # continue chain through this reaction node's "Output"
            out2 = node.outputs.get("Reaction Output") or node.outputs.get("Output")
            if out2:
                for lk in out2.links:
                    if lk.to_node:
                        queue.append(lk.to_node)
        else:
            # passthrough (e.g. reroute) — traverse all outputs
            for sock in node.outputs:
                for lk in sock.links:
                    if lk.to_node:
                        queue.append(lk.to_node)


def _sync_interaction_links_from_graph(trigger_node: Node, inter_index: int) -> None:
    """
    Read the connected reaction-node chain and write InteractionDefinition.reaction_links
    as an ordered list of indices into scn.reactions. Removing a node or unlinking re-writes
    the list accordingly.
    """
    scn = _scene()
    if not scn or not (0 <= inter_index < len(scn.custom_interactions)):
        return
    inter = scn.custom_interactions[inter_index]

    ordered_indices = []
    for rnode in _iter_downstream_reaction_nodes_from(trigger_node):
        idx = getattr(rnode, "reaction_index", -1)
        if 0 <= idx < len(scn.reactions) and idx not in ordered_indices:
            ordered_indices.append(idx)

    # Rebuild links to match ordered_indices exactly
    links = inter.reaction_links
    try:
        links.clear()
    except AttributeError:
        while len(links):
            links.remove(len(links) - 1)

    for ridx in ordered_indices:
        link = links.add()
        link.reaction_index = ridx
    inter.reaction_links_index = max(0, min(getattr(inter, "reaction_links_index", 0), len(links) - 1))

def _reindex_trigger_nodes_after_inter_remove(removed_index: int) -> None:
    """
    After removing scene.custom_interactions[removed_index], all subsequent items shift -1.
    This walks every Exploratory node tree and fixes each Trigger node's
    node-local interaction_index so it still points to the same logical Interaction.
    """
    for ng in bpy.data.node_groups:
        if getattr(ng, "bl_idname", "") != "ExploratoryNodesTreeType":
            continue
        for node in ng.nodes:
            # Detect our trigger nodes by presence of the index field (safe & future-proof)
            if not hasattr(node, "interaction_index"):
                continue

            idx = getattr(node, "interaction_index", -1)
            if idx < 0:
                continue

            if idx == removed_index:
                # This node is being freed (or points at the removed slot).
                # Mark invalid to avoid mis-pointing if Blender keeps it around during batch deletes.
                try:
                    node.interaction_index = -1
                except Exception:
                    pass
            elif idx > removed_index:
                # Shift down by one to track the same logical item.
                try:
                    node.interaction_index = idx - 1
                except Exception:
                    pass


# ──────────────────────────────────────────────────────────────────────────────
# Base Trigger Node (creates/owns a real Interaction by index)
# ──────────────────────────────────────────────────────────────────────────────

class _TriggerNodeKind(TriggerNodeBase):
    """
    Every Trigger node owns exactly one real InteractionDefinition (by index).
    Deleting the node deletes that interaction. The node UI edits the real object.
    """
    KIND = None  # "PROXIMITY" | "COLLISION" | "INTERACT" | "OBJECTIVE_UPDATE" | "TIMER_COMPLETE"
    # Subtle mid-red body tint (header is theme-controlled)

    interaction_index: bpy.props.IntProperty(name="Interaction Index", default=-1, min=-1)

    # UI convenience mirrors for the two enum pickers that are expensive to reach by index each draw
    node_objective_index: bpy.props.EnumProperty(
        name="Objective",
        description="Select which objective this trigger watches",
        items=enum_objective_items,
        update=lambda self, ctx: self._on_objective_changed(ctx)
    )
    node_timer_objective_index: bpy.props.EnumProperty(
        name="Timer Objective",
        description="Select which timer objective to watch for completion",
        items=enum_objective_items,
        update=lambda self, ctx: self._on_timer_objective_changed(ctx)
    )

    _EXPL_TINT_TRIGGER = (0.24, 0.18, 0.18)
    def _tint(self):
        try:
            self.use_custom_color = True
            self.color = self._EXPL_TINT_TRIGGER
        except Exception:
            pass

    def init(self, context):
        # sockets + sizing
        self.outputs.new('TriggerOutputSocketType', "Trigger Output")
        self.width = 300

        self._tint()
        self.interaction_index = _ensure_interaction(self.KIND or "PROXIMITY")


    def copy(self, node):
        """
        Duplicate this trigger node by delegating to the canonical operator:
        bpy.ops.exploratory.duplicate_interaction(index=src_idx)

        Guarantees:
        • A brand-new InteractionDefinition is created by the operator.
        • All trigger data is copied by the operator.
        • Reaction links are cleared by the operator (as implemented there).
        • This node is re-bound to the NEW interaction index.
        • Any duplicated graph wires from this node are removed so the new
            interaction stays link-empty until the user reconnects nodes.
        """
        scn = _scene()
        self.width = getattr(node, "width", 300)

        if not scn or not hasattr(scn, "custom_interactions"):
            self.interaction_index = -1
            return

        # Source interaction index taken from the ORIGINAL node being duplicated
        src_idx = getattr(node, "interaction_index", -1)
        if not (0 <= src_idx < len(scn.custom_interactions)):
            # No valid source; just leave this node unbound (nothing else to do)
            self.interaction_index = -1
            return

        # 1) Call the canonical duplicate operator (EXEC_DEFAULT to avoid popups)
        try:
            res = bpy.ops.exploratory.duplicate_interaction('EXEC_DEFAULT', index=src_idx)
            if 'CANCELLED' in res:
                # Operator refused; keep unbound so we don't alias the source
                self.interaction_index = -1
                return
        except Exception:
            # If the operator errored, don't fall back to custom copy logic
            self.interaction_index = -1
            return

        # 2) The operator appends a new Interaction at the end and clears links.
        new_idx = len(scn.custom_interactions) - 1
        self.interaction_index = new_idx

        # 3) Make sure the trigger_type matches THIS node’s kind (belt & suspenders)
        try:
            scn.custom_interactions[new_idx].trigger_type = (self.KIND or "PROXIMITY")
        except Exception:
            pass

        # 4) Remove any auto-duplicated outgoing wires from this duplicated node.
        #    (Prevents update() from re-synchronizing links from copied wires.)
        try:
            ntree = getattr(self, "id_data", None)
            out_sock = self.outputs.get("Trigger Output")
            if ntree and out_sock and hasattr(ntree, "links"):
                for lk in list(out_sock.links):
                    try:
                        ntree.links.remove(lk)
                    except Exception:
                        pass
        except Exception:
            pass


    def free(self):
        """
        Deleting the node deletes the owned Interaction.
        Other trigger nodes must not be affected.
        """
        idx = getattr(self, "interaction_index", -1)
        if idx >= 0:
            _delete_interaction_at(idx)
        # Mark this node as unbound so any late draw/update won't touch stale indices
        self.interaction_index = -1


    def update(self):
        """
        IMPORTANT CHANGE:
        Do NOT create new interactions during update() anymore.
        This prevents the 'first add creates 2 interactions' bug that happens
        when Blender triggers an early update during node construction.
        We only:
          • validate the existing index,
          • keep trigger_type in sync,
          • and resync linked reactions from downstream nodes.
        If the index is invalid (e.g., user deleted it from the N-panel),
        we simply bail out and do nothing; no auto-recreation here.
        """
        scn = _scene()
        if not scn:
            return

        # If our owned interaction no longer exists, do NOT recreate here.
        if not (0 <= self.interaction_index < len(getattr(scn, "custom_interactions", []))):
            return

        inter = scn.custom_interactions[self.interaction_index]

        # Keep its trigger_type pinned to the node kind
        kind = self.KIND or "PROXIMITY"
        try:
            if getattr(inter, "trigger_type", "") != kind:
                inter.trigger_type = kind
        except Exception:
            pass

        # Rewrite linked reactions to reflect the visible node graph
        _sync_interaction_links_from_graph(self, self.interaction_index)

    # ── Execution: traversal for debug/manual evaluation path (optional) ──
    def execute_trigger(self, context):
        pass

    # ── Two reactive pickers that mirror into the real Interaction object ──
    def _on_objective_changed(self, context):
        scn = _scene()
        if not scn or not (0 <= self.interaction_index < len(scn.custom_interactions)):
            return
        inter = scn.custom_interactions[self.interaction_index]
        if getattr(inter, "trigger_type", "") == "OBJECTIVE_UPDATE":
            inter.objective_index = self.node_objective_index

    def _on_timer_objective_changed(self, context):
        scn = _scene()
        if not scn or not (0 <= self.interaction_index < len(scn.custom_interactions)):
            return
        inter = scn.custom_interactions[self.interaction_index]
        if getattr(inter, "trigger_type", "") == "TIMER_COMPLETE":
            inter.timer_objective_index = self.node_timer_objective_index

    # ── Per-kind drawers (edit the backed Interaction by index) ──
    def _draw_proximity(self, layout, scn, inter):
        layout.prop(inter, "use_character", text="Use Character as A")
        if inter.use_character:
            char = getattr(scn, "target_armature", None)
            layout.label(text=f"Object A: {char.name if char else '—'}", icon='ARMATURE_DATA')
        else:
            layout.prop(inter, "proximity_object_a", text="Object A")
        layout.prop(inter, "proximity_object_b", text="Object B")
        layout.prop(inter, "proximity_distance", text="Distance")

    def _draw_collision(self, layout, scn, inter):
        layout.prop(inter, "use_character", text="Use Character as A")
        if inter.use_character:
            char = getattr(scn, "target_armature", None)
            layout.label(text=f"Object A: {char.name if char else '—'}", icon='ARMATURE_DATA')
        else:
            layout.prop(inter, "collision_object_a", text="Object A")
        layout.prop(inter, "collision_object_b", text="Object B")
        layout.prop(inter, "collision_margin", text="Margin")

    def _draw_interact(self, layout, _scn, inter):
        layout.prop(inter, "interact_object", text="Object")
        layout.prop(inter, "interact_distance", text="Distance")

    def _draw_objective_update(self, layout, _scn, inter):
        layout.prop(self, "node_objective_index", text="Objective")
        layout.prop(inter, "objective_condition", text="Condition")
        if getattr(inter, "objective_condition", "") in {"EQUALS", "AT_LEAST"}:
            layout.prop(inter, "objective_condition_value", text="Value")

    def _draw_timer_complete(self, layout, _scn, inter):
        layout.prop(self, "node_timer_objective_index", text="Timer Objective")

    # ── Node UI ──
    def draw_buttons(self, context, layout):
        scn = _scene()
        if not scn or not (0 <= self.interaction_index < len(scn.custom_interactions)):
            layout.label(text="(Creating Interaction…)", icon='TIME')
            return

        inter = scn.custom_interactions[self.interaction_index]
        nice = {
            "PROXIMITY": "Proximity",
            "COLLISION": "Collision",
            "INTERACT": "Interact Key",
            "OBJECTIVE_UPDATE": "Objective Update",
            "TIMER_COMPLETE": "Timer Complete",
        }.get(self.KIND or "PROXIMITY", self.KIND or "PROXIMITY")

        layout.label(text=nice, icon='HAND')

        if self.KIND == "PROXIMITY":
            self._draw_proximity(layout, scn, inter)
        elif self.KIND == "COLLISION":
            self._draw_collision(layout, scn, inter)
        elif self.KIND == "INTERACT":
            self._draw_interact(layout, scn, inter)
        elif self.KIND == "OBJECTIVE_UPDATE":
            self._draw_objective_update(layout, scn, inter)
        elif self.KIND == "TIMER_COMPLETE":
            self._draw_timer_complete(layout, scn, inter)

        layout.separator()
        layout.label(text="Options:")
        layout.prop(inter, "trigger_mode", text="Mode")
        if getattr(inter, "trigger_mode", "") == "COOLDOWN":
            layout.prop(inter, "trigger_cooldown", text="Cooldown")
        layout.prop(inter, "trigger_delay", text="Delay (sec)")


# ──────────────────────────────────────────────────────────────────────────────
# Concrete Trigger Nodes
# ──────────────────────────────────────────────────────────────────────────────

class ProximityTriggerNode(_TriggerNodeKind):
    bl_idname = 'ProximityTriggerNodeType'
    bl_label  = 'Proximity'
    KIND = "PROXIMITY"

class CollisionTriggerNode(_TriggerNodeKind):
    bl_idname = 'CollisionTriggerNodeType'
    bl_label  = 'Collision'
    KIND = "COLLISION"

class InteractTriggerNode(_TriggerNodeKind):
    bl_idname = 'InteractTriggerNodeType'
    bl_label  = 'Interact Key'
    KIND = "INTERACT"

class ObjectiveUpdateTriggerNode(_TriggerNodeKind):
    bl_idname = 'ObjectiveUpdateTriggerNodeType'
    bl_label  = 'Objective Update'
    KIND = "OBJECTIVE_UPDATE"

class TimerCompleteTriggerNode(_TriggerNodeKind):
    bl_idname = 'TimerCompleteTriggerNodeType'
    bl_label  = 'Timer Complete'
    KIND = "TIMER_COMPLETE"
