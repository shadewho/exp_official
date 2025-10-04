# File: Exp_Nodes/trigger_nodes.py
import bpy
from bpy.types import Node
from .base_nodes import TriggerNodeBase
from .reaction_nodes import ReactionOutputSocket  # ensure socket classes loaded
from .objective_nodes import enum_objective_items

# Interactions live in the Scene collections you already have:
from ..Exp_Game.interactions.exp_interaction_definition import InteractionDefinition


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
    scn.custom_interactions.remove(index)
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


# ──────────────────────────────────────────────────────────────────────────────
# Base Trigger Node (creates/owns a real Interaction by index)
# ──────────────────────────────────────────────────────────────────────────────

class _TriggerNodeKind(TriggerNodeBase):
    """
    Every Trigger node owns exactly one real InteractionDefinition (by index).
    Deleting the node deletes that interaction. The node UI edits the real object.
    """
    bl_icon = 'HAND'
    KIND = None  # "PROXIMITY" | "COLLISION" | "INTERACT" | "OBJECTIVE_UPDATE" | "TIMER_COMPLETE"

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

    def init(self, context):
        self.outputs.new('TriggerOutputSocketType', "Trigger Output")
        self.width = 300
        self.interaction_index = _ensure_interaction(self.KIND or "PROXIMITY")

    def copy(self, node):
        """
        Duplicating a trigger: make a totally fresh Interaction (no link copying),
        as requested.
        """
        self.interaction_index = _ensure_interaction(self.KIND or "PROXIMITY")
        self.width = getattr(node, "width", 300)

    def free(self):
        """
        Deleting the node deletes the owned Interaction.
        Reactions themselves are NOT deleted; only links are lost by virtue of the inter removal.
        """
        if self.interaction_index >= 0:
            _delete_interaction_at(self.interaction_index)
        self.interaction_index = -1

    def update(self):
        """
        Any graph change: assert interaction exists, correct trigger_type,
        then rewrite its linked reactions to mirror the current downstream chain.
        """
        scn = _scene()
        if not scn:
            return

        # Ensure we have a real interaction
        if not (0 <= self.interaction_index < len(scn.custom_interactions)):
            self.interaction_index = _ensure_interaction(self.KIND or "PROXIMITY")
            if self.interaction_index < 0:
                return

        inter = scn.custom_interactions[self.interaction_index]
        # Keep its trigger_type in sync with the node kind
        kind = self.KIND or "PROXIMITY"
        try:
            if getattr(inter, "trigger_type", "") != kind:
                inter.trigger_type = kind
        except Exception:
            pass

        # Rewrite the linked reaction indices from the node graph
        _sync_interaction_links_from_graph(self, self.interaction_index)

    # ── Execution: traversal for debug/manual evaluation path (optional) ──
    def execute_trigger(self, context):
        # Left as a no-op; runtime firing is handled in exp_interactions.py
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
