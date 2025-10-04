import bpy
from .base_nodes import TriggerNodeBase
from .objective_nodes import enum_objective_items
from ..Exp_Game.interactions.exp_interaction_definition import (
    new_interaction, InteractionDefinition
)

# ───────────────────────── helpers ─────────────────────────

def _safe_scene() -> bpy.types.Scene | None:
    scn = getattr(bpy.context, "scene", None)
    if scn:
        return scn
    if bpy.data.scenes:
        return bpy.data.scenes[0]
    return None

def _has_inter_collection(scene: bpy.types.Scene | None) -> bool:
    return bool(scene and hasattr(scene, "custom_interactions"))

def _find_inter_by_uid(scene: bpy.types.Scene | None, uid: str) -> InteractionDefinition | None:
    if not _has_inter_collection(scene) or not uid:
        return None
    for it in scene.custom_interactions:
        if getattr(it, "uid", "") == uid:
            return it
    return None

def _lookup_backend(node_self) -> InteractionDefinition | None:
    """Lookup only. Never creates/writes. Also flips backend_ready when found."""
    scn = _safe_scene()
    if not _has_inter_collection(scn):
        return None
    uid = getattr(node_self, "interaction_uid", "")
    inter = _find_inter_by_uid(scn, uid) if uid else None
    if inter:
        try:
            node_self.backend_ready = True
            node_self.ensure_pending = False
        except Exception:
            pass
    return inter

def _schedule_create_backend(node_self):
    """
    Debounced creation of a fresh Interaction (NO copy-from).
    Avoids enum-copy issues and multi-create storms.
    """
    if getattr(node_self, "ensure_pending", False):
        return
    node_self.ensure_pending = True
    kind = getattr(node_self, "KIND", "PROXIMITY")

    def _do():
        scn = _safe_scene()
        if not _has_inter_collection(scn):
            # Retry a bit later if scene props not registered yet.
            node_self.ensure_pending = False
            return 0.05

        # If something else already created it, stop.
        if _lookup_backend(node_self):
            node_self.ensure_pending = False
            node_self.backend_ready = True
            return None

        inter = new_interaction(scn, name="AutoInteraction")
        inter.trigger_type = kind
        try:
            node_self.interaction_uid = inter.uid
            node_self.backend_ready = True
        except Exception:
            pass

        node_self.ensure_pending = False
        return None

    bpy.app.timers.register(_do, first_interval=0.0)

def _schedule_delete_backend(node_self):
    """
    Delete the backend Interaction on a timer (legal write context).
    IMPORTANT: Before deletion, persist all downstream reaction nodes (copy their
    backend Reaction data into each node's local_reaction and detach).
    """
    uid = getattr(node_self, "interaction_uid", "")
    if not uid:
        return

    def _do():
        scn = _safe_scene()
        if not _has_inter_collection(scn):
            return None

        # 1) Persist downstream reactions (detach but keep node data)
        _persist_downstream_reactions(node_self, owning_inter_uid=uid)

        # 2) Remove this Interaction
        for i, it in enumerate(scn.custom_interactions):
            if getattr(it, "uid", "") == uid:
                scn.custom_interactions.remove(i)
                break

        # 3) Clear node markers
        try:
            node_self.interaction_uid = ""
            node_self.backend_ready = False
            node_self.ensure_pending = False
        except Exception:
            pass
        return None

    bpy.app.timers.register(_do, first_interval=0.0)

def _schedule_setattr_inter(inter_uid: str, attr: str, value):
    """Generic safe setter for InteractionDefinition attributes."""
    if not inter_uid:
        return
    def _do():
        scn = _safe_scene()
        if not _has_inter_collection(scn):
            return None
        inter = _find_inter_by_uid(scn, inter_uid)
        if not inter:
            return None
        try:
            setattr(inter, attr, value)
        except Exception:
            pass
        return None
    bpy.app.timers.register(_do, first_interval=0.0)

# ───────────────────────── reaction persistence utilities ─────────────────────────
# Reaction node idnames (match your reaction_nodes.py)
_REACTION_NODE_IDS = {
    "ReactionCustomActionNodeType",
    "ReactionCharActionNodeType",
    "ReactionSoundNodeType",
    "ReactionPropertyNodeType",
    "ReactionTransformNodeType",
    "ReactionCustomTextNodeType",
    "ReactionObjectiveCounterNodeType",
    "ReactionObjectiveTimerNodeType",
    "ReactionMobilityGameNodeType",
}

def _is_reaction_node(n: bpy.types.Node) -> bool:
    return getattr(n, "bl_idname", "") in _REACTION_NODE_IDS

def _copy_reaction_values(src, dst):
    """
    Best-effort copy of simple RNA fields from src → dst.
    Skips rna_type, collections, and read-only props.
    Compatible with Blender 4.x RNA API.
    """
    if not src or not dst:
        return
    for prop in src.bl_rna.properties:
        ident = prop.identifier
        if ident == "rna_type":
            continue
        if getattr(prop, "is_readonly", False) or getattr(prop, "type", None) == 'COLLECTION':
            continue
        try:
            val = getattr(src, ident)
        except Exception:
            continue
        try:
            setattr(dst, ident, val)
        except Exception:
            # Some pointer/enum mismatches can fail; ignore.
            pass

def _persist_downstream_reactions(trigger_node: bpy.types.Node, owning_inter_uid: str):
    """
    Walk the node graph from trigger_node's output through reaction chains.
    For every reaction node that currently points into owning_inter_uid, copy its
    backend Reaction item into node.local_reaction and clear linkage (reaction_uid/owning_inter_uid).
    """
    node_tree = trigger_node.id_data
    if not node_tree:
        return

    scn = _safe_scene()
    inter = _find_inter_by_uid(scn, owning_inter_uid) if scn else None
    if not inter or not hasattr(inter, "reactions"):
        return

    # Build adjacency from links
    to_visit = []
    visited = set()

    out_sock = trigger_node.outputs.get("Trigger Output")
    if out_sock:
        for lk in out_sock.links:
            if lk.to_node:
                to_visit.append(lk.to_node)

    while to_visit:
        n = to_visit.pop()
        if n in visited:
            continue
        visited.add(n)

        # If it's a reaction node, detach & persist if it belongs to this interaction
        if _is_reaction_node(n):
            react_uid = getattr(n, "reaction_uid", "")
            owning_uid = getattr(n, "owning_inter_uid", "")
            if react_uid and owning_uid == owning_inter_uid:
                # find backend reaction item by uid
                backend = None
                for itm in inter.reactions:
                    if getattr(itm, "uid", "") == react_uid:
                        backend = itm
                        break
                # copy fields to local_reaction if present
                local = getattr(n, "local_reaction", None)
                if backend and local:
                    _copy_reaction_values(backend, local)
                # clear linkage so node survives without an Interaction
                try:
                    n.reaction_uid = ""
                except Exception:
                    pass
                try:
                    n.owning_inter_uid = ""
                except Exception:
                    pass

            # Continue traversal through this reaction node's output(s) to reach chained reactions
            out2 = n.outputs.get("Output") or n.outputs.get("Reaction Output")
            if out2:
                for lk in out2.links:
                    if lk.to_node:
                        to_visit.append(lk.to_node)

        else:
            # Non-reaction nodes (e.g., rerouters) — still traverse their outputs
            for sock in n.outputs:
                for lk in sock.links:
                    if lk.to_node:
                        to_visit.append(lk.to_node)

# ───────────────────────── shared base ─────────────────────────

class _TriggerNodeKind(TriggerNodeBase):
    """
    Base class for concrete Trigger nodes.
    Each subclass sets KIND to one of:
      PROXIMITY | COLLISION | INTERACT | OBJECTIVE_UPDATE | TIMER_COMPLETE
    """
    bl_icon = 'HAND'
    KIND = None  # must be overridden

    # stable reference to our auto-created Interaction
    interaction_uid: bpy.props.StringProperty(name="Interaction UID", default="")

    # debounce flags
    ensure_pending: bpy.props.BoolProperty(name="Ensure Pending", default=False)
    backend_ready: bpy.props.BoolProperty(name="Backend Ready", default=False)

    # Only some kinds expose a node-side picker that mirrors the backend enum(s)
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

    # ── Node life cycle ────────────────────────────────────────────
    def init(self, context):
        self.outputs.new('TriggerOutputSocketType', "Trigger Output")
        self.width = 300
        self.backend_ready = False
        self.ensure_pending = False
        self._debounced_ensure_backend()

    def copy(self, node):
        """
        Duplicating a trigger → ALWAYS create a fresh, empty Interaction.
        (We do NOT copy fields; avoids enum mismatch and honors your request.)
        """
        self.interaction_uid = ""  # detached first; creation is deferred
        self.backend_ready = False
        self.ensure_pending = False
        _schedule_create_backend(self)

    def free(self):
        """
        Deleting the trigger:
          • persist all downstream reaction nodes (copy backend → local, detach)
          • delete only this Interaction (reactions inside it are going away,
            but each reaction node keeps its data locally for reuse)
        """
        _schedule_delete_backend(self)

    def _debounced_ensure_backend(self):
        """
        Ensure backend exists and has the correct type, but never write immediately.
        Debounced with ensure_pending/backend_ready flags.
        """
        inter = _lookup_backend(self)
        if inter:
            if getattr(inter, "trigger_type", None) != self.KIND:
                _schedule_setattr_inter(inter.uid, "trigger_type", self.KIND)
            return
        if self.ensure_pending or self.backend_ready:
            return
        _schedule_create_backend(self)

    def update(self):
        # keep it light; might be called often
        self._debounced_ensure_backend()

    # ── Execution ──────────────────────────────────────────────────
    def execute_trigger(self, context):
        if "Trigger Output" in self.outputs:
            for link in self.outputs["Trigger Output"].links:
                to_node = link.to_node
                if hasattr(to_node, "execute_reaction"):
                    to_node.execute_reaction(context)

    # ── UI helpers ─────────────────────────────────────────────────
    def _on_objective_changed(self, context):
        if self.KIND != "OBJECTIVE_UPDATE":
            return
        inter = _lookup_backend(self)
        if inter and inter.trigger_type == "OBJECTIVE_UPDATE":
            _schedule_setattr_inter(inter.uid, "objective_index", self.node_objective_index)

    def _on_timer_objective_changed(self, context):
        if self.KIND != "TIMER_COMPLETE":
            return
        inter = _lookup_backend(self)
        if inter and inter.trigger_type == "TIMER_COMPLETE":
            _schedule_setattr_inter(inter.uid, "timer_objective_index", self.node_timer_objective_index)

    # ── per-kind drawers ───────────────────────────────────────────
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
        if inter.objective_condition in {"EQUALS", "AT_LEAST"}:
            layout.prop(inter, "objective_condition_value", text="Value")

    def _draw_timer_complete(self, layout, _scn, inter):
        layout.prop(self, "node_timer_objective_index", text="Timer Objective")

    # ── main UI ────────────────────────────────────────────────────
    def draw_buttons(self, context, layout):
        scn   = _safe_scene()
        inter = _lookup_backend(self)

        if not scn or not inter:
            # DO NOT schedule here (UI redraws often). Just show status.
            layout.label(text="(Creating backend…)", icon='TIME')
            return

        # Title
        nice = {
            "PROXIMITY": "Proximity",
            "COLLISION": "Collision",
            "INTERACT": "Interact Key",
            "OBJECTIVE_UPDATE": "Objective Update",
            "TIMER_COMPLETE": "Timer Complete",
        }.get(self.KIND, self.KIND)
        layout.label(text=nice, icon='HAND')

        # Per-kind fields
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
        if inter.trigger_mode == "COOLDOWN":
            layout.prop(inter, "trigger_cooldown", text="Cooldown")
        layout.prop(inter, "trigger_delay", text="Delay (sec)")

# ───────────────────────── 5 concrete trigger nodes ─────────────────────────

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
