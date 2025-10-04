import bpy
import uuid
from .base_nodes import ReactionNodeBase

# Optional: standalone editing uses ReactionDefinition if available
try:
    from ..Exp_Game.reactions.exp_reaction_definition import ReactionDefinition
except Exception:
    ReactionDefinition = None


# ───────────────────────── helpers ─────────────────────────
def _scene():
    scn = getattr(bpy.context, "scene", None)
    return scn if scn else (bpy.data.scenes[0] if bpy.data.scenes else None)

def _find_inter_by_uid(scene, uid: str):
    if not scene or not hasattr(scene, "custom_interactions") or not uid:
        return None
    for it in scene.custom_interactions:
        if getattr(it, "uid", "") == uid:
            return it
    return None

# Trigger node ids we support (upstream root for reaction chains)
_TRIGGER_NODE_IDS = {
    "ProximityTriggerNodeType",
    "CollisionTriggerNodeType",
    "InteractTriggerNodeType",
    "ObjectiveUpdateTriggerNodeType",
    "TimerCompleteTriggerNodeType",
}

def _resolve_inter_from_trigger_node(trigger_node):
    scn = _scene()
    if not scn:
        return None
    uid = getattr(trigger_node, "interaction_uid", "")
    if uid:
        return _find_inter_by_uid(scn, uid)
    return None

def _resolve_inter_from_any_upstream(reaction_node, _visited=None):
    """
    Walk upstream from a Reaction node until we find a Trigger node.
    If we hit another Reaction first, keep recursing.
    """
    if _visited is None:
        _visited = set()
    if reaction_node in _visited:
        return None
    _visited.add(reaction_node)

    sock = reaction_node.inputs.get("Trigger Input")
    if not sock or not sock.links:
        return None

    upstream = sock.links[0].from_node
    if getattr(upstream, "bl_idname", "") in _TRIGGER_NODE_IDS:
        return _resolve_inter_from_trigger_node(upstream)
    if getattr(upstream, "bl_idname", "") in _ALL_REACTION_NODE_IDS:
        return _resolve_inter_from_any_upstream(upstream, _visited)
    return None


# ───────────────────────── RNA copy helpers ─────────────────────────
def _copy_reaction_values(src, dst):
    """
    Best-effort copy of simple RNA fields from src → dst.
    Skips rna_type, collections, and read-only props.
    """
    if not src or not dst:
        return
    rna = src.bl_rna
    for prop in rna.properties:
        # Skip internal
        if prop.identifier in {"rna_type"}:
            continue
        # Read-only or collections: skip
        if prop.is_readonly or prop.is_collection:
            continue
        try:
            val = getattr(src, prop.identifier)
        except Exception:
            continue
        try:
            setattr(dst, prop.identifier, val)
        except Exception:
            # Some pointer types may fail if incompatible; ignore gracefully
            pass


# ───────────────────────── sockets ─────────────────────────
class ReactionTriggerInputSocket(bpy.types.NodeSocket):
    bl_idname = "ReactionTriggerInputSocketType"
    bl_label  = "Reaction Trigger Input Socket"
    def draw(self, context, layout, node, text):
        layout.label(text=text)
    def draw_color(self, context, node):
        return (0.4, 0.4, 1.0, 1.0)

class ReactionOutputSocket(bpy.types.NodeSocket):
    bl_idname = "ReactionOutputSocketType"
    bl_label  = "Reaction Output Socket"
    def draw(self, context, layout, node, text):
        layout.label(text=text)
    def draw_color(self, context, node):
        return (0.7, 1.0, 0.7, 1.0)


# ───────────── base class for all specific reaction nodes ─────────────
class _ReactionNodeKind(ReactionNodeBase):
    """Base for concrete reaction nodes. Subclasses must define KIND."""
    KIND = None  # e.g. "CUSTOM_ACTION"

    # Node → Reaction linkage (node is the owner)
    reaction_uid: bpy.props.StringProperty(name="Reaction UID", default="")
    owning_inter_uid: bpy.props.StringProperty(name="Owning Interaction UID", default="")

    # Standalone edit buffer (no writes to Scene during draw/update)
    local_reaction: bpy.props.PointerProperty(
        name="Standalone Reaction",
        type=ReactionDefinition if ReactionDefinition else bpy.types.PropertyGroup
    )

    # ── lifecycle ──
    def init(self, context):
        self.inputs.new("ReactionTriggerInputSocketType", "Trigger Input")
        self.outputs.new("ReactionOutputSocketType",      "Output")
        self.width = 300

    def copy(self, node):
        """
        Duplicated nodes start detached: no backend link yet.
        They will attach to whichever trigger they're connected to.
        """
        self.reaction_uid = ""
        self.owning_inter_uid = ""

    def free(self):
        """
        Node deleted → delete its backend reaction (if any).
        Safe even if links are already gone.
        """
        uid = self.reaction_uid
        ow_uid = self.owning_inter_uid
        if not uid:
            return
        scn = _scene()
        if not scn or not hasattr(scn, "custom_interactions"):
            self.reaction_uid = ""
            self.owning_inter_uid = ""
            return

        def _delete():
            inter = _find_inter_by_uid(scn, ow_uid) if ow_uid else None
            if inter and hasattr(inter, "reactions"):
                for i in range(len(inter.reactions) - 1, -1, -1):
                    r = inter.reactions[i]
                    if getattr(r, "uid", "") == uid:
                        inter.reactions.remove(i)
                        break
            try:
                self.reaction_uid = ""
                self.owning_inter_uid = ""
            except Exception:
                pass
            return None

        bpy.app.timers.register(_delete, first_interval=0.0)

    def update(self):
        """
        Authoring-time sync:
          - If connected upstream to a Trigger → ensure backend reaction exists (create if missing)
          - If connected to a *different* Interaction than last time → MOVE the backend (no leaks, keep data)
          - If disconnected                     → keep backend but do nothing until free() (user may reconnect)
        All writes are deferred via timers to avoid illegal RNA writes during draw/update.
        """
        inter = _resolve_inter_from_any_upstream(self)
        inter_uid_now = getattr(inter, "uid", "") if inter else ""

        # Nothing to write if no upstream trigger
        if not inter:
            return

        react_uid = self.reaction_uid
        kind = self.KIND
        prev_owner = self.owning_inter_uid

        def _ensure_or_move():
            scn = _scene()
            if not scn:
                return None

            cur_inter = _find_inter_by_uid(scn, inter_uid_now)
            if not cur_inter or not hasattr(cur_inter, "reactions"):
                return None

            # CASE 1: No backend yet → create new on current interaction
            if not react_uid:
                r = cur_inter.reactions.add()
                r.name = "Reaction_%d" % len(cur_inter.reactions)
                new_uid = str(uuid.uuid4())
                r.uid = new_uid
                try:
                    if hasattr(r, "reaction_type") and kind and r.reaction_type != kind:
                        r.reaction_type = kind
                except Exception:
                    pass
                try:
                    self.reaction_uid = new_uid
                    self.owning_inter_uid = inter_uid_now
                except Exception:
                    pass
                return None

            # We have a backend uid, find it either in previous owner or current
            prev_inter = _find_inter_by_uid(scn, prev_owner) if prev_owner else None
            r_in_cur = None
            if cur_inter:
                for it in cur_inter.reactions:
                    if getattr(it, "uid", "") == react_uid:
                        r_in_cur = it
                        break

            # CASE 2: It already lives in current interaction → ensure type, update owner marker
            if r_in_cur is not None:
                try:
                    if hasattr(r_in_cur, "reaction_type") and kind and r_in_cur.reaction_type != kind:
                        r_in_cur.reaction_type = kind
                    self.owning_inter_uid = inter_uid_now
                except Exception:
                    pass
                return None

            # CASE 3: It lives in previous interaction → MOVE it to current
            r_in_prev = None
            if prev_inter and hasattr(prev_inter, "reactions"):
                for it in prev_inter.reactions:
                    if getattr(it, "uid", "") == react_uid:
                        r_in_prev = it
                        break

            if r_in_prev is not None:
                # Create new item in current, copy data, delete old
                new_r = cur_inter.reactions.add()
                new_r.name = r_in_prev.name
                new_r.uid = react_uid  # keep same id so node keeps tracking it
                _copy_reaction_values(r_in_prev, new_r)
                try:
                    if hasattr(new_r, "reaction_type") and kind and new_r.reaction_type != kind:
                        new_r.reaction_type = kind
                except Exception:
                    pass

                # remove old
                for i in range(len(prev_inter.reactions) - 1, -1, -1):
                    if getattr(prev_inter.reactions[i], "uid", "") == react_uid:
                        prev_inter.reactions.remove(i)
                        break

                # update owner marker
                try:
                    self.owning_inter_uid = inter_uid_now
                except Exception:
                    pass
                return None

            # CASE 4: Not found anywhere (stale marker) → recreate in current
            r = cur_inter.reactions.add()
            r.name = "Reaction_%d" % len(cur_inter.reactions)
            r.uid = react_uid  # reuse uid if we have one
            try:
                if hasattr(r, "reaction_type") and kind and r.reaction_type != kind:
                    r.reaction_type = kind
            except Exception:
                pass
            try:
                self.owning_inter_uid = inter_uid_now
            except Exception:
                pass
            return None

        bpy.app.timers.register(_ensure_or_move, first_interval=0.0)

    # ── shared field drawers (read-only from Scene; no writes during draw) ──
    def _draw_custom_action(self, box, r):
        box.prop(r, "custom_action_message", text="Notes")
        box.prop_search(r, "custom_action_target", bpy.context.scene, "objects", text="Object")
        box.prop_search(r, "custom_action_action", bpy.data, "actions", text="Action")
        box.prop(r, "custom_action_loop", text="Loop?")
        if r.custom_action_loop:
            box.prop(r, "custom_action_loop_duration", text="Loop Duration")

    def _draw_char_action(self, box, r):
        box.prop_search(r, "char_action_ref", bpy.data, "actions", text="Action")
        box.prop(r, "char_action_mode", text="Mode")
        if r.char_action_mode == 'LOOP':
            box.prop(r, "char_action_loop_duration", text="Loop Duration")

    def _draw_sound(self, box, r):
        box.label(text="Play Packed Sound")
        box.prop(r, "sound_volume", text="Relative Volume")
        box.prop(r, "sound_use_distance", text="Use Distance?")
        box.prop(r, "sound_pointer", text="Sound Datablock")
        box.prop(r, "sound_play_mode", text="Mode")
        if r.sound_play_mode == "DURATION":
            box.prop(r, "sound_duration", text="Duration")
        if r.sound_use_distance:
            box.prop(r, "sound_distance_object", text="Distance Obj")
            box.prop(r, "sound_max_distance", text="Max Distance")

    def _draw_property(self, box, r):
        box.prop(r, "property_data_path", text="Data Path")
        row = box.row(); row.label(text=f"Detected Type: {r.property_type}")
        box.prop(r, "property_transition_duration", text="Duration")
        box.prop(r, "property_reset", text="Reset?")
        if r.property_reset:
            box.prop(r, "property_reset_delay", text="Reset Delay")
        if r.property_type == "BOOL":
            box.prop(r, "bool_value", text="New Bool Value")
        elif r.property_type == "INT":
            box.prop(r, "int_value", text="New Int Value")
        elif r.property_type == "FLOAT":
            box.prop(r, "float_value", text="New Float Value")
        elif r.property_type == "STRING":
            box.prop(r, "string_value", text="New String Value")
        elif r.property_type == "VECTOR":
            box.label(text=f"Vector length: {r.vector_length}")
            box.prop(r, "vector_value", text="New Vector")
        else:
            box.label(text="No property detected or invalid path.")

    def _draw_transform(self, box, r):
        box.prop_search(r, "transform_object", bpy.context.scene, "objects", text="Object")
        box.prop(r, "transform_mode", text="Mode")
        if r.transform_mode == "TO_OBJECT":
            box.prop_search(r, "transform_to_object", bpy.context.scene, "objects", text="To Object")
        if r.transform_mode in {"OFFSET", "TO_LOCATION", "LOCAL_OFFSET"}:
            box.prop(r, "transform_location", text="Location")
            box.prop(r, "transform_rotation", text="Rotation")
            box.prop(r, "transform_scale", text="Scale")
        box.prop(r, "transform_duration", text="Duration")
        if hasattr(r, "transform_distance"):
            box.prop(r, "transform_distance", text="Distance")

    def _draw_custom_ui_text(self, box, r):
        box.prop(r, "custom_text_subtype", text="Subtype")
        if r.custom_text_subtype == "STATIC":
            box.prop(r, "custom_text_value", text="Text")
            box.prop(r, "custom_text_indefinite", text="Indefinite?")
            if not r.custom_text_indefinite:
                box.prop(r, "custom_text_duration", text="Duration")
            box.prop(r, "custom_text_anchor", text="Anchor")
            box.prop(r, "custom_text_scale", text="Scale")
            box.prop(r, "custom_text_margin_x", text="Margin X")
            box.prop(r, "custom_text_margin_y", text="Margin Y")
            box.prop(r, "custom_text_color", text="Color")
        elif r.custom_text_subtype == "OBJECTIVE":
            if hasattr(r, "text_objective_index"):
                box.prop(r, "text_objective_index", text="Objective")
            for fld, lbl in [
                ("custom_text_prefix","Prefix"),
                ("custom_text_include_counter","Show Counter"),
                ("custom_text_suffix","Suffix"),
            ]:
                if hasattr(r, fld):
                    box.prop(r, fld, text=lbl)
            box.prop(r, "custom_text_indefinite", text="Indefinite?")
            if not r.custom_text_indefinite:
                box.prop(r, "custom_text_duration", text="Duration")
            box.prop(r, "custom_text_anchor", text="Anchor")
            box.prop(r, "custom_text_scale", text="Scale")
            box.prop(r, "custom_text_margin_x", text="Margin X")
            box.prop(r, "custom_text_margin_y", text="Margin Y")
            box.prop(r, "custom_text_color", text="Color")
        elif r.custom_text_subtype == "OBJECTIVE_TIMER_DISPLAY":
            if hasattr(r, "text_objective_index"):
                box.prop(r, "text_objective_index", text="Objective")
            box.prop(r, "custom_text_indefinite", text="Indefinite?")
            if not r.custom_text_indefinite:
                box.prop(r, "custom_text_duration", text="Duration")
            box.prop(r, "custom_text_anchor", text="Anchor")
            box.prop(r, "custom_text_scale", text="Scale")
            box.prop(r, "custom_text_margin_x", text="Margin X")
            box.prop(r, "custom_text_margin_y", text="Margin Y")
            box.prop(r, "custom_text_color", text="Color")

    def _draw_objective_counter(self, box, r):
        box.prop(r, "objective_index", text="Objective")
        box.prop(r, "objective_op", text="Operation")
        if r.objective_op in ("ADD", "SUBTRACT"):
            box.prop(r, "objective_amount", text="Amount")

    def _draw_objective_timer(self, box, r):
        box.prop(r, "objective_index", text="Timer Objective")
        box.prop(r, "objective_timer_op", text="Timer Operation")

    def _draw_mobility_game(self, box, r):
        mg = r.mobility_game_settings
        box.prop(mg, "allow_movement", text="Allow Movement")
        box.prop(mg, "allow_jump", text="Allow Jump")
        box.prop(mg, "allow_sprint", text="Allow Sprint")

    # ── main UI (no writes here) ──
    def _linked_reaction(self):
        inter = _resolve_inter_from_any_upstream(self)
        if not inter:
            return None
        uid = self.reaction_uid
        if not uid:
            return None
        for r in inter.reactions:
            if getattr(r, "uid", "") == uid:
                return r
        return None

    def draw_buttons(self, context, layout):
        layout.active = True

        # Try to fetch the linked backend for editing
        r = self._linked_reaction()
        if r:
            box = layout.box()
            box.label(text=f"{self.bl_label}", icon='MODIFIER')
            self._draw_fields(box, r)
            return

        # Disconnected or not yet ensured: draw standalone editor
        layout.label(text=f"{self.bl_label} (Standalone)", icon='DECORATE')
        if self.local_reaction:
            box = layout.box()
            self._draw_fields(box, self.local_reaction)
        else:
            layout.label(text="(No local data)", icon='ERROR')

    # Subclasses implement this to draw type-specific fields only
    def _draw_fields(self, box, r):
        box.label(text="No fields implemented", icon='INFO')


# ───────────── concrete reaction nodes (one per type) ─────────────
class ReactionCustomActionNode(_ReactionNodeKind):
    bl_idname = "ReactionCustomActionNodeType"
    bl_label  = "Custom Action"
    KIND = "CUSTOM_ACTION"
    def _draw_fields(self, box, r): self._draw_custom_action(box, r)

class ReactionCharActionNode(_ReactionNodeKind):
    bl_idname = "ReactionCharActionNodeType"
    bl_label  = "Character Action"
    KIND = "CHAR_ACTION"
    def _draw_fields(self, box, r): self._draw_char_action(box, r)

class ReactionSoundNode(_ReactionNodeKind):
    bl_idname = "ReactionSoundNodeType"
    bl_label  = "Play Sound"
    KIND = "SOUND"
    def _draw_fields(self, box, r): self._draw_sound(box, r)

class ReactionPropertyNode(_ReactionNodeKind):
    bl_idname = "ReactionPropertyNodeType"
    bl_label  = "Property Value"
    KIND = "PROPERTY"
    def _draw_fields(self, box, r): self._draw_property(box, r)

class ReactionTransformNode(_ReactionNodeKind):
    bl_idname = "ReactionTransformNodeType"
    bl_label  = "Transform"
    KIND = "TRANSFORM"
    def _draw_fields(self, box, r): self._draw_transform(box, r)

class ReactionCustomTextNode(_ReactionNodeKind):
    bl_idname = "ReactionCustomTextNodeType"
    bl_label  = "Custom UI Text"
    KIND = "CUSTOM_UI_TEXT"
    def _draw_fields(self, box, r): self._draw_custom_ui_text(box, r)

class ReactionObjectiveCounterNode(_ReactionNodeKind):
    bl_idname = "ReactionObjectiveCounterNodeType"
    bl_label  = "Objective Counter"
    KIND = "OBJECTIVE_COUNTER"
    def _draw_fields(self, box, r): self._draw_objective_counter(box, r)

class ReactionObjectiveTimerNode(_ReactionNodeKind):
    bl_idname = "ReactionObjectiveTimerNodeType"
    bl_label  = "Objective Timer"
    KIND = "OBJECTIVE_TIMER"
    def _draw_fields(self, box, r): self._draw_objective_timer(box, r)

class ReactionMobilityGameNode(_ReactionNodeKind):
    bl_idname = "ReactionMobilityGameNodeType"
    bl_label  = "Mobility & Game"
    KIND = "MOBILITY_GAME"
    def _draw_fields(self, box, r): self._draw_mobility_game(box, r)


# Node idnames for recursion helper
_ALL_REACTION_NODE_IDS = {
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


# ────────────────── optional operator (menu button uses this) ──────────────────
class NODE_OT_add_reaction_to_node(bpy.types.Operator):
    """Ensure a linked reaction in the owning Interaction of this node's chain."""
    bl_idname = "node.add_reaction_to_node"
    bl_label  = "Add Reaction to Node"

    def execute(self, context):
        node_tree = context.space_data.edit_tree
        active = node_tree.nodes.active if node_tree else None
        if not active or getattr(active, "bl_idname", "") not in _ALL_REACTION_NODE_IDS:
            self.report({'WARNING'}, "Active node is not a Reaction node.")
            return {'CANCELLED'}

        inter = _resolve_inter_from_any_upstream(active)
        if not inter:
            self.report({'INFO'}, "This node is not connected (upstream) to a Trigger.")
            return {'CANCELLED'}

        inter_uid = getattr(inter, "uid", "")
        react_uid = getattr(active, "reaction_uid", "")
        kind = getattr(active, "KIND", "")

        def _ensure():
            scn = _scene()
            if not scn:
                return None
            inter2 = _find_inter_by_uid(scn, inter_uid)
            if not inter2 or not hasattr(inter2, "reactions"):
                return None
            r = None
            if react_uid:
                for it in inter2.reactions:
                    if getattr(it, "uid", "") == react_uid:
                        r = it
                        break
            if r is None:
                r = inter2.reactions.add()
                r.name = "Reaction_%d" % len(inter2.reactions)
                new_uid = str(uuid.uuid4())
                r.uid = new_uid
                try:
                    active.reaction_uid = new_uid
                    active.owning_inter_uid = inter_uid
                except Exception:
                    pass
            try:
                if hasattr(r, "reaction_type") and kind and r.reaction_type != kind:
                    r.reaction_type = kind
            except Exception:
                pass
            return None

        bpy.app.timers.register(_ensure, first_interval=0.0)
        self.report({'INFO'}, "Linked reaction will be created/ensured.")
        return {'FINISHED'}


# ───────────────────────── registration ─────────────────────────
_CLASSES = [
    ReactionTriggerInputSocket,
    ReactionOutputSocket,
    ReactionCustomActionNode,
    ReactionCharActionNode,
    ReactionSoundNode,
    ReactionPropertyNode,
    ReactionTransformNode,
    ReactionCustomTextNode,
    ReactionObjectiveCounterNode,
    ReactionObjectiveTimerNode,
    ReactionMobilityGameNode,
    NODE_OT_add_reaction_to_node,
]

def register():
    for c in _CLASSES:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
