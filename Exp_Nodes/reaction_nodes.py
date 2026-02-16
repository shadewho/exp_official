# File: Exp_Nodes/reaction_nodes.py
import bpy
from bpy.types import Node
from .base_nodes import ReactionNodeBase, has_invalid_link, INVALID_LINK_COLOR

# ───────────────────────── helpers ─────────────────────────

def _scene() -> bpy.types.Scene | None:
    scn = getattr(bpy.context, "scene", None)
    if scn:
        return scn
    return bpy.data.scenes[0] if bpy.data.scenes else None

def _enum_action_key_items(self, context):
    scn = getattr(context, "scene", None) or _scene()
    items = []
    if scn and hasattr(scn, "action_keys") and scn.action_keys:
        for it in scn.action_keys:
            nm = getattr(it, "name", "")
            if nm:
                items.append((nm, nm, f"Action: {nm}"))
    if not items:
        items.append(("", "(No Action Keys)", "Add a 'Create Action Key' node first"))
    return items

def _fix_interaction_reaction_indices_after_remove(removed_index: int) -> None:
    scn = _scene()
    if not scn or not hasattr(scn, "custom_interactions"):
        return
    for inter in scn.custom_interactions:
        links = getattr(inter, "reaction_links", None)
        if not links:
            continue
        # remove links pointing to removed index
        to_remove = [i for i, l in enumerate(links) if getattr(l, "reaction_index", -1) == removed_index]
        for i in reversed(to_remove):
            links.remove(i)
        # shift higher indices down by 1
        for l in links:
            idx = getattr(l, "reaction_index", -1)
            if idx > removed_index:
                l.reaction_index = idx - 1

def _reindex_reaction_nodes_after_remove(removed_index: int) -> None:
    """
    After removing scene.reactions[removed_index], all subsequent items shift -1.
    Fix every node that carries a 'reaction_index' (not just those whose
    bl_idname starts with 'Reaction'), including UtilityDelayNode.
    """
    for ng in bpy.data.node_groups:
        if getattr(ng, "bl_idname", "") != "ExploratoryNodesTreeType":
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


def _ensure_reaction(kind_label: str) -> int:
    scn = _scene()
    if not scn:
        return -1
    r = scn.reactions.add()
    r.name = f"{kind_label}_{len(scn.reactions)}"
    if hasattr(r, "reaction_type"):
        try:
            r.reaction_type = kind_label
        except Exception:
            pass
    scn.reactions_index = len(scn.reactions) - 1
    return scn.reactions_index


def _duplicate_reaction_via_operator(src_index: int) -> int:
    scn = _scene()
    if not scn or not (0 <= src_index < len(scn.reactions)):
        return -1
    # preferred path: your operator
    try:
        res = bpy.ops.exploratory.duplicate_global_reaction(index=src_index)
        if 'CANCELLED' not in res:
            return len(scn.reactions) - 1
    except Exception:
        pass
    # fallback shallow copy
    try:
        src = scn.reactions[src_index]
        dst = scn.reactions.add()
        dst.name = f"{getattr(src, 'name', 'Reaction')} (Copy)"
        for prop in src.bl_rna.properties:
            ident = prop.identifier
            if ident == "rna_type" or getattr(prop, "is_readonly", False) or getattr(prop, "is_collection", False):
                continue
            try:
                setattr(dst, ident, getattr(src, ident))
            except Exception:
                pass
        return len(scn.reactions) - 1
    except Exception:
        return -1


def _force_kind(r, kind: str):
    """Make sure the stored reaction type matches the node's KIND."""
    if hasattr(r, "reaction_type"):
        try:
            if r.reaction_type != kind:
                r.reaction_type = kind
        except Exception:
            pass


def _draw_common_fields(layout, r, kind: str):
    """DEPRECATED: Legacy fallback - all nodes now have custom draw_buttons().
    Kept only for safety if somehow called from old code."""
    _force_kind(r, kind)
    header = layout.box()
    header.prop(r, "name", text="Name")
    header.label(text=f"({kind})")



# ───────────────────────── sockets ─────────────────────────

class ReactionTriggerInputSocket(bpy.types.NodeSocket):
    bl_idname = "ReactionTriggerInputSocketType"
    bl_label  = "Reaction Trigger Input"

    _RICH_RED = (0.92, 0.18, 0.18, 1.0)
    _BLUE     = (0.15, 0.55, 1.0, 1.0)
    _MULTI_FROM_TYPES = {
        "TriggerOutputSocketType",
        "ReactionOutputSocketType",
    }

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        if has_invalid_link(self):
            return INVALID_LINK_COLOR
        # Multi-link red: incoming from output with >1 outgoing links
        try:
            for lk in getattr(self, "links", []):
                fs = getattr(lk, "from_socket", None)
                if fs and getattr(fs, "bl_idname", "") in self._MULTI_FROM_TYPES:
                    if getattr(fs, "is_output", False) and len(getattr(fs, "links", [])) > 1:
                        return self._RICH_RED
        except Exception:
            pass
        return self._BLUE

class ReactionOutputSocket(bpy.types.NodeSocket):
    bl_idname = "ReactionOutputSocketType"
    bl_label  = "Reaction Output"

    _RICH_RED = (0.92, 0.18, 0.18, 1.0)
    _BLUE     = (0.15, 0.55, 1.0, 1.0)

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        if has_invalid_link(self):
            return INVALID_LINK_COLOR
        # solid red if this output has multiple outgoing links
        try:
            if self.is_output and len(self.links) > 1:
                return self._RICH_RED
        except Exception:
            pass
        return self._BLUE

# ─────────────────────────────────────────────────────────
# Dynamic Property Sockets - draw reaction properties inline
# ─────────────────────────────────────────────────────────

def _get_reaction_for_node(node):
    """Get the ReactionDefinition for a reaction node."""
    scn = _scene()
    idx = getattr(node, "reaction_index", -1)
    if scn and 0 <= idx < len(getattr(scn, "reactions", [])):
        return scn.reactions[idx]
    return None


class DynamicObjectInputSocket(bpy.types.NodeSocket):
    """DEPRECATED: Use ExpObjectSocketType with reaction_prop instead.
    Kept registered for backwards compatibility with old .blend files."""
    bl_idname = "DynamicObjectInputSocketType"
    bl_label = "Object (Dynamic)"

    prop_name: bpy.props.StringProperty(default="")
    _COLOR = (0.90, 0.50, 0.20, 1.0)  # Orange

    def draw(self, context, layout, node, text):
        if self.is_linked:
            layout.label(text=text)
        else:
            r = _get_reaction_for_node(node)
            if r and self.prop_name:
                layout.prop_search(r, self.prop_name, bpy.context.scene, "objects", text=text)
            else:
                layout.label(text=text)

    def draw_color(self, context, node):
        if has_invalid_link(self):
            return INVALID_LINK_COLOR
        return self._COLOR


class DynamicBoolInputSocket(bpy.types.NodeSocket):
    """DEPRECATED: Use ExpBoolSocketType with reaction_prop instead.
    Kept registered for backwards compatibility with old .blend files."""
    bl_idname = "DynamicBoolInputSocketType"
    bl_label = "Bool (Dynamic)"

    prop_name: bpy.props.StringProperty(default="")
    _COLOR = (0.78, 0.55, 0.78, 1.0)  # Light pink (Blender standard)

    def draw(self, context, layout, node, text):
        if self.is_linked:
            layout.label(text=text)
        else:
            r = _get_reaction_for_node(node)
            if r and self.prop_name:
                layout.prop(r, self.prop_name, text=text)
            else:
                layout.label(text=text)

    def draw_color(self, context, node):
        if has_invalid_link(self):
            return INVALID_LINK_COLOR
        return self._COLOR


class DynamicFloatInputSocket(bpy.types.NodeSocket):
    """DEPRECATED: Use ExpFloatSocketType with reaction_prop instead.
    Kept registered for backwards compatibility with old .blend files."""
    bl_idname = "DynamicFloatInputSocketType"
    bl_label = "Float (Dynamic)"

    prop_name: bpy.props.StringProperty(default="")
    _COLOR = (0.63, 0.63, 0.63, 1.0)  # Gray

    def draw(self, context, layout, node, text):
        if self.is_linked:
            layout.label(text=text)
        else:
            r = _get_reaction_for_node(node)
            if r and self.prop_name:
                layout.prop(r, self.prop_name, text=text)
            else:
                layout.label(text=text)

    def draw_color(self, context, node):
        if has_invalid_link(self):
            return INVALID_LINK_COLOR
        return self._COLOR


class DynamicActionInputSocket(bpy.types.NodeSocket):
    """DEPRECATED: Use ExpActionSocketType with reaction_prop instead.
    Kept registered for backwards compatibility with old .blend files."""
    bl_idname = "DynamicActionInputSocketType"
    bl_label = "Action (Dynamic)"

    prop_name: bpy.props.StringProperty(default="")
    _COLOR = (0.95, 0.85, 0.30, 1.0)  # Yellow (action color)

    def draw(self, context, layout, node, text):
        if self.is_linked:
            layout.label(text=text)
        else:
            r = _get_reaction_for_node(node)
            if r and self.prop_name:
                layout.prop_search(r, self.prop_name, bpy.data, "actions", text=text)
            else:
                layout.label(text=text)

    def draw_color(self, context, node):
        if has_invalid_link(self):
            return INVALID_LINK_COLOR
        return self._COLOR


# ───────────────────────── base class ─────────────────────────

class _ReactionNodeKind(ReactionNodeBase):
    """Base for concrete reaction nodes. Subclasses must set KIND to a valid ReactionDefinition.reaction_type."""
    KIND = None

    reaction_index: bpy.props.IntProperty(name="Reaction Index", default=-1, min=-1)


    # subtle mid-green (body) to contrast triggers/counters/timers without shouting
    _EXPL_TINT_REACTION = (0.18, 0.24, 0.18)
    def _tint(self):
        try:
            self.use_custom_color = True
            self.color = self._EXPL_TINT_REACTION
        except Exception:
            pass

    def init(self, context):
        self.inputs.new("ReactionTriggerInputSocketType", "Reaction Input")
        self.outputs.new("ReactionOutputSocketType",      "Reaction Output")
        self.width = 300
        self._tint()
        self.reaction_index = _ensure_reaction(self.KIND or "CUSTOM_ACTION")

    def copy(self, node):
        src_idx = getattr(node, "reaction_index", -1)
        self.reaction_index = _duplicate_reaction_via_operator(src_idx)
        self.width = getattr(node, "width", 300)

    def free(self):
        """
        Deleting a Reaction node should:
          1) Remove its reaction from scene.reactions
          2) Fix InteractionDefinition.reaction_links indices
          3) Fix every other Reaction node's reaction_index so they don't drift
        """
        scn = _scene()
        idx = self.reaction_index
        if scn and hasattr(scn, "reactions") and 0 <= idx < len(scn.reactions):
            # 1) remove the actual reaction item
            scn.reactions.remove(idx)
            # 2) repair interaction link collections
            _fix_interaction_reaction_indices_after_remove(idx)
            # 3) repair all other nodes' local indices
            _reindex_reaction_nodes_after_remove(idx)

        self.reaction_index = -1

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND or "CUSTOM_ACTION")
        layout.box().prop(r, "name", text="Name")

    def execute_reaction(self, context):
        pass

class ReactionHitscanNode(_ReactionNodeKind):
    bl_idname = "ReactionHitscanNodeType"
    bl_label  = "Hitscan"
    KIND = "HITSCAN"

    def init(self, context):
        super().init(context)
        # Outputs
        self.outputs.new("ExpBoolSocketType",   "Impact")
        self.outputs.new("ExpVectorSocketType", "Impact Location")
        self.outputs.new("ExpObjectSocketType", "Hit Object")
        self.outputs.new("ExpVectorSocketType", "Hit Normal")
        # Inputs
        s = self.inputs.new("ExpObjectSocketType", "Origin Object")
        s.reaction_prop = "proj_origin_object"
        s.use_prop_search = True
        s = self.inputs.new("ExpVectorSocketType", "Offset")
        s.reaction_prop = "proj_origin_offset"
        s = self.inputs.new("ExpObjectSocketType", "Place Object")
        s.reaction_prop = "proj_object"
        s.use_prop_search = True
        s = self.inputs.new("ExpFloatSocketType", "Max Range")
        s.reaction_prop = "proj_max_range"
        s = self.inputs.new("ExpBoolSocketType", "Align to Direction")
        s.reaction_prop = "proj_align_object_to_velocity"
        s = self.inputs.new("ExpBoolSocketType", "Place at Impact")
        s.reaction_prop = "proj_place_hitscan_object"
        s = self.inputs.new("ExpFloatSocketType", "Lifetime")
        s.reaction_prop = "proj_lifetime"
        s = self.inputs.new("ExpIntSocketType", "Max Active")
        s.reaction_prop = "proj_pool_limit"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)
        box = layout.box()
        box.prop(r, "name", text="Name")
        box.prop(r, "proj_aim_source", text="Aim Source")


class ReactionProjectileNode(_ReactionNodeKind):
    bl_idname = "ReactionProjectileNodeType"
    bl_label  = "Projectile"
    KIND = "PROJECTILE"

    def init(self, context):
        super().init(context)
        # Outputs
        self.outputs.new("ExpBoolSocketType",   "Impact")
        self.outputs.new("ExpVectorSocketType", "Impact Location")
        self.outputs.new("ExpObjectSocketType", "Hit Object")
        self.outputs.new("ExpVectorSocketType", "Hit Normal")
        # Inputs
        s = self.inputs.new("ExpObjectSocketType", "Origin Object")
        s.reaction_prop = "proj_origin_object"
        s.use_prop_search = True
        s = self.inputs.new("ExpVectorSocketType", "Offset")
        s.reaction_prop = "proj_origin_offset"
        s = self.inputs.new("ExpObjectSocketType", "Projectile Object")
        s.reaction_prop = "proj_object"
        s.use_prop_search = True
        s = self.inputs.new("ExpFloatSocketType", "Speed")
        s.reaction_prop = "proj_speed"
        s = self.inputs.new("ExpFloatSocketType", "Gravity")
        s.reaction_prop = "proj_gravity"
        s = self.inputs.new("ExpFloatSocketType", "Lifetime")
        s.reaction_prop = "proj_lifetime"
        s = self.inputs.new("ExpBoolSocketType", "Stop on Contact")
        s.reaction_prop = "proj_on_contact_stop"
        s = self.inputs.new("ExpIntSocketType", "Max Active")
        s.reaction_prop = "proj_pool_limit"
        s = self.inputs.new("ExpBoolSocketType", "Align to Velocity")
        s.reaction_prop = "proj_align_object_to_velocity"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)
        box = layout.box()
        box.prop(r, "name", text="Name")
        box.prop(r, "proj_aim_source", text="Aim Source")




class ReactionActionKeysNode(_ReactionNodeKind):
    bl_idname = "ReactionActionKeysNodeType"
    bl_label  = "Action Keys"
    KIND = "ACTION_KEYS"

    node_action_key: bpy.props.EnumProperty(
        name="Action",
        items=_enum_action_key_items,
        description="Choose which Action Key this reaction will enable/disable/toggle",
        update=lambda self, ctx: self._on_node_action_changed(ctx),
    )

    def _on_node_action_changed(self, context):
        scn = _scene()
        idx = getattr(self, "reaction_index", -1)
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            return
        r = scn.reactions[idx]
        name = getattr(self, "node_action_key", "") or ""

        # CRITICAL: Find and set action_key_index FIRST, before setting action_key_name!
        # The _update_action_key_name callback fires when action_key_name changes and
        # uses action_key_index to write to scene.action_keys. If we set name first,
        # the callback uses the OLD index and corrupts the wrong scene entry.
        if hasattr(scn, "action_keys"):
            match = -1
            try:
                for i, it in enumerate(scn.action_keys):
                    if getattr(it, "name", "") == name:
                        match = i
                        break
            except Exception:
                pass
            try:
                r.action_key_index = match
            except Exception:
                pass

        # NOW set the name - callback will use the correct index we just set
        try:
            r.action_key_name = name
            r.action_key_id   = name
        except Exception:
            pass

    def update(self):
        # Keep node_action_key and r.action_key_name in sync (both directions)
        scn = _scene()
        idx = getattr(self, "reaction_index", -1)
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            return

        r = scn.reactions[idx]
        reaction_name = getattr(r, "action_key_name", "") or getattr(r, "action_key_id", "")
        node_name = getattr(self, "node_action_key", "") or ""

        try:
            valid = {it[0] for it in _enum_action_key_items(self, bpy.context)}

            # If reaction has a name but node doesn't match, sync node FROM reaction
            if reaction_name and reaction_name in valid:
                if node_name != reaction_name:
                    self.node_action_key = reaction_name

            # If reaction is EMPTY but node has a valid selection, sync reaction FROM node
            elif not reaction_name and node_name and node_name in valid:
                # Find the action_key_index
                if hasattr(scn, "action_keys"):
                    match = -1
                    for i, it in enumerate(scn.action_keys):
                        if getattr(it, "name", "") == node_name:
                            match = i
                            break
                    try:
                        r.action_key_index = match
                    except Exception:
                        pass
                try:
                    r.action_key_name = node_name
                    r.action_key_id = node_name
                except Exception:
                    pass
        except Exception:
            pass

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]

        # Ensure action_key_name is synced whenever the node is drawn
        node_name = getattr(self, "node_action_key", "") or ""
        reaction_name = getattr(r, "action_key_name", "") or ""
        if node_name and not reaction_name:
            # Node has a selection but reaction doesn't - sync it
            try:
                r.action_key_name = node_name
                r.action_key_id = node_name
            except Exception:
                pass

        box = layout.box()
        box.prop(r, "name", text="Name")

        row = layout.row(align=True)
        row.prop(r, "action_key_op", text="Operation")

        layout.prop(self, "node_action_key", text="Action")


class UtilityDelayNode(_ReactionNodeKind):
    bl_idname = "UtilityDelayNodeType"
    bl_label  = "Delay"
    KIND = "DELAY"

    _EXPL_TINT_UTILITY = (0.35, 0.35, 0.35)

    def _tint(self):
        try:
            self.use_custom_color = True
            self.color = self._EXPL_TINT_UTILITY
        except Exception:
            pass

    def init(self, context):
        super().init(context)
        s = self.inputs.new("ExpFloatSocketType", "Delay (sec)")
        s.reaction_prop = "utility_delay_seconds"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Delay Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]

        box = layout.box()
        box.prop(r, "name", text="Name")

        info = layout.box()
        info.label(text="Delays all reactions AFTER this node by the amount above.", icon='TIME')

class ReactionTrackingNode(_ReactionNodeKind):
    bl_idname = "ReactionTrackingNodeType"
    bl_label  = "Track To"
    KIND = "TRACK_TO"

    def init(self, context):
        super().init(context)
        # From (Mover)
        s = self.inputs.new("ExpObjectSocketType", "From Object")
        s.reaction_prop = "track_from_object"
        s.use_prop_search = True
        # To (Target)
        s = self.inputs.new("ExpObjectSocketType", "To Object")
        s.reaction_prop = "track_to_object"
        s.use_prop_search = True
        # Options
        s = self.inputs.new("ExpFloatSocketType", "Speed")
        s.reaction_prop = "track_speed"
        s = self.inputs.new("ExpFloatSocketType", "Arrive Radius")
        s.reaction_prop = "track_arrive_radius"
        s = self.inputs.new("ExpBoolSocketType", "Respect Proxy Meshes")
        s.reaction_prop = "track_respect_proxy_meshes"
        s = self.inputs.new("ExpBoolSocketType", "Gravity")
        s.reaction_prop = "track_use_gravity"
        s = self.inputs.new("ExpFloatSocketType", "Max Runtime")
        s.reaction_prop = "track_max_runtime"
        # Face Object
        s = self.inputs.new("ExpBoolSocketType", "Face Enable")
        s.reaction_prop = "track_face_enabled"
        s = self.inputs.new("ExpObjectSocketType", "Face Object")
        s.reaction_prop = "track_face_object"
        s.use_prop_search = True

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)

        box = layout.box()
        box.prop(r, "name", text="Name")
        box.prop(r, "track_mode", text="Nav Mode")
        if getattr(r, "track_face_enabled", False):
            box.prop(r, "track_face_axis", text="Face Axis")


class ReactionParentingNode(_ReactionNodeKind):
    bl_idname = "ReactionParentingNodeType"
    bl_label  = "Parent / Unparent"
    KIND = "PARENTING"

    def init(self, context):
        super().init(context)

        # Target (child)
        s = self.inputs.new("ExpObjectSocketType", "Target Object")
        s.reaction_prop = "parenting_target_object"
        s.use_prop_search = True
        # Parent
        s = self.inputs.new("ExpObjectSocketType", "Parent Object")
        s.reaction_prop = "parenting_parent_object"
        s.use_prop_search = True
        # Vector input for local offset
        s_offset = self.inputs.new("ExpVectorSocketType", "Local Offset")
        s_offset.reaction_prop = "parenting_local_offset"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]

        box = layout.box()
        box.prop(r, "name", text="Name")
        box.prop(r, "parenting_op", text="Operation")
        box.prop(r, "parenting_bone_name", text="Bone")



# ───────────────────────── concrete nodes ─────────────────────────

class ReactionCustomActionNode(_ReactionNodeKind):
    bl_idname = "ReactionCustomActionNodeType"
    bl_label  = "Custom Action"
    KIND = "CUSTOM_ACTION"

    def init(self, context):
        super().init(context)
        s = self.inputs.new("ExpActionSocketType", "Action")
        s.reaction_prop = "custom_action_action"
        s = self.inputs.new("ExpObjectSocketType", "Object")
        s.reaction_prop = "custom_action_target"
        s.use_prop_search = True
        s = self.inputs.new("ExpBoolSocketType", "Loop?")
        s.reaction_prop = "custom_action_loop"
        s = self.inputs.new("ExpFloatSocketType", "Loop Duration")
        s.reaction_prop = "custom_action_loop_duration"
        s = self.inputs.new("ExpFloatSocketType", "Speed")
        s.reaction_prop = "custom_action_speed"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(scn.reactions)):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)
        header = layout.box()
        header.prop(r, "name", text="Name")
        header.prop(r, "custom_action_message", text="Notes")


class ReactionCharActionNode(_ReactionNodeKind):
    bl_idname = "ReactionCharActionNodeType"
    bl_label  = "Character Action"
    KIND = "CHAR_ACTION"

    def init(self, context):
        super().init(context)
        s = self.inputs.new("ExpActionSocketType", "Action")
        s.reaction_prop = "char_action_ref"
        s = self.inputs.new("ExpFloatSocketType", "Speed")
        s.reaction_prop = "char_action_speed"
        s = self.inputs.new("ExpFloatSocketType", "Blend Time")
        s.reaction_prop = "char_action_blend_time"
        s = self.inputs.new("ExpFloatSocketType", "Loop Duration")
        s.reaction_prop = "char_action_loop_duration"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)
        header = layout.box()
        header.prop(r, "name", text="Name")
        header.prop(r, "char_action_bone_group", text="Body Part")
        header.prop(r, "char_action_mode", text="Mode")


class ReactionSoundNode(_ReactionNodeKind):
    bl_idname = "ReactionSoundNodeType"
    bl_label  = "Play Sound"
    KIND = "SOUND"

    def init(self, context):
        super().init(context)
        s = self.inputs.new("ExpFloatSocketType", "Volume")
        s.reaction_prop = "sound_volume"
        s = self.inputs.new("ExpBoolSocketType", "Use Distance?")
        s.reaction_prop = "sound_use_distance"
        s = self.inputs.new("ExpObjectSocketType", "Distance Object")
        s.reaction_prop = "sound_distance_object"
        s.use_prop_search = True
        s = self.inputs.new("ExpFloatSocketType", "Max Distance")
        s.reaction_prop = "sound_max_distance"
        s = self.inputs.new("ExpFloatSocketType", "Duration")
        s.reaction_prop = "sound_duration"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)
        header = layout.box()
        header.prop(r, "name", text="Name")
        header.prop(r, "sound_pointer", text="Sound")
        header.prop(r, "sound_play_mode", text="Mode")
        pack = layout.box()
        pack.label(text="Custom sounds must be packed into the .blend.", icon='INFO')
        row = pack.row(align=True)
        row.operator("exp_audio.pack_all_sounds", text="Pack All Sounds", icon='PACKAGE')
        ridx = -1
        for i, rx in enumerate(getattr(scn, "reactions", [])):
            if rx == r:
                ridx = i
                break
        test_box = layout.box()
        row = test_box.row(align=True)
        op = row.operator("exp_audio.test_reaction_sound", text="Test Sound", icon='PLAY')
        op.reaction_index = ridx


class ReactionPropertyNode(_ReactionNodeKind):
    bl_idname = "ReactionPropertyNodeType"
    bl_label  = "Property Value"
    KIND = "PROPERTY"

    def init(self, context):
        super().init(context)
        s = self.inputs.new("ExpFloatSocketType", "Duration")
        s.reaction_prop = "property_transition_duration"
        s = self.inputs.new("ExpBoolSocketType", "Reset After")
        s.reaction_prop = "property_reset"
        s = self.inputs.new("ExpFloatSocketType", "Reset Delay")
        s.reaction_prop = "property_reset_delay"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)
        header = layout.box()
        header.prop(r, "name", text="Name")
        header.label(text="Paste a Blender full data path (Right-Click > Copy Full Data Path).", icon='INFO')
        header.prop(r, "property_data_path", text="Full Data Path")
        row = header.row()
        row.label(text=f"Detected Type: {getattr(r, 'property_type', 'NONE')}")
        pt = getattr(r, "property_type", "NONE")
        if pt == "BOOL":
            header.prop(r, "bool_value", text="Target Bool")
            header.prop(r, "default_bool_value", text="Default Bool")
        elif pt == "INT":
            header.prop(r, "int_value", text="Target Int")
            header.prop(r, "default_int_value", text="Default Int")
        elif pt == "FLOAT":
            header.prop(r, "float_value", text="Target Float")
            header.prop(r, "default_float_value", text="Default Float")
        elif pt == "STRING":
            header.prop(r, "string_value", text="Target String")
            header.prop(r, "default_string_value", text="Default String")
        elif pt == "VECTOR":
            header.label(text=f"Vector length: {getattr(r, 'vector_length', 3)}")
            header.prop(r, "vector_value", text="Target Vector")
            header.prop(r, "default_vector_value", text="Default Vector")

class ReactionTransformNode(_ReactionNodeKind):
    bl_idname = "ReactionTransformNodeType"
    bl_label  = "Transform"
    KIND = "TRANSFORM"

    def init(self, context):
        # Keep base Reaction Input/Output
        super().init(context)

        # Target Object socket
        s = self.inputs.new("ExpObjectSocketType", "Target Object")
        s.reaction_prop = "transform_object"
        s.use_prop_search = True

        # TO_OBJECT source object (hidden until mode is TO_OBJECT)
        s = self.inputs.new("ExpObjectSocketType", "To Object")
        s.reaction_prop = "transform_to_object"
        s.use_prop_search = True
        s.hide = True

        # TO_BONE armature (hidden until mode is TO_BONE)
        s = self.inputs.new("ExpObjectSocketType", "Armature")
        s.reaction_prop = "transform_to_armature"
        s.use_prop_search = True
        s.hide = True

        # Vector inputs with inline property drawing (unified socket type)
        s_loc = self.inputs.new("ExpVectorSocketType", "Location")
        s_loc.reaction_prop = "transform_location"

        s_rot = self.inputs.new("ExpVectorSocketType", "Rotation")
        s_rot.reaction_prop = "transform_rotation"

        s_scl = self.inputs.new("ExpVectorSocketType", "Scale")
        s_scl.reaction_prop = "transform_scale"

        # Duration as float input
        s_dur = self.inputs.new("ExpFloatSocketType", "Duration")
        s_dur.reaction_prop = "transform_duration"

    def draw_buttons(self, context, layout):
        """Custom draw that doesn't duplicate socket-drawn fields."""
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(scn.reactions)):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]

        # Name
        header = layout.box()
        header.prop(r, "name", text="Name")

        # Mode
        header.prop(r, "transform_mode", text="Mode")
        mode = getattr(r, "transform_mode", "OFFSET")

        # TO_OBJECT mode specific fields
        if mode == "TO_OBJECT":
            col = header.column(align=True)
            col.label(text="Copy Channels:")
            col.prop(r, "transform_use_location", text="Location")
            col.prop(r, "transform_use_rotation", text="Rotation")
            col.prop(r, "transform_use_scale", text="Scale")

        # TO_BONE mode specific fields
        elif mode == "TO_BONE":
            header.prop(r, "transform_bone_name", text="Bone (name)")

class ReactionCustomTextNode(_ReactionNodeKind):
    bl_idname = "ReactionCustomTextNodeType"
    bl_label  = "Custom UI Text"
    KIND = "CUSTOM_UI_TEXT"

    def init(self, context):
        super().init(context)
        s = self.inputs.new("ExpIntSocketType", "Scale")
        s.reaction_prop = "custom_text_scale"
        s = self.inputs.new("ExpIntSocketType", "Margin X")
        s.reaction_prop = "custom_text_margin_x"
        s = self.inputs.new("ExpIntSocketType", "Margin Y")
        s.reaction_prop = "custom_text_margin_y"
        s = self.inputs.new("ExpFloatSocketType", "Duration")
        s.reaction_prop = "custom_text_duration"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)
        header = layout.box()
        header.prop(r, "name", text="Name")
        header.prop(r, "custom_text_subtype", text="Subtype")
        subtype = getattr(r, "custom_text_subtype", "STATIC")
        content = header.box()
        content.label(text="Text Content")
        if subtype == "STATIC":
            content.prop(r, "custom_text_value", text="Text")
        elif subtype == "COUNTER_DISPLAY":
            content.prop(r, "text_counter_index", text="Counter")
            fmt = content.box()
            fmt.label(text="Counter Formatting")
            fmt.prop(r, "custom_text_prefix", text="Prefix")
            fmt.prop(r, "custom_text_include_counter", text="Show Counter")
            fmt.prop(r, "custom_text_suffix", text="Suffix")
        elif subtype == "TIMER_DISPLAY":
            content.prop(r, "text_timer_index", text="Timer")
        timing = header.box()
        timing.label(text="Display Timing")
        timing.prop(r, "custom_text_indefinite", text="Indefinite")
        header.prop(r, "custom_text_anchor", text="Anchor")
        header.prop(r, "custom_text_font", text="Font")
        header.prop(r, "custom_text_color", text="Color")
        preview_box = layout.box()
        preview_box.label(text="Note: preview in fullscreen for best results.", icon='INFO')
        row = preview_box.row(align=True)
        op = row.operator("exploratory.preview_custom_text", text="Preview Custom Text", icon='HIDE_OFF')
        op.duration = 5.0


class ReactionCounterUpdateNode(_ReactionNodeKind):
    bl_idname = "ReactionCounterUpdateNodeType"
    bl_label  = "Counter Update"
    KIND = "COUNTER_UPDATE"

    def init(self, context):
        super().init(context)
        s = self.inputs.new("ExpIntSocketType", "Amount")
        s.reaction_prop = "counter_amount"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)
        header = layout.box()
        header.prop(r, "name", text="Name")
        header.prop(r, "counter_index", text="Counter")
        header.prop(r, "counter_op", text="Operation")


class ReactionTimerControlNode(_ReactionNodeKind):
    bl_idname = "ReactionTimerControlNodeType"
    bl_label  = "Timer Control"
    KIND = "TIMER_CONTROL"

    def init(self, context):
        super().init(context)
        s = self.inputs.new("ExpBoolSocketType", "Interruptible")
        s.reaction_prop = "interruptible"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)
        header = layout.box()
        header.prop(r, "name", text="Name")
        header.prop(r, "timer_index", text="Timer")
        header.prop(r, "timer_op", text="Timer Operation")


class ReactionMobilityNode(_ReactionNodeKind):
    bl_idname = "ReactionMobilityNodeType"
    bl_label  = "Mobility"
    KIND = "MOBILITY"

    def init(self, context):
        super().init(context)
        s = self.inputs.new("ExpBoolSocketType", "Allow Movement")
        s.reaction_prop = "mob_allow_movement"
        s = self.inputs.new("ExpBoolSocketType", "Allow Jump")
        s.reaction_prop = "mob_allow_jump"
        s = self.inputs.new("ExpBoolSocketType", "Allow Sprint")
        s.reaction_prop = "mob_allow_sprint"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)
        header = layout.box()
        header.prop(r, "name", text="Name")


class ReactionMeshVisibilityNode(_ReactionNodeKind):
    bl_idname = "ReactionMeshVisibilityNodeType"
    bl_label  = "Mesh Visibility"
    KIND = "MESH_VISIBILITY"

    def init(self, context):
        super().init(context)
        s = self.inputs.new("ExpObjectSocketType", "Mesh Object")
        s.reaction_prop = "mesh_vis_object"
        s.use_prop_search = True

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)
        header = layout.box()
        header.prop(r, "name", text="Name")
        vs = getattr(r, "mesh_visibility", None)
        if vs:
            header.prop(vs, "mesh_action", text="Action")


class ReactionResetGameNode(_ReactionNodeKind):
    bl_idname = "ReactionResetGameNodeType"
    bl_label  = "Reset Game"
    KIND = "RESET_GAME"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)
        header = layout.box()
        header.prop(r, "name", text="Name")
        header.label(text="Reset Game on trigger", icon='FILE_REFRESH')


class ReactionCrosshairsNode(_ReactionNodeKind):
    bl_idname = "ReactionCrosshairsNodeType"
    bl_label  = "Enable Crosshairs"
    KIND = "ENABLE_CROSSHAIRS"

    def init(self, context):
        super().init(context)
        s = self.inputs.new("ExpIntSocketType", "Arm Length")
        s.reaction_prop = "crosshair_length_px"
        s = self.inputs.new("ExpIntSocketType", "Gap")
        s.reaction_prop = "crosshair_gap_px"
        s = self.inputs.new("ExpIntSocketType", "Thickness")
        s.reaction_prop = "crosshair_thickness_px"
        s = self.inputs.new("ExpIntSocketType", "Dot Radius")
        s.reaction_prop = "crosshair_dot_radius_px"
        s = self.inputs.new("ExpBoolSocketType", "Indefinite")
        s.reaction_prop = "crosshair_indefinite"
        s = self.inputs.new("ExpFloatSocketType", "Duration")
        s.reaction_prop = "crosshair_duration"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)
        header = layout.box()
        header.prop(r, "name", text="Name")
        header.prop(r, "crosshair_style", text="Style")
        header.prop(r, "crosshair_color", text="Color")


class ReactionEnableHealthNode(_ReactionNodeKind):
    bl_idname = "ReactionEnableHealthNodeType"
    bl_label  = "Enable Health"
    KIND = "ENABLE_HEALTH"

    def init(self, context):
        super().init(context)

        # Object socket - accepts any object (can connect ObjectDataNode with "use character")
        s_obj = self.inputs.new("ExpObjectSocketType", "Target Object")
        s_obj.reaction_prop = "health_target_object"
        s_obj.use_prop_search = True

        # Float sockets for health values
        s_start = self.inputs.new("ExpFloatSocketType", "Start Value")
        s_start.reaction_prop = "health_start_value"

        s_min = self.inputs.new("ExpFloatSocketType", "Min Value")
        s_min.reaction_prop = "health_min_value"

        s_max = self.inputs.new("ExpFloatSocketType", "Max Value")
        s_max.reaction_prop = "health_max_value"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)

        box = layout.box()
        box.prop(r, "name", text="Name")

        # Info box
        info = layout.box()
        info.label(text="Attach health to any object.", icon='FUND')


class ReactionDisplayHealthUINode(_ReactionNodeKind):
    bl_idname = "ReactionDisplayHealthUINodeType"
    bl_label  = "Display Health UI"
    KIND = "DISPLAY_HEALTH_UI"

    def _is_target_character(self):
        """Check if the Target Object socket is linked to an ObjectDataNode with use_character=True."""
        sock = self.inputs.get("Target Object")
        if not sock or not sock.is_linked:
            return False
        src_node = sock.links[0].from_node
        return (getattr(src_node, 'bl_idname', '') == 'ObjectDataNodeType'
                and getattr(src_node, 'use_character', False))

    def _update_socket_visibility(self):
        """Toggle HUD vs world-space sockets based on target type."""
        is_char = self._is_target_character()

        hud_names = ("Scale", "Offset X", "Offset Y")
        world_names = ("World Scale", "World Offset Horizontal", "World Offset Vertical")

        for name in hud_names:
            sock = self.inputs.get(name)
            if sock:
                sock.hide = not is_char

        for name in world_names:
            sock = self.inputs.get(name)
            if sock:
                sock.hide = is_char

    def init(self, context):
        super().init(context)

        # Object socket - which object's health to display
        s_obj = self.inputs.new("ExpObjectSocketType", "Target Object")
        s_obj.reaction_prop = "health_ui_target_object"
        s_obj.use_prop_search = True

        # HUD sockets (for character target)
        s_scale = self.inputs.new("ExpIntSocketType", "Scale")
        s_scale.reaction_prop = "health_ui_scale"

        s_offset_x = self.inputs.new("ExpIntSocketType", "Offset X")
        s_offset_x.reaction_prop = "health_ui_offset_x"

        s_offset_y = self.inputs.new("ExpIntSocketType", "Offset Y")
        s_offset_y.reaction_prop = "health_ui_offset_y"

        # World-space sockets (for non-character targets)
        s_wscale = self.inputs.new("ExpIntSocketType", "World Scale")
        s_wscale.reaction_prop = "health_ui_world_scale"

        s_woffh = self.inputs.new("ExpIntSocketType", "World Offset Horizontal")
        s_woffh.reaction_prop = "health_ui_world_offset_h"

        s_woffv = self.inputs.new("ExpIntSocketType", "World Offset Vertical")
        s_woffv.reaction_prop = "health_ui_world_offset_v"

        # Default: hide HUD sockets (no target connected = non-character)
        for name in ("Scale", "Offset X", "Offset Y"):
            sock = self.inputs.get(name)
            if sock:
                sock.hide = True

    def update(self):
        """Called when links change - update socket visibility."""
        self._update_socket_visibility()

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)

        box = layout.box()
        box.prop(r, "name", text="Name")

        is_char = self._is_target_character()

        if is_char:
            # Character target - show HUD options
            layout.prop(r, "health_ui_position")
            info = layout.box()
            info.label(text="HUD health bar overlay.", icon='HEART')
        else:
            # Non-character target - show world-space options
            layout.prop(r, "health_ui_world_style")
            layout.prop(r, "health_ui_world_show_through")
            info = layout.box()
            info.label(text="World-space health display above object.", icon='HEART')


class ReactionAdjustHealthNode(_ReactionNodeKind):
    bl_idname = "ReactionAdjustHealthNodeType"
    bl_label  = "Adjust Health"
    KIND = "ADJUST_HEALTH"

    def init(self, context):
        super().init(context)

        s_obj = self.inputs.new("ExpObjectSocketType", "Target Object")
        s_obj.reaction_prop = "adjust_health_target_object"
        s_obj.use_prop_search = True

        s_amount = self.inputs.new("ExpFloatSocketType", "Amount")
        s_amount.reaction_prop = "adjust_health_amount"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _force_kind(r, self.KIND)

        box = layout.box()
        box.prop(r, "name", text="Name")

        info = layout.box()
        info.label(text="Positive = heal, negative = damage.", icon='HEART')