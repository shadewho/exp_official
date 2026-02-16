# Exploratory/Exp_Nodes/utility_nodes.py
import bpy
import math
import random
from bpy.types import Node

# Store API (central registry)
from ..Exp_Game.props_and_utils.exp_utility_store import (
    # Float
    create_float_slot, set_float, get_float, float_slot_exists,
    # Integer
    create_int_slot, set_int, get_int, int_slot_exists,
    # Boolean
    create_bool_slot, set_bool, get_bool, bool_slot_exists,
    # Object
    create_object_slot, set_object, get_object, get_object_name, object_slot_exists,
    # Collection
    create_collection_slot, set_collection, get_collection, get_collection_name, collection_slot_exists,
    # Action
    create_action_slot, set_action, get_action, get_action_name, action_slot_exists,
    # Float Vector
    create_floatvec_slot, set_floatvec, get_floatvec, slot_exists,
)
from .base_nodes import _ExploratoryNodeOnly, has_invalid_link, INVALID_LINK_COLOR
from ..Exp_Game.props_and_utils.trackers import EQUALITY_ONLY_OPERATORS

EXPL_TREE_ID = "ExploratoryNodesTreeType"


# ═══════════════════════════════════════════════════════════
# UNIFIED SOCKET TYPES
# ═══════════════════════════════════════════════════════════
#
# ONE socket type per data type - all sockets of the same type
# can connect to each other regardless of which node they're on.
#
# Features:
# - prop_name: If set on an INPUT socket, draws the node property
#              inline when not connected (for inline editing)
# - Works for both inputs and outputs with same bl_idname
# ═══════════════════════════════════════════════════════════

# Colors for socket types (consistent visual language)
_COLOR_FLOAT      = (0.63, 0.63, 0.63, 1.0)  # Gray
_COLOR_INT        = (0.37, 0.56, 0.36, 1.0)  # Dark green (Blender standard)
_COLOR_BOOL       = (0.78, 0.55, 0.78, 1.0)  # Light pink (Blender standard)
_COLOR_OBJECT     = (0.90, 0.50, 0.20, 1.0)  # Orange
_COLOR_COLLECTION = (0.95, 0.95, 0.30, 1.0)  # Yellow
_COLOR_ACTION     = (0.95, 0.85, 0.30, 1.0)  # Gold
_COLOR_VECTOR     = (0.65, 0.40, 0.95, 1.0)  # Purple


def _get_reaction_for_socket(node):
    """
    Helper for unified sockets to access reaction properties.
    Returns the ReactionDefinition if node is a reaction node, else None.
    """
    if not hasattr(node, 'reaction_index'):
        return None
    idx = getattr(node, 'reaction_index', -1)
    if idx < 0:
        return None
    scn = bpy.context.scene
    reactions = getattr(scn, 'reactions', None)
    if reactions and 0 <= idx < len(reactions):
        return reactions[idx]
    return None


def _get_interaction_for_socket(node):
    """Returns InteractionDefinition if node has interaction_index."""
    if not hasattr(node, 'interaction_index'):
        return None
    idx = getattr(node, 'interaction_index', -1)
    if idx < 0:
        return None
    scn = bpy.context.scene
    interactions = getattr(scn, 'custom_interactions', None)
    if interactions and 0 <= idx < len(interactions):
        return interactions[idx]
    return None


def _get_counter_for_socket(node):
    """Returns CounterDefinition if node has counter_index."""
    if not hasattr(node, 'counter_index'):
        return None
    idx = getattr(node, 'counter_index', -1)
    if idx < 0:
        return None
    scn = bpy.context.scene
    counters = getattr(scn, 'counters', None)
    if counters and 0 <= idx < len(counters):
        return counters[idx]
    return None


def _get_timer_for_socket(node):
    """Returns TimerDefinition if node has timer_index."""
    if not hasattr(node, 'timer_index'):
        return None
    idx = getattr(node, 'timer_index', -1)
    if idx < 0:
        return None
    scn = bpy.context.scene
    timers = getattr(scn, 'timers', None)
    if timers and 0 <= idx < len(timers):
        return timers[idx]
    return None


# ─────────────────────────────────────────────────────────
# Float Socket (unified)
# ─────────────────────────────────────────────────────────
class ExpFloatSocket(bpy.types.NodeSocket):
    """Unified float socket - works for both input and output."""
    bl_idname = "ExpFloatSocketType"
    bl_label = "Float"

    prop_name: bpy.props.StringProperty(
        default="",
        description="Node property to draw inline when not connected"
    )
    reaction_prop: bpy.props.StringProperty(
        default="",
        description="Reaction property to draw inline (for reaction nodes)"
    )
    interaction_prop: bpy.props.StringProperty(
        default="",
        description="Interaction property to draw inline (for trigger nodes)"
    )
    counter_prop: bpy.props.StringProperty(
        default="",
        description="Counter property to draw inline (for counter nodes)"
    )
    timer_prop: bpy.props.StringProperty(
        default="",
        description="Timer property to draw inline (for timer nodes)"
    )

    def draw(self, context, layout, node, text):
        if not self.is_output and not self.is_linked:
            if self.reaction_prop:
                r = _get_reaction_for_socket(node)
                if r and hasattr(r, self.reaction_prop):
                    layout.prop(r, self.reaction_prop, text=text or "")
                    return
            if self.interaction_prop:
                inter = _get_interaction_for_socket(node)
                if inter and hasattr(inter, self.interaction_prop):
                    layout.prop(inter, self.interaction_prop, text=text or "")
                    return
            if self.counter_prop:
                c = _get_counter_for_socket(node)
                if c and hasattr(c, self.counter_prop):
                    layout.prop(c, self.counter_prop, text=text or "")
                    return
            if self.timer_prop:
                t = _get_timer_for_socket(node)
                if t and hasattr(t, self.timer_prop):
                    layout.prop(t, self.timer_prop, text=text or "")
                    return
            if self.prop_name and hasattr(node, self.prop_name):
                layout.prop(node, self.prop_name, text=text or "")
                return
        layout.label(text=text or ("Float" if self.is_output else "Input"))

    def draw_color(self, context, node):
        if has_invalid_link(self):
            return INVALID_LINK_COLOR
        return _COLOR_FLOAT


# ─────────────────────────────────────────────────────────
# Integer Socket (unified)
# ─────────────────────────────────────────────────────────
class ExpIntSocket(bpy.types.NodeSocket):
    """Unified integer socket - works for both input and output."""
    bl_idname = "ExpIntSocketType"
    bl_label = "Integer"

    prop_name: bpy.props.StringProperty(
        default="",
        description="Node property to draw inline when not connected"
    )
    reaction_prop: bpy.props.StringProperty(
        default="",
        description="Reaction property to draw inline (for reaction nodes)"
    )
    interaction_prop: bpy.props.StringProperty(
        default="",
        description="Interaction property to draw inline (for trigger nodes)"
    )
    counter_prop: bpy.props.StringProperty(
        default="",
        description="Counter property to draw inline (for counter nodes)"
    )
    timer_prop: bpy.props.StringProperty(
        default="",
        description="Timer property to draw inline (for timer nodes)"
    )

    def draw(self, context, layout, node, text):
        if not self.is_output and not self.is_linked:
            if self.reaction_prop:
                r = _get_reaction_for_socket(node)
                if r and hasattr(r, self.reaction_prop):
                    layout.prop(r, self.reaction_prop, text=text or "")
                    return
            if self.interaction_prop:
                inter = _get_interaction_for_socket(node)
                if inter and hasattr(inter, self.interaction_prop):
                    layout.prop(inter, self.interaction_prop, text=text or "")
                    return
            if self.counter_prop:
                c = _get_counter_for_socket(node)
                if c and hasattr(c, self.counter_prop):
                    layout.prop(c, self.counter_prop, text=text or "")
                    return
            if self.timer_prop:
                t = _get_timer_for_socket(node)
                if t and hasattr(t, self.timer_prop):
                    layout.prop(t, self.timer_prop, text=text or "")
                    return
            if self.prop_name and hasattr(node, self.prop_name):
                layout.prop(node, self.prop_name, text=text or "")
                return
        layout.label(text=text or ("Integer" if self.is_output else "Input"))

    def draw_color(self, context, node):
        if has_invalid_link(self):
            return INVALID_LINK_COLOR
        return _COLOR_INT


# ─────────────────────────────────────────────────────────
# Boolean Socket (unified)
# ─────────────────────────────────────────────────────────
class ExpBoolSocket(bpy.types.NodeSocket):
    """Unified boolean socket - works for both input and output."""
    bl_idname = "ExpBoolSocketType"
    bl_label = "Boolean"

    prop_name: bpy.props.StringProperty(
        default="",
        description="Node property to draw inline when not connected"
    )
    reaction_prop: bpy.props.StringProperty(
        default="",
        description="Reaction property to draw inline (for reaction nodes)"
    )
    interaction_prop: bpy.props.StringProperty(
        default="",
        description="Interaction property to draw inline (for trigger nodes)"
    )
    counter_prop: bpy.props.StringProperty(
        default="",
        description="Counter property to draw inline (for counter nodes)"
    )
    timer_prop: bpy.props.StringProperty(
        default="",
        description="Timer property to draw inline (for timer nodes)"
    )

    def draw(self, context, layout, node, text):
        if not self.is_output and not self.is_linked:
            if self.reaction_prop:
                r = _get_reaction_for_socket(node)
                if r and hasattr(r, self.reaction_prop):
                    layout.prop(r, self.reaction_prop, text=text or "")
                    return
            if self.interaction_prop:
                inter = _get_interaction_for_socket(node)
                if inter and hasattr(inter, self.interaction_prop):
                    layout.prop(inter, self.interaction_prop, text=text or "")
                    return
            if self.counter_prop:
                c = _get_counter_for_socket(node)
                if c and hasattr(c, self.counter_prop):
                    layout.prop(c, self.counter_prop, text=text or "")
                    return
            if self.timer_prop:
                t = _get_timer_for_socket(node)
                if t and hasattr(t, self.timer_prop):
                    layout.prop(t, self.timer_prop, text=text or "")
                    return
            if self.prop_name and hasattr(node, self.prop_name):
                layout.prop(node, self.prop_name, text=text or "")
                return
        layout.label(text=text or ("Result" if self.is_output else "Input"))

    def draw_color(self, context, node):
        if has_invalid_link(self):
            return INVALID_LINK_COLOR
        return _COLOR_BOOL


# ─────────────────────────────────────────────────────────
# Object Socket (unified)
# ─────────────────────────────────────────────────────────
class ExpObjectSocket(bpy.types.NodeSocket):
    """Unified object socket - works for both input and output."""
    bl_idname = "ExpObjectSocketType"
    bl_label = "Object"

    prop_name: bpy.props.StringProperty(
        default="",
        description="Node property to draw inline when not connected"
    )
    reaction_prop: bpy.props.StringProperty(
        default="",
        description="Reaction property to draw inline (for reaction nodes)"
    )
    interaction_prop: bpy.props.StringProperty(
        default="",
        description="Interaction property to draw inline (for trigger nodes)"
    )
    counter_prop: bpy.props.StringProperty(
        default="",
        description="Counter property to draw inline (for counter nodes)"
    )
    timer_prop: bpy.props.StringProperty(
        default="",
        description="Timer property to draw inline (for timer nodes)"
    )
    use_prop_search: bpy.props.BoolProperty(
        default=False,
        description="Use prop_search for objects instead of direct prop"
    )

    def draw(self, context, layout, node, text):
        if not self.is_output and not self.is_linked:
            if self.reaction_prop:
                r = _get_reaction_for_socket(node)
                if r and hasattr(r, self.reaction_prop):
                    if self.use_prop_search:
                        layout.prop_search(r, self.reaction_prop, bpy.context.scene, "objects", text=text or "")
                    else:
                        layout.prop(r, self.reaction_prop, text="")
                    return
            if self.interaction_prop:
                inter = _get_interaction_for_socket(node)
                if inter and hasattr(inter, self.interaction_prop):
                    if self.use_prop_search:
                        layout.prop_search(inter, self.interaction_prop, bpy.context.scene, "objects", text=text or "")
                    else:
                        layout.prop(inter, self.interaction_prop, text="")
                    return
            if self.counter_prop:
                c = _get_counter_for_socket(node)
                if c and hasattr(c, self.counter_prop):
                    if self.use_prop_search:
                        layout.prop_search(c, self.counter_prop, bpy.context.scene, "objects", text=text or "")
                    else:
                        layout.prop(c, self.counter_prop, text="")
                    return
            if self.timer_prop:
                t = _get_timer_for_socket(node)
                if t and hasattr(t, self.timer_prop):
                    if self.use_prop_search:
                        layout.prop_search(t, self.timer_prop, bpy.context.scene, "objects", text=text or "")
                    else:
                        layout.prop(t, self.timer_prop, text="")
                    return
            if self.prop_name and hasattr(node, self.prop_name):
                layout.prop(node, self.prop_name, text="")
                return
        layout.label(text=text or ("Object" if self.is_output else "Input"))

    def draw_color(self, context, node):
        if has_invalid_link(self):
            return INVALID_LINK_COLOR
        return _COLOR_OBJECT


# ─────────────────────────────────────────────────────────
# Collection Socket (unified)
# ─────────────────────────────────────────────────────────
class ExpCollectionSocket(bpy.types.NodeSocket):
    """Unified collection socket - works for both input and output."""
    bl_idname = "ExpCollectionSocketType"
    bl_label = "Collection"

    prop_name: bpy.props.StringProperty(
        default="",
        description="Node property to draw inline when not connected"
    )
    reaction_prop: bpy.props.StringProperty(
        default="",
        description="Reaction property to draw inline (for reaction nodes)"
    )
    interaction_prop: bpy.props.StringProperty(
        default="",
        description="Interaction property to draw inline (for trigger nodes)"
    )
    counter_prop: bpy.props.StringProperty(
        default="",
        description="Counter property to draw inline (for counter nodes)"
    )
    timer_prop: bpy.props.StringProperty(
        default="",
        description="Timer property to draw inline (for timer nodes)"
    )

    def draw(self, context, layout, node, text):
        if not self.is_output and not self.is_linked:
            if self.reaction_prop:
                r = _get_reaction_for_socket(node)
                if r and hasattr(r, self.reaction_prop):
                    layout.prop(r, self.reaction_prop, text="")
                    return
            if self.interaction_prop:
                inter = _get_interaction_for_socket(node)
                if inter and hasattr(inter, self.interaction_prop):
                    layout.prop(inter, self.interaction_prop, text="")
                    return
            if self.counter_prop:
                c = _get_counter_for_socket(node)
                if c and hasattr(c, self.counter_prop):
                    layout.prop(c, self.counter_prop, text="")
                    return
            if self.timer_prop:
                t = _get_timer_for_socket(node)
                if t and hasattr(t, self.timer_prop):
                    layout.prop(t, self.timer_prop, text="")
                    return
            if self.prop_name and hasattr(node, self.prop_name):
                layout.prop(node, self.prop_name, text="")
                return
        layout.label(text=text or ("Collection" if self.is_output else "Input"))

    def draw_color(self, context, node):
        if has_invalid_link(self):
            return INVALID_LINK_COLOR
        return _COLOR_COLLECTION


# ─────────────────────────────────────────────────────────
# Action Socket (unified)
# ─────────────────────────────────────────────────────────
class ExpActionSocket(bpy.types.NodeSocket):
    """Unified action socket - works for both input and output."""
    bl_idname = "ExpActionSocketType"
    bl_label = "Action"

    prop_name: bpy.props.StringProperty(
        default="",
        description="Node property to draw inline when not connected"
    )
    reaction_prop: bpy.props.StringProperty(
        default="",
        description="Reaction property to draw inline (for reaction nodes)"
    )
    interaction_prop: bpy.props.StringProperty(
        default="",
        description="Interaction property to draw inline (for trigger nodes)"
    )
    counter_prop: bpy.props.StringProperty(
        default="",
        description="Counter property to draw inline (for counter nodes)"
    )
    timer_prop: bpy.props.StringProperty(
        default="",
        description="Timer property to draw inline (for timer nodes)"
    )

    def draw(self, context, layout, node, text):
        if not self.is_output and not self.is_linked:
            if self.reaction_prop:
                r = _get_reaction_for_socket(node)
                if r and hasattr(r, self.reaction_prop):
                    layout.prop_search(r, self.reaction_prop, bpy.data, "actions", text=text or "")
                    return
            if self.interaction_prop:
                inter = _get_interaction_for_socket(node)
                if inter and hasattr(inter, self.interaction_prop):
                    layout.prop_search(inter, self.interaction_prop, bpy.data, "actions", text=text or "")
                    return
            if self.counter_prop:
                c = _get_counter_for_socket(node)
                if c and hasattr(c, self.counter_prop):
                    layout.prop_search(c, self.counter_prop, bpy.data, "actions", text=text or "")
                    return
            if self.timer_prop:
                t = _get_timer_for_socket(node)
                if t and hasattr(t, self.timer_prop):
                    layout.prop_search(t, self.timer_prop, bpy.data, "actions", text=text or "")
                    return
            if self.prop_name and hasattr(node, self.prop_name):
                layout.prop_search(node, self.prop_name, bpy.data, "actions", text=text or "")
                return
        layout.label(text=text or ("Action" if self.is_output else "Input"))

    def draw_color(self, context, node):
        if has_invalid_link(self):
            return INVALID_LINK_COLOR
        return _COLOR_ACTION


# ─────────────────────────────────────────────────────────
# Vector Socket (unified)
# ─────────────────────────────────────────────────────────
class ExpVectorSocket(bpy.types.NodeSocket):
    """Unified vector socket - works for both input and output."""
    bl_idname = "ExpVectorSocketType"
    bl_label = "Vector"

    prop_name: bpy.props.StringProperty(
        default="",
        description="Node property to draw inline when not connected"
    )
    reaction_prop: bpy.props.StringProperty(
        default="",
        description="Reaction property to draw inline (for reaction nodes)"
    )
    interaction_prop: bpy.props.StringProperty(
        default="",
        description="Interaction property to draw inline (for trigger nodes)"
    )
    counter_prop: bpy.props.StringProperty(
        default="",
        description="Counter property to draw inline (for counter nodes)"
    )
    timer_prop: bpy.props.StringProperty(
        default="",
        description="Timer property to draw inline (for timer nodes)"
    )

    def draw(self, context, layout, node, text):
        if not self.is_output and not self.is_linked:
            if self.reaction_prop:
                r = _get_reaction_for_socket(node)
                if r and hasattr(r, self.reaction_prop):
                    col = layout.column(align=True)
                    col.label(text=text)
                    col.prop(r, self.reaction_prop, index=0, text="X")
                    col.prop(r, self.reaction_prop, index=1, text="Y")
                    col.prop(r, self.reaction_prop, index=2, text="Z")
                    return
            if self.interaction_prop:
                inter = _get_interaction_for_socket(node)
                if inter and hasattr(inter, self.interaction_prop):
                    col = layout.column(align=True)
                    col.label(text=text)
                    col.prop(inter, self.interaction_prop, index=0, text="X")
                    col.prop(inter, self.interaction_prop, index=1, text="Y")
                    col.prop(inter, self.interaction_prop, index=2, text="Z")
                    return
            if self.counter_prop:
                c = _get_counter_for_socket(node)
                if c and hasattr(c, self.counter_prop):
                    col = layout.column(align=True)
                    col.label(text=text)
                    col.prop(c, self.counter_prop, index=0, text="X")
                    col.prop(c, self.counter_prop, index=1, text="Y")
                    col.prop(c, self.counter_prop, index=2, text="Z")
                    return
            if self.timer_prop:
                t = _get_timer_for_socket(node)
                if t and hasattr(t, self.timer_prop):
                    col = layout.column(align=True)
                    col.label(text=text)
                    col.prop(t, self.timer_prop, index=0, text="X")
                    col.prop(t, self.timer_prop, index=1, text="Y")
                    col.prop(t, self.timer_prop, index=2, text="Z")
                    return
            if self.prop_name and hasattr(node, self.prop_name):
                col = layout.column(align=True)
                col.label(text=text)
                col.prop(node, self.prop_name, index=0, text="X")
                col.prop(node, self.prop_name, index=1, text="Y")
                col.prop(node, self.prop_name, index=2, text="Z")
                return
        layout.label(text=text or ("Vector" if self.is_output else "Input"))

    def draw_color(self, context, node):
        if has_invalid_link(self):
            return INVALID_LINK_COLOR
        return _COLOR_VECTOR



# ═══════════════════════════════════════════════════════════
# DATA NODES - Core procedural elements
# Each node has input + output. If input is linked, use incoming value.
# If not linked, show manual value field.
# ═══════════════════════════════════════════════════════════


def _input_linked(node, socket_name):
    """Check if an input socket is linked."""
    inp = node.inputs.get(socket_name)
    return inp and inp.is_linked


def _resolve_upstream_float(node, socket_name):
    """Get float value from upstream node if linked."""
    inp = node.inputs.get(socket_name)
    if not inp or not inp.is_linked:
        return None
    link = inp.links[0]
    src_node = link.from_node
    # Multi-output support: check for socket-specific export first
    if hasattr(src_node, "export_float_by_socket"):
        return src_node.export_float_by_socket(link.from_socket.name)
    if hasattr(src_node, "export_float"):
        return src_node.export_float()
    return None


def _resolve_upstream_int(node, socket_name):
    """Get int value from upstream node if linked."""
    inp = node.inputs.get(socket_name)
    if not inp or not inp.is_linked:
        return None
    link = inp.links[0]
    src_node = link.from_node
    if hasattr(src_node, "export_int"):
        return src_node.export_int()
    return None


def _resolve_upstream_bool(node, socket_name):
    """Get bool value from upstream node if linked."""
    inp = node.inputs.get(socket_name)
    if not inp or not inp.is_linked:
        return None
    link = inp.links[0]
    src_node = link.from_node
    if hasattr(src_node, "export_bool"):
        return src_node.export_bool()
    return None


def _resolve_upstream_object(node, socket_name):
    """Get object from upstream node if linked."""
    inp = node.inputs.get(socket_name)
    if not inp or not inp.is_linked:
        return None
    link = inp.links[0]
    src_node = link.from_node
    if hasattr(src_node, "export_object"):
        return src_node.export_object()
    return None


def _resolve_upstream_collection(node, socket_name):
    """Get collection from upstream node if linked."""
    inp = node.inputs.get(socket_name)
    if not inp or not inp.is_linked:
        return None
    link = inp.links[0]
    src_node = link.from_node
    if hasattr(src_node, "export_collection"):
        return src_node.export_collection()
    return None


def _resolve_upstream_vector(node, socket_name):
    """Get vector from upstream node if linked."""
    inp = node.inputs.get(socket_name)
    if not inp or not inp.is_linked:
        return None
    link = inp.links[0]
    src_node = link.from_node
    if hasattr(src_node, "export_vector"):
        return src_node.export_vector()
    return None


def _resolve_upstream_action(node, socket_name):
    """Get action from upstream node if linked."""
    inp = node.inputs.get(socket_name)
    if not inp or not inp.is_linked:
        return None
    link = inp.links[0]
    src_node = link.from_node
    if hasattr(src_node, "export_action"):
        return src_node.export_action()
    return None


# ─────────────────────────────────────────────────────────
# Float Data Node
# ─────────────────────────────────────────────────────────
class FloatDataNode(_ExploratoryNodeOnly, Node):
    """Float value node - input or manual value, outputs to other nodes."""
    bl_idname = "FloatDataNodeType"
    bl_label = "Float"
    bl_icon = 'PREFERENCES'

    slot_uid: bpy.props.StringProperty(name="UID", default="")
    slot_name: bpy.props.StringProperty(
        name="Name",
        default="Float",
        update=lambda self, ctx: self._sync_name(ctx)
    )
    value: bpy.props.FloatProperty(
        name="Value",
        default=0.0,
        update=lambda self, ctx: self._sync_value(ctx)
    )

    def _ensure_uid(self, context):
        scn = context.scene if context else bpy.context.scene
        if not self.slot_uid or not float_slot_exists(self.slot_uid):
            self.slot_uid = create_float_slot(scn, name=self.slot_name or "Float")

    def _sync_name(self, context):
        scn = context.scene if context else bpy.context.scene
        coll = getattr(scn, "utility_floats", None)
        if coll:
            for it in coll:
                if getattr(it, "uid", "") == self.slot_uid:
                    it.name = self.slot_name or "Float"
                    break

    def _sync_value(self, context):
        if self.slot_uid:
            set_float(self.slot_uid, self.value)

    def init(self, context):
        self.width = 180
        self.inputs.new("ExpFloatSocketType", "Input")
        self.outputs.new("ExpFloatSocketType", "Value")
        self._ensure_uid(context)
        self._sync_value(context)

    def update(self):
        self._ensure_uid(bpy.context)

    def draw_buttons(self, context, layout):
        # Don't call _ensure_uid here - writing not allowed in draw context
        layout.prop(self, "slot_name", text="Name")
        # Only show value field if input not connected
        if not _input_linked(self, "Input"):
            layout.prop(self, "value", text="Value")

    def export_float(self):
        """API for other nodes to read this value."""
        # If input is linked, use upstream value
        upstream = _resolve_upstream_float(self, "Input")
        if upstream is not None:
            return upstream
        # Otherwise use stored/manual value
        has_val, val, _ = get_float(self.slot_uid)
        return val if has_val else self.value


# ─────────────────────────────────────────────────────────
# Integer Data Node
# ─────────────────────────────────────────────────────────
class IntDataNode(_ExploratoryNodeOnly, Node):
    """Integer value node - input or manual value, outputs to other nodes."""
    bl_idname = "IntDataNodeType"
    bl_label = "Integer"
    bl_icon = 'LINENUMBERS_ON'

    slot_uid: bpy.props.StringProperty(name="UID", default="")
    slot_name: bpy.props.StringProperty(
        name="Name",
        default="Integer",
        update=lambda self, ctx: self._sync_name(ctx)
    )
    value: bpy.props.IntProperty(
        name="Value",
        default=0,
        update=lambda self, ctx: self._sync_value(ctx)
    )

    def _ensure_uid(self, context):
        scn = context.scene if context else bpy.context.scene
        if not self.slot_uid or not int_slot_exists(self.slot_uid):
            self.slot_uid = create_int_slot(scn, name=self.slot_name or "Integer")

    def _sync_name(self, context):
        scn = context.scene if context else bpy.context.scene
        coll = getattr(scn, "utility_ints", None)
        if coll:
            for it in coll:
                if getattr(it, "uid", "") == self.slot_uid:
                    it.name = self.slot_name or "Integer"
                    break

    def _sync_value(self, context):
        if self.slot_uid:
            set_int(self.slot_uid, self.value)

    def init(self, context):
        self.width = 180
        self.inputs.new("ExpIntSocketType", "Input")
        self.outputs.new("ExpIntSocketType", "Value")
        self._ensure_uid(context)
        self._sync_value(context)

    def update(self):
        self._ensure_uid(bpy.context)

    def draw_buttons(self, context, layout):
        # Don't call _ensure_uid here - writing not allowed in draw context
        layout.prop(self, "slot_name", text="Name")
        if not _input_linked(self, "Input"):
            layout.prop(self, "value", text="Value")

    def export_int(self):
        """API for other nodes to read this value."""
        upstream = _resolve_upstream_int(self, "Input")
        if upstream is not None:
            return upstream
        has_val, val, _ = get_int(self.slot_uid)
        return val if has_val else self.value


# ─────────────────────────────────────────────────────────
# Boolean Data Node
# ─────────────────────────────────────────────────────────
class BoolDataNode(_ExploratoryNodeOnly, Node):
    """Boolean value node - input or manual value, outputs to other nodes."""
    bl_idname = "BoolDataNodeType"
    bl_label = "Boolean"
    bl_icon = 'CHECKBOX_HLT'

    slot_uid: bpy.props.StringProperty(name="UID", default="")
    slot_name: bpy.props.StringProperty(
        name="Name",
        default="Boolean",
        update=lambda self, ctx: self._sync_name(ctx)
    )
    value: bpy.props.BoolProperty(
        name="Value",
        default=False,
        update=lambda self, ctx: self._sync_value(ctx)
    )

    def _ensure_uid(self, context):
        scn = context.scene if context else bpy.context.scene
        if not self.slot_uid or not bool_slot_exists(self.slot_uid):
            self.slot_uid = create_bool_slot(scn, name=self.slot_name or "Boolean")

    def _sync_name(self, context):
        scn = context.scene if context else bpy.context.scene
        coll = getattr(scn, "utility_bools", None)
        if coll:
            for it in coll:
                if getattr(it, "uid", "") == self.slot_uid:
                    it.name = self.slot_name or "Boolean"
                    break

    def _sync_value(self, context):
        if self.slot_uid:
            set_bool(self.slot_uid, self.value)

    def init(self, context):
        self.width = 180
        self.inputs.new("ExpBoolSocketType", "Input")
        self.outputs.new("ExpBoolSocketType", "Value")
        self._ensure_uid(context)
        self._sync_value(context)

    def update(self):
        self._ensure_uid(bpy.context)

    def draw_buttons(self, context, layout):
        # Don't call _ensure_uid here - writing not allowed in draw context
        layout.prop(self, "slot_name", text="Name")
        if not _input_linked(self, "Input"):
            layout.prop(self, "value", text="Value")

    def export_bool(self):
        """API for other nodes to read this value."""
        upstream = _resolve_upstream_bool(self, "Input")
        if upstream is not None:
            return upstream
        has_val, val, _ = get_bool(self.slot_uid)
        return val if has_val else self.value

    def write_from_graph(self, value, *, timestamp=None):
        self.value = bool(value)
        if self.slot_uid:
            set_bool(self.slot_uid, bool(value))


# ─────────────────────────────────────────────────────────
# Object Data Node
# ─────────────────────────────────────────────────────────
class ObjectDataNode(_ExploratoryNodeOnly, Node):
    """Object reference node - input or manual selection, outputs to other nodes."""
    bl_idname = "ObjectDataNodeType"
    bl_label = "Object"
    bl_icon = 'OBJECT_DATA'

    slot_uid: bpy.props.StringProperty(name="UID", default="")
    slot_name: bpy.props.StringProperty(
        name="Name",
        default="Object",
        update=lambda self, ctx: self._sync_name(ctx)
    )
    target_object: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Object",
        update=lambda self, ctx: self._sync_value(ctx)
    )
    use_character: bpy.props.BoolProperty(
        name="Use Character",
        default=False,
        description="Use the scene's main character (scene.target_armature). "
                    "Enable this instead of picking the armature directly, since "
                    "the character is recreated at game start"
    )

    def _ensure_uid(self, context):
        scn = context.scene if context else bpy.context.scene
        if not self.slot_uid or not object_slot_exists(self.slot_uid):
            self.slot_uid = create_object_slot(scn, name=self.slot_name or "Object")

    def _sync_name(self, context):
        scn = context.scene if context else bpy.context.scene
        coll = getattr(scn, "utility_objects", None)
        if coll:
            for it in coll:
                if getattr(it, "uid", "") == self.slot_uid:
                    it.name = self.slot_name or "Object"
                    break

    def _sync_value(self, context):
        if self.slot_uid:
            set_object(self.slot_uid, self.target_object)

    def init(self, context):
        self.width = 200
        self.inputs.new("ExpObjectSocketType", "Input")
        self.outputs.new("ExpObjectSocketType", "Object")
        self._ensure_uid(context)
        self._sync_value(context)

    def update(self):
        self._ensure_uid(bpy.context)

    def draw_buttons(self, context, layout):
        # Don't call _ensure_uid here - writing not allowed in draw context
        layout.prop(self, "slot_name", text="Name")
        if not _input_linked(self, "Input"):
            layout.prop(self, "use_character", text="Use Character")
            if self.use_character:
                # Show indicator that character will be used
                box = layout.box()
                box.label(text="→ Character", icon='ARMATURE_DATA')
            else:
                layout.prop(self, "target_object", text="")

    def export_object(self):
        """API for other nodes to read this object."""
        upstream = _resolve_upstream_object(self, "Input")
        if upstream is not None:
            return upstream
        # Use character if toggle is enabled
        if self.use_character:
            scn = bpy.context.scene
            return getattr(scn, 'target_armature', None)
        has_val, obj, _ = get_object(self.slot_uid)
        return obj if has_val else self.target_object

    def export_object_name(self):
        """API for engine worker (string-based)."""
        # Use character if toggle is enabled
        if self.use_character:
            scn = bpy.context.scene
            char = getattr(scn, 'target_armature', None)
            return char.name if char else ""
        obj = self.export_object()
        return obj.name if obj else ""

    def write_from_graph(self, obj_or_name, *, timestamp=None):
        if isinstance(obj_or_name, str):
            if self.slot_uid:
                set_object(self.slot_uid, obj_or_name)
        elif obj_or_name is not None:
            self.target_object = obj_or_name
            if self.slot_uid:
                set_object(self.slot_uid, obj_or_name)
        else:
            self.target_object = None
            if self.slot_uid:
                set_object(self.slot_uid, None)


# ─────────────────────────────────────────────────────────
# Collection Data Node
# ─────────────────────────────────────────────────────────
class CollectionDataNode(_ExploratoryNodeOnly, Node):
    """Collection reference node - input or manual selection, outputs to other nodes."""
    bl_idname = "CollectionDataNodeType"
    bl_label = "Collection"
    bl_icon = 'OUTLINER_COLLECTION'

    slot_uid: bpy.props.StringProperty(name="UID", default="")
    slot_name: bpy.props.StringProperty(
        name="Name",
        default="Collection",
        update=lambda self, ctx: self._sync_name(ctx)
    )
    target_collection: bpy.props.PointerProperty(
        type=bpy.types.Collection,
        name="Collection",
        update=lambda self, ctx: self._sync_value(ctx)
    )

    def _ensure_uid(self, context):
        scn = context.scene if context else bpy.context.scene
        if not self.slot_uid or not collection_slot_exists(self.slot_uid):
            self.slot_uid = create_collection_slot(scn, name=self.slot_name or "Collection")

    def _sync_name(self, context):
        scn = context.scene if context else bpy.context.scene
        coll = getattr(scn, "utility_collections", None)
        if coll:
            for it in coll:
                if getattr(it, "uid", "") == self.slot_uid:
                    it.name = self.slot_name or "Collection"
                    break

    def _sync_value(self, context):
        if self.slot_uid:
            set_collection(self.slot_uid, self.target_collection)

    def init(self, context):
        self.width = 200
        self.inputs.new("ExpCollectionSocketType", "Input")
        self.outputs.new("ExpCollectionSocketType", "Collection")
        self._ensure_uid(context)
        self._sync_value(context)

    def update(self):
        self._ensure_uid(bpy.context)

    def draw_buttons(self, context, layout):
        # Don't call _ensure_uid here - writing not allowed in draw context
        layout.prop(self, "slot_name", text="Name")
        if not _input_linked(self, "Input"):
            layout.prop(self, "target_collection", text="")

    def export_collection(self):
        """API for other nodes to read this collection."""
        upstream = _resolve_upstream_collection(self, "Input")
        if upstream is not None:
            return upstream
        has_val, coll, _ = get_collection(self.slot_uid)
        return coll if has_val else self.target_collection

    def export_collection_name(self):
        """API for engine worker (string-based)."""
        coll = self.export_collection()
        return coll.name if coll else ""


# ─────────────────────────────────────────────────────────
# Float Vector Data Node
# ─────────────────────────────────────────────────────────
class FloatVectorDataNode(_ExploratoryNodeOnly, Node):
    """Float Vector value node - input or manual value, outputs to other nodes."""
    bl_idname = "FloatVectorDataNodeType"
    bl_label = "Float Vector"
    bl_icon = 'EMPTY_SINGLE_ARROW'

    slot_uid: bpy.props.StringProperty(name="UID", default="")
    slot_name: bpy.props.StringProperty(
        name="Name",
        default="Vector",
        update=lambda self, ctx: self._sync_name(ctx)
    )
    value: bpy.props.FloatVectorProperty(
        name="Value",
        size=3,
        subtype='TRANSLATION',
        default=(0.0, 0.0, 0.0),
        update=lambda self, ctx: self._sync_value(ctx)
    )

    def _ensure_uid(self, context):
        scn = context.scene if context else bpy.context.scene
        if not self.slot_uid or not slot_exists(self.slot_uid):
            self.slot_uid = create_floatvec_slot(scn, name=self.slot_name or "Vector")

    def _sync_name(self, context):
        scn = context.scene if context else bpy.context.scene
        coll = getattr(scn, "utility_float_vectors", None)
        if coll:
            for it in coll:
                if getattr(it, "uid", "") == self.slot_uid:
                    it.name = self.slot_name or "Vector"
                    break

    def _sync_value(self, context):
        if self.slot_uid:
            set_floatvec(self.slot_uid, self.value)

    def init(self, context):
        self.width = 200
        self.inputs.new("ExpVectorSocketType", "Input")
        self.outputs.new("ExpVectorSocketType", "Vector")
        self._ensure_uid(context)
        self._sync_value(context)

    def update(self):
        self._ensure_uid(bpy.context)

    def draw_buttons(self, context, layout):
        # Don't call _ensure_uid here - writing not allowed in draw context
        layout.prop(self, "slot_name", text="Name")
        if not _input_linked(self, "Input"):
            col = layout.column(align=True)
            col.prop(self, "value", text="")

    def export_vector(self):
        """API for other nodes to read this vector."""
        upstream = _resolve_upstream_vector(self, "Input")
        if upstream is not None:
            return upstream
        has_val, vec, _ = get_floatvec(self.slot_uid)
        return vec if has_val else tuple(self.value)

    def write_from_graph(self, vec, *, timestamp=None):
        v = (float(vec[0]), float(vec[1]), float(vec[2]))
        self.value = v
        if self.slot_uid:
            set_floatvec(self.slot_uid, v)


# ─────────────────────────────────────────────────────────
# Action Data Node
# ─────────────────────────────────────────────────────────
class ActionDataNode(_ExploratoryNodeOnly, Node):
    """Action reference node - input or manual selection, outputs to other nodes."""
    bl_idname = "ActionDataNodeType"
    bl_label = "Action"
    bl_icon = 'ACTION'

    slot_uid: bpy.props.StringProperty(name="UID", default="")
    slot_name: bpy.props.StringProperty(
        name="Name",
        default="Action",
        update=lambda self, ctx: self._sync_name(ctx)
    )
    target_action: bpy.props.PointerProperty(
        type=bpy.types.Action,
        name="Action",
        update=lambda self, ctx: self._sync_value(ctx)
    )

    def _ensure_uid(self, context):
        scn = context.scene if context else bpy.context.scene
        if not self.slot_uid or not action_slot_exists(self.slot_uid):
            self.slot_uid = create_action_slot(scn, name=self.slot_name or "Action")

    def _sync_name(self, context):
        scn = context.scene if context else bpy.context.scene
        coll = getattr(scn, "utility_actions", None)
        if coll:
            for it in coll:
                if getattr(it, "uid", "") == self.slot_uid:
                    it.name = self.slot_name or "Action"
                    break

    def _sync_value(self, context):
        if self.slot_uid:
            set_action(self.slot_uid, self.target_action)

    def init(self, context):
        self.width = 200
        self.inputs.new("ExpActionSocketType", "Input")
        self.outputs.new("ExpActionSocketType", "Action")
        self._ensure_uid(context)
        self._sync_value(context)

    def update(self):
        self._ensure_uid(bpy.context)

    def draw_buttons(self, context, layout):
        # Don't call _ensure_uid here - writing not allowed in draw context
        layout.prop(self, "slot_name", text="Name")
        if not _input_linked(self, "Input"):
            layout.prop(self, "target_action", text="")

    def export_action(self):
        """API for other nodes to read this action."""
        upstream = _resolve_upstream_action(self, "Input")
        if upstream is not None:
            return upstream
        has_val, action, _ = get_action(self.slot_uid)
        return action if has_val else self.target_action

    def export_action_name(self):
        """API for engine worker (string-based)."""
        action = self.export_action()
        return action.name if action else ""


# ─────────────────────────────────────────────────────────
# Compare Node
# ─────────────────────────────────────────────────────────

_NUMERIC_TYPES = {'FLOAT', 'INT'}

_COMPARE_NUMERIC_OPS = [
    ('EQ', "==", "Equal"),
    ('NE', "!=", "Not equal"),
    ('LT', "<", "Less than"),
    ('LE', "<=", "Less than or equal"),
    ('GE', ">=", "Greater than or equal"),
    ('GT', ">", "Greater than"),
]

def _get_operation_items(self, context):
    if self.data_type in _NUMERIC_TYPES:
        return _COMPARE_NUMERIC_OPS
    return EQUALITY_ONLY_OPERATORS

class CompareNode(_ExploratoryNodeOnly, Node):
    """Compare two values and output a boolean result."""
    bl_idname = "CompareNodeType"
    bl_label  = "Compare"
    bl_icon   = 'NONE'

    data_type: bpy.props.EnumProperty(
        name="Data Type",
        items=[
            ('FLOAT',      "Float",      ""),
            ('INT',        "Integer",    ""),
            ('BOOL',       "Boolean",    ""),
            ('OBJECT',     "Object",     ""),
            ('COLLECTION', "Collection", ""),
            ('ACTION',     "Action",     ""),
        ],
        default='FLOAT',
        update=lambda self, ctx: self._on_data_type_changed(ctx),
    )

    operation: bpy.props.EnumProperty(
        name="Operation",
        items=_get_operation_items,
    )

    epsilon: bpy.props.FloatProperty(
        name="Epsilon",
        description="Tolerance for float comparison",
        default=0.0001,
        min=0.0,
        soft_max=1.0,
        precision=6,
    )

    # Manual fallback properties (used when input sockets are not connected)
    value_a_float: bpy.props.FloatProperty(name="A")
    value_b_float: bpy.props.FloatProperty(name="B")
    value_a_int: bpy.props.IntProperty(name="A")
    value_b_int: bpy.props.IntProperty(name="B")
    value_a_bool: bpy.props.BoolProperty(name="A")
    value_b_bool: bpy.props.BoolProperty(name="B")
    value_a_object: bpy.props.PointerProperty(type=bpy.types.Object, name="A")
    value_b_object: bpy.props.PointerProperty(type=bpy.types.Object, name="B")
    value_a_collection: bpy.props.PointerProperty(type=bpy.types.Collection, name="A")
    value_b_collection: bpy.props.PointerProperty(type=bpy.types.Collection, name="B")
    value_a_action: bpy.props.PointerProperty(type=bpy.types.Action, name="A")
    value_b_action: bpy.props.PointerProperty(type=bpy.types.Action, name="B")

    def _on_data_type_changed(self, context):
        self._rebuild_inputs(context)
        if self.data_type not in _NUMERIC_TYPES:
            if self.operation not in ('EQ', 'NE'):
                self.operation = 'EQ'

    def _rebuild_inputs(self, context):
        type_to_socket = {
            'FLOAT':      "ExpFloatSocketType",
            'INT':        "ExpIntSocketType",
            'BOOL':       "ExpBoolSocketType",
            'OBJECT':     "ExpObjectSocketType",
            'COLLECTION': "ExpCollectionSocketType",
            'ACTION':     "ExpActionSocketType",
        }
        self.inputs.clear()
        socket_type = type_to_socket[self.data_type]
        dt_lower = self.data_type.lower()

        sock_a = self.inputs.new(socket_type, "A")
        sock_b = self.inputs.new(socket_type, "B")

        sock_a.prop_name = f"value_a_{dt_lower}"
        sock_b.prop_name = f"value_b_{dt_lower}"

        if self.data_type == 'OBJECT':
            sock_a.use_prop_search = True
            sock_b.use_prop_search = True

    def init(self, context):
        self.width = 160
        self._rebuild_inputs(context)
        self.outputs.new("ExpBoolSocketType", "Result")

    def draw_buttons(self, context, layout):
        layout.prop(self, "data_type", text="")
        layout.prop(self, "operation", text="")
        if self.data_type == 'FLOAT':
            layout.prop(self, "epsilon")

    def _compare_float(self, a, b, op):
        eps = self.epsilon
        if op == 'EQ': return abs(a - b) <= eps
        if op == 'NE': return abs(a - b) > eps
        if op == 'LT': return a < b
        if op == 'LE': return a <= b
        if op == 'GE': return a >= b
        if op == 'GT': return a > b
        return False

    @staticmethod
    def _compare_numeric(a, b, op):
        if op == 'EQ': return a == b
        if op == 'NE': return a != b
        if op == 'LT': return a < b
        if op == 'LE': return a <= b
        if op == 'GE': return a >= b
        if op == 'GT': return a > b
        return False

    def export_bool(self):
        dt = self.data_type
        op = self.operation

        if dt == 'FLOAT':
            a = _resolve_upstream_float(self, "A")
            if a is None:
                a = self.value_a_float
            b = _resolve_upstream_float(self, "B")
            if b is None:
                b = self.value_b_float
            return self._compare_float(a, b, op)

        elif dt == 'INT':
            a = _resolve_upstream_int(self, "A")
            if a is None:
                a = self.value_a_int
            b = _resolve_upstream_int(self, "B")
            if b is None:
                b = self.value_b_int
            return self._compare_numeric(a, b, op)

        elif dt == 'BOOL':
            a = _resolve_upstream_bool(self, "A")
            if a is None:
                a = self.value_a_bool
            b = _resolve_upstream_bool(self, "B")
            if b is None:
                b = self.value_b_bool
            return (a == b) if op == 'EQ' else (a != b)

        elif dt == 'OBJECT':
            a = _resolve_upstream_object(self, "A")
            if a is None:
                a = self.value_a_object
            b = _resolve_upstream_object(self, "B")
            if b is None:
                b = self.value_b_object
            return (a is b) if op == 'EQ' else (a is not b)

        elif dt == 'COLLECTION':
            a = _resolve_upstream_collection(self, "A")
            if a is None:
                a = self.value_a_collection
            b = _resolve_upstream_collection(self, "B")
            if b is None:
                b = self.value_b_collection
            return (a is b) if op == 'EQ' else (a is not b)

        elif dt == 'ACTION':
            a = _resolve_upstream_action(self, "A")
            if a is None:
                a = self.value_a_action
            b = _resolve_upstream_action(self, "B")
            if b is None:
                b = self.value_b_action
            return (a is b) if op == 'EQ' else (a is not b)

        return False


# ─────────────────────────────────────────────────────────
# Split Vector Node
# ─────────────────────────────────────────────────────────
class SplitVectorNode(_ExploratoryNodeOnly, Node):
    """Split a vector into its X, Y, Z float components."""
    bl_idname = "SplitVectorNodeType"
    bl_label  = "Split Vector"
    bl_icon   = 'NONE'

    value_vector: bpy.props.FloatVectorProperty(name="Vector", size=3)

    def init(self, context):
        self.width = 150
        sock = self.inputs.new("ExpVectorSocketType", "Vector")
        sock.prop_name = "value_vector"
        self.outputs.new("ExpFloatSocketType", "X")
        self.outputs.new("ExpFloatSocketType", "Y")
        self.outputs.new("ExpFloatSocketType", "Z")

    def _get_vector(self):
        upstream = _resolve_upstream_vector(self, "Vector")
        if upstream is not None:
            return upstream
        return tuple(self.value_vector)

    def export_float_by_socket(self, socket_name):
        """Multi-output export: returns the component matching the output socket."""
        vec = self._get_vector()
        if socket_name == "X": return vec[0]
        if socket_name == "Y": return vec[1]
        if socket_name == "Z": return vec[2]
        return 0.0

    def export_float(self):
        """Fallback: returns X component."""
        return self._get_vector()[0]


# ─────────────────────────────────────────────────────────
# Combine Vector Node
# ─────────────────────────────────────────────────────────
class CombineVectorNode(_ExploratoryNodeOnly, Node):
    """Combine X, Y, Z float values into a vector."""
    bl_idname = "CombineVectorNodeType"
    bl_label  = "Combine Vector"
    bl_icon   = 'NONE'

    value_x: bpy.props.FloatProperty(name="X", default=0.0)
    value_y: bpy.props.FloatProperty(name="Y", default=0.0)
    value_z: bpy.props.FloatProperty(name="Z", default=0.0)

    def init(self, context):
        self.width = 150
        sx = self.inputs.new("ExpFloatSocketType", "X")
        sx.prop_name = "value_x"
        sy = self.inputs.new("ExpFloatSocketType", "Y")
        sy.prop_name = "value_y"
        sz = self.inputs.new("ExpFloatSocketType", "Z")
        sz.prop_name = "value_z"
        self.outputs.new("ExpVectorSocketType", "Vector")

    def export_vector(self):
        x = _resolve_upstream_float(self, "X")
        if x is None:
            x = self.value_x
        y = _resolve_upstream_float(self, "Y")
        if y is None:
            y = self.value_y
        z = _resolve_upstream_float(self, "Z")
        if z is None:
            z = self.value_z
        return (x, y, z)


# ─────────────────────────────────────────────────────────
# Float Math Node
# ─────────────────────────────────────────────────────────

_UNARY_FLOAT_OPS = {'ABS', 'NEGATE', 'SQRT', 'FLOOR', 'CEIL', 'ROUND'}

class FloatMathNode(_ExploratoryNodeOnly, Node):
    """Perform a math operation on one or two float values."""
    bl_idname = "FloatMathNodeType"
    bl_label  = "Float Math"
    bl_icon   = 'NONE'

    operation: bpy.props.EnumProperty(
        name="Operation",
        items=[
            ('ADD',      "Add",         "A + B"),
            ('SUBTRACT', "Subtract",    "A − B"),
            ('MULTIPLY', "Multiply",    "A × B"),
            ('DIVIDE',   "Divide",      "A / B (safe)"),
            ('POWER',    "Power",       "A ^ B"),
            ('MODULO',   "Modulo",      "A mod B"),
            ('MIN',      "Minimum",     "min(A, B)"),
            ('MAX',      "Maximum",     "max(A, B)"),
            ('ABS',      "Absolute",    "|A|"),
            ('NEGATE',   "Negate",      "−A"),
            ('SQRT',     "Square Root", "√A"),
            ('FLOOR',    "Floor",       "Round A down"),
            ('CEIL',     "Ceiling",     "Round A up"),
            ('ROUND',    "Round",       "Round A"),
        ],
        default='ADD',
        update=lambda self, ctx: self._rebuild_inputs(ctx),
    )

    value_a: bpy.props.FloatProperty(name="A", default=0.0)
    value_b: bpy.props.FloatProperty(name="B", default=0.0)

    def _rebuild_inputs(self, context):
        self.inputs.clear()
        sock_a = self.inputs.new("ExpFloatSocketType", "A")
        sock_a.prop_name = "value_a"
        if self.operation not in _UNARY_FLOAT_OPS:
            sock_b = self.inputs.new("ExpFloatSocketType", "B")
            sock_b.prop_name = "value_b"

    def init(self, context):
        self.width = 160
        self._rebuild_inputs(context)
        self.outputs.new("ExpFloatSocketType", "Result")

    def draw_buttons(self, context, layout):
        layout.prop(self, "operation", text="")

    def export_float(self):
        a = _resolve_upstream_float(self, "A")
        if a is None:
            a = self.value_a

        op = self.operation

        # Unary operations
        if op == 'ABS':    return abs(a)
        if op == 'NEGATE': return -a
        if op == 'SQRT':   return math.sqrt(a) if a >= 0.0 else 0.0
        if op == 'FLOOR':  return math.floor(a)
        if op == 'CEIL':   return math.ceil(a)
        if op == 'ROUND':  return round(a)

        # Binary operations
        b = _resolve_upstream_float(self, "B")
        if b is None:
            b = self.value_b

        if op == 'ADD':      return a + b
        if op == 'SUBTRACT': return a - b
        if op == 'MULTIPLY': return a * b
        if op == 'DIVIDE':   return a / b if b != 0.0 else 0.0
        if op == 'POWER':    return a ** b
        if op == 'MODULO':   return a % b if b != 0.0 else 0.0
        if op == 'MIN':      return min(a, b)
        if op == 'MAX':      return max(a, b)
        return 0.0


# ─────────────────────────────────────────────────────────
# Integer Math Node
# ─────────────────────────────────────────────────────────

_UNARY_INT_OPS = {'ABS', 'NEGATE'}

class IntMathNode(_ExploratoryNodeOnly, Node):
    """Perform a math operation on one or two integer values."""
    bl_idname = "IntMathNodeType"
    bl_label  = "Int Math"
    bl_icon   = 'NONE'

    operation: bpy.props.EnumProperty(
        name="Operation",
        items=[
            ('ADD',      "Add",      "A + B"),
            ('SUBTRACT', "Subtract", "A − B"),
            ('MULTIPLY', "Multiply", "A × B"),
            ('DIVIDE',   "Divide",   "A / B (integer)"),
            ('POWER',    "Power",    "A ^ B"),
            ('MODULO',   "Modulo",   "A mod B"),
            ('MIN',      "Minimum",  "min(A, B)"),
            ('MAX',      "Maximum",  "max(A, B)"),
            ('ABS',      "Absolute", "|A|"),
            ('NEGATE',   "Negate",   "−A"),
        ],
        default='ADD',
        update=lambda self, ctx: self._rebuild_inputs(ctx),
    )

    value_a: bpy.props.IntProperty(name="A", default=0)
    value_b: bpy.props.IntProperty(name="B", default=0)

    def _rebuild_inputs(self, context):
        self.inputs.clear()
        sock_a = self.inputs.new("ExpIntSocketType", "A")
        sock_a.prop_name = "value_a"
        if self.operation not in _UNARY_INT_OPS:
            sock_b = self.inputs.new("ExpIntSocketType", "B")
            sock_b.prop_name = "value_b"

    def init(self, context):
        self.width = 160
        self._rebuild_inputs(context)
        self.outputs.new("ExpIntSocketType", "Result")

    def draw_buttons(self, context, layout):
        layout.prop(self, "operation", text="")

    def export_int(self):
        a = _resolve_upstream_int(self, "A")
        if a is None:
            a = self.value_a

        op = self.operation

        # Unary operations
        if op == 'ABS':    return abs(a)
        if op == 'NEGATE': return -a

        # Binary operations
        b = _resolve_upstream_int(self, "B")
        if b is None:
            b = self.value_b

        if op == 'ADD':      return a + b
        if op == 'SUBTRACT': return a - b
        if op == 'MULTIPLY': return a * b
        if op == 'DIVIDE':   return a // b if b != 0 else 0
        if op == 'POWER':    return int(a ** b)
        if op == 'MODULO':   return a % b if b != 0 else 0
        if op == 'MIN':      return min(a, b)
        if op == 'MAX':      return max(a, b)
        return 0


# ─────────────────────────────────────────────────────────
# Random Float Node
# ─────────────────────────────────────────────────────────
class RandomFloatNode(_ExploratoryNodeOnly, Node):
    """Generate a random float between Min and Max each evaluation."""
    bl_idname = "RandomFloatNodeType"
    bl_label  = "Random Float"
    bl_icon   = 'NONE'

    value_min: bpy.props.FloatProperty(name="Min", default=0.0)
    value_max: bpy.props.FloatProperty(name="Max", default=1.0)
    use_seed: bpy.props.BoolProperty(
        name="Use Seed",
        description="Use a fixed seed for repeatable results",
        default=False,
    )
    seed: bpy.props.IntProperty(
        name="Seed",
        default=0,
        min=0,
    )

    def init(self, context):
        self.width = 160
        s_min = self.inputs.new("ExpFloatSocketType", "Min")
        s_min.prop_name = "value_min"
        s_max = self.inputs.new("ExpFloatSocketType", "Max")
        s_max.prop_name = "value_max"
        self.outputs.new("ExpFloatSocketType", "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "use_seed")
        if self.use_seed:
            layout.prop(self, "seed")

    def export_float(self):
        lo = _resolve_upstream_float(self, "Min")
        if lo is None:
            lo = self.value_min
        hi = _resolve_upstream_float(self, "Max")
        if hi is None:
            hi = self.value_max
        if lo > hi:
            lo, hi = hi, lo
        if self.use_seed:
            rng = random.Random(self.seed)
            return rng.uniform(lo, hi)
        return random.uniform(lo, hi)


# ─────────────────────────────────────────────────────────
# Random Integer Node
# ─────────────────────────────────────────────────────────
class RandomIntNode(_ExploratoryNodeOnly, Node):
    """Generate a random integer between Min and Max (inclusive) each evaluation."""
    bl_idname = "RandomIntNodeType"
    bl_label  = "Random Int"
    bl_icon   = 'NONE'

    value_min: bpy.props.IntProperty(name="Min", default=0)
    value_max: bpy.props.IntProperty(name="Max", default=10)
    use_seed: bpy.props.BoolProperty(
        name="Use Seed",
        description="Use a fixed seed for repeatable results",
        default=False,
    )
    seed: bpy.props.IntProperty(
        name="Seed",
        default=0,
        min=0,
    )

    def init(self, context):
        self.width = 160
        s_min = self.inputs.new("ExpIntSocketType", "Min")
        s_min.prop_name = "value_min"
        s_max = self.inputs.new("ExpIntSocketType", "Max")
        s_max.prop_name = "value_max"
        self.outputs.new("ExpIntSocketType", "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "use_seed")
        if self.use_seed:
            layout.prop(self, "seed")

    def export_int(self):
        lo = _resolve_upstream_int(self, "Min")
        if lo is None:
            lo = self.value_min
        hi = _resolve_upstream_int(self, "Max")
        if hi is None:
            hi = self.value_max
        if lo > hi:
            lo, hi = hi, lo
        if self.use_seed:
            rng = random.Random(self.seed)
            return rng.randint(lo, hi)
        return random.randint(lo, hi)
