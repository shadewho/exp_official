# Exploratory/Exp_Nodes/utility_nodes.py
import bpy
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
from .base_nodes import _ExploratoryNodeOnly

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
_COLOR_INT        = (0.35, 0.55, 0.80, 1.0)  # Blue
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

    def draw(self, context, layout, node, text):
        if not self.is_output and not self.is_linked:
            # Try reaction property first (for reaction nodes)
            if self.reaction_prop:
                r = _get_reaction_for_socket(node)
                if r and hasattr(r, self.reaction_prop):
                    layout.prop(r, self.reaction_prop, text=text or "")
                    return
            # Then try node property
            if self.prop_name and hasattr(node, self.prop_name):
                layout.prop(node, self.prop_name, text=text or "")
                return
        layout.label(text=text or ("Float" if self.is_output else "Input"))

    def draw_color(self, context, node):
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

    def draw(self, context, layout, node, text):
        if not self.is_output and not self.is_linked:
            if self.reaction_prop:
                r = _get_reaction_for_socket(node)
                if r and hasattr(r, self.reaction_prop):
                    layout.prop(r, self.reaction_prop, text=text or "")
                    return
            if self.prop_name and hasattr(node, self.prop_name):
                layout.prop(node, self.prop_name, text=text or "")
                return
        layout.label(text=text or ("Integer" if self.is_output else "Input"))

    def draw_color(self, context, node):
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

    def draw(self, context, layout, node, text):
        if not self.is_output and not self.is_linked:
            if self.reaction_prop:
                r = _get_reaction_for_socket(node)
                if r and hasattr(r, self.reaction_prop):
                    layout.prop(r, self.reaction_prop, text=text or "")
                    return
            if self.prop_name and hasattr(node, self.prop_name):
                layout.prop(node, self.prop_name, text=text or "")
                return
        layout.label(text=text or ("Result" if self.is_output else "Input"))

    def draw_color(self, context, node):
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
            if self.prop_name and hasattr(node, self.prop_name):
                layout.prop(node, self.prop_name, text="")
                return
        layout.label(text=text or ("Object" if self.is_output else "Input"))

    def draw_color(self, context, node):
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

    def draw(self, context, layout, node, text):
        if not self.is_output and not self.is_linked:
            if self.reaction_prop:
                r = _get_reaction_for_socket(node)
                if r and hasattr(r, self.reaction_prop):
                    layout.prop(r, self.reaction_prop, text="")
                    return
            if self.prop_name and hasattr(node, self.prop_name):
                layout.prop(node, self.prop_name, text="")
                return
        layout.label(text=text or ("Collection" if self.is_output else "Input"))

    def draw_color(self, context, node):
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

    def draw(self, context, layout, node, text):
        if not self.is_output and not self.is_linked:
            if self.reaction_prop:
                r = _get_reaction_for_socket(node)
                if r and hasattr(r, self.reaction_prop):
                    layout.prop_search(r, self.reaction_prop, bpy.data, "actions", text=text or "")
                    return
            if self.prop_name and hasattr(node, self.prop_name):
                layout.prop_search(node, self.prop_name, bpy.data, "actions", text=text or "")
                return
        layout.label(text=text or ("Action" if self.is_output else "Input"))

    def draw_color(self, context, node):
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

    def draw(self, context, layout, node, text):
        if not self.is_output and not self.is_linked:
            if self.reaction_prop:
                r = _get_reaction_for_socket(node)
                if r and hasattr(r, self.reaction_prop):
                    # Draw with X, Y, Z labels in a column
                    col = layout.column(align=True)
                    col.prop(r, self.reaction_prop, index=0, text="X")
                    col.prop(r, self.reaction_prop, index=1, text="Y")
                    col.prop(r, self.reaction_prop, index=2, text="Z")
                    return
            if self.prop_name and hasattr(node, self.prop_name):
                # Draw with X, Y, Z labels in a column
                col = layout.column(align=True)
                col.prop(node, self.prop_name, index=0, text="X")
                col.prop(node, self.prop_name, index=1, text="Y")
                col.prop(node, self.prop_name, index=2, text="Z")
                return
        layout.label(text=text or ("Vector" if self.is_output else "Input"))

    def draw_color(self, context, node):
        return _COLOR_VECTOR


# ═══════════════════════════════════════════════════════════
# LEGACY SOCKET ALIASES (for backwards compatibility)
# These just reference the unified types
# ═══════════════════════════════════════════════════════════
# Note: Old nodes using these will need updating, but the
# bl_idname aliases help with transition

FloatInputSocket = ExpFloatSocket
FloatOutputSocket = ExpFloatSocket
IntInputSocket = ExpIntSocket
IntOutputSocket = ExpIntSocket
BoolInputSocket = ExpBoolSocket
BoolOutputSocket = ExpBoolSocket
ObjectInputSocket = ExpObjectSocket
ObjectOutputSocket = ExpObjectSocket
CollectionInputSocket = ExpCollectionSocket
CollectionOutputSocket = ExpCollectionSocket
ActionInputSocket = ExpActionSocket
ActionOutputSocket = ExpActionSocket
FloatVectorInputSocket = ExpVectorSocket
FloatVectorOutputSocket = ExpVectorSocket


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
    # Check for export method
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
