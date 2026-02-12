# File: Exp_Nodes/base_nodes.py
import bpy

# ─────────────────────────────────────────────────────────
# Restrict ALL Exploratory nodes to ONLY our node tree type
# ─────────────────────────────────────────────────────────
class _ExploratoryNodeOnly:
    """
    Mixin to gate nodes to ExploratoryNodesTreeType only.

    We use both:
      • bl_tree -> filters compatibility so nodes don't appear in other trees' add menus
      • poll    -> extra safety if someone tries to instantiate anyway
    """
    bl_tree = 'ExploratoryNodesTreeType'

    @classmethod
    def poll(cls, ntree):
        return bool(ntree) and getattr(ntree, "bl_idname", "") == "ExploratoryNodesTreeType"


class TriggerNodeBase(_ExploratoryNodeOnly, bpy.types.Node):
    bl_label = "Trigger Node Base"

    def execute_trigger(self, context):
        pass


class ReactionNodeBase(_ExploratoryNodeOnly, bpy.types.Node):
    """Base class for reaction nodes."""
    bl_label = "Reaction Node Base"

    def execute_reaction(self, context):
        pass


# ─────────────────────────────────────────────────────────
# Socket type-compatibility validation
# ─────────────────────────────────────────────────────────
# Sockets within the same group can connect; cross-group = invalid.

_SOCKET_TYPE_GROUP = {
    # Flow chain (trigger → reaction → reaction)
    "TriggerOutputSocketType":          "FLOW",
    "ReactionTriggerInputSocketType":   "FLOW",
    "ReactionOutputSocketType":         "FLOW",

    # Bool
    "ExpBoolSocketType":                "BOOL",
    "DynamicBoolInputSocketType":       "BOOL",
    "TriggerInputSocketType":           "BOOL",

    # Float
    "ExpFloatSocketType":               "FLOAT",
    "DynamicFloatInputSocketType":      "FLOAT",

    # Integer
    "ExpIntSocketType":                 "INT",

    # Object
    "ExpObjectSocketType":              "OBJECT",
    "DynamicObjectInputSocketType":     "OBJECT",

    # Collection
    "ExpCollectionSocketType":          "COLLECTION",

    # Action
    "ExpActionSocketType":              "ACTION",
    "DynamicActionInputSocketType":     "ACTION",

    # Vector
    "ExpVectorSocketType":              "VECTOR",
}

INVALID_LINK_COLOR = (0.92, 0.18, 0.18, 1.0)  # Red


def has_invalid_link(socket) -> bool:
    """Return True if any link on *socket* connects to an incompatible type."""
    try:
        links = getattr(socket, "links", None)
        if not links:
            return False

        my_group = _SOCKET_TYPE_GROUP.get(getattr(socket, "bl_idname", ""))
        if my_group is None:
            return False

        for lk in links:
            other = getattr(lk, "to_socket", None) if socket.is_output \
                else getattr(lk, "from_socket", None)
            if other is None:
                continue
            other_group = _SOCKET_TYPE_GROUP.get(
                getattr(other, "bl_idname", ""))
            if other_group is not None and my_group != other_group:
                return True
    except Exception:
        pass
    return False

