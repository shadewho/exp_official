# Exploratory/Exp_Nodes/utility_nodes.py
import bpy
from bpy.types import Node

# Store API (central registry)
from ..Exp_Game.props_and_utils.exp_utility_store import (
    create_floatvec_slot, set_floatvec, get_floatvec, slot_exists
)

EXPL_TREE_ID = "ExploratoryNodesTreeType"


# ─────────────────────────────────────────────────────────
# Gate to our custom node tree
# ─────────────────────────────────────────────────────────
class _ExploratoryNodeOnly:
    bl_tree = EXPL_TREE_ID

    @classmethod
    def poll(cls, ntree):
        return bool(ntree) and getattr(ntree, "bl_idname", "") == EXPL_TREE_ID


# ─────────────────────────────────────────────────────────
# Data sockets (Float Vector)
# ─────────────────────────────────────────────────────────
class FloatVectorInputSocket(bpy.types.NodeSocket):
    bl_idname = "FloatVectorInputSocketType"
    bl_label  = "Float Vector (In)"
    _PURPLE   = (0.65, 0.40, 0.95, 1.0)

    def draw(self, context, layout, node, text):
        layout.label(text=text or "Vector In")

    def draw_color(self, context, node):
        return self._PURPLE


class FloatVectorOutputSocket(bpy.types.NodeSocket):
    bl_idname = "FloatVectorOutputSocketType"
    bl_label  = "Float Vector (Out)"
    _PURPLE   = (0.65, 0.40, 0.95, 1.0)

    def draw(self, context, layout, node, text):
        layout.label(text=text or "Vector Out")

    def draw_color(self, context, node):
        return self._PURPLE


# ─────────────────────────────────────────────────────────
# Node: Capture Float Vector (read-only UI)
# ─────────────────────────────────────────────────────────
class UtilityCaptureFloatVectorNode(_ExploratoryNodeOnly, Node):
    """
    Unique utility node that owns a storage slot for a 3-float vector.
    Holds the last written value until overwritten by the graph.
    Not an Interaction or Reaction. No manual write/clear controls.
    """
    bl_idname = "UtilityCaptureFloatVectorNodeType"
    bl_label  = "Capture Float Vector"
    bl_icon   = 'EMPTY_SINGLE_ARROW'

    # Bind to scene store by UID; show/edit friendly name via property mirror
    capture_uid: bpy.props.StringProperty(name="UID", default="")
    capture_name: bpy.props.StringProperty(
        name="Name",
        default="Capture Vector",
        description="Friendly name for this capture slot",
        update=lambda self, ctx: self._on_name_changed(ctx)
    )

    # ---------------- internals ----------------

    def _ensure_uid(self, context):
        """Ensure a valid storage slot exists."""
        scn = context.scene if context else bpy.context.scene
        if not self.capture_uid or not slot_exists(self.capture_uid):
            self.capture_uid = create_floatvec_slot(scn, name=self.capture_name or "Capture Vector")

    # --------------- node lifetime ---------------

    def init(self, context):
        self.width = 280
        self.inputs.new("FloatVectorInputSocketType",   "Vector In")
        self.outputs.new("FloatVectorOutputSocketType", "Vector Out")
        self._ensure_uid(context)

    def free(self):
        # Don’t auto-delete the slot to avoid accidental loss during undo/redo.
        pass

    def update(self):
        # Heal missing UID/slot on any graph edit.
        self._ensure_uid(bpy.context)

    # --------------- drawing / UI ----------------

    def draw_buttons(self, context, layout):
        self._ensure_uid(context)

        box = layout.box()
        box.prop(self, "capture_name", text="Name")

        row = box.row(align=True); row.enabled = False
        row.prop(self, "capture_uid", text="UID")

        has_val, vec, ts = get_floatvec(self.capture_uid) if self.capture_uid else (False, (0.0,0.0,0.0), 0.0)
        stat = layout.box()
        stat.label(text="Current Value")
        col = stat.column(align=True)
        col.label(text=f"X: {vec[0]:.4f}")
        col.label(text=f"Y: {vec[1]:.4f}")
        col.label(text=f"Z: {vec[2]:.4f}")
        col = stat.column(align=True)
        col.label(text=f"Has Value: {'Yes' if has_val else 'No'}")

    # Keep store name in sync when renamed in the node
    def _on_name_changed(self, context):
        scn = context.scene if context else bpy.context.scene
        try:
            coll = getattr(scn, "utility_float_vectors", None)
            if not coll:
                return
            for it in coll:
                if getattr(it, "uid", "") == self.capture_uid:
                    it.name = self.capture_name or "Capture Vector"
                    break
        except Exception:
            pass

    # ---------------- generic sync layer ----------------

    def _coerce_vec3(self, vec3):
        try:
            x, y, z = float(vec3[0]), float(vec3[1]), float(vec3[2])
            return (x, y, z)
        except Exception:
            return None

    def _upstream_endpoint(self, node, socket):
        """
        Follow reroutes upstream. Return (node, socket) at the first non-reroute.
        """
        cur_node, cur_sock = node, socket
        # Handle Blender's native Reroute node
        while getattr(cur_node, "bl_idname", "") == "NodeReroute":
            in_socks = getattr(cur_node, "inputs", [])
            if not in_socks or not in_socks[0].is_linked:
                return (cur_node, cur_sock)
            lk = in_socks[0].links[0]
            cur_node = getattr(lk, "from_node", cur_node)
            cur_sock = getattr(lk, "from_socket", cur_sock)
        return (cur_node, cur_sock)

    def _resolve_vec_from_input_socket(self, in_sock):
        """
        Try to get a 3-float vector for a FloatVectorInputSocket by walking one link upstream.
        Supports:
          • Another node that defines export_vector() -> (x,y,z)
          • Another Capture node (reads its stored value)
        """
        if not getattr(in_sock, "is_linked", False):
            return None
        # Use first link (deterministic). If you want last, swap [-1].
        lk = in_sock.links[0]
        src_node = getattr(lk, "from_node", None)
        src_sock = getattr(lk, "from_socket", None)
        if not src_node:
            return None

        src_node, src_sock = self._upstream_endpoint(src_node, src_sock)

        # 1) Generic provider contract
        provider = getattr(src_node, "export_vector", None)
        if callable(provider):
            try:
                v = provider()
                return self._coerce_vec3(v)
            except Exception:
                pass

        # 2) Capture → read store
        if getattr(src_node, "bl_idname", "") == "UtilityCaptureFloatVectorNodeType":
            has_v, v, _ts = get_floatvec(getattr(src_node, "capture_uid", ""))
            if has_v:
                return self._coerce_vec3(v)

        # Unknown provider
        return None

    def _sync_all_vector_inputs_to_reactions(self):
        """
        Generic pass over all Exploratory node trees:
        For any node that has FloatVectorInput sockets with an 'exp_vec_target'
        metadata string on the socket, resolve the upstream vector and write it
        into that node's ReactionDefinition property before executors run.
        """
        import bpy
        scn = bpy.context.scene
        if not scn or not hasattr(scn, "reactions"):
            return

        for nt in bpy.data.node_groups:
            if getattr(nt, "bl_idname", "") != EXPL_TREE_ID:
                continue

            for node in nt.nodes:
                # Only nodes tied to a ReactionDefinition need syncing
                r_idx = getattr(node, "reaction_index", -1)
                if r_idx < 0 or r_idx >= len(getattr(scn, "reactions", [])):
                    continue
                reaction = scn.reactions[r_idx]

                for in_sock in getattr(node, "inputs", []):
                    if getattr(in_sock, "bl_idname", "") != "FloatVectorInputSocketType":
                        continue
                    if not getattr(in_sock, "is_linked", False):
                        continue

                    # Socket-level metadata declares which field to set
                    try:
                        target_prop = in_sock.get("exp_vec_target", "")
                    except Exception:
                        target_prop = ""
                    if not target_prop:
                        continue
                    if not hasattr(reaction, target_prop):
                        continue

                    vec = self._resolve_vec_from_input_socket(in_sock)
                    if vec is None:
                        continue

                    try:
                        setattr(reaction, target_prop, vec)
                    except Exception:
                        # Defensive: ignore bad writes but keep other sockets going
                        pass

    # Programmatic write hook (used by hitscan/projectile impact location, etc.)
    # Stores the value, then runs a GENERIC sync (no node-type branching).
    def write_from_graph(self, vec3, timestamp: float | None = None) -> bool:
        self._ensure_uid(bpy.context)

        v = self._coerce_vec3(vec3)
        if v is None:
            return False

        ok = set_floatvec(self.capture_uid, v, timestamp=timestamp)
        if not ok:
            return False

        # Generic: push to any bound vector inputs across the whole graph.
        self._sync_all_vector_inputs_to_reactions()
        return True


# ─────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────
_CLASSES = [
    FloatVectorInputSocket,
    FloatVectorOutputSocket,
    UtilityCaptureFloatVectorNode,
]

def register():
    for c in _CLASSES:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
