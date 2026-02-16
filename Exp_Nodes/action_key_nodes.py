# File: Exp_Nodes/action_key_nodes.py
import bpy
from bpy.types import Node
from bpy.props import StringProperty, IntProperty, BoolProperty
from .base_nodes import _ExploratoryNodeOnly

EXPL_TREE_ID = "ExploratoryNodesTreeType"


def _scene() -> bpy.types.Scene | None:
    scn = getattr(bpy.context, "scene", None)
    if scn:
        return scn
    return bpy.data.scenes[0] if bpy.data.scenes else None


def _unique_action_key_name(scn, base="Action"):
    existing = {getattr(it, "name", "") for it in getattr(scn, "action_keys", [])}
    if base and base not in existing:
        return base
    i = 1
    while True:
        cand = f"{base} {i}"
        if cand not in existing:
            return cand
        i += 1


def _create_action_key(scn, base="Action", enabled_default=False):
    item = scn.action_keys.add()
    item.name = _unique_action_key_name(scn, base=base)
    try:
        item.enabled_default = bool(enabled_default)
    except Exception:
        pass
    scn.action_keys_index = len(scn.action_keys) - 1
    return scn.action_keys_index, item.name


def _reindex_create_nodes_after_remove(removed_index: int):
    """Shift other Create Action Key nodes' indices down by one."""
    for ng in bpy.data.node_groups:
        if getattr(ng, "bl_idname", "") != EXPL_TREE_ID:
            continue
        for node in ng.nodes:
            if getattr(node, "bl_idname", "") != "CreateActionKeyNodeType":
                continue
            idx = getattr(node, "action_key_index", -1)
            if idx < 0:
                continue
            if idx == removed_index:
                try:
                    node.action_key_index = -1
                    node.action_key_name = ""
                except Exception:
                    pass
            elif idx > removed_index:
                try:
                    node.action_key_index = idx - 1
                    scn = _scene()
                    if scn and 0 <= node.action_key_index < len(scn.action_keys):
                        node.action_key_name = scn.action_keys[node.action_key_index].name
                except Exception:
                    pass


def _propagate_rename(old_name: str, new_name: str):
    """Rename everywhere that stores the string name (safe, no cross-module imports)."""
    if not old_name or not new_name or old_name == new_name:
        return
    scn = _scene()
    if not scn:
        return

    # Interactions (triggers)
    for inter in getattr(scn, "custom_interactions", []):
        if getattr(inter, "trigger_type", "") == "ACTION":
            if getattr(inter, "action_key_id", "") == old_name:
                try:
                    inter.action_key_id = new_name
                except Exception:
                    pass

    # Reactions (ACTION_KEYS)
    for r in getattr(scn, "reactions", []):
        if getattr(r, "reaction_type", "") == "ACTION_KEYS":
            if getattr(r, "action_key_name", "") == old_name:
                try:
                    r.action_key_name = new_name
                    r.action_key_id   = new_name
                except Exception:
                    pass


def _propagate_delete(removed_index: int, old_name: str):
    """Clear or reindex anything that referenced this action key."""
    scn = _scene()
    if not scn:
        return

    # Interactions (triggers): clear if matching old string
    for inter in getattr(scn, "custom_interactions", []):
        if getattr(inter, "trigger_type", "") == "ACTION":
            if getattr(inter, "action_key_id", "") == old_name:
                try:
                    inter.action_key_id = ""
                except Exception:
                    pass

    # Reactions (ACTION_KEYS): clear if same; shift indices above removed
    for r in getattr(scn, "reactions", []):
        if getattr(r, "reaction_type", "") != "ACTION_KEYS":
            continue
        try:
            ridx = getattr(r, "action_key_index", -1)
            rname = getattr(r, "action_key_name", "")
            if rname == old_name:
                r.action_key_name  = ""
                r.action_key_id    = ""
                r.action_key_index = -1
            elif ridx > removed_index:
                r.action_key_index = ridx - 1
        except Exception:
            pass


class CreateActionKeyNode(_ExploratoryNodeOnly, Node):
    """Standalone: creates/owns exactly one Action Key entry (no sockets)."""
    bl_idname = "CreateActionKeyNodeType"
    bl_label  = "Create Action Key"

    action_key_index: IntProperty(name="Action Index", default=-1, min=-1)

    # NOTE: guard attribute to prevent recursive callbacks
    # (plain Python attribute; not an RNA prop)
    _ak_guard = False

    # Class-level set to track nodes currently being initialized
    # This catches the window between Blender property copy and copy() call
    _initializing_pointers: set = set()

    def _name_changed(self, context):
        # prevent re-entrant loops
        if getattr(self, "_ak_guard", False):
            return

        # CRITICAL: Check if we're mid-copy (index not yet updated)
        # During node duplication, Blender copies properties BEFORE calling copy().
        # If _copying is set, skip all updates - copy() will handle it.
        if getattr(self, "_copying", False):
            return

        scn = _scene()
        if not scn or not hasattr(scn, "action_keys"):
            return
        idx = getattr(self, "action_key_index", -1)
        if not (0 <= idx < len(scn.action_keys)):
            return

        # SAFETY: Verify this node actually owns this action key index.
        # Multiple nodes pointing to the same index is a bug state.
        if not self._verify_ownership(scn, idx):
            return

        new_name = (getattr(self, "action_key_name", "") or "").strip()
        if not new_name:
            return

        # Enforce uniqueness (exclude our current slot)
        exist = {it.name for i, it in enumerate(scn.action_keys) if i != idx}
        if new_name in exist:
            new_name = _unique_action_key_name(scn, base=new_name)

        old_name = scn.action_keys[idx].name
        if old_name == new_name:
            return

        # Update scene entry; reflect on node without re-entering
        try:
            scn.action_keys[idx].name = new_name
            self._ak_guard = True
            self.action_key_name = new_name
        finally:
            self._ak_guard = False

        _propagate_rename(old_name, new_name)

    def _verify_ownership(self, scn, idx) -> bool:
        """
        Verify this node is the rightful owner of action_key_index.
        Returns False if another node with MATCHING name already owns this index.

        During duplication, the copy temporarily has the original's index but
        hasn't been through copy() yet. We detect this by checking if the OTHER
        node's name matches the scene entry - if so, THAT node is the real owner.
        """
        if not (0 <= idx < len(scn.action_keys)):
            return False

        scene_name = scn.action_keys[idx].name
        my_ptr = self.as_pointer()
        my_name = getattr(self, "action_key_name", "")

        for ng in bpy.data.node_groups:
            if getattr(ng, "bl_idname", "") != EXPL_TREE_ID:
                continue
            for node in ng.nodes:
                if getattr(node, "bl_idname", "") != "CreateActionKeyNodeType":
                    continue
                if node.as_pointer() == my_ptr:
                    continue  # Skip self
                if getattr(node, "action_key_index", -1) == idx:
                    # Another node has this index. Check who's the real owner.
                    other_name = getattr(node, "action_key_name", "")

                    # If OTHER node's name matches scene, it's the owner - we're stale
                    if other_name == scene_name and my_name != scene_name:
                        return False

                    # If OUR name matches scene, we're the owner - other is stale
                    if my_name == scene_name and other_name != scene_name:
                        continue  # Ignore stale copy

                    # If both match or neither match, it's a true conflict
                    if other_name == my_name:
                        return False
        return True

    def _enabled_changed(self, context):
        # Skip if we're mid-copy
        if getattr(self, "_copying", False):
            return

        scn = _scene()
        if not scn or not hasattr(scn, "action_keys"):
            return
        idx = getattr(self, "action_key_index", -1)
        if not (0 <= idx < len(scn.action_keys)):
            return

        # Verify ownership before modifying
        if not self._verify_ownership(scn, idx):
            return

        try:
            scn.action_keys[idx].enabled_default = bool(self.enabled_default)
        except Exception:
            pass

    action_key_name: StringProperty(
        name="Name",
        default="Action",
        update=_name_changed,
    )
    enabled_default: BoolProperty(
        name="Enabled by Default",
        default=False,
        update=_enabled_changed,
    )

    _EXPL_TINT_ACTION = (0.28, 0.2, 0.3)

    def _tint(self):
        try:
            self.use_custom_color = True
            self.color = self._EXPL_TINT_ACTION
        except Exception:
            pass

    def init(self, context):
        self.width = 220
        self._tint()
        scn = _scene()
        if not scn or not hasattr(scn, "action_keys"):
            self.action_key_index = -1
            return

        idx, nm = _create_action_key(scn, base="Action", enabled_default=False)
        self.action_key_index = idx
        # Set node label without re-entering callback
        try:
            self._ak_guard = True
            self.action_key_name = nm
        finally:
            self._ak_guard = False
        try:
            self.enabled_default = bool(scn.action_keys[idx].enabled_default)
        except Exception:
            pass

        # Inline Bool socket for enabled_default
        s = self.inputs.new("ExpBoolSocketType", "Enabled by Default")
        s.prop_name = "enabled_default"

    def copy(self, node):
        """
        Called when node is duplicated. MUST create a NEW unique action key.
        Blender copies all properties BEFORE calling this, so self already has
        the original's action_key_index/name. We MUST override them.
        """
        # Mark that we're mid-copy to prevent callbacks from messing with state
        self._copying = True

        try:
            self.width = getattr(node, "width", 220)
            scn = _scene()
            if not scn or not hasattr(scn, "action_keys"):
                self.action_key_index = -1
                self.action_key_name = ""
                self.enabled_default = False
                return

            base = getattr(node, "action_key_name", "") or "Action"
            default_flag = bool(getattr(node, "enabled_default", False))

            # Create NEW action key - this MUST return a unique name
            idx, nm = _create_action_key(scn, base=base, enabled_default=default_flag)

            # Update our index to point to the NEW action key
            self.action_key_index = idx

            # Update our name (guarded to prevent _name_changed callback)
            try:
                self._ak_guard = True
                self.action_key_name = nm
            finally:
                self._ak_guard = False

            self.enabled_default = default_flag
        finally:
            self._copying = False

    def free(self):
        scn = _scene()
        idx = getattr(self, "action_key_index", -1)
        old_name = getattr(self, "action_key_name", "")
        # Remove Scene list entry if valid
        if scn and hasattr(scn, "action_keys") and 0 <= idx < len(scn.action_keys):
            try:
                scn.action_keys.remove(idx)
            except Exception:
                pass
            # Cascade: clean references + reindex surviving nodes/holders
            _propagate_delete(idx, old_name)
            _reindex_create_nodes_after_remove(idx)
        # Mark node unbound
        self.action_key_index = -1
        self.action_key_name = ""

    def update(self):
        """
        Called when node graph changes.
        NODE is source of truth - push node state TO scene, never pull FROM scene.
        """
        # Skip if mid-copy
        if getattr(self, "_copying", False):
            return

        scn = _scene()
        if not scn or not hasattr(scn, "action_keys"):
            return

        idx = getattr(self, "action_key_index", -1)
        if not (0 <= idx < len(scn.action_keys)):
            return

        # CRITICAL: Verify ownership before modifying scene data
        if not self._verify_ownership(scn, idx):
            return

        # Push node name TO scene (node is source of truth)
        node_name = getattr(self, "action_key_name", "")
        scene_name = scn.action_keys[idx].name

        if node_name and node_name != scene_name:
            # Ensure uniqueness before writing
            exist = {it.name for i, it in enumerate(scn.action_keys) if i != idx}
            if node_name in exist:
                # Name conflict - generate unique name
                node_name = _unique_action_key_name(scn, base=node_name)
                try:
                    self._ak_guard = True
                    self.action_key_name = node_name
                finally:
                    self._ak_guard = False

            try:
                old_name = scene_name
                scn.action_keys[idx].name = node_name
                if old_name != node_name:
                    _propagate_rename(old_name, node_name)
            except Exception:
                pass

        # Push enabled_default TO scene
        try:
            node_flag = bool(getattr(self, "enabled_default", False))
            scene_flag = bool(getattr(scn.action_keys[idx], "enabled_default", False))
            if node_flag != scene_flag:
                scn.action_keys[idx].enabled_default = node_flag
        except Exception:
            pass

    def draw_buttons(self, context, layout):
        box = layout.box()
        box.prop(self, "action_key_name", text="Name")
        info = layout.box()
        info.label(text="This node defines the key. No links needed.", icon='INFO')