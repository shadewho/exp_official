# File: Exp_Nodes/objective_nodes.py
import bpy
from bpy.types import Node
from .base_nodes import _ExploratoryNodeOnly
# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _scene() -> bpy.types.Scene | None:
    scn = getattr(bpy.context, "scene", None)
    if scn:
        return scn
    return bpy.data.scenes[0] if bpy.data.scenes else None


def _create_objective() -> int:
    """
    Create via the canonical operator so systems/exp_objectives.py remains the source of truth.
    Returns the new objective index, or -1 on failure.
    """
    scn = _scene()
    if not scn:
        return -1

    try:
        # Use EXEC_DEFAULT to avoid popups and respect current scene context.
        res = bpy.ops.exploratory.add_objective('EXEC_DEFAULT')
        if 'CANCELLED' in res:
            return -1
    except Exception:
        return -1

    # Operator appends to scene.objectives and updates objectives_index.
    # New item is always the last one.
    return len(scn.objectives) - 1 if hasattr(scn, "objectives") else -1


def _duplicate_objective(src_index: int) -> int:
    """
    Duplicate by calling the Add operator (to create a new, valid item)
    and then shallow-copy the editable properties from the source.
    This still centralizes the creation path through your operator.
    """
    scn = _scene()
    if not scn or not hasattr(scn, "objectives"):
        return -1
    if not (0 <= src_index < len(scn.objectives)):
        return -1

    # 1) Create a brand new objective through the operator (source of truth).
    new_index = _create_objective()
    if new_index < 0:
        return -1

    src = scn.objectives[src_index]
    dst = scn.objectives[new_index]

    # 2) Copy over writable, non-collection props (robust against schema drift).
    for prop in src.bl_rna.properties:
        ident = prop.identifier
        if ident == "rna_type" or getattr(prop, "is_readonly", False) or getattr(prop, "is_collection", False):
            continue
        try:
            setattr(dst, ident, getattr(src, ident))
        except Exception:
            pass

    # Friendly default name that mirrors your prior behavior
    try:
        dst.name = f"{getattr(src, 'name', 'Objective')} (Copy)"
    except Exception:
        pass

    # Return the new index
    scn.objectives_index = new_index
    return new_index


def _delete_objective_and_fix_indices(removed_index: int) -> None:
    """
    Delete via the canonical Remove operator, then fix node-stored indices.
    We *must* do the node index repair here because only the node layer
    knows which nodes are holding raw indices.
    """
    scn = _scene()
    if not scn or not hasattr(scn, "objectives"):
        return
    if not (0 <= removed_index < len(scn.objectives)):
        return

    # 1) Remove the item through your operator.
    try:
        res = bpy.ops.exploratory.remove_objective('EXEC_DEFAULT', index=removed_index)
        if 'CANCELLED' in res:
            return
    except Exception:
        return

    # 2) Clamp the active index like your operator does.
    scn.objectives_index = max(0, min(removed_index, len(scn.objectives) - 1))

    # 3) Repair any ObjectiveNode indices that were pointing into the shifted list.
    for ng in bpy.data.node_groups:
        if getattr(ng, "bl_idname", "") != "ExploratoryNodesTreeType":
            continue
        for node in ng.nodes:
            if getattr(node, "bl_idname", "") == "ObjectiveNodeType":
                idx = getattr(node, "objective_index", -1)
                if idx > removed_index:
                    node.objective_index = idx - 1
                elif idx == removed_index:
                    # The node's owned objective was deleted; mark invalid.
                    node.objective_index = -1



# ─────────────────────────────────────────────────────────────
# Objective Node (always owns exactly one real Objective)
# ─────────────────────────────────────────────────────────────

class ObjectiveNode(_ExploratoryNodeOnly, bpy.types.Node):
    """
    An Objective Node that ALWAYS owns a real ObjectiveDefinition in Scene.objectives.
    - Creating the node creates an objective and binds it.
    - Duplicating the node duplicates the bound objective and binds the copy.
    - Deleting the node deletes the bound objective.
    No add/remove operators. If the node exists, its objective exists.
    """
    bl_idname = "ObjectiveNodeType"
    bl_label = "Objective"
    bl_icon = 'TRACKER'

    objective_index: bpy.props.IntProperty(name="Objective Index", default=-1, min=-1)

    def init(self, context):
        self.width = 300
        self.objective_index = _create_objective()

    def copy(self, node):
        src_idx = getattr(node, "objective_index", -1)
        if src_idx == -1:
            self.objective_index = _create_objective()
        else:
            self.objective_index = _duplicate_objective(src_idx)
        self.width = getattr(node, "width", 300)


    def free(self):
        # Deleting the node MUST delete its objective.
        idx = getattr(self, "objective_index", -1)
        if idx >= 0:
            _delete_objective_and_fix_indices(idx)
        self.objective_index = -1

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.objective_index
        if not scn or not hasattr(scn, "objectives") or not (0 <= idx < len(scn.objectives)):
            layout.label(text="(No Objective Bound)", icon='ERROR')
            return

        objv = scn.objectives[idx]

        # — Basic Info (matches N-panel) —
        info_box = layout.box()
        info_box.prop(objv, "name", text="Name")
        info_box.prop(objv, "description", text="Description")

        # — Counter Settings (matches N-panel) —
        counter_box = layout.box()
        counter_box.label(text="Objective Counter Settings")
        counter_box.prop(objv, "default_value", text="Default Value")
        counter_box.label(text="(Value at start/reset)")
        counter_box.prop(objv, "use_min_limit", text="Enable Min")
        if objv.use_min_limit:
            counter_box.prop(objv, "min_value", text="Min Value")
        counter_box.prop(objv, "use_max_limit", text="Enable Max")
        if objv.use_max_limit:
            counter_box.prop(objv, "max_value", text="Max Value")

        # — Timer Settings (matches N-panel) —
        timer_box = layout.box()
        timer_box.label(text="Objective Timer Settings")
        timer_box.prop(objv, "timer_mode", text="Mode")
        timer_box.prop(objv, "timer_start_value", text="Start Value")
        timer_box.prop(objv, "timer_end_value", text="End Value")


    def draw_label(self):
        scn = _scene()
        idx = self.objective_index
        if scn and hasattr(scn, "objectives") and 0 <= idx < len(scn.objectives):
            nm = getattr(scn.objectives[idx], "name", "")
            return nm if nm else "Objective"
        return "Objective"
