# File: Exp_Nodes/counter_timer_nodes.py
# Counter and Timer utility nodes for Exploratory.

import bpy
from .base_nodes import _ExploratoryNodeOnly

EXPL_TREE_ID = "ExploratoryNodesTreeType"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _scene() -> bpy.types.Scene | None:
    scn = getattr(bpy.context, "scene", None)
    if scn:
        return scn
    return bpy.data.scenes[0] if bpy.data.scenes else None


# ─────────────────────────────────────────────────────────────
# Counter Node Helpers
# ─────────────────────────────────────────────────────────────

def _create_counter() -> int:
    """Create a counter via the canonical operator. Returns the new index, or -1 on failure."""
    scn = _scene()
    if not scn:
        return -1

    try:
        res = bpy.ops.exploratory.add_counter('EXEC_DEFAULT')
        if 'CANCELLED' in res:
            return -1
    except Exception:
        return -1

    return len(scn.counters) - 1 if hasattr(scn, "counters") else -1


def _duplicate_counter(src_index: int) -> int:
    """Duplicate a counter by creating a new one and copying properties."""
    scn = _scene()
    if not scn or not hasattr(scn, "counters"):
        return -1
    if not (0 <= src_index < len(scn.counters)):
        return -1

    new_index = _create_counter()
    if new_index < 0:
        return -1

    src = scn.counters[src_index]
    dst = scn.counters[new_index]

    for prop in src.bl_rna.properties:
        ident = prop.identifier
        if ident == "rna_type" or getattr(prop, "is_readonly", False) or getattr(prop, "is_collection", False):
            continue
        try:
            setattr(dst, ident, getattr(src, ident))
        except Exception:
            pass

    try:
        dst.name = f"{getattr(src, 'name', 'Counter')} (Copy)"
    except Exception:
        pass

    scn.counters_index = new_index
    return new_index


def _delete_counter_and_fix_indices(removed_index: int) -> None:
    """Delete a counter via operator and repair node indices."""
    scn = _scene()
    if not scn or not hasattr(scn, "counters"):
        return
    if not (0 <= removed_index < len(scn.counters)):
        return

    try:
        res = bpy.ops.exploratory.remove_counter('EXEC_DEFAULT', index=removed_index)
        if 'CANCELLED' in res:
            return
    except Exception:
        return

    scn.counters_index = max(0, min(removed_index, len(scn.counters) - 1))

    # Repair CounterNode indices
    for ng in bpy.data.node_groups:
        if getattr(ng, "bl_idname", "") != EXPL_TREE_ID:
            continue
        for node in ng.nodes:
            if getattr(node, "bl_idname", "") == "CounterNodeType":
                idx = getattr(node, "counter_index", -1)
                if idx > removed_index:
                    node.counter_index = idx - 1
                elif idx == removed_index:
                    node.counter_index = -1


# ─────────────────────────────────────────────────────────────
# Timer Node Helpers
# ─────────────────────────────────────────────────────────────

def _create_timer() -> int:
    """Create a timer via the canonical operator. Returns the new index, or -1 on failure."""
    scn = _scene()
    if not scn:
        return -1

    try:
        res = bpy.ops.exploratory.add_timer('EXEC_DEFAULT')
        if 'CANCELLED' in res:
            return -1
    except Exception:
        return -1

    return len(scn.timers) - 1 if hasattr(scn, "timers") else -1


def _duplicate_timer(src_index: int) -> int:
    """Duplicate a timer by creating a new one and copying properties."""
    scn = _scene()
    if not scn or not hasattr(scn, "timers"):
        return -1
    if not (0 <= src_index < len(scn.timers)):
        return -1

    new_index = _create_timer()
    if new_index < 0:
        return -1

    src = scn.timers[src_index]
    dst = scn.timers[new_index]

    for prop in src.bl_rna.properties:
        ident = prop.identifier
        if ident == "rna_type" or getattr(prop, "is_readonly", False) or getattr(prop, "is_collection", False):
            continue
        try:
            setattr(dst, ident, getattr(src, ident))
        except Exception:
            pass

    try:
        dst.name = f"{getattr(src, 'name', 'Timer')} (Copy)"
    except Exception:
        pass

    scn.timers_index = new_index
    return new_index


def _delete_timer_and_fix_indices(removed_index: int) -> None:
    """Delete a timer via operator and repair node indices."""
    scn = _scene()
    if not scn or not hasattr(scn, "timers"):
        return
    if not (0 <= removed_index < len(scn.timers)):
        return

    try:
        res = bpy.ops.exploratory.remove_timer('EXEC_DEFAULT', index=removed_index)
        if 'CANCELLED' in res:
            return
    except Exception:
        return

    scn.timers_index = max(0, min(removed_index, len(scn.timers) - 1))

    # Repair TimerNode indices
    for ng in bpy.data.node_groups:
        if getattr(ng, "bl_idname", "") != EXPL_TREE_ID:
            continue
        for node in ng.nodes:
            if getattr(node, "bl_idname", "") == "TimerNodeType":
                idx = getattr(node, "timer_index", -1)
                if idx > removed_index:
                    node.timer_index = idx - 1
                elif idx == removed_index:
                    node.timer_index = -1


# ─────────────────────────────────────────────────────────────
# Counter Node
# ─────────────────────────────────────────────────────────────

class CounterNode(_ExploratoryNodeOnly, bpy.types.Node):
    """
    A Counter Node that owns a real CounterDefinition in Scene.counters.
    - Creating the node creates a counter and binds it.
    - Duplicating the node duplicates the bound counter.
    - Deleting the node deletes the bound counter.
    """
    bl_idname = "CounterNodeType"
    bl_label = "Counter"

    _EXPL_TINT_COUNTER = (0.28, 0.38, 0.52)

    counter_index: bpy.props.IntProperty(name="Counter Index", default=-1, min=-1)

    def _tint(self):
        try:
            self.use_custom_color = True
            self.color = self._EXPL_TINT_COUNTER
        except Exception:
            pass

    def init(self, context):
        self.width = 300
        self._tint()
        self.counter_index = _create_counter()
        # Inline data sockets
        s = self.inputs.new("ExpIntSocketType", "Default Value")
        s.counter_prop = "default_value"
        s = self.inputs.new("ExpBoolSocketType", "Enable Min")
        s.counter_prop = "use_min_limit"
        s = self.inputs.new("ExpIntSocketType", "Min Value")
        s.counter_prop = "min_value"
        s = self.inputs.new("ExpBoolSocketType", "Enable Max")
        s.counter_prop = "use_max_limit"
        s = self.inputs.new("ExpIntSocketType", "Max Value")
        s.counter_prop = "max_value"

    def copy(self, node):
        src_idx = getattr(node, "counter_index", -1)
        if src_idx == -1:
            self.counter_index = _create_counter()
        else:
            self.counter_index = _duplicate_counter(src_idx)
        self.width = getattr(node, "width", 300)

    def free(self):
        idx = getattr(self, "counter_index", -1)
        if idx >= 0:
            _delete_counter_and_fix_indices(idx)
        self.counter_index = -1

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.counter_index
        if not scn or not hasattr(scn, "counters") or not (0 <= idx < len(scn.counters)):
            layout.label(text="(No Counter Bound)", icon='ERROR')
            return

        counter = scn.counters[idx]

        info_box = layout.box()
        info_box.prop(counter, "name", text="Name")
        info_box.prop(counter, "description", text="Description")

    def draw_label(self):
        scn = _scene()
        idx = self.counter_index
        if scn and hasattr(scn, "counters") and 0 <= idx < len(scn.counters):
            nm = getattr(scn.counters[idx], "name", "")
            return nm if nm else "Counter"
        return "Counter"


# ─────────────────────────────────────────────────────────────
# Timer Node
# ─────────────────────────────────────────────────────────────

class TimerNode(_ExploratoryNodeOnly, bpy.types.Node):
    """
    A Timer Node that owns a real TimerDefinition in Scene.timers.
    - Creating the node creates a timer and binds it.
    - Duplicating the node duplicates the bound timer.
    - Deleting the node deletes the bound timer.
    """
    bl_idname = "TimerNodeType"
    bl_label = "Timer"

    _EXPL_TINT_TIMER = (0.28, 0.38, 0.52)

    timer_index: bpy.props.IntProperty(name="Timer Index", default=-1, min=-1)

    def _tint(self):
        try:
            self.use_custom_color = True
            self.color = self._EXPL_TINT_TIMER
        except Exception:
            pass

    def init(self, context):
        self.width = 300
        self._tint()
        self.timer_index = _create_timer()
        # Inline data sockets
        s = self.inputs.new("ExpFloatSocketType", "Start Value")
        s.timer_prop = "start_value"
        s = self.inputs.new("ExpFloatSocketType", "End Value")
        s.timer_prop = "end_value"

    def copy(self, node):
        src_idx = getattr(node, "timer_index", -1)
        if src_idx == -1:
            self.timer_index = _create_timer()
        else:
            self.timer_index = _duplicate_timer(src_idx)
        self.width = getattr(node, "width", 300)

    def free(self):
        idx = getattr(self, "timer_index", -1)
        if idx >= 0:
            _delete_timer_and_fix_indices(idx)
        self.timer_index = -1

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.timer_index
        if not scn or not hasattr(scn, "timers") or not (0 <= idx < len(scn.timers)):
            layout.label(text="(No Timer Bound)", icon='ERROR')
            return

        timer = scn.timers[idx]

        info_box = layout.box()
        info_box.prop(timer, "name", text="Name")
        info_box.prop(timer, "description", text="Description")

        settings_box = layout.box()
        settings_box.prop(timer, "timer_mode", text="Mode")

    def draw_label(self):
        scn = _scene()
        idx = self.timer_index
        if scn and hasattr(scn, "timers") and 0 <= idx < len(scn.timers):
            nm = getattr(scn.timers[idx], "name", "")
            return nm if nm else "Timer"
        return "Timer"
#test