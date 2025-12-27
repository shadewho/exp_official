# Exp_Nodes/tracker_nodes.py
"""
Tracker Nodes - Worker-Offloaded Conditional Logic

PERFORMANCE:
- All evaluation happens on the WORKER thread
- Main thread only handles UI and dispatch
- Serialized to worker at game start

Node Types:
- DistanceTrackerNode: Distance between two objects
- ContactTrackerNode: Mesh collision detection
- StateTrackerNode: Character state monitoring
- InputTrackerNode: Input action monitoring (FORWARD, JUMP, etc)
- LogicGateNode: AND/OR/NOT for combining bools

SOCKETS:
- Uses unified socket types from utility_nodes.py
- ExpBoolSocketType, ExpObjectSocketType, ExpFloatSocketType
- All sockets of same type can connect regardless of source node
"""

import bpy
from bpy.types import Node
from ..Exp_Game.props_and_utils.trackers import (
    COMPARE_OPERATORS, CHAR_STATES, INPUT_ACTIONS,
)


# ══════════════════════════════════════════════════════════════════════════════
# BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════

EXPL_TREE_ID = "ExploratoryNodesTreeType"


class _ExploratoryNodeOnly:
    """Mixin that restricts a node to our custom tree."""
    @classmethod
    def poll(cls, ntree):
        return getattr(ntree, "bl_idname", "") == EXPL_TREE_ID


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED SOCKET TYPE NAMES (from utility_nodes.py)
# ══════════════════════════════════════════════════════════════════════════════
# These are the bl_idname strings for the unified sockets.
# Using these ensures all sockets of the same data type can connect.

BOOL_SOCKET = "ExpBoolSocketType"
FLOAT_SOCKET = "ExpFloatSocketType"
OBJECT_SOCKET = "ExpObjectSocketType"
VECTOR_SOCKET = "ExpVectorSocketType"


# ══════════════════════════════════════════════════════════════════════════════
# DISTANCE TRACKER NODE
# ══════════════════════════════════════════════════════════════════════════════

class DistanceTrackerNode(_ExploratoryNodeOnly, Node):
    """Tracks distance between two objects. Outputs Bool."""
    bl_idname = "DistanceTrackerNodeType"
    bl_label = "Distance Tracker"

    object_a: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Object A",
        description="First object for distance check"
    )
    object_b: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Object B",
        description="Second object for distance check"
    )
    operator: bpy.props.EnumProperty(
        name="Operator",
        items=COMPARE_OPERATORS,
        default='LT'
    )
    distance: bpy.props.FloatProperty(
        name="Distance",
        default=5.0,
        min=0.0,
        unit='LENGTH'
    )
    eval_hz: bpy.props.IntProperty(
        name="Eval Rate",
        default=10,
        min=1,
        max=30,
        description="Checks per second (worker thread)"
    )

    def init(self, context):
        self.width = 200
        # Object inputs with inline property drawing
        sock_a = self.inputs.new(OBJECT_SOCKET, "Object A")
        sock_a.prop_name = "object_a"
        sock_b = self.inputs.new(OBJECT_SOCKET, "Object B")
        sock_b.prop_name = "object_b"
        # Bool output
        self.outputs.new(BOOL_SOCKET, "Result")

    def draw_buttons(self, context, layout):
        # Operator and distance on same row
        row = layout.row(align=True)
        row.prop(self, "operator", text="")
        row.prop(self, "distance", text="")

        layout.prop(self, "eval_hz", text="Hz")

    def export_condition(self):
        """Export condition data for worker serialization."""
        return {
            'type': 'DISTANCE',
            'object_a': self.object_a.name if self.object_a else "",
            'object_b': self.object_b.name if self.object_b else "",
            'op': self.operator,
            'value': self.distance,
            'eval_hz': self.eval_hz,
        }


# ══════════════════════════════════════════════════════════════════════════════
# STATE TRACKER NODE
# ══════════════════════════════════════════════════════════════════════════════

class StateTrackerNode(_ExploratoryNodeOnly, Node):
    """Tracks character state (grounded, sprinting, etc). Outputs Bool."""
    bl_idname = "StateTrackerNodeType"
    bl_label = "State Tracker"

    state: bpy.props.EnumProperty(
        name="State",
        items=CHAR_STATES,
        default='GROUNDED'
    )
    equals: bpy.props.BoolProperty(
        name="Is",
        default=True,
        description="True = is in state, False = is NOT in state"
    )
    eval_hz: bpy.props.IntProperty(
        name="Eval Rate",
        default=10,
        min=1,
        max=30,
        description="Checks per second (worker thread)"
    )

    def init(self, context):
        self.width = 180
        self.outputs.new(BOOL_SOCKET, "Result")

    def draw_buttons(self, context, layout):
        layout.prop(self, "state", text="")

        row = layout.row()
        row.prop(self, "equals", text="Is" if self.equals else "Is NOT")

        layout.prop(self, "eval_hz", text="Hz")

    def export_condition(self):
        """Export condition data for worker serialization."""
        return {
            'type': 'CHAR_STATE',
            'state': self.state,
            'equals': self.equals,
            'eval_hz': self.eval_hz,
        }


# ══════════════════════════════════════════════════════════════════════════════
# CONTACT TRACKER NODE
# ══════════════════════════════════════════════════════════════════════════════

class ContactTrackerNode(_ExploratoryNodeOnly, Node):
    """Tracks mesh collision/contact. Outputs Bool."""
    bl_idname = "ContactTrackerNodeType"
    bl_label = "Contact Tracker"

    contact_object: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Object",
        description="Object to check for contacts"
    )
    contact_target: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Target",
        description="Target object to check contact with"
    )
    contact_collection: bpy.props.PointerProperty(
        type=bpy.types.Collection,
        name="Collection",
        description="Collection of objects to check contact with"
    )
    use_collection: bpy.props.BoolProperty(
        name="Use Collection",
        default=False
    )
    eval_hz: bpy.props.IntProperty(
        name="Eval Rate",
        default=10,
        min=1,
        max=30,
        description="Checks per second (worker thread)"
    )

    def init(self, context):
        self.width = 200
        # Object inputs with inline property drawing
        sock_obj = self.inputs.new(OBJECT_SOCKET, "Object")
        sock_obj.prop_name = "contact_object"
        sock_tgt = self.inputs.new(OBJECT_SOCKET, "Target")
        sock_tgt.prop_name = "contact_target"
        self.outputs.new(BOOL_SOCKET, "Result")

    def draw_buttons(self, context, layout):
        layout.prop(self, "use_collection")
        if self.use_collection:
            layout.prop(self, "contact_collection", text="")

        layout.prop(self, "eval_hz", text="Hz")

    def export_condition(self):
        """Export condition data for worker serialization."""
        targets = []
        if self.use_collection and self.contact_collection:
            targets = [obj.name for obj in self.contact_collection.objects]
        elif self.contact_target:
            targets = [self.contact_target.name]

        return {
            'type': 'CONTACT',
            'object': self.contact_object.name if self.contact_object else "",
            'targets': targets,
            'eval_hz': self.eval_hz,
        }


# ══════════════════════════════════════════════════════════════════════════════
# INPUT TRACKER NODE
# ══════════════════════════════════════════════════════════════════════════════

class InputTrackerNode(_ExploratoryNodeOnly, Node):
    """Tracks input action state (FORWARD, JUMP, etc). Outputs Bool."""
    bl_idname = "InputTrackerNodeType"
    bl_label = "Input Tracker"

    input_action: bpy.props.EnumProperty(
        name="Action",
        items=INPUT_ACTIONS,
        default='FORWARD',
        description="Input action to monitor"
    )
    is_pressed: bpy.props.BoolProperty(
        name="Pressed",
        default=True,
        description="True = check if pressed, False = check if NOT pressed"
    )
    eval_hz: bpy.props.IntProperty(
        name="Eval Rate",
        default=30,
        min=1,
        max=30,
        description="Checks per second (worker thread)"
    )

    def init(self, context):
        self.width = 180
        self.outputs.new(BOOL_SOCKET, "Result")

    def draw_buttons(self, context, layout):
        layout.prop(self, "input_action", text="")

        row = layout.row()
        row.prop(self, "is_pressed", text="Pressed" if self.is_pressed else "Not Pressed")

        layout.prop(self, "eval_hz", text="Hz")

    def export_condition(self):
        """Export condition data for worker serialization."""
        return {
            'type': 'INPUT',
            'action': self.input_action,
            'is_pressed': self.is_pressed,
            'eval_hz': self.eval_hz,
        }


# ══════════════════════════════════════════════════════════════════════════════
# GAME TIME TRACKER NODE
# ══════════════════════════════════════════════════════════════════════════════

class GameTimeTrackerNode(_ExploratoryNodeOnly, Node):
    """Tracks elapsed game time in seconds. Outputs Float."""
    bl_idname = "GameTimeTrackerNodeType"
    bl_label = "Game Time"

    compare_enabled: bpy.props.BoolProperty(
        name="Compare",
        default=True,
        description="Enable time comparison"
    )
    operator: bpy.props.EnumProperty(
        name="Operator",
        items=COMPARE_OPERATORS,
        default='GE'
    )
    time_threshold: bpy.props.FloatProperty(
        name="Time",
        default=10.0,
        min=0.0,
        description="Time threshold in seconds"
    )
    eval_hz: bpy.props.IntProperty(
        name="Eval Rate",
        default=10,
        min=1,
        max=30,
        description="Checks per second (worker thread)"
    )

    def init(self, context):
        self.width = 180
        # Float output for raw time value
        self.outputs.new(FLOAT_SOCKET, "Time")
        # Bool output for comparison result
        self.outputs.new(BOOL_SOCKET, "Result")

    def draw_buttons(self, context, layout):
        layout.prop(self, "compare_enabled")
        if self.compare_enabled:
            row = layout.row(align=True)
            row.prop(self, "operator", text="")
            row.prop(self, "time_threshold", text="")

        layout.prop(self, "eval_hz", text="Hz")

    def export_condition(self):
        """Export condition data for worker serialization."""
        return {
            'type': 'GAME_TIME',
            'compare_enabled': self.compare_enabled,
            'op': self.operator,
            'value': self.time_threshold,
            'eval_hz': self.eval_hz,
        }


# ══════════════════════════════════════════════════════════════════════════════
# LOGIC GATE NODES (Dynamic inputs)
# ══════════════════════════════════════════════════════════════════════════════

class LogicAndNode(_ExploratoryNodeOnly, Node):
    """AND gate - all inputs must be true. Outputs Bool. Dynamic inputs."""
    bl_idname = "LogicAndNodeType"
    bl_label = "AND"

    def init(self, context):
        self.width = 100
        self.inputs.new(BOOL_SOCKET, "Input")
        self.inputs.new(BOOL_SOCKET, "Input")
        self.outputs.new(BOOL_SOCKET, "Result")

    def update(self):
        """Add/remove inputs dynamically based on connections."""
        self._update_dynamic_inputs()

    def _update_dynamic_inputs(self):
        # Count connected and unconnected inputs
        connected = sum(1 for inp in self.inputs if inp.is_linked)
        total = len(self.inputs)

        # Always keep exactly one empty input at the bottom
        unconnected = total - connected

        if unconnected == 0:
            # All inputs connected - add a new one
            self.inputs.new(BOOL_SOCKET, "Input")
        elif unconnected > 1:
            # More than one empty - remove extras from the end
            for i in range(len(self.inputs) - 1, -1, -1):
                if not self.inputs[i].is_linked and unconnected > 1:
                    self.inputs.remove(self.inputs[i])
                    unconnected -= 1

    def draw_buttons(self, context, layout):
        pass


class LogicOrNode(_ExploratoryNodeOnly, Node):
    """OR gate - any input must be true. Outputs Bool. Dynamic inputs."""
    bl_idname = "LogicOrNodeType"
    bl_label = "OR"

    def init(self, context):
        self.width = 100
        self.inputs.new(BOOL_SOCKET, "Input")
        self.inputs.new(BOOL_SOCKET, "Input")
        self.outputs.new(BOOL_SOCKET, "Result")

    def update(self):
        """Add/remove inputs dynamically based on connections."""
        self._update_dynamic_inputs()

    def _update_dynamic_inputs(self):
        connected = sum(1 for inp in self.inputs if inp.is_linked)
        total = len(self.inputs)
        unconnected = total - connected

        if unconnected == 0:
            self.inputs.new(BOOL_SOCKET, "Input")
        elif unconnected > 1:
            for i in range(len(self.inputs) - 1, -1, -1):
                if not self.inputs[i].is_linked and unconnected > 1:
                    self.inputs.remove(self.inputs[i])
                    unconnected -= 1

    def draw_buttons(self, context, layout):
        pass


class LogicNotNode(_ExploratoryNodeOnly, Node):
    """NOT gate - inverts input. Outputs Bool."""
    bl_idname = "LogicNotNodeType"
    bl_label = "NOT"

    def init(self, context):
        self.width = 80
        self.inputs.new(BOOL_SOCKET, "Input")
        self.outputs.new(BOOL_SOCKET, "Result")

    def draw_buttons(self, context, layout):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRATION
# ══════════════════════════════════════════════════════════════════════════════
# NOTE: Sockets are registered in utility_nodes.py (unified socket types)
# This file only registers the tracker node classes.

classes = [
    # Tracker nodes
    DistanceTrackerNode,
    StateTrackerNode,
    ContactTrackerNode,
    InputTrackerNode,
    GameTimeTrackerNode,
    # Logic gates
    LogicAndNode,
    LogicOrNode,
    LogicNotNode,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
