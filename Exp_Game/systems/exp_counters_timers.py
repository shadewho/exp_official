# File: exp_counters_timers.py
# Dedicated counter and timer utilities for Exploratory game system.
# Replaces the old unified "objectives" system with clean separation.

import bpy
from ..props_and_utils.exp_time import get_game_time


# ─────────────────────────────────────────────────────────────
# Counter Definition
# ─────────────────────────────────────────────────────────────

def _find_counter_node(scn, counter_self):
    """Find the CounterNode owning this counter definition."""
    c_idx = -1
    for i, c in enumerate(scn.counters):
        if c == counter_self:
            c_idx = i
            break
    if c_idx < 0:
        return None, -1
    for ng in bpy.data.node_groups:
        if getattr(ng, "bl_idname", "") != "ExploratoryNodesTreeType":
            continue
        for node in ng.nodes:
            if getattr(node, "bl_idname", "") == "CounterNodeType":
                if getattr(node, "counter_index", -1) == c_idx:
                    return node, c_idx
    return None, c_idx


def _update_counter_min_limit(self, context):
    """Hide/show Min Value socket on the owning counter node (runs outside draw context)."""
    scn = context.scene if context else None
    if not scn or not hasattr(scn, "counters"):
        return
    node, _ = _find_counter_node(scn, self)
    if node:
        min_sock = node.inputs.get("Min Value")
        if min_sock:
            min_sock.hide = not bool(self.use_min_limit)


def _update_counter_max_limit(self, context):
    """Hide/show Max Value socket on the owning counter node (runs outside draw context)."""
    scn = context.scene if context else None
    if not scn or not hasattr(scn, "counters"):
        return
    node, _ = _find_counter_node(scn, self)
    if node:
        max_sock = node.inputs.get("Max Value")
        if max_sock:
            max_sock.hide = not bool(self.use_max_limit)


class CounterDefinition(bpy.types.PropertyGroup):
    """A named integer counter with optional min/max clamping."""

    name: bpy.props.StringProperty(default="Counter")
    description: bpy.props.StringProperty(default="")

    default_value: bpy.props.IntProperty(
        name="Default Value",
        default=0,
        description="Value at game start/reset"
    )

    current_value: bpy.props.IntProperty(
        name="Current Value",
        default=0,
        update=lambda self, context: self._clamp_current(context),
        description="Current counter value (auto-clamped if limits enabled)"
    )

    use_min_limit: bpy.props.BoolProperty(
        name="Enable Minimum",
        default=False,
        description="Clamp current_value to min_value if enabled",
        update=lambda self, ctx: _update_counter_min_limit(self, ctx),
    )
    min_value: bpy.props.IntProperty(
        name="Minimum Value",
        default=0,
        description="Lowest allowed current_value"
    )

    use_max_limit: bpy.props.BoolProperty(
        name="Enable Maximum",
        default=False,
        description="Clamp current_value to max_value if enabled",
        update=lambda self, ctx: _update_counter_max_limit(self, ctx),
    )
    max_value: bpy.props.IntProperty(
        name="Maximum Value",
        default=100,
        description="Highest allowed current_value"
    )

    def _clamp_current(self, context):
        """Force current_value into [min_value..max_value] if limits are enabled."""
        if self.use_min_limit and self.current_value < self.min_value:
            self.current_value = self.min_value
        if self.use_max_limit and self.current_value > self.max_value:
            self.current_value = self.max_value


# ─────────────────────────────────────────────────────────────
# Timer Definition
# ─────────────────────────────────────────────────────────────

class TimerDefinition(bpy.types.PropertyGroup):
    """A named timer that counts up or down."""

    name: bpy.props.StringProperty(default="Timer")
    description: bpy.props.StringProperty(default="")

    timer_mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ("COUNT_UP", "Count Up", "Timer counts upward from start to end"),
            ("COUNT_DOWN", "Count Down", "Timer counts downward from start to end"),
        ],
        default="COUNT_DOWN"
    )

    start_value: bpy.props.FloatProperty(
        name="Start Value",
        default=30.0,
        min=0.0,
        description="Timer value at start (seconds)"
    )

    end_value: bpy.props.FloatProperty(
        name="End Value",
        default=0.0,
        min=0.0,
        description="Timer value at completion (seconds)"
    )

    current_value: bpy.props.FloatProperty(
        name="Current Value",
        default=0.0,
        description="Current timer value (seconds)"
    )

    is_active: bpy.props.BoolProperty(
        name="Active",
        default=False,
        description="Whether the timer is currently running"
    )

    just_finished: bpy.props.BoolProperty(
        name="Just Finished",
        default=False,
        description="Set to True on the frame the timer completes"
    )

    prev_time: bpy.props.FloatProperty(default=0.0)

    def start(self, game_time: float):
        """Start/restart the timer from its start_value."""
        self.current_value = self.start_value
        self.prev_time = game_time
        self.is_active = True
        self.just_finished = False

    def stop(self):
        """Stop the timer."""
        self.is_active = False
        self.just_finished = False

    def is_complete(self) -> bool:
        """Check if timer has reached its end_value."""
        if self.timer_mode == "COUNT_UP":
            return self.current_value >= self.end_value
        else:
            return self.current_value <= self.end_value


# ─────────────────────────────────────────────────────────────
# Runtime Functions
# ─────────────────────────────────────────────────────────────

def update_all_timers(scene):
    """Update all active timers. Called each game loop tick."""
    now = get_game_time()

    for timer in scene.timers:
        if not timer.is_active:
            continue

        dt = now - timer.prev_time
        timer.prev_time = now

        if timer.timer_mode == "COUNT_UP":
            timer.current_value += dt
            if timer.current_value >= timer.end_value:
                timer.current_value = timer.end_value
                timer.just_finished = True
        else:  # COUNT_DOWN
            timer.current_value -= dt
            if timer.current_value <= timer.end_value:
                timer.current_value = timer.end_value
                timer.just_finished = True


def reset_all_counters(scene):
    """Reset all counters to their default values."""
    for counter in scene.counters:
        counter.current_value = counter.default_value


def reset_all_timers(scene):
    """Reset all timers to their start values and deactivate."""
    for timer in scene.timers:
        timer.is_active = False
        timer.current_value = timer.start_value
        timer.prev_time = 0.0
        timer.just_finished = False


# ─────────────────────────────────────────────────────────────
# Operators
# ─────────────────────────────────────────────────────────────

class EXPLORATORY_OT_AddCounter(bpy.types.Operator):
    bl_idname = "exploratory.add_counter"
    bl_label = "Add Counter"

    def execute(self, context):
        scn = context.scene
        new_item = scn.counters.add()
        new_item.name = f"Counter_{len(scn.counters)}"
        scn.counters_index = len(scn.counters) - 1
        return {'FINISHED'}


class EXPLORATORY_OT_RemoveCounter(bpy.types.Operator):
    bl_idname = "exploratory.remove_counter"
    bl_label = "Remove Counter"

    index: bpy.props.IntProperty()

    def execute(self, context):
        scn = context.scene
        if 0 <= self.index < len(scn.counters):
            scn.counters.remove(self.index)
            scn.counters_index = max(0, min(self.index, len(scn.counters) - 1))
        return {'FINISHED'}


class EXPLORATORY_OT_AddTimer(bpy.types.Operator):
    bl_idname = "exploratory.add_timer"
    bl_label = "Add Timer"

    def execute(self, context):
        scn = context.scene
        new_item = scn.timers.add()
        new_item.name = f"Timer_{len(scn.timers)}"
        scn.timers_index = len(scn.timers) - 1
        return {'FINISHED'}


class EXPLORATORY_OT_RemoveTimer(bpy.types.Operator):
    bl_idname = "exploratory.remove_timer"
    bl_label = "Remove Timer"

    index: bpy.props.IntProperty()

    def execute(self, context):
        scn = context.scene
        if 0 <= self.index < len(scn.timers):
            scn.timers.remove(self.index)
            scn.timers_index = max(0, min(self.index, len(scn.timers) - 1))
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────

def register_counter_timer_properties():
    bpy.utils.register_class(CounterDefinition)
    bpy.utils.register_class(TimerDefinition)
    bpy.utils.register_class(EXPLORATORY_OT_AddCounter)
    bpy.utils.register_class(EXPLORATORY_OT_RemoveCounter)
    bpy.utils.register_class(EXPLORATORY_OT_AddTimer)
    bpy.utils.register_class(EXPLORATORY_OT_RemoveTimer)

    bpy.types.Scene.counters = bpy.props.CollectionProperty(type=CounterDefinition)
    bpy.types.Scene.counters_index = bpy.props.IntProperty(default=0)
    bpy.types.Scene.timers = bpy.props.CollectionProperty(type=TimerDefinition)
    bpy.types.Scene.timers_index = bpy.props.IntProperty(default=0)


def unregister_counter_timer_properties():
    if hasattr(bpy.types.Scene, 'counters'):
        del bpy.types.Scene.counters
    if hasattr(bpy.types.Scene, 'counters_index'):
        del bpy.types.Scene.counters_index
    if hasattr(bpy.types.Scene, 'timers'):
        del bpy.types.Scene.timers
    if hasattr(bpy.types.Scene, 'timers_index'):
        del bpy.types.Scene.timers_index

    bpy.utils.unregister_class(EXPLORATORY_OT_RemoveTimer)
    bpy.utils.unregister_class(EXPLORATORY_OT_AddTimer)
    bpy.utils.unregister_class(EXPLORATORY_OT_RemoveCounter)
    bpy.utils.unregister_class(EXPLORATORY_OT_AddCounter)
    bpy.utils.unregister_class(TimerDefinition)
    bpy.utils.unregister_class(CounterDefinition)
