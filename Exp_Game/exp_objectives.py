# File: exp_objectives.py
import bpy
from .exp_time import get_game_time

# File: exp_objectives.py

class ObjectiveDefinition(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(default="Objective")
    description: bpy.props.StringProperty(default="")

    default_value: bpy.props.IntProperty(default=0)
    current_value: bpy.props.IntProperty(default=0)

    timer_mode: bpy.props.EnumProperty(
        name="Timer Mode",
        items=[("COUNT_UP","Count Up","..."),
               ("COUNT_DOWN","Count Down","...")],
        default="COUNT_DOWN"
    )

    timer_start_value: bpy.props.FloatProperty(default=0.0)
    timer_end_value:   bpy.props.FloatProperty(default=30.0)
    timer_value:       bpy.props.FloatProperty(default=0.0)
    timer_active:      bpy.props.BoolProperty(default=False)

    # We add this:
    just_finished: bpy.props.BoolProperty(default=False)

    prev_timer_time: bpy.props.FloatProperty(default=0.0)

    def start_timer(self, game_time: float):
        self.timer_value = self.timer_start_value
        self.prev_timer_time = game_time
        self.timer_active = True
        self.just_finished = False  # Always reset

    def stop_timer(self):
        self.timer_active = False
        self.just_finished = False

    def is_timer_complete(self) -> bool:
        """
        We'll keep this simple: return True if we've passed the boundary
        (and were active). We won't require timer_active to remain True.
        """
        # Option A: remove the requirement for timer_active entirely:
        if self.timer_mode == "COUNT_UP":
            return (self.timer_value >= self.timer_end_value)
        else:
            return (self.timer_value <= self.timer_end_value)



def update_all_objective_timers(scene):
    now = get_game_time()
    for objv in scene.objectives:
        if not objv.timer_active:
            # skip
            continue

        dt = now - objv.prev_timer_time
        objv.prev_timer_time = now
        old_val = objv.timer_value

        if objv.timer_mode == "COUNT_UP":
            objv.timer_value += dt
            # If we cross end_value => clamp + set just_finished
            if objv.timer_value >= objv.timer_end_value:
                objv.timer_value = objv.timer_end_value
                # DO NOT set timer_active=false here:
                # Instead, do:
                objv.just_finished = True
        else:  # COUNT_DOWN
            objv.timer_value -= dt
            if objv.timer_value <= objv.timer_end_value:
                objv.timer_value = objv.timer_end_value
                objv.just_finished = True

def reset_all_objectives(scene):
    """
    At the start of the modal, reset each Objective so that
    timer_value = timer_start_value, timer_active = False,
    and current_value = default_value.
    """
    for objv in scene.objectives:
        objv.timer_active = False
        objv.timer_value  = objv.timer_start_value
        objv.prev_timer_time = 0.0
        objv.current_value = objv.default_value  # Now resetting the counter




class EXPLORATORY_OT_AddObjective(bpy.types.Operator):
    bl_idname = "exploratory.add_objective"
    bl_label = "Add Objective"

    def execute(self, context):
        scn = context.scene
        new_item = scn.objectives.add()
        new_item.name = f"Objective_{len(scn.objectives)}"
        scn.objectives_index = len(scn.objectives) - 1
        return {'FINISHED'}


class EXPLORATORY_OT_RemoveObjective(bpy.types.Operator):
    bl_idname = "exploratory.remove_objective"
    bl_label = "Remove Objective"

    index: bpy.props.IntProperty()

    def execute(self, context):
        scn = context.scene
        if 0 <= self.index < len(scn.objectives):
            scn.objectives.remove(self.index)
            scn.objectives_index = max(0, min(self.index, len(scn.objectives) - 1))
        return {'FINISHED'}


def register_objective_properties():
    bpy.utils.register_class(ObjectiveDefinition)
    bpy.utils.register_class(EXPLORATORY_OT_AddObjective)
    bpy.utils.register_class(EXPLORATORY_OT_RemoveObjective)
    bpy.types.Scene.objectives = bpy.props.CollectionProperty(type=ObjectiveDefinition)
    bpy.types.Scene.objectives_index = bpy.props.IntProperty(default=0)

def unregister_objective_properties():
    del bpy.types.Scene.objectives
    del bpy.types.Scene.objectives_index
    bpy.utils.unregister_class(ObjectiveDefinition)
    bpy.utils.unregister_class(EXPLORATORY_OT_AddObjective)
    bpy.utils.unregister_class(EXPLORATORY_OT_RemoveObjective)
