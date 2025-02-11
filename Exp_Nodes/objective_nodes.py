import bpy
from bpy.types import Node

# Callback that creates an enum list from scene.objectives
def enum_objective_items(self, context):
    scene = context.scene
    items = []
    if hasattr(scene, "objectives"):
        for i, obj in enumerate(scene.objectives):
            # Each item is a tuple: (identifier as string, name, description)
            items.append((str(i), obj.name, f"Objective: {obj.name}"))
    if not items:
        items.append(("0", "No Objectives", ""))
    return items

class ObjectiveNode(bpy.types.Node):
    """An Objective Node that lets you select an objective from the scene
       and displays its properties similar to the N panel."""
    bl_idname = "ObjectiveNodeType"
    bl_label = "Objective Node"
    bl_icon = 'TRACKER'  # Use a valid icon (e.g., "TRACKER" appears in Blender's list)

    # An enum property that lists all available objectives
    objective_item: bpy.props.EnumProperty(
         name="Objective",
         items=enum_objective_items,
         update=lambda self, context: self.update_objective(context)
    )
    # Also store the index as an integer for convenience
    objective_index: bpy.props.IntProperty(name="Objective Index", default=-1)

    def init(self, context):
        # This node has no sockets.
        self.width = 300

        # Set the default objective if available.
        items = enum_objective_items(self, context)
        if items:
            self.objective_item = items[0][0]
            try:
                self.objective_index = int(items[0][0])
            except:
                self.objective_index = -1

    def update_objective(self, context):
        try:
            self.objective_index = int(self.objective_item)
        except Exception as e:
            print("Error updating objective index:", e)
            self.objective_index = -1

    def draw_buttons(self, context, layout):
        # Display an enum dropdown for selecting an objective.
        layout.label(text="Select Objective:")
        layout.prop(self, "objective_item", text="")  # Dropdown without label

        # Place operator buttons to add and remove objectives.
        row = layout.row(align=True)
        row.operator("exploratory.add_objective", text="", icon='ADD')
        op = row.operator("exploratory.remove_objective", text="", icon='TRASH')
        op.index = self.objective_index  # Pass the current objective index to the removal operator

        # If the selected objective is valid, display its details.
        scene = context.scene
        if hasattr(scene, "objectives") and 0 <= self.objective_index < len(scene.objectives):
            objv = scene.objectives[self.objective_index]
            box = layout.box()
            box.prop(objv, "name", text="Name")
            box.prop(objv, "description", text="Description")
            box.separator()
            box.label(text=f"Default Value: {objv.default_value}")
            box.label(text=f"Current Value: {objv.current_value}")
            box.separator()
            box.prop(objv, "timer_mode", text="Timer Mode")
            box.prop(objv, "timer_start_value", text="Start Value")
            box.prop(objv, "timer_end_value", text="End Value")
            box.label(text=f"Current Timer: {objv.timer_value:.1f}")
            box.label(text=f"Timer Active?: {objv.timer_active}")
        else:
            layout.label(text="No valid objective found", icon='ERROR')

    def draw_label(self):
        # Return the name of the selected objective if possible.
        scene = bpy.context.scene
        if hasattr(scene, "objectives") and 0 <= self.objective_index < len(scene.objectives):
            return scene.objectives[self.objective_index].name
     
