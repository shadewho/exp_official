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
       and displays (or edits) its properties similar to the N panel."""
    bl_idname = "ObjectiveNodeType"
    bl_label = "Objective Node"
    bl_icon = 'TRACKER'
    
    # An enum property that lists all available objectives.
    objective_item: bpy.props.EnumProperty(
         name="Objective",
         items=enum_objective_items,
         update=lambda self, context: self.update_objective(context)
    )
    # Also store the index as an integer for convenience.
    objective_index: bpy.props.IntProperty(name="Objective Index", default=-1)
    
    def init(self, context):
        self.width = 300
        # Initialize the objective based on current scene objectives, if any.
        items = enum_objective_items(self, context)
        if items:
            # Select the first objective by default
            self.objective_item = items[0][0]
            try:
                self.objective_index = int(items[0][0])
            except:
                self.objective_index = -1
    
    def update_objective(self, context):
        try:
            self.objective_index = int(self.objective_item)
            print(f"Objective updated: {self.objective_item} => {self.objective_index}")
        except Exception as e:
            print("Error updating objective index:", e)
            self.objective_index = -1

    def draw_buttons(self, context, layout):
        layout.label(text="Select Objective:")
        layout.prop(self, "objective_item", text="")  # Draw dropdown without a label

        # Add and Remove operators
        row = layout.row(align=True)
        row.operator("exploratory.add_objective", text="", icon='ADD')
        op = row.operator("exploratory.remove_objective", text="", icon='TRASH')
        op.index = self.objective_index

        # Display editable properties from the chosen objective:
        scene = context.scene
        if hasattr(scene, "objectives") and 0 <= self.objective_index < len(scene.objectives):
            objv = scene.objectives[self.objective_index]
            box = layout.box()
            box.prop(objv, "name", text="Name")
            box.prop(objv, "description", text="Description")
            box.separator()
            box.prop(objv, "default_value", text="Default Value")
            box.prop(objv, "current_value", text="Current Value")
            box.separator()
            box.prop(objv, "timer_mode", text="Timer Mode")
            box.prop(objv, "timer_start_value", text="Start Value")
            box.prop(objv, "timer_end_value", text="End Value")
            box.label(text=f"Current Timer: {objv.timer_value:.1f}")
            box.label(text=f"Timer Active?: {objv.timer_active}")
        else:
            layout.label(text="No valid objective found", icon='ERROR')

    def draw_label(self):
        scene = bpy.context.scene
        if hasattr(scene, "objectives") and 0 <= self.objective_index < len(scene.objectives):
            return scene.objectives[self.objective_index].name
        return "Objective Node"

    # Override the copy method so that duplicating this node creates a new objective.
    def copy(self, node):
        scene = bpy.context.scene

        # Add a new objective to the scene's objectives collection.
        new_obj = scene.objectives.add()
        new_obj.name = f"Objective_{len(scene.objectives)}"
        new_obj.description = ""  # Optionally, you could copy from the original if desired.
        # Set any default values you need, for example:
        new_obj.default_value = 0
        new_obj.current_value = 0
        # You can also initialize timer settings if needed.

        # Assign the new objective's index to this duplicated node.
        self.objective_index = len(scene.objectives) - 1
        self.objective_item = str(self.objective_index)
        print(f"Duplicated Objective Node now uses objective index {self.objective_index}")
