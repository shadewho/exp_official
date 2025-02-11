import bpy
from .base_nodes import TriggerNodeBase

class TriggerNode(TriggerNodeBase):
    bl_idname = 'TriggerNodeType'
    bl_label = 'Trigger Node'
    bl_icon = 'HAND'
    
    # This property stores the owner (interaction) index.
    interaction_owner: bpy.props.IntProperty(name="Owner Index", default=-1)
    
    def init(self, context):
        # Create an input socket to receive the Interaction Node’s data.
        self.inputs.new('InteractionSocketType', "Interaction")
        # Create an output socket for the trigger signal using our custom purple socket.
        self.outputs.new('TriggerOutputSocketType', "Trigger Output")
        self.width = 300
    
    def update(self):
        """
        Automatically called when the node is updated.
        It checks whether the "Interaction" input socket is connected.
        If connected, it reads the linked node's interaction_index
        and stores it into this node's interaction_owner property.
        """
        if self.inputs["Interaction"].links:
            linked_node = self.inputs["Interaction"].links[0].from_node
            if hasattr(linked_node, "interaction_index"):
                self.interaction_owner = linked_node.interaction_index
                print(f"Trigger node updated: interaction_owner = {self.interaction_owner}")
            else:
                print("Connected node does not have an interaction_index!")
        else:
            print("Trigger node not connected to any Interaction Node!")
    
    def execute_trigger(self, context):
        # Loop through all outgoing links from the "Trigger Output" socket
        if "Trigger Output" in self.outputs:
            for link in self.outputs["Trigger Output"].links:
                reaction_node = link.to_node
                if hasattr(reaction_node, "execute_reaction"):
                    reaction_node.execute_reaction(context)
    
    def draw_buttons(self, context, layout):
        """
        In the UI we simply read the already updated interaction_owner.
        We do not attempt to write during drawing.
        """
        if self.inputs["Interaction"].links:
            # Read the already updated property.
            owner_index = self.interaction_owner
            scene = context.scene
            if (hasattr(scene, "custom_interactions") and 
                0 <= owner_index < len(scene.custom_interactions)):
                inter = scene.custom_interactions[owner_index]
                layout.label(text="Trigger Data:")
                layout.prop(inter, "trigger_type", text="Trigger Type")
                if inter.trigger_type == "PROXIMITY":
                    layout.prop(inter, "proximity_distance", text="Distance")
                    layout.prop(inter, "proximity_object_a", text="Object A")
                    layout.prop(inter, "proximity_object_b", text="Object B")
                elif inter.trigger_type == "COLLISION":
                    layout.prop(inter, "collision_object_a", text="Object A")
                    layout.prop(inter, "collision_object_b", text="Object B")
                elif inter.trigger_type == "INTERACT":
                    layout.prop(inter, "interact_object", text="Interact Object")
                    layout.prop(inter, "interact_distance", text="Distance")
                elif inter.trigger_type == "OBJECTIVE_UPDATE":
                    layout.prop_search(inter, "objective_index", scene, "objectives", text="Objective")
                    layout.prop(inter, "objective_condition", text="Condition")
                    if inter.objective_condition in {"EQUALS", "AT_LEAST"}:
                        layout.prop(inter, "objective_condition_value", text="Value")
                elif inter.trigger_type == "TIMER_COMPLETE":
                    layout.prop(inter, "timer_objective_index", text="Timer Objective")
                
                layout.separator()
                layout.label(text="Trigger Options:")
                layout.prop(inter, "trigger_mode", text="Mode")
                if inter.trigger_mode == "COOLDOWN":
                    layout.prop(inter, "trigger_cooldown", text="Cooldown")
                layout.prop(inter, "trigger_delay", text="Delay (sec)")
            else:
                layout.label(text="Invalid interaction data", icon='ERROR')
        else:
            layout.label(text="No Interaction Connected", icon='ERROR')
    
    def draw_label(self):
        return "Trigger Node"

# Registration functions (optional for testing as a standalone module)
def register():
    bpy.utils.register_class(TriggerNode)

def unregister():
    bpy.utils.unregister_class(TriggerNode)

if __name__ == "__main__":
    register()
