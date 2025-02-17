import bpy
from .base_nodes import TriggerNodeBase
# Import the objective items callback used elsewhere in your add-on.
from .objective_nodes import enum_objective_items  

class TriggerNode(TriggerNodeBase):
    bl_idname = 'TriggerNodeType'
    bl_label = 'Trigger Node'
    bl_icon = 'HAND'
    
    # Stores the index (in scene.custom_interactions) of the linked InteractionDefinition.
    interaction_owner: bpy.props.IntProperty(name="Owner Index", default=-1)
    
    # NEW: Each trigger node now holds its own objective selection as an EnumProperty.
    node_objective_index: bpy.props.EnumProperty(
        name="Objective",
        description="Select which objective to update",
        items=enum_objective_items,
        update=lambda self, context: self.update_objective_index(context)
    )
    
    def update_objective_index(self, context):
        """
        Called whenever the node's objective selection changes.
        If the node is connected to an InteractionDefinition whose trigger type is OBJECTIVE_UPDATE,
        then update that InteractionDefinition's objective_index with this node's selection.
        """
        if self.inputs["Interaction"].links:
            scene = context.scene
            if (hasattr(scene, "custom_interactions") and 
                0 <= self.interaction_owner < len(scene.custom_interactions)):
                inter = scene.custom_interactions[self.interaction_owner]
                inter.objective_index = self.node_objective_index
                print(f"Updated interaction objective_index to {self.node_objective_index}")
    
    def init(self, context):
        # Create an input socket to receive data from an Interaction Node.
        self.inputs.new('InteractionSocketType', "Interaction")
        # Create an output socket for the trigger signal.
        self.outputs.new('TriggerOutputSocketType', "Trigger Output")
        self.width = 300
    
    def update(self):
        """
        Automatically called when the node is updated.
        If the "Interaction" input is connected, retrieve the linked node’s interaction_index.
        For objective update triggers, immediately override the connected interaction’s objective_index
        with the node's own selection.
        """
        if self.inputs["Interaction"].links:
            linked_node = self.inputs["Interaction"].links[0].from_node
            if hasattr(linked_node, "interaction_index"):
                self.interaction_owner = linked_node.interaction_index
                print(f"Trigger node updated: interaction_owner = {self.interaction_owner}")
                # If the connected interaction is an OBJECTIVE_UPDATE trigger,
                # update its objective_index with this node's selection.
                scene = bpy.context.scene
                if (hasattr(scene, "custom_interactions") and 
                    0 <= self.interaction_owner < len(scene.custom_interactions)):
                    inter = scene.custom_interactions[self.interaction_owner]
                    if inter.trigger_type == "OBJECTIVE_UPDATE":
                        inter.objective_index = self.node_objective_index
            else:
                print("Connected node does not have an interaction_index!")
        else:
            print("Trigger node not connected to any Interaction Node!")
    
    def execute_trigger(self, context):
        # Loop through outgoing links from the "Trigger Output" socket and execute reactions.
        if "Trigger Output" in self.outputs:
            for link in self.outputs["Trigger Output"].links:
                reaction_node = link.to_node
                if hasattr(reaction_node, "execute_reaction"):
                    reaction_node.execute_reaction(context)
    
    def draw_buttons(self, context, layout):
        """
        Draw the UI for this trigger node.
        When the connected interaction's trigger type is OBJECTIVE_UPDATE,
        show the objective dropdown (using this node's node_objective_index property).
        """
        if self.inputs["Interaction"].links:
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
                    # Display the objective dropdown using the node's own property.
                    layout.prop(self, "node_objective_index", text="Objective")
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
