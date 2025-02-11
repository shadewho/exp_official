# Exp_Nodes/interaction_nodes.py
import bpy
from bpy.types import Node
# Import your game’s InteractionDefinition from your game module.
# Adjust the relative import as needed.
from ..Exp_Game.interaction_definition import InteractionDefinition

class InteractionNode(bpy.types.Node):
    """A node representing an interaction.
    Its stored index (interaction_index) is used to look up the corresponding interaction
    in the scene's custom_interactions collection. The node displays the interaction’s
    name and description only when such an interaction is linked."""
    bl_idname = 'InteractionNodeType'
    bl_label = 'Interaction Node'
    bl_icon = 'ACTION'
    
    # This property stores the index of the interaction in scene.custom_interactions
    interaction_index: bpy.props.IntProperty(
        name="Interaction Index",
        default=-1
    )
    
    
    def init(self, context):
        # Create an output socket.
        self.outputs.new('NodeSocketFloat', "Output")
        # Do not automatically create an interaction here.
        # We let the user press the button to add/link one.
    
    def draw_buttons(self, context, layout):
        scene = context.scene
        # Check if the scene has custom_interactions and the stored index is valid.
        if hasattr(scene, "custom_interactions") and (0 <= self.interaction_index < len(scene.custom_interactions)):
            inter = scene.custom_interactions[self.interaction_index]
            # Display the interaction’s name and description.
            layout.label(text="Name: " + inter.name)
            layout.label(text="Description: " + inter.description)
            # (Optionally, you could use layout.prop() to allow editing.)
            # For example:
            # layout.prop(inter, "name", text="Name")
            # layout.prop(inter, "description", text="Description")
        else:
            # No valid interaction is linked.
            layout.label(text="No interaction linked")
            # Provide a button to add a new interaction.
            # This calls your existing operator that creates a new interaction.
            layout.operator("exploratory.add_interaction", text="Add Interaction", icon='ADD')
    
    def draw_label(self):
        # If there is a linked interaction, use its name; otherwise, use a default.
        scene = bpy.context.scene
        if hasattr(scene, "custom_interactions") and (0 <= self.interaction_index < len(scene.custom_interactions)):
            return scene.custom_interactions[self.interaction_index].name
        else:
            return "Interaction Node"
