import bpy
from bpy.types import Node
from ..Exp_Game.exp_interactions import InteractionDefinition

class InteractionNode(bpy.types.Node):
    """A node representing an interaction.
    When no valid interaction is linked, it shows a button that calls an operator
    to create a new interaction. Once linked, it displays the interaction’s name
    and description (from scene.custom_interactions)."""
    bl_idname = "InteractionNodeType"
    bl_label = "Interaction Node"
    bl_icon = "ACTION"
    
    # This property stores the index of the interaction in scene.custom_interactions.
    interaction_index: bpy.props.IntProperty(
        name="Interaction Index",
        default=-1
    )
    
    def init(self, context):
        self.interaction_index = -1  # no interaction linked yet
        # Create an output socket using our custom InteractionSocket.
        self.outputs.new("InteractionSocketType", "Output")
        self.width = 300
    
    def draw_buttons(self, context, layout):
        scene = context.scene
        if self.interaction_index < 0 or not (hasattr(scene, "custom_interactions") and (0 <= self.interaction_index < len(scene.custom_interactions))):
            layout.label(text="No interaction linked")
            layout.operator("node.add_interaction_to_node", text="Add Interaction", icon='ADD')
        else:
            inter = scene.custom_interactions[self.interaction_index]
            layout.prop(inter, "name", text="Name")
            layout.prop(inter, "description", text="Description")
    
    def draw_label(self):
        scene = bpy.context.scene
        if self.interaction_index >= 0 and hasattr(scene, "custom_interactions") and (0 <= self.interaction_index < len(scene.custom_interactions)):
            return scene.custom_interactions[self.interaction_index].name
        else:
            return "Interaction Node"


# Operator to add a new interaction and link it to the active Interaction Node.
class NODE_OT_add_interaction_to_node(bpy.types.Operator):
    """Add a new interaction to the scene and link it to the active Interaction Node"""
    bl_idname = "node.add_interaction_to_node"
    bl_label = "Add Interaction to Node"
    
    def execute(self, context):
        node_tree = context.space_data.edit_tree
        active_node = node_tree.nodes.active
        if not active_node or active_node.bl_idname != "InteractionNodeType":
            self.report({'WARNING'}, "Active node is not an Interaction Node.")
            return {'CANCELLED'}
        scene = context.scene
        if not hasattr(scene, "custom_interactions"):
            self.report({'WARNING'}, "Scene does not have custom_interactions!")
            return {'CANCELLED'}
        # Create a new interaction using your standard logic.
        new_inter = scene.custom_interactions.add()
        new_inter.name = "Interaction_%d" % len(scene.custom_interactions)
        new_inter.description = ""
        new_inter.trigger_type = "PROXIMITY"  # Adjust default as needed.
        # Update the node's stored index.
        active_node.interaction_index = len(scene.custom_interactions) - 1
        self.report({'INFO'}, "New interaction added and linked.")
        return {'FINISHED'}