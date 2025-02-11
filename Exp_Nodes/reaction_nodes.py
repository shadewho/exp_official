import bpy
from .base_nodes import ReactionNodeBase

# Define a custom socket for the Reaction Node input ("Trigger Input")
class ReactionTriggerInputSocket(bpy.types.NodeSocket):
    bl_idname = "ReactionTriggerInputSocketType"
    bl_label = "Reaction Trigger Input Socket"
    
    def draw(self, context, layout, node, text):
        layout.label(text=text)
    
    def draw_color(self, context, node):
        # Lighter blue (RGB: 0.4, 0.4, 1.0)
        return (0.4, 0.4, 1.0, 1.0)

# Define a custom socket for the Reaction Node output ("Output")
class ReactionOutputSocket(bpy.types.NodeSocket):
    bl_idname = "ReactionOutputSocketType"
    bl_label = "Reaction Output Socket"
    
    def draw(self, context, layout, node, text):
        layout.label(text=text)
    
    def draw_color(self, context, node):
        # Light green (for example, RGB: 0.7, 1.0, 0.7)
        return (0.7, 1.0, 0.7, 1.0)

class ReactionNode(ReactionNodeBase):
    bl_idname = "ReactionNodeType"
    bl_label = "Reaction Node"
    bl_icon = 'MODIFIER'
    
    def init(self, context):
        # Create one input socket named "Trigger Input" using our custom blue socket.
        self.inputs.new("ReactionTriggerInputSocketType", "Trigger Input")
        # Create one output socket named "Output" using our custom light-green socket.
        self.outputs.new("ReactionOutputSocketType", "Output")
        self.width = 300
    
    def draw_buttons(self, context, layout):
        # 1. Check if a trigger is connected.
        trigger_socket = self.inputs.get("Trigger Input")
        if not trigger_socket or not trigger_socket.links:
            layout.label(text="No trigger connected", icon='ERROR')
            return

        # 2. Get the trigger node (the first link)
        trigger_node = trigger_socket.links[0].from_node
        if not hasattr(trigger_node, "interaction_owner"):
            layout.label(text="Trigger missing interaction data", icon='ERROR')
            return

        # 3. Retrieve the owner index from the trigger node.
        owner_index = trigger_node.interaction_owner
        scene = context.scene

        # 4. Validate the index against scene.custom_interactions.
        if owner_index < 0 or owner_index >= len(scene.custom_interactions):
            layout.label(text="Invalid interaction index", icon='ERROR')
            return

        # 5. Get the corresponding interaction.
        inter = scene.custom_interactions[owner_index]
        
        # 6. Check if there are any reactions added.
        if not inter.reactions or len(inter.reactions) == 0:
            layout.label(text="No reactions added", icon='ERROR')
            # Provide an operator button to add a reaction:
            layout.operator("exploratory.add_reaction_to_interaction", text="Add Reaction", icon='ADD')
            return

        # 7. If there is reaction data, check that the reactions_index is valid.
        r_idx = inter.reactions_index
        if r_idx < 0 or r_idx >= len(inter.reactions):
            layout.label(text="Select a reaction", icon='INFO')
            return

        # 8. Get the active reaction and display its properties.
        reaction = inter.reactions[r_idx]
        box = layout.box()
        box.label(text="Current Reaction", icon='OBJECT_DATA')
        box.prop(reaction, "name", text="Name")
        box.prop(reaction, "reaction_type", text="Type")
        
        if reaction.reaction_type == "CUSTOM_ACTION":
            box.prop(reaction, "custom_action_message", text="Notes")
            box.prop_search(reaction, "custom_action_target", bpy.context.scene, "objects", text="Object")
            box.prop_search(reaction, "custom_action_action", bpy.data, "actions", text="Action")
            box.prop(reaction, "custom_action_loop", text="Loop?")
            if reaction.custom_action_loop:
                box.prop(reaction, "custom_action_loop_duration", text="Loop Duration")
        elif reaction.reaction_type == "CHAR_ACTION":
            box.prop_search(reaction, "char_action_ref", bpy.data, "actions", text="Action")
            box.prop(reaction, "char_action_mode", text="Mode")
            if reaction.char_action_mode == 'LOOP':
                box.prop(reaction, "char_action_loop_duration", text="Loop Duration")
        elif reaction.reaction_type == "OBJECTIVE_COUNTER":
            box.prop(reaction, "objective_index", text="Objective")
            box.prop(reaction, "objective_op", text="Operation")
            if reaction.objective_op in ("ADD", "SUBTRACT"):
                box.prop(reaction, "objective_amount", text="Amount")
        elif reaction.reaction_type == "PROPERTY":
            box.prop(reaction, "property_data_path", text="Data Path")
            row = box.row()
            row.label(text=f"Detected Type: {reaction.property_type}")
            box.prop(reaction, "property_transition_duration", text="Duration")
            box.prop(reaction, "property_reset", text="Reset?")
            if reaction.property_reset:
                box.prop(reaction, "property_reset_delay", text="Reset Delay")
            if reaction.property_type == "BOOL":
                box.prop(reaction, "bool_value", text="New Bool Value")
            elif reaction.property_type == "INT":
                box.prop(reaction, "int_value", text="New Int Value")
            elif reaction.property_type == "FLOAT":
                box.prop(reaction, "float_value", text="New Float Value")
            elif reaction.property_type == "STRING":
                box.prop(reaction, "string_value", text="New String Value")
            elif reaction.property_type == "VECTOR":
                box.label(text=f"Vector length: {reaction.vector_length}")
                box.prop(reaction, "vector_value", text="New Vector")
            else:
                box.label(text="No property detected or invalid path.")
        elif reaction.reaction_type == "TRANSFORM":
            box.prop_search(reaction, "transform_object", bpy.context.scene, "objects", text="Object")
            box.prop(reaction, "transform_mode", text="Mode")
            if reaction.transform_mode == "TO_OBJECT":
                box.prop_search(reaction, "transform_to_object", bpy.context.scene, "objects", text="To Object")
            if reaction.transform_mode in {"OFFSET", "TO_LOCATION", "LOCAL_OFFSET"}:
                box.prop(reaction, "transform_location", text="Location")
                box.prop(reaction, "transform_rotation", text="Rotation")
                box.prop(reaction, "transform_scale", text="Scale")
            box.prop(reaction, "transform_duration", text="Duration")
            box.prop(reaction, "transform_distance", text="Distance")
        elif reaction.reaction_type == "CUSTOM_UI_TEXT":
            box.prop(reaction, "custom_text_subtype", text="Subtype")
            if reaction.custom_text_subtype == "STATIC":
                box.prop(reaction, "custom_text_value", text="Text")
                box.prop(reaction, "custom_text_indefinite", text="Indefinite?")
                if not reaction.custom_text_indefinite:
                    box.prop(reaction, "custom_text_duration", text="Duration")
                box.prop(reaction, "custom_text_anchor", text="Anchor")
                box.prop(reaction, "custom_text_scale", text="Scale")
                box.prop(reaction, "custom_text_margin_x", text="Margin X")
                box.prop(reaction, "custom_text_margin_y", text="Margin Y")
                box.prop(reaction, "custom_text_color", text="Color")
            elif reaction.custom_text_subtype == "OBJECTIVE":
                box.prop(reaction, "text_objective_index", text="Objective")
                box.prop(reaction, "text_objective_format", text="Format")
                box.prop(reaction, "custom_text_indefinite", text="Indefinite?")
                if not reaction.custom_text_indefinite:
                    box.prop(reaction, "custom_text_duration", text="Duration")
                box.prop(reaction, "custom_text_anchor", text="Anchor")
                box.prop(reaction, "custom_text_scale", text="Scale")
                box.prop(reaction, "custom_text_margin_x", text="Margin X")
                box.prop(reaction, "custom_text_margin_y", text="Margin Y")
                box.prop(reaction, "custom_text_color", text="Color")
            elif reaction.custom_text_subtype == "OBJECTIVE_TIMER_DISPLAY":
                box.prop(reaction, "text_objective_index", text="Objective")
                box.prop(reaction, "custom_text_indefinite", text="Indefinite?")
                if not reaction.custom_text_indefinite:
                    box.prop(reaction, "custom_text_duration", text="Duration")
                box.prop(reaction, "custom_text_anchor", text="Anchor")
                box.prop(reaction, "custom_text_scale", text="Scale")
                box.prop(reaction, "custom_text_margin_x", text="Margin X")
                box.prop(reaction, "custom_text_margin_y", text="Margin Y")
                box.prop(reaction, "custom_text_color", text="Color")
        elif reaction.reaction_type == "OBJECTIVE_TIMER":
            box.prop(reaction, "objective_index", text="Timer Objective")
            box.prop(reaction, "objective_timer_op", text="Timer Operation")
        elif reaction.reaction_type == "MOBILITY_GAME":
            mg = reaction.mobility_game_settings
            box.prop(mg, "allow_movement", text="Allow Movement")
            box.prop(mg, "allow_jump", text="Allow Jump")
            box.prop(mg, "allow_sprint", text="Allow Sprint")
        elif reaction.reaction_type == "SOUND":
            box.label(text="Play Packed Sound")
            box.prop(reaction, "sound_volume", text="Relative Volume")
            box.prop(reaction, "sound_use_distance", text="Use Distance?")
            box.prop(reaction, "sound_pointer", text="Sound Datablock")
            box.prop(reaction, "sound_play_mode", text="Mode")
            if reaction.sound_play_mode == "DURATION":
                box.prop(reaction, "sound_duration", text="Duration")
            if reaction.sound_use_distance:
                box.prop(reaction, "sound_distance_object", text="Distance Obj")
                box.prop(reaction, "sound_max_distance", text="Max Distance")
        else:
            box.label(text="Reaction type not recognized", icon='ERROR')
    
    def draw_label(self):
        return "Reaction Node"

# Registration functions for this file
def register():
    bpy.utils.register_class(ReactionTriggerInputSocket)
    bpy.utils.register_class(ReactionOutputSocket)
    bpy.utils.register_class(ReactionNode)

def unregister():
    bpy.utils.unregister_class(ReactionNode)
    bpy.utils.unregister_class(ReactionOutputSocket)
    bpy.utils.unregister_class(ReactionTriggerInputSocket)

if __name__ == "__main__":
    register()
