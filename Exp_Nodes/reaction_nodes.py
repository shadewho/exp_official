import bpy
from .base_nodes import ReactionNodeBase

class PlaySoundReactionNode(ReactionNodeBase):
    """A node that triggers playing a sound."""
    bl_idname = 'PlaySoundReactionNodeType'
    bl_label = 'Play Sound Reaction'
    
    sound_name: bpy.props.StringProperty(name="Sound Name", default="DefaultSound")
    volume: bpy.props.FloatProperty(name="Volume", default=1.0, min=0.0, max=1.0)
    
    def init(self, context):
        self.inputs.new('NodeSocketFloat', "Trigger In")
        self.outputs.new('NodeSocketFloat', "Output")
        
    def draw_buttons(self, context, layout):
        layout.prop(self, "sound_name", text="Sound")
        layout.prop(self, "volume", text="Volume")
    
    def execute_reaction(self, context):
        print(f"PlaySoundReactionNode: Playing sound '{self.sound_name}' at volume {self.volume}.")
        # Placeholder for integrating your game logic later.
