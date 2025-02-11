import bpy
from .base_nodes import TriggerNodeBase

class KeyPressTriggerNode(TriggerNodeBase):
    """A node that triggers when a specific key is pressed."""
    bl_idname = 'KeyPressTriggerNodeType'
    bl_label = 'Key Press Trigger'
    
    trigger_key: bpy.props.StringProperty(name="Trigger Key", default="SPACE")
    
    def init(self, context):
        self.outputs.new('NodeSocketFloat', "Trigger Out")
    
    def draw_buttons(self, context, layout):
        layout.prop(self, "trigger_key", text="Key")
        layout.prop(self, "trigger_delay", text="Delay")
    
    def execute_trigger(self, context):
        print(f"KeyPressTriggerNode: Trigger fired for key {self.trigger_key} after delay {self.trigger_delay}.")
        if self.outputs:
            self.outputs[0].default_value = 1.0
