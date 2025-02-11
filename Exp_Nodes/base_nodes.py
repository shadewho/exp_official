import bpy
from bpy.types import Node

class TriggerNodeBase(bpy.types.Node):
    """Base class for trigger nodes."""
    bl_label = "Trigger Node Base"
    
    # Common properties (e.g., trigger delay) can be defined here.
    trigger_delay: bpy.props.FloatProperty(name="Trigger Delay", default=0.0)
    
    def execute_trigger(self, context):
        # This method should be overridden by concrete trigger nodes.
        self.report({'INFO'}, "Trigger executed from base class.")

class ReactionNodeBase(bpy.types.Node):
    """Base class for reaction nodes."""
    bl_label = "Reaction Node Base"
    
    def execute_reaction(self, context):
        # This method should be overridden by concrete reaction nodes.
        self.report({'INFO'}, "Reaction executed from base class.")
