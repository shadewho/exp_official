import bpy
from bpy.types import Node

import bpy
from bpy.types import Node

class TriggerNodeBase(bpy.types.Node):
    bl_label = "Trigger Node Base"
    
    def execute_trigger(self, context):
        self.report({'INFO'}, "Trigger executed from base class.")

class ReactionNodeBase(bpy.types.Node):
    """Base class for reaction nodes."""
    bl_label = "Reaction Node Base"
    
    def execute_reaction(self, context):
        # This method should be overridden by concrete reaction nodes.
        self.report({'INFO'}, "Reaction executed from base class.")
