# File: Exp_Nodes/base_nodes.py
import bpy
from bpy.types import Node

# ─────────────────────────────────────────────────────────
# Restrict ALL Exploratory nodes to ONLY our node tree type
# ─────────────────────────────────────────────────────────
class _ExploratoryNodeOnly:
    """
    Mixin to gate nodes to ExploratoryNodesTreeType only.

    We use both:
      • bl_tree -> filters compatibility so nodes don't appear in other trees' add menus
      • poll    -> extra safety if someone tries to instantiate anyway
    """
    bl_tree = 'ExploratoryNodesTreeType'

    @classmethod
    def poll(cls, ntree):
        return bool(ntree) and getattr(ntree, "bl_idname", "") == "ExploratoryNodesTreeType"


class TriggerNodeBase(_ExploratoryNodeOnly, bpy.types.Node):
    bl_label = "Trigger Node Base"

    def execute_trigger(self, context):
        # Optional hook in concrete trigger nodes
        self.report({'INFO'}, "Trigger executed from base class.")


class ReactionNodeBase(_ExploratoryNodeOnly, bpy.types.Node):
    """Base class for reaction nodes."""
    bl_label = "Reaction Node Base"

    def execute_reaction(self, context):
        # Optional hook in concrete reaction nodes
        self.report({'INFO'}, "Reaction executed from base class.")
