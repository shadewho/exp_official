#interaction_nodes.py
import bpy
from bpy.types import Node
from ..Exp_Game.interactions.exp_interactions import InteractionDefinition

# --- Callback for populating the dropdown list from scene.custom_interactions ---
def enum_interaction_items_callback(self, context):
    scene = context.scene
    items = []
    if hasattr(scene, "custom_interactions"):
        for i, inter in enumerate(scene.custom_interactions):
            # Each item is a tuple: (identifier, name, description)
            items.append((str(i), inter.name, ""))
    if not items:
        items.append(("0", "No Interaction", ""))
    return items

# --- Update callback for when the dropdown selection changes ---
def update_interaction_index(self, context):
    try:
        self.interaction_index = int(self.interaction_item)
    except Exception:
        self.interaction_index = -1

# --- The Interaction Node ---
class InteractionNode(bpy.types.Node):
    """A node representing an interaction.
    Displays a dropdown listing scene interactions plus +/– buttons.
    The selected item is the node’s associated interaction."""
    bl_idname = "InteractionNodeType"
    bl_label = "Interaction Node"
    bl_icon = "ACTION"
    
    # Internal storage of the index (as an integer)
    interaction_index: bpy.props.IntProperty(
        name="Interaction Index",
        default=-1
    )
    # EnumProperty for the dropdown; its items are generated dynamically
    interaction_item: bpy.props.EnumProperty(
        name="Interaction",
        items=enum_interaction_items_callback,
        update=update_interaction_index
    )
    
    def init(self, context):
        # Initialize with no interaction linked
        self.interaction_index = -1
        self.interaction_item = "0"  # default (even if there is no interaction yet)
        # Create an output socket (using your custom socket type)
        self.outputs.new("InteractionSocketType", "Output")
        self.width = 300  # Make the node wider

    def copy(self, node):
        # When duplicating, create a new interaction (so the duplicated node isn't linked to the original)
        scene = bpy.context.scene
        new_inter = scene.custom_interactions.add()
        new_inter.name = "Interaction_%d" % len(scene.custom_interactions)
        new_inter.description = ""
        new_inter.trigger_type = "PROXIMITY"  # Or set your default trigger type here
        self.interaction_index = len(scene.custom_interactions) - 1
        self.interaction_item = str(self.interaction_index)
    def draw_buttons(self, context, layout):
        scene = context.scene

        # Create a split row: 15% for buttons, 85% for the dropdown.
        row = layout.row(align=True)
        split = row.split(factor=0.15)
        
        # Left column: Skinny add and delete buttons.
        col_buttons = split.column(align=True)
        col_buttons.operator("node.add_interaction_to_node", text="", icon='ADD')
        col_buttons.operator("node.remove_interaction_from_node", text="", icon='TRASH')
        
        # Right column: The dropdown listing interactions.
        col_dropdown = split.column(align=True)
        col_dropdown.prop(self, "interaction_item", text="")  # No label so it stays compact
        
        # Optionally, display additional details below.
        if (self.interaction_index >= 0 and hasattr(scene, "custom_interactions") and 
            (0 <= self.interaction_index < len(scene.custom_interactions))):
            inter = scene.custom_interactions[self.interaction_index]
            layout.prop(inter, "name", text="Name")
            layout.prop(inter, "description", text="Description")
        else:
            layout.label(text="No interaction linked", icon='INFO')




    def draw_label(self):
        scene = bpy.context.scene
        if (self.interaction_index >= 0 and hasattr(scene, "custom_interactions") and
            (0 <= self.interaction_index < len(scene.custom_interactions))):
            return scene.custom_interactions[self.interaction_index].name
        else:
            return "Interaction Node"

# --- Operator to Add a New Interaction ---
class NODE_OT_add_interaction_to_node(bpy.types.Operator):
    """Add a new interaction and link it to this node"""
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
        new_inter = scene.custom_interactions.add()
        new_inter.name = "Interaction_%d" % len(scene.custom_interactions)
        new_inter.description = ""
        new_inter.trigger_type = "PROXIMITY"  # Set default as needed
        active_node.interaction_index = len(scene.custom_interactions) - 1
        active_node.interaction_item = str(active_node.interaction_index)
        self.report({'INFO'}, "New interaction added and linked.")
        return {'FINISHED'}

# --- Operator to Remove the Selected Interaction ---
class NODE_OT_remove_interaction_from_node(bpy.types.Operator):
    """Remove the currently selected interaction from the node (with confirmation)"""
    bl_idname = "node.remove_interaction_from_node"
    bl_label = "Remove Interaction from Node"
    
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)
    
    def execute(self, context):
        node_tree = context.space_data.edit_tree
        active_node = node_tree.nodes.active
        if not active_node or active_node.bl_idname != "InteractionNodeType":
            self.report({'WARNING'}, "Active node is not an Interaction Node.")
            return {'CANCELLED'}
        scene = context.scene
        idx = active_node.interaction_index
        if idx < 0 or idx >= len(scene.custom_interactions):
            self.report({'WARNING'}, "No valid interaction to remove.")
            return {'CANCELLED'}
        # Remove using your existing removal operator call (or do it directly)
        bpy.ops.exploratory.remove_interaction(index=idx)
        active_node.interaction_index = -1
        active_node.interaction_item = "0"
        context.area.tag_redraw()
        self.report({'INFO'}, "Interaction removed.")
        return {'FINISHED'}
