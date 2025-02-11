# Exp_Nodes/node_editor.py
import bpy
from bpy.types import NodeTree, Node, NodeSocket
from nodeitems_utils import NodeCategory, NodeItem, register_node_categories, unregister_node_categories
# -----------------------------
# Define the custom node tree
# -----------------------------
class ExploratoryNodesTree(bpy.types.NodeTree):
    """A custom node tree for the Exploratory Node Editor"""
    bl_idname = 'ExploratoryNodesTreeType'
    bl_label = 'Exploratory Node Editor Tree'
    bl_icon = 'NODETREE'

# -----------------------------
# Define a custom socket (optional)
# -----------------------------

# NEW: Custom socket for Interaction connections (blue)
class InteractionSocket(bpy.types.NodeSocket):
    bl_idname = 'InteractionSocketType'
    bl_label = 'Interaction Socket'
    
    def draw(self, context, layout, node, text):
        layout.label(text=text)
    
    def draw_color(self, context, node):
        # Blue color (you can adjust the values as you like)
        return (0.4, 0.4, 1.0, 1.0)

# NEW: Custom socket for Trigger Node outputs (purple)
class TriggerOutputSocket(bpy.types.NodeSocket):
    bl_idname = 'TriggerOutputSocketType'
    bl_label = 'Trigger Output Socket'
    
    def draw(self, context, layout, node, text):
        layout.label(text=text)
    
    def draw_color(self, context, node):
        # Purple color (for example: 0.5 red, 0.0 green, 0.5 blue)
        return (0.8, 0.3, 0.8, 1.0)


# -----------------------------
# Define Node Categories for the Shift+A menu
# -----------------------------
node_categories = [
    NodeCategory("TRIGGERS", "Triggers", items=[
        NodeItem("TriggerNodeType"),
    ]),
    NodeCategory("REACTIONS", "Reactions", items=[
        NodeItem("ReactionNodeType"),
    ]),
    NodeCategory("INTERACTIONS", "Interactions", items=[
        NodeItem("InteractionNodeType"),
    ]),
    NodeCategory("OBJECTIVES", "Objectives", items=[
        NodeItem("ObjectiveNodeType"),
    ]),
]

# -----------------------------
# Operator and Panel for the Node Editor sidebar
# -----------------------------
class NODE_OT_create_exploratory_node_tree(bpy.types.Operator):
    bl_idname = "node.create_exploratory_node_tree"
    bl_label = "Create Exploratory Node Tree"
    
    def execute(self, context):
        new_tree = bpy.data.node_groups.new("Exploratory Node Tree", 'ExploratoryNodesTreeType')
        # Give the new node tree a fake user to prevent it from being removed automatically.
        new_tree.use_fake_user = True
        
        for area in context.screen.areas:
            if area.type == 'NODE_EDITOR':
                space = area.spaces.active
                space.tree_type = 'ExploratoryNodesTreeType'
                space.node_tree = new_tree
                break
        self.report({'INFO'}, "Exploratory Node Tree created (fake user enabled).")
        return {'FINISHED'}

# ----------------------------------------------------------
# Operator: Delete an Exploratory Node Tree with Confirmation
# ----------------------------------------------------------
class NODE_OT_delete_exploratory_node_tree(bpy.types.Operator):
    """Delete the selected Exploratory Node Tree (with confirmation)"""
    bl_idname = "node.delete_exploratory_node_tree"
    bl_label = "Delete Exploratory Node Tree"

    tree_name: bpy.props.StringProperty()

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        node_tree = bpy.data.node_groups.get(self.tree_name)
        if node_tree:
            bpy.data.node_groups.remove(node_tree)
            self.report({'INFO'}, f"Deleted node tree: {self.tree_name}")
        else:
            self.report({'WARNING'}, "Node tree not found")
        return {'FINISHED'}

# ----------------------------------------------------------
# Panel: Exploratory Node Editor with Node Tree List
# ----------------------------------------------------------
class NODE_PT_exploratory_panel(bpy.types.Panel):
    bl_label = "Exploratory Node Editor"
    bl_idname = "NODE_PT_exploratory_panel"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    
    @classmethod
    def poll(cls, context):
        # Show this panel only when the active node tree is of our custom type.
        return context.space_data.tree_type == 'ExploratoryNodesTreeType'
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        # --- Existing UI: Create Node Tree & instructions ---
        layout.operator("node.create_exploratory_node_tree", icon='NODETREE')
        layout.separator()
        layout.label(text="Use Shift+A to add nodes by category.")
        
        # --- New Section: List all Node Trees ---
        layout.separator()
        layout.label(text="Existing Node Trees:")
        col = layout.column(align=True)
        for nt in bpy.data.node_groups:
            if nt.bl_idname == "ExploratoryNodesTreeType":
                # Wrap each entry in a box so that it has a colored background.
                box = col.box()
                row = box.row(align=True)
                row.label(text=nt.name)
                # Delete button: when pressed, a confirmation will pop up.
                del_op = row.operator("node.delete_exploratory_node_tree", text="", icon='TRASH')
                del_op.tree_name = nt.name


class NODE_PT_exploratory_proxy(bpy.types.Panel):
    bl_label = "Proxy Meshes"
    bl_idname = "NODE_PT_exploratory_proxy"
    bl_space_type = 'NODE_EDITOR'  # Changed from 'VIEW_3D' to 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    @classmethod
    def poll(cls, context):
        return context.scene.main_category == 'CREATE'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.separator()
        layout.label(text="Proxy Meshes")
        row = layout.row()
        row.template_list(
            "EXPLORATORY_UL_ProxyMeshList",  # Your UIList class name
            "",
            scene,
            "proxy_meshes",          # The collection property
            scene,
            "proxy_meshes_index",    # The integer property for the active item
            rows=4
        )

        # The side column with add/remove operators
        col = row.column(align=True)
        col.operator("exploratory.add_proxy_mesh", text="", icon='ADD')
        remove_op = col.operator("exploratory.remove_proxy_mesh", text="", icon='REMOVE')
        remove_op.index = scene.proxy_meshes_index

        layout.separator()

        # Show details for the currently selected proxy mesh
        idx = scene.proxy_meshes_index
        if 0 <= idx < len(scene.proxy_meshes):
            entry = scene.proxy_meshes[idx]
            box = layout.box()
            box.label(text="Selected Proxy Mesh Details:")
            box.prop(entry, "name", text="Name")
            box.prop(entry, "mesh_object", text="Mesh")
            box.prop(entry, "is_moving", text="Is Moving?")

        layout.separator()
        layout.label(text="Spawn Object")
        layout.prop(scene, "spawn_object", text="")


class NODE_PT_character_actions_panel(bpy.types.Panel):
    bl_label = "Character, Actions, and Audio"
    bl_idname = "NODE_PT_character_actions_panel"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        # Show this panel when our custom node tree is active.
        return context.space_data.tree_type == 'ExploratoryNodesTreeType'
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.separator()
        layout.label(text="Character, Actions, and Audio:")

        # Target Armature and Animation Slots
        layout.prop(scene, "target_armature", text="Target Armature")
        layout.label(text="Animation Slots")
        layout.prop(scene.character_actions, "idle_action")
        layout.prop(scene.character_actions, "walk_action")
        layout.prop(scene.character_actions, "run_action")
        layout.prop(scene.character_actions, "jump_action")
        layout.prop(scene.character_actions, "fall_action")
        layout.prop(scene.character_actions, "land_action")

        layout.separator()
        layout.label(text="Action Speeds:")
        char_actions = scene.character_actions
        layout.prop(char_actions, "idle_speed")
        layout.prop(char_actions, "walk_speed")
        layout.prop(char_actions, "run_speed")
        layout.prop(char_actions, "jump_speed")
        layout.prop(char_actions, "fall_speed")
        layout.prop(char_actions, "land_speed")

        layout.separator()
        layout.label(text="Audio Control:")
        layout.prop(scene, "enable_audio", text="Enable Audio")
        layout.prop(scene, "audio_level", text="Master Volume")

        layout.label(text="Audio Pointers:")
        row = layout.row()
        row.prop(scene.character_audio, "walk_sound", text="Walk Sound")
        op = row.operator("exp_audio.test_sound_pointer", text="Test")
        op.sound_slot = "walk_sound"

        row = layout.row()
        row.prop(scene.character_audio, "run_sound", text="Run Sound")
        op = row.operator("exp_audio.test_sound_pointer", text="Test")
        op.sound_slot = "run_sound"

        row = layout.row()
        row.prop(scene.character_audio, "jump_sound", text="Jump Sound")
        op = row.operator("exp_audio.test_sound_pointer", text="Test")
        op.sound_slot = "jump_sound"

        row = layout.row()
        row.prop(scene.character_audio, "fall_sound", text="Fall Sound")
        op = row.operator("exp_audio.test_sound_pointer", text="Test")
        op.sound_slot = "fall_sound"

        row = layout.row()
        row.prop(scene.character_audio, "land_sound", text="Land Sound")
        op = row.operator("exp_audio.test_sound_pointer", text="Test")
        op.sound_slot = "land_sound"