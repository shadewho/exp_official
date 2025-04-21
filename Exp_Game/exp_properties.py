# File: exp_properties.py

import bpy
import os

bpy.types.Action.action_speed = bpy.props.FloatProperty(
    name="Action Speed",
    description="Speed multiplier for this action",
    default=1.0,
    min=0.1,
    max=5.0
)
bpy.types.Sound.sound_speed = bpy.props.FloatProperty(
    name="Sound Speed",
    description="Speed multiplier for this sound",
    default=1.0,
    min=0.1,
    max=5.0
)

def list_actions(self, context):
    items = []
    for act in bpy.data.actions:
        items.append((act.name, act.name, ""))
    if not items:
        items.append(("None","No Actions Found",""))
    return items

###---------------proxy meshes-------------###
class ProxyMeshEntry(bpy.types.PropertyGroup):
    """
    One entry in our list of 'proxy meshes'.
    Each entry has a name and a pointer to an object.
    """
    name: bpy.props.StringProperty(
        name="Name",
        default="ProxyMesh"
    )

    mesh_object: bpy.props.PointerProperty(
        name="Mesh Object",
        type=bpy.types.Object,
        description="Mesh used for collisions"
    )

    is_moving: bpy.props.BoolProperty(
        name="Is Moving",
        default=False,
        description="If True, this mesh may move each frame and we’ll handle it differently."
    )

    register_distance: bpy.props.FloatProperty(
        name="Register Distance",
        default=0.0,
        min=0.0,
        description="Distance from the player at which this dynamic mesh is active. 0 = always active."
    )
    
    hide_during_game: bpy.props.BoolProperty(
        name="Hide During Game",
        default=False,
        description="If enabled, the proxy mesh will be hidden during gameplay."
    )
class EXPLORATORY_UL_ProxyMeshList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            # Show only the name field:
            layout.label(text=item.name)
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.name)

class EXPLORATORY_OT_AddProxyMesh(bpy.types.Operator):
    """Add a new ProxyMeshEntry to the scene's proxy_meshes list."""
    bl_idname = "exploratory.add_proxy_mesh"
    bl_label = "Add Proxy Mesh"

    def execute(self, context):
        scene = context.scene
        new_item = scene.proxy_meshes.add()
        new_item.name = f"ProxyMesh_{len(scene.proxy_meshes)}"
        scene.proxy_meshes_index = len(scene.proxy_meshes) - 1
        return {'FINISHED'}


class EXPLORATORY_OT_RemoveProxyMesh(bpy.types.Operator):
    """Remove a ProxyMeshEntry from scene.proxy_meshes."""
    bl_idname = "exploratory.remove_proxy_mesh"
    bl_label = "Remove Proxy Mesh"

    index: bpy.props.IntProperty()

    def execute(self, context):
        scene = context.scene
        if 0 <= self.index < len(scene.proxy_meshes):
            scene.proxy_meshes.remove(self.index)
            scene.proxy_meshes_index = max(0, min(self.index, len(scene.proxy_meshes) - 1))
        return {'FINISHED'}


# ------------------------------------------------------------------------
# 2) add_scene_properties() => animation
# ------------------------------------------------------------------------
def add_scene_properties():
    # ================
    #Explore/Studio Enum
    bpy.types.Scene.main_category = bpy.props.EnumProperty(
    name="Main Category",
    description="Select between Explore and Create modes",
    items=[
        ("EXPLORE", "Explore", "Show Explore UI"),
        ("CREATE", "Create", "Show Create UI")
    ],
    default="EXPLORE"
)

    #---proxy mesh --#
    bpy.types.Scene.proxy_meshes = bpy.props.CollectionProperty(type=ProxyMeshEntry)
    bpy.types.Scene.proxy_meshes_index = bpy.props.IntProperty(default=0)


    # ================
    #ANIMATION, Armature, Spawn Object
    # ===============
    bpy.types.Scene.spawn_object = bpy.props.PointerProperty(
        name="Spawn Object",
        type=bpy.types.Object,
        description="Which object is used as the spawn location?"
    )
    bpy.types.Scene.target_armature = bpy.props.PointerProperty(
        name="Target Armature",
        type=bpy.types.Object,
        description="Armature (or main character object) for the Third Person system"
    )

    bpy.types.Scene.orbit_distance = bpy.props.FloatProperty(
        name="Orbit Distance",
        default=2.0,
        min=1.0,    
        max=10.0
    )
    bpy.types.Scene.zoom_factor = bpy.props.FloatProperty(
        name="Zoom Factor",
        default=2.0,
        min=0.1,
        max=15.0
    )
    bpy.types.Scene.pitch_angle = bpy.props.FloatProperty(
        name="Pitch Angle",
        default=15.0,
        min=-180.0,
        max=180.0
    )
    # How far the camera pulls in when avoiding obstacles
    bpy.types.Scene.camera_collision_buffer = bpy.props.FloatProperty(
        name="Collision Buffer",
        default=4.1,
        min=0.0,
        description="Meters to pull the camera toward the character when a collision is detected"
    )

    ###AUDIO GLOBALS###
    bpy.types.Scene.enable_audio = bpy.props.BoolProperty(
        name="Enable Audio",
        description="If off, do not load/play any sounds at all",
        default=True
    )

    bpy.types.Scene.audio_level = bpy.props.FloatProperty(
        name="Audio Volume",
        description="Master volume from 0.0 to 1.0",
        default=0.5,
        min=0.0,
        max=1.0
    )

class CharacterActionsPG(bpy.types.PropertyGroup):
    """
    This property group will hold pointers to Actions instead of storing their names.
    """
    # Example for each typical slot:
    idle_action: bpy.props.PointerProperty(
        name="Idle Action",
        type=bpy.types.Action,
        description="Action to use for the idle state"
    )
    walk_action: bpy.props.PointerProperty(
        name="Walk Action",
        type=bpy.types.Action,
        description="Action to use for the walk state"
    )
    run_action: bpy.props.PointerProperty(
        name="Run Action",
        type=bpy.types.Action,
        description="Action to use for the run state"
    )
    jump_action: bpy.props.PointerProperty(
        name="Jump Action",
        type=bpy.types.Action,
        description="Action to use for the jump state"
    )
    fall_action: bpy.props.PointerProperty(
        name="Fall Action",
        type=bpy.types.Action,
        description="Action to use for the fall state"
    )
    land_action: bpy.props.PointerProperty(
        name="Land Action",
        type=bpy.types.Action,
        description="Action to use for the land state"
    )

def remove_scene_properties():
    del bpy.types.Scene.target_armature
    del bpy.types.Scene.spawn_object
    del bpy.types.Scene.orbit_distance
    del bpy.types.Scene.zoom_factor
    del bpy.types.Scene.pitch_angle

    del bpy.types.Scene.proxy_meshes
    del bpy.types.Scene.proxy_meshes_index

    del bpy.types.Scene.enable_audio
    del bpy.types.Scene.audio_level