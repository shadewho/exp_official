# File: exp_properties.py

import bpy

# —————————————
# Helpers & UI classes (unchanged)
# —————————————

def list_actions(self, context):
    """Build a list of all actions for dropdowns."""
    items = [(act.name, act.name, "") for act in bpy.data.actions]
    if not items:
        items.append(("None", "No Actions Found", ""))
    return items

class ProxyMeshEntry(bpy.types.PropertyGroup):
    """One entry in our list of 'proxy meshes'."""
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
        description="If True, this mesh may move each frame"
    )
    register_distance: bpy.props.FloatProperty(
        name="Register Distance",
        default=0.0, min=0.0,
        description="Active within this distance (0=always)"
    )
    hide_during_game: bpy.props.BoolProperty(
        name="Hide During Game",
        default=False,
        description="If enabled, proxy hides in gameplay"
    )

class EXPLORATORY_UL_ProxyMeshList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        text = item.name
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.label(text=text)
        else:
            layout.alignment = 'CENTER'
            layout.label(text=text)

class EXPLORATORY_OT_AddProxyMesh(bpy.types.Operator):
    bl_idname = "exploratory.add_proxy_mesh"
    bl_label  = "Add Proxy Mesh"
    def execute(self, context):
        sc = context.scene
        new = sc.proxy_meshes.add()
        new.name = f"ProxyMesh_{len(sc.proxy_meshes)}"
        sc.proxy_meshes_index = len(sc.proxy_meshes) - 1
        return {'FINISHED'}

class EXPLORATORY_OT_RemoveProxyMesh(bpy.types.Operator):
    bl_idname = "exploratory.remove_proxy_mesh"
    bl_label  = "Remove Proxy Mesh"
    index: bpy.props.IntProperty()
    def execute(self, context):
        sc = context.scene
        if 0 <= self.index < len(sc.proxy_meshes):
            sc.proxy_meshes.remove(self.index)
            sc.proxy_meshes_index = max(0, min(self.index, len(sc.proxy_meshes)-1))
        return {'FINISHED'}

class CharacterActionsPG(bpy.types.PropertyGroup):
    """Holds pointers to character action tracks."""
    idle_action: bpy.props.PointerProperty(
        name="Idle Action", type=bpy.types.Action
    )
    walk_action: bpy.props.PointerProperty(
        name="Walk Action", type=bpy.types.Action
    )
    run_action : bpy.props.PointerProperty(
        name="Run Action", type=bpy.types.Action
    )
    jump_action: bpy.props.PointerProperty(
        name="Jump Action", type=bpy.types.Action
    )
    fall_action: bpy.props.PointerProperty(
        name="Fall Action", type=bpy.types.Action
    )
    land_action: bpy.props.PointerProperty(
        name="Land Action", type=bpy.types.Action
    )

# —————————————
# Scene-level prop definitions
# —————————————

def add_scene_properties():
    """Register all bpy.types.Scene.* properties."""
    sc = bpy.types.Scene

    sc.main_category = bpy.props.EnumProperty(
        name="Main Category",
        description="Explore vs Create mode",
        items=[
            ("EXPLORE","Explore","Browse worlds"),
            ("CREATE","Create","Build world")
        ],
        default="EXPLORE"
    )

    sc.proxy_meshes       = bpy.props.CollectionProperty(type=ProxyMeshEntry)
    sc.proxy_meshes_index = bpy.props.IntProperty(default=0)

    sc.spawn_object             = bpy.props.PointerProperty(type=bpy.types.Object)
    sc.spawn_use_nearest_z_surface = bpy.props.BoolProperty(
        name="Find Nearest Z Surface", default=True
    )
    sc.target_armature = bpy.props.PointerProperty(type=bpy.types.Object)

    sc.character_spawn_lock   = bpy.props.BoolProperty(default=False)
    sc.character_actions_lock = bpy.props.BoolProperty(default=False)
    sc.character_audio_lock   = bpy.props.BoolProperty(default=False)

    sc.orbit_distance           = bpy.props.FloatProperty(default=2.0, min=1.0, max=10.0)
    sc.zoom_factor              = bpy.props.FloatProperty(default=2.0, min=0.1, max=15.0)
    sc.pitch_angle              = bpy.props.FloatProperty(default=15.0, min=-180.0, max=180.0)
    sc.camera_collision_buffer  = bpy.props.FloatProperty(default=4.1, min=0.0)

def remove_scene_properties():
    """Remove all bpy.types.Scene.* properties added above."""
    attrs = (
        "main_category",
        "proxy_meshes", "proxy_meshes_index",
        "spawn_object", "spawn_use_nearest_z_surface", "target_armature",
        "character_spawn_lock", "character_actions_lock", "character_audio_lock",
        "orbit_distance", "zoom_factor", "pitch_angle", "camera_collision_buffer",
    )
    for name in attrs:
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)

# —————————————
# Module-level register/unregister
# —————————————

def register_props():
    """Call this in your init.register() to install all props."""
    # 1) Action & Sound extras
    bpy.types.Action.action_speed = bpy.props.FloatProperty(
        name="Action Speed",
        description="Speed multiplier",
        default=1.0, min=0.1, max=5.0
    )
    bpy.types.Sound.sound_speed  = bpy.props.FloatProperty(
        name="Sound Speed",
        description="Speed multiplier",
        default=1.0, min=0.1, max=5.0
    )

    # 2) Scene props
    add_scene_properties()

def unregister_props():
    """Call this in your init.unregister() to tear down all props."""
    # 1) remove scene props
    remove_scene_properties()

    # 2) remove Action & Sound extras
    if hasattr(bpy.types.Action, "action_speed"):
        delattr(bpy.types.Action, "action_speed")
    if hasattr(bpy.types.Sound, "sound_speed"):
        delattr(bpy.types.Sound, "sound_speed")
