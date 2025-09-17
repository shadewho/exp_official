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


class CharacterPhysicsConfigPG(bpy.types.PropertyGroup):
    radius: bpy.props.FloatProperty(
        name="Capsule Radius",
        description="Collider radius in meters. Larger values make the character wider and can prevent squeezing through tight gaps.",
        default=0.3, min=0.05, max=1.0
    )
    height: bpy.props.FloatProperty(
        name="Capsule Height",
        description="Total capsule height in meters from feet to head. Must be tall enough to avoid ceiling collisions.",
        default=2.0, min=0.8, max=3.0
    )
    slope_limit_deg: bpy.props.FloatProperty(
        name="Slope Limit (°)",
        description="Maximum ground slope (degrees from horizontal) considered walkable. Steeper surfaces are treated as too steep.",
        default=50.0, min=0.0, max=89.0
    )
    step_height: bpy.props.FloatProperty(
        name="Step Height",
        description="Max ledge height (meters) the controller will auto-step up while grounded (no jump required).",
        default=0.45, min=0.0, max=0.8
    )
    snap_down: bpy.props.FloatProperty(
        name="Snap Distance",
        description="Downward snap distance (meters per step) to keep the character glued to ground over small dips and edges.",
        default=0.20, min=0.0, max=2.0
    )
    gravity: bpy.props.FloatProperty(
        name="Gravity m/s²",
        description="Downward acceleration applied when not grounded. Use a negative value (e.g. -9.81).",
        default=-20.0, min=-50.0, max=0.0
    )
    max_speed_walk: bpy.props.FloatProperty(
        name="Walk Speed",
        description="Target maximum horizontal speed (m/s) when walking.",
        default=3.0, min=0.1, max=20.0
    )
    max_speed_run: bpy.props.FloatProperty(
        name="Run Speed",
        description="Target maximum horizontal speed (m/s) when running (run key held).",
        default=9.0, min=0.1, max=30.0
    )
    accel_ground: bpy.props.FloatProperty(
        name="Ground Accel",
        description="Rate of horizontal acceleration toward the target speed while on ground. Higher = snappier control.",
        default=20.0, min=1.0, max=100.0
    )
    accel_air: bpy.props.FloatProperty(
        name="Air Accel",
        description="Rate of horizontal acceleration while airborne. Lower = floatier, higher = tighter midair control.",
        default=8.0, min=0.0, max=100.0
    )
    coyote_time: bpy.props.FloatProperty(
        name="Coyote Time",
        description="Late-jump grace period (seconds) after stepping off a ledge during which pressing jump still triggers a jump.",
        default=0.10, min=0.0, max=0.5
    )
    jump_buffer: bpy.props.FloatProperty(
        name="Jump Buffer",
        description="Early-jump grace period (seconds): if jump is pressed slightly before landing, it will trigger on landing.",
        default=0.10, min=0.0, max=0.5
    )
    jump_speed: bpy.props.FloatProperty(
        name="Jump Speed",
        description="Initial upward velocity (m/s) applied when a jump starts. Higher values yield higher jumps.",
        default=8.0, min=0.0, max=30.0
    )
    fixed_hz: bpy.props.IntProperty(
        name="Physics Hz",
        description="Fixed physics update frequency (steps per second). Higher values are smoother but cost more CPU.",
        default=30, min=30, max=30
    )


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
    bpy.types.Scene.char_physics = bpy.props.PointerProperty(type=CharacterPhysicsConfigPG)

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
    bpy.types.Scene.spawn_use_nearest_z_surface = bpy.props.BoolProperty(
        name="Find Nearest Z Surface",
        default=True,
        description="If enabled and spawn object is a mesh, place character on the nearest surface along global Z"
    )

    bpy.types.Scene.target_armature = bpy.props.PointerProperty(
        name="Target Armature",
        type=bpy.types.Object,
        description="Armature (or main character object) for the Third Person system"
    )
    #-----LOCKS LOCKS LOCKS --------#
    bpy.types.Scene.character_spawn_lock = bpy.props.BoolProperty(
        name="Lock Character Spawn",
        default=False,
        description="If on, building or removing the character is disabled"
    )
    bpy.types.Scene.character_actions_lock = bpy.props.BoolProperty(
        name="Lock Character Actions",
        default=False,
        description="If on, building or changing the character’s actions is disabled"
    )
    # Lock out automatic audio appending
    bpy.types.Scene.character_audio_lock = bpy.props.BoolProperty(
        name="Lock Character Audio",
        default=False,
        description="If on, building or changing the character’s audio is disabled"
    )

    bpy.types.Scene.orbit_distance = bpy.props.FloatProperty(
        name="Orbit Distance",
        default=2.0,
        min=1.0,    
        max=10.0
    )
    bpy.types.Scene.zoom_factor = bpy.props.FloatProperty(
        name="Zoom Factor",
        default=4.0,
        min=0.1,
        max=15.0
    )
    bpy.types.Scene.pitch_angle = bpy.props.FloatProperty(
        name="Pitch Angle",
        default=15.0,
        min=-180.0,
        max=180.0
    )

    # Toggle: Live Performance Overlay (drawn in the 3D Viewport)
    bpy.types.Scene.show_live_performance_overlay = bpy.props.BoolProperty(
        name="Live Performance Overlay",
        description="Show a live performance meter in the viewport while playing",
        default=False
    )
    # Live Performance HUD tuning
    bpy.types.Scene.live_perf_scale = bpy.props.IntProperty(
        name="HUD Scale",
        description="Smaller values make the single-row HUD less intrusive",
        default=2, min=1, max=4
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
    del bpy.types.Scene.show_live_performance_overlay