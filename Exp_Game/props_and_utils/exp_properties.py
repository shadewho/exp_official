# File: exp_properties.py

import bpy
import math

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
        default=-21.0, min=-50.0, max=0.0
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
        default=30, min=10, max=30
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


    # ─────────────────────────────────────────────────────────
    # "Filter Create Panels" - multi-select flags enum
    # ─────────────────────────────────────────────────────────
    bpy.types.Scene.create_panels_filter = bpy.props.EnumProperty(
        name="Filter 'Create' Panels",
        description="Show only selected panels in the Create section",
        items=[
            ("CHAR",   "Character / Actions / Audio", "Character, actions, and audio panel"),
            ("PROXY",  "Proxy Mesh & Spawn",          "Proxy mesh list and spawn settings"),
            ("UPLOAD", "Upload Helper",               "Six-step upload helper"),
            ("PERF",   "Performance",                 "Live performance + culling tools"),
            ("PHYS",   "Character Physics & View",    "Kinematic controller & view tuning"),
            ("DEV",    "Developer Tools",             "Debug toggles and diagnostic tools"),
        ],
        options={'ENUM_FLAG'},
        default={'CHAR', 'PROXY', 'UPLOAD', 'PERF', 'PHYS', 'DEV'}
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
    bpy.types.Scene.pitch_angle = bpy.props.FloatProperty(
        name="Pitch Angle",
        default=15.0,
        min=-180.0,
        max=180.0
    )

    bpy.types.Scene.view_obstruction_enabled = bpy.props.BoolProperty(
        name="View Obstruction",
        description="If OFF, camera obstruction checks are completely disabled (no raycasts or dynamic map used).",
        default=True
    )
    bpy.types.Scene.view_mode = bpy.props.EnumProperty(
        name="View Mode",
        description="Third-person orbit, first-person, or locked camera",
        items=[
            ('THIRD',  "Third Person", "Orbit behind character"),
            ('FIRST',  "First Person", "Camera at head height (no orbit)"),
            ('LOCKED', "Locked",       "Fixed yaw/pitch + distance; no obstruction logic"),
        ],
        default='THIRD'
    )
    bpy.types.Scene.view_locked_move_axis = bpy.props.EnumProperty(
        name="Locked Move Axis",
        description=(
            "In LOCKED view, constrain character movement to a global axis. "
            "When set, W/S are ignored and only A/D move along the selected global axis."
        ),
        items=[
            ('OFF', "Off", "No axis constraint"),
            ('X',   "Global X (A⇄D)", "Left/Right move strictly along ±X"),
            ('Y',   "Global Y (A⇄D)", "Left/Right move strictly along ±Y"),
        ],
        default='OFF'
    )

    # LOCKED + Axis: optional 180° facing flip
    bpy.types.Scene.view_locked_flip_axis = bpy.props.BoolProperty(
        name="Flip Axis Direction (180°)",
        description="In LOCKED view with Axis Lock, reverse the facing along the selected world axis.",
        default=False,
    )
    bpy.types.Scene.view_locked_pitch = bpy.props.FloatProperty(
        name="Pitch (°)", subtype='ANGLE', unit='ROTATION',
        default=math.radians(60.0), soft_min=math.radians(-89.9), soft_max=math.radians(89.9),
        description="Positive = look up, negative = look down"
    )
    bpy.types.Scene.view_locked_yaw = bpy.props.FloatProperty(
        name="Yaw (°)", subtype='ANGLE', unit='ROTATION',
        default=0.0, soft_min=-math.pi, soft_max= math.pi,
        description="0° = -Y, 90° = +X, 180° = +Y, -90° = -X (Blender view coords)"
    )
    bpy.types.Scene.view_locked_distance = bpy.props.FloatProperty(
        name="Distance",
        default=6.0, min=0.0, soft_max=50.0,
        description="Boom length along the locked direction from the head anchor"
    )

    bpy.types.Scene.view_projection = bpy.props.EnumProperty(
        name="Projection",
        description="Viewport projection mode",
        items=[('PERSP', "Perspective", ""), ('ORTHO', "Orthographic", "")],
        default='PERSP'
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
    bpy.types.Scene.viewport_lens_mm = bpy.props.FloatProperty(
        name="Viewport Lens (mm)",
        description="3D View lens in millimeters for all VIEW_3D areas while playing",
        default=55.0, min=1.0, max=300.0
    )

    # First-person view: which pose bone to aim with camera pitch
    bpy.types.Scene.fpv_view_bone = bpy.props.StringProperty(
        name="FPV Target Bone",
        description="Pose bone on the Target Armature that receives camera pitch in FIRST view",
        default=""
    )
    bpy.types.Scene.fpv_invert_pitch = bpy.props.BoolProperty(
        name="Invert FPV Pitch",
        description="Invert mapping of mouse Y to FPV bone pitch",
        default=False
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
    # Safe property deletion with hasattr checks
    if hasattr(bpy.types.Scene, 'target_armature'):
        del bpy.types.Scene.target_armature
    if hasattr(bpy.types.Scene, 'spawn_object'):
        del bpy.types.Scene.spawn_object
    if hasattr(bpy.types.Scene, 'orbit_distance'):
        del bpy.types.Scene.orbit_distance
    if hasattr(bpy.types.Scene, 'zoom_factor'):
        del bpy.types.Scene.zoom_factor
    if hasattr(bpy.types.Scene, 'pitch_angle'):
        del bpy.types.Scene.pitch_angle

    if hasattr(bpy.types.Scene, 'proxy_meshes'):
        del bpy.types.Scene.proxy_meshes
    if hasattr(bpy.types.Scene, 'proxy_meshes_index'):
        del bpy.types.Scene.proxy_meshes_index
    if hasattr(bpy.types.Scene, 'create_panels_filter'):
        del bpy.types.Scene.create_panels_filter
    if hasattr(bpy.types.Scene, 'view_mode'):
        del bpy.types.Scene.view_mode
    if hasattr(bpy.types.Scene, 'view_locked_pitch'):
        del bpy.types.Scene.view_locked_pitch
    if hasattr(bpy.types.Scene, 'view_locked_yaw'):
        del bpy.types.Scene.view_locked_yaw
    if hasattr(bpy.types.Scene, 'view_locked_distance'):
        del bpy.types.Scene.view_locked_distance
    if hasattr(bpy.types.Scene, 'view_projection'):
        del bpy.types.Scene.view_projection
    if hasattr(bpy.types.Scene, 'view_locked_move_axis'):
        del bpy.types.Scene.view_locked_move_axis