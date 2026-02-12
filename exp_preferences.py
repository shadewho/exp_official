import bpy
from bpy.props import BoolProperty
import os

# ------------------------------------------------------------------------
# Utility: Where is this addon?
# ------------------------------------------------------------------------
def get_addon_path():
    """
    Return the absolute path to this add-on's folder (where __init__.py is).
    """
    import inspect
    return os.path.dirname(inspect.getfile(get_addon_path))



# ------------------------------------------------------------------------
# 1) Keybind Operator
# ------------------------------------------------------------------------
class EXPLORATORY_OT_SetKeybind(bpy.types.Operator):
    """Operator that temporarily listens for the next pressed key and updates the add-on preference property."""
    bl_idname = "exploratory.set_keybind"
    bl_label = "Set Keybind"
    bl_description = "Click then press the key you want to assign."

    target_prop: bpy.props.StringProperty(default="")

    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Press any key to assign...")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type in {'MOUSEMOVE','INBETWEEN_MOUSEMOVE'}:
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS':
            prefs = context.preferences.addons["Exploratory"].preferences
            if hasattr(prefs, self.target_prop):
                setattr(prefs, self.target_prop, event.type)
                self.report({'INFO'}, f"Keybind set to: {event.type}")
            else:
                self.report({'WARNING'}, f"Property not found: {self.target_prop}")
            return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE','ESC'}:
            self.report({'INFO'}, "Cancelled.")
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

# ------------------------------------------------------------------------
# Update Callback: Start Game Keymap
# ------------------------------------------------------------------------
def _update_start_game_keymap(self, context):
    """
    Called when the user changes the Start Game keybind in preferences.
    Updates the VIEW_3D keymap to use the new key.
    """
    try:
        from .Exp_Game import update_start_game_keymap
        update_start_game_keymap()
    except Exception as e:
        print(f"[Exploratory] Failed to update start game keymap: {e}")

# ------------------------------------------------------------------------
# 3) The actual Addon Preferences
# ------------------------------------------------------------------------
class ExploratoryAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = "Exploratory"

    keep_preferences: BoolProperty(
    name="Keep Preferences",
    description="If unchecked, all saved preferences will be deleted",
    default=True,
    )
    # ----------------------------------------------------------------
    # (A) Keybinds + Performance
    # ----------------------------------------------------------------
    key_forward:  bpy.props.StringProperty(default="W", name="Forward Key")
    key_backward: bpy.props.StringProperty(default="S", name="Backward Key")
    key_left:     bpy.props.StringProperty(default="A", name="Left Key")
    key_right:    bpy.props.StringProperty(default="D", name="Right Key")
    key_jump:     bpy.props.StringProperty(default="SPACE", name="Jump Key")
    key_run:      bpy.props.StringProperty(default="LEFT_SHIFT", name="Run Modifier")
    key_action:   bpy.props.StringProperty(default="LEFTMOUSE",  name="Action Key")
    key_interact: bpy.props.StringProperty(default="E",          name="Interact Key")
    key_end_game: bpy.props.StringProperty(default="ESC",         name="End Game Key")
    key_reset:    bpy.props.StringProperty(default="R", name="Reset Key")
    key_start_game: bpy.props.StringProperty(
        default="P",
        name="Start Game Key (3D Viewport)",
        description="Key to start the game in the 3D Viewport",
        update=lambda self, ctx: _update_start_game_keymap(self, ctx)
    )

    mouse_sensitivity: bpy.props.FloatProperty(
        name="Mouse Sensitivity", default=2.0, min=0.0, max=10.0
    )
    performance_mode: bpy.props.BoolProperty(name="Performance Mode", default=False)

    # ----------------------------------------------------------------
    # (B) Skin
    # ----------------------------------------------------------------
    skin_use_default: bpy.props.BoolProperty(name="Use Default Skin", default=True)
    skin_custom_blend: bpy.props.StringProperty(subtype='FILE_PATH', default="")

    # ----------------------------------------------------------------
    # (C) AUDIO MASTER PROPERTIES
    # ----------------------------------------------------------------
    enable_audio: bpy.props.BoolProperty(
        name="Enable Audio",
        description="Global master audio mute/unmute",
        default=True
    )
    audio_level: bpy.props.FloatProperty(
        name="Audio Volume",
        description="Global master volume (0.0-1.0)",
        default=0.5,
        min=0.0,
        max=1.0
    )

    # ----------------------------------------------------------------
    # (D) Performance
    # ----------------------------------------------------------------
    performance_level: bpy.props.EnumProperty(
        name="Performance Level",
        description="Choose a performance preset",
        items=[
            ("LOW", "Low", "Optimized for performance with minimal visual quality"),
            ("MEDIUM", "Medium", "Balanced quality and performance"),
            ("HIGH", "High", "Full quality settings (default)"),
        ],
        default="MEDIUM"
    )

    # --- Shadows toggle for gameplay ---
    enable_shadows_in_game: bpy.props.BoolProperty(
        name="Enable Shadows",
        description="Toggle Eevee Next shadows ON/OFF when the game starts",
        default=True
    )

    # --- Viewport shading mode to use on game start ---
    viewport_shading_mode: bpy.props.EnumProperty(
        name="Viewport Shading on Start",
        description="Shading mode to switch all 3D Viewports to when the game starts",
        items=[
            ("RENDERED",  "Rendered",   "Full rendered shading"),
            ("MATERIAL",  "Material",   "Material/LookDev shading"),
            ("SOLID",     "Solid",      "Solid shading"),
            ("WIREFRAME", "Wireframe",  "Wireframe shading"),
        ],
        default="RENDERED",
    )

    # --- Viewport Preview Pixel Size (Render) ---
    # Mirrors Scene.render.preview_pixel_size; 'AUTO' is default.
    preview_pixel_size: bpy.props.EnumProperty(
        name="Preview Pixel Size",
        description="Pixel size used for preview renders in the viewport",
        items=[
            ("AUTO", "Automatic", "Let Blender choose automatically"),
            ("1",    "1x",        "1:1 preview pixels"),
            ("2",    "2x",        "Double-sized preview pixels"),
            ("4",    "4x",        "Quad-sized preview pixels"),
            ("8",    "8x",        "Eight-times preview pixels"),
        ],
        default="AUTO",
    )

    # ----------------------------------------------------------------
    # DRAW
    # ----------------------------------------------------------------
    def draw(self, context):
        layout = self.layout

        # ----------------------------------------------------------------
        # Persistence toggle
        # ----------------------------------------------------------------
        layout.prop(self, "keep_preferences", text="Keep Preferences")
        layout.separator()

        # 1) Keybind box
        box = layout.box()
        box.label(text="Keybinds")
        for label_txt, prop_nm in [
            ("Start Game:", "key_start_game"),
            ("Forward:", "key_forward"),
            ("Backward:", "key_backward"),
            ("Left:",     "key_left"),
            ("Right:",    "key_right"),
            ("Jump:",     "key_jump"),
            ("Run:",      "key_run"),
            ("Reset:",    "key_reset"),
            ("Action:",   "key_action"),
            ("Interact:", "key_interact"),
            ("End Game:",  "key_end_game"),
        ]:
            row = box.row()
            row.label(text=label_txt)
            row.operator("exploratory.set_keybind", text=getattr(self, prop_nm)).target_prop = prop_nm

        layout.separator()
        layout.prop(self, "mouse_sensitivity")

        #---------------
        #Performance
        #---------------
        layout.separator()
        layout.label(text="Performance Settings:")
        layout.prop(self, "performance_level")
        layout.separator()
        layout.prop(self, "viewport_shading_mode")
        layout.separator()
        layout.prop(self, "preview_pixel_size")
        layout.prop(self, "enable_shadows_in_game")


        #--------------------
        #MASTER AUDIO SETTINGS
        #-------------------
        box = layout.box()
        box.label(text="Global Audio", icon='SOUND')

        # mimic the N-panel split
        split = box.split(factor=0.5, align=True)
        col = split.column(align=True)
        icon = 'RADIOBUT_ON' if self.enable_audio else 'RADIOBUT_OFF'
        col.prop(
            self,
            "enable_audio",
            text="Master Volume",
            icon=icon
        )
        split.column(align=True).prop(
            self,
            "audio_level",
            text="Volume",
            slider=True
        )

        # 2) Skin
        layout.separator()
        layout.label(text="Character Selection:")
        b_s = layout.box()
        b_s.prop(self, "skin_use_default", text="Use Default Character?")
        if not self.skin_use_default:
            b_s.prop(self, "skin_custom_blend", text="Custom Char .Blend")

        # 3) Animations are now configured in the N-panel (Animation Slots)
        layout.separator()
        box = layout.box()
        box.label(text="Animations + Sounds", icon='ACTION')
        box.label(text="Configure in the N-panel > Animation Slots")