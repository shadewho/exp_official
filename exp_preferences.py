import bpy
from bpy.props import (
    BoolProperty, StringProperty, IntProperty,
    FloatProperty, EnumProperty, CollectionProperty,
)
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

def _tag_prefs_redraw(context):
    """Ping UI to refresh enum lists after a path toggle."""
    try:
        for win in context.window_manager.windows:
            scr = getattr(win, "screen", None)
            if not scr:
                continue
            for area in scr.areas:
                if area.type in {'PREFERENCES', 'VIEW_3D'}:
                    area.tag_redraw()
    except Exception:
        pass

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
# Asset Pack Entry PropertyGroup
# ------------------------------------------------------------------------
class AssetPackEntry(bpy.types.PropertyGroup):
    filepath: StringProperty(subtype='FILE_PATH')
    enabled: BoolProperty(default=True)

# ------------------------------------------------------------------------
# Asset Pack UIList
# ------------------------------------------------------------------------
class EXPLORATORY_UL_AssetPackList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.prop(item, "filepath", text="")

# ------------------------------------------------------------------------
# Add / Remove operators
# ------------------------------------------------------------------------
class EXPLORATORY_OT_AddAssetPack(bpy.types.Operator):
    bl_idname = "exploratory.add_asset_pack"
    bl_label = "Add Asset Pack"
    bl_description = "Add a new .blend file entry to the asset pack list"

    def execute(self, context):
        prefs = context.preferences.addons["Exploratory"].preferences
        prefs.asset_packs.add()
        prefs.asset_packs_index = len(prefs.asset_packs) - 1
        return {'FINISHED'}

class EXPLORATORY_OT_RemoveAssetPack(bpy.types.Operator):
    bl_idname = "exploratory.remove_asset_pack"
    bl_label = "Remove Asset Pack"
    bl_description = "Remove the selected entry from the asset pack list"

    def execute(self, context):
        prefs = context.preferences.addons["Exploratory"].preferences
        idx = prefs.asset_packs_index
        if 0 <= idx < len(prefs.asset_packs):
            prefs.asset_packs.remove(idx)
            prefs.asset_packs_index = min(idx, len(prefs.asset_packs) - 1)
        return {'FINISHED'}

# ------------------------------------------------------------------------
# Keybind Operator
# ------------------------------------------------------------------------
class EXPLORATORY_OT_SetKeybind(bpy.types.Operator):
    """Operator that temporarily listens for the next pressed key and updates the add-on preference property."""
    bl_idname = "exploratory.set_keybind"
    bl_label = "Set Keybind"
    bl_description = "Click then press the key you want to assign."

    target_prop: StringProperty(default="")

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
# Addon Preferences
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
    key_forward:  StringProperty(default="W", name="Forward Key")
    key_backward: StringProperty(default="S", name="Backward Key")
    key_left:     StringProperty(default="A", name="Left Key")
    key_right:    StringProperty(default="D", name="Right Key")
    key_jump:     StringProperty(default="SPACE", name="Jump Key")
    key_run:      StringProperty(default="LEFT_SHIFT", name="Run Modifier")
    key_action:   StringProperty(default="LEFTMOUSE",  name="Action Key")
    key_interact: StringProperty(default="E",          name="Interact Key")
    key_end_game: StringProperty(default="ESC",         name="End Game Key")
    key_reset:    StringProperty(default="R", name="Reset Key")
    key_start_game: StringProperty(
        default="P",
        name="Start Game Key (3D Viewport)",
        description="Key to start the game in the 3D Viewport",
        update=lambda self, ctx: _update_start_game_keymap(self, ctx)
    )

    mouse_sensitivity: FloatProperty(
        name="Mouse Sensitivity", default=2.0, min=0.0, max=10.0
    )
    performance_mode: BoolProperty(name="Performance Mode", default=False)

    # ----------------------------------------------------------------
    # Audio Master
    # ----------------------------------------------------------------
    enable_audio: BoolProperty(
        name="Enable Audio",
        description="Global master audio mute/unmute",
        default=True
    )
    audio_level: FloatProperty(
        name="Audio Volume",
        description="Global master volume (0.0\u20131.0)",
        default=0.5,
        min=0.0,
        max=1.0
    )

    # ----------------------------------------------------------------
    # Performance
    # ----------------------------------------------------------------
    performance_level: EnumProperty(
        name="Performance Level",
        description="Choose a performance preset",
        items=[
            ("LOW", "Low", "Optimized for performance with minimal visual quality"),
            ("MEDIUM", "Medium", "Balanced quality and performance"),
            ("HIGH", "High", "Full quality settings (default)"),
        ],
        default="MEDIUM"
    )

    enable_shadows_in_game: BoolProperty(
        name="Enable Shadows",
        description="Toggle Eevee Next shadows ON/OFF when the game starts",
        default=True
    )

    viewport_shading_mode: EnumProperty(
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

    preview_pixel_size: EnumProperty(
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
    # Asset Packs
    # ----------------------------------------------------------------
    asset_packs: CollectionProperty(type=AssetPackEntry)
    asset_packs_index: IntProperty(default=0)

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

        # ----------------------------------------------------------------
        # Asset Packs
        # ----------------------------------------------------------------
        layout.separator()
        box = layout.box()
        box.label(text="Asset Packs", icon='PACKAGE')

        row = box.row()
        row.template_list(
            "EXPLORATORY_UL_AssetPackList", "",
            self, "asset_packs",
            self, "asset_packs_index",
            rows=3,
        )

        col = row.column(align=True)
        col.operator("exploratory.add_asset_pack", icon='ADD', text="")
        col.operator("exploratory.remove_asset_pack", icon='REMOVE', text="")

        if self.asset_packs and 0 <= self.asset_packs_index < len(self.asset_packs):
            entry = self.asset_packs[self.asset_packs_index]
            sub = box.box()
            sub.prop(entry, "filepath", text="Blend File")
            sub.prop(entry, "enabled", text="Enabled")

        box.label(text="Add .blend files with marked assets.", icon='INFO')
