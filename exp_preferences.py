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
# Asset Pack Scan Cache
# ------------------------------------------------------------------------
_asset_scan_cache = None
_scan_generation = 0

def _invalidate_asset_scan():
    """Mark the scan cache as stale and schedule a deferred rescan."""
    global _asset_scan_cache, _scan_generation
    _asset_scan_cache = None
    _scan_generation += 1
    gen = _scan_generation

    def _deferred():
        if _scan_generation != gen:
            return None  # superseded by a newer invalidation
        _run_preview_scan()
        return None

    try:
        bpy.app.timers.register(_deferred, first_interval=0.3)
    except Exception:
        pass

def _on_asset_pack_changed(self, context):
    """Update callback fired when an AssetPackEntry filepath or enabled changes."""
    _invalidate_asset_scan()
    _tag_prefs_redraw(context)

def _run_preview_scan():
    """Execute the preview scan and cache results."""
    global _asset_scan_cache
    try:
        prefs = bpy.context.preferences.addons["Exploratory"].preferences
        pack_paths = []
        for e in prefs.asset_packs:
            if not e.enabled or not e.filepath:
                continue
            resolved = bpy.path.abspath(e.filepath)
            if os.path.isfile(resolved):
                pack_paths.append(resolved)
        from .Exp_Game.props_and_utils.exp_asset_marking import preview_scan_packs
        _asset_scan_cache = preview_scan_packs(pack_paths)
    except Exception as ex:
        print(f"[Exploratory] Preview scan error: {ex}")
        _asset_scan_cache = {}
    _tag_prefs_redraw(bpy.context)

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
    filepath: StringProperty(subtype='FILE_PATH', update=_on_asset_pack_changed)
    enabled: BoolProperty(default=True, update=_on_asset_pack_changed)

# ------------------------------------------------------------------------
# Asset Pack UIList
# ------------------------------------------------------------------------
class EXPLORATORY_UL_AssetPackList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        filename = os.path.basename(item.filepath) if item.filepath else "(no file set)"
        row.label(text=filename)

# ------------------------------------------------------------------------
# Add / Remove operators
# ------------------------------------------------------------------------
class EXPLORATORY_OT_AddAssetPack(bpy.types.Operator):
    bl_idname = "exploratory.add_asset_pack"
    bl_label = "Add Asset Pack"
    bl_description = "Browse for a .blend file to add as an asset pack"

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.blend", options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        prefs = context.preferences.addons["Exploratory"].preferences
        entry = prefs.asset_packs.add()
        entry.filepath = self.filepath
        prefs.asset_packs_index = len(prefs.asset_packs) - 1
        _invalidate_asset_scan()
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
        _invalidate_asset_scan()
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

        # ── Asset Summary ──────────────────────────────────────────
        if _asset_scan_cache is not None and self.asset_packs:
            _ROLE_LABELS = {
                "SKIN": "Skin",
                "IDLE": "Idle", "WALK": "Walk", "RUN": "Run",
                "JUMP": "Jump", "FALL": "Fall", "LAND": "Land",
                "WALK_SOUND": "Walk Sound", "RUN_SOUND": "Run Sound",
                "JUMP_SOUND": "Jump Sound", "FALL_SOUND": "Fall Sound",
                "LAND_SOUND": "Land Sound",
            }
            _DISPLAY_ORDER = [
                "SKIN",
                "IDLE", "WALK", "RUN", "JUMP", "FALL", "LAND",
                "WALK_SOUND", "RUN_SOUND", "JUMP_SOUND", "FALL_SOUND", "LAND_SOUND",
            ]

            summary_box = box.box()
            summary_box.label(text="Detected Assets", icon='VIEWZOOM')

            for role in _DISPLAY_ORDER:
                info = _asset_scan_cache.get(role, {"count": 0, "sources": []})
                count = info["count"]
                sources = info["sources"]

                row = summary_box.row()
                if count == 0:
                    row.label(text=_ROLE_LABELS[role], icon='RADIOBUT_OFF')
                    row.label(text="Default")
                elif count == 1:
                    row.label(text=_ROLE_LABELS[role], icon='RADIOBUT_ON')
                    row.label(text=sources[0])
                else:
                    row.label(text=_ROLE_LABELS[role], icon='RADIOBUT_ON')
                    row.label(text=f"{count} packs (random)")

        box.label(text="Add .blend files with marked assets.", icon='INFO')
