import bpy
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
# Utility: List actions from a .blend
# ------------------------------------------------------------------------
def list_actions_in_blend(blend_path):
    """
    Returns a list of (identifier, name, description) for all actions in `blend_path`.
    If file is invalid or no actions exist, return [].
    """
    if not blend_path or not os.path.isfile(blend_path):
        return []
    try:
        with bpy.data.libraries.load(blend_path, link=False) as (data_from, _):
            action_names = data_from.actions
    except:
        action_names = []

    items = []
    for a_name in action_names:
        items.append((a_name, a_name, f"Action: {a_name}"))
    return items

# ------------------------------------------------------------------------
# Utility: List sounds from a .blend
# ------------------------------------------------------------------------
def list_sounds_in_blend(blend_path: str):
    """
    If file is invalid or no sounds exist, return [].
    """
    if not blend_path or not os.path.isfile(blend_path):
        return []
    try:
        with bpy.data.libraries.load(blend_path, link=False) as (data_from, _):
            sound_names = data_from.sounds
    except:
        sound_names = []

    items = []
    for s_name in sound_names:
        items.append((s_name, s_name, f"Sound: {s_name}"))
    return items

# ------------------------------------------------------------------------
# A row-drawing helper for Action + Sound
# ------------------------------------------------------------------------
def draw_action_sound_row(
    prefs, layout,
    label: str,

    # Action props
    act_use_def: str,
    act_blend:   str,
    act_enum:    str,

    # Sound props
    snd_use_def: str = None,
    snd_blend:   str = None,
    snd_enum:    str = None
):
    """
    Draw a single "row" inside its own box, with:
      - A left label for the name (Idle, Walk, Run, etc.)
      - A split: left column for action, right column for sound
      - If snd_* is None => we show “(No Sound)”.
    """
    # Put this entire row in its own box:
    box = layout.box()

    # Top row for the label
    row_top = box.row(align=True)
    row_top.label(text=label)

    # Now split the rest of the box horizontally
    split = box.split(factor=0.5, align=True)

    # Left column => Action
    colA = split.column(align=True)
    # Force left alignment so it doesn't jump around
    colA.alignment = 'LEFT'
    colA.prop(prefs, act_use_def, text="Use Default Action?")
    if not getattr(prefs, act_use_def):
        colA.prop(prefs, act_blend, text="Action Blend")
        colA.prop(prefs, act_enum,  text="Action")

    # Right column => Sound or N/A
    colB = split.column(align=True)
    colB.alignment = 'LEFT'
    if snd_use_def is None:
        # e.g. Idle has no sound
        colB.label(text="(No Sound)")
    else:
        colB.prop(prefs, snd_use_def, text="Use Default Sound?")
        if not getattr(prefs, snd_use_def):
            colB.prop(prefs, snd_blend, text="Sound Blend")
            colB.prop(prefs, snd_enum,  text="Sound")


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
# 2) Update Callbacks (Actions)
# ------------------------------------------------------------------------
def update_idle_action_settings(self, context):
    if self.idle_use_default_action:
        self["idle_actions"] = []
    else:
        path = self.idle_custom_blend_action
        act_list = list_actions_in_blend(path)
        names_only = [a[0] for a in act_list]
        self["idle_actions"] = names_only

def update_walk_action_settings(self, context):
    if self.walk_use_default_action:
        self["walk_actions"] = []
    else:
        path = self.walk_custom_blend_action
        act_list = list_actions_in_blend(path)
        names_only = [a[0] for a in act_list]
        self["walk_actions"] = names_only

def update_run_action_settings(self, context):
    if self.run_use_default_action:
        self["run_actions"] = []
    else:
        path = self.run_custom_blend_action
        act_list = list_actions_in_blend(path)
        names_only = [a[0] for a in act_list]
        self["run_actions"] = names_only

def update_jump_action_settings(self, context):
    if self.jump_use_default_action:
        self["jump_actions"] = []
    else:
        path = self.jump_custom_blend_action
        act_list = list_actions_in_blend(path)
        names_only = [a[0] for a in act_list]
        self["jump_actions"] = names_only

def update_fall_action_settings(self, context):
    if self.fall_use_default_action:
        self["fall_actions"] = []
    else:
        path = self.fall_custom_blend_action
        act_list = list_actions_in_blend(path)
        names_only = [a[0] for a in act_list]
        self["fall_actions"] = names_only

def update_land_action_settings(self, context):
    if self.land_use_default_action:
        self["land_actions"] = []
    else:
        path = self.land_custom_blend_action
        act_list = list_actions_in_blend(path)
        names_only = [a[0] for a in act_list]
        self["land_actions"] = names_only

# ------------------------------------------------------------------------
# 2b) Update Callbacks (Sounds)
# ------------------------------------------------------------------------
def update_walk_sound_settings(self, context):
    if self.walk_use_default_sound:
        self["walk_sounds"] = []
    else:
        path = self.walk_custom_blend_sound
        snd_list = list_sounds_in_blend(path)
        names_only = [s[0] for s in snd_list]
        self["walk_sounds"] = names_only

def update_run_sound_settings(self, context):
    if self.run_use_default_sound:
        self["run_sounds"] = []
    else:
        path = self.run_custom_blend_sound
        snd_list = list_sounds_in_blend(path)
        names_only = [s[0] for s in snd_list]
        self["run_sounds"] = names_only

def update_jump_sound_settings(self, context):
    if self.jump_use_default_sound:
        self["jump_sounds"] = []
    else:
        path = self.jump_custom_blend_sound
        snd_list = list_sounds_in_blend(path)
        names_only = [s[0] for s in snd_list]
        self["jump_sounds"] = names_only

def update_fall_sound_settings(self, context):
    if self.fall_use_default_sound:
        self["fall_sounds"] = []
    else:
        path = self.fall_custom_blend_sound
        snd_list = list_sounds_in_blend(path)
        names_only = [s[0] for s in snd_list]
        self["fall_sounds"] = names_only

def update_land_sound_settings(self, context):
    if self.land_use_default_sound:
        self["land_sounds"] = []
    else:
        path = self.land_custom_blend_sound
        snd_list = list_sounds_in_blend(path)
        names_only = [s[0] for s in snd_list]
        self["land_sounds"] = names_only

# ------------------------------------------------------------------------
# 3) The actual Addon Preferences
# ------------------------------------------------------------------------
class ExploratoryAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = "Exploratory"

    # ----------------------------------------------------------------
    # (A) Keybinds + Performance
    # ----------------------------------------------------------------
    key_forward:  bpy.props.StringProperty(default="W", name="Forward Key")
    key_backward: bpy.props.StringProperty(default="S", name="Backward Key")
    key_left:     bpy.props.StringProperty(default="A", name="Left Key")
    key_right:    bpy.props.StringProperty(default="D", name="Right Key")
    key_jump:     bpy.props.StringProperty(default="SPACE", name="Jump Key")
    key_run:      bpy.props.StringProperty(default="LEFT_SHIFT", name="Run Modifier")
    key_interact: bpy.props.StringProperty(default="LEFTMOUSE", name="Interact Key")
    key_reset:    bpy.props.StringProperty(default="R", name="Reset Key")

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
    # (C) Idle (action only, no sound)
    # ----------------------------------------------------------------
    idle_use_default_action: bpy.props.BoolProperty(
        name="Use Default Idle Action",
        default=True,
        update=update_idle_action_settings
    )
    idle_custom_blend_action: bpy.props.StringProperty(
        subtype='FILE_PATH',
        default="",
        update=update_idle_action_settings
    )
    def idle_action_items(self, context):
        arr = self.get("idle_actions", [])
        return [(n, n, f"Action: {n}") for n in arr]
    idle_action_enum_prop: bpy.props.EnumProperty(
        name="Idle Action",
        items=idle_action_items
    )

    # ----------------------------------------------------------------
    # (D 1) AUDIO MASTER PROPERTIES
    # ----------------------------------------------------------------
    enable_audio: bpy.props.BoolProperty(
        name="Enable Audio",
        description="Global master audio mute/unmute",
        default=True
    )
    audio_level: bpy.props.FloatProperty(
        name="Audio Volume",
        description="Global master volume (0.0–1.0)",
        default=0.5,
        min=0.0,
        max=1.0
    )
    # ----------------------------------------------------------------
    # (D 2) Walk => separate booleans for Action vs Sound
    # ----------------------------------------------------------------
    # Action
    walk_use_default_action: bpy.props.BoolProperty(
        name="Use Default Walk Action",
        default=True,
        update=update_walk_action_settings
    )
    walk_custom_blend_action: bpy.props.StringProperty(
        name="Custom Walk Action Blend",
        subtype='FILE_PATH',
        default="",
        update=update_walk_action_settings
    )
    def walk_action_items(self, context):
        arr = self.get("walk_actions", [])
        return [(n, n, f"Action: {n}") for n in arr]
    walk_action_enum_prop: bpy.props.EnumProperty(
        name="Walk Action",
        items=walk_action_items
    )

    # Sound
    walk_use_default_sound: bpy.props.BoolProperty(
        name="Use Default Walk Sound",
        default=True,
        update=update_walk_sound_settings
    )
    walk_custom_blend_sound: bpy.props.StringProperty(
        name="Custom Walk Sound Blend",
        subtype='FILE_PATH',
        default="",
        update=update_walk_sound_settings
    )
    def walk_sound_items(self, context):
        arr = self.get("walk_sounds", [])
        return [(n, n, f"Sound: {n}") for n in arr]
    walk_sound_enum_prop: bpy.props.EnumProperty(
        name="Walk Sound",
        items=walk_sound_items
    )

    # ----------------------------------------------------------------
    # (E) Run => separate booleans for Action vs Sound
    # ----------------------------------------------------------------
    run_use_default_action: bpy.props.BoolProperty(
        name="Use Default Run Action",
        default=True,
        update=update_run_action_settings
    )
    run_custom_blend_action: bpy.props.StringProperty(
        name="Custom Run Action Blend",
        subtype='FILE_PATH',
        default="",
        update=update_run_action_settings
    )
    def run_action_items(self, context):
        arr = self.get("run_actions", [])
        return [(n, n, f"Action: {n}") for n in arr]
    run_action_enum_prop: bpy.props.EnumProperty(
        name="Run Action",
        items=run_action_items
    )

    run_use_default_sound: bpy.props.BoolProperty(
        name="Use Default Run Sound",
        default=True,
        update=update_run_sound_settings
    )
    run_custom_blend_sound: bpy.props.StringProperty(
        name="Custom Run Sound Blend",
        subtype='FILE_PATH',
        default="",
        update=update_run_sound_settings
    )
    def run_sound_items(self, context):
        arr = self.get("run_sounds", [])
        return [(n, n, f"Sound: {n}") for n in arr]
    run_sound_enum_prop: bpy.props.EnumProperty(
        name="Run Sound",
        items=run_sound_items
    )

    # ----------------------------------------------------------------
    # (F) Jump => separate booleans for Action vs Sound
    # ----------------------------------------------------------------
    jump_use_default_action: bpy.props.BoolProperty(
        name="Use Default Jump Action",
        default=True,
        update=update_jump_action_settings
    )
    jump_custom_blend_action: bpy.props.StringProperty(
        name="Custom Jump Action Blend",
        subtype='FILE_PATH',
        default="",
        update=update_jump_action_settings
    )
    def jump_action_items(self, context):
        arr = self.get("jump_actions", [])
        return [(n, n, f"Action: {n}") for n in arr]
    jump_action_enum_prop: bpy.props.EnumProperty(
        name="Jump Action",
        items=jump_action_items
    )

    jump_use_default_sound: bpy.props.BoolProperty(
        name="Use Default Jump Sound",
        default=True,
        update=update_jump_sound_settings
    )
    jump_custom_blend_sound: bpy.props.StringProperty(
        name="Custom Jump Sound Blend",
        subtype='FILE_PATH',
        default="",
        update=update_jump_sound_settings
    )
    def jump_sound_items(self, context):
        arr = self.get("jump_sounds", [])
        return [(n, n, f"Sound: {n}") for n in arr]
    jump_sound_enum_prop: bpy.props.EnumProperty(
        name="Jump Sound",
        items=jump_sound_items
    )

    # ----------------------------------------------------------------
    # (G) Fall => separate booleans for Action vs Sound
    # ----------------------------------------------------------------
    fall_use_default_action: bpy.props.BoolProperty(
        name="Use Default Fall Action",
        default=True,
        update=update_fall_action_settings
    )
    fall_custom_blend_action: bpy.props.StringProperty(
        name="Custom Fall Action Blend",
        subtype='FILE_PATH',
        default="",
        update=update_fall_action_settings
    )
    def fall_action_items(self, context):
        arr = self.get("fall_actions", [])
        return [(n, n, f"Action: {n}") for n in arr]
    fall_action_enum_prop: bpy.props.EnumProperty(
        name="Fall Action",
        items=fall_action_items
    )

    fall_use_default_sound: bpy.props.BoolProperty(
        name="Use Default Fall Sound",
        default=True,
        update=update_fall_sound_settings
    )
    fall_custom_blend_sound: bpy.props.StringProperty(
        name="Custom Fall Sound Blend",
        subtype='FILE_PATH',
        default="",
        update=update_fall_sound_settings
    )
    def fall_sound_items(self, context):
        arr = self.get("fall_sounds", [])
        return [(n, n, f"Sound: {n}") for n in arr]
    fall_sound_enum_prop: bpy.props.EnumProperty(
        name="Fall Sound",
        items=fall_sound_items
    )

    # ----------------------------------------------------------------
    # (H) Land => separate booleans for Action vs Sound
    # ----------------------------------------------------------------
    land_use_default_action: bpy.props.BoolProperty(
        name="Use Default Land Action",
        default=True,
        update=update_land_action_settings
    )
    land_custom_blend_action: bpy.props.StringProperty(
        name="Custom Land Action Blend",
        subtype='FILE_PATH',
        default="",
        update=update_land_action_settings
    )
    def land_action_items(self, context):
        arr = self.get("land_actions", [])
        return [(n, n, f"Action: {n}") for n in arr]
    land_action_enum_prop: bpy.props.EnumProperty(
        name="Land Action",
        items=land_action_items
    )

    land_use_default_sound: bpy.props.BoolProperty(
        name="Use Default Land Sound",
        default=True,
        update=update_land_sound_settings
    )
    land_custom_blend_sound: bpy.props.StringProperty(
        name="Custom Land Sound Blend",
        subtype='FILE_PATH',
        default="",
        update=update_land_sound_settings
    )
    def land_sound_items(self, context):
        arr = self.get("land_sounds", [])
        return [(n, n, f"Sound: {n}") for n in arr]
    land_sound_enum_prop: bpy.props.EnumProperty(
        name="Land Sound",
        items=land_sound_items
    )
    # ----------------------------------------------------------------
    # (I) Performance
    # ----------------------------------------------------------------
    performance_level: bpy.props.EnumProperty(
        name="Performance Level",
        description="Choose a performance preset",
        items=[
            ("LOW", "Low", "Optimized for performance with minimal visual quality"),
            ("MEDIUM", "Medium", "Balanced quality and performance"),
            ("HIGH", "High", "Full quality settings (default)"),
            ("CUSTOM", "Custom", "Use custom performance settings")
        ],
        default="HIGH"
    )

    # ----------------------------------------------------------------
    # DRAW
    # ----------------------------------------------------------------
    def draw(self, context):
        layout = self.layout

        # 1) Keybind box
        box = layout.box()
        box.label(text="Keybinds")
        for label_txt, prop_nm in [
            ("Forward:", "key_forward"),
            ("Backward:", "key_backward"),
            ("Left:",     "key_left"),
            ("Right:",    "key_right"),
            ("Jump:",     "key_jump"),
            ("Run:",      "key_run"),
            ("Reset:",    "key_reset"),
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

        # 3) Actions + Sounds (Single Table)
        layout.separator()
        layout.label(text="Character Actions + Sounds")

        # IDLE => no sound
        draw_action_sound_row(
            prefs=self, layout=layout,
            label="Idle",
            act_use_def="idle_use_default_action",
            act_blend="idle_custom_blend_action",
            act_enum="idle_action_enum_prop",

            snd_use_def=None
        )

        # Walk
        draw_action_sound_row(
            prefs=self, layout=layout,
            label="Walk",
            act_use_def="walk_use_default_action",
            act_blend="walk_custom_blend_action",
            act_enum="walk_action_enum_prop",

            snd_use_def="walk_use_default_sound",
            snd_blend="walk_custom_blend_sound",
            snd_enum="walk_sound_enum_prop"
        )

        # Run
        draw_action_sound_row(
            prefs=self, layout=layout,
            label="Run",
            act_use_def="run_use_default_action",
            act_blend="run_custom_blend_action",
            act_enum="run_action_enum_prop",

            snd_use_def="run_use_default_sound",
            snd_blend="run_custom_blend_sound",
            snd_enum="run_sound_enum_prop"
        )

        # Jump
        draw_action_sound_row(
            prefs=self, layout=layout,
            label="Jump",
            act_use_def="jump_use_default_action",
            act_blend="jump_custom_blend_action",
            act_enum="jump_action_enum_prop",

            snd_use_def="jump_use_default_sound",
            snd_blend="jump_custom_blend_sound",
            snd_enum="jump_sound_enum_prop"
        )

        # Fall
        draw_action_sound_row(
            prefs=self, layout=layout,
            label="Fall",
            act_use_def="fall_use_default_action",
            act_blend="fall_custom_blend_action",
            act_enum="fall_action_enum_prop",

            snd_use_def="fall_use_default_sound",
            snd_blend="fall_custom_blend_sound",
            snd_enum="fall_sound_enum_prop"
        )

        # Land
        draw_action_sound_row(
            prefs=self, layout=layout,
            label="Land",
            act_use_def="land_use_default_action",
            act_blend="land_custom_blend_action",
            act_enum="land_action_enum_prop",

            snd_use_def="land_use_default_sound",
            snd_blend="land_custom_blend_sound",
            snd_enum="land_sound_enum_prop"
        )


# ------------------------------------------------------------------------
# 4) The operator that appends skin + actions
# ------------------------------------------------------------------------
class EXPLORATORY_OT_BuildCharacter(bpy.types.Operator):
    """
    One operator that:
      1) Appends the skin (default or custom)
      2) Appends each action (default or custom)
      3) Assigns appended actions to scene.character_actions pointers
    """
    bl_idname = "exploratory.build_character"
    bl_label = "Build Character (Skin + Actions)"

    DEFAULT_SKIN_BLEND = os.path.join(
        get_addon_path(),
        "Exp_Game", "exp_assets", "Skins", "exp_default_char.blend"
    )
    DEFAULT_ANIMS_BLEND = os.path.join(
        get_addon_path(),
        "Exp_Game", "exp_assets", "Animations", "exp_animations.blend"
    )

    DEFAULT_IDLE_NAME = "exp_idle"
    DEFAULT_WALK_NAME = "exp_walk"
    DEFAULT_RUN_NAME  = "exp_run"
    DEFAULT_JUMP_NAME = "exp_jump"
    DEFAULT_FALL_NAME = "exp_fall"
    DEFAULT_LAND_NAME = "exp_land"

    def execute(self, context):
        prefs = context.preferences.addons["Exploratory"].preferences
        scene = context.scene

        # ─── 1) Skin ───────────────────────────────────────────────────────────
        skin_blend = (
            self.DEFAULT_SKIN_BLEND
            if prefs.skin_use_default
            else prefs.skin_custom_blend
        )
        if scene.character_spawn_lock:
            self.report(
                {'INFO'},
                "Character spawn is locked; skipping skin append."
            )
        else:
            # If the armature already exists in that blend, remove it first
            try:
                with bpy.data.libraries.load(skin_blend, link=False) as (df, _):
                    lib_names = set(df.objects)
            except Exception as e:
                self.report({'WARNING'}, f"Could not read {skin_blend}: {e}")
                lib_names = set()

            existing_arm = scene.target_armature
            if existing_arm and existing_arm.name in lib_names:
                bpy.ops.exploratory.remove_character('EXEC_DEFAULT')

            # Append skin objects
            self.append_all_skin_objects(
                use_default  = prefs.skin_use_default,
                custom_blend = prefs.skin_custom_blend
            )

        # ─── 2) Actions ───────────────────────────────────────────────────────
        if scene.character_actions_lock:
            self.report(
                {'INFO'},
                "Character actions lock is ON; skipping action assignment."
            )
        else:
            char_actions = scene.character_actions

            def process_action(state_label, use_default, custom_blend,
                               chosen_name, default_name, target_attr):
                blend = (
                    self.DEFAULT_ANIMS_BLEND
                    if use_default
                    else custom_blend
                )
                action_name = default_name if use_default else chosen_name
                if not action_name:
                    print(f"[{state_label}] No action name; skipping.")
                    return

                # Append if not already loaded
                if not bpy.data.actions.get(action_name) and os.path.isfile(blend):
                    with bpy.data.libraries.load(blend, link=False) as (df, dt):
                        if action_name in df.actions:
                            dt.actions = [action_name]

                act = bpy.data.actions.get(action_name)
                if act:
                    setattr(char_actions, target_attr, act)
                    print(f"[{state_label}] => {act.name}")

            process_action(
                "Idle",
                prefs.idle_use_default_action,
                prefs.idle_custom_blend_action,
                prefs.idle_action_enum_prop,
                self.DEFAULT_IDLE_NAME,
                "idle_action"
            )
            process_action(
                "Walk",
                prefs.walk_use_default_action,
                prefs.walk_custom_blend_action,
                prefs.walk_action_enum_prop,
                self.DEFAULT_WALK_NAME,
                "walk_action"
            )
            process_action(
                "Run",
                prefs.run_use_default_action,
                prefs.run_custom_blend_action,
                prefs.run_action_enum_prop,
                self.DEFAULT_RUN_NAME,
                "run_action"
            )
            process_action(
                "Jump",
                prefs.jump_use_default_action,
                prefs.jump_custom_blend_action,
                prefs.jump_action_enum_prop,
                self.DEFAULT_JUMP_NAME,
                "jump_action"
            )
            process_action(
                "Fall",
                prefs.fall_use_default_action,
                prefs.fall_custom_blend_action,
                prefs.fall_action_enum_prop,
                self.DEFAULT_FALL_NAME,
                "fall_action"
            )
            process_action(
                "Land",
                prefs.land_use_default_action,
                prefs.land_custom_blend_action,
                prefs.land_action_enum_prop,
                self.DEFAULT_LAND_NAME,
                "land_action"
            )

        self.report({'INFO'}, "Build Character complete!")
        return {'FINISHED'}

    # ----------------------------------------------------------------
    # Exactly the same method for skin
    # ----------------------------------------------------------------
    def append_all_skin_objects(self, use_default, custom_blend):
        import os
        import bpy
        from .exp_preferences import get_addon_path

        scene = bpy.context.scene

        # Decide .blend file
        if use_default:
            blend_path = os.path.join(get_addon_path(), "Exp_Game", "exp_assets", "Skins", "exp_default_char.blend")
        else:
            blend_path = custom_blend

        if not blend_path or not os.path.isfile(blend_path):
            self.report({'WARNING'}, f"Invalid blend file: {blend_path}")
            return

        print(f"[AppendSkin] Checking objects from: {blend_path}")

        with bpy.data.libraries.load(blend_path, link=False) as (data_from, _):
            all_obj_names_in_lib = data_from.objects

        to_append = []
        existing_armature = None

        for lib_obj_name in all_obj_names_in_lib:
            if lib_obj_name in bpy.data.objects:
                print(f"Skipping '{lib_obj_name}' (already in file).")
                existing = bpy.data.objects.get(lib_obj_name)
                if existing and existing.type == 'ARMATURE':
                    existing_armature = existing
            else:
                to_append.append(lib_obj_name)

        if not to_append:
            if existing_armature:
                scene.target_armature = existing_armature
                print(f"No new objects appended. Using existing armature '{existing_armature.name}'.")
            else:
                print("No objects to append and no existing armature found.")
            return

        appended_objects = []
        with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
            data_to.objects = to_append

        for obj in data_to.objects:
            if not obj:
                continue
            bpy.context.scene.collection.objects.link(obj)
            appended_objects.append(obj)
            print(f"Appended: {obj.name}")
            if obj.type == 'ARMATURE':
                scene.target_armature = obj
                print(f"Assigned scene.target_armature => {obj.name}")

        self.ensure_skin_in_scene_collection([o.name for o in appended_objects])
        print(f"[AppendSkin] Done. Appended {len(appended_objects)} objects from {blend_path}.")

    def ensure_skin_in_scene_collection(self, object_names):
        scene = bpy.context.scene
        for name in object_names:
            if any(o.name == name for o in scene.collection.objects):
                continue
            data_obj = bpy.data.objects.get(name)
            if data_obj:
                scene.collection.objects.link(data_obj)
                print(f"Linked '{name}' to scene collection.")
            else:
                print(f"Object '{name}' not found in bpy.data.objects.")

    # ----------------------------------------------------------------
    # Helper: ensures an action is in bpy.data.actions
    # ----------------------------------------------------------------
    def ensure_action_in_file(self, blend_path, action_name, state_label):
        if not blend_path or not os.path.isfile(blend_path):
            print(f"[{state_label}] Invalid blend path: {blend_path}")
            return None

        existing = bpy.data.actions.get(action_name)
        if existing:
            print(f"[{state_label}] Action '{action_name}' already loaded.")
            return existing

        print(f"[{state_label}] Appending '{action_name}' from {blend_path}")
        with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
            if action_name in data_from.actions:
                data_to.actions = [action_name]
            else:
                print(f"[{state_label}] Action '{action_name}' not found in {blend_path}")
                return None

        return bpy.data.actions.get(action_name)
