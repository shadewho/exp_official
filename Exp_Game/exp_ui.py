# File: exp_ui.py
import bpy
from .props_and_utils.exp_asset_marking import (
    ACTION_ROLES, SOUND_ROLES, find_marked, _pg_attr_for_role,
)


def _is_create_panel_enabled(scene, key: str) -> bool:
    flags = getattr(scene, "create_panels_filter", None)
    # If the property doesn't exist yet, default to visible.
    if flags is None:
        return True
    # If the property exists but is an empty set, hide all.
    if hasattr(flags, "__len__") and len(flags) == 0:
        return False
    return (key in flags)

# ─────────────────────────────────────────────────────────
# Operator: shows a popup to edit the filter flags
# ─────────────────────────────────────────────────────────
class EXP_OT_FilterCreatePanels(bpy.types.Operator):
    bl_idname = "exploratory.filter_create_panels"
    bl_label = "Filter Panels"
    bl_description = "Choose which panels to show/hide"
    bl_options = {'INTERNAL'}

    _ITEMS = [
        ("CHAR",   "Character / Actions / Audio"),
        ("PROXY",  "Proxy Mesh & Spawn"),
        ("PHYS",   "Physics"),
        ("VIEW",   "View"),
        ("DEV",    "Developer Tools"),
        ("ASSETS", "Assets"),
    ]

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Show these panels:")

        col = layout.column(align=True)
        # each flag as its own vertical toggle with proper text
        for ident, label in self._ITEMS:
            col.prop_enum(context.scene, "create_panels_filter", ident, text=label)

    def execute(self, context):
        return {'FINISHED'}
    
# --------------------------------------------------------------------
# Exploratory Modal Panel
# --------------------------------------------------------------------
class ExploratoryPanel(bpy.types.Panel):
    bl_label = "Exploratory"
    bl_idname = "VIEW3D_PT_exploratory_modal"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        col = layout.column(align=True)

        # Play button
        row = col.row()
        row.scale_y = 2.0
        play_op = row.operator(
            "exploratory.start_game",
            text="▶     Play"
        )

        # Demo world button
        col.separator()
        col.operator(
            "exploratory.append_demo_scene",
            text="Demo",
            icon='FILE_MOVIE'
        )

        # Panel filter
        col.separator()
        col.operator("exploratory.filter_create_panels", text="Filter Panels", icon='FILTER')

# --------------------------------------------------------------------
# Character, Actions, Audio
# --------------------------------------------------------------------
class ExploratoryCharacterPanel(bpy.types.Panel):
    bl_label = "Character, Actions, Audio"
    bl_idname = "VIEW3D_PT_exploratory_character"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return _is_create_panel_enabled(context.scene, 'CHAR')

    def draw(self, context):
        layout = self.layout
        scene  = context.scene
        ca     = scene.character_actions
        audio  = scene.character_audio
        prefs  = context.preferences.addons["Exploratory"].preferences

        # ─── Character ───
        box = layout.box()
        box.label(text="Character", icon='SETTINGS')

        # Target Armature picker
        split = box.split(factor=0.4, align=True)
        colL  = split.column()
        colR  = split.column()
        colL.label(text="Target Armature (Character)")
        colR.prop(scene, "target_armature", text="")

        box.separator()

        # Lock + description
        row = box.row(align=True)
        icon = 'LOCKED' if scene.character_spawn_lock else 'UNLOCKED'
        row.prop(
            scene,
            "character_spawn_lock",
            text="Character Lock",
            icon=icon,
            toggle=True
        )
        char_col = box.column(align=True)
        char_col.label(text="• If ON the character must be set manually.")
        char_col.label(text="• If ON the character will not be removed or changed.")
        char_col.label(text="• Useful for testing a character")
        char_col.label(text="• Useful for setting a custom world characters")
        char_col.separator()
        char_col.label(text="• If OFF the character will be built automatically")
        char_col.label(text="• If OFF the character will be removed then re-appended on game start.")
        char_col.label(text="• Uses the character defined in preferences.")
        # ─── Build Character Button ───
        box.separator()
        row = box.row(align=True)
        row.scale_y = 1.0
        row.operator(
            "exploratory.build_character",
            text="Build Character",
            icon='ARMATURE_DATA'
        )
        #Build Armature (append only the armature from Armature.blend)
        row = box.row(align=True)
        row.operator(
            "exploratory.build_armature",
            text="Build Armature",
            icon='ARMATURE_DATA'
        )

        # ─── Animations & Speeds ───
        box = layout.box()
        box.label(text="Animations", icon='ACTION')

        # ─── Actions Lock ───
        row = box.row(align=True)
        icon = 'LOCKED' if scene.character_actions_lock else 'UNLOCKED'
        row.prop(
            scene,
            "character_actions_lock",
            text="Actions Lock",
            icon=icon,
            toggle=True
        )
        action_col = box.column(align=True)
        action_col.label(text="• If ON the actions must be set manually.")
        action_col.label(text="• Useful for custom scene action assignments or testing.")
        action_col.label(text="• Useful for creating a world with custom actions.")
        action_col.separator()
        action_col.label(text="• If OFF audio will be appended from preferences.")
        action_col.label(text="• Uses the actions defined in preferences.")
        box.separator()

        def anim_row(action_attr, label):
            split = box.split(factor=0.75, align=True)
            colA  = split.column()
            colB  = split.column()
            act = getattr(ca, action_attr)
            colA.prop(ca, action_attr, text=label)
            if act:
                colB.prop(act, "action_speed", text="Speed")
            else:
                colB.label(text="—")

        anim_row("idle_action", "Idle")
        anim_row("walk_action", "Walk")
        anim_row("run_action",  "Run")
        anim_row("jump_action", "Jump")
        anim_row("fall_action", "Fall")
        anim_row("land_action", "Land")

        # ─── Blend Time ───
        box.separator()
        row = box.row()
        row.prop(ca, "blend_time", text="Blend")

        # ─── Audio ───
        box = layout.box()
        box.label(text="Audio", icon='SOUND')

        # ─── Audio Lock ───
        row = box.row(align=True)
        icon = 'LOCKED' if scene.character_audio_lock else 'UNLOCKED'
        row.prop(
            scene,
            "character_audio_lock",
            text="Audio Lock",
            icon=icon,
            toggle=True
        )
        audio_col = box.column(align=True)
        audio_col.label(text="• If ON the audio must be set manually.")
        audio_col.label(text="• Useful for custom scene audio assignments or testing.")
        audio_col.label(text="• Useful for creating a world with custom audio.")
        audio_col.separator()
        audio_col.label(text="• If OFF audio will be appended from preferences.")
        audio_col.label(text="• Uses the audio defined in preferences.")
        box.separator()

        split = box.split(factor=0.5, align=True)
        col = split.column(align=True)
        icon = 'RADIOBUT_ON' if prefs.enable_audio else 'RADIOBUT_OFF'
        col.prop(
            prefs,
            "enable_audio",
            text="Master Volume",
            icon=icon
        )
        split.column(align=True).prop(
            prefs,
            "audio_level",
            text="Volume",
            slider=True
        )

        def sound_row(prop_name, label):
            row = box.row(align=True)
            snd = getattr(audio, prop_name)

            # the pointer field
            row.prop(audio, prop_name, text=label)
            # new Load button
            load_op = row.operator(
                "exp_audio.load_character_audio_file",
                text="",
                icon='FILE_FOLDER'
            )
            load_op.sound_slot = prop_name

            if snd:
                sub = box.row(align=True)
                sub.prop(snd, "sound_speed", text="Speed")
                test = sub.operator(
                    "exp_audio.test_sound_pointer",
                    text="Test",
                    icon='PLAY'
                )
                test.sound_slot = prop_name
            else:
                box.label(text=f"{label}: (none)")

        sound_row("walk_sound", "Walk")
        sound_row("run_sound",  "Run")
        sound_row("jump_sound", "Jump")
        sound_row("fall_sound", "Fall")
        sound_row("land_sound", "Land")

        box.separator()
        box.operator(
            "exp_audio.pack_all_sounds",
            text="Pack All Sounds",
            icon='PACKAGE'
        )
        box.label(text="For distribution, please pack all custom audio into the .blend file.")

# --------------------------------------------------------------------
# Proxy Meshes
# --------------------------------------------------------------------
class ExploratoryProxyMeshPanel(bpy.types.Panel):
    bl_label = "Proxy Mesh and Spawn"
    bl_idname = "VIEW3D_PT_exploratory_proxy"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return _is_create_panel_enabled(context.scene, 'PROXY')

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Proxy Mesh List
        layout.separator()
        layout.label(text="Proxy Meshes")
        row = layout.row()
        row.template_list(
            "EXPLORATORY_UL_ProxyMeshList",
            "",
            scene,
            "proxy_meshes",
            scene,
            "proxy_meshes_index",
            rows=4
        )
        col = row.column(align=True)
        col.operator("exploratory.add_proxy_mesh", text="", icon='ADD')
        remove_op = col.operator("exploratory.remove_proxy_mesh", text="", icon='REMOVE')
        remove_op.index = scene.proxy_meshes_index

        # Details for selected proxy mesh
        layout.separator()
        idx = scene.proxy_meshes_index
        if 0 <= idx < len(scene.proxy_meshes):
            entry = scene.proxy_meshes[idx]
            box = layout.box()
            box.label(text="Selected Proxy Mesh Details:")
            box.prop(entry, "name", text="Name")
            box.prop(entry, "mesh_object", text="Mesh")
            box.prop(entry, "is_moving", text="Dynamic Mesh (Moving)")
            if entry.is_moving:
                dyn_box = box.box()
                warn_grp = dyn_box.column(align=True)
                warn_grp.label(text="Dynamic mesh: AABB-gated automatically")
                warn_grp.label(text="Only transforms when player nearby")
            box.prop(entry, "hide_during_game", text="Hide During Game")

        # Spawn Object section in its own box
        layout.separator()
        spawn_box = layout.box()
        spawn_box.label(text="Spawn Object", icon='EMPTY_AXIS')
        spawn_box.prop(scene, "spawn_object", text="Object")
        spawn_box.prop(scene, "spawn_use_nearest_z_surface", text="Find Nearest Z Surface")


# -----------------------------
# UI: Character Physics (grouped)
# -----------------------------
class VIEW3D_PT_Exploratory_PhysicsTuning(bpy.types.Panel):
    bl_label = "Physics"
    bl_idname = "VIEW3D_PT_exploratory_physics"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return _is_create_panel_enabled(context.scene, 'PHYS')

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        p = scene.char_physics
        if p is None:
            layout.label(text="Character physics not initialized")
            return

        # --- Warning label at top ---
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Experimental", icon='ERROR')
        col.label(text="Changing these values may affect:")
        col.label(text="- Character physics")
        col.label(text="- Mesh collisions")
        col.label(text="- Audio/animation timing")

        # --- Collider ---
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Collider", icon='MESH_CAPSULE')
        row = col.row(align=True)
        row.prop(p, "radius")
        row.prop(p, "height")

        # --- Grounding & Steps ---
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Grounding & Steps", icon='TRIA_DOWN_BAR')
        col.prop(p, "slope_limit_deg")
        row = col.row(align=True)
        row.prop(p, "step_height")
        row.prop(p, "snap_down")

        # --- Movement ---
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Movement", icon='ORIENTATION_GLOBAL')
        col.prop(p, "gravity")
        row = col.row(align=True)
        row.prop(p, "max_speed_walk")
        row.prop(p, "max_speed_run")
        row = col.row(align=True)
        row.prop(p, "accel_ground")
        row.prop(p, "accel_air")

        # --- Jumping & Forgiveness ---
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Jumping & Forgiveness", icon='OUTLINER_OB_FORCE_FIELD')
        row = col.row(align=True)
        row.prop(p, "jump_speed")
        row.prop(p, "coyote_time")
        col.prop(p, "jump_buffer")

        # --- Steep Slope Settings ---
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Steep Slope Settings", icon='SURFACE_NCURVE')
        row = col.row(align=True)
        row.prop(p, "steep_slide_gain", text="Slide Acceleration")
        col.prop(p, "steep_min_speed", text="Minimum Slide Speed")


# -----------------------------
# UI: View
# -----------------------------
class VIEW3D_PT_Exploratory_View(bpy.types.Panel):
    bl_label = "View"
    bl_idname = "VIEW3D_PT_exploratory_view"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return _is_create_panel_enabled(context.scene, 'VIEW')

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        col = layout.column(align=True)
        col.prop(scene, "view_projection", text="Projection")
        col.prop(scene, "view_mode", text="View Mode")

        # Always-visible view controls
        col.prop(scene, "viewport_lens_mm", text="Viewport Lens (mm)")
        if scene.view_mode != 'FIRST':
            col.prop(scene, "orbit_distance", text="Orbit Distance")
            col.prop(scene, "zoom_factor", text="Zoom Factor")

        # LOCKED mode settings
        if scene.view_mode == 'LOCKED':
            col.separator()
            col.label(text="Locked View Settings:")
            col.prop(scene, "view_locked_yaw", text="Yaw")
            col.prop(scene, "view_locked_pitch", text="Pitch")
            col.prop(scene, "view_locked_distance", text="Distance")

            # Axis lock + flip
            if hasattr(scene, "view_locked_move_axis"):
                col.prop(scene, "view_locked_move_axis", text="Axis Lock (Movement)")
                row = col.row(align=True)
                row.enabled = (getattr(scene, "view_locked_move_axis", 'OFF') != 'OFF')
                row.prop(scene, "view_locked_flip_axis", text="Flip Axis Direction (180°)")

        # THIRD PERSON mode settings
        if scene.view_mode == 'THIRD':
            col.separator()
            row = col.row(align=True)
            row.prop(scene, "view_obstruction_enabled", text="Enable View Obstruction")

        # FIRST PERSON mode settings
        if scene.view_mode == 'FIRST':
            col.separator()
            box_info = col.box()
            ic = box_info.column(align=True)
            ic.label(text="FPV Tips:", icon='INFO')
            ic.label(text="Assign a character armature")
            ic.label(text="Set the FPV target bone")
            ic.label(text="The view is from the target bone")
            ic.label(text="Parent handheld objects to it")
            ic.label(text="Animate and assign to actions/nodes")
            col.separator()
            col.label(text="First Person Settings:")
            arm = getattr(scene, "target_armature", None)
            row = col.row(align=True)
            if arm and arm.type == 'ARMATURE':
                row.prop_search(scene, "fpv_view_bone", arm.data, "bones", text="FPV Target Bone")
            else:
                row.enabled = False
                row.label(text="FPV Target Bone: — (set Target Armature)")
            col.prop(scene, "fpv_invert_pitch", text="Invert FPV Pitch")
            col.operator("exploratory.test_fpv", icon='HIDE_OFF')


# --------------------------------------------------------------------
# Assets — mark datablocks for game use
# --------------------------------------------------------------------
class VIEW3D_PT_Exploratory_Assets(bpy.types.Panel):
    bl_label = "Assets"
    bl_idname = "VIEW3D_PT_exploratory_assets"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return _is_create_panel_enabled(context.scene, 'ASSETS')

    # ── helpers used by draw() ──────────────────────────────────
    @staticmethod
    def _mark_btn(layout, role, db):
        """Draw a Mark or Unmark button for *role* depending on state."""
        marked = find_marked(role)
        if marked is not None and db is not None and marked == db:
            op = layout.operator(
                "exploratory.unmark_asset", text="", icon='CHECKMARK',
            )
            op.role = role
        else:
            op = layout.operator(
                "exploratory.mark_asset", text="", icon='ADD',
            )
            op.role = role

    # ── main draw ───────────────────────────────────────────────
    def draw(self, context):
        layout = self.layout
        pg = context.scene.asset_marking

        # ─── Character (Skin) ───
        box = layout.box()
        box.label(text="Character", icon='ARMATURE_DATA')
        row = box.row(align=True)
        row.prop(pg, "skin", text="")
        self._mark_btn(row, "SKIN", pg.skin)
        marked_skin = find_marked("SKIN")
        if marked_skin is not None:
            box.label(text=f"Marked: {marked_skin.name}", icon='CHECKMARK')

        # ─── Actions ───
        box = layout.box()
        box.label(text="Actions", icon='ACTION')
        for role in ACTION_ROLES:
            attr = _pg_attr_for_role(role)
            row = box.row(align=True)
            row.label(text=role.capitalize())
            row.prop(pg, attr, text="")
            self._mark_btn(row, role, getattr(pg, attr))
        marked_actions = [
            find_marked(r) for r in ACTION_ROLES if find_marked(r) is not None
        ]
        if marked_actions:
            col = box.column(align=True)
            for db in marked_actions:
                col.label(text=f"  {db.get('exp_asset_role')}: {db.name}")

        # ─── Audio ───
        box = layout.box()
        box.label(text="Audio", icon='SOUND')
        for role in SOUND_ROLES:
            attr = _pg_attr_for_role(role)
            label = role.replace("_SOUND", "").capitalize()
            row = box.row(align=True)
            row.label(text=label)
            row.prop(pg, attr, text="")
            self._mark_btn(row, role, getattr(pg, attr))
        marked_sounds = [
            find_marked(r) for r in SOUND_ROLES if find_marked(r) is not None
        ]
        if marked_sounds:
            col = box.column(align=True)
            for db in marked_sounds:
                col.label(text=f"  {db.get('exp_asset_role')}: {db.name}")