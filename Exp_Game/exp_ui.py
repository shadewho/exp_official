# File: exp_ui.py
import bpy
from .props_and_utils.exp_utilities import (
    get_game_world)
from ..Exp_UI.main_config import UPLOAD_URL


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
    bl_label = "Filter Create Panels"
    bl_description = "Choose which Create panels to show/hide"
    bl_options = {'INTERNAL'}

    _ITEMS = [
        ("CHAR",   "Character / Actions / Audio"),
        ("PROXY",  "Proxy Mesh & Spawn"),
        ("UPLOAD", "Upload Helper"),
        ("PERF",   "Performance"),
        ("PHYS",   "Character Physics & View"),
        ("DEV",    "Developer Tools"),
        ("ANIM2",  "Animation 2.0 (Test)"),
    ]

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Show these panels in 'Create':")

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
    bl_label = "Exploratory (BETA v1.5)"
    bl_idname = "VIEW3D_PT_exploratory_modal"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # ─── Main navigation ────────────────────────────────────────────
        row = layout.row(align=True)
        row.scale_x = 1.5   # make buttons wider
        row.scale_y = 1.5   # make buttons taller
        row.prop(scene, "main_category", expand=True)

        layout.separator()

        # ─── CREATE MODE ──────────────────────────────────────────────
        if scene.main_category == 'CREATE':
            col = layout.column(align=True)
            
            # # Play in windowed mode
            # op = col.operator(
            #     "view3d.exp_modal",
            #     text="▶     Play Windowed (Slower)"
            # )
            # op.launched_from_ui = False
            
            # Play in fullscreen

            row = col.row()
            row.scale_y = 2.0
            play_op = row.operator(
                "exploratory.start_game",
                text="▶     Play"
            )
                 

            # ---Append demo world (button sits right under the Play buttons)
            col.separator()
            col.operator(
                "exploratory.append_demo_scene",
                text="Demo",
                icon='FILE_MOVIE'
            )

            col.separator()
            col.operator("exploratory.filter_create_panels", text="Filter Create Panels", icon='FILTER')

# --------------------------------------------------------------------
# Character, Actions, Audio (only visible in Explore mode)
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
        return (context.scene.main_category == 'CREATE'
                and _is_create_panel_enabled(context.scene, 'CHAR'))

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
# Proxy Meshes (only visible in Create mode)
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
        return (context.scene.main_category == 'CREATE'
                and _is_create_panel_enabled(context.scene, 'PROXY'))

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


# ─── Upload Helper Panel (6-step flow) ─────────────────────────────────────────
class VIEW3D_PT_Exploratory_UploadHelper(bpy.types.Panel):
    bl_label = "Upload Helper"
    bl_idname = "VIEW3D_PT_Exploratory_UploadHelper"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (context.scene.main_category == 'CREATE'
                and _is_create_panel_enabled(context.scene, 'UPLOAD'))

    def draw(self, context):
        layout = self.layout
        scene  = context.scene

        # Header
        layout.label(text="Ready to Upload?", icon='EVENT_UP_ARROW')
        layout.label(text="Follow the steps below, top to bottom.")

        # ────────────────────────────────────────────────────────────
        # Step 1: Set Game World
        # ────────────────────────────────────────────────────────────
        step1 = layout.box()
        step1.label(text="Step 1: Set Game World", icon='WORLD')
        row = step1.row(align=True)
        row.operator("exploratory.set_game_world", text="Set Current Scene as Game World", icon='WORLD')

        game_world = get_game_world()
        step1.separator()
        if game_world:
            step1.label(text=f"Current Game World: {game_world.name}", icon='CHECKMARK')
        else:
            warn = step1.column(align=True); warn.alert = True
            warn.label(text="No Game World Assigned!", icon='ERROR')
            warn.label(text="Switch to your target scene and click the button above.")

        # ────────────────────────────────────────────────────────────
        # Step 2: Remove Character (clean materials/images from it)
        # ────────────────────────────────────────────────────────────
        step2 = layout.box()
        step2.label(text="Step 2: Remove Character", icon='ARMATURE_DATA')
        step2.label(text="Removes the character and purges its materials/images if unused.")
        step2.operator("exploratory.remove_character", text="Remove Character", icon='TRASH')

        # ────────────────────────────────────────────────────────────
        # Step 3: Pack Data
        # ────────────────────────────────────────────────────────────
        step3 = layout.box()
        step3.label(text="Step 3: Pack Assets", icon='PACKAGE')
        step3.label(text="Embed all external assets and sounds into this .blend.")

        col = step3.column(align=True)
        col.operator("file.pack_all", text="Pack Data", icon='PACKAGE')
        col.operator("exp_audio.pack_all_sounds", text="Pack Sounds", icon='SOUND')

        # ────────────────────────────────────────────────────────────
        # Step 4: Purge Unused Data
        # ────────────────────────────────────────────────────────────
        step4 = layout.box()
        step4.label(text="Step 4: Purge Unused Data", icon='TRASH')
        step4.label(text="Remove orphaned datablocks to minimize file size.")
        op = step4.operator("outliner.orphans_purge", text="Purge Orphans (Recursive)", icon='TRASH')
        op.do_recursive = True

        # ────────────────────────────────────────────────────────────
        # Step 5: Save & Refresh File Size
        # ────────────────────────────────────────────────────────────
        step5 = layout.box()
        step5.label(text="Step 5: Save & Refresh File Size", icon='FILE_TICK')

        row5 = step5.row(align=True)
        row5.operator("wm.save_mainfile", text="Save .blend", icon='FILE_BLEND')
        row5.operator("exploratory.uploadhelper_refresh_size", text="Refresh Size", icon='FILE_TICK')

        cap_mib  = 500.0
        size_mib = getattr(scene, "upload_helper_file_size", 0.0)
        pct      = (size_mib / cap_mib) * 100.0 if cap_mib else 0.0

        step5.separator()
        if not bpy.data.filepath:
            step5.label(text="File not saved yet.", icon='ERROR')
        step5.label(text=f"Current Size: {size_mib:.2f} MiB / {cap_mib:.0f} MiB  ({pct:4.1f}%)", icon='FILE')

        # ────────────────────────────────────────────────────────────
        # Step 6: Upload Page
        # ────────────────────────────────────────────────────────────
        step6 = layout.box()
        step6.label(text="Step 6: Ready to upload!", icon='URL')
        row6 = step6.row(align=True)
        op_url = row6.operator("wm.url_open", text="Open Upload Page", icon='URL')
        op_url.url = UPLOAD_URL

        # Optional: quick tips
        layout.separator()
        tips = layout.box()
        tips.label(text="Tips", icon='INFO')
        tips.label(text="• Keep textures modest in resolution.")
        tips.label(text="• Delete test/unused meshes, actions, images.")
        tips.label(text="• Redo steps upon further edits.")



class VIEW3D_PT_Exploratory_Performance(bpy.types.Panel):
    bl_label = "Performance"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_idname = "VIEW3D_PT_exploratory_performance"
    bl_options = {'DEFAULT_CLOSED'}


    @classmethod
    def poll(cls, context):
        return (context.scene.main_category == 'CREATE'
                and _is_create_panel_enabled(context.scene, 'PERF'))

    def draw(self, context):
        layout = self.layout
        scene  = context.scene
        # ─── Culling ─────────────────────────
        layout.label(text="Cull Distance Entries")
        self._draw_entry_list(layout, scene)
        self._draw_entry_details(layout, scene)

    def _draw_entry_list(self, layout, scene):
        row = layout.row()
        row.template_list(
            "EXP_PERFORMANCE_UL_List", "", 
            scene, "performance_entries",
            scene, "performance_entries_index",
            rows=4
        )
        col = row.column(align=True)
        col.operator("exploratory.add_performance_entry", icon='ADD', text="")
        rem = col.operator("exploratory.remove_performance_entry", icon='REMOVE', text="")
        rem.index = scene.performance_entries_index

    def _draw_entry_details(self, layout, scene):
        idx = scene.performance_entries_index
        if not (0 <= idx < len(scene.performance_entries)):
            return

        entry = scene.performance_entries[idx]
        box = layout.box()
        box.label(text=f"Entry: {entry.name}")

        # ─── Name & Enable ─────────────────────────────────────────
        box.prop(entry, "name", text="Label")

        # ─── Trigger Settings ──────────────────────────────────────
        trigger_box = box.box()
        trigger_box.label(text="Trigger Settings")
        trigger_box.prop(entry, "trigger_type", text="Mode")
        if entry.trigger_type == 'BOX':
            trigger_box.prop(entry, "trigger_mesh", text="Bounding Mesh")
            trigger_box.prop(entry, "hide_trigger_mesh", text="Hide Mesh During Game")

        # ─── Target Selection ──────────────────────────────────────
        target_box = box.box()
        target_box.label(text="Cull Target")
        target_box.prop(entry, "use_collection", text="Cull a Collection?")
        if entry.use_collection:
            target_box.prop(entry, "target_collection", text="Collection")
            target_box.prop(entry, "cascade_collections", text="Cascade Into Child Collections")
            excl = target_box.row()
            excl.enabled = (entry.trigger_type != 'BOX')
            excl.prop(entry, "exclude_collection", text="Hide Entire Collection")
        else:
            target_box.prop(entry, "target_object", text="Object")

        # ─── Distance Settings ─────────────────────────────────────
        dist_box = box.box()
        dist_box.label(text="Distance Settings")
        # Only allow numeric distance for RADIAL mode
        dist_row = dist_box.row()
        dist_row.enabled = (entry.trigger_type == 'RADIAL')
        dist_row.prop(entry, "cull_distance", text="Cull Radius")

        # ─── Placeholder Options ───────────────────────────────────
        ph_box = box.box()
        ph_box.prop(entry, "has_placeholder", text="Enable Placeholder")

        if entry.has_placeholder:
            ph_box.label(text="Placeholder Settings")
            ph_box.prop(entry, "placeholder_use_collection", text="Use Collection as Placeholder")
            if entry.placeholder_use_collection:
                ph_box.prop(entry, "placeholder_collection", text="Placeholder Collection")
            else:
                ph_box.prop(entry, "placeholder_object", text="Placeholder Object")

# -----------------------------
# UI: Character Physics (grouped)
# -----------------------------
class VIEW3D_PT_Exploratory_PhysicsTuning(bpy.types.Panel):
    bl_label = "Character Physics and View"
    bl_idname = "VIEW3D_PT_exploratory_physics"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (getattr(context.scene, "main_category", "EXPLORE") == 'CREATE'
                and _is_create_panel_enabled(context.scene, 'PHYS'))

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

        # --- View / Camera ---
        box = layout.box()
        col = box.column(align=True)
        col.label(text="View / Camera", icon='HIDE_OFF')
        col.prop(scene, "view_projection", text="Projection")
        col.prop(scene, "view_mode", text="View Mode")

        # Always-visible view controls
        col.prop(scene, "viewport_lens_mm", text="Viewport Lens (mm)")
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
            col.label(text="First Person Settings:")
            arm = getattr(scene, "target_armature", None)
            row = col.row(align=True)
            if arm and arm.type == 'ARMATURE':
                row.prop_search(scene, "fpv_view_bone", arm.data, "bones", text="FPV Target Bone")
            else:
                row.enabled = False
                row.label(text="FPV Target Bone: — (set Target Armature)")
            col.prop(scene, "fpv_invert_pitch", text="Invert FPV Pitch")