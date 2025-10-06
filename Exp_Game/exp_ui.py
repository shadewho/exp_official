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
        ("STUDIO", "Interactions & Reactions"),
        ("OBJ",    "Objectives & Timers"),
        ("UPLOAD", "Upload Helper"),
        ("PERF",   "Performance"),
        ("PHYS",   "Character Physics & View"),
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
    bl_label = "Exploratory (BETA v1.3)"
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
                warn_grp.label(text="⚠ Mesh rebuild every frame!")
                warn_grp.label(text="Optimize mesh and use Register Distance")
                warn_grp.label(text="Very expensive calculations for complex geometry!")
                dyn_box.separator()
                dist_grp = dyn_box.column(align=True)
                dist_grp.prop(entry, "register_distance", text="Register Distance")
                dist_grp.label(text="Distance at which dynamic mesh updates.")
            box.prop(entry, "hide_during_game", text="Hide During Game")

        # Spawn Object section in its own box
        layout.separator()
        spawn_box = layout.box()
        spawn_box.label(text="Spawn Object", icon='EMPTY_AXIS')
        spawn_box.prop(scene, "spawn_object", text="Object")
        spawn_box.prop(scene, "spawn_use_nearest_z_surface", text="Find Nearest Z Surface")


###############################################################
# the "Studio" Panel in the same "Exploratory" tab
###############################################################


class EXPLORATORY_UL_CustomInteractions(bpy.types.UIList):
    """
    This UIList will display each item in `scene.custom_interactions`.
    We'll show the item.name in each row.
    """
    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        # item is an InteractionDefinition
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "name", text="", emboss=False, icon='DOT')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.name)


class EXPLORATORY_UL_ReactionsInInteraction(bpy.types.UIList):
    """Shows links (by index) to global reactions; displays the reaction's name if valid."""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        # item is a ReactionLinkPG (inter.reaction_links[i])
        scn = context.scene
        name = "—"
        i = getattr(item, "reaction_index", -1)
        if 0 <= i < len(scn.reactions):
            name = scn.reactions[i].name
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text=name, icon='DOT')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=name)


class VIEW3D_PT_Exploratory_Studio(bpy.types.Panel):
    """Your main Interactions panel."""
    bl_label = "Interactions"
    bl_idname = "VIEW3D_PT_exploratory_studio"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (context.scene.main_category == 'CREATE'
                and _is_create_panel_enabled(context.scene, 'STUDIO'))
    def draw(self, context):
        layout = self.layout
        scene = context.scene

        ######################
        # (A) The Interaction List
        ######################

        row = layout.row()
        row.template_list(
            "EXPLORATORY_UL_CustomInteractions",
            "",
            scene,
            "custom_interactions",
            scene,
            "custom_interactions_index",
            rows=3
        )
        col = row.column(align=True)
        col.operator("exploratory.add_interaction", text="", icon='ADD')
        remove_op = col.operator("exploratory.remove_interaction", text="", icon='REMOVE')
        col.operator("exploratory.duplicate_interaction", text="", icon='DUPLICATE')
        remove_op.index = scene.custom_interactions_index

        layout.separator()

        # "Current Interaction" box
        box = layout.box()
        box.label(text="Current Interaction", icon='OBJECT_DATA')

        if not scene.custom_interactions:
            box.label(text="(No interactions defined)", icon='ERROR')
            return

        i_idx = scene.custom_interactions_index
        if i_idx < 0 or i_idx >= len(scene.custom_interactions):
            box.label(text="(Select an interaction)", icon='INFO')
            return

        inter = scene.custom_interactions[i_idx]

        # Show the basics
        box.prop(inter, "name", text="Name")
        box.prop(inter, "description", text="Description")

        # Trigger Type & Fields
        box.prop(inter, "trigger_type", text="Trigger")

        if inter.trigger_type == "PROXIMITY":
            box.prop(inter, "proximity_distance", text="Distance")
            box.prop(inter, "use_character", text="Use Character as Object A")
            if inter.use_character:
                # show a read‑only label instead of the picker
                char = context.scene.target_armature
                box.label(text=f"Object A: {char.name if char else '—'}")
            else:
                box.prop(inter, "proximity_object_a", text="Object A")
            box.prop(inter, "proximity_object_b", text="Object B")

        elif inter.trigger_type == "COLLISION":
            box.prop(inter, "use_character", text="Use Character as Object A")
            if inter.use_character:
                char = context.scene.target_armature
                box.label(text=f"Object A: {char.name if char else '—'}")
            else:
                box.prop(inter, "collision_object_a", text="Object A")
            box.prop(inter, "collision_object_b", text="Object B")
            box.prop(inter, "collision_margin", text="Collision Margin")



        elif inter.trigger_type == "INTERACT":
            box.prop(inter, "interact_object", text="Object to Interact")
            box.prop(inter, "interact_distance", text="Distance")

        elif inter.trigger_type == "OBJECTIVE_UPDATE":
            # Let user pick which objective
            box.prop(inter, "objective_index", text="Objective")
            box.prop(inter, "objective_condition", text="Condition")

            # If user picks EQUALS or AT_LEAST, show the condition_value
            if inter.objective_condition in {"EQUALS", "AT_LEAST"}:
                box.prop(inter, "objective_condition_value", text="Value")
        elif inter.trigger_type == "TIMER_COMPLETE":
            # Just show the Timer Objective dropdown
            box.prop(inter, "timer_objective_index", text="Timer Objective")



        # (NEW) One-Shot / Cooldown UI
        col_trigger = box.column(align=True)
        col_trigger.label(text="Trigger Options:")
        col_trigger.prop(inter, "trigger_mode", text="Mode")
        if inter.trigger_mode == "COOLDOWN":
            col_trigger.prop(inter, "trigger_cooldown", text="Cooldown")

        box.prop(inter, "trigger_delay", text="Trigger Delay (sec)")


        layout.separator()

        ######################
        # (B) Linked Reactions (global library)
        ######################
        sub = box.box()
        sub.label(text="Linked Reactions (from global library)", icon='OBJECT_DATA')

        row2 = sub.row()
        row2.template_list(
            "EXPLORATORY_UL_ReactionsInInteraction",
            "",
            inter,                    # InteractionDefinition
            "reaction_links",         # Collection of links (ReactionLinkPG)
            inter,                    # Active data
            "reaction_links_index",   # Active index
            rows=5
        )
        col2 = row2.column(align=True)

        # Link the currently selected global reaction (scene.reactions_index)
        col2.operator("exploratory.add_reaction_to_interaction", text="", icon='ADD')

        # Unlink the currently selected link
        rem_op = col2.operator("exploratory.remove_reaction_from_interaction", text="", icon='REMOVE')
        rem_op.index = inter.reaction_links_index

        col2.separator()
        # Create a brand new global reaction and link it here
        col2.operator("exploratory.create_reaction_and_link", text="", icon='PLUS')

        # (No embedded reaction editor here anymore; editing happens in the Reactions panel)
        info = sub.box()
        info.label(text="Edit reactions in the separate 'Reactions' panel.", icon='INFO')
        info.label(text="Tip: select a reaction in the library, then click + to link it.")



# ─────────────────────────────────────────────────────────
# Interactions and reactions -- Duplicate
# ─────────────────────────────────────────────────────────
def _deep_copy_pg(src, dst, skip: set[str] = frozenset()):
    import bpy
    from bpy.types import ID as _ID
    for prop in src.bl_rna.properties:
        ident = prop.identifier
        if ident in {"rna_type"} or ident in skip:
            continue
        if getattr(prop, "is_readonly", False):
            continue
        try:
            value = getattr(src, ident)
        except Exception:
            continue
        try:
            if prop.type == 'POINTER':
                if isinstance(value, _ID) or value is None:
                    setattr(dst, ident, value)
                else:
                    sub_dst = getattr(dst, ident)
                    _deep_copy_pg(value, sub_dst)
            elif prop.type == 'COLLECTION':
                dst_coll = getattr(dst, ident)
                try:
                    dst_coll.clear()
                except AttributeError:
                    while len(dst_coll):
                        dst_coll.remove(len(dst_coll) - 1)
                for src_item in value:
                    dst_item = dst_coll.add()
                    _deep_copy_pg(src_item, dst_item)
            else:
                setattr(dst, ident, value)
        except Exception:
            pass


class EXPLORATORY_OT_DuplicateGlobalReaction(bpy.types.Operator):
    """Duplicate the selected Reaction in the global library."""
    bl_idname = "exploratory.duplicate_global_reaction"
    bl_label = "Duplicate Reaction"
    bl_options = {'REGISTER', 'UNDO'}

    index: bpy.props.IntProperty(name="Index", default=-1)

    def execute(self, context):
        scn = context.scene
        src_idx = self.index if self.index >= 0 else scn.reactions_index
        if not (0 <= src_idx < len(scn.reactions)):
            self.report({'WARNING'}, "No valid Reaction selected.")
            return {'CANCELLED'}

        src = scn.reactions[src_idx]
        dst = scn.reactions.add()

        _deep_copy_pg(src, dst)

        try:
            dst.name = f"{src.name} (Copy)"
        except Exception:
            pass

        scn.reactions_index = len(scn.reactions) - 1
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────
# Reactions Panel (Global Library + Editor)
# ─────────────────────────────────────────────────────────

class EXPLORATORY_UL_ReactionLibrary(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "name", text="", emboss=False, icon='DOT')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.name)


class EXPLORATORY_OT_AddGlobalReaction(bpy.types.Operator):
    bl_idname = "exploratory.add_global_reaction"
    bl_label = "Add Reaction"

    def execute(self, context):
        scn = context.scene
        r = scn.reactions.add()
        r.name = f"Reaction_{len(scn.reactions)}"
        scn.reactions_index = len(scn.reactions) - 1
        return {'FINISHED'}


class EXPLORATORY_OT_RemoveGlobalReaction(bpy.types.Operator):
    bl_idname = "exploratory.remove_global_reaction"
    bl_label = "Remove Reaction"

    index: bpy.props.IntProperty()

    def execute(self, context):
        scn = context.scene
        if 0 <= self.index < len(scn.reactions):
            scn.reactions.remove(self.index)
            scn.reactions_index = max(0, min(self.index, len(scn.reactions) - 1))
        return {'FINISHED'}


class VIEW3D_PT_Exploratory_Reactions(bpy.types.Panel):
    bl_label = "Reactions"
    bl_idname = "VIEW3D_PT_exploratory_reactions"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        # Keep independent from Interactions filter—show when in CREATE.
        return (context.scene.main_category == 'CREATE')

    def draw(self, context):
        layout = self.layout
        scn = context.scene

        # Library list
        row = layout.row()
        row.template_list(
            "EXPLORATORY_UL_ReactionLibrary", "",
            scn, "reactions",
            scn, "reactions_index",
            rows=8
        )
        col = row.column(align=True)
        col.operator("exploratory.add_global_reaction", text="", icon='ADD')
        rem = col.operator("exploratory.remove_global_reaction", text="", icon='REMOVE')
        col.operator("exploratory.duplicate_global_reaction", text="", icon='DUPLICATE')
        rem.index = scn.reactions_index

        idx = scn.reactions_index
        if not (0 <= idx < len(scn.reactions)):
            layout.label(text="(No reactions in library)", icon='ERROR')
            return

        r = scn.reactions[idx]
        box = layout.box()
        box.label(text="Reaction", icon='OBJECT_DATA')
        box.prop(r, "name", text="Name")
        box.prop(r, "reaction_type", text="Type")

        # ===== Per-type UI (mirrors your previous fields) =====

        if r.reaction_type == "CUSTOM_ACTION":
            box.prop(r, "custom_action_message", text="Notes")
            box.prop_search(r, "custom_action_target", bpy.context.scene, "objects", text="Object")
            box.prop_search(r, "custom_action_action", bpy.data, "actions", text="Action")
            box.prop(r, "custom_action_loop", text="Loop?")
            if r.custom_action_loop:
                box.prop(r, "custom_action_loop_duration", text="Loop Duration")

        elif r.reaction_type == "CHAR_ACTION":
            box.prop_search(r, "char_action_ref", bpy.data, "actions", text="Action")
            box.prop(r, "char_action_mode", text="Mode")
            if r.char_action_mode == 'LOOP':
                box.prop(r, "char_action_loop_duration", text="Loop Duration")

        elif r.reaction_type == "OBJECTIVE_COUNTER":
            box.prop(r, "objective_index", text="Objective")
            box.prop(r, "objective_op", text="Operation")
            if r.objective_op in ("ADD", "SUBTRACT"):
                box.prop(r, "objective_amount", text="Amount")

        elif r.reaction_type == "PROPERTY":
            box.label(text="Paste a Blender data path (Right-Click → Copy Data Path).", icon='INFO')
            box.prop(r, "property_data_path", text="Data Path")
            row = box.row()
            row.label(text=f"Detected Type: {r.property_type}")
            box.prop(r, "property_transition_duration", text="Duration")
            box.prop(r, "property_reset", text="Reset after")
            if r.property_reset:
                box.prop(r, "property_reset_delay", text="Reset when target value is reached")
            if r.property_type == "BOOL":
                box.prop(r, "bool_value", text="Target Bool")
                box.prop(r, "default_bool_value", text="Default Bool")
            elif r.property_type == "INT":
                box.prop(r, "int_value", text="Target Int")
                box.prop(r, "default_int_value", text="Default Int")
            elif r.property_type == "FLOAT":
                box.prop(r, "float_value", text="Target Float")
                box.prop(r, "default_float_value", text="Default Float")
            elif r.property_type == "STRING":
                box.prop(r, "string_value", text="Target String")
                box.prop(r, "default_string_value", text="Default String")
            elif r.property_type == "VECTOR":
                box.label(text=f"Vector length: {r.vector_length}")
                box.prop(r, "vector_value", text="Target Vector")
                box.prop(r, "default_vector_value", text="Default Vector")

        elif r.reaction_type == "TRANSFORM":
            box.prop(r, "use_character", text="Use Character as Target")
            if r.use_character:
                char = context.scene.target_armature
                box.label(text=f"Character: {char.name if char else '—'}", icon='ARMATURE_DATA')
            else:
                box.prop_search(r, "transform_object", context.scene, "objects", text="Object")
            box.prop(r, "transform_mode", text="Mode")
            if r.transform_mode == "TO_OBJECT":
                box.prop_search(r, "transform_to_object", context.scene, "objects", text="To Object")
                colX = box.column(align=True)
                colX.label(text="Copy Channels:")
                colX.prop(r, "transform_use_location", text="Location")
                colX.prop(r, "transform_use_rotation", text="Rotation")
                colX.prop(r, "transform_use_scale",    text="Scale")
            if r.transform_mode in {"OFFSET", "LOCAL_OFFSET", "TO_LOCATION"}:
                box.prop(r, "transform_location", text="Location")
                box.prop(r, "transform_rotation", text="Rotation")
                box.prop(r, "transform_scale",    text="Scale")
            box.prop(r, "transform_duration", text="Duration")

        elif r.reaction_type == "CUSTOM_UI_TEXT":
            box.prop(r, "custom_text_subtype", text="Subtype")
            subtype = r.custom_text_subtype
            content_box = box.box()
            content_box.label(text="Text Content")
            if subtype == 'STATIC':
                content_box.prop(r, "custom_text_value", text="Text")
            else:
                content_box.prop(r, "text_objective_index", text="Objective")
            if subtype == 'OBJECTIVE':
                fmt = content_box.box()
                fmt.label(text="Counter Formatting")
                fmt.prop(r, "custom_text_prefix", text="Prefix")
                fmt.prop(r, "custom_text_include_counter", text="Show Counter")
                fmt.prop(r, "custom_text_suffix", text="Suffix")
            timing = box.box()
            timing.label(text="Display Timing")
            timing.prop(r, "custom_text_indefinite", text="Indefinite")
            if not r.custom_text_indefinite:
                timing.prop(r, "custom_text_duration", text="Duration (sec)")
            layout_box = box.box()
            layout_box.label(text="Position & Size")
            layout_box.prop(r, "custom_text_anchor", text="Anchor")
            layout_box.prop(r, "custom_text_scale", text="Scale")
            margins = layout_box.column(align=True)
            margins.prop(r, "custom_text_margin_x", text="Margin X")
            margins.prop(r, "custom_text_margin_y", text="Margin Y")
            style = box.box()
            style.label(text="Appearance")
            style.prop(r, "custom_text_font",  text="Font")
            style.prop(r, "custom_text_color", text="Color")
            preview_box = box.box()
            preview_box.label(text="Note: preview in fullscreen for best results.", icon='INFO')

            row = preview_box.row(align=True)
            op  = row.operator("exploratory.preview_custom_text", text="Preview Custom Text", icon='HIDE_OFF')
            op.duration = 5.0  # default preview length (seconds)


        elif r.reaction_type == "OBJECTIVE_TIMER":
            box.prop(r, "objective_index", text="Timer Objective")
            box.prop(r, "objective_timer_op", text="Timer Operation")
            box.prop(r, "interruptible", text="Interruptible")

        elif r.reaction_type == "MOBILITY_GAME":
            mg = r.mobility_game_settings
            box.label(text="Character Mobility")
            box.prop(mg, "allow_movement", text="Allow Movement")
            box.prop(mg, "allow_jump", text="Allow Jump")
            box.prop(mg, "allow_sprint", text="Allow Sprint")
            box.separator()
            box.label(text="Mesh Visibility Trigger")
            box.prop(mg, "mesh_object", text="Mesh Object")
            box.prop(mg, "mesh_action", text="Action")
            box.separator()
            box.label(text="Game Reset")
            box.prop(mg, "reset_game", text="Reset Game")

        elif r.reaction_type == "SOUND":
            box.prop(r, "sound_pointer", text="Sound")

            params = box.box()
            params.label(text="Parameters")
            params.prop(r, "sound_volume", text="Relative Volume")
            params.prop(r, "sound_use_distance", text="Use Distance?")
            if r.sound_use_distance:
                dist_box = params.box()
                dist_box.prop(r, "sound_distance_object", text="Distance Object")
                dist_box.prop(r, "sound_max_distance", text="Max Distance")
            params.prop(r, "sound_play_mode", text="Mode")
            if r.sound_play_mode == "DURATION":
                params.prop(r, "sound_duration", text="Duration")

            # ───── Pack helper (message + button in ONE box at bottom) ─────
            pack = box.box()
            pack.label(text="Custom sounds must be packed into the .blend.", icon='INFO')
            row = pack.row(align=True)
            row.operator("exp_audio.pack_all_sounds", text="Pack All Sounds", icon='PACKAGE')
            test_box = box.box()
            row = test_box.row(align=True)
            op  = row.operator("exp_audio.test_reaction_sound", text="Test Sound", icon='PLAY')
            op.reaction_index = scn.reactions_index  # current selection in the global library





##########OBJECTIVES PANEL######################
class EXPLORATORY_UL_Objectives(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        # item is an ObjectiveDefinition
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "name", text="", emboss=False, icon='DOT')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.name)


# -------------------------------------
# (Objectives panel)
# -------------------------------------
class VIEW3D_PT_Objectives(bpy.types.Panel):
    bl_label = "Objectives and Timers"
    bl_idname = "VIEW3D_PT_objectives"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (context.scene.main_category == 'CREATE'
                and _is_create_panel_enabled(context.scene, 'OBJ'))

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # List
        row = layout.row()
        row.template_list(
            "EXPLORATORY_UL_Objectives", "", 
            scene, "objectives", 
            scene, "objectives_index", 
            rows=3
        )
        col = row.column(align=True)
        col.operator("exploratory.add_objective", icon='ADD', text="")
        rem = col.operator("exploratory.remove_objective", icon='REMOVE', text="")
        rem.index = scene.objectives_index

        # Details
        idx = scene.objectives_index
        if 0 <= idx < len(scene.objectives):
            objv = scene.objectives[idx]

            # Basic Info
            info_box = layout.box()
            info_box.prop(objv, "name", text="Name")
            info_box.prop(objv, "description", text="Description")

            # Counter Settings
            counter_box = layout.box()
            counter_box.label(text="Objective Counter Settings")
            counter_box.prop(objv, "default_value", text="Default Value")
            counter_box.label(text="(Value at start/reset)")
            counter_box.prop(objv, "use_min_limit", text="Enable Min")
            if objv.use_min_limit:
                counter_box.prop(objv, "min_value", text="Min Value")
            counter_box.prop(objv, "use_max_limit", text="Enable Max")
            if objv.use_max_limit:
                counter_box.prop(objv, "max_value", text="Max Value")

            # Timer Settings
            timer_box = layout.box()
            timer_box.label(text="Objective Timer Settings")
            timer_box.prop(objv, "timer_mode", text="Mode")
            timer_box.prop(objv, "timer_start_value", text="Start Value")
            timer_box.prop(objv, "timer_end_value", text="End Value")




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
        # ─── Live Performance ─────────────────────────
        top = layout.column(align=True)
        row = top.row(align=True)
        row.prop(
            scene,
            "show_live_performance_overlay",
            text="Show Live Performance",
            icon='HIDE_OFF'
        )
        if scene.show_live_performance_overlay:
            top.prop(scene, "live_perf_scale", text="Scale")
        layout.separator()

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



