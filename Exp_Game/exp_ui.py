# File: exp_ui.py
import bpy
from .exp_utilities import (
    get_game_world)
# --------------------------------------------------------------------
# Exploratory Modal Panel
# --------------------------------------------------------------------
class ExploratoryPanel(bpy.types.Panel):
    bl_label = "Exploratory Modal Panel"
    bl_idname = "VIEW3D_PT_exploratory_modal"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Always show the mode toggle:
        layout.prop(scene, "main_category", expand=True)
        layout.separator()
        layout.separator()

        # Only show the modal operator button if in CREATE mode:
        if scene.main_category == 'CREATE':
            op = layout.operator("view3d.exp_modal", text="Start Game (Not Fullscreen)")
            op.launched_from_ui = False  # Direct modal launch.
            op.should_revert_workspace = False  # Do not revert workspace.
            layout.operator("exploratory.start_game", text="Start Game (Fullscreen)")

            layout.separator()

            # Provide a button to mark the current scene as the game world
            layout.operator("exploratory.set_game_world", text="Set Game World", icon='WORLD')

            game_world = get_game_world()
            if game_world:
                layout.label(text=f"Game World: {game_world.name}")

            else:
                row = layout.row()
                row.alert = True
                row.label(text="No Game World Assigned!")
                
                row = layout.row()
                row.alert = True
                row.label(text="Switch to your desired scene and click 'Set Game World' to assign it.")





# --------------------------------------------------------------------
# Character, Actions, Audio (only visible in Explore mode)
# --------------------------------------------------------------------
class ExploratoryCharacterPanel(bpy.types.Panel):
    bl_label = "Character, Actions, Audio"
    bl_idname = "VIEW3D_PT_exploratory_character"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    @classmethod
    def poll(cls, context):
        return context.scene.main_category == 'CREATE'

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
        char_col.label(text="• If OFF the character filepath set in preferences will be used.")
        char_col.label(text="• Easy and stable for working character filepaths (see preferences)")

        # ─── Build Character Button ───
        box.separator()
        btn = box.row()
        btn.scale_y = 1.0
        btn.operator(
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
        action_col.label(text="• Useful for creating a world with custom actions.")
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
        audio_col.label(text="• Uses the audio settings defined in preferences.")
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
            row = box.row()
            snd = getattr(audio, prop_name)
            row.prop(audio, prop_name, text=label)
            if snd:
                sub = box.row(align=True)
                sub.prop(snd, "sound_speed", text="Speed")
                op = sub.operator("exp_audio.test_sound_pointer", text="Test", icon='PLAY')
                op.sound_slot = prop_name
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

# --------------------------------------------------------------------
# Proxy Meshes (only visible in Create mode)
# --------------------------------------------------------------------
class ExploratoryProxyMeshPanel(bpy.types.Panel):
    bl_label = "Proxy Meshes"
    bl_idname = "VIEW3D_PT_exploratory_proxy"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    # Show only when main_category is set to "CREATE"
    @classmethod
    def poll(cls, context):
        return context.scene.main_category == 'CREATE'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.separator()
        layout.label(text="Proxy Meshes")
        row = layout.row()
        row.template_list(
            "EXPLORATORY_UL_ProxyMeshList",  # Your UIList class name
            "",
            scene,
            "proxy_meshes",          # The collection property
            scene,
            "proxy_meshes_index",    # The integer property for the active item
            rows=4
        )

        # The side column with add/remove operators
        col = row.column(align=True)
        col.operator("exploratory.add_proxy_mesh", text="", icon='ADD')
        remove_op = col.operator("exploratory.remove_proxy_mesh", text="", icon='REMOVE')
        remove_op.index = scene.proxy_meshes_index

        layout.separator()

        # Show details for the currently selected proxy mesh
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

                # — Warning Group —
                warn_grp = dyn_box.column(align=True)
                warn_grp.label(text="⚠ Mesh rebuild every frame!")
                warn_grp.label(text="Optimize mesh and use Register Distance")
                warn_grp.label(text="Very expensive calculations for complex geometry!")

                dyn_box.separator()

                # — Register Distance Group —
                dist_grp = dyn_box.column(align=True)
                dist_grp.prop(entry, "register_distance", text="Register Distance")
                dist_grp.label(text="Register Distance = Distance from mesh before calculations are considered.")
                dist_grp.label(text="0.0 = Always on.")

                
            box.prop(entry, "hide_during_game", text="Hide During Game")


        layout.separator()
        layout.label(text="Spawn Object")
        layout.prop(scene, "spawn_object", text="")


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
    """UIList that displays ReactionDefinition items within one Interaction."""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        # 'data' is the InteractionDefinition
        # 'item' is the ReactionDefinition
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "name", text="", emboss=False, icon='DOT')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.name)

class VIEW3D_PT_Exploratory_Studio(bpy.types.Panel):
    """Your main Interactions panel, extended with Reactions sub-list."""
    bl_label = "Custom Interactions"
    bl_idname = "VIEW3D_PT_exploratory_studio"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.scene.main_category == "CREATE"
    def draw(self, context):
        layout = self.layout
        scene = context.scene

        ######################
        # (A) The Interaction List
        ######################
        layout.separator()
        layout.operator("view3d.exp_modal", text="Start Exploratory Modal")
        layout.separator()

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
            box.prop(inter, "use_character", text="Use Character?")
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
            box.prop_search(inter, "objective_index", scene, "objectives", text="Objective")
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
        # (B) Reactions sub-list
        ######################
        sub = box.box()
        sub.label(text="Reactions for this Interaction", icon='OBJECT_DATA')

        row2 = sub.row()
        row2.template_list(
            "EXPLORATORY_UL_ReactionsInInteraction",  # The nested UIList
            "",
            inter,               # Data is the InteractionDefinition
            "reactions",         # The CollectionProperty of ReactionDefinitions
            inter,               # Active data
            "reactions_index",   # The property for the active reaction index
            rows=3
        )
        col2 = row2.column(align=True)
        col2.operator("exploratory.add_reaction_to_interaction", text="", icon='ADD')
        rem_op = col2.operator("exploratory.remove_reaction_from_interaction", text="", icon='REMOVE')
        rem_op.index = inter.reactions_index

        # "Current Reaction" box
        if not inter.reactions:
            sub.label(text="(No reactions added)", icon='ERROR')
            return

        r_idx = inter.reactions_index
        if r_idx < 0 or r_idx >= len(inter.reactions):
            sub.label(text="(Select a reaction)", icon='INFO')
            return

        reaction = inter.reactions[r_idx]
        box2 = sub.box()
        box2.label(text="Current Reaction", icon='OBJECT_DATA')
        box2.prop(reaction, "name", text="Name")
        box2.prop(reaction, "reaction_type", text="Type")

        #custom object action --------------
        if reaction.reaction_type == "CUSTOM_ACTION":
            box2.prop(reaction, "custom_action_message", text="Notes")
            box2.prop_search(reaction, "custom_action_target", bpy.context.scene, "objects", text="Object")
            box2.prop_search(reaction, "custom_action_action", bpy.data, "actions", text="Action")
            box2.prop(reaction, "custom_action_loop", text="Loop?")
            if reaction.custom_action_loop:
                box2.prop(reaction, "custom_action_loop_duration", text="Loop Duration")

        #custom character action -------------------
        elif reaction.reaction_type == "CHAR_ACTION":
            # Show the pointer to the Action
            box2.prop_search(reaction, "char_action_ref", bpy.data, "actions", text="Action")

            # Show the new dropdown
            box2.prop(reaction, "char_action_mode", text="Mode")

            # If user selected 'LOOP', show the loop duration
            if reaction.char_action_mode == 'LOOP':
                box2.prop(reaction, "char_action_loop_duration", text="Loop Duration")

        #--Objective reaction----------------------------#
        elif reaction.reaction_type == "OBJECTIVE_COUNTER":
            # 1) Which objective to increment/decrement?
            box2.prop(reaction, "objective_index", text="Objective")

            # 2) The main operation (Add/Sub/Reset)
            box2.prop(reaction, "objective_op", text="Operation")

            # 3) If Add or Subtract, let user specify how much
            if reaction.objective_op in ("ADD", "SUBTRACT"):
                box2.prop(reaction, "objective_amount", text="Amount")



        #-Property reaction----------------------------#
        elif reaction.reaction_type == "PROPERTY":
            box2.label(
                text="For node properties, copy the properties from the node graph (e.g. the specific node)",
                icon='INFO'
            )
            box2.label(
                text="Group input properties do not reflect real-time changes in the node graph.",
                icon='INFO'
            )
            box2.prop(reaction, "property_data_path", text="Data Path")

            # Show the detected property type
            row = box2.row()
            row.label(text=f"Detected Type: {reaction.property_type}")
            box2.prop(reaction, "property_transition_duration", text="Duration")
            box2.prop(reaction, "property_reset", text="Reset after")
            if reaction.property_reset:
                box2.prop(reaction, "property_reset_delay", text="Reset when target value is reached")
            
            # Display input fields based on the detected type.
            if reaction.property_type == "BOOL":
                box2.prop(reaction, "bool_value", text="Target Bool Value")
                box2.prop(reaction, "default_bool_value", text="Default Bool Value")
            elif reaction.property_type == "INT":
                box2.prop(reaction, "int_value", text="Target Int Value")
                box2.prop(reaction, "default_int_value", text="Default Int Value")
            elif reaction.property_type == "FLOAT":
                box2.prop(reaction, "float_value", text="Target Float Value")
                box2.prop(reaction, "default_float_value", text="Default Float Value")
            elif reaction.property_type == "STRING":
                box2.prop(reaction, "string_value", text="Target String Value")
                box2.prop(reaction, "default_string_value", text="Default String Value")
            elif reaction.property_type == "VECTOR":
                box2.label(text=f"Vector length: {reaction.vector_length}")
                box2.prop(reaction, "vector_value", text="Target Vector")
                box2.prop(reaction, "default_vector_value", text="Default Vector")
            else:
                box2.label(text="No property detected or invalid path.")


        #custom transform reaction ----------
        elif reaction.reaction_type == "TRANSFORM":
            box2.prop_search(reaction, "transform_object", bpy.context.scene, "objects", text="Object")

            # 1) The new transform mode dropdown
            box2.prop(reaction, "transform_mode", text="Mode")

            # 2) If "TO_OBJECT" => show target object pointer
            if reaction.transform_mode == "TO_OBJECT":
                box2.prop_search(reaction, "transform_to_object", bpy.context.scene, "objects", text="To Object")

            # Show location/rotation/scale fields if relevant
            if reaction.transform_mode in {"OFFSET","TO_LOCATION","LOCAL_OFFSET"}:
                box2.prop(reaction, "transform_location", text="Location")
                box2.prop(reaction, "transform_rotation", text="Rotation")
                box2.prop(reaction, "transform_scale", text="Scale")

            box2.prop(reaction, "transform_duration", text="Duration")


        #custom UI text reaction ---------------
        elif reaction.reaction_type == "CUSTOM_UI_TEXT":
            # 1) Subtype selector
            box2.prop(reaction, "custom_text_subtype", text="Subtype")
            subtype = reaction.custom_text_subtype
            box2.label(text="Note: Preview custom text in fullscreen mode for best results.")

            # ── A) Content ─────────────────────────────────────────────────────────
            content_box = box2.box()
            content_box.label(text="Text Content")
            if subtype == 'STATIC':
                content_box.prop(reaction, "custom_text_value", text="Text")
            else:
                content_box.prop(reaction, "text_objective_index", text="Objective")

            # ── B) Counter Formatting (OBJECTIVE only) ─────────────────────────────
            if subtype == 'OBJECTIVE':
                fmt = content_box.box()
                fmt.label(text="Counter Formatting")
                fmt.prop(reaction, "custom_text_prefix", text="Prefix")
                fmt.prop(reaction, "custom_text_include_counter", text="Show Counter")
                fmt.prop(reaction, "custom_text_suffix", text="Suffix")

            # ── C) Display Timing ───────────────────────────────────────────────────
            timing_box = box2.box()
            timing_box.label(text="Display Timing")
            timing_box.prop(reaction, "custom_text_indefinite", text="Indefinite")
            if not reaction.custom_text_indefinite:
                timing_box.prop(reaction, "custom_text_duration", text="Duration (sec)")

            # ── D) Position & Size ─────────────────────────────────────────────────
            layout_box = box2.box()
            layout_box.label(text="Position & Size")
            layout_box.prop(reaction, "custom_text_anchor", text="Anchor")
            layout_box.prop(reaction, "custom_text_scale", text="Scale")
            margins = layout_box.column(align=True)
            margins.prop(reaction, "custom_text_margin_x", text="Margin X")
            margins.prop(reaction, "custom_text_margin_y", text="Margin Y")

            # ── E) Appearance ──────────────────────────────────────────────────────
            style_box = box2.box()
            style_box.label(text="Appearance")
            style_box.prop(reaction, "custom_text_color", text="Color")

            # ── F) Preview Button ───────────────────────────────────────────────────
            box2.separator()
            box2.operator("exploratory.preview_custom_text", text="Preview Text")


        # ===============================
        # Objective Timer Start/Stop
        # ===============================
        elif reaction.reaction_type == "OBJECTIVE_TIMER":
            box2.prop(reaction, "objective_index", text="Timer Objective")
            box2.prop(reaction, "objective_timer_op", text="Timer Operation")
            box2.prop(reaction, "interruptible", text="Interruptible")


        # ===============================
        # Character and Game Reactions
        # ===============================
        elif reaction.reaction_type == "MOBILITY_GAME":
            mg = reaction.mobility_game_settings
            box2.label(text="Character Mobility")
            box2.prop(mg, "allow_movement", text="Allow Movement")
            box2.prop(mg, "allow_jump", text="Allow Jump")
            box2.prop(mg, "allow_sprint", text="Allow Sprint")
            
            box2.separator()
            
            box2.label(text="Mesh Visibility Trigger")
            box2.prop(mg, "mesh_object", text="Mesh Object")
            box2.prop(mg, "mesh_action", text="Action")

        # ===============================
        # Sound Reaction
        # ===============================
        elif reaction.reaction_type == "SOUND":
            # ───── Notice ─────
            note_box = box2.box()
            note_box.label(text="Notice: Custom Sounds must be packed into the .blend to work in-game.")
            note_box.label(text="Reason: Custom sounds can't rely on external file sources.")

            # ───── Packing ─────
            pack_box = box2.box()
            pack_box.operator(
                "exp_audio.pack_all_sounds",
                text="Pack All Sounds",
                icon='PACKAGE'
            )

            # ───── Load / Test ─────
            io_box = box2.box()
            io_box.label(text="Load / Test Sound")
            row = io_box.row(align=True)
            row.prop(reaction, "sound_pointer", text="Sound")

            load_op = row.operator(
                "exp_audio.load_audio_file",
                text="",
                icon='FILE_FOLDER'
            )
            load_op.interaction_index = scene.custom_interactions_index
            load_op.reaction_index    = inter.reactions_index

            test_op = row.operator(
                "exp_audio.test_reaction_sound",
                text="",
                icon='PLAY'
            )
            test_op.interaction_index = scene.custom_interactions_index
            test_op.reaction_index    = inter.reactions_index

            # ───── Parameters ─────
            params_box = box2.box()
            params_box.label(text="Parameters")
            params_box.prop(reaction, "sound_volume", text="Relative Volume")
            params_box.prop(reaction, "sound_use_distance", text="Use Distance?")

            if reaction.sound_use_distance:
                dist_box = params_box.box()
                dist_box.prop(reaction, "sound_distance_object", text="Distance Object")
                dist_box.prop(reaction, "sound_max_distance", text="Max Distance")

            params_box.prop(reaction, "sound_play_mode", text="Mode")
            if reaction.sound_play_mode == "DURATION":
                params_box.prop(reaction, "sound_duration", text="Duration")








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
    bl_label = "Objectives"
    bl_idname = "VIEW3D_PT_objectives"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):

        return context.scene.main_category == "CREATE"
    def draw(self, context):
        layout = self.layout
        scene = context.scene

        row = layout.row()
        row.template_list(
            "EXPLORATORY_UL_Objectives",
            "",
            scene, "objectives",
            scene, "objectives_index",
            rows=3
        )
        col = row.column(align=True)
        col.operator("exploratory.add_objective", text="", icon='ADD')
        rem = col.operator("exploratory.remove_objective", text="", icon='REMOVE')
        rem.index = scene.objectives_index

        idx = scene.objectives_index
        if 0 <= idx < len(scene.objectives):
            objv = scene.objectives[idx]
            box = layout.box()
            box.prop(objv, "name", text="Name")
            box.prop(objv, "description", text="Description")

            box.label(text="Objective Counter:")
            box.prop(objv, "default_value", text="Default Value")
            box.label(text="Default Value = value when the game starts or resets.")
            box.separator()

            # ─── NEW: optional min/max limits UI ────────────────────────
            box.separator()
            box.label(text="Value Limits (optional):")
            box.prop(objv, "use_min_limit", text="Enable Minimum")
            if objv.use_min_limit:
                box.prop(objv, "min_value", text="Minimum Value")
            box.prop(objv, "use_max_limit", text="Enable Maximum")
            if objv.use_max_limit:
                box.prop(objv, "max_value", text="Maximum Value")

            box.separator()
            box.label(text="Timer Settings:")
            box.prop(objv, "timer_mode",       text="Mode")
            box.prop(objv, "timer_start_value",text="Start Value")
            box.prop(objv, "timer_end_value",  text="End Value")

            box.label(text=f"Current Timer Value: {objv.timer_value:.1f}")