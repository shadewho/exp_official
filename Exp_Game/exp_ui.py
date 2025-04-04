# File: exp_ui.py
import bpy
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
            op.launched_from_ui = False  # Indicates it's a direct modal launch.
            op.should_revert_workspace = False  # Don't revert workspace for direct modal.

            # Button to launch the full-screen game (which switches workspaces).
            layout.operator("exploratory.start_game", text="Start Game (Fullscreen)")




# --------------------------------------------------------------------
# NEW PANEL: Character, Actions, Audio (only visible in Explore mode)
# --------------------------------------------------------------------
class ExploratoryCharacterPanel(bpy.types.Panel):
    bl_label = "Character, Actions, Audio"
    bl_idname = "VIEW3D_PT_exploratory_character"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    # Optionally, you can hide this panel when in Create mode:
    @classmethod
    def poll(cls, context):
        return context.scene.main_category == 'CREATE'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "target_armature", text="Target Armature")
        layout.label(text="Animation Slots")
        layout.prop(scene.character_actions, "idle_action")
        layout.prop(scene.character_actions, "walk_action")
        layout.prop(scene.character_actions, "run_action")
        layout.prop(scene.character_actions, "jump_action")
        layout.prop(scene.character_actions, "fall_action")
        layout.prop(scene.character_actions, "land_action")

        layout.separator()
        layout.label(text="Action Speeds:")
        char_actions = scene.character_actions
        # Instead of showing scene-level speed properties, show the speed from the Action data-block.
        if char_actions.idle_action:
            layout.prop(char_actions.idle_action, "action_speed", text="Idle Speed")
        else:
            layout.label(text="Idle Action not assigned")

        if char_actions.walk_action:
            layout.prop(char_actions.walk_action, "action_speed", text="Walk Speed")
        else:
            layout.label(text="Walk Action not assigned")

        if char_actions.run_action:
            layout.prop(char_actions.run_action, "action_speed", text="Run Speed")
        else:
            layout.label(text="Run Action not assigned")

        if char_actions.jump_action:
            layout.prop(char_actions.jump_action, "action_speed", text="Jump Speed")
        else:
            layout.label(text="Jump Action not assigned")

        if char_actions.fall_action:
            layout.prop(char_actions.fall_action, "action_speed", text="Fall Speed")
        else:
            layout.label(text="Fall Action not assigned")

        if char_actions.land_action:
            layout.prop(char_actions.land_action, "action_speed", text="Land Speed")
        else:
            layout.label(text="Land Action not assigned")

        layout.separator()
        layout.label(text="Audio Pointers:")

        # Walk Sound
        row = layout.row()
        row.prop(scene.character_audio, "walk_sound", text="Walk Sound")

        if scene.character_audio.walk_sound:
            row = layout.row(align=True)
            row.prop(scene.character_audio.walk_sound, "sound_speed", text="Speed")
            op = row.operator("exp_audio.test_sound_pointer", text="Test", icon='PLAY')
            op.sound_slot = "walk_sound"
        else:
            row = layout.row()
            row.label(text="(No Walk Sound)")

        # Run Sound
        row = layout.row()
        row.prop(scene.character_audio, "run_sound", text="Run Sound")

        if scene.character_audio.run_sound:
            row = layout.row(align=True)
            row.prop(scene.character_audio.run_sound, "sound_speed", text="Speed")
            op = row.operator("exp_audio.test_sound_pointer", text="Test", icon='PLAY')
            op.sound_slot = "run_sound"
        else:
            row = layout.row()
            row.label(text="(No Run Sound)")

        # Jump Sound
        row = layout.row()
        row.prop(scene.character_audio, "jump_sound", text="Jump Sound")

        if scene.character_audio.jump_sound:
            row = layout.row(align=True)
            row.prop(scene.character_audio.jump_sound, "sound_speed", text="Speed")
            op = row.operator("exp_audio.test_sound_pointer", text="Test", icon='PLAY')
            op.sound_slot = "jump_sound"
        else:
            row = layout.row()
            row.label(text="(No Jump Sound)")

        # Fall Sound
        row = layout.row()
        row.prop(scene.character_audio, "fall_sound", text="Fall Sound")

        if scene.character_audio.fall_sound:
            row = layout.row(align=True)
            row.prop(scene.character_audio.fall_sound, "sound_speed", text="Speed")
            op = row.operator("exp_audio.test_sound_pointer", text="Test", icon='PLAY')
            op.sound_slot = "fall_sound"
        else:
            row = layout.row()
            row.label(text="(No Fall Sound)")

        # Land Sound
        row = layout.row()
        row.prop(scene.character_audio, "land_sound", text="Land Sound")

        if scene.character_audio.land_sound:
            row = layout.row(align=True)
            row.prop(scene.character_audio.land_sound, "sound_speed", text="Speed")
            op = row.operator("exp_audio.test_sound_pointer", text="Test", icon='PLAY')
            op.sound_slot = "land_sound"
        else:
            row = layout.row()
            row.label(text="(No Land Sound)")


# --------------------------------------------------------------------
# NEW PANEL: Proxy Meshes (only visible in Create mode)
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
            box.prop(entry, "is_moving", text="Is Moving")
            # New line to display the hide_during_game boolean:
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
            box.prop(inter, "proximity_object_a", text="Object A")
            box.prop(inter, "proximity_object_b", text="Object B")

        elif inter.trigger_type == "COLLISION":
            box.prop(inter, "collision_object_a", text="Object A")
            box.prop(inter, "collision_object_b", text="Object B")

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
            # The user pastes the entire path, e.g. "bpy.data.materials['MyMat'].node_tree.nodes..."
            box2.prop(reaction, "property_data_path", text="Data Path")

            # Show the detected property type
            row = box2.row()
            row.label(text=f"Detected Type: {reaction.property_type}")
            box2.prop(reaction, "property_transition_duration", text="Duration")
            box2.prop(reaction, "property_reset", text="Reset?")
            if reaction.property_reset:
                box2.prop(reaction, "property_reset_delay", text="Reset Delay")
            # Depending on the type, show a different field:
            if reaction.property_type == "BOOL":
                box2.prop(reaction, "bool_value", text="New Bool Value")
            elif reaction.property_type == "INT":
                box2.prop(reaction, "int_value", text="New Int Value")
            elif reaction.property_type == "FLOAT":
                box2.prop(reaction, "float_value", text="New Float Value")
            elif reaction.property_type == "STRING":
                box2.prop(reaction, "string_value", text="New String Value")
            elif reaction.property_type == "VECTOR":
                box2.label(text=f"Vector length: {reaction.vector_length}")
                box2.prop(reaction, "vector_value", text="New Vector")
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
            box2.prop(reaction, "transform_distance", text="Distance")


        #custom UI text reaction ---------------
        elif reaction.reaction_type == "CUSTOM_UI_TEXT":
            # 1) Always show the dropdown for which subtype of custom UI text
            box2.prop(reaction, "custom_text_subtype", text="Subtype")

            # ===============================
            #  A) Static Text
            # ===============================
            if reaction.custom_text_subtype == "STATIC":
                box2.prop(reaction, "custom_text_value", text="Text")
                box2.prop(reaction, "custom_text_indefinite", text="Indefinite?")
                if not reaction.custom_text_indefinite:
                    box2.prop(reaction, "custom_text_duration", text="Duration")

                box2.prop(reaction, "custom_text_anchor", text="Anchor")
                box2.prop(reaction, "custom_text_scale", text="Scale")
                box2.prop(reaction, "custom_text_margin_x", text="Margin X")
                box2.prop(reaction, "custom_text_margin_y", text="Margin Y")
                box2.prop(reaction, "custom_text_color", text="Color")

            # ===============================
            #  B) Objective Display
            # ===============================
            elif reaction.custom_text_subtype == "OBJECTIVE":
                # This subtype (sometimes called OBJECTIVE_DISPLAY) shows objective.current_value
                box2.prop(reaction, "text_objective_index", text="Objective")
                box2.prop(reaction, "text_objective_format", text="Format")  # e.g. "Coins: {value}"

                box2.prop(reaction, "custom_text_indefinite", text="Indefinite?")
                if not reaction.custom_text_indefinite:
                    box2.prop(reaction, "custom_text_duration", text="Duration")

                box2.prop(reaction, "custom_text_anchor", text="Anchor")
                box2.prop(reaction, "custom_text_scale", text="Scale")
                box2.prop(reaction, "custom_text_margin_x", text="Margin X")
                box2.prop(reaction, "custom_text_margin_y", text="Margin Y")
                box2.prop(reaction, "custom_text_color", text="Color")

            # ===============================
            #  C) Objective Timer Display
            # ===============================
            elif reaction.custom_text_subtype == "OBJECTIVE_TIMER_DISPLAY":
                # This subtype shows an objective’s timer countdown
                box2.prop(reaction, "text_objective_index", text="Objective")
                box2.prop(reaction, "custom_text_indefinite", text="Indefinite?")
                if not reaction.custom_text_indefinite:
                    box2.prop(reaction, "custom_text_duration", text="Duration")

                box2.prop(reaction, "custom_text_anchor", text="Anchor")
                box2.prop(reaction, "custom_text_scale", text="Scale")
                box2.prop(reaction, "custom_text_margin_x", text="Margin X")
                box2.prop(reaction, "custom_text_margin_y", text="Margin Y")
                box2.prop(reaction, "custom_text_color", text="Color")

        # ===============================
        # Objective Timer Start/Stop
        # ===============================
        elif reaction.reaction_type == "OBJECTIVE_TIMER":
            # Show the Objective dropdown
            box2.prop(reaction, "objective_index", text="Timer Objective")

            # Let the user pick START or STOP
            box2.prop(reaction, "objective_timer_op", text="Timer Operation")


        # ===============================
        # Character and Game Reactions
        # ===============================
        elif reaction.reaction_type == "MOBILITY_GAME":
            mg = reaction.mobility_game_settings
            box2.prop(mg, "allow_movement", text="Allow Movement")
            box2.prop(mg, "allow_jump", text="Allow Jump")
            box2.prop(mg, "allow_sprint", text="Allow Sprint")

        # ===============================
        # Sound Reaction
        # ===============================
        elif reaction.reaction_type == "SOUND":
            box2.label(text="Play Packed Sound")
            box2.prop(reaction, "sound_volume", text="Relative Volume")
            box2.prop(reaction, "sound_use_distance", text="Use Distance?")


            box2.prop(reaction, "sound_pointer", text="Sound Datablock")
            box2.prop(reaction, "sound_play_mode", text="Mode")

            if reaction.sound_play_mode == "DURATION":
                box2.prop(reaction, "sound_duration", text="Duration")

            if reaction.sound_use_distance:
                box2.prop(reaction, "sound_distance_object", text="Distance Obj")
                box2.prop(reaction, "sound_max_distance",   text="Max Distance")






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

            box.label(text="Integer Counter:")
            box.prop(objv, "default_value", text="Default Value")
            box.prop(objv, "current_value", text="Current Value")

            box.separator()
            box.label(text="Timer Settings:")
            box.prop(objv, "timer_mode",       text="Mode")
            box.prop(objv, "timer_start_value",text="Start Value")
            box.prop(objv, "timer_end_value",  text="End Value")

            box.label(text=f"Current Timer Value: {objv.timer_value:.1f}")