#Exploratory/Exp_Nodes/npaneltriggerreactionlists.py

import bpy

EXPL_TREE_ID = "ExploratoryNodesTreeType"

def _in_exploratory_editor(context) -> bool:
    sd = getattr(context, "space_data", None)
    return bool(sd) and getattr(sd, "tree_type", "") == EXPL_TREE_ID

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
    bl_label = "Interactions (Legacy)"
    bl_idname = "VIEW3D_PT_exploratory_studio"
    bl_space_type = 'NODE_EDITOR'   
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return _in_exploratory_editor(context)

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        row = layout.row()
        row.alert = True
        row.label(text="Warning: Legacy system. Use Nodes.", icon='ERROR')

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
    bl_label = "Reactions (Legacy)"
    bl_idname = "VIEW3D_PT_exploratory_reactions"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}


    @classmethod
    def poll(cls, context):
        return _in_exploratory_editor(context)

    def draw(self, context):
        layout = self.layout
        scn = context.scene

        row = layout.row()
        row.alert = True
        row.label(text="Warning: Legacy system. Use Nodes.", icon='ERROR')

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

        elif r.reaction_type == "MOBILITY":
            ms = getattr(r, "mobility_settings", None)
            if ms:
                box.label(text="Character Mobility")
                box.prop(ms, "allow_movement", text="Allow Movement")
                box.prop(ms, "allow_jump",     text="Allow Jump")
                box.prop(ms, "allow_sprint",   text="Allow Sprint")

        elif r.reaction_type == "MESH_VISIBILITY":
            vs = getattr(r, "mesh_visibility", None)
            if vs:
                box.label(text="Mesh Visibility")
                box.prop(vs, "mesh_object", text="Mesh Object")
                box.prop(vs, "mesh_action", text="Action")

        elif r.reaction_type == "RESET_GAME":
            box.label(text="Reset Game")


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
    bl_label = "Objectives and Timers (Legacy)"
    bl_idname = "VIEW3D_PT_objectives"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}


    @classmethod
    def poll(cls, context):
        return _in_exploratory_editor(context)

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        row = layout.row()
        row.alert = True
        row.label(text="Warning: Legacy system. Use Nodes.", icon='ERROR')

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

