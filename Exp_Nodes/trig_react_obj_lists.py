# Exploratory/Exp_Nodes/npaneltriggerreactionlists.py

import bpy

EXPL_TREE_ID = "ExploratoryNodesTreeType"

def _in_exploratory_editor(context) -> bool:
    sd = getattr(context, "space_data", None)
    return bool(sd) and getattr(sd, "tree_type", "") == EXPL_TREE_ID


# ─────────────────────────────────────────────────────────
# UI Lists
# ─────────────────────────────────────────────────────────

class EXPLORATORY_UL_CustomInteractions(bpy.types.UIList):
    """
    Shows items in scene.custom_interactions (InteractionDefinition).
    """
    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "name", text="", emboss=False, icon='DOT')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.name)


class EXPLORATORY_UL_ReactionsInInteraction(bpy.types.UIList):
    """
    Shows InteractionDefinition.reaction_links[*].reaction_index resolved to scene.reactions[*].name
    """
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        scn  = context.scene
        name = "—"
        i = getattr(item, "reaction_index", -1)
        if 0 <= i < len(getattr(scn, "reactions", [])):
            name = scn.reactions[i].name
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.label(text=name, icon='DOT')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=name)


class EXPLORATORY_UL_ReactionLibrary(bpy.types.UIList):
    """
    Shows items in scene.reactions (ReactionDefinition).
    """
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "name", text="", emboss=False, icon='DOT')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.name)


class EXPLORATORY_UL_Objectives(bpy.types.UIList):
    """
    Shows items in scene.objectives (ObjectiveDefinition).
    """
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "name", text="", emboss=False, icon='DOT')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.name)


# ─────────────────────────────────────────────────────────
# Small helpers (UI only; no new data/registration)
# ─────────────────────────────────────────────────────────

def _scn():
    return getattr(bpy.context, "scene", None)

def _action_keys_exist(scn):
    try:
        return bool(getattr(scn, "action_keys", None)) and len(scn.action_keys) > 0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────
# Interactions Panel (N-panel)
# ─────────────────────────────────────────────────────────

class VIEW3D_PT_Exploratory_Studio(bpy.types.Panel):
    bl_label = "Interactions"
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
        scn = context.scene

        # List + add/remove/duplicate
        row = layout.row()
        row.template_list(
            "EXPLORATORY_UL_CustomInteractions",
            "",
            scn, "custom_interactions",
            scn, "custom_interactions_index",
            rows=5
        )
        col = row.column(align=True)
        col.operator("exploratory.add_interaction", text="", icon='ADD')
        rem = col.operator("exploratory.remove_interaction", text="", icon='REMOVE')
        col.operator("exploratory.duplicate_interaction", text="", icon='DUPLICATE')
        rem.index = scn.custom_interactions_index

        # Selection guard
        idx = scn.custom_interactions_index
        if not (0 <= idx < len(getattr(scn, "custom_interactions", []))):
            layout.label(text="Select an interaction to edit.", icon='INFO')
            return

        inter = scn.custom_interactions[idx]
        box = layout.box()
        box.prop(inter, "name", text="Name")
        box.prop(inter, "description", text="Description")

        # Trigger type and per-type fields (mirror node UI)
        box.prop(inter, "trigger_type", text="Trigger")

        if inter.trigger_type == "PROXIMITY":
            box.prop(inter, "use_character", text="Use Character as A")
            if inter.use_character:
                char = getattr(scn, "target_armature", None)
                box.label(text=f"Object A: {char.name if char else '—'}", icon='ARMATURE_DATA')
            else:
                box.prop(inter, "proximity_object_a", text="Object A")
            box.prop(inter, "proximity_object_b", text="Object B")
            box.prop(inter, "proximity_distance", text="Distance")

        elif inter.trigger_type == "COLLISION":
            box.prop(inter, "use_character", text="Use Character as A")
            if inter.use_character:
                char = getattr(scn, "target_armature", None)
                box.label(text=f"Object A: {char.name if char else '—'}", icon='ARMATURE_DATA')
            else:
                box.prop(inter, "collision_object_a", text="Object A")
            box.prop(inter, "collision_object_b", text="Object B")
            box.prop(inter, "collision_margin", text="Margin")

        elif inter.trigger_type == "INTERACT":
            box.prop(inter, "interact_object", text="Object")
            box.prop(inter, "interact_distance", text="Distance")

        elif inter.trigger_type == "ACTION":
            # Choose from Scene.action_keys by name, store into InteractionDefinition.action_key_id (string)
            if _action_keys_exist(scn):
                box.prop_search(inter, "action_key_id", scn, "action_keys", text="Action Key")
            else:
                row = box.row(align=True)
                row.enabled = False
                row.prop(inter, "action_key_id", text="Action Key")
                box.label(text="(No Action Keys. Add a 'Create Action Key' node.)", icon='INFO')

        elif inter.trigger_type == "OBJECTIVE_UPDATE":
            box.prop(inter, "objective_index", text="Objective")
            box.prop(inter, "objective_condition", text="Condition")
            if getattr(inter, "objective_condition", "") in {"EQUALS", "AT_LEAST"}:
                box.prop(inter, "objective_condition_value", text="Value")

        elif inter.trigger_type == "TIMER_COMPLETE":
            box.prop(inter, "timer_objective_index", text="Timer Objective")

        elif inter.trigger_type == "ON_GAME_START":
            box.label(text="Fires once when the game starts.", icon='TIME')

        # Shared trigger options
        opt = box.box()
        opt.label(text="Options")
        opt.prop(inter, "trigger_mode", text="Mode")
        if getattr(inter, "trigger_mode", "") == "COOLDOWN":
            opt.prop(inter, "trigger_cooldown", text="Cooldown")
        opt.prop(inter, "trigger_delay", text="Delay (sec)")

        # Linked reactions (ordered)
        sub = layout.box()
        sub.label(text="Linked Reactions")
        row2 = sub.row()
        row2.template_list(
            "EXPLORATORY_UL_ReactionsInInteraction", "",
            inter, "reaction_links",
            inter, "reaction_links_index",
            rows=6
        )
        col2 = row2.column(align=True)
        col2.operator("exploratory.add_reaction_to_interaction", text="", icon='ADD')
        rem2 = col2.operator("exploratory.remove_reaction_from_interaction", text="", icon='REMOVE')
        rem2.index = getattr(inter, "reaction_links_index", 0)
        col2.separator()
        col2.operator("exploratory.create_reaction_and_link", text="", icon='PLUS')


# ─────────────────────────────────────────────────────────
# Reaction Library + Editor (N-panel)
# ─────────────────────────────────────────────────────────

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
        if 0 <= self.index < len(getattr(scn, "reactions", [])):
            scn.reactions.remove(self.index)
            scn.reactions_index = max(0, min(self.index, len(scn.reactions) - 1))
        return {'FINISHED'}


class VIEW3D_PT_Exploratory_Reactions(bpy.types.Panel):
    bl_label = "Reactions"
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

        # Library list controls
        row = layout.row()
        row.template_list(
            "EXPLORATORY_UL_ReactionLibrary", "",
            scn, "reactions",
            scn, "reactions_index",
            rows=10
        )
        col = row.column(align=True)
        col.operator("exploratory.add_global_reaction", text="", icon='ADD')
        rem = col.operator("exploratory.remove_global_reaction", text="", icon='REMOVE')
        col.operator("exploratory.duplicate_global_reaction", text="", icon='DUPLICATE')
        rem.index = scn.reactions_index

        idx = scn.reactions_index
        if not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="No reaction selected.", icon='INFO')
            return

        r = scn.reactions[idx]

        # Header (name + type)
        box = layout.box()
        box.prop(r, "name", text="Name")
        box.prop(r, "reaction_type", text="Type")

        # Per-type UI to mirror node drawers (ALL current kinds)

        t = getattr(r, "reaction_type", "")

        if t == "CUSTOM_ACTION":
            box.prop_search(r, "custom_action_target", bpy.context.scene, "objects", text="Object")
            box.prop_search(r, "custom_action_action", bpy.data, "actions", text="Action")
            box.prop(r, "custom_action_message", text="Notes")
            box.prop(r, "custom_action_loop", text="Loop?")
            if getattr(r, "custom_action_loop", False):
                box.prop(r, "custom_action_loop_duration", text="Loop Duration")

        elif t == "CHAR_ACTION":
            box.prop_search(r, "char_action_ref", bpy.data, "actions", text="Action")
            box.prop(r, "char_action_mode", text="Mode")
            if getattr(r, "char_action_mode", "") == 'LOOP':
                box.prop(r, "char_action_loop_duration", text="Loop Duration")

        elif t == "SOUND":
            box.prop(r, "sound_pointer", text="Sound")
            params = box.box()
            params.label(text="Parameters")
            params.prop(r, "sound_volume", text="Relative Volume")
            params.prop(r, "sound_use_distance", text="Use Distance?")
            if getattr(r, "sound_use_distance", False):
                d = params.box()
                d.prop(r, "sound_distance_object", text="Distance Object")
                d.prop(r, "sound_max_distance", text="Max Distance")
            params.prop(r, "sound_play_mode", text="Mode")
            if getattr(r, "sound_play_mode", "") == "DURATION":
                params.prop(r, "sound_duration", text="Duration")
            pack = box.box()
            pack.label(text="Custom sounds must be packed.", icon='INFO')
            rowp = pack.row(align=True)
            rowp.operator("exp_audio.pack_all_sounds", text="Pack All", icon='PACKAGE')
            # quick test button
            test = box.box()
            rowt = test.row(align=True)
            op = rowt.operator("exp_audio.test_reaction_sound", text="Test Sound", icon='PLAY')
            op.reaction_index = idx

        elif t == "PROPERTY":
            box.label(text="Use Right-Click → Copy Full Data Path.", icon='INFO')
            box.prop(r, "property_data_path", text="Full Data Path")
            rowt = box.row(); rowt.label(text=f"Detected Type: {getattr(r, 'property_type', 'NONE')}")
            box.prop(r, "property_transition_duration", text="Duration")
            box.prop(r, "property_reset", text="Reset after")
            if getattr(r, "property_reset", False):
                box.prop(r, "property_reset_delay", text="Reset delay")

            pt = getattr(r, "property_type", "NONE")
            if pt == "BOOL":
                box.prop(r, "bool_value", text="Target Bool")
                box.prop(r, "default_bool_value", text="Default Bool")
            elif pt == "INT":
                box.prop(r, "int_value", text="Target Int")
                box.prop(r, "default_int_value", text="Default Int")
            elif pt == "FLOAT":
                box.prop(r, "float_value", text="Target Float")
                box.prop(r, "default_float_value", text="Default Float")
            elif pt == "STRING":
                box.prop(r, "string_value", text="Target String")
                box.prop(r, "default_string_value", text="Default String")
            elif pt == "VECTOR":
                box.label(text=f"Vector length: {getattr(r, 'vector_length', 3)}")
                box.prop(r, "vector_value", text="Target Vector")
                box.prop(r, "default_vector_value", text="Default Vector")

        elif t == "TRANSFORM":
            box.prop(r, "use_character", text="Use Character as Target")
            if getattr(r, "use_character", False):
                char = getattr(scn, "target_armature", None)
                box.label(text=f"Character: {char.name if char else '—'}", icon='ARMATURE_DATA')
            else:
                box.prop_search(r, "transform_object", scn, "objects", text="Object")
            box.prop(r, "transform_mode", text="Mode")
            if getattr(r, "transform_mode", "") == "TO_OBJECT":
                box.prop_search(r, "transform_to_object", scn, "objects", text="To Object")
                col = box.column(align=True)
                col.label(text="Copy Channels:")
                col.prop(r, "transform_use_location", text="Location")
                col.prop(r, "transform_use_rotation", text="Rotation")
                col.prop(r, "transform_use_scale",    text="Scale")
            if getattr(r, "transform_mode", "") in {"OFFSET", "LOCAL_OFFSET", "TO_LOCATION"}:
                box.prop(r, "transform_location", text="Location")
                box.prop(r, "transform_rotation", text="Rotation")
                box.prop(r, "transform_scale",    text="Scale")
            box.prop(r, "transform_duration", text="Duration")

        elif t == "CUSTOM_UI_TEXT":
            box.prop(r, "custom_text_subtype", text="Subtype")
            subtype = getattr(r, "custom_text_subtype", "STATIC")

            content = box.box()
            content.label(text="Text Content")
            if subtype == "STATIC":
                content.prop(r, "custom_text_value", text="Text")
            else:
                content.prop(r, "text_objective_index", text="Objective")

            if subtype == "OBJECTIVE":
                fmt = content.box()
                fmt.label(text="Counter Formatting")
                fmt.prop(r, "custom_text_prefix", text="Prefix")
                fmt.prop(r, "custom_text_include_counter", text="Show Counter")
                fmt.prop(r, "custom_text_suffix", text="Suffix")

            timing = box.box()
            timing.label(text="Display Timing")
            timing.prop(r, "custom_text_indefinite", text="Indefinite")
            if not getattr(r, "custom_text_indefinite", False):
                timing.prop(r, "custom_text_duration", text="Duration (sec)")

            layout_box = box.box()
            layout_box.label(text="Position & Size")
            layout_box.prop(r, "custom_text_anchor", text="Anchor")
            layout_box.prop(r, "custom_text_scale", text="Scale")
            m = layout_box.column(align=True)
            m.prop(r, "custom_text_margin_x", text="Margin X")
            m.prop(r, "custom_text_margin_y", text="Margin Y")

            style = box.box()
            style.label(text="Appearance")
            style.prop(r, "custom_text_font",  text="Font")
            style.prop(r, "custom_text_color", text="Color")

            preview = box.box()
            rowp = preview.row(align=True)
            op  = rowp.operator("exploratory.preview_custom_text", text="Preview (5s)", icon='HIDE_OFF')
            op.duration = 5.0

        elif t == "ENABLE_CROSSHAIRS":
            box.prop(r, "crosshair_style", text="Style")
            dims = box.box()
            dims.label(text="Dimensions (px)")
            dims.prop(r, "crosshair_length_px",     text="Arm Length")
            dims.prop(r, "crosshair_gap_px",        text="Gap")
            dims.prop(r, "crosshair_thickness_px",  text="Thickness")
            dims.prop(r, "crosshair_dot_radius_px", text="Dot Radius")
            box.prop(r, "crosshair_color", text="Color")
            tbox = box.box()
            tbox.label(text="Timing")
            tbox.prop(r, "crosshair_indefinite", text="Indefinite")
            if not getattr(r, "crosshair_indefinite", True):
                tbox.prop(r, "crosshair_duration", text="Duration (sec)")

        elif t == "OBJECTIVE_COUNTER":
            box.prop(r, "objective_index", text="Objective")
            box.prop(r, "objective_op", text="Operation")
            if getattr(r, "objective_op", "") in {"ADD", "SUBTRACT"}:
                box.prop(r, "objective_amount", text="Amount")

        elif t == "OBJECTIVE_TIMER":
            box.prop(r, "objective_index", text="Timer Objective")
            box.prop(r, "objective_timer_op", text="Timer Operation")
            box.prop(r, "interruptible", text="Interruptible")

        elif t == "MOBILITY":
            ms = getattr(r, "mobility_settings", None)
            if ms:
                box.label(text="Character Mobility")
                box.prop(ms, "allow_movement", text="Allow Movement")
                box.prop(ms, "allow_jump",     text="Allow Jump")
                box.prop(ms, "allow_sprint",   text="Allow Sprint")

        elif t == "MESH_VISIBILITY":
            vs = getattr(r, "mesh_visibility", None)
            if vs:
                box.label(text="Mesh Visibility")
                box.prop(vs, "mesh_object", text="Mesh Object")
                box.prop(vs, "mesh_action", text="Action")

        elif t == "RESET_GAME":
            box.label(text="Reset the game on trigger.", icon='FILE_REFRESH')

        # ───── NEW: DELAY utility reaction ─────
        elif t == "DELAY":
            box.prop(r, "utility_delay_seconds", text="Delay (sec)")
            box.label(text="Delays all subsequent reactions in the chain.", icon='TIME')

        # ───── NEW: PROJECTILE reaction ─────
        elif t == "PROJECTILE":
            org = box.box(); org.label(text="Origin")
            org.prop(r, "proj_use_character_origin", text="Use Character Origin")
            if not getattr(r, "proj_use_character_origin", False):
                org.prop_search(r, "proj_origin_object", scn, "objects", text="Origin Object")
            org.prop(r, "proj_origin_offset", text="Offset")

            aim = box.box(); aim.label(text="Aim")
            aim.prop(r, "proj_aim_source", text="Aim Source")

            vis = box.box(); vis.label(text="Visual (Optional)")
            vis.prop_search(r, "proj_object", scn, "objects", text="Projectile Object")
            vis.prop(r, "proj_align_object_to_velocity", text="Align to Velocity")

            pj = box.box(); pj.label(text="Projectile")
            pj.prop(r, "proj_speed", text="Speed")
            pj.prop(r, "proj_gravity", text="Gravity")
            pj.prop(r, "proj_lifetime", text="Lifetime")
            pj.prop(r, "proj_on_contact_stop", text="Stop on Contact")
            pj.prop(r, "proj_pool_limit", text="Max Active")

        # ───── NEW: HITSCAN reaction ─────
        elif t == "HITSCAN":
            org = box.box(); org.label(text="Origin")
            org.prop(r, "proj_use_character_origin", text="Use Character Origin")
            if not getattr(r, "proj_use_character_origin", False):
                org.prop_search(r, "proj_origin_object", scn, "objects", text="Origin Object")
            org.prop(r, "proj_origin_offset", text="Offset")

            aim = box.box(); aim.label(text="Aim")
            aim.prop(r, "proj_aim_source", text="Aim Source")

            vis = box.box(); vis.label(text="Visual (Optional)")
            vis.prop_search(r, "proj_object", scn, "objects", text="Place Object")
            vis.prop(r, "proj_align_object_to_velocity", text="Align to Direction")

            hs = box.box(); hs.label(text="Hitscan")
            hs.prop(r, "proj_max_range", text="Max Range")
            hs.prop(r, "proj_place_hitscan_object", text="Place Object at Impact")

        # ───── NEW: ACTION KEYS reaction (enable/disable/toggle) ─────
        elif t == "ACTION_KEYS":
            row = box.row(align=True)
            row.prop(r, "action_key_op", text="Operation")
            if _action_keys_exist(scn):
                box.prop_search(r, "action_key_name", scn, "action_keys", text="Action")
            else:
                row2 = box.row(align=True)
                row2.enabled = False
                row2.prop(r, "action_key_name", text="Action")
                box.label(text="(No Action Keys. Add a 'Create Action Key' node.)", icon='INFO')


# ─────────────────────────────────────────────────────────
# Objectives Panel (N-panel)
# ─────────────────────────────────────────────────────────

class VIEW3D_PT_Objectives(bpy.types.Panel):
    bl_label = "Objectives & Timers"
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
        scn = context.scene

        # List
        row = layout.row()
        row.template_list(
            "EXPLORATORY_UL_Objectives", "",
            scn, "objectives",
            scn, "objectives_index",
            rows=5
        )
        col = row.column(align=True)
        col.operator("exploratory.add_objective", icon='ADD', text="")
        rem = col.operator("exploratory.remove_objective", icon='REMOVE', text="")
        rem.index = scn.objectives_index

        idx = scn.objectives_index
        if not (0 <= idx < len(getattr(scn, "objectives", []))):
            layout.label(text="Select an objective to edit.", icon='INFO')
            return

        objv = scn.objectives[idx]

        # Basic
        info = layout.box()
        info.prop(objv, "name", text="Name")
        info.prop(objv, "description", text="Description")

        # Counter
        counter = layout.box()
        counter.label(text="Counter")
        counter.prop(objv, "default_value", text="Default Value")
        counter.prop(objv, "use_min_limit", text="Enable Min")
        if getattr(objv, "use_min_limit", False):
            counter.prop(objv, "min_value", text="Min Value")
        counter.prop(objv, "use_max_limit", text="Enable Max")
        if getattr(objv, "use_max_limit", False):
            counter.prop(objv, "max_value", text="Max Value")

        # Timer
        timer = layout.box()
        timer.label(text="Timer")
        timer.prop(objv, "timer_mode", text="Mode")
        timer.prop(objv, "timer_start_value", text="Start Value")
        timer.prop(objv, "timer_end_value", text="End Value")
