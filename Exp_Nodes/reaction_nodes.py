# File: Exp_Nodes/reaction_nodes.py
import bpy
from bpy.types import Node
from .base_nodes import ReactionNodeBase

# ───────────────────────── helpers ─────────────────────────

def _scene() -> bpy.types.Scene | None:
    scn = getattr(bpy.context, "scene", None)
    if scn:
        return scn
    return bpy.data.scenes[0] if bpy.data.scenes else None

def _enum_action_key_items(self, context):
    scn = getattr(context, "scene", None) or _scene()
    items = []
    if scn and hasattr(scn, "action_keys") and scn.action_keys:
        for it in scn.action_keys:
            nm = getattr(it, "name", "")
            if nm:
                items.append((nm, nm, f"Action: {nm}"))
    if not items:
        items.append(("", "(No Action Keys)", "Add a 'Create Action Key' node first"))
    return items

def _fix_interaction_reaction_indices_after_remove(removed_index: int) -> None:
    scn = _scene()
    if not scn or not hasattr(scn, "custom_interactions"):
        return
    for inter in scn.custom_interactions:
        links = getattr(inter, "reaction_links", None)
        if not links:
            continue
        # remove links pointing to removed index
        to_remove = [i for i, l in enumerate(links) if getattr(l, "reaction_index", -1) == removed_index]
        for i in reversed(to_remove):
            links.remove(i)
        # shift higher indices down by 1
        for l in links:
            idx = getattr(l, "reaction_index", -1)
            if idx > removed_index:
                l.reaction_index = idx - 1

def _reindex_reaction_nodes_after_remove(removed_index: int) -> None:
    """
    After removing scene.reactions[removed_index], all subsequent items shift -1.
    This walks every Exploratory node tree and fixes each Reaction node's
    node-local reaction_index so it still points to the same logical Reaction.
    """
    for ng in bpy.data.node_groups:
        if getattr(ng, "bl_idname", "") != "ExploratoryNodesTreeType":
            continue
        for node in ng.nodes:
            # Only touch our reaction nodes that carry a reaction_index
            blid = getattr(node, "bl_idname", "")
            if not blid.startswith("Reaction"):
                continue
            if not hasattr(node, "reaction_index"):
                continue

            idx = getattr(node, "reaction_index", -1)
            if idx < 0:
                continue

            if idx == removed_index:
                # This node is being freed (or points at the removed slot).
                # If somehow we got here for a surviving node, mark invalid to avoid mis-pointing.
                try:
                    node.reaction_index = -1
                except Exception:
                    pass
            elif idx > removed_index:
                # Shift down by one to track the same logical item.
                try:
                    node.reaction_index = idx - 1
                except Exception:
                    pass


def _ensure_reaction(kind_label: str) -> int:
    scn = _scene()
    if not scn:
        return -1
    r = scn.reactions.add()
    r.name = f"{kind_label}_{len(scn.reactions)}"
    if hasattr(r, "reaction_type"):
        try:
            r.reaction_type = kind_label
        except Exception:
            pass
    scn.reactions_index = len(scn.reactions) - 1
    return scn.reactions_index


def _duplicate_reaction_via_operator(src_index: int) -> int:
    scn = _scene()
    if not scn or not (0 <= src_index < len(scn.reactions)):
        return -1
    # preferred path: your operator
    try:
        res = bpy.ops.exploratory.duplicate_global_reaction(index=src_index)
        if 'CANCELLED' not in res:
            return len(scn.reactions) - 1
    except Exception:
        pass
    # fallback shallow copy
    try:
        src = scn.reactions[src_index]
        dst = scn.reactions.add()
        dst.name = f"{getattr(src, 'name', 'Reaction')} (Copy)"
        for prop in src.bl_rna.properties:
            ident = prop.identifier
            if ident == "rna_type" or getattr(prop, "is_readonly", False) or getattr(prop, "is_collection", False):
                continue
            try:
                setattr(dst, ident, getattr(src, ident))
            except Exception:
                pass
        return len(scn.reactions) - 1
    except Exception:
        return -1


def _force_kind(r, kind: str):
    """Make sure the stored reaction type matches the node's KIND."""
    if hasattr(r, "reaction_type"):
        try:
            if r.reaction_type != kind:
                r.reaction_type = kind
        except Exception:
            pass


def _draw_common_fields(layout, r, kind: str):
    """Draw reaction fields without showing 'Type' or any identifier."""
    # keep Name editable
    header = layout.box()
    header.prop(r, "name", text="Name")

    # ensure type stays pinned to node KIND silently
    _force_kind(r, kind)

    t = kind  # use pinned kind instead of reading the enum back

    if t == "CUSTOM_ACTION":
        header.prop(r, "custom_action_message", text="Notes")
        header.prop_search(r, "custom_action_target", bpy.context.scene, "objects", text="Object")
        header.prop_search(r, "custom_action_action", bpy.data, "actions", text="Action")
        header.prop(r, "custom_action_loop", text="Loop?")
        if getattr(r, "custom_action_loop", False):
            header.prop(r, "custom_action_loop_duration", text="Loop Duration")

    elif t == "CHAR_ACTION":
        header.prop_search(r, "char_action_ref", bpy.data, "actions", text="Action")
        header.prop(r, "char_action_mode", text="Mode")
        if getattr(r, "char_action_mode", "") == "LOOP":
            header.prop(r, "char_action_loop_duration", text="Loop Duration")

    elif t == "SOUND":
        # Sound pointer
        header.prop(r, "sound_pointer", text="Sound")

        # Parameters
        params = header.box()
        params.label(text="Parameters")
        params.prop(r, "sound_volume", text="Relative Volume")
        params.prop(r, "sound_use_distance", text="Use Distance?")
        if getattr(r, "sound_use_distance", False):
            dist_box = params.box()
            dist_box.prop(r, "sound_distance_object", text="Distance Object")
            dist_box.prop(r, "sound_max_distance", text="Max Distance")
        params.prop(r, "sound_play_mode", text="Mode")
        if getattr(r, "sound_play_mode", "") == "DURATION":
            params.prop(r, "sound_duration", text="Duration")

        # Pack helper (message + button in ONE box at bottom)
        pack = header.box()
        pack.label(text="Custom sounds must be packed into the .blend.", icon='INFO')
        row = pack.row(align=True)
        row.operator("exp_audio.pack_all_sounds", text="Pack All Sounds", icon='PACKAGE')
        

        # ---- Test button (preview playback) ----
        scn = bpy.context.scene
        ridx = -1
        for i, rx in enumerate(getattr(scn, "reactions", [])):
            if rx == r:
                ridx = i
                break
        test_box = header.box()
        row = test_box.row(align=True)
        op  = row.operator("exp_audio.test_reaction_sound", text="Test Sound", icon='PLAY')
        op.reaction_index = ridx


    elif t == "PROPERTY":
        header.label(text="Paste a Blender full data path (Right-Click → Copy Full Data Path).", icon='INFO')
        header.prop(r, "property_data_path", text="Full Data Path")
        row = header.row(); row.label(text=f"Detected Type: {getattr(r, 'property_type', 'NONE')}")
        header.prop(r, "property_transition_duration", text="Duration")
        header.prop(r, "property_reset", text="Reset after")
        if getattr(r, "property_reset", False):
            header.prop(r, "property_reset_delay", text="Reset delay")

        pt = getattr(r, "property_type", "NONE")
        if pt == "BOOL":
            header.prop(r, "bool_value", text="Target Bool")
            header.prop(r, "default_bool_value", text="Default Bool")
        elif pt == "INT":
            header.prop(r, "int_value", text="Target Int")
            header.prop(r, "default_int_value", text="Default Int")
        elif pt == "FLOAT":
            header.prop(r, "float_value", text="Target Float")
            header.prop(r, "default_float_value", text="Default Float")
        elif pt == "STRING":
            header.prop(r, "string_value", text="Target String")
            header.prop(r, "default_string_value", text="Default String")
        elif pt == "VECTOR":
            header.label(text=f"Vector length: {getattr(r, 'vector_length', 3)}")
            header.prop(r, "vector_value", text="Target Vector")
            header.prop(r, "default_vector_value", text="Default Vector")

    elif t == "TRANSFORM":
        header.prop(r, "use_character", text="Use Character as Target")
        if getattr(r, "use_character", False):
            char = bpy.context.scene.target_armature
            header.label(text=f"Character: {char.name if char else '—'}", icon='ARMATURE_DATA')
        else:
            header.prop_search(r, "transform_object", bpy.context.scene, "objects", text="Object")
        header.prop(r, "transform_mode", text="Mode")
        if getattr(r, "transform_mode", "") == "TO_OBJECT":
            header.prop_search(r, "transform_to_object", bpy.context.scene, "objects", text="To Object")
            col = header.column(align=True)
            col.label(text="Copy Channels:")
            col.prop(r, "transform_use_location", text="Location")
            col.prop(r, "transform_use_rotation", text="Rotation")
            col.prop(r, "transform_use_scale",    text="Scale")
        if getattr(r, "transform_mode", "") in {"OFFSET","LOCAL_OFFSET","TO_LOCATION"}:
            header.prop(r, "transform_location", text="Location")
            header.prop(r, "transform_rotation", text="Rotation")
            header.prop(r, "transform_scale",    text="Scale")
        header.prop(r, "transform_duration", text="Duration")

    elif t == "CUSTOM_UI_TEXT":
        header.prop(r, "custom_text_subtype", text="Subtype")
        subtype = getattr(r, "custom_text_subtype", "STATIC")

        # Text content
        content = header.box()
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

        # Timing
        timing = header.box()
        timing.label(text="Display Timing")
        timing.prop(r, "custom_text_indefinite", text="Indefinite")
        if not getattr(r, "custom_text_indefinite", False):
            timing.prop(r, "custom_text_duration", text="Duration (sec)")

        # Position & Size
        layout_box = header.box()
        layout_box.label(text="Position & Size")
        layout_box.prop(r, "custom_text_anchor", text="Anchor")
        layout_box.prop(r, "custom_text_scale", text="Scale")
        margins = layout_box.column(align=True)
        margins.prop(r, "custom_text_margin_x", text="Margin X")
        margins.prop(r, "custom_text_margin_y", text="Margin Y")

        # Appearance
        style = header.box()
        style.label(text="Appearance")
        style.prop(r, "custom_text_font",  text="Font")
        style.prop(r, "custom_text_color", text="Color")

        # Preview (same structure as N-panel)
        preview_box = header.box()
        preview_box.label(text="Note: preview in fullscreen for best results.", icon='INFO')
        row = preview_box.row(align=True)
        op  = row.operator("exploratory.preview_custom_text", text="Preview Custom Text", icon='HIDE_OFF')
        op.duration = 5.0

    elif t == "ENABLE_CROSSHAIRS":
        header.prop(r, "crosshair_style", text="Style")

        dims = header.box()
        dims.label(text="Dimensions (pixels)")
        dims.prop(r, "crosshair_length_px",    text="Arm Length")
        dims.prop(r, "crosshair_gap_px",       text="Gap")
        dims.prop(r, "crosshair_thickness_px", text="Thickness")
        dims.prop(r, "crosshair_dot_radius_px", text="Dot Radius")

        header.prop(r, "crosshair_color", text="Color")

        timing = header.box()
        timing.label(text="Timing")
        timing.prop(r, "crosshair_indefinite", text="Indefinite")
        if not getattr(r, "crosshair_indefinite", True):
            timing.prop(r, "crosshair_duration", text="Duration (sec)")

    elif t == "OBJECTIVE_COUNTER":
        header.prop(r, "objective_index", text="Objective")
        header.prop(r, "objective_op", text="Operation")
        if getattr(r, "objective_op", "") in {"ADD","SUBTRACT"}:
            header.prop(r, "objective_amount", text="Amount")

    elif t == "OBJECTIVE_TIMER":
        header.prop(r, "objective_index", text="Timer Objective")
        header.prop(r, "objective_timer_op", text="Timer Operation")
        header.prop(r, "interruptible", text="Interruptible")

    elif t == "MOBILITY":
        ms = getattr(r, "mobility_settings", None)
        if ms:
            header.label(text="Mobility")
            header.prop(ms, "allow_movement", text="Allow Movement")
            header.prop(ms, "allow_jump",     text="Allow Jump")
            header.prop(ms, "allow_sprint",   text="Allow Sprint")

    elif t == "MESH_VISIBILITY":
        vs = getattr(r, "mesh_visibility", None)
        if vs:
            header.label(text="Mesh Visibility")
            header.prop(vs, "mesh_object", text="Mesh Object")
            header.prop(vs, "mesh_action", text="Action")

    elif t == "RESET_GAME":
        header.label(text="Reset Game on trigger", icon='FILE_REFRESH')

# ───────────────────────── sockets ─────────────────────────

class ReactionTriggerInputSocket(bpy.types.NodeSocket):
    bl_idname = "ReactionTriggerInputSocketType"
    bl_label  = "Reaction Trigger Input"
    def draw(self, context, layout, node, text): layout.label(text=text)
    def draw_color(self, context, node): return (0.15, 0.55, 1.0, 1.0)

class ReactionOutputSocket(bpy.types.NodeSocket):
    bl_idname = "ReactionOutputSocketType"
    bl_label  = "Reaction Output"
    def draw(self, context, layout, node, text): layout.label(text=text)
    def draw_color(self, context, node): return (0.15, 0.55, 1.0, 1.0)


# ───────────────────────── base class ─────────────────────────

class _ReactionNodeKind(ReactionNodeBase):
    """Base for concrete reaction nodes. Subclasses must set KIND to a valid ReactionDefinition.reaction_type."""
    KIND = None

    reaction_index: bpy.props.IntProperty(name="Reaction Index", default=-1, min=-1)


    # subtle mid-green (body) to contrast triggers/objectives without shouting
    _EXPL_TINT_REACTION = (0.18, 0.24, 0.18)
    def _tint(self):
        try:
            self.use_custom_color = True
            self.color = self._EXPL_TINT_REACTION
        except Exception:
            pass

    def init(self, context):
        self.inputs.new("ReactionTriggerInputSocketType", "Reaction Input")
        self.outputs.new("ReactionOutputSocketType",      "Reaction Output")
        self.width = 300
        self._tint()
        self.reaction_index = _ensure_reaction(self.KIND or "CUSTOM_ACTION")

    def copy(self, node):
        src_idx = getattr(node, "reaction_index", -1)
        self.reaction_index = _duplicate_reaction_via_operator(src_idx)
        self.width = getattr(node, "width", 300)

    def free(self):
        """
        Deleting a Reaction node should:
          1) Remove its reaction from scene.reactions
          2) Fix InteractionDefinition.reaction_links indices
          3) Fix every other Reaction node's reaction_index so they don't drift
        """
        scn = _scene()
        idx = self.reaction_index
        if scn and hasattr(scn, "reactions") and 0 <= idx < len(scn.reactions):
            # 1) remove the actual reaction item
            scn.reactions.remove(idx)
            # 2) repair interaction link collections
            _fix_interaction_reaction_indices_after_remove(idx)
            # 3) repair all other nodes' local indices
            _reindex_reaction_nodes_after_remove(idx)

        self.reaction_index = -1

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(scn.reactions)):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]
        _draw_common_fields(layout, r, self.KIND or "CUSTOM_ACTION")

    def execute_reaction(self, context):
        pass

class ReactionHitscanNode(_ReactionNodeKind):
    bl_idname = "ReactionHitscanNodeType"
    bl_label  = "Hitscan"
    KIND = "HITSCAN"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]

        box = layout.box()
        box.prop(r, "name", text="Name")

        org = layout.box()
        org.label(text="Origin")
        org.prop(r, "proj_use_character_origin", text="Use Character Origin")
        if not r.proj_use_character_origin:
            org.prop_search(r, "proj_origin_object", bpy.context.scene, "objects", text="Origin Object")
        org.prop(r, "proj_origin_offset", text="Offset")

        aim = layout.box()
        aim.label(text="Aim")
        aim.prop(r, "proj_aim_source", text="Aim Source")

        vis = layout.box()
        vis.label(text="Visual (Optional)")
        vis.prop_search(r, "proj_object", bpy.context.scene, "objects", text="Place Object")
        vis.prop(r, "proj_align_object_to_velocity", text="Align to Direction")

        hs = layout.box()
        hs.label(text="Hitscan")
        hs.prop(r, "proj_max_range", text="Max Range")
        hs.prop(r, "proj_place_hitscan_object", text="Place Object at Impact")


class ReactionProjectileNode(_ReactionNodeKind):
    bl_idname = "ReactionProjectileNodeType"
    bl_label  = "Projectile"
    KIND = "PROJECTILE"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]

        box = layout.box()
        box.prop(r, "name", text="Name")

        org = layout.box()
        org.label(text="Origin")
        org.prop(r, "proj_use_character_origin", text="Use Character Origin")
        if not r.proj_use_character_origin:
            org.prop_search(r, "proj_origin_object", bpy.context.scene, "objects", text="Origin Object")
        org.prop(r, "proj_origin_offset", text="Offset")

        aim = layout.box()
        aim.label(text="Aim")
        aim.prop(r, "proj_aim_source", text="Aim Source")

        vis = layout.box()
        vis.label(text="Visual (Optional)")
        vis.prop_search(r, "proj_object", bpy.context.scene, "objects", text="Projectile Object")
        vis.prop(r, "proj_align_object_to_velocity", text="Align to Velocity")

        pj = layout.box()
        pj.label(text="Projectile")
        pj.prop(r, "proj_speed", text="Speed")
        pj.prop(r, "proj_gravity", text="Gravity")
        pj.prop(r, "proj_lifetime", text="Lifetime")
        pj.prop(r, "proj_on_contact_stop", text="Stop on Contact")
        pj.prop(r, "proj_pool_limit", text="Max Active")



class ReactionActionKeysNode(_ReactionNodeKind):
    bl_idname = "ReactionActionKeysNodeType"
    bl_label  = "Action Keys"
    KIND = "ACTION_KEYS"

    node_action_key: bpy.props.EnumProperty(
        name="Action",
        items=_enum_action_key_items,
        description="Choose which Action Key this reaction will enable/disable/toggle",
        update=lambda self, ctx: self._on_node_action_changed(ctx),
    )

    def _on_node_action_changed(self, context):
        scn = _scene()
        idx = getattr(self, "reaction_index", -1)
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            return
        r = scn.reactions[idx]
        name = getattr(self, "node_action_key", "") or ""
        # mirror into ReactionDefinition for executor
        try:
            r.action_key_name = name
            r.action_key_id   = name
        except Exception:
            pass
        # best-effort index mirror for reindexing on list edits
        if hasattr(scn, "action_keys"):
            match = -1
            try:
                for i, it in enumerate(scn.action_keys):
                    if getattr(it, "name", "") == name:
                        match = i
                        break
            except Exception:
                pass
            try:
                r.action_key_index = match
            except Exception:
                pass

    def update(self):
        # keep chooser in sync with ReactionDefinition (no base update required)
        scn = _scene()
        idx = getattr(self, "reaction_index", -1)
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            return

        r = scn.reactions[idx]
        want = getattr(r, "action_key_name", "") or getattr(r, "action_key_id", "")
        if not want:
            return

        try:
            valid = {it[0] for it in _enum_action_key_items(self, bpy.context)}
            if want in valid and getattr(self, "node_action_key", "") != want:
                self.node_action_key = want
        except Exception:
            pass

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(getattr(scn, "reactions", []))):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]

        box = layout.box()
        box.prop(r, "name", text="Name")

        row = layout.row(align=True)
        row.prop(r, "action_key_op", text="Operation")

        layout.prop(self, "node_action_key", text="Action")

        info = layout.box()
        info.label(text="Pick an existing Action Key. No creation here.", icon='INFO')

# ───────────────────────── concrete nodes ─────────────────────────

class ReactionCustomActionNode(_ReactionNodeKind):
    bl_idname = "ReactionCustomActionNodeType"
    bl_label  = "Custom Action"
    KIND = "CUSTOM_ACTION"

class ReactionCharActionNode(_ReactionNodeKind):
    bl_idname = "ReactionCharActionNodeType"
    bl_label  = "Character Action"
    KIND = "CHAR_ACTION"

class ReactionSoundNode(_ReactionNodeKind):
    bl_idname = "ReactionSoundNodeType"
    bl_label  = "Play Sound"
    KIND = "SOUND"

class ReactionPropertyNode(_ReactionNodeKind):
    bl_idname = "ReactionPropertyNodeType"
    bl_label  = "Property Value"
    KIND = "PROPERTY"

class ReactionTransformNode(_ReactionNodeKind):
    bl_idname = "ReactionTransformNodeType"
    bl_label  = "Transform"
    KIND = "TRANSFORM"

class ReactionCustomTextNode(_ReactionNodeKind):
    bl_idname = "ReactionCustomTextNodeType"
    bl_label  = "Custom UI Text"
    KIND = "CUSTOM_UI_TEXT"

class ReactionObjectiveCounterNode(_ReactionNodeKind):
    bl_idname = "ReactionObjectiveCounterNodeType"
    bl_label  = "Objective Counter"
    KIND = "OBJECTIVE_COUNTER"

class ReactionObjectiveTimerNode(_ReactionNodeKind):
    bl_idname = "ReactionObjectiveTimerNodeType"
    bl_label  = "Objective Timer"
    KIND = "OBJECTIVE_TIMER"

class ReactionMobilityNode(_ReactionNodeKind):
    bl_idname = "ReactionMobilityNodeType"
    bl_label  = "Mobility"
    KIND = "MOBILITY"

class ReactionMeshVisibilityNode(_ReactionNodeKind):
    bl_idname = "ReactionMeshVisibilityNodeType"
    bl_label  = "Mesh Visibility"
    KIND = "MESH_VISIBILITY"

class ReactionResetGameNode(_ReactionNodeKind):
    bl_idname = "ReactionResetGameNodeType"
    bl_label  = "Reset Game"
    KIND = "RESET_GAME"

class ReactionCrosshairsNode(_ReactionNodeKind):
    bl_idname = "ReactionCrosshairsNodeType"
    bl_label  = "Enable Crosshairs"
    KIND = "ENABLE_CROSSHAIRS"

# ───────────────────────── registration ─────────────────────────

_CLASSES = [
    ReactionTriggerInputSocket,
    ReactionOutputSocket,
    ReactionCustomActionNode,
    ReactionCharActionNode,
    ReactionSoundNode,
    ReactionPropertyNode,
    ReactionTransformNode,
    ReactionCustomTextNode,
    ReactionObjectiveCounterNode,
    ReactionObjectiveTimerNode,
    ReactionMobilityNode,
    ReactionMeshVisibilityNode,
    ReactionResetGameNode,
]


def register():
    for c in _CLASSES:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
