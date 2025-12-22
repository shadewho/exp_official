# Exploratory/Exp_Nodes/trig_react_obj_lists.py
# Read-only visualization of interactions, reactions, and objectives.
# Includes orphan/stale detection for debugging.

import bpy

EXPL_TREE_ID = "ExploratoryNodesTreeType"


def _in_exploratory_editor(context) -> bool:
    sd = getattr(context, "space_data", None)
    return bool(sd) and getattr(sd, "tree_type", "") == EXPL_TREE_ID


# ─────────────────────────────────────────────────────────
# Orphan/Stale Detection Helpers
# ─────────────────────────────────────────────────────────

def _collect_referenced_indices():
    """
    Scan all Exploratory node trees and collect which indices are actually
    referenced by nodes. Returns dict with sets of referenced indices.
    """
    referenced = {
        "interactions": set(),
        "reactions": set(),
        "objectives": set(),
        "action_keys": set(),
    }

    for ng in bpy.data.node_groups:
        if getattr(ng, "bl_idname", "") != EXPL_TREE_ID:
            continue

        for node in ng.nodes:
            # Trigger nodes reference interactions
            if hasattr(node, "interaction_index"):
                idx = getattr(node, "interaction_index", -1)
                if idx >= 0:
                    referenced["interactions"].add(idx)

            # Reaction nodes reference reactions
            if hasattr(node, "reaction_index"):
                idx = getattr(node, "reaction_index", -1)
                if idx >= 0:
                    referenced["reactions"].add(idx)

            # Objective nodes reference objectives
            if getattr(node, "bl_idname", "") == "ObjectiveNodeType":
                idx = getattr(node, "objective_index", -1)
                if idx >= 0:
                    referenced["objectives"].add(idx)

            # Action key nodes reference action keys
            if getattr(node, "bl_idname", "") == "CreateActionKeyNodeType":
                idx = getattr(node, "action_key_index", -1)
                if idx >= 0:
                    referenced["action_keys"].add(idx)

    return referenced


def _find_invalid_reaction_links(scn):
    """
    Find reaction_links that point to non-existent reaction indices.
    Returns list of (interaction_index, interaction_name, bad_reaction_index).
    """
    invalid = []
    interactions = getattr(scn, "custom_interactions", [])
    reactions = getattr(scn, "reactions", [])
    num_reactions = len(reactions)

    for i, inter in enumerate(interactions):
        links = getattr(inter, "reaction_links", [])
        for link in links:
            ridx = getattr(link, "reaction_index", -1)
            if ridx < 0 or ridx >= num_reactions:
                invalid.append((i, inter.name, ridx))

    return invalid


# ─────────────────────────────────────────────────────────
# Delete Orphaned Operator
# ─────────────────────────────────────────────────────────

class EXPL_OT_delete_orphaned(bpy.types.Operator):
    """Delete all orphaned interactions, reactions, and objectives that are not referenced by any node"""
    bl_idname = "exploratory.delete_orphaned"
    bl_label = "Delete Orphaned"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _in_exploratory_editor(context)

    def execute(self, context):
        scn = context.scene
        referenced = _collect_referenced_indices()

        deleted_counts = {"interactions": 0, "reactions": 0, "objectives": 0}

        # Delete orphaned reactions (reverse order to preserve indices)
        reactions = getattr(scn, "reactions", None)
        if reactions:
            ref_react = referenced["reactions"]
            to_delete = [i for i in range(len(reactions)) if i not in ref_react]
            for idx in reversed(to_delete):
                reactions.remove(idx)
                deleted_counts["reactions"] += 1

        # Delete orphaned interactions (reverse order)
        interactions = getattr(scn, "custom_interactions", None)
        if interactions:
            ref_inter = referenced["interactions"]
            to_delete = [i for i in range(len(interactions)) if i not in ref_inter]
            for idx in reversed(to_delete):
                interactions.remove(idx)
                deleted_counts["interactions"] += 1

        # Delete orphaned objectives (reverse order)
        objectives = getattr(scn, "objectives", None)
        if objectives:
            ref_obj = referenced["objectives"]
            to_delete = [i for i in range(len(objectives)) if i not in ref_obj]
            for idx in reversed(to_delete):
                objectives.remove(idx)
                deleted_counts["objectives"] += 1

        total = sum(deleted_counts.values())
        if total > 0:
            parts = []
            if deleted_counts["interactions"]:
                parts.append(f"{deleted_counts['interactions']} interactions")
            if deleted_counts["reactions"]:
                parts.append(f"{deleted_counts['reactions']} reactions")
            if deleted_counts["objectives"]:
                parts.append(f"{deleted_counts['objectives']} objectives")
            self.report({'INFO'}, f"Deleted {', '.join(parts)}")
        else:
            self.report({'INFO'}, "No orphaned items found")

        return {'FINISHED'}


# ─────────────────────────────────────────────────────────
# Interactions & Chains Panel
# ─────────────────────────────────────────────────────────

class VIEW3D_PT_Exploratory_Studio(bpy.types.Panel):
    bl_label = "Interactions & Chains"
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

        interactions = getattr(scn, "custom_interactions", [])
        reactions = getattr(scn, "reactions", [])

        # Collect what's actually referenced by nodes
        referenced = _collect_referenced_indices()
        ref_inter = referenced["interactions"]
        ref_react = referenced["reactions"]

        # Find invalid links
        invalid_links = _find_invalid_reaction_links(scn)

        if not interactions:
            layout.label(text="No interactions defined.", icon='INFO')
        else:
            # Show valid interactions with their chains
            for i, inter in enumerate(interactions):
                is_orphan = i not in ref_inter

                box = layout.box()
                if is_orphan:
                    box.alert = True

                row = box.row()
                icon = 'ORPHAN_DATA' if is_orphan else 'PLAY'
                row.label(text=inter.name, icon=icon)
                row.label(text=f"({inter.trigger_type})")

                if is_orphan:
                    box.label(text=f"ORPHAN - no node references idx {i}", icon='ERROR')

                # Chained reactions
                links = getattr(inter, "reaction_links", [])
                if links:
                    for link in links:
                        idx = getattr(link, "reaction_index", -1)
                        if 0 <= idx < len(reactions):
                            r = reactions[idx]
                            sub = box.row()
                            sub.label(text="", icon='FORWARD')
                            sub.label(text=r.name)
                            sub.label(text=f"({r.reaction_type})")
                        else:
                            sub = box.row()
                            sub.alert = True
                            sub.label(text="", icon='FORWARD')
                            sub.label(text=f"INVALID LINK → idx {idx}", icon='ERROR')
                else:
                    box.label(text="  (no reactions linked)", icon='BLANK1')

        # Show invalid links summary
        if invalid_links:
            warn_box = layout.box()
            warn_box.alert = True
            warn_box.label(text=f"{len(invalid_links)} INVALID LINK(S)", icon='ERROR')
            for inter_idx, inter_name, bad_idx in invalid_links:
                warn_box.label(text=f"  [{inter_idx}] {inter_name} → reaction {bad_idx}")


# ─────────────────────────────────────────────────────────
# Reactions Library Panel (Read-Only)
# ─────────────────────────────────────────────────────────

class VIEW3D_PT_Exploratory_Reactions(bpy.types.Panel):
    bl_label = "Reactions Library"
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

        reactions = getattr(scn, "reactions", [])
        referenced = _collect_referenced_indices()
        ref_react = referenced["reactions"]

        if not reactions:
            layout.label(text="No reactions defined.", icon='INFO')
            return

        # Count orphans
        orphan_count = sum(1 for i in range(len(reactions)) if i not in ref_react)

        header = f"{len(reactions)} reaction(s)"
        if orphan_count:
            header += f" ({orphan_count} orphaned)"
        layout.label(text=header)

        # Delete orphaned button (only show if there are orphans)
        if orphan_count:
            layout.operator("exploratory.delete_orphaned", icon='TRASH')

        for i, r in enumerate(reactions):
            is_orphan = i not in ref_react
            row = layout.row()
            if is_orphan:
                row.alert = True
            icon = 'ORPHAN_DATA' if is_orphan else 'DOT'
            row.label(text=f"{i}: {r.name}", icon=icon)
            row.label(text=f"({r.reaction_type})")


# ─────────────────────────────────────────────────────────
# Objectives Panel (Read-Only)
# ─────────────────────────────────────────────────────────

class VIEW3D_PT_Objectives(bpy.types.Panel):
    bl_label = "Objectives"
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

        objectives = getattr(scn, "objectives", [])
        referenced = _collect_referenced_indices()
        ref_obj = referenced["objectives"]

        if not objectives:
            layout.label(text="No objectives defined.", icon='INFO')
            return

        # Count orphans
        orphan_count = sum(1 for i in range(len(objectives)) if i not in ref_obj)

        if orphan_count:
            layout.label(text=f"{orphan_count} orphaned objective(s)", icon='ERROR')

        for i, obj in enumerate(objectives):
            is_orphan = i not in ref_obj

            row = layout.row()
            if is_orphan:
                row.alert = True

            # Determine type based on timer_mode
            timer_mode = getattr(obj, "timer_mode", "NONE")
            if timer_mode != "NONE":
                obj_type = "Timer"
                base_icon = 'TIME'
            else:
                obj_type = "Counter"
                base_icon = 'LINENUMBERS_ON'

            icon = 'ORPHAN_DATA' if is_orphan else base_icon
            row.label(text=f"{i}: {obj.name}", icon=icon)
            row.label(text=f"({obj_type})")
