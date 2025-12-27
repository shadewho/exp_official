# Exploratory/Exp_Nodes/trig_react_obj_lists.py
# Read-only visualization of interactions, reactions, counters, and timers.
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
        "counters": set(),
        "timers": set(),
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

            # Counter nodes reference counters
            if getattr(node, "bl_idname", "") == "CounterNodeType":
                idx = getattr(node, "counter_index", -1)
                if idx >= 0:
                    referenced["counters"].add(idx)

            # Timer nodes reference timers
            if getattr(node, "bl_idname", "") == "TimerNodeType":
                idx = getattr(node, "timer_index", -1)
                if idx >= 0:
                    referenced["timers"].add(idx)

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
    """Delete all orphaned interactions, reactions, counters, and timers that are not referenced by any node"""
    bl_idname = "exploratory.delete_orphaned"
    bl_label = "Delete Orphaned"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _in_exploratory_editor(context)

    def execute(self, context):
        scn = context.scene
        referenced = _collect_referenced_indices()

        deleted_counts = {"interactions": 0, "reactions": 0, "counters": 0, "timers": 0}

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

        # Delete orphaned counters (reverse order)
        counters = getattr(scn, "counters", None)
        if counters:
            ref_cnt = referenced["counters"]
            to_delete = [i for i in range(len(counters)) if i not in ref_cnt]
            for idx in reversed(to_delete):
                counters.remove(idx)
                deleted_counts["counters"] += 1

        # Delete orphaned timers (reverse order)
        timers = getattr(scn, "timers", None)
        if timers:
            ref_tmr = referenced["timers"]
            to_delete = [i for i in range(len(timers)) if i not in ref_tmr]
            for idx in reversed(to_delete):
                timers.remove(idx)
                deleted_counts["timers"] += 1

        total = sum(deleted_counts.values())
        if total > 0:
            parts = []
            if deleted_counts["interactions"]:
                parts.append(f"{deleted_counts['interactions']} interactions")
            if deleted_counts["reactions"]:
                parts.append(f"{deleted_counts['reactions']} reactions")
            if deleted_counts["counters"]:
                parts.append(f"{deleted_counts['counters']} counters")
            if deleted_counts["timers"]:
                parts.append(f"{deleted_counts['timers']} timers")
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
# Counters Panel (Read-Only)
# ─────────────────────────────────────────────────────────

class VIEW3D_PT_Counters(bpy.types.Panel):
    bl_label = "Counters"
    bl_idname = "VIEW3D_PT_counters"
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

        counters = getattr(scn, "counters", [])
        referenced = _collect_referenced_indices()
        ref_cnt = referenced["counters"]

        if not counters:
            layout.label(text="No counters defined.", icon='INFO')
            return

        # Count orphans
        orphan_count = sum(1 for i in range(len(counters)) if i not in ref_cnt)

        header = f"{len(counters)} counter(s)"
        if orphan_count:
            header += f" ({orphan_count} orphaned)"
        layout.label(text=header)

        for i, cnt in enumerate(counters):
            is_orphan = i not in ref_cnt

            row = layout.row()
            if is_orphan:
                row.alert = True

            icon = 'ORPHAN_DATA' if is_orphan else 'LINENUMBERS_ON'
            row.label(text=f"{i}: {cnt.name}", icon=icon)
            row.label(text=f"(default: {cnt.default_value})")


# ─────────────────────────────────────────────────────────
# Timers Panel (Read-Only)
# ─────────────────────────────────────────────────────────

class VIEW3D_PT_Timers(bpy.types.Panel):
    bl_label = "Timers"
    bl_idname = "VIEW3D_PT_timers"
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

        timers = getattr(scn, "timers", [])
        referenced = _collect_referenced_indices()
        ref_tmr = referenced["timers"]

        if not timers:
            layout.label(text="No timers defined.", icon='INFO')
            return

        # Count orphans
        orphan_count = sum(1 for i in range(len(timers)) if i not in ref_tmr)

        header = f"{len(timers)} timer(s)"
        if orphan_count:
            header += f" ({orphan_count} orphaned)"
        layout.label(text=header)

        for i, tmr in enumerate(timers):
            is_orphan = i not in ref_tmr

            row = layout.row()
            if is_orphan:
                row.alert = True

            icon = 'ORPHAN_DATA' if is_orphan else 'TIME'
            row.label(text=f"{i}: {tmr.name}", icon=icon)
            row.label(text=f"({tmr.timer_mode})")


# ─────────────────────────────────────────────────────────
# Action Keys Panel (Read-Only)
# ─────────────────────────────────────────────────────────

class VIEW3D_PT_ActionKeys(bpy.types.Panel):
    bl_label = "Action Keys"
    bl_idname = "VIEW3D_PT_action_keys"
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

        action_keys = getattr(scn, "action_keys", [])
        referenced = _collect_referenced_indices()
        ref_ak = referenced["action_keys"]

        if not action_keys:
            layout.label(text="No action keys defined.", icon='INFO')
            return

        # Count orphans
        orphan_count = sum(1 for i in range(len(action_keys)) if i not in ref_ak)

        header = f"{len(action_keys)} action key(s)"
        if orphan_count:
            header += f" ({orphan_count} orphaned)"
        layout.label(text=header)

        for i, ak in enumerate(action_keys):
            is_orphan = i not in ref_ak

            row = layout.row()
            if is_orphan:
                row.alert = True

            icon = 'ORPHAN_DATA' if is_orphan else 'EVENT_A'
            enabled = getattr(ak, "enabled_default", False)
            enabled_txt = "ON" if enabled else "OFF"
            row.label(text=f"{i}: {ak.name}", icon=icon)
            row.label(text=f"(default: {enabled_txt})")


# ─────────────────────────────────────────────────────────
# Trackers Panel (Read-Only)
# ─────────────────────────────────────────────────────────

def _collect_tracker_nodes():
    """Collect all tracker nodes from all Exploratory trees."""
    trackers = []
    tracker_types = {
        "DistanceTrackerNodeType": "Distance",
        "StateTrackerNodeType": "State",
        "ContactTrackerNodeType": "Contact",
        "InputTrackerNodeType": "Input",
        "GameTimeTrackerNodeType": "Game Time",
        "LogicAndNodeType": "AND Gate",
        "LogicOrNodeType": "OR Gate",
        "LogicNotNodeType": "NOT Gate",
    }

    for ng in bpy.data.node_groups:
        if getattr(ng, "bl_idname", "") != EXPL_TREE_ID:
            continue
        for node in ng.nodes:
            bl_id = getattr(node, "bl_idname", "")
            if bl_id in tracker_types:
                trackers.append({
                    "tree": ng.name,
                    "node": node.name,
                    "type": tracker_types[bl_id],
                    "bl_idname": bl_id,
                })
    return trackers


class VIEW3D_PT_Trackers(bpy.types.Panel):
    bl_label = "Trackers"
    bl_idname = "VIEW3D_PT_trackers"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return _in_exploratory_editor(context)

    def draw(self, context):
        layout = self.layout

        trackers = _collect_tracker_nodes()

        if not trackers:
            layout.label(text="No trackers defined.", icon='INFO')
            return

        # Group by type
        by_type = {}
        for t in trackers:
            by_type.setdefault(t["type"], []).append(t)

        layout.label(text=f"{len(trackers)} tracker(s)")

        for tracker_type, items in sorted(by_type.items()):
            box = layout.box()
            box.label(text=f"{tracker_type} ({len(items)})", icon='VIEWZOOM')
            for item in items:
                row = box.row()
                row.label(text=f"  {item['node']}")
                row.label(text=f"[{item['tree']}]")


# ─────────────────────────────────────────────────────────
# Utilities Panel (Read-Only)
# ─────────────────────────────────────────────────────────

def _collect_utility_nodes():
    """Collect all utility nodes (Delay, Data nodes) from all Exploratory trees."""
    utilities = []
    utility_types = {
        "UtilityDelayNodeType": "Delay",
        "FloatDataNodeType": "Float",
        "IntDataNodeType": "Integer",
        "BoolDataNodeType": "Boolean",
        "ObjectDataNodeType": "Object",
        "CollectionDataNodeType": "Collection",
        "ActionDataNodeType": "Action",
        "FloatVectorDataNodeType": "Float Vector",
    }

    for ng in bpy.data.node_groups:
        if getattr(ng, "bl_idname", "") != EXPL_TREE_ID:
            continue
        for node in ng.nodes:
            bl_id = getattr(node, "bl_idname", "")
            if bl_id in utility_types:
                utilities.append({
                    "tree": ng.name,
                    "node": node.name,
                    "type": utility_types[bl_id],
                    "bl_idname": bl_id,
                })
    return utilities


class VIEW3D_PT_Utilities(bpy.types.Panel):
    bl_label = "Utilities"
    bl_idname = "VIEW3D_PT_utilities"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return _in_exploratory_editor(context)

    def draw(self, context):
        layout = self.layout

        utilities = _collect_utility_nodes()

        if not utilities:
            layout.label(text="No utility nodes defined.", icon='INFO')
            return

        # Group by type
        by_type = {}
        for u in utilities:
            by_type.setdefault(u["type"], []).append(u)

        layout.label(text=f"{len(utilities)} utility node(s)")

        for util_type, items in sorted(by_type.items()):
            box = layout.box()
            box.label(text=f"{util_type} ({len(items)})", icon='MODIFIER')
            for item in items:
                row = box.row()
                row.label(text=f"  {item['node']}")
                row.label(text=f"[{item['tree']}]")
