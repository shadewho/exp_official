# File: exp_interactions.py

import bpy
import time
from mathutils import Vector
from ..developer.dev_debug_gate import should_print_debug
from ..reactions.exp_reactions import ( execute_property_reaction,
                             execute_char_action_reaction, execute_custom_ui_text_reaction,
                             execute_objective_counter_reaction,
                             execute_objective_timer_reaction, execute_sound_reaction,
)
from ..reactions.exp_transforms import execute_transform_reaction
from ..reactions.exp_projectiles import execute_projectile_reaction, execute_hitscan_reaction
from ..animations.exp_custom_animations import execute_custom_action_reaction
from ..props_and_utils.exp_time import get_game_time
from ..systems.exp_objectives import update_all_objective_timers
from ..reactions.exp_mobility_and_game_reactions import (
    execute_mobility_reaction,
    execute_mesh_visibility_reaction,
    execute_reset_game_reaction
)

from ..reactions.exp_crosshairs import execute_crosshairs_reaction
from ..reactions import exp_custom_ui
from ..reactions.exp_action_keys import (
    is_enabled as is_action_key_enabled,
    execute_action_key_reaction,
    seed_defaults_from_scene,
)
from ..reactions.exp_parenting import execute_parenting_reaction
from ..reactions.exp_tracking import execute_tracking_reaction

# Global list to hold pending trigger tasks.
_pending_reaction_batches = []
_pending_trigger_tasks = []


def apply_interaction_check_result(engine_result):
    """
    Apply worker result for interaction checks (diagnostic only for Sprint 1.2).
    Called from game loop when INTERACTION_CHECK_BATCH job completes.

    For now, this just logs worker execution to prove offload works.
    Future: Could optimize to skip trigger handlers for non-triggered interactions.
    """
    debug_offload = should_print_debug("interactions")

    if not debug_offload:
        return  # Silent if debug disabled

    result_data = engine_result.result
    triggered_indices = result_data.get("triggered_indices", [])
    total_count = result_data.get("count", 0)
    calc_time_us = result_data.get("calc_time_us", 0.0)
    worker_time_ms = engine_result.processing_time * 1000.0

    print(f"[InteractionOffload] Workers checked {total_count} interactions, {len(triggered_indices)} triggered (worker: {worker_time_ms:.3f}ms, calc: {calc_time_us:.1f}µs)")
def _execute_reaction_now(r):
    """
    Execute a single reaction immediately (no scheduling).
    'DELAY' is handled at the sequencing level and is ignored here.
    """
    t = getattr(r, "reaction_type", "")

    if t == "DELAY":
        return  # sequencing-only marker

    # --- map reaction types to executors (same mapping you already use) ---
    if t == "CUSTOM_ACTION":
        execute_custom_action_reaction(r)
    elif t == "OBJECTIVE_COUNTER":
        execute_objective_counter_reaction(r)
    elif t == "OBJECTIVE_TIMER":
        execute_objective_timer_reaction(r)
    elif t == "CHAR_ACTION":
        execute_char_action_reaction(r)
    elif t == "SOUND":
        execute_sound_reaction(r)
    elif t == "TRANSFORM":
        execute_transform_reaction(r)
    elif t == "PROPERTY":
        execute_property_reaction(r)
    elif t == "CUSTOM_UI_TEXT":
        execute_custom_ui_text_reaction(r)
    elif t == "ENABLE_CROSSHAIRS":
        execute_crosshairs_reaction(r)
    elif t == "HITSCAN":
        execute_hitscan_reaction(r)
    elif t == "PROJECTILE":
        execute_projectile_reaction(r)
    elif t == "MOBILITY":
        execute_mobility_reaction(r)
    elif t == "MESH_VISIBILITY":
        execute_mesh_visibility_reaction(r)
    elif t == "RESET_GAME":
        execute_reset_game_reaction(r)
    elif t == "ACTION_KEYS":
        execute_action_key_reaction(r)
    elif r.reaction_type == "PARENTING":
        execute_parenting_reaction(r)
    elif t == "TRACK_TO":
        execute_tracking_reaction(r)


def _execute_reaction_list_now(reactions):
    """Execute a list of reactions immediately (skips any DELAY markers)."""
    for r in (reactions or []):
        if getattr(r, "reaction_type", "") != "DELAY":
            _execute_reaction_now(r)


def schedule_reaction_batch(reactions, fire_time):
    """Queue a batch (list of ReactionDefinition) to run at fire_time."""
    if not reactions:
        return
    _pending_reaction_batches.append({
        "reactions": list(reactions),
        "fire_time": float(fire_time)
    })


def process_pending_reaction_batches():
    """Run any queued reaction batches whose time has arrived."""
    now = get_game_time()
    for task in _pending_reaction_batches[:]:
        if now >= task["fire_time"]:
            _execute_reaction_list_now(task["reactions"])
            _pending_reaction_batches.remove(task)



# -------------------------------------------------------------------
# Find reactions linked to an interaction and run them
# -------------------------------------------------------------------
def _get_linked_reactions(inter, scene=None):
    """
    Resolve the interaction's link list to actual ReactionDefinition objects.
    Invalid indices are skipped.
    """
    if scene is None:
        scene = bpy.context.scene
    out = []
    links = getattr(inter, "reaction_links", [])
    for link in links:
        i = getattr(link, "reaction_index", -1)
        if 0 <= i < len(scene.reactions):
            out.append(scene.reactions[i])
    return out
def _is_reaction_already_linked(inter, reaction_index: int) -> bool:
    """True if this Interaction already links the given global reaction index."""
    for link in getattr(inter, "reaction_links", []):
        if getattr(link, "reaction_index", -1) == reaction_index:
            return True
    return False

def _fire_interaction(inter, current_time):
    """
    Runs the linked reactions for 'inter', honoring trigger_delay scheduling.
    """
    if inter.trigger_delay > 0.0:
        schedule_trigger(inter, current_time + inter.trigger_delay)
    else:
        run_reactions(_get_linked_reactions(inter))
    inter.has_fired = True
    inter.last_trigger_time = current_time


# -------------------------------------------------------------------
# Dynamic trigger_mode items based on trigger_type
# -------------------------------------------------------------------
def trigger_mode_items(self, context):
    ttype = getattr(self, "trigger_type", "")
    if ttype == "ON_GAME_START":
        # Game-start triggers should be single-shot only
        return [("ONE_SHOT", "One-Shot", "Fire once at game start/reset")]
    
    # always include One‑Shot and Cooldown
    items = [
        ("ONE_SHOT", "One-Shot",    "Fire once only, never again"),
        ("COOLDOWN", "Cooldown",    "Allow repeated fires but only after cooldown"),
    ]
    # but only proximity/collision get the Enter‑Only mode
    if self.trigger_type in {"PROXIMITY", "COLLISION"}:
        items.insert(1, ("ENTER_ONLY", "Enter Only",
                         "Fire on each new enter/press; resets on release/exit"))
    return items

###############################################################################
# 2) Operators for Adding/Removing Interactions
###############################################################################
class EXPLORATORY_OT_AddInteraction(bpy.types.Operator):
    """Add a new InteractionDefinition to the scene custom_interactions."""
    bl_idname = "exploratory.add_interaction"
    bl_label = "Add Interaction"

    def execute(self, context):
        scene = context.scene
        new_item = scene.custom_interactions.add()
        new_item.name = f"Interaction_{len(scene.custom_interactions)}"
        scene.custom_interactions_index = len(scene.custom_interactions) - 1
        return {'FINISHED'}


class EXPLORATORY_OT_RemoveInteraction(bpy.types.Operator):
    """Remove an InteractionDefinition from scene.custom_interactions by index."""
    bl_idname = "exploratory.remove_interaction"
    bl_label = "Remove Interaction"

    index: bpy.props.IntProperty()

    def execute(self, context):
        scene = context.scene
        if 0 <= self.index < len(scene.custom_interactions):
            scene.custom_interactions.remove(self.index)
            scene.custom_interactions_index = max(
                0, 
                min(self.index, len(scene.custom_interactions) - 1)
            )
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────
# Utility: deep-copy any PropertyGroup (handles nested pointers & collections)
# ─────────────────────────────────────────────────────────
def _deep_copy_pg(src, dst, skip: set[str] = frozenset()):
    """
    Copies writable RNA properties from src -> dst.
    - Pointers to ID datablocks (Object, Action, Sound, etc.) are assigned directly.
    - Pointers to PropertyGroups are copied recursively.
    - Collections are recreated item-by-item (recursive).
    """
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
                # Datablock pointer → assign; PropertyGroup pointer → recurse
                if isinstance(value, _ID) or value is None:
                    setattr(dst, ident, value)
                else:
                    sub_dst = getattr(dst, ident)
                    _deep_copy_pg(value, sub_dst)
            elif prop.type == 'COLLECTION':
                dst_coll = getattr(dst, ident)
                # Clear destination collection
                try:
                    dst_coll.clear()
                except AttributeError:
                    while len(dst_coll):
                        dst_coll.remove(len(dst_coll) - 1)
                # Rebuild items
                for src_item in value:
                    dst_item = dst_coll.add()
                    _deep_copy_pg(src_item, dst_item)
            else:
                setattr(dst, ident, value)
        except Exception:
            # Be robust; skip anything Blender won't let us set
            pass


# ─────────────────────────────────────────────────────────
# Duplicate Interaction operator
# ─────────────────────────────────────────────────────────
class EXPLORATORY_OT_DuplicateInteraction(bpy.types.Operator):
    """
    Duplicate the currently selected Interaction, copying all trigger/settings,
    but **NOT** carrying over any linked reactions.
    """
    bl_idname = "exploratory.duplicate_interaction"
    bl_label = "Duplicate Interaction"
    bl_options = {'REGISTER', 'UNDO'}

    # Optional: allow duplicating a specific index; defaults to current selection
    index: bpy.props.IntProperty(name="Index", default=-1)

    def execute(self, context):
        scn = context.scene
        src_idx = self.index if self.index >= 0 else scn.custom_interactions_index
        if not (0 <= src_idx < len(scn.custom_interactions)):
            self.report({'WARNING'}, "No valid Interaction selected.")
            return {'CANCELLED'}

        src = scn.custom_interactions[src_idx]
        dst = scn.custom_interactions.add()

        # Copy everything EXCEPT reaction_links
        _deep_copy_pg(src, dst, skip={"reaction_links", "reaction_links_index"})

        # Ensure the duplicate has zero links (explicit)
        try:
            dst.reaction_links.clear()
        except AttributeError:
            while len(dst.reaction_links):
                dst.reaction_links.remove(len(dst.reaction_links) - 1)
        dst.reaction_links_index = 0

        # Keep trigger_type/mode consistent (in case copy order matters)
        try:
            dst.trigger_type = src.trigger_type
            dst.trigger_mode = src.trigger_mode
        except Exception:
            pass

        # Friendly name
        try:
            base = getattr(src, "name", "Interaction")
            dst.name = f"{base} (Copy)"
        except Exception:
            pass

        scn.custom_interactions_index = len(scn.custom_interactions) - 1
        return {'FINISHED'}


###############################################################################
# 3) Operators for Adding/Removing Reactions within an Interaction
###############################################################################
class EXPLORATORY_OT_AddReactionToInteraction(bpy.types.Operator):
    """Link the currently selected global Reaction (scene.reactions_index) to the selected Interaction."""
    bl_idname = "exploratory.add_reaction_to_interaction"
    bl_label = "Link Selected Reaction"

    def execute(self, context):
        scene = context.scene
        i_idx = scene.custom_interactions_index
        if not (0 <= i_idx < len(scene.custom_interactions)):
            self.report({'WARNING'}, "No valid Interaction selected.")
            return {'CANCELLED'}

        r_idx = scene.reactions_index
        if not (0 <= r_idx < len(scene.reactions)):
            self.report({'WARNING'}, "No valid Reaction selected in the Reactions panel.")
            return {'CANCELLED'}

        inter = scene.custom_interactions[i_idx]

        # Enforce uniqueness
        if _is_reaction_already_linked(inter, r_idx):
            # Focus the existing link instead of adding a duplicate
            for i, l in enumerate(inter.reaction_links):
                if l.reaction_index == r_idx:
                    inter.reaction_links_index = i
                    break
            self.report({'INFO'}, "Reaction already linked to this interaction.")
            return {'CANCELLED'}

        link = inter.reaction_links.add()
        link.reaction_index = r_idx
        inter.reaction_links_index = len(inter.reaction_links) - 1
        return {'FINISHED'}


class EXPLORATORY_OT_RemoveReactionFromInteraction(bpy.types.Operator):
    """Unlink a Reaction from the Interaction by the link's index."""
    bl_idname = "exploratory.remove_reaction_from_interaction"
    bl_label = "Unlink Reaction"

    index: bpy.props.IntProperty()

    def execute(self, context):
        scene = context.scene
        i_idx = scene.custom_interactions_index
        if not (0 <= i_idx < len(scene.custom_interactions)):
            return {'CANCELLED'}

        inter = scene.custom_interactions[i_idx]
        if 0 <= self.index < len(inter.reaction_links):
            inter.reaction_links.remove(self.index)
            inter.reaction_links_index = max(0, min(self.index, len(inter.reaction_links) - 1))
        return {'FINISHED'}
    
class EXPLORATORY_OT_CreateReactionAndLink(bpy.types.Operator):
    """Create a new Reaction in the global library and link it to the current Interaction."""
    bl_idname = "exploratory.create_reaction_and_link"
    bl_label = "New Reaction + Link"

    def execute(self, context):
        scene = context.scene

        # Create a new global reaction
        new_r = scene.reactions.add()
        new_r.name = f"Reaction_{len(scene.reactions)}"
        scene.reactions_index = len(scene.reactions) - 1
        r_idx = scene.reactions_index

        # Link it to the current interaction if any (guard against dupes)
        i_idx = scene.custom_interactions_index
        if 0 <= i_idx < len(scene.custom_interactions):
            inter = scene.custom_interactions[i_idx]
            if not _is_reaction_already_linked(inter, r_idx):
                link = inter.reaction_links.add()
                link.reaction_index = r_idx
                inter.reaction_links_index = len(inter.reaction_links) - 1
            else:
                # Focus the existing link if it already exists
                for i, l in enumerate(inter.reaction_links):
                    if l.reaction_index == r_idx:
                        inter.reaction_links_index = i
                        break

        return {'FINISHED'}


###############################################################################
# 4) The Main check_interactions() Logic
###############################################################################
def _submit_interaction_worker_jobs(context):
    """
    Submit PROXIMITY and COLLISION checks to workers.
    Returns list of interaction data for result matching.
    """
    from ..modal.exp_modal import get_active_modal_operator

    scene = context.scene
    modal_op = get_active_modal_operator()
    engine = getattr(modal_op, "engine", None) if modal_op else None

    if not engine or not engine.is_alive():
        return None  # No workers available

    debug_offload = should_print_debug("interactions")

    # Snapshot interactions that need worker checking
    interactions_data = []
    interaction_map = []  # Maps worker indices back to scene interactions

    submit_start = time.perf_counter() if debug_offload else 0.0

    for inter in scene.custom_interactions:
        t = inter.trigger_type

        if t == "PROXIMITY":
            # pick A: either character or chosen object
            if inter.use_character:
                obj_a = scene.target_armature
            else:
                obj_a = inter.proximity_object_a
            obj_b = inter.proximity_object_b

            if obj_a and obj_b:
                inter_data = {
                    "type": "PROXIMITY",
                    "obj_a_pos": (obj_a.location.x, obj_a.location.y, obj_a.location.z),
                    "obj_b_pos": (obj_b.location.x, obj_b.location.y, obj_b.location.z),
                    "threshold": float(inter.proximity_distance)
                }
                interactions_data.append(inter_data)
                interaction_map.append(inter)

        elif t == "COLLISION":
            # pick A: either character or chosen object
            if inter.use_character:
                obj_a = scene.target_armature
            else:
                obj_a = inter.collision_object_a
            obj_b = inter.collision_object_b

            if obj_a and obj_b:
                # Calculate AABBs (same as bounding_sphere_collision)
                def get_aabb(obj):
                    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
                    return (
                        min(pt.x for pt in corners), max(pt.x for pt in corners),
                        min(pt.y for pt in corners), max(pt.y for pt in corners),
                        min(pt.z for pt in corners), max(pt.z for pt in corners)
                    )

                inter_data = {
                    "type": "COLLISION",
                    "aabb_a": get_aabb(obj_a),
                    "aabb_b": get_aabb(obj_b),
                    "margin": float(inter.collision_margin)
                }
                interactions_data.append(inter_data)
                interaction_map.append(inter)

    # Submit if we have interactions to check
    if interactions_data:
        player_loc = scene.target_armature.location if scene.target_armature else (0, 0, 0)

        job_data = {
            "interactions": interactions_data,
            "player_position": (player_loc.x, player_loc.y, player_loc.z)
        }

        job_id = engine.submit_job("INTERACTION_CHECK_BATCH", job_data)

        submit_end = time.perf_counter() if debug_offload else 0.0

        if debug_offload:
            submit_time_ms = (submit_end - submit_start) * 1000.0
            print(f"[InteractionOffload] Submitted job {job_id} with {len(interactions_data)} interactions (submit: {submit_time_ms:.3f}ms)")

        return interaction_map

    return None


def check_interactions(context):
    scene = context.scene
    current_time = get_game_time()

    # ─── CLAMP ANY INVALID trigger_mode VALUES ───
    for inter in scene.custom_interactions:
        allowed = [item[0] for item in trigger_mode_items(inter, context)]
        if inter.trigger_mode not in allowed:
            inter.trigger_mode = allowed[0]

    pressed_this_frame = was_interact_pressed()
    pressed_action   = was_action_pressed()

    update_all_objective_timers(context.scene)

    # ========== WORKER OFFLOAD: PROXIMITY & COLLISION CHECKS ==========
    # Submit proximity/collision checks to workers
    # Note: We still run handlers on main thread, just pre-filtered by worker results
    interaction_map = _submit_interaction_worker_jobs(context)
    # Worker results will be applied via apply_interaction_check_result() in game loop

    for inter in scene.custom_interactions:
        t = inter.trigger_type

        if t == "PROXIMITY":
            handle_proximity_trigger(inter, current_time)
        elif t == "COLLISION":
            handle_collision_trigger(inter, current_time)
        elif t == "INTERACT":
            handle_interact_trigger(inter, current_time, pressed_this_frame, context)
        elif t == "ACTION":  #
            handle_action_trigger(inter, current_time, pressed_action)
        elif t == "OBJECTIVE_UPDATE":
            handle_objective_update_trigger(inter, current_time)
        elif t == "TIMER_COMPLETE":
            handle_timer_complete_trigger(inter)
        elif t == "ON_GAME_START":
            handle_on_game_start_trigger(inter, current_time)
        elif t == "EXTERNAL":
            handle_external_trigger(inter, current_time)



    # Now process any pending triggers that have matured.
    process_pending_triggers()

    # Clear edge-like key flags at frame end
    global _global_interact_pressed, _global_action_pressed
    _global_interact_pressed = False
    _global_action_pressed   = False


###############################################################################
# 4)trigger helpers
###############################################################################
def schedule_trigger(interaction, fire_time):
    """
    Schedule the given interaction's reactions to fire at fire_time (game time).
    """
    _pending_trigger_tasks.append({
        "interaction": interaction,
        "fire_time": fire_time
    })

def process_pending_triggers():
    """
    Checks pending trigger tasks and fires their reactions when due.
    Also processes any matured reaction batches (Utilities → Delay).
    """
    current_time = get_game_time()
    for task in _pending_trigger_tasks[:]:
        if current_time >= task["fire_time"]:
            inter = task["interaction"]
            run_reactions(_get_linked_reactions(inter))
            _pending_trigger_tasks.remove(task)

    # New: also run matured reaction batches each frame
    process_pending_reaction_batches()


# -------------------------------------------------------------------
# 4.1) Proximity Trigger:
# -------------------------------------------------------------------
def handle_proximity_trigger(inter, current_time):
    scene = bpy.context.scene

    # pick A: either the user‑chosen object or the character armature
    if inter.use_character:
        obj_a = scene.target_armature
    else:
        obj_a = inter.proximity_object_a
    obj_b = inter.proximity_object_b
    dist_thresh = inter.proximity_distance

    if not obj_a or not obj_b:
        return

    dist = (obj_a.location - obj_b.location).length
    inside_now = (dist <= dist_thresh)
    was_in_zone = inter.is_in_zone
    inter.is_in_zone = inside_now

    # 1) ONE_SHOT: never re‑fire
    if inter.trigger_mode == "ONE_SHOT" and inter.has_fired:
        return

    # 2) ENTER
    if not was_in_zone and inside_now:
        def fire():
            if inter.trigger_delay > 0.0:
                schedule_trigger(inter, current_time + inter.trigger_delay)
            else:
                run_reactions(_get_linked_reactions(inter))
            inter.has_fired = True
            inter.last_trigger_time = current_time

        if inter.trigger_mode in {"ONE_SHOT", "ENTER_ONLY"}:
            fire()
        elif inter.trigger_mode == "COOLDOWN":
            time_since = current_time - inter.last_trigger_time
            if (not inter.has_fired) or (time_since >= inter.trigger_cooldown):
                fire()

    # 3) LEAVE → only ENTER_ONLY resets here
    elif was_in_zone and not inside_now:
        if inter.trigger_mode == "ENTER_ONLY":
            inter.has_fired = False

    # 4) STILL INSIDE: allow COOLDOWN to retrigger when timer expires
    else:
        if inside_now and inter.trigger_mode == "COOLDOWN" and inter.has_fired:
            time_since = current_time - inter.last_trigger_time
            if time_since >= inter.trigger_cooldown:
                if inter.trigger_delay > 0.0:
                    schedule_trigger(inter, current_time + inter.trigger_delay)
                else:
                    run_reactions(_get_linked_reactions(inter))
                inter.has_fired = True
                inter.last_trigger_time = current_time



# ─────────────────────────────────────────────────────────
# External Trigger (boolean input)
# ─────────────────────────────────────────────────────────
def handle_external_trigger(inter, current_time):
    """
    External boolean-driven trigger.
      • ONE_SHOT  → first True fires, then never again
      • COOLDOWN  → while True, re-fire whenever cooldown has elapsed
    Delay is honored through the standard _fire_interaction path.
    """
    gate = bool(getattr(inter, "external_signal", False))

    # If not True, nothing to do (we do not require an edge; COOLDOWN can repeat while held True)
    if not gate:
        return

    if can_fire_trigger(inter, current_time):
        _fire_interaction(inter, current_time)

# ─────────────────────────────────────────────────────────
# Game Start trigger (fires once per game start/reset)
# ─────────────────────────────────────────────────────────
def handle_on_game_start_trigger(inter, current_time):
    """
    Fire once after gameplay begins (or after a reset). Subsequent frames do nothing
    until reset_all_interactions() clears 'has_fired'.
    Honors the per-interaction trigger_delay just like other triggers.
    """
    if inter.has_fired:
        return

    if getattr(inter, "trigger_delay", 0.0) > 0.0:
        schedule_trigger(inter, current_time + inter.trigger_delay)
    else:
        run_reactions(_get_linked_reactions(inter))

    inter.has_fired = True
    inter.last_trigger_time = current_time


# -------------------------------------------------------------------
# 4.2) Collision Trigger: ENTER/LEAVE (updated)
# -------------------------------------------------------------------
def handle_collision_trigger(inter, current_time):
    scene = bpy.context.scene

    # pick A: either the user‑chosen object or the character armature
    if inter.use_character:
        obj_a = scene.target_armature
    else:
        obj_a = inter.collision_object_a
    obj_b = inter.collision_object_b

    if not obj_a or not obj_b:
        return

    margin = inter.collision_margin
    colliding_now = bounding_sphere_collision(obj_a, obj_b, margin=margin)

    # ENTER
    if (not inter.is_in_zone) and colliding_now:
        inter.is_in_zone = True
        if can_fire_trigger(inter, current_time):
            if inter.trigger_delay > 0.0:
                schedule_trigger(inter, current_time + inter.trigger_delay)
            else:
                run_reactions(_get_linked_reactions(inter))
            inter.has_fired = True
            inter.last_trigger_time = current_time

    # LEAVE
    elif inter.is_in_zone and (not colliding_now):
        inter.is_in_zone = False
        reset_interaction_if_needed(inter)

    # STILL COLLIDING
    elif colliding_now and inter.is_in_zone:
        if inter.trigger_mode == "COOLDOWN":
            if can_fire_trigger(inter, current_time):
                if inter.trigger_delay > 0.0:
                    schedule_trigger(inter, current_time + inter.trigger_delay)
                else:
                    run_reactions(_get_linked_reactions(inter))
                inter.has_fired = True
                inter.last_trigger_time = current_time

###############################################################################
# reset interactions
###############################################################################
def reset_all_interactions(scene):

    seed_defaults_from_scene(scene)

    # Clear any queued triggers and delayed reaction batches
    global _pending_trigger_tasks, _pending_reaction_batches
    try:
        _pending_trigger_tasks.clear()
    except Exception:
        _pending_trigger_tasks = []
    try:
        _pending_reaction_batches.clear()
    except Exception:
        _pending_reaction_batches = []

    now = get_game_time()
    for inter in scene.custom_interactions:
        inter.has_fired = False
        inter.last_trigger_time = now
        inter.is_in_zone = False

        if inter.trigger_type == "OBJECTIVE_UPDATE":
            if inter.objective_index.isdigit():
                idx = int(inter.objective_index)
                if 0 <= idx < len(scene.objectives):
                    inter.last_known_count = scene.objectives[idx].current_value
                else:
                    inter.last_known_count = -1
            else:
                inter.last_known_count = -1


###############################################################################
# Action Key Trigger
###############################################################################
def handle_action_trigger(inter, current_time, pressed_now):
    """
    Action key trigger gated by a named Action Key string.
      • ONE_SHOT   → fire once and never again
      • COOLDOWN   → fire if cooldown elapsed since last fire
    Fires only if:
      1) An Action Key name is set on this interaction,
      2) That name exists in Scene.action_keys *right now*, and
      3) That name is currently enabled via the Action Keys reaction.
    """
    key_id = (getattr(inter, "action_key_id", "") or "").strip()
    if not key_id:
        return

    # Hard gate: if the key no longer exists in the Scene registry, ignore.
    scn = bpy.context.scene
    exists = False
    try:
        for it in getattr(scn, "action_keys", []):
            if getattr(it, "name", "") == key_id:
                exists = True
                break
    except Exception:
        exists = False
    if not exists:
        return

    # Also require enabled flag
    if not is_action_key_enabled(key_id):
        return

    # One-shot already fired?
    if inter.trigger_mode == "ONE_SHOT" and inter.has_fired:
        return

    if not pressed_now:
        if inter.trigger_mode != "ONE_SHOT" and inter.has_fired:
            if inter.trigger_mode == 'COOLDOWN':
                time_since = current_time - inter.last_trigger_time
                if time_since >= inter.trigger_cooldown:
                    inter.has_fired = False
            else:
                inter.has_fired = False
        return

    if can_fire_trigger(inter, current_time):
        if inter.trigger_delay > 0.0:
            schedule_trigger(inter, current_time + inter.trigger_delay)
        else:
            run_reactions(_get_linked_reactions(inter))
        inter.has_fired = True
        inter.last_trigger_time = current_time


###############################################################################
# 4.3) Interact Trigger
###############################################################################
def handle_interact_trigger(inter, current_time, pressed_now, context):
    # If it's one_shot and we've already fired => do nothing
    if inter.trigger_mode == "ONE_SHOT" and inter.has_fired:
        return

    # Find the player's character object
    char_obj = context.scene.target_armature
    if not char_obj:
        return

    # Check if we should display the interact prompt
    if inter.interact_object:
        distance = (char_obj.location - inter.interact_object.location).length
        if distance <= inter.interact_distance:
            prefs = context.preferences.addons["Exploratory"].preferences
            interact_key = prefs.key_interact
            exp_custom_ui.add_text_reaction(
                text_str=f"Press [{interact_key}] to interact",
                anchor='BOTTOM_CENTER',
                margin_x=0.5,
                margin_y=16,
                scale=3,
                end_time=get_game_time() + 0.1,
                color=(1, 1, 1, 1)
            )
    # If user did NOT press interact key => reset logic.
    if not pressed_now:
        if inter.trigger_mode != "ONE_SHOT" and inter.has_fired:
            if inter.trigger_mode == 'COOLDOWN':
                time_since = current_time - inter.last_trigger_time
                if time_since >= inter.trigger_cooldown:
                    inter.has_fired = False
            else:
                inter.has_fired = False
        return

    # The key was pressed this frame => check distance again.
    if inter.interact_object:
        dist = (char_obj.location - inter.interact_object.location).length
        if dist > inter.interact_distance:
            return  # Too far; do nothing

    if can_fire_trigger(inter, current_time):
        if inter.trigger_delay > 0.0:
            schedule_trigger(inter, current_time + inter.trigger_delay)
        else:
            run_reactions(_get_linked_reactions(inter))
        inter.has_fired = True
        inter.last_trigger_time = current_time


###############################################################################
# 4.4) Trigger Firing / Reset Helpers
###############################################################################
def can_fire_trigger(inter, current_time):
    # If One-Shot and we already fired, disallow forever.
    if inter.trigger_mode == "ONE_SHOT" and inter.has_fired:
        return False

    # If COOLDOWN and we already fired, check time
    if inter.trigger_mode == "COOLDOWN" and inter.has_fired:
        time_since = current_time - inter.last_trigger_time
        if time_since < inter.trigger_cooldown:
            return False

    # If ENTER_ONLY => we only require that has_fired be False at press time
    # but that’s handled by resetting has_fired on release/exit. So no extra check needed.
    return True




def reset_interaction_if_needed(inter):
    """
    Only ENTER_ONLY should clear has_fired on exit.
    COOLDOWN holds has_fired=True until cooldown expires.
    ONE_SHOT never resets.
    """
    if inter.trigger_mode == "ENTER_ONLY":
        inter.has_fired = False


###############################################################################
# 5) Reaction Execution
###############################################################################
# In exp_interactions.py (inside run_reactions)

def run_reactions(reactions):
    """
    Execute reactions honoring Utilities → Delay nodes in the sequence.
    Any 'DELAY' item partitions the list; segments after it are queued by
    the cumulative delay time.

    Assumes the execution order is the Interaction's link order.
    """
    if not reactions:
        return

    now = get_game_time()
    fire_time = now
    segment = []

    for r in reactions:
        if getattr(r, "reaction_type", "") == "DELAY":
            # flush current segment at current fire_time
            if segment:
                schedule_reaction_batch(segment, fire_time)
                segment = []
            # accumulate delay
            d = float(getattr(r, "utility_delay_seconds", 0.0) or 0.0)
            if d > 0.0:
                fire_time += d
        else:
            segment.append(r)

    # flush trailing segment
    if segment:
        schedule_reaction_batch(segment, fire_time)
        



###############################################################################
# 6) Collision Helper
###############################################################################

def bounding_sphere_collision(obj_a, obj_b, margin=0.0):
    """
    Actually uses Axis-Aligned Bounding Boxes (AABB).
    This version lets you specify a 'margin' so we count collisions
    if they're within 'margin' distance of touching.
    """

    if not obj_a or not obj_b:
        return False

    # Helper to get the min/max of an object's world AABB
    def get_world_aabb(obj):
        corners = []
        for corner_local in obj.bound_box:
            corner_world = obj.matrix_world @ Vector(corner_local)
            corners.append(corner_world)

        min_x = min(pt.x for pt in corners)
        max_x = max(pt.x for pt in corners)
        min_y = min(pt.y for pt in corners)
        max_y = max(pt.y for pt in corners)
        min_z = min(pt.z for pt in corners)
        max_z = max(pt.z for pt in corners)
        return (min_x, max_x, min_y, max_y, min_z, max_z)

    # Get each object's AABB
    A_minx, A_maxx, A_miny, A_maxy, A_minz, A_maxz = get_world_aabb(obj_a)
    B_minx, B_maxx, B_miny, B_maxy, B_minz, B_maxz = get_world_aabb(obj_b)

    # Inflate each AABB by margin on all sides
    A_minx -= margin
    A_maxx += margin
    A_miny -= margin
    A_maxy += margin
    A_minz -= margin
    A_maxz += margin

    B_minx -= margin
    B_maxx += margin
    B_miny -= margin
    B_maxy += margin
    B_minz -= margin
    B_maxz += margin

    # Check overlap in X, Y, and Z
    overlap_x = (A_minx <= B_maxx) and (A_maxx >= B_minx)
    overlap_y = (A_miny <= B_maxy) and (A_maxy >= B_miny)
    overlap_z = (A_minz <= B_maxz) and (A_maxz >= B_minz)

    return (overlap_x and overlap_y and overlap_z)


def approximate_bounding_sphere_radius(obj):
    """
    Approximate bounding-sphere radius from bounding-box diagonal & local scale.
    """
    local_min = Vector(obj.bound_box[0])
    local_max = Vector(obj.bound_box[0])

    for corner in obj.bound_box:
        local_min.x = min(local_min.x, corner[0])
        local_min.y = min(local_min.y, corner[1])
        local_min.z = min(local_min.z, corner[2])

        local_max.x = max(local_max.x, corner[0])
        local_max.y = max(local_max.y, corner[1])
        local_max.z = max(local_max.z, corner[2])

    bb_size = local_max - local_min
    local_radius = 0.5 * bb_size.length

    scale_factors = obj.scale
    max_scale = max(abs(scale_factors.x), abs(scale_factors.y), abs(scale_factors.z))
    return local_radius * max_scale


###############################################################################
# 7) Interact Helper
###############################################################################

_global_interact_pressed = False  # <--- global placeholders
_global_action_pressed   = False
def set_interact_pressed(value: bool):
    """
    Called by the modal operator to tell us the Interact key was pressed or released.
    """
    global _global_interact_pressed
    _global_interact_pressed = value

def was_interact_pressed():

    """
    Return True once when the Interact key is pressed. 
    Resets to False so it doesn't continuously fire if the user holds the key.
    """
    global _global_interact_pressed
    if _global_interact_pressed:
        return True
    return False

def set_action_pressed(value: bool):
    """
    Called by the modal operator to tell us the Action key was pressed or released.
    """
    global _global_action_pressed
    _global_action_pressed = bool(value)

def was_action_pressed():
    """
    Edge-like usage: check once per frame in check_interactions();
    the flag is cleared at the end of check_interactions().
    """
    return bool(_global_action_pressed)


###############################################################################
# 8) Objective Trigger Helper
###############################################################################

def handle_objective_update_trigger(inter, current_time):
    scene = bpy.context.scene

    # 1) Validate the objective index
    if not inter.objective_index.isdigit():
        return
    idx = int(inter.objective_index)
    if idx < 0 or idx >= len(scene.objectives):
        return

    objv    = scene.objectives[idx]
    old_val = inter.last_known_count
    new_val = objv.current_value

    # 2) If ONE_SHOT and already fired, bail out
    if inter.trigger_mode == "ONE_SHOT" and inter.has_fired:
        inter.last_known_count = new_val
        return

    # 3) Decide whether the condition is met right now
    should_fire = False
    cond_value  = inter.objective_condition_value

    if inter.objective_condition == "CHANGED":
        should_fire = (new_val != old_val)

    elif inter.objective_condition == "INCREASED":
        should_fire = (new_val > old_val)

    elif inter.objective_condition == "DECREASED":
        should_fire = (new_val < old_val and old_val >= 0)

    elif inter.objective_condition == "EQUALS":
        should_fire = (new_val == cond_value)

    elif inter.objective_condition == "AT_LEAST":
        should_fire = (new_val >= cond_value)

    # 4) If it’s met, fire—honoring COOLDOWN
    if should_fire:
        def _fire():
            if inter.trigger_delay > 0.0:
                schedule_trigger(inter, current_time + inter.trigger_delay)
            else:
                run_reactions(_get_linked_reactions(inter))
            inter.has_fired = True
            inter.last_trigger_time = current_time

        if inter.trigger_mode == "COOLDOWN":
            # Only fire if we haven't yet, or cooldown has passed
            elapsed = current_time - inter.last_trigger_time
            if (not inter.has_fired) or (elapsed >= inter.trigger_cooldown):
                _fire()
        else:
            # ONE_SHOT or default
            _fire()

    # 5) Store for next frame
    inter.last_known_count = new_val


def handle_timer_complete_trigger(inter):
    scene = bpy.context.scene
    idx_str = inter.timer_objective_index
    if not idx_str.isdigit():
        return

    idx = int(idx_str)
    if idx < 0 or idx >= len(scene.objectives):
        return

    objv = scene.objectives[idx]
    if objv.just_finished:
        current_time = get_game_time()
        if can_fire_trigger(inter, current_time):
            if inter.trigger_delay > 0.0:
                schedule_trigger(inter, current_time + inter.trigger_delay)
            else:
                run_reactions(_get_linked_reactions(inter))
            inter.has_fired = True
            inter.last_trigger_time = current_time

            objv.timer_active = False
            objv.just_finished = False
