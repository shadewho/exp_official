#Exploratory/Exp_Game/exp_game_reset.py

import bpy
from ..props_and_utils.exp_time import init_time
from ..systems.exp_objectives import reset_all_objectives
from ..reactions.exp_reactions import reset_all_tasks, _assign_safely
from ..interactions.exp_interactions import reset_all_interactions
from ..startup_and_reset.exp_spawn import spawn_user
from ..audio import exp_globals
from ..reactions.exp_custom_ui import clear_all_text
from ..systems.exp_performance import rearm_performance_after_reset
from ..reactions.exp_crosshairs import disable_crosshairs
from ..reactions.exp_tracking import clear as clear_tracking_tasks

def capture_scene_state(self, context):
    """
    Fills self._initial_game_state with enough data
    so we can restore the entire scene later.
    """
    scene = context.scene

    # Clear any old data
    self._initial_game_state.clear()

    # 1) object transforms + viewport visibility + parent inverse
    self._initial_game_state["object_transforms"] = {}
    for obj in scene.objects:
        self._initial_game_state["object_transforms"][obj.name] = {
            "location":      obj.location.copy(),
            "rotation":      obj.rotation_euler.copy(),
            "scale":         obj.scale.copy(),
            "hide_viewport": obj.hide_viewport,
            "parent_inv":    obj.matrix_parent_inverse.copy(),  # ← NEW
        }
        

    # 1.5) original parent relationships for all objects
    try:
        from ..reactions import exp_parenting as _parenting
        _parenting.capture_original_parents(self, context)
    except Exception as e:
        print(f"[WARN] capture_original_parents failed: {e}")

    # 2) Add another subdict for scene‐level data if you want
    self._initial_game_state["scene_data"] = {}
    mg = scene.mobility_game
    self._initial_game_state["scene_data"]["mobility_flags"] = {
        "allow_movement": mg.allow_movement,
        "allow_jump":     mg.allow_jump,
        "allow_sprint":   mg.allow_sprint,
    }

def restore_scene_state(modal_op, context):
    """
    Correct order:
      A) restore original parents (also removes runtime Child-Of constraints)
      B) then restore local transforms and visibility
    """
    state = getattr(modal_op, "_initial_game_state", None)
    if not state:
        return

    # A) parents first (so local transforms below mean the same thing again)
    try:
        from ..reactions import exp_parenting as _parenting
        _parenting.restore_original_parents(modal_op, context)
    except Exception as e:
        print(f"[WARN] restore_original_parents failed: {e}")

    # B) Per-object transforms + viewport visibility
    obj_transforms = state.get("object_transforms", {})
    for obj_name, xform_data in obj_transforms.items():
        obj = bpy.data.objects.get(obj_name)
        if not obj:
            continue
        try:
            obj.location       = xform_data["location"]
            obj.rotation_euler = xform_data["rotation"]
            obj.scale          = xform_data["scale"]
        except Exception:
            pass

        # Restore viewport visibility if it was captured.
        try:
            if "hide_viewport" in xform_data:
                obj.hide_viewport = xform_data["hide_viewport"]
        except Exception:
            pass


def apply_hide_during_game(modal_op, context):
    """
    Ensure all proxy meshes marked 'hide_during_game' are hidden.
    Preserves the original hide state in modal_op._proxy_mesh_original_states once.
    """
    if not hasattr(modal_op, "_proxy_mesh_original_states"):
        modal_op._proxy_mesh_original_states = {}

    for entry in context.scene.proxy_meshes:
        obj = entry.mesh_object
        if not obj:
            continue

        if entry.hide_during_game:
            # Remember original only once so cancel() can restore it later
            if obj.name not in modal_op._proxy_mesh_original_states:
                modal_op._proxy_mesh_original_states[obj.name] = bool(obj.hide_viewport)
            obj.hide_viewport = True


def reset_property_reactions(scene):
    """
    Reset all PROPERTY-type reactions to their default_* values at game start.

    Works with the new architecture:
      • Uses InteractionDefinition.reaction_links (indices → scene.reactions)
      • Also resets standalone (unlinked) reactions in scene.reactions
      • Deduplicates by property_data_path
    """
    # 1) Gather reactions referenced by interactions
    to_visit = []
    for inter in getattr(scene, "custom_interactions", []):
        for link in getattr(inter, "reaction_links", []):
            idx = getattr(link, "reaction_index", -1)
            if 0 <= idx < len(scene.reactions):
                to_visit.append(scene.reactions[idx])

    # 2) Include all global reactions so standalones are reset too
    to_visit.extend(getattr(scene, "reactions", []))

    # 3) Reset each unique PROPERTY path to its declared default_* value
    seen_paths = set()

    for r in to_visit:
        if getattr(r, "reaction_type", None) != "PROPERTY":
            continue

        path_str = (getattr(r, "property_data_path", "") or "").strip()
        if not path_str or path_str in seen_paths:
            continue

        ptype = getattr(r, "property_type", "NONE")
        if ptype == "BOOL":
            def_val = bool(r.default_bool_value)
        elif ptype == "INT":
            def_val = int(r.default_int_value)
        elif ptype == "FLOAT":
            def_val = float(r.default_float_value)
        elif ptype == "STRING":
            def_val = r.default_string_value
        elif ptype == "VECTOR":
            def_val = list(r.default_vector_value[:getattr(r, "vector_length", 3)])
        else:
            # Unknown/undetected → skip
            continue

        # Assign safely (casts component-wise to the underlying property type)
        _assign_safely(path_str, def_val)
        seen_paths.add(path_str)

class EXPLORATORY_OT_ResetGame(bpy.types.Operator):
    bl_idname = "exploratory.reset_game"
    bl_label  = "Reset Game"

    # new property: skip the restore step when True
    skip_restore: bpy.props.BoolProperty(
        name="Skip Restore",
        description="If true, do not restore transforms (used when called on cancel)",
        default=False,
    )

    def execute(self, context):
        modal_op = exp_globals.ACTIVE_MODAL_OP
        if not modal_op:
            self.report({'WARNING'}, "No active ExpModal found.")
            return {'CANCELLED'}

        # --- NLA critical section so scrubbing can't fight the wipe ---
        from ..animations.exp_animations import nla_guard_enter, nla_guard_exit, get_global_animation_manager
        nla_guard_enter()
        try:
            # Quiesce the animation manager so no stale strip refs remain
            try:
                mgr = get_global_animation_manager()
                if mgr:
                    mgr.active_actions.clear()
                    mgr.one_time_in_progress = False
                    mgr.last_action_name = None
            except Exception:
                pass

            # ─── 0) Reset the game clock first ───────────────────────────
            #    so that any "last_trigger_time" stamps use the new zero baseline.
            init_time()

            # ─── 0.5) Stop custom actions and rewind exp_custom strips ───
            # Do this BEFORE restoring object transforms, so strips can't "fight" the pose.
            try:
                from ..animations.exp_custom_animations import stop_custom_actions_and_rewind_strips
                stop_custom_actions_and_rewind_strips()
            except Exception as e:
                print(f"[WARN] stop_custom_actions_and_rewind_strips failed: {e}")

            # ─── 0.6) Stop all playing sounds ─────────────────────────────────
            try:
                exp_globals.stop_all_sounds()
            except Exception as e:
                print(f"[WARN] stop_all_sounds failed: {e}")

            # 1) only restore the scene if skip_restore is False
            if not self.skip_restore:
                restore_scene_state(modal_op, context)
                apply_hide_during_game(modal_op, context)

            # ---- Reset dynamic platform state so no stale deltas apply after reset ----
            # Clear BVH caches so they rebuild against the restored transforms
            modal_op.cached_dynamic_bvhs = {}
            modal_op.dynamic_bvh_map = {}

            # Drop any notion of being "on a platform"
            modal_op.grounded_platform = None

            # Re-seed prev positions/matrices to CURRENT matrices post-restore
            modal_op.platform_motion_map = {}
            modal_op.platform_delta_map = {}

            if hasattr(modal_op, "moving_meshes"):
                modal_op.platform_prev_positions = {}
                modal_op.platform_prev_matrices  = {}
                for dyn_obj in modal_op.moving_meshes:
                    if dyn_obj:
                        modal_op.platform_prev_positions[dyn_obj] = dyn_obj.matrix_world.translation.copy()
                        modal_op.platform_prev_matrices[dyn_obj]  = dyn_obj.matrix_world.copy()

            # Optional: suppress one frame of platform-delta application (see Patch 3)
            from ..props_and_utils.exp_time import get_game_time
            modal_op._suppress_platform_delta_until = get_game_time() + 1e-6

            # ─── 2) Reset interactions, tasks, objectives, and properties ─
            #    Now that game_time == 0.0, last_trigger_time will be set to 0.0.
            reset_all_interactions(context.scene)
            reset_all_tasks()
            clear_tracking_tasks()
            reset_all_objectives(context.scene)
            reset_property_reactions(context.scene)

            # ─── 3) Clear any on-screen text and crpsshair and respawn the user ───────
            clear_all_text()
            try:
                disable_crosshairs()
            except Exception:
                pass
            spawn_user()

            # Re-arm performance culling after restoring visibility
            rearm_performance_after_reset(modal_op, context)

            self.report({'INFO'}, "Game fully reset.")
            return {'FINISHED'}

        finally:
            # Leave the NLA critical section so scrubbing can resume next tick
            try:
                nla_guard_exit()
            except Exception:
                pass



def setattr_recursive(scene, dotted_path, value):
    """
    scene property keys might be dotted like "mobility_game.allow_movement",
    we can parse them. If you just store them as "allow_movement" that belongs 
    directly to the mobility_game pointer, you could manually do 
    scene.mobility_game.allow_movement = ...
    """
    parts = dotted_path.split('.')
    target = scene
    for p in parts[:-1]:
        target = getattr(target, p)
    setattr(target, parts[-1], value)




#--------------------------------------------------------
#reset the state of the armature and the view 3d camera
#--------------------------------------------------------
def capture_initial_cam_state(modal_op, context):
    """
    Capture just the VIEW_3D camera state into modal_op._initial_session_state
    so we can restore it later.
    """
    # start fresh
    modal_op._initial_session_state = {}

    # find the first 3D View and record its view settings
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            r3d = area.spaces.active.region_3d
            modal_op._initial_session_state["cam_loc"]  = r3d.view_location.copy()
            modal_op._initial_session_state["cam_rot"]  = r3d.view_rotation.copy()
            modal_op._initial_session_state["cam_dist"] = r3d.view_distance
            break


def capture_initial_character_state(modal_op, context):
    """
    After spawn_user() has positioned the character, capture its final transform.
    """
    arm = context.scene.target_armature
    if arm:
        modal_op._initial_session_state["char_loc"] = arm.location.copy()
        modal_op._initial_session_state["char_rot"] = arm.rotation_euler.copy()


def restore_initial_session_state(modal_op, context):
    """
    Restore camera always; restore character only when
    modal_op.launched_from_ui is False.
    """
    state = getattr(modal_op, "_initial_session_state", None)
    if not state:
        return

    # ─── A) Restore camera on every VIEW_3D ────────────────────────
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                r3d = area.spaces.active.region_3d
                # camera
                if "cam_loc" in state:
                    r3d.view_location = state["cam_loc"]
                if "cam_rot" in state:
                    r3d.view_rotation = state["cam_rot"]
                if "cam_dist" in state:
                    r3d.view_distance = state["cam_dist"]

    # ─── B) Restore character only when NOT launched_from_ui ───────
    if not getattr(modal_op, "launched_from_ui", False):
        arm = context.scene.target_armature
        if arm and "char_loc" in state and "char_rot" in state:
            arm.location       = state["char_loc"]
            arm.rotation_euler = state["char_rot"]
