#Exploratory/Exp_Game/exp_game_reset.py

import bpy
from ..props_and_utils.exp_time import init_time
from ..systems.exp_objectives import reset_all_objectives
from ..reactions.exp_reactions import reset_all_tasks, _set_property_value
from ..interactions.exp_interactions import reset_all_interactions
from ..startup_and_reset.exp_spawn import spawn_user
from ..audio import exp_globals
from ..reactions.exp_custom_ui import clear_all_text

def capture_scene_state(self, context):
    """
    Fills self._initial_game_state with enough data
    so we can restore the entire scene later.
    """
    scene = context.scene

    # Clear any old data
    self._initial_game_state.clear()

    # 1) Make a subdict for all object transforms
    self._initial_game_state["object_transforms"] = {}

    for obj in scene.objects:
        # Capture the object's transform data plus viewport visibility.
        self._initial_game_state["object_transforms"][obj.name] = {
            "location": obj.location.copy(),
            "rotation": obj.rotation_euler.copy(),
            "scale":    obj.scale.copy(),
            "hide_viewport": obj.hide_viewport,   # <-- Capture the viewport visibility.
        }
        

    # 2) Add another subdict for scene‐level data if you want
    self._initial_game_state["scene_data"] = {}
    mg = scene.mobility_game
    self._initial_game_state["scene_data"]["mobility_flags"] = {
        "allow_movement": mg.allow_movement,
        "allow_jump":     mg.allow_jump,
        "allow_sprint":   mg.allow_sprint,
    }

def restore_scene_state(modal_op, context):
    state = getattr(modal_op, "_initial_game_state", None)
    if not state:
        return

    # A) Per-object transforms
    obj_transforms = state.get("object_transforms", {})
    for obj_name, xform_data in obj_transforms.items():
        obj = bpy.data.objects.get(obj_name)
        if obj:
            obj.location       = xform_data["location"]
            obj.rotation_euler = xform_data["rotation"]
            obj.scale          = xform_data["scale"]

            # Restore viewport visibility if it was captured.
            if "hide_viewport" in xform_data:
                obj.hide_viewport = xform_data["hide_viewport"]


def reset_property_reactions(scene):
    """
    For each Interaction in the scene, find any PROPERTY reaction and reset the target property
    to the user-defined default value.
    """
    # Assuming custom interactions are stored on scene.custom_interactions
    for inter in scene.custom_interactions:
        for reaction in inter.reactions:
            if reaction.reaction_type == "PROPERTY":
                path_str = reaction.property_data_path.strip()
                if not path_str:
                    continue
                # Choose the default value based on the detected property type:
                if reaction.property_type == "BOOL":
                    def_val = reaction.default_bool_value
                elif reaction.property_type == "INT":
                    def_val = reaction.default_int_value
                elif reaction.property_type == "FLOAT":
                    def_val = reaction.default_float_value
                elif reaction.property_type == "STRING":
                    def_val = reaction.default_string_value
                elif reaction.property_type == "VECTOR":
                    def_val = list(reaction.default_vector_value[:reaction.vector_length])
                else:
                    continue

                # Use the helper (_set_property_value) to assign the default value:
                _set_property_value(path_str, def_val)

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

        # ─── 0) Reset the game clock first ───────────────────────────
        #    so that any "last_trigger_time" stamps use the new zero baseline.
        init_time()

        # 1) only restore the scene if skip_restore is False
        if not self.skip_restore:
            restore_scene_state(modal_op, context)

        # ─── 2) Reset interactions, tasks, objectives, and properties ─
        #    Now that game_time == 0.0, last_trigger_time will be set to 0.0.
        reset_all_interactions(context.scene)
        reset_all_tasks()
        reset_all_objectives(context.scene)
        reset_property_reactions(context.scene)

        # ─── 3) Clear any on-screen text and respawn the user ───────
        clear_all_text()
        spawn_user()

        self.report({'INFO'}, "Game fully reset.")
        return {'FINISHED'}



def setattr_recursive(scene, dotted_path, value):
    """
    If your scene property keys might be dotted like "mobility_game.allow_movement",
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
