import bpy
from .exp_time import init_time
from .exp_objectives import reset_all_objectives
from .exp_reactions import reset_all_tasks
from .exp_interactions import reset_all_interactions
from .exp_spawn import spawn_user
from . import exp_globals
from .exp_custom_ui import clear_all_text

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
        # Store a sub‐dictionary for each object’s name
        self._initial_game_state["object_transforms"][obj.name] = {
            "location": obj.location.copy(),
            "rotation": obj.rotation_euler.copy(),
            "scale":    obj.scale.copy(),
        }
        # If you later want to store more (like hide_viewport, etc.),
        # just add more fields here:
        # e.g. "hide_viewport": obj.hide_viewport,
        # e.g. "some_custom_flag": your_own_logic
        

    # 2) Add another subdict for scene‐level data if you want
    self._initial_game_state["scene_data"] = {}
    mg = scene.mobility_game
    self._initial_game_state["scene_data"]["mobility_flags"] = {
        "allow_movement": mg.allow_movement,
        "allow_jump":     mg.allow_jump,
        "allow_sprint":   mg.allow_sprint,
    }

    print("capture_scene_state: Done capturing all transforms + some scene props.")

def restore_scene_state(modal_op, context):
    state = getattr(modal_op, "_initial_game_state", None)
    if not state:
        print("No stored state found—cannot restore.")
        return

    # A) Per-object transforms
    obj_transforms = state.get("object_transforms", {})
    for obj_name, xform_data in obj_transforms.items():
        obj = bpy.data.objects.get(obj_name)
        if obj:
            obj.location       = xform_data["location"]
            obj.rotation_euler = xform_data["rotation"]
            obj.scale          = xform_data["scale"]



class EXPLORATORY_OT_ResetGame(bpy.types.Operator):
    bl_idname = "exploratory.reset_game"
    bl_label = "Reset Game"

    def execute(self, context):
        modal_op = exp_globals.ACTIVE_MODAL_OP
        if not modal_op:
            self.report({'WARNING'}, "No active ExpModal found.")
            return {'CANCELLED'}

        # 1) Actually restore transforms & scene data:
        restore_scene_state(modal_op, context)

        # 2) Reset interactions, tasks, time, etc.
        reset_all_interactions(context.scene)
        reset_all_tasks()
        reset_all_objectives(context.scene)
        init_time()
        clear_all_text()

        # 3) Re-spawn if you want
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
