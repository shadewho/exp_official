#exp_startup.py


import bpy
import os
from ..exp_preferences import get_addon_path
from .exp_utilities import get_game_world

# ────────────────────────────────────
# Global for restoring workspace
# ────────────────────────────────────
ORIGINAL_WORKSPACE_NAME = None
ORIGINAL_SCENE_NAME     = None


def center_cursor_in_3d_view(context, margin=50):
    """
    Centers the cursor in the 3D View region with a specified margin to prevent it
    from being too close to the edges.
    
    Args:
        context (bpy.types.Context): The current Blender context.
        margin (int): The margin in pixels to keep the cursor away from the edges.
    """
    region = context.region
    if region.type != 'WINDOW':  # Ensure we're working within the 3D View region
        return

    # Calculate the center of the 3D View region
    center_x = region.x + region.width // 2
    center_y = region.y + region.height // 2

    # Adjust cursor to stay within the margins
    adjusted_x = max(region.x + margin, min(center_x, region.x + region.width - margin))
    adjusted_y = max(region.y + margin, min(center_y, region.y + region.height - margin))

    # Warp the cursor to the adjusted position
    context.window.cursor_warp(adjusted_x, adjusted_y)


def clear_old_dynamic_references(self):
    """
    Clears any old dynamic references from previous runs,
    so we can start fresh when we invoke the modal again.
    """
    self.moving_meshes = []
    self.dynamic_bvh_map = {}
    self.platform_prev_positions = {}
    self.platform_prev_matrices = {}
    if hasattr(self, "platform_motion_map"):
        self.platform_motion_map = {}
    if hasattr(self, "grounded_platform"):
        self.grounded_platform = None


#----------------------------------------------------------
# Ensure Object Mode if user calls from non-fullscreen modal
#----------------------------------------------------------
def ensure_object_mode(context):
    """Switch to Object Mode if in a 3D View and not already in Object mode."""
    if context.area and context.area.type == 'VIEW_3D':
        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')


#----------------------------------------------------------
### Snapshot of User Settings (No Global)
#----------------------------------------------------------
def set_viewport_shading_rendered():
    """
    Sets the shading mode of all VIEW_3D areas to 'RENDERED'.
    This must be done using the correct context (the current screen's areas).
    """
    # Loop over all areas in the current window's screen
    for area in bpy.context.window.screen.areas:
        if area.type == 'VIEW_3D':
            # The active space in a VIEW_3D area holds the shading settings.
            space = area.spaces.active
            if hasattr(space, "shading"):
                space.shading.type = 'RENDERED'
            else:
                print("No shading property found for area:", area)


def ensure_timeline_at_zero():
    """
    Ensures that the scene's current frame is 0.
    This is important so that the NLA strips and animations are in sync when the modal starts.
    """
    scene = bpy.context.scene
    if scene.frame_current != 0:
        scene.frame_current = 0
        print("Timeline frame reset to 0.")


def record_user_settings(scene):
    """
    Records a snapshot of user settings that you may later want to restore.
    Extend this list as needed.
    The settings are stored as a custom property on the scene.
    """
    # Create (or clear) a custom dictionary on the scene to store settings
    scene["exploratory_original_settings"] = {}
    original_settings = scene["exploratory_original_settings"]

    # Record the render engine type:
    original_settings["render_engine"] = scene.render.engine

    # Record Eevee-specific settings (if available)
    if hasattr(scene, "eevee"):
        original_settings["use_taa_reprojection"] = scene.eevee.use_taa_reprojection
        original_settings["use_shadow_jitter_viewport"] = scene.eevee.use_shadow_jitter_viewport
        original_settings["taa_samples"] = scene.eevee.taa_samples
        original_settings["use_raytracing"] = scene.eevee.use_raytracing
        original_settings["shadow_ray_count"] = scene.eevee.shadow_ray_count
        original_settings["shadow_step_count"] = scene.eevee.shadow_step_count
        original_settings["shadow_resolution_scale"] = scene.eevee.shadow_resolution_scale
    # Record 3D Viewport lens settings for each VIEW_3D area.
    # Convert the pointer to a string so that it can be used as a key.
    original_settings["viewport_lens"] = {}
    for area in bpy.context.window.screen.areas:
        if area.type == 'VIEW_3D':
            key = str(area.as_pointer())
            original_settings["viewport_lens"][key] = area.spaces.active.lens

    print("User settings recorded:", original_settings)


def apply_performance_settings(scene, performance_level):
    """
    Applies performance settings based on the given performance_level.
    If performance_level is "CUSTOM", this function does nothing.
    Otherwise, it forces the render engine to Eevee Next,
    applies the desired Eevee settings, sets the viewport shading to 'RENDERED',
    and sets the viewport lens to 55mm.
    """
    # If the user chose CUSTOM, do nothing.
    if performance_level == "CUSTOM":
        print("Custom performance mode selected; skipping performance settings.")
        return

    # Force the render engine to Eevee Next.
    scene.render.engine = "BLENDER_EEVEE_NEXT"

    if hasattr(scene, "eevee"):
        # Force the two Eevee settings regardless of performance level:
        scene.eevee.use_taa_reprojection = False
        scene.eevee.use_shadow_jitter_viewport = False
        scene.eevee.use_raytracing = False
        scene.eevee.shadow_ray_count = 1
        scene.eevee.shadow_step_count = 1

        # Adjust TAA samples based on performance level:
        if performance_level == "LOW":
            scene.eevee.taa_samples = 4
            scene.eevee.shadow_resolution_scale = 0.0
        elif performance_level == "MEDIUM":
            scene.eevee.taa_samples = 8
            scene.eevee.shadow_resolution_scale = 0.25
        elif performance_level == "HIGH":
            scene.eevee.taa_samples = 16
            scene.eevee.shadow_resolution_scale = 0.5

    # Now set all VIEW_3D areas’ shading mode to 'RENDERED'
    set_viewport_shading_rendered()

    # Set the viewport lens for each VIEW_3D area to 55mm.
    for area in bpy.context.window.screen.areas:
        if area.type == 'VIEW_3D':
            space = area.spaces.active
            space.lens = 55
            # Set the viewport to perspective mode.
            space.region_3d.view_perspective = 'PERSP'

    print("Performance settings applied:", {
        "render_engine": scene.render.engine,
        "taa_samples": scene.eevee.taa_samples if hasattr(scene, "eevee") else "N/A",
        "use_taa_reprojection": scene.eevee.use_taa_reprojection if hasattr(scene, "eevee") else "N/A",
        "use_shadow_jitter_viewport": scene.eevee.use_shadow_jitter_viewport if hasattr(scene, "eevee") else "N/A",
        "use_raytracing": scene.eevee.use_raytracing if hasattr(scene, "eevee") else "N/A",
        "shadow_ray_count": scene.eevee.shadow_ray_count if hasattr(scene, "eevee") else "N/A",
        "shadow_step_count": scene.eevee.shadow_step_count if hasattr(scene, "eevee") else "N/A",
        "shadow_resolution_scale": scene.eevee.shadow_resolution_scale if hasattr(scene, "eevee") else "N/A",
        "viewport_lens": 55,
    })


def restore_user_settings(scene):
    """
    Restores settings from the snapshot stored as a custom property on the scene.
    If a property wasn’t recorded, it leaves the current value unchanged.
    """
    original_settings = scene.get("exploratory_original_settings")
    if not original_settings:
        print("No settings were recorded; nothing to restore.")
        return

    # Restore the render engine type:
    scene.render.engine = original_settings.get("render_engine", scene.render.engine)

    if hasattr(scene, "eevee"):
        scene.eevee.use_taa_reprojection = original_settings.get("use_taa_reprojection", scene.eevee.use_taa_reprojection)
        scene.eevee.use_shadow_jitter_viewport = original_settings.get("use_shadow_jitter_viewport", scene.eevee.use_shadow_jitter_viewport)
        scene.eevee.taa_samples = original_settings.get("taa_samples", scene.eevee.taa_samples)
        scene.eevee.use_raytracing = original_settings.get("use_raytracing", scene.eevee.use_raytracing)
        scene.eevee.shadow_ray_count = original_settings.get("shadow_ray_count", scene.eevee.shadow_ray_count)
        scene.eevee.shadow_step_count = original_settings.get("shadow_step_count", scene.eevee.shadow_step_count)
        scene.eevee.shadow_resolution_scale = original_settings.get("shadow_resolution_scale", scene.eevee.shadow_resolution_scale)
        

    # Restore 3D Viewport lens settings for each VIEW_3D area.
    viewport_lens = original_settings.get("viewport_lens")
    if viewport_lens:
        for area in bpy.context.window.screen.areas:
            if area.type == 'VIEW_3D':
                key = str(area.as_pointer())
                if key in viewport_lens:
                    area.spaces.active.lens = viewport_lens[key]

    print("User settings restored:", original_settings)

    # Optionally, remove the custom property after restoring settings.
    del scene["exploratory_original_settings"]


def move_armature_and_children_to_scene(target_armature, destination_scene):
    """
    Moves the target armature and all objects parented (directly or indirectly)
    to it into the destination scene's master collection.
    Optionally, it unlinks these objects from other scenes.
    """
    # Build a list of objects to move: the armature plus all its children.
    objects_to_move = [target_armature] + list(target_armature.children_recursive)
    
    # For each object, remove it from all scenes (if desired) and link it to the destination scene.
    for obj in objects_to_move:
        # Unlink from any scene that is not the destination.
        for sc in bpy.data.scenes:
            if sc != destination_scene:
                try:
                    sc.collection.objects.unlink(obj)
                    print(f"Unlinked {obj.name} from scene {sc.name}")
                except Exception:
                    pass
        # Link the object to the destination scene if it's not already there.
        if not any(o == obj for o in destination_scene.collection.objects):
            destination_scene.collection.objects.link(obj)
            print(f"Linked {obj.name} to scene {destination_scene.name}")


def append_and_switch_to_game_workspace(context, workspace_name="exp_game"):
    """
    Appends a workspace from game_screen.blend (if not already present)
    and switches the current window to it.
    """

    scene = context.scene

    # 1) Store the original workspace name (if not already stored)
    if "original_workspace_name" not in scene:
        original_ws = context.window.workspace
        scene["original_workspace_name"] = original_ws.name
        print("Stored original workspace:", original_ws.name)
    else:
        print("Original workspace already stored as:", scene["original_workspace_name"])

    # 2) Build the path to your .blend file using get_addon_path()
    blend_path = os.path.join(get_addon_path(), "Exp_Game", "exp_assets", "Game_Screen", "game_screen.blend")

    # 3) Define the directory inside the blend file where workspaces are stored.
    directory = blend_path + "/WorkSpace/"
    filepath  = os.path.join(directory, workspace_name)

    # 4) Append the workspace if not already present.
    if workspace_name not in bpy.data.workspaces:
        try:
            bpy.ops.wm.append(
                filepath=filepath,
                directory=directory,
                filename=workspace_name
            )
            print(f"Appended workspace '{workspace_name}' from {blend_path}")
        except Exception as e:
            print(f"Failed to append workspace '{workspace_name}': {e}")

    # 5) Switch the current window to the appended workspace.
    new_ws = bpy.data.workspaces.get(workspace_name)
    if new_ws:
        context.window.workspace = new_ws
        print("Switched to new workspace:", new_ws.name)
    else:
        print(f"Workspace '{workspace_name}' not found after append.")

def revert_to_original_workspace(context):
    """
    Switch back to the workspace we started from, by using the
    global ORIGINAL_WORKSPACE_NAME slot instead of any scene property.
    """
    global ORIGINAL_WORKSPACE_NAME

    # ── 1) Read and clear the saved workspace name
    original_name = ORIGINAL_WORKSPACE_NAME
    print(f"[DEBUG] revert_to_original_workspace sees global: {original_name}")
    ORIGINAL_WORKSPACE_NAME = None

    # ── 2) Resolve a bpy.types.Workspace
    original_ws = bpy.data.workspaces.get(original_name)
    if not original_ws:
        original_ws = bpy.data.workspaces.get("Layout")
        if not original_ws and bpy.data.workspaces:
            original_ws = bpy.data.workspaces[0]
        print(f"[DEBUG] Falling back to workspace: {original_ws.name if original_ws else 'None'}")

    # ── 3) Delete the temporary "exp_game" workspace, then switch back
    def delete_and_revert():
        # a) delete exp_game if it exists, to avoid clutter
        game_ws = bpy.data.workspaces.get("exp_game")
        if game_ws:
            context.window.workspace = game_ws
            override = {
                'window': context.window,
                'screen': context.window.screen,
            }
            # find a VIEW_3D area/region for the override
            for area in context.window.screen.areas:
                if area.type == 'VIEW_3D':
                    override['area'] = area
                    for reg in area.regions:
                        if reg.type == 'WINDOW':
                            override['region'] = reg
                            break
                    break
            with bpy.context.temp_override(**override):
                bpy.ops.workspace.delete('EXEC_DEFAULT')

        # b) switch back to the original
        if original_ws:
            context.window.workspace = original_ws
            print(f"[DEBUG] Switched back to workspace: {original_ws.name}")
        return None  # stop the timer

    # register the timer so it runs after the modal cleanup
    bpy.app.timers.register(delete_and_revert, first_interval=0.1)

def revert_to_original_scene(context):
    """
    Switch back to the scene we started from, using either:
      1) the name stored on WindowManager by explore_icon_handler, or
      2) the global ORIGINAL_SCENE_NAME passed into EXP_GAME_OT_StartGame.
    """
    global ORIGINAL_SCENE_NAME

    # 1) Try to pull from the window_manager (and remove it)
    orig_name = context.window_manager.pop('original_scene', None)

    # 2) If that wasn't set (or was already popped), fall back to the global
    if not orig_name:
        orig_name = ORIGINAL_SCENE_NAME

    # 3) Clear the global so it won't persist
    ORIGINAL_SCENE_NAME = None

    # 4) If that scene still exists, switch back to it
    if orig_name and orig_name in bpy.data.scenes:
        context.window.scene = bpy.data.scenes[orig_name]
        print(f"[DEBUG] Switched back to original scene: {orig_name}")
    else:
        print(f"[DEBUG] Original scene '{orig_name}' not found; skipping scene revert.")
def delayed_invoke_modal(from_ui):
    for window in bpy.context.window_manager.windows:
        if window.workspace.name == "exp_game":
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    override_ctx = bpy.context.copy()
                    override_ctx['window'] = window
                    override_ctx['screen'] = window.screen
                    override_ctx['area'] = area
                    for reg in area.regions:
                        if reg.type == 'WINDOW':
                            override_ctx['region'] = reg
                            break
                    with bpy.context.temp_override(**override_ctx):
                        bpy.ops.view3d.exp_modal('INVOKE_DEFAULT', launched_from_ui=from_ui)
                    return None
    return 0.1

#########################################
#Append workspace then start the game
#########################################
class EXP_GAME_OT_StartGame(bpy.types.Operator):
    bl_idname = "exploratory.start_game"
    bl_label = "Start Game"

    launched_from_ui: bpy.props.BoolProperty(
        name="Launched from UI",
        default=False
    )
    original_workspace_name: bpy.props.StringProperty(
        name="Original Workspace Name",
        default=""
    )
    original_scene_name: bpy.props.StringProperty(
        name="Original Scene Name",
        default=""
    )

    def execute(self, context):
        global ORIGINAL_WORKSPACE_NAME, ORIGINAL_SCENE_NAME

        # ─── Store original workspace ───────────────────────────
        if self.original_workspace_name:
            ORIGINAL_WORKSPACE_NAME = self.original_workspace_name
        else:
            ORIGINAL_WORKSPACE_NAME = context.window.workspace.name
        print(f"[DEBUG] ORIGINAL_WORKSPACE_NAME = {ORIGINAL_WORKSPACE_NAME}")

        # ─── Store original scene ───────────────────────────────
        if self.original_scene_name:
            ORIGINAL_SCENE_NAME = self.original_scene_name
        else:
            ORIGINAL_SCENE_NAME = context.window.scene.name
        print(f"[DEBUG] ORIGINAL_SCENE_NAME     = {ORIGINAL_SCENE_NAME}")

        # ─── Switch into the exp_game workspace ────────────────
        append_and_switch_to_game_workspace(context, "exp_game")

        # ─── Schedule the modal launch ─────────────────────────
        from_ui = self.launched_from_ui
        def start_modal():
            delayed_invoke_modal(from_ui)
            return None
        bpy.app.timers.register(start_modal, first_interval=0.2)

        return {'FINISHED'}
    

    