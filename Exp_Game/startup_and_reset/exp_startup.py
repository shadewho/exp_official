#exp_startup.py


import bpy
import os
from ...exp_preferences import get_addon_path
from ..props_and_utils.exp_utilities import get_game_world
from .exp_fullscreen import enter_fullscreen_once
# ────────────────────────────────────
# Global for restoring scene
# ────────────────────────────────────
ORIGINAL_SCENE_NAME     = None

def center_cursor_in_3d_view(context, margin=50):
    """
    Warp the mouse cursor to the center of a VIEW_3D/WINDOW region.
    Robust against contexts that don't carry a region (fullscreen swaps, timers, UI overlays).
    If no suitable region exists, do nothing (no exception).
    """
    import bpy

    # Prefer the window in the incoming context; fall back to any window.
    win = getattr(context, "window", None)
    region = getattr(context, "region", None)
    region_is_ok = (region is not None) and (getattr(region, "type", None) == 'WINDOW')

    def _find_view3d_window_region():
        nonlocal win
        wm = bpy.context.window_manager
        windows = []
        if win is not None:
            windows.append(win)
        if wm:
            for w in wm.windows:
                if w not in windows:
                    windows.append(w)
        for w in windows:
            screen = getattr(w, "screen", None)
            if not screen:
                continue
            for area in screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                reg = next((r for r in area.regions if r.type == 'WINDOW'), None)
                if reg:
                    return w, reg
        return None, None

    if not region_is_ok:
        win, region = _find_view3d_window_region()

    # Nothing valid? Bail silently.
    if not win or not region:
        print("[center_cursor_in_3d_view] No VIEW_3D/WINDOW region; skipping cursor warp.")
        return
    if getattr(region, "width", 0) <= 0 or getattr(region, "height", 0) <= 0:
        print("[center_cursor_in_3d_view] Region has zero size; skipping cursor warp.")
        return

    # Compute safe center inside region margins
    cx = region.x + region.width // 2
    cy = region.y + region.height // 2
    x = max(region.x + margin, min(cx, region.x + region.width  - margin))
    y = max(region.y + margin, min(cy, region.y + region.height - margin))

    try:
        win.cursor_warp(x, y)
    except Exception as e:
        print(f"[center_cursor_in_3d_view] Cursor warp failed: {e}")



def clear_old_dynamic_references(self):
    """
    Clears any old dynamic references from previous runs,
    so we can start fresh when we invoke the modal again.
    """
    self.moving_meshes = []
    self.platform_prev_positions = {}
    self.platform_prev_quaternions = {}  # OPTIMIZED: 4 floats vs 16 for matrices
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
def set_viewport_shading(mode: str = "RENDERED"):
    """
    Sets the shading mode of all VIEW_3D areas on the *current* window's screen.
    Valid modes in Blender: 'WIREFRAME', 'SOLID', 'MATERIAL', 'RENDERED'.
    """
    valid = {"WIREFRAME", "SOLID", "MATERIAL", "RENDERED"}
    if mode not in valid:
        mode = "RENDERED"

    win = bpy.context.window
    if not win or not win.screen:
        return

    for area in win.screen.areas:
        if area.type == 'VIEW_3D':
            space = area.spaces.active
            if hasattr(space, "shading"):
                try:
                    space.shading.type = mode
                except Exception as e:
                    print(f"[WARN] Failed to set shading on {area}: {e}")


def ensure_timeline_at_zero():
    """
    Ensures that the scene's current frame is 0.
    This is important so that the NLA strips and animations are in sync when the modal starts.
    """
    scene = bpy.context.scene
    if scene.frame_current != 0:
        scene.frame_current = 0


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
        original_settings["use_shadows"] = getattr(scene.eevee, "use_shadows", True)
        original_settings["preview_pixel_size"] = scene.render.preview_pixel_size
    # Record 3D Viewport lens settings for each VIEW_3D area.
    # Convert the pointer to a string so that it can be used as a key.
    original_settings["viewport_shading_type"] = {}
    original_settings["viewport_lens"] = {}
    original_settings["viewport_show_shadows"] = {}

    win = bpy.context.window
    if win and win.screen:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                key = str(area.as_pointer())
                space = area.spaces.active
                # lens
                original_settings["viewport_lens"][key] = space.lens
                # shading type (defensive)
                stype = None
                if hasattr(space, "shading"):
                    try:
                        stype = space.shading.type
                    except Exception:
                        stype = None
                original_settings["viewport_shading_type"][key] = stype


def apply_performance_settings(scene, performance_level):
    """
    Applies performance settings based on the given performance_level.
    Otherwise, it forces the render engine to Eevee Next,
    applies the desired Eevee settings, sets the viewport shading to 'RENDERED',
    and sets the viewport lens to 55mm.
    """
    if performance_level not in {"LOW", "MEDIUM", "HIGH"}:
        performance_level = "LOW"

    # Force the render engine to Eevee Next.
    scene.render.engine = "BLENDER_EEVEE"

    prefs = bpy.context.preferences.addons["Exploratory"].preferences

    if hasattr(scene, "eevee"):
        # Force the two Eevee settings regardless of performance level:
        scene.eevee.use_taa_reprojection = False
        scene.eevee.use_shadow_jitter_viewport = False
        scene.eevee.use_raytracing = False
        scene.eevee.shadow_ray_count = 1
        scene.eevee.shadow_step_count = 1

    if hasattr(scene.eevee, "use_shadows"):
        scene.eevee.use_shadows = bool(prefs.enable_shadows_in_game)

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
    # Apply user-chosen viewport shading for gameplay
    try:
        set_viewport_shading(getattr(prefs, "viewport_shading_mode", "RENDERED"))
    except Exception as e:
        print(f"[WARN] Failed to apply preferred shading: {e}")

    # Apply preview pixel size from preferences
    try:
        # This is an Enum on Scene.render; strings like 'AUTO','1','2','4','8' are valid.
        scene.render.preview_pixel_size = getattr(prefs, "preview_pixel_size", "AUTO")
    except Exception as e:
        print(f"[WARN] Failed to set preview_pixel_size: {e}")

    # Set the viewport lens for each VIEW_3D area to 55mm.
    lens_mm = float(getattr(scene, "viewport_lens_mm", 55.0))
    for area in bpy.context.window.screen.areas:
        if area.type == 'VIEW_3D':
            space = area.spaces.active
            space.lens = lens_mm
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
        "use_shadows": (scene.eevee.use_shadows if hasattr(scene, "eevee") and hasattr(scene.eevee, "use_shadows") else "N/A"),

    })


def restore_user_settings(scene):
    """
    Restores settings from the snapshot stored as a custom property on the scene.
    If a property wasn’t recorded, it leaves the current value unchanged.
    """
    original_settings = scene.get("exploratory_original_settings")
    if not original_settings:
        return

    # Render engine
    scene.render.engine = original_settings.get("render_engine", scene.render.engine)

    # Eevee block
    if hasattr(scene, "eevee"):
        scene.eevee.use_taa_reprojection      = original_settings.get("use_taa_reprojection", scene.eevee.use_taa_reprojection)
        scene.eevee.use_shadow_jitter_viewport = original_settings.get("use_shadow_jitter_viewport", scene.eevee.use_shadow_jitter_viewport)
        scene.eevee.taa_samples               = original_settings.get("taa_samples", scene.eevee.taa_samples)
        scene.eevee.use_raytracing            = original_settings.get("use_raytracing", scene.eevee.use_raytracing)
        scene.eevee.shadow_ray_count          = original_settings.get("shadow_ray_count", scene.eevee.shadow_ray_count)
        scene.eevee.shadow_step_count         = original_settings.get("shadow_step_count", scene.eevee.shadow_step_count)
        scene.eevee.shadow_resolution_scale   = original_settings.get("shadow_resolution_scale", scene.eevee.shadow_resolution_scale)

        if hasattr(scene.eevee, "use_shadows"):
            scene.eevee.use_shadows = original_settings.get("use_shadows", scene.eevee.use_shadows)

    pps = original_settings.get("preview_pixel_size")
    if pps is not None:
        try:
            scene.render.preview_pixel_size = pps
        except Exception as e:
            print(f"[WARN] Failed to restore preview_pixel_size: {e}")

    # Per-viewport lens
    viewport_lens = original_settings.get("viewport_lens")
    if viewport_lens:
        win = bpy.context.window
        if win and win.screen:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    key = str(area.as_pointer())
                    if key in viewport_lens:
                        area.spaces.active.lens = viewport_lens[key]

    # Per-viewport shading
    viewport_types = original_settings.get("viewport_shading_type")
    if viewport_types:
        win = bpy.context.window
        if win and win.screen:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    key = str(area.as_pointer())
                    stype = viewport_types.get(key)
                    if stype:
                        try:
                            area.spaces.active.shading.type = stype
                        except Exception as e:
                            print(f"[WARN] Failed to restore shading for {area}: {e}")

    # Cleanup snapshot
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
                except Exception:
                    pass
        # Link the object to the destination scene if it's not already there.
        if not any(o == obj for o in destination_scene.collection.objects):
            destination_scene.collection.objects.link(obj)

def disable_live_perf_overlay_next_tick():
    """
    Keep the name for backward compat.
    On next tick: disable the Developer HUD (dev_hud_enable) on all scenes.
    """

    def _do_disable():
        try:
            # Current scene first
            sc = getattr(bpy.context, "scene", None)
            if sc and hasattr(sc, "dev_hud_enable"):
                sc.dev_hud_enable = False

            # Ensure it’s off everywhere (in case of scene switches)
            for s in bpy.data.scenes:
                if hasattr(s, "dev_hud_enable"):
                    s.dev_hud_enable = False
        except Exception as e:
            print(f"[WARN] disable_live_perf_overlay_next_tick failed: {e}")
        return None  # run once

    bpy.app.timers.register(_do_disable, first_interval=0.0)



#---------clear custom actions in NLA --- LEGACY (no longer used)
def clear_all_exp_custom_strips():
    """
    Legacy function - no longer needed with unified animation system.
    The new system doesn't use NLA tracks; animations are applied directly.
    """
    pass


#----------------------------------------------------------------------------------------


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
    else:
        print(f"[DEBUG] Original scene '{orig_name}' not found; skipping scene revert.")
        

#########################################
#Fullscreen then start the game
#########################################


def invoke_modal_in_current_view3d(launched_from_ui: bool):
    """Find a valid VIEW_3D + WINDOW region in the current window and invoke the modal there."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                if region:
                    override_ctx = {
                        'window': window,
                        'screen': window.screen,
                        'area':   area,
                        'region': region,
                    }
                    with bpy.context.temp_override(**override_ctx):
                        bpy.ops.view3d.exp_modal('INVOKE_DEFAULT', launched_from_ui=launched_from_ui)
                    return
    print("[StartGameFS] No VIEW_3D/WINDOW region available to invoke modal.")


class EXP_GAME_OT_StartGame(bpy.types.Operator):
    """Fullscreen Game Start"""
    
    bl_idname = "exploratory.start_game"
    bl_label = "Start Game"

    launched_from_ui: bpy.props.BoolProperty(
        name="Launched from UI",
        default=False
    )

    def execute(self, context):
        # capture before the timer so we never touch self later
        from_ui = bool(self.launched_from_ui)

        if from_ui:
            try:
                disable_live_perf_overlay_next_tick()
            except Exception:
                pass


        enter_fullscreen_once()

        def _start_modal_after_fs():
            invoke_modal_in_current_view3d(from_ui)  # no self here
            return None
        
        delay = 0.4 if from_ui else 0.2  # ← give the UI/area/handlers a beat to settle
        bpy.app.timers.register(_start_modal_after_fs, first_interval=delay)
        return {'FINISHED'}

