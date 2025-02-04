#exp_startup.py


import bpy

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

        # Adjust TAA samples based on performance level:
        if performance_level == "LOW":
            scene.eevee.taa_samples = 4
        elif performance_level == "MEDIUM":
            scene.eevee.taa_samples = 8
        elif performance_level == "HIGH":
            scene.eevee.taa_samples = 16

    # Now set all VIEW_3D areas’ shading mode to 'RENDERED'
    set_viewport_shading_rendered()

    # Set the viewport lens for each VIEW_3D area to 55mm.
    for area in bpy.context.window.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces.active.lens = 55

    print("Performance settings applied:", {
        "render_engine": scene.render.engine,
        "taa_samples": scene.eevee.taa_samples if hasattr(scene, "eevee") else "N/A",
        "use_taa_reprojection": scene.eevee.use_taa_reprojection if hasattr(scene, "eevee") else "N/A",
        "use_shadow_jitter_viewport": scene.eevee.use_shadow_jitter_viewport if hasattr(scene, "eevee") else "N/A",
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
