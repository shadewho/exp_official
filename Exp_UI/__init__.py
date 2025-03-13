bl_info = {
    "name": "UI_TEST_ADDON_3",
    "blender": (4, 2, 0),
    "author": "Your Name",
    "description": "A simple add-on for UI EXP purposes in Blender.",
    "version": (1, 0, 2),
}


import bpy
from bpy.app.handlers import persistent

from .auth import token_expiry_check, is_internet_available, clear_token
# Import operators, panels, and other modules as before.
from .backend import LOGIN_OT_WebApp, LOGOUT_OT_WebApp, DOWNLOAD_CODE_OT_File
from .panel import (
    VIEW3D_PT_PackageDisplay_Login,
    VIEW3D_PT_PackageDisplay_FilterAndScene,
    VIEW3D_PT_PackageDisplay_CurrentItem,
    VIEW3D_PT_SubscriptionUsage
)
from .social_operators import (
    LIKE_PACKAGE_OT_WebApp, COMMENT_PACKAGE_OT_WebApp, 
    OPEN_URL_OT_WebApp, EXPLORATORY_UL_Comments, REFRESH_USAGE_OT_WebApp,
    POPUP_SOCIAL_DETAILS_OT
)
from .addon_data import MyAddonComment, MyAddonSceneProps
from bpy.props import PointerProperty
from .image_button_UI.operators import (
    FETCH_PAGE_THREADED_OT_WebApp,
    PACKAGE_OT_Display,
    REMOVE_PACKAGE_OT_Display,
    APPLY_FILTERS_SHOWUI_OT
)
from .cache_memory_operators import (
    CLEAR_ALL_DATA_OT_WebApp, CLEAR_THUMBNAILS_ONLY_OT_WebApp, 
    REFRESH_FILTERS_OT_WebApp, PRELOAD_METADATA_OT_WebApp, preload_metadata_timer
)

# Import functions from our cache module.
from .image_button_UI.cache import preload_in_memory_thumbnails, clear_image_datablocks

# List all classes that will be registered.
classes = (
    LOGIN_OT_WebApp,
    LOGOUT_OT_WebApp,
    DOWNLOAD_CODE_OT_File,
    LIKE_PACKAGE_OT_WebApp,
    COMMENT_PACKAGE_OT_WebApp,
    OPEN_URL_OT_WebApp,
    VIEW3D_PT_PackageDisplay_Login,
    VIEW3D_PT_PackageDisplay_FilterAndScene,
    VIEW3D_PT_PackageDisplay_CurrentItem,
    VIEW3D_PT_SubscriptionUsage,
    EXPLORATORY_UL_Comments,
    REFRESH_USAGE_OT_WebApp,
    FETCH_PAGE_THREADED_OT_WebApp,
    REMOVE_PACKAGE_OT_Display,
    PACKAGE_OT_Display,
    APPLY_FILTERS_SHOWUI_OT,
    CLEAR_THUMBNAILS_ONLY_OT_WebApp,
    CLEAR_ALL_DATA_OT_WebApp,
    REFRESH_FILTERS_OT_WebApp,
    PRELOAD_METADATA_OT_WebApp,
    POPUP_SOCIAL_DETAILS_OT
)

# --- Persistent Handler ---
@persistent
def on_blend_load(dummy):
    """
    This handler is called when a new blend file is loaded.
    It clears out our cached image datablocks so that stale references are not used.
    """
    print("[INFO] New blend file loaded; clearing image datablocks.")
    clear_image_datablocks()

def connectivity_check_timer():
    if not is_internet_available():
        # Log out and disable UI
        clear_token()
        if bpy.context.scene.get("my_addon_data"):
            bpy.context.scene.my_addon_data.is_from_webapp = False
        bpy.context.scene.ui_current_mode = "GAME"
        print("[INFO] No internet connection detected. User logged out and UI disabled.")
    return 10.0  # Check every 10 seconds

# --- Registration ---
def register():
    # Register the persistent load handler if not already registered.
    if on_blend_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_blend_load)
    
    # Register a timer for preloading metadata.
    bpy.app.timers.register(preload_metadata_timer, first_interval=30.0)

    # Register the token expiry check timer.
    bpy.app.timers.register(token_expiry_check, first_interval=60.0)

    # Register the comment and main property group classes.
    bpy.utils.register_class(MyAddonComment)
    bpy.utils.register_class(MyAddonSceneProps)
    bpy.types.Scene.my_addon_data = PointerProperty(type=MyAddonSceneProps)

    # Register UI mode and other scene properties.
    bpy.types.Scene.ui_current_mode = bpy.props.EnumProperty(
        name="UI Current Mode",
        description="Which UI mode is currently shown",
        items=[
            ("BROWSE", "Browse Thumbnails", ""),
            ("DETAIL", "Item Detail View", ""),
            ("LOADING", "Loading Progress", ""),
            ("GAME", "Game Mode", "")
        ],
        default="BROWSE"
    )
    bpy.types.Scene.show_image_buttons = bpy.props.BoolProperty(
        name="Show Package Thumbnails",
        description="Toggle the display of package thumbnail buttons",
        default=False
    )
    bpy.types.Scene.selected_thumbnail = bpy.props.StringProperty(
        name="Selected Thumbnail",
        description="Path to the selected thumbnail image",
        default=""
    )
    bpy.types.Scene.package_item_type = bpy.props.EnumProperty(
        name="Item Type",
        description="World or Shop",
        items=[
            ('world', 'World', ''),
            ('shop_item', 'Shop Item', '')
        ],
        default='world'
    )
    bpy.types.Scene.package_sort_by = bpy.props.EnumProperty(
        name="Sort By",
        description="Sorting Method",
        items=[
            ('newest', 'Newest', ''),
            ('oldest', 'Oldest', ''),
            ('popular', 'Popular', ''),
            ('random', 'Random', ''),
            ('featured', 'Featured', '')
        ],
        default='newest'
    )
    bpy.types.Scene.package_search_query = bpy.props.StringProperty(
        name="Search Query",
        description="Filter packages by name or author",
        default=""
    )
    bpy.types.Scene.download_code = bpy.props.StringProperty(
        name="Download Code",
        description="Enter the download code for the .blend item"
    )
    bpy.types.Scene.current_thumbnail_page = bpy.props.IntProperty(
        name="Current Thumbnail Page",
        description="Tracks the current page number for thumbnail navigation",
        default=1,
        min=1
    )
    bpy.types.Scene.total_thumbnail_pages = bpy.props.IntProperty(
        name="Total Thumbnail Pages",
        description="Tracks the total number of pages available for thumbnails",
        default=1,
        min=1
    )
    bpy.types.Scene.show_loading_image = bpy.props.BoolProperty(
        name="Show Loading",
        description="Toggle the loading indicator on/off",
        default=False
    )
    bpy.types.Scene.comment_text = bpy.props.StringProperty(name="Comment", default="")
    bpy.types.Scene.show_loading = bpy.props.BoolProperty(
        name="Show Loading",
        description="Display the loading indicator",
        default=False
    )
    bpy.types.Scene.download_progress = bpy.props.FloatProperty(
        name="Download Progress",
        description="Download progress (0.0 to 1.0)",
        default=0.0,
        min=0.0,
        max=1.0
    )

    # Register all addon classes.
    for cls in classes:
        bpy.utils.register_class(cls)

    # Preload in-memory thumbnails from the disk cache.
    preload_in_memory_thumbnails()

def unregister():
    # Remove all properties from bpy.types.Scene.
    del bpy.types.Scene.show_image_buttons
    del bpy.types.Scene.ui_current_mode
    del bpy.types.Scene.selected_thumbnail
    del bpy.types.Scene.package_item_type
    del bpy.types.Scene.package_sort_by
    del bpy.types.Scene.package_search_query
    del bpy.types.Scene.my_addon_data
    del bpy.types.Scene.current_thumbnail_page
    del bpy.types.Scene.total_thumbnail_pages
    del bpy.types.Scene.download_code
    del bpy.types.Scene.comment_text
    del bpy.types.Scene.show_loading
    del bpy.types.Scene.download_progress

    bpy.utils.unregister_class(MyAddonComment)
    bpy.utils.unregister_class(MyAddonSceneProps)

    # Unregister addon classes in reverse order.
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    # Remove the persistent handler if it was registered.
    if on_blend_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_blend_load)

if __name__ == "__main__":
    register()
