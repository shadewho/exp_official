#Exploratory/Exp_UI/__init__.py
import bpy
from bpy.app.handlers import persistent
from bpy.props import PointerProperty
from .internet.helpers import is_internet_available, clear_token
from .auth.helpers import token_expiry_check

from .panel import (
    VIEW3D_PT_PackageDisplay_Login,
    VIEW3D_PT_PackageDisplay_FilterAndScene,
    VIEW3D_PT_PackageDisplay_CurrentItem,
    VIEW3D_PT_ProfileAccount,
    VIEW3D_PT_SettingsAndUpdate,)


from .packages.social_operators import (
    LIKE_PACKAGE_OT_WebApp, COMMENT_PACKAGE_OT_WebApp, 
    OPEN_URL_OT_WebApp, EXPLORATORY_UL_Comments,
    POPUP_SOCIAL_DETAILS_OT
)
from .packages.operators import DOWNLOAD_CODE_OT_File

from .cache_system.operators.clear import (CLEAR_ALL_DATA_OT_WebApp, CLEAR_THUMBNAILS_ONLY_OT_WebApp)
from .cache_system.operators.metadata import PRELOAD_METADATA_OT_WebApp
from .cache_system.operators.refresh import REFRESH_FILTERS_OT_WebApp
from .cache_system.manager import CacheManager

# Import functions from our cache module.
from .cache_system.persistence import clear_image_datablocks
from .cache_system.preload import preload_in_memory_thumbnails, preload_metadata_timer
from .auth.helpers import auto_refresh_usage

#Auth operators
from .auth.operators import (
    LOGIN_OT_WebApp, LOGOUT_OT_WebApp, REFRESH_USAGE_OT_WebApp, OPEN_DOCS_OT)

#interface operators
from .interface.operators.apply_filters import APPLY_FILTERS_SHOWUI_OT
from .interface.operators.display import PACKAGE_OT_Display
from .interface.operators.fetch import FETCH_PAGE_THREADED_OT_WebApp
from .interface.operators.remove import REMOVE_PACKAGE_OT_Display


#evemts operators
from .events.operators import VOTE_MAP_OT_WebApp

from .interface.properties import register as register_ui_props, unregister as unregister_ui_props


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
    VIEW3D_PT_ProfileAccount,
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
    POPUP_SOCIAL_DETAILS_OT,
    VOTE_MAP_OT_WebApp,
    VIEW3D_PT_SettingsAndUpdate,
    OPEN_DOCS_OT
)
from .packages.properties import MyAddonComment, PackageProps
from .events.properties import register as register_event_props, unregister as unregister_event_props
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

    register_event_props()

    if on_blend_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_blend_load)
    
    # Register a timer for preloading metadata.
    bpy.app.timers.register(preload_metadata_timer, first_interval=30.0)

    # Register the token expiry check timer.
    bpy.app.timers.register(token_expiry_check, first_interval=60.0)

    bpy.app.timers.register(auto_refresh_usage, first_interval=1.0)

    # Register the comment and main property group classes.
    bpy.utils.register_class(MyAddonComment)
    bpy.utils.register_class(PackageProps)
    bpy.types.Scene.my_addon_data = PointerProperty(type=PackageProps)

    register_ui_props()

    bpy.types.Scene.package_item_type = bpy.props.EnumProperty(
        name="Item Type",
        description="Select a type: World, Shop Item, or Event (maps in voting/winners)",
        items=[
            ('world', 'World', ''),
            ('shop_item', 'Shop', ''),
            ('event', 'Event', 'Playable event maps (voting and winners)')
        ],
        default='world',
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
        default='newest',
    )

    bpy.types.Scene.package_search_query = bpy.props.StringProperty(
        name="Search Query",
        description="Filter packages by name or author",
        default="",
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

    bpy.types.Scene.comment_text = bpy.props.StringProperty(name="Comment", default="")


    #get this out of init and put where it belongs!!
    bpy.types.Scene.download_progress = bpy.props.FloatProperty(
        name="Download Progress",
        description="Download progress (0.0 to 1.0)",
        default=0.0,
        min=0.0,
        max=1.0
    )
    ##is this even used anywhere???? maybe display ops?
    bpy.types.Scene.last_filter_signature = bpy.props.StringProperty(
        name="Last Filter Signature",
        description="Filters that produced the cached package list",
        default=""
    )

    # Register all addon classes.
    for cls in classes:
        bpy.utils.register_class(cls)

    # Preload in-memory thumbnails from the disk cache.
    preload_in_memory_thumbnails()

def unregister():
    # Remove all properties from bpy.types.Scene.
    del bpy.types.Scene.package_item_type
    del bpy.types.Scene.package_sort_by
    del bpy.types.Scene.package_search_query
    del bpy.types.Scene.my_addon_data
    del bpy.types.Scene.current_thumbnail_page
    del bpy.types.Scene.total_thumbnail_pages
    del bpy.types.Scene.download_code
    del bpy.types.Scene.comment_text
    del bpy.types.Scene.download_progress
    del bpy.types.Scene.last_filter_signature

    bpy.utils.unregister_class(MyAddonComment)
    bpy.utils.unregister_class(PackageProps)

    # Unregister addon classes in reverse order.
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    # Remove the persistent handler if it was registered.
    if on_blend_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_blend_load)
    unregister_ui_props()
    unregister_event_props()

if __name__ == "__main__":
    register()
