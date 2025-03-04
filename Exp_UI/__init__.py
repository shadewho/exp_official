# __init__.py
bl_info = {
    "name": "UI_TEST_ADDON_3",
    "blender": (4, 2, 0),
    "author": "Your Name",
    "description": "A simple add-on for UI EXP purposes in Blender.",
    "version": (1, 0, 2),
}

from .helper_functions import on_filter_changed

from .backend import LOGIN_OT_WebApp, LOGOUT_OT_WebApp, DOWNLOAD_CODE_OT_File
from .panel import (
    VIEW3D_PT_PackageDisplay_Login,
    VIEW3D_PT_PackageDisplay_FilterAndScene,
    VIEW3D_PT_PackageDisplay_CurrentItem,
    VIEW3D_PT_SubscriptionUsage
)

from .social_operators import (LIKE_PACKAGE_OT_WebApp, COMMENT_PACKAGE_OT_WebApp, 
                               OPEN_URL_OT_WebApp, EXPLORATORY_UL_Comments, REFRESH_USAGE_OT_WebApp,
                               POPUP_SOCIAL_DETAILS_OT
)
from .addon_data import MyAddonComment, MyAddonSceneProps
from bpy.props import PointerProperty
import bpy
from .image_button_UI.operators import (
    FETCH_PAGE_THREADED_OT_WebApp,
    PACKAGE_OT_Display,
    REMOVE_PACKAGE_OT_Display,
    APPLY_FILTERS_SHOWUI_OT

)

from .cache_memory_operators import (CLEAR_ALL_DATA_OT_WebApp,
                                     CLEAR_THUMBNAILS_ONLY_OT_WebApp, 
                                     REFRESH_FILTERS_OT_WebApp,
                                     PRELOAD_METADATA_OT_WebApp,
                                     preload_metadata_timer
)

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

def register():
    bpy.app.timers.register(preload_metadata_timer, first_interval=5.0)
    # 1) Register your comment sub-class
    bpy.utils.register_class(MyAddonComment)
    # 2) Register your main property group
    bpy.utils.register_class(MyAddonSceneProps)

    # 3) Attach the property group to bpy.types.Scene
    bpy.types.Scene.my_addon_data = PointerProperty(type=MyAddonSceneProps)

    # Add properties for login fields
    bpy.types.Scene.username = bpy.props.StringProperty(name="Username")
    bpy.types.Scene.password = bpy.props.StringProperty(name="Password", subtype='PASSWORD')

    # UI MODE PROPERTY
    bpy.types.Scene.ui_current_mode = bpy.props.EnumProperty(
        name="UI Current Mode",
        description="Which UI mode is currently shown",
        items=[
            ("BROWSE", "Browse Thumbnails", ""),
            ("DETAIL", "Item Detail View", "")
        ],
        default="BROWSE"  # Start in the thumbnail/browse mode
    )

    # Add property for toggling image display
    bpy.types.Scene.show_image_buttons = bpy.props.BoolProperty(
        name="Show Package Thumbnails",
        description="Toggle the display of package thumbnail buttons",
        default=False
    )
    
    # Property to store the selected thumbnail image path
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
        default='world',
        update=on_filter_changed  # Use our new callback here.
    )

    bpy.types.Scene.package_sort_by = bpy.props.EnumProperty(
        name="Sort By",
        description="Sorting Method",
        items=[
            ('newest', 'Newest', ''),
            ('oldest', 'Oldest', ''),
            ('popular', 'Popular', ''),
            ('random', 'Random', '')
        ],
        default='newest',
        update=on_filter_changed
    )

    bpy.types.Scene.package_search_query = bpy.props.StringProperty(
        name="Search Query",
        description="Filter packages by name or author",
        default="",
        update=on_filter_changed
    )

    #DOWNLOAD CODE
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

    
    # Register other add-on classes, properties, and handlers here

    # Register backend, panel, and social operators
    for cls in classes:
        bpy.utils.register_class(cls)

    # *** New Code: Preload in-memory thumbnails from disk cache ***
    from .image_button_UI.cache import preload_in_memory_thumbnails
    preload_in_memory_thumbnails()

def unregister():
    # Remove properties
    del bpy.types.Scene.username
    del bpy.types.Scene.password
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

    bpy.utils.unregister_class(MyAddonComment)
    bpy.utils.unregister_class(MyAddonSceneProps)

    # Unregister backend, panel, and social operators
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()