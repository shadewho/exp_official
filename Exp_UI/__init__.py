#Exploratory/Exp_UI/__init__.py


# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

import bpy
import time
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
    POPUP_SOCIAL_DETAILS_OT, COMMENT_PACKAGE_INLINE_OT_WebApp
)
from .packages.operators import DOWNLOAD_CODE_OT_File

from .cache_system.operators.clear import (CLEAR_ALL_DATA_OT_WebApp, CLEAR_THUMBNAILS_ONLY_OT_WebApp)
from .cache_system.operators.refresh import REFRESH_FILTERS_OT_WebApp
from .cache_system.manager import CacheManager

# Import functions from our cache module.
from .cache_system.persistence import clear_image_datablocks
from .cache_system.preload import start_cache_worker, stop_cache_worker
from .auth.helpers import auto_refresh_usage
from .download_and_explore.cleanup import cleanup_world_downloads
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

from .open_addon_prefs import OPEN_ADDON_PREFS_OT

#test db
# from .cache_system.db import (
#     DB_INSPECT_OT_ShowCacheDB,
#     VIEW3D_PT_CacheDB,)

# List all classes that will be registered.
classes = (
    LOGIN_OT_WebApp,
    LOGOUT_OT_WebApp,
    DOWNLOAD_CODE_OT_File,
    LIKE_PACKAGE_OT_WebApp,
    COMMENT_PACKAGE_OT_WebApp,
    COMMENT_PACKAGE_INLINE_OT_WebApp,
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
    POPUP_SOCIAL_DETAILS_OT,
    VOTE_MAP_OT_WebApp,
    VIEW3D_PT_SettingsAndUpdate,
    OPEN_DOCS_OT,
    OPEN_ADDON_PREFS_OT,

    # #test db
    # DB_INSPECT_OT_ShowCacheDB,
    # VIEW3D_PT_CacheDB
)
from .packages.properties import MyAddonComment, PackageProps
from .cache_system.db import init_db
from .events.properties import register as register_event_props, unregister as unregister_event_props
from .events.utilities import on_package_item_type_update
from .interface.drawing.fonts import reset_font 
from .interface.operators.fetch import fetch_page_queue

@persistent
def on_blend_load(dummy):
    """
    Runs after every successful File → Open.
    • Purges GPU images safely
    • Invalidates cached font so the next draw reloads it
    • Clears stale background‑fetch tasks
    • Flags the modal draw‑loop to rebuild its data list
    """
    # Force Browse mode on every file-open
    try:
        bpy.context.scene.ui_current_mode = 'BROWSE'
    except Exception as e:
        print(f"[WARN] Could not set UI mode to Browse on load: {e}")

    # 1. image/texture cleanup – wrapped so an error can’t abort the rest
    try:
        clear_image_datablocks()
    except Exception as e:
        print(f"[WARN] clear_image_datablocks failed: {e!s}")

    # 2. make sure font‑ID is re‑loaded in the new file
    reset_font()

    # 3. drop any queued fetch tasks that still point to the old RNA
    try:
        fetch_page_queue.queue.clear()
    except Exception:
        pass

    # 4. empty the old draw data & request a fresh build
    if hasattr(bpy.types.Scene, "gpu_image_buttons_data"):
        bpy.types.Scene.gpu_image_buttons_data.clear()

    bpy.types.Scene.package_ui_dirty = True   # modal operator sees this
    print("[INFO] New blend file loaded – full UI reset.")

def connectivity_check_timer():
    if not is_internet_available():
        # Log out and disable UI
        clear_token()
        if bpy.context.scene.get("my_addon_data"):
            bpy.context.scene.my_addon_data.is_from_webapp = False
        bpy.context.scene.ui_current_mode = "GAME"
        print ("GAME GAME GAME GAME GAME GAME GAME GAME GAME GAME GAME GAME GAME GAME GAME GAME ")
        print("[INFO] No internet connection detected. User logged out and UI disabled.")
    return 10.0  # Check every 10 seconds

def on_sort_by_update(self, context):
    # Whenever the user explicitly picks “Random”:
    if context.scene.package_sort_by == 'random':
        # 1️⃣  Bump the signature to something new
        context.scene.last_filter_signature = str(time.time())
        # 2️⃣  Reset to page 1
        context.scene.current_thumbnail_page = 1
        # 3️⃣  Force a fresh fetch of that first page
        try:
            bpy.ops.webapp.fetch_page('EXEC_DEFAULT', page_number=1)
        except Exception as e:
            print(f"[ERROR] failed to refresh random page: {e}")
# --- Registration ---
def register():

    #Clear out the World Downloads folder every time we register
    try:
        cleanup_world_downloads() 
    except Exception as e:
        print(f"[INFO] Failed to clear downloads folder at register: {e}")

    init_db()

    register_event_props()

    if on_blend_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_blend_load)
    

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
            ('world',     'World',     ''),
            ('shop_item', 'Shop',      ''),
            ('event',     'Event',     'Playable event maps (voting and winners)'),
        ],
        default='world',
        update=on_package_item_type_update,   # ← hook in your updater here
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
        update=on_sort_by_update,
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

    # ── finally start the background cache thread ───────────────────
    start_cache_worker()


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
    stop_cache_worker()

if __name__ == "__main__":
    register()
