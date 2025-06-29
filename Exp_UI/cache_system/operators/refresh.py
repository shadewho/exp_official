#Exploratory/Exp_UI/cache_system/refresh.py
import bpy

class REFRESH_FILTERS_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.refresh_filters"
    bl_label = "Refresh Filters (Threaded)"
    
    def execute(self, context):
        scene = context.scene
        
        # 1) Clear old data/caches, remove old thumbnails
        if hasattr(bpy.types.Scene, "fetched_packages_data"):
            bpy.types.Scene.fetched_packages_data.clear()
        if hasattr(bpy.types.Scene, "all_pages_data"):
            bpy.types.Scene.all_pages_data.clear()
        # clear on-disk thumbnails and GPU images if you like…

        # 2) Start fresh on page 1, show spinner
        scene.current_thumbnail_page = 1
        scene.show_loading_image = True
        
        # 3) Use your threaded fetch operator 
        #    so we get progressive loading, page count, arrow logic, etc.
        bpy.ops.webapp.fetch_page('EXEC_DEFAULT', page_number=1)
        
        self.report({'INFO'}, "Threaded refresh started.")
        return {'FINISHED'}
