#Exploratory/Exp_UI/cache_system/operators/metadata.py
import bpy

from ..download_helpers import background_fetch_metadata
from ..manager import cache_manager, ensure_package_data

class PRELOAD_METADATA_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.preload_metadata"
    bl_label = "Preload Metadata"
    bl_description = (
        "Starts background threads to preload metadata (e.g. JSON and images) "
        "for all packages that aren’t already cached."
    )

    def execute(self, context):
        scene = context.scene

        # If there's no package data at all, prime the cache for the current type
        if not cache_manager.get_package_data():
            # PASS the file_type (and you can pass a second arg for limit if you like)
            if not ensure_package_data(scene.package_item_type):
                self.report(
                    {'WARNING'},
                    f"No package data available to preload for “{scene.package_item_type}”."
                )
                return {'CANCELLED'}

        total_preloaded = 0
        for page, packages in cache_manager.get_package_data().items():
            for pkg in packages:
                pkg_id = pkg.get("file_id")
                if pkg_id is None:
                    continue
                # Only spawn a background fetch if we don’t already have metadata
                if cache_manager.get_metadata(pkg_id) is None:
                    background_fetch_metadata(pkg_id)
                    total_preloaded += 1

        self.report({'INFO'}, f"Started preloading metadata for {total_preloaded} packages.")
        return {'FINISHED'}