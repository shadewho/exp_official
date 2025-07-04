# File: Exploratory/Exp_UI/cache_system/operators/metadata.py

import bpy
from ..download_helpers import background_fetch_metadata
from ..manager import cache_manager
from ...internet.helpers import is_internet_available

class PRELOAD_METADATA_OT_WebApp(bpy.types.Operator):
    bl_idname = "webapp.preload_metadata"
    bl_label = "Preload Metadata"
    bl_description = (
        "Starts background threads to preload metadata (e.g. JSON and thumbnails) "
        "for all packages that aren’t already cached."
    )
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        # Don’t run in GAME mode
        if scene.ui_current_mode == 'GAME':
            return False
        # Allow if we already have packages to refresh, or if we need to prime and have internet
        return bool(cache_manager.get_package_data()) or is_internet_available()

    def execute(self, context):
        scene = context.scene

        # 1) Prime cache if empty
        packages = cache_manager.get_package_data()
        if not packages:
            success = cache_manager.ensure_package_data(scene.package_item_type)
            if not success:
                self.report(
                    {'WARNING'},
                    f"No package data to preload for '{scene.package_item_type}'."
                )
                return {'CANCELLED'}
            packages = cache_manager.get_package_data()

        # 2) Queue any missing metadata
        total_queued = 0
        for pkg_list in packages.values():
            for pkg in pkg_list:
                pkg_id = pkg.get("file_id")
                if pkg_id is None:
                    continue
                if cache_manager.get_metadata(pkg_id) is None:
                    background_fetch_metadata(pkg_id)
                    total_queued += 1

        self.report({'INFO'}, f"Queued metadata preload for {total_queued} package(s).")
        return {'FINISHED'}
