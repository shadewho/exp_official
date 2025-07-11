# Exploratory/Exp_Game/exp_performance.py

import bpy
import mathutils

from bpy.types import PropertyGroup, Operator, UIList

# ─── 1) Data ──────────────────────────────────────────────────

class PerformanceEntry(PropertyGroup):
    name: bpy.props.StringProperty(
        name="Name",
        default="Entry"
    )
    use_collection: bpy.props.BoolProperty(
        name="Cull a Collection",
        default=False,
        description="If true, cull entire collection; otherwise a single object"
    )
    target_object: bpy.props.PointerProperty(
        name="Object",
        type=bpy.types.Object,
        description="Object to cull beyond distance"
    )
    target_collection: bpy.props.PointerProperty(
        name="Collection",
        type=bpy.types.Collection,
        description="Collection to cull beyond distance"
    )
    cull_distance: bpy.props.FloatProperty(
        name="Cull Distance",
        default=10.0,
        min=0.0,
        description="Distance from character beyond which to hide"
    )

# ─── 2) UIList & Operators ─────────────────────────────────────

class EXP_PERFORMANCE_UL_List(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.name)

class EXP_OT_AddPerformanceEntry(Operator):
    bl_idname = "exploratory.add_performance_entry"
    bl_label  = "Add Entry"
    def execute(self, context):
        scene = context.scene
        item = scene.performance_entries.add()
        item.name = f"Entry_{len(scene.performance_entries)}"
        scene.performance_entries_index = len(scene.performance_entries) - 1
        return {'FINISHED'}

class EXP_OT_RemovePerformanceEntry(Operator):
    bl_idname = "exploratory.remove_performance_entry"
    bl_label  = "Remove Entry"
    index: bpy.props.IntProperty()
    def execute(self, context):
        scene = context.scene
        if 0 <= self.index < len(scene.performance_entries):
            scene.performance_entries.remove(self.index)
            scene.performance_entries_index = min(self.index, len(scene.performance_entries)-1)
        return {'FINISHED'}

# ─── 3) Runtime Helpers ─────────────────────────────────────────

def init_performance_state(modal_op, context):
    """Record each entry’s original hide_viewport state."""
    modal_op._perf_orig = {}
    for entry in context.scene.performance_entries:
        if entry.use_collection and entry.target_collection:
            for obj in entry.target_collection.objects:
                modal_op._perf_orig[obj.name] = obj.hide_viewport
        elif entry.target_object:
            modal_op._perf_orig[entry.target_object.name] = entry.target_object.hide_viewport

def update_performance_culling(modal_op, context):
    """Hide/unhide each entry based on distance to modal_op.target_object."""
    char = modal_op.target_object
    if not char:
        return

    char_loc = char.matrix_world.translation

    for entry in context.scene.performance_entries:
        def apply(obj):
            dist = (obj.matrix_world.translation - char_loc).length
            hide = dist > entry.cull_distance
            # restore original when back in range
            obj.hide_viewport = hide

        if entry.use_collection and entry.target_collection:
            for obj in entry.target_collection.objects:
                apply(obj)
        elif entry.target_object:
            apply(entry.target_object)

# ─── 4) Registration ────────────────────────────────────────────

classes = (
    PerformanceEntry,
    EXP_PERFORMANCE_UL_List,
    EXP_OT_AddPerformanceEntry,
    EXP_OT_RemovePerformanceEntry,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.performance_entries = bpy.props.CollectionProperty(type=PerformanceEntry)
    bpy.types.Scene.performance_entries_index = bpy.props.IntProperty(default=0)

def unregister():
    del bpy.types.Scene.performance_entries
    del bpy.types.Scene.performance_entries_index
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
