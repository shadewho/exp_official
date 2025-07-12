# File: Exp_Game/exp_performance.py

import bpy
import mathutils
from bpy.types import PropertyGroup, UIList, Operator
from bpy.props import (
    StringProperty,
    BoolProperty,
    FloatProperty,
    PointerProperty,
    CollectionProperty,
    IntProperty,
    EnumProperty,
)

# ─── 0) Snapshot / Restore Helpers ─────────────────────────────

def find_layer_collection(layer_coll, target_coll):
    """Recursively find the LayerCollection matching target_coll."""
    if layer_coll.collection == target_coll:
        return layer_coll
    for child in layer_coll.children:
        found = find_layer_collection(child, target_coll)
        if found:
            return found
    return None

def init_performance_state(operator, context):
    """
    Capture original hide_viewport flags and view_layer.exclude flags
    for all performance entries, so we can restore them later.
    """
    operator._perf_prev_states = {}
    scene = context.scene

    for entry in scene.performance_entries:
        key = entry.name
        state = {
            "objects": {},       # obj_name -> hide_viewport
            "collections": {},   # (view_layer.name, coll_name) -> exclude_flag
        }

        # Real target
        if entry.use_collection and entry.target_collection:
            coll = entry.target_collection
            # capture per-layer exclude
            for vl in scene.view_layers:
                lc = find_layer_collection(vl.layer_collection, coll)
                if lc:
                    state["collections"][(vl.name, coll.name)] = lc.exclude
            # if radial-hide, capture each object's hide_viewport
            if entry.trigger_type == 'RADIAL' and not entry.exclude_collection:
                for obj in coll.objects:
                    state["objects"][obj.name] = obj.hide_viewport

        elif entry.target_object:
            obj = entry.target_object
            state["objects"][obj.name] = obj.hide_viewport

        # Placeholder target
        if entry.has_placeholder:
            if entry.placeholder_use_collection and entry.placeholder_collection:
                plc = entry.placeholder_collection
                for vl in scene.view_layers:
                    lc = find_layer_collection(vl.layer_collection, plc)
                    if lc:
                        state["collections"][(vl.name, plc.name)] = lc.exclude
            elif entry.placeholder_object:
                obj = entry.placeholder_object
                state["objects"][obj.name] = obj.hide_viewport
                    # ─── NEW: capture & hide the trigger mesh ──────────────
        if entry.trigger_type == 'BOX' and entry.hide_trigger_mesh and entry.trigger_mesh:
            mesh = entry.trigger_mesh
            state["objects"][mesh.name] = mesh.hide_viewport
            mesh.hide_viewport = True


        operator._perf_prev_states[key] = state

def restore_performance_state(operator, context):
    """
    Revert hide_viewport and view_layer.exclude flags from snapshot.
    """
    scene = context.scene
    for entry in scene.performance_entries:
        key = entry.name
        state = operator._perf_prev_states.get(key)
        if not state:
            continue

        # restore objects (including trigger_mesh)
        for obj_name, hide_flag in state["objects"].items():
            obj = bpy.data.objects.get(obj_name)
            if obj:
                obj.hide_viewport = hide_flag

        # restore collection exclude
        for (vl_name, coll_name), excl_flag in state["collections"].items():
            vl = scene.view_layers.get(vl_name)
            if not vl:
                continue
            coll = bpy.data.collections.get(coll_name)
            if not coll:
                continue
            lc = find_layer_collection(vl.layer_collection, coll)
            if lc:
                lc.exclude = excl_flag


# ─── 1) Data ──────────────────────────────────────────────────

class PerformanceEntry(PropertyGroup):
    name: StringProperty(
        name="Name",
        default="New Cull Entry"
    )
    use_collection: BoolProperty(
        name="Use Collection",
        description="Cull applies to a collection if true, otherwise to a single object",
        default=False,
    )
    target_collection: PointerProperty(
        name="Collection",
        type=bpy.types.Collection,
    )
    target_object: PointerProperty(
        name="Object",
        type=bpy.types.Object,
    )
    exclude_collection: BoolProperty(
        name="Exclude Entire Collection",
        description=(
            "When culling a collection radially, exclude the whole thing "
            "from the view-layer instead of hiding each object individually"
        ),
        default=True,
    )
    cull_distance: FloatProperty(
        name="Cull Distance",
        description="Distance beyond which the real entry is hidden/excluded",
        default=10.0,
        min=0.0,
    )

    # ─── new trigger props ───────────────────
    trigger_type: EnumProperty(
        name="Trigger Type",
        description="Choose how to trigger culled objects",
        items=[
            ('RADIAL', 'Radial', 'Use radial distance from character'),
            ('BOX',    'Bounding Box', 'Use bounding box trigger mesh'),
        ],
        default='RADIAL'
    )
    trigger_mesh: PointerProperty(
        name="Trigger Mesh",
        description="Mesh object used as bounding box trigger",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH'
    )
    hide_trigger_mesh: BoolProperty(
        name="Hide Trigger Mesh on Start",
        description="Automatically hide this mesh when you launch the game modal",
        default=False,
    )

    # ─── placeholder props ───────────────────
    has_placeholder: BoolProperty(
        name="Use Placeholder",
        description="Show a cheaper proxy when the real entry is out of range",
        default=False,
    )
    placeholder_use_collection: BoolProperty(
        name="Placeholder is Collection",
        description="If true, placeholder_collection is used; otherwise placeholder_object",
        default=False,
    )
    placeholder_collection: PointerProperty(
        name="Placeholder Collection",
        type=bpy.types.Collection,
    )
    placeholder_object: PointerProperty(
        name="Placeholder Object",
        type=bpy.types.Object,
    )


# ─── 2) UI List ────────────────────────────────────────────────

class EXP_PERFORMANCE_UL_List(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "name", text="", emboss=False, icon='OUTLINER_OB_EMPTY')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.name)


# ─── 4) Operators ─────────────────────────────────────────────

class EXPLORATORY_OT_AddPerformanceEntry(Operator):
    bl_idname = "exploratory.add_performance_entry"
    bl_label  = "Add Cull Entry"

    def execute(self, context):
        entry = context.scene.performance_entries.add()
        entry.name = f"Cull {len(context.scene.performance_entries)}"
        context.scene.performance_entries_index = len(context.scene.performance_entries) - 1
        return {'FINISHED'}

class EXPLORATORY_OT_RemovePerformanceEntry(Operator):
    bl_idname = "exploratory.remove_performance_entry"
    bl_label  = "Remove Cull Entry"

    index: IntProperty()

    def execute(self, context):
        scene = context.scene
        if 0 <= self.index < len(scene.performance_entries):
            scene.performance_entries.remove(self.index)
            scene.performance_entries_index = min(self.index, len(scene.performance_entries) - 1)
        return {'FINISHED'}


# ─── 5) Per-Frame Culling ──────────────────────────────────────

def update_performance_culling(operator, context):
    """
    Called each frame in ExpModal.modal():
      - hides/excludes the real entry when out-of-range (radial or box)
      - shows the placeholder when out-of-range
    """
    scene = context.scene
    ref = scene.target_armature
    if not ref:
        return
    ref_loc = ref.matrix_world.translation

    for entry in scene.performance_entries:
        # determine whether the real entry is “in” the trigger
        real_in = False

        if entry.trigger_type == 'RADIAL':
            dist = entry.cull_distance
            if entry.use_collection and entry.target_collection:
                # any object in collection within distance?
                real_in = any((obj.matrix_world.translation - ref_loc).length <= dist
                              for obj in entry.target_collection.objects)
            elif entry.target_object:
                d = (entry.target_object.matrix_world.translation - ref_loc).length
                real_in = (d <= dist)
        else:  # BOX trigger
            mesh = entry.trigger_mesh
            if mesh and mesh.type == 'MESH':
                # compute world-space AABB
                world_verts = [mesh.matrix_world @ mathutils.Vector(c) for c in mesh.bound_box]
                min_v = mathutils.Vector((min(v[i] for v in world_verts) for i in range(3)))
                max_v = mathutils.Vector((max(v[i] for v in world_verts) for i in range(3)))
                p = ref_loc
                real_in = (min_v.x <= p.x <= max_v.x and
                           min_v.y <= p.y <= max_v.y and
                           min_v.z <= p.z <= max_v.z)
            else:
                real_in = False

        # ─── apply to the “real” target ───────────────────────
        if entry.use_collection and entry.target_collection:
            coll = entry.target_collection
            if entry.trigger_type == 'BOX' or entry.exclude_collection:
                # hide/unhide whole collection via view-layer exclude
                for vl in scene.view_layers:
                    lc = find_layer_collection(vl.layer_collection, coll)
                    if lc:
                        lc.exclude = not real_in
            else:
                # radial + per-object hide
                for obj in coll.objects:
                    obj.hide_viewport = (obj.matrix_world.translation - ref_loc).length > entry.cull_distance

        elif entry.target_object:
            if entry.trigger_type == 'BOX':
                entry.target_object.hide_viewport = not real_in
            else:
                d = (entry.target_object.matrix_world.translation - ref_loc).length
                entry.target_object.hide_viewport = (d > entry.cull_distance)

        # ─── apply to the placeholder ───────────────────────
        if entry.has_placeholder:
            if entry.placeholder_use_collection and entry.placeholder_collection:
                plc = entry.placeholder_collection
                for vl in scene.view_layers:
                    lc = find_layer_collection(vl.layer_collection, plc)
                    if lc:
                        # show placeholder when real_in is False
                        lc.exclude = real_in
            elif entry.placeholder_object:
                # hide placeholder when real_in is True
                entry.placeholder_object.hide_viewport = real_in


# ─── 6) Registration ───────────────────────────────────────────

classes = (
    PerformanceEntry,
    EXP_PERFORMANCE_UL_List,
    EXPLORATORY_OT_AddPerformanceEntry,
    EXPLORATORY_OT_RemovePerformanceEntry,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.performance_entries = CollectionProperty(type=PerformanceEntry)
    bpy.types.Scene.performance_entries_index = IntProperty(default=0)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.performance_entries
    del bpy.types.Scene.performance_entries_index
