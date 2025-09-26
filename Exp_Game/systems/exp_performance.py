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

# Use the real-time game clock you already run each TIMER tick
from ..props_and_utils.exp_time import get_game_time
from .exp_threads import compute_cull_batch
# ──────────────────────────────────────────────────────────────────────────────
# Tunables
# ──────────────────────────────────────────────────────────────────────────────
CULL_UPDATE_HZ = 1.0                 # frequency for culling decisions (~5 Hz)
CULL_HYSTERESIS_RATIO = 0.0         # 10% of cull_distance
CULL_HYSTERESIS_MIN = 0.0            # at least 1 meter of margin
MAX_PROP_WRITES_PER_TICK = 100       # safety budget for per-object hide writes


# ─── 0) Snapshot / Restore Helpers ─────────────────────────────

def _is_force_hidden_during_game(operator, obj):
    """
    Returns True if obj should remain hidden for the whole game session
    because the modal marked it as 'hide during game'.
    """
    names = getattr(operator, "_force_hide_names", None)
    return bool(obj and names and obj.name in names)

def rearm_performance_after_reset(operator, context):
    """
    After a ResetGame, restore 'game-time' visibility rules:
      - Re-hide trigger meshes that are marked 'Hide Mesh During Game'
      - Rebuild runtime caches for active view layer
      - Force next culling pass to apply regardless of previous state
    """
    scene = context.scene

    # Re-hide trigger meshes that should be hidden during gameplay
    for entry in scene.performance_entries:
        if entry.trigger_type == 'BOX' and entry.hide_trigger_mesh and entry.trigger_mesh:
            try:
                entry.trigger_mesh.hide_viewport = True
            except Exception:
                pass

    # Rebuild caches and force immediate evaluation
    _build_perf_runtime_cache(operator, context)
    if hasattr(operator, "_perf_runtime"):
        for rec in operator._perf_runtime.values():
            rec["last_in"] = None      # force "state changed" behavior
            rec["next_update"] = 0.0   # run on the very next tick
            rec["scan_idx"] = 0

def find_layer_collection(layer_coll, target_coll):
    """Recursively find the LayerCollection matching target_coll."""
    if layer_coll.collection == target_coll:
        return layer_coll
    for child in layer_coll.children:
        found = find_layer_collection(child, target_coll)
        if found:
            return found
    return None


def iter_collection_and_descendants(root_coll):
    """Yield root_coll and every child collection (depth-first)."""
    if not root_coll:
        return
    yield root_coll
    for child in root_coll.children:
        yield from iter_collection_and_descendants(child)


def objects_in_collection(coll, recursive=False):
    """Return a set of objects in 'coll'. If recursive=True, include children."""
    objs = set(coll.objects) if coll else set()
    if recursive and coll:
        for child in coll.children:
            objs |= objects_in_collection(child, recursive=True)
    return objs


def _build_perf_runtime_cache(operator, context):
    """
    Build per-entry runtime caches so we don't traverse the scene every frame.

    Structure:
    operator._perf_runtime: {
        entry_ptr (int): {
            "layer_colls":   [LayerCollection, ...],  # active view layer only
            "obj_list":      [Object, ...],           # flattened objects to test distance
            "pl_layer_coll": LayerCollection | None,  # placeholder collection (active layer)
            "pl_obj":        Object | None,           # placeholder object
            "last_in":       None | bool,             # last "in-range" decision
            "next_update":   float,                   # next game_time to evaluate
            "scan_idx":      int,                     # round-robin cursor for per-object writes
        },
        ...
    }
    """
    operator._perf_runtime = {}
    scene = context.scene
    active_vl = context.view_layer

    for entry in scene.performance_entries:
        rec = {
            "layer_colls": [],
            "obj_list": [],
            "pl_layer_coll": None,
            "pl_obj": None,
            "last_in": None,
            "next_update": 0.0,
            "scan_idx": 0,  # NEW: distribute per-object writes fairly
        }

        # Real target cache
        if entry.use_collection and entry.target_collection:
            root = entry.target_collection
            colls = list(iter_collection_and_descendants(root)) if entry.cascade_collections else [root]

            # Resolve LayerCollections only for the ACTIVE view layer
            for coll in colls:
                lc = find_layer_collection(active_vl.layer_collection, coll)
                if lc:
                    rec["layer_colls"].append(lc)

            # Flatten the objects once
            for coll in colls:
                rec["obj_list"].extend(list(objects_in_collection(coll, recursive=False)))

        elif entry.target_object:
            rec["obj_list"].append(entry.target_object)

        # Placeholder cache (active view layer only)
        if entry.has_placeholder:
            if entry.placeholder_use_collection and entry.placeholder_collection:
                rec["pl_layer_coll"] = find_layer_collection(active_vl.layer_collection, entry.placeholder_collection)
            elif entry.placeholder_object:
                rec["pl_obj"] = entry.placeholder_object

        operator._perf_runtime[entry.as_pointer()] = rec


def init_performance_state(operator, context):
    """
    Capture original hide_viewport flags and view_layer.exclude flags
    for all performance entries, so we can restore them later.
    Also builds a runtime cache for fast per-tick decisions.
    """
    operator._perf_prev_states = {}
    scene = context.scene

    for entry in scene.performance_entries:
        key = entry.name
        state = {
            "objects": {},       # obj_name -> hide_viewport
            "collections": {},   # (view_layer.name, coll_name) -> exclude_flag
        }

        # Real target snapshot
        if entry.use_collection and entry.target_collection:
            root_coll = entry.target_collection
            colls_to_track = list(iter_collection_and_descendants(root_coll)) if entry.cascade_collections else [root_coll]

            # capture per-layer exclude for all relevant collections (ALL view layers, for correct restore)
            for coll in colls_to_track:
                for vl in scene.view_layers:
                    lc = find_layer_collection(vl.layer_collection, coll)
                    if lc:
                        state["collections"][(vl.name, coll.name)] = lc.exclude

            # if radial per-object culling, capture each object's hide_viewport
            if entry.trigger_type == 'RADIAL' and not entry.exclude_collection:
                objs = set()
                for coll in colls_to_track:
                    objs |= objects_in_collection(coll, recursive=False)
                for obj in objs:
                    state["objects"][obj.name] = obj.hide_viewport

        elif entry.target_object:
            obj = entry.target_object
            state["objects"][obj.name] = obj.hide_viewport

        # Placeholder snapshot
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

        # Trigger mesh snapshot & early hide if requested
        if entry.trigger_type == 'BOX' and entry.hide_trigger_mesh and entry.trigger_mesh:
            mesh = entry.trigger_mesh
            state["objects"][mesh.name] = mesh.hide_viewport
            mesh.hide_viewport = True

        operator._perf_prev_states[key] = state

    # Build runtime caches for fast updates on the ACTIVE view layer
    _build_perf_runtime_cache(operator, context)


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

        # restore objects (including trigger_mesh and placeholder object)
        for obj_name, hide_flag in state["objects"].items():
            obj = bpy.data.objects.get(obj_name)
            if obj:
                obj.hide_viewport = hide_flag

        # restore collection exclude (across all view layers)
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

    # Drop runtime caches; they will be rebuilt next time
    if hasattr(operator, "_perf_runtime"):
        operator._perf_runtime.clear()


# ─── 1) Data: Properties ──────────────────────────────────────────────────

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
            "When enabled, hides the whole collection at once "
            "instead of culling its objects one by one"
        ),
        default=True,
    )
    cull_distance: FloatProperty(
        name="Cull Distance",
        description="Distance beyond which the real entry is hidden/excluded",
        default=10.0,
        min=0.0,
    )

    # Trigger mode
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

    # Placeholder props
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
    cascade_collections: BoolProperty(
        name="Cascade Into Child Collections",
        description=(
            "If enabled and a collection is the target, culling also applies to "
            "all nested child collections (and their objects)"
        ),
        default=False,
    )


# ─── 2) UI List ────────────────────────────────────────────────

class EXP_PERFORMANCE_UL_List(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "name", text="", emboss=False, icon='VIEW_PERSPECTIVE')
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


# ─── 5) Per-Frame Culling (Throttled + Cached + Hysteresis) ───

def update_performance_culling(operator, context):
    """
    Called each frame in ExpModal.modal().

    Optimizations:
      - Throttled per-entry to CULL_UPDATE_HZ using get_game_time()
      - Hysteresis around cull_distance to prevent flapping (entry-level)
      - Only writes properties when the value actually changes
      - Operates on ACTIVE view layer only
      - Uses caches built at init to avoid per-tick recursion/aggregation
      - Per-object mode ALWAYS runs (even if entry-level in_range didn't change)
    """
    scene = context.scene
    ref = scene.target_armature
    if not ref:
        return
    ref_loc = ref.matrix_world.translation

    # Ensure caches exist
    if not hasattr(operator, "_perf_runtime"):
        _build_perf_runtime_cache(operator, context)

    game_t = get_game_time()

    for entry in scene.performance_entries:
        key = entry.as_pointer()
        rec = operator._perf_runtime.get(key)
        if not rec:
            _build_perf_runtime_cache(operator, context)
            rec = operator._perf_runtime.get(key)
            if not rec:
                continue

        # Throttle per entry
        if game_t < rec["next_update"]:
            continue
        rec["next_update"] = game_t + (1.0 / CULL_UPDATE_HZ)

        # Determine per-object vs collection-exclude mode
        per_object_mode = (
            entry.use_collection and entry.target_collection and
            entry.trigger_type == 'RADIAL' and
            not entry.exclude_collection
        )

        # Entry-level in_range (used for placeholders & collection-exclude)
        base = entry.cull_distance if entry.trigger_type == 'RADIAL' else 0.0
        margin = max(CULL_HYSTERESIS_MIN, base * CULL_HYSTERESIS_RATIO)

        if entry.trigger_type == 'RADIAL':
            objs = rec["obj_list"]
            if objs:
                if rec["last_in"] is None:
                    threshold = base
                elif rec["last_in"]:
                    threshold = base + margin   # need to be clearly outside to flip false
                else:
                    threshold = max(0.0, base - margin)  # clearly inside to flip true

                in_range = any((o.matrix_world.translation - ref_loc).length <= threshold for o in objs)
            else:
                in_range = False
        else:
            # BOX trigger
            mesh = entry.trigger_mesh
            if mesh and mesh.type == 'MESH':
                world_verts = [mesh.matrix_world @ mathutils.Vector(c) for c in mesh.bound_box]
                min_v = mathutils.Vector((min(v[i] for v in world_verts) for i in range(3)))
                max_v = mathutils.Vector((max(v[i] for v in world_verts) for i in range(3)))
                p = ref_loc
                in_range = (min_v.x <= p.x <= max_v.x and
                            min_v.y <= p.y <= max_v.y and
                            min_v.z <= p.z <= max_v.z)
            else:
                in_range = False

        # ─────────────────────────────────────────────────────────
        # A) COLLECTION-EXCLUDE / BOX / SINGLE-OBJECT modes
        #    -> only act when entry-level state changes
        # ─────────────────────────────────────────────────────────
        if not per_object_mode:
            if rec["last_in"] is None or in_range != rec["last_in"]:
                if entry.use_collection and entry.target_collection:
                    if entry.trigger_type == 'BOX' or entry.exclude_collection:
                        desired_excl = not in_range
                        for lc in rec["layer_colls"]:
                            if lc and lc.exclude != desired_excl:
                                lc.exclude = desired_excl
                    elif entry.target_object:
                        if _is_force_hidden_during_game(operator, entry.target_object):
                            desired_hidden = True
                        else:
                            if entry.trigger_type == 'BOX':
                                desired_hidden = not in_range
                            else:
                                d = (entry.target_object.matrix_world.translation - ref_loc).length
                                desired_hidden = (d > entry.cull_distance)
                        if entry.target_object.hide_viewport != desired_hidden:
                            entry.target_object.hide_viewport = desired_hidden

        # ─────────────────────────────────────────────────────────
        # B) PER-OBJECT mode (RADIAL + not exclude_collection)
        #    -> ALWAYS run (throttled), regardless of in_range changes
        # ─────────────────────────────────────────────────────────
        else:
            objs = rec["obj_list"]
            if objs:
                # If the thread engine is available, submit a coalesced batch job.
                if hasattr(operator, "_thread_eng") and operator._thread_eng:
                    # Snapshot names & positions on the main thread (read-only)
                    names = [o.name for o in objs]
                    poss  = [tuple(o.matrix_world.translation) for o in objs]

                    # Round-robin cursor and batch size (respect your write budget)
                    start_idx = rec.get("scan_idx", 0)
                    max_batch = min(MAX_PROP_WRITES_PER_TICK, len(names))

                    # Threshold and identity for this entry
                    threshold = float(entry.cull_distance)
                    entry_ptr = entry.as_pointer()
                    job_key   = f"cull:{entry_ptr}"

                    # Per-entry version to coalesce newer work
                    ver = rec.get("cull_ver", 0) + 1
                    rec["cull_ver"] = ver

                    # Submit/replace latest work for this entry.
                    operator._thread_eng.submit_latest(
                        key=job_key,
                        version=ver,
                        fn=compute_cull_batch,             # runs in worker (no bpy)
                        entry_ptr=entry_ptr,
                        obj_names=names,
                        obj_positions=poss,
                        ref_loc=tuple(ref_loc),            # (x,y,z)
                        thresh=threshold,
                        start=start_idx,
                        max_count=max_batch,
                    )

                    # Note: we advance rec["scan_idx"] when the result is applied
                    # in ExpModal.modal() after polling thread results.
                else:
                    # ── Fallback (no thread engine): original synchronous path ──
                    n = len(objs)
                    start = rec.get("scan_idx", 0)
                    i = 0
                    changes = 0
                    thresh = entry.cull_distance  # object-level check uses base threshold

                    # Round-robin across the list to avoid starvation
                    while i < n and changes < MAX_PROP_WRITES_PER_TICK:
                        idx = (start + i) % n
                        o = objs[idx]

                        if _is_force_hidden_during_game(operator, o):
                            desired_hidden = True
                        else:
                            far = (o.matrix_world.translation - ref_loc).length > thresh
                            desired_hidden = far

                        if o.hide_viewport != desired_hidden:
                            o.hide_viewport = desired_hidden
                            changes += 1
                        i += 1

                    rec["scan_idx"] = (start + i) % n  # advance cursor even if no changes


        # ─────────────────────────────────────────────────────────
        # C) Placeholder toggling follows entry-level in_range
        # ─────────────────────────────────────────────────────────
        if entry.has_placeholder:
            if rec["pl_layer_coll"]:
                desired_excl = in_range  # exclude placeholder when real visible
                if rec["pl_layer_coll"].exclude != desired_excl:
                    rec["pl_layer_coll"].exclude = desired_excl
            elif rec["pl_obj"]:
                desired_hidden = in_range  # hide placeholder when real visible
                if rec["pl_obj"].hide_viewport != desired_hidden:
                    rec["pl_obj"].hide_viewport = desired_hidden

        # Save new state
        rec["last_in"] = in_range



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
