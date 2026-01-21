# Exploratory/Exp_Game/props_and_utils/exp_utility_store.py
import bpy
import time
from uuid import uuid4
from bpy.types import PropertyGroup


# ─────────────────────────────────────────────────────────
# Scene-level registry for Utility Data Nodes
# Core types: Float, Int, Bool, FloatVector, Object, ObjectCollection
# ─────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════
# FLOAT
# ═══════════════════════════════════════════════════════════

class UtilityFloatSlotPG(PropertyGroup):
    """Storage slot for a single float value."""
    uid: bpy.props.StringProperty(name="UID", default="")
    name: bpy.props.StringProperty(name="Name", default="Float")
    has_value: bpy.props.BoolProperty(name="Has Value", default=False)
    value: bpy.props.FloatProperty(name="Value", default=0.0)
    updated_at: bpy.props.FloatProperty(name="Updated At", default=0.0)


# ═══════════════════════════════════════════════════════════
# INTEGER
# ═══════════════════════════════════════════════════════════

class UtilityIntSlotPG(PropertyGroup):
    """Storage slot for a single integer value."""
    uid: bpy.props.StringProperty(name="UID", default="")
    name: bpy.props.StringProperty(name="Name", default="Integer")
    has_value: bpy.props.BoolProperty(name="Has Value", default=False)
    value: bpy.props.IntProperty(name="Value", default=0)
    updated_at: bpy.props.FloatProperty(name="Updated At", default=0.0)


# ═══════════════════════════════════════════════════════════
# BOOLEAN
# ═══════════════════════════════════════════════════════════

class UtilityBoolSlotPG(PropertyGroup):
    """Storage slot for a boolean value."""
    uid: bpy.props.StringProperty(name="UID", default="")
    name: bpy.props.StringProperty(name="Name", default="Boolean")
    has_value: bpy.props.BoolProperty(name="Has Value", default=False)
    value: bpy.props.BoolProperty(name="Value", default=False)
    updated_at: bpy.props.FloatProperty(name="Updated At", default=0.0)


# ═══════════════════════════════════════════════════════════
# OBJECT REFERENCE
# ═══════════════════════════════════════════════════════════

class UtilityObjectSlotPG(PropertyGroup):
    """Storage slot for an object reference (by name)."""
    uid: bpy.props.StringProperty(name="UID", default="")
    name: bpy.props.StringProperty(name="Name", default="Object")
    has_value: bpy.props.BoolProperty(name="Has Value", default=False)
    object_name: bpy.props.StringProperty(
        name="Object Name",
        default="",
        description="Name of the referenced object"
    )
    updated_at: bpy.props.FloatProperty(name="Updated At", default=0.0)


# ═══════════════════════════════════════════════════════════
# OBJECT COLLECTION REFERENCE
# ═══════════════════════════════════════════════════════════

class UtilityObjectCollectionSlotPG(PropertyGroup):
    """Storage slot for a collection reference (by name)."""
    uid: bpy.props.StringProperty(name="UID", default="")
    name: bpy.props.StringProperty(name="Name", default="Collection")
    has_value: bpy.props.BoolProperty(name="Has Value", default=False)
    collection_name: bpy.props.StringProperty(
        name="Collection Name",
        default="",
        description="Name of the referenced collection"
    )
    updated_at: bpy.props.FloatProperty(name="Updated At", default=0.0)


# ═══════════════════════════════════════════════════════════
# ACTION REFERENCE
# ═══════════════════════════════════════════════════════════

class UtilityActionSlotPG(PropertyGroup):
    """Storage slot for an action reference (by name)."""
    uid: bpy.props.StringProperty(name="UID", default="")
    name: bpy.props.StringProperty(name="Name", default="Action")
    has_value: bpy.props.BoolProperty(name="Has Value", default=False)
    action_name: bpy.props.StringProperty(
        name="Action Name",
        default="",
        description="Name of the referenced action"
    )
    updated_at: bpy.props.FloatProperty(name="Updated At", default=0.0)


# ═══════════════════════════════════════════════════════════
# FLOAT VECTOR (existing)
# ═══════════════════════════════════════════════════════════

class UtilityFloatVectorSlotPG(PropertyGroup):
    """
    A single, unique storage slot for a 3D float vector.
    Owned by exactly one CaptureFloatVector node via uid.
    """
    uid: bpy.props.StringProperty(
        name="UID",
        default="",
        description="Stable unique identifier for this slot (uuid4)"
    )
    name: bpy.props.StringProperty(
        name="Name",
        default="Capture Vector",
        description="Friendly name for this capture slot"
    )
    has_value: bpy.props.BoolProperty(
        name="Has Value",
        default=False,
        description="True once a value has been written at least once"
    )
    value: bpy.props.FloatVectorProperty(
        name="Value",
        size=3,
        subtype='TRANSLATION',
        default=(0.0, 0.0, 0.0),
        description="Stored 3D vector"
    )
    updated_at: bpy.props.FloatProperty(
        name="Updated At (sec)",
        default=0.0,
        description="Game time or wall time of last write (seconds)"
    )


# ─────────────────────────────────────────────────────────
# Internal helpers (generic)
# ─────────────────────────────────────────────────────────

def _ensure_uid(slot) -> None:
    if not getattr(slot, "uid", ""):
        slot.uid = str(uuid4())


def _get_collection(scn: bpy.types.Scene, attr_name: str):
    return getattr(scn, attr_name, None)


# ─────────────────────────────────────────────────────────
# PERF: UID Index Cache for O(1) slot lookups (replaces O(n) iteration)
# ─────────────────────────────────────────────────────────
# Structure: {scene_id: {attr_name: {uid: slot}}}
_uid_index_cache: dict = {}

def _get_uid_index(scn: bpy.types.Scene, attr_name: str) -> dict:
    """Get or build the UID -> slot index for a collection."""
    scene_id = id(scn)
    if scene_id not in _uid_index_cache:
        _uid_index_cache[scene_id] = {}

    scene_cache = _uid_index_cache[scene_id]
    if attr_name not in scene_cache:
        # Build the index
        coll = getattr(scn, attr_name, None)
        index = {}
        if coll:
            for slot in coll:
                uid = getattr(slot, "uid", "")
                if uid:
                    index[uid] = slot
        scene_cache[attr_name] = index

    return scene_cache[attr_name]


def _invalidate_uid_index(scn: bpy.types.Scene, attr_name: str):
    """Invalidate the UID index when slots are added/removed."""
    scene_id = id(scn)
    if scene_id in _uid_index_cache:
        _uid_index_cache[scene_id].pop(attr_name, None)


def _register_slot_in_index(scn: bpy.types.Scene, attr_name: str, slot):
    """Register a newly created slot in the index."""
    uid = getattr(slot, "uid", "")
    if not uid:
        return
    scene_id = id(scn)
    if scene_id not in _uid_index_cache:
        _uid_index_cache[scene_id] = {}
    if attr_name not in _uid_index_cache[scene_id]:
        # Index not built yet, will be built on first lookup
        return
    _uid_index_cache[scene_id][attr_name][uid] = slot


def clear_uid_index_cache():
    """Clear the entire UID index cache (call on scene change or game reset)."""
    global _uid_index_cache
    _uid_index_cache.clear()


def _find_by_uid(scn: bpy.types.Scene, attr_name: str, uid: str):
    """O(1) lookup using indexed cache instead of O(n) iteration."""
    if not uid:
        return None
    index = _get_uid_index(scn, attr_name)
    return index.get(uid)


def _slot_exists(attr_name: str, uid: str) -> bool:
    scn = bpy.context.scene
    return _find_by_uid(scn, attr_name, uid) is not None


# ═══════════════════════════════════════════════════════════
# FLOAT API
# ═══════════════════════════════════════════════════════════

def create_float_slot(scn: bpy.types.Scene, name: str = "Float") -> str:
    coll = _get_collection(scn, "utility_floats")
    if coll is None:
        return ""
    item = coll.add()
    item.name = name or "Float"
    _ensure_uid(item)
    item.has_value = False
    item.value = 0.0
    item.updated_at = 0.0
    # PERF: Register in index for O(1) lookup
    _register_slot_in_index(scn, "utility_floats", item)
    return item.uid


def set_float(uid: str, value: float, timestamp: float | None = None) -> bool:
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_floats", uid)
    if not slot:
        return False
    slot.value = float(value)
    slot.has_value = True
    slot.updated_at = float(timestamp) if timestamp is not None else time.perf_counter()
    return True


def get_float(uid: str):
    """Returns (has_value, value, updated_at)."""
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_floats", uid)
    if not slot:
        return (False, 0.0, 0.0)
    return (bool(slot.has_value), float(slot.value), float(slot.updated_at))


def float_slot_exists(uid: str) -> bool:
    return _slot_exists("utility_floats", uid)


# ═══════════════════════════════════════════════════════════
# INTEGER API
# ═══════════════════════════════════════════════════════════

def create_int_slot(scn: bpy.types.Scene, name: str = "Integer") -> str:
    coll = _get_collection(scn, "utility_ints")
    if coll is None:
        return ""
    item = coll.add()
    item.name = name or "Integer"
    _ensure_uid(item)
    item.has_value = False
    item.value = 0
    item.updated_at = 0.0
    # PERF: Register in index for O(1) lookup
    _register_slot_in_index(scn, "utility_ints", item)
    return item.uid


def set_int(uid: str, value: int, timestamp: float | None = None) -> bool:
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_ints", uid)
    if not slot:
        return False
    slot.value = int(value)
    slot.has_value = True
    slot.updated_at = float(timestamp) if timestamp is not None else time.perf_counter()
    return True


def get_int(uid: str):
    """Returns (has_value, value, updated_at)."""
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_ints", uid)
    if not slot:
        return (False, 0, 0.0)
    return (bool(slot.has_value), int(slot.value), float(slot.updated_at))


def int_slot_exists(uid: str) -> bool:
    return _slot_exists("utility_ints", uid)


# ═══════════════════════════════════════════════════════════
# BOOLEAN API
# ═══════════════════════════════════════════════════════════

def create_bool_slot(scn: bpy.types.Scene, name: str = "Boolean") -> str:
    coll = _get_collection(scn, "utility_bools")
    if coll is None:
        return ""
    item = coll.add()
    item.name = name or "Boolean"
    _ensure_uid(item)
    item.has_value = False
    item.value = False
    item.updated_at = 0.0
    # PERF: Register in index for O(1) lookup
    _register_slot_in_index(scn, "utility_bools", item)
    return item.uid


def set_bool(uid: str, value: bool, timestamp: float | None = None) -> bool:
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_bools", uid)
    if not slot:
        return False
    slot.value = bool(value)
    slot.has_value = True
    slot.updated_at = float(timestamp) if timestamp is not None else time.perf_counter()
    return True


def get_bool(uid: str):
    """Returns (has_value, value, updated_at)."""
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_bools", uid)
    if not slot:
        return (False, False, 0.0)
    return (bool(slot.has_value), bool(slot.value), float(slot.updated_at))


def bool_slot_exists(uid: str) -> bool:
    return _slot_exists("utility_bools", uid)


# ═══════════════════════════════════════════════════════════
# OBJECT REFERENCE API
# ═══════════════════════════════════════════════════════════

def create_object_slot(scn: bpy.types.Scene, name: str = "Object") -> str:
    coll = _get_collection(scn, "utility_objects")
    if coll is None:
        return ""
    item = coll.add()
    item.name = name or "Object"
    _ensure_uid(item)
    item.has_value = False
    item.object_name = ""
    item.updated_at = 0.0
    # PERF: Register in index for O(1) lookup
    _register_slot_in_index(scn, "utility_objects", item)
    return item.uid


def set_object(uid: str, obj, timestamp: float | None = None) -> bool:
    """Set object reference. Accepts bpy.types.Object or object name string."""
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_objects", uid)
    if not slot:
        return False
    if obj is None:
        slot.object_name = ""
        slot.has_value = False
    elif isinstance(obj, str):
        slot.object_name = obj
        slot.has_value = True
    else:
        slot.object_name = getattr(obj, "name", "")
        slot.has_value = bool(slot.object_name)
    slot.updated_at = float(timestamp) if timestamp is not None else time.perf_counter()
    return True


def get_object(uid: str):
    """Returns (has_value, object_or_None, updated_at)."""
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_objects", uid)
    if not slot:
        return (False, None, 0.0)
    obj = bpy.data.objects.get(slot.object_name) if slot.object_name else None
    return (bool(slot.has_value), obj, float(slot.updated_at))


def get_object_name(uid: str):
    """Returns (has_value, object_name, updated_at) - for engine worker use."""
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_objects", uid)
    if not slot:
        return (False, "", 0.0)
    return (bool(slot.has_value), slot.object_name, float(slot.updated_at))


def object_slot_exists(uid: str) -> bool:
    return _slot_exists("utility_objects", uid)


# ═══════════════════════════════════════════════════════════
# OBJECT COLLECTION REFERENCE API
# ═══════════════════════════════════════════════════════════

def create_collection_slot(scn: bpy.types.Scene, name: str = "Collection") -> str:
    coll = _get_collection(scn, "utility_collections")
    if coll is None:
        return ""
    item = coll.add()
    item.name = name or "Collection"
    _ensure_uid(item)
    item.has_value = False
    item.collection_name = ""
    item.updated_at = 0.0
    _register_slot_in_index(scn, "utility_collections", item)
    return item.uid


def set_collection(uid: str, collection, timestamp: float | None = None) -> bool:
    """Set collection reference. Accepts bpy.types.Collection or collection name string."""
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_collections", uid)
    if not slot:
        return False
    if collection is None:
        slot.collection_name = ""
        slot.has_value = False
    elif isinstance(collection, str):
        slot.collection_name = collection
        slot.has_value = True
    else:
        slot.collection_name = getattr(collection, "name", "")
        slot.has_value = bool(slot.collection_name)
    slot.updated_at = float(timestamp) if timestamp is not None else time.perf_counter()
    return True


def get_collection(uid: str):
    """Returns (has_value, collection_or_None, updated_at)."""
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_collections", uid)
    if not slot:
        return (False, None, 0.0)
    coll = bpy.data.collections.get(slot.collection_name) if slot.collection_name else None
    return (bool(slot.has_value), coll, float(slot.updated_at))


def get_collection_name(uid: str):
    """Returns (has_value, collection_name, updated_at) - for engine worker use."""
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_collections", uid)
    if not slot:
        return (False, "", 0.0)
    return (bool(slot.has_value), slot.collection_name, float(slot.updated_at))


def collection_slot_exists(uid: str) -> bool:
    return _slot_exists("utility_collections", uid)


# ═══════════════════════════════════════════════════════════
# ACTION REFERENCE API
# ═══════════════════════════════════════════════════════════

def create_action_slot(scn: bpy.types.Scene, name: str = "Action") -> str:
    coll = _get_collection(scn, "utility_actions")
    if coll is None:
        return ""
    item = coll.add()
    item.name = name or "Action"
    _ensure_uid(item)
    item.has_value = False
    item.action_name = ""
    item.updated_at = 0.0
    _register_slot_in_index(scn, "utility_actions", item)
    return item.uid


def set_action(uid: str, action, timestamp: float | None = None) -> bool:
    """Set action reference. Accepts bpy.types.Action or action name string."""
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_actions", uid)
    if not slot:
        return False
    if action is None:
        slot.action_name = ""
        slot.has_value = False
    elif isinstance(action, str):
        slot.action_name = action
        slot.has_value = True
    else:
        slot.action_name = getattr(action, "name", "")
        slot.has_value = bool(slot.action_name)
    slot.updated_at = float(timestamp) if timestamp is not None else time.perf_counter()
    return True


def get_action(uid: str):
    """Returns (has_value, action_or_None, updated_at)."""
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_actions", uid)
    if not slot:
        return (False, None, 0.0)
    action = bpy.data.actions.get(slot.action_name) if slot.action_name else None
    return (bool(slot.has_value), action, float(slot.updated_at))


def get_action_name(uid: str):
    """Returns (has_value, action_name, updated_at) - for engine worker use."""
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_actions", uid)
    if not slot:
        return (False, "", 0.0)
    return (bool(slot.has_value), slot.action_name, float(slot.updated_at))


def action_slot_exists(uid: str) -> bool:
    return _slot_exists("utility_actions", uid)


# ═══════════════════════════════════════════════════════════
# FLOAT VECTOR API (existing, refactored)
# ═══════════════════════════════════════════════════════════

def create_floatvec_slot(scn: bpy.types.Scene, name: str = "Float Vector") -> str:
    coll = _get_collection(scn, "utility_float_vectors")
    if coll is None:
        return ""
    item = coll.add()
    item.name = name or "Float Vector"
    _ensure_uid(item)
    item.has_value = False
    item.value = (0.0, 0.0, 0.0)
    item.updated_at = 0.0
    _register_slot_in_index(scn, "utility_float_vectors", item)
    return item.uid


def set_floatvec(uid: str, vec3, timestamp: float | None = None) -> bool:
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_float_vectors", uid)
    if not slot:
        return False
    try:
        x, y, z = float(vec3[0]), float(vec3[1]), float(vec3[2])
    except Exception:
        return False
    slot.value = (x, y, z)
    slot.has_value = True
    slot.updated_at = float(timestamp) if timestamp is not None else time.perf_counter()
    return True


def get_floatvec(uid: str):
    """Returns (has_value, (x,y,z), updated_at)."""
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_float_vectors", uid)
    if not slot:
        return (False, (0.0, 0.0, 0.0), 0.0)
    v = tuple(slot.value) if slot.value else (0.0, 0.0, 0.0)
    return (bool(slot.has_value), (float(v[0]), float(v[1]), float(v[2])), float(slot.updated_at))


def clear_floatvec(uid: str) -> bool:
    scn = bpy.context.scene
    slot = _find_by_uid(scn, "utility_float_vectors", uid)
    if not slot:
        return False
    slot.value = (0.0, 0.0, 0.0)
    slot.has_value = False
    slot.updated_at = 0.0
    return True


def slot_exists(uid: str) -> bool:
    """Legacy: check if float vector slot exists."""
    return _slot_exists("utility_float_vectors", uid)

# ─────────────────────────────────────────────────────────
# Register / attach to Scene
# ─────────────────────────────────────────────────────────

_SLOT_CLASSES = [
    UtilityFloatSlotPG,
    UtilityIntSlotPG,
    UtilityBoolSlotPG,
    UtilityObjectSlotPG,
    UtilityObjectCollectionSlotPG,
    UtilityActionSlotPG,
    UtilityFloatVectorSlotPG,
]


def register_utility_store_properties():
    for cls in _SLOT_CLASSES:
        bpy.utils.register_class(cls)

    # Float
    bpy.types.Scene.utility_floats = bpy.props.CollectionProperty(type=UtilityFloatSlotPG)

    # Integer
    bpy.types.Scene.utility_ints = bpy.props.CollectionProperty(type=UtilityIntSlotPG)

    # Boolean
    bpy.types.Scene.utility_bools = bpy.props.CollectionProperty(type=UtilityBoolSlotPG)

    # Object Reference
    bpy.types.Scene.utility_objects = bpy.props.CollectionProperty(type=UtilityObjectSlotPG)

    # Object Collection Reference
    bpy.types.Scene.utility_collections = bpy.props.CollectionProperty(type=UtilityObjectCollectionSlotPG)

    # Action Reference
    bpy.types.Scene.utility_actions = bpy.props.CollectionProperty(type=UtilityActionSlotPG)

    # Float Vector (existing)
    bpy.types.Scene.utility_float_vectors = bpy.props.CollectionProperty(type=UtilityFloatVectorSlotPG)
    bpy.types.Scene.utility_float_vectors_index = bpy.props.IntProperty(default=0)


def unregister_utility_store_properties():
    attrs = [
        "utility_floats",
        "utility_ints",
        "utility_bools",
        "utility_objects",
        "utility_collections",
        "utility_actions",
        "utility_float_vectors",
        "utility_float_vectors_index",
    ]
    for attr in attrs:
        try:
            delattr(bpy.types.Scene, attr)
        except Exception:
            pass

    for cls in reversed(_SLOT_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
