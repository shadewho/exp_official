# Exploratory/Exp_Game/props_and_utils/exp_utility_store.py
import bpy
import time
from uuid import uuid4
from bpy.types import PropertyGroup


# ─────────────────────────────────────────────────────────
# Scene-level registry for Utility Nodes (FloatVector first)
# ─────────────────────────────────────────────────────────

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
# Internal helpers
# ─────────────────────────────────────────────────────────

def _ensure_uid(slot: UtilityFloatVectorSlotPG) -> None:
    if not getattr(slot, "uid", ""):
        slot.uid = str(uuid4())

def _slots(scn: bpy.types.Scene):
    return getattr(scn, "utility_float_vectors", None)

def _find_slot_by_uid(scn: bpy.types.Scene, uid: str):
    coll = _slots(scn)
    if not coll:
        return None
    for it in coll:
        if getattr(it, "uid", "") == uid:
            return it
    return None


# ─────────────────────────────────────────────────────────
# Public API (import from anywhere)
# ─────────────────────────────────────────────────────────

def create_floatvec_slot(scn: bpy.types.Scene, name: str = "Capture Vector") -> str:
    """
    Create a new float-vector capture slot. Returns its UID.
    """
    coll = _slots(scn)
    if coll is None:
        return ""
    item = coll.add()
    item.name = name or "Capture Vector"
    _ensure_uid(item)
    item.has_value = False
    item.value = (0.0, 0.0, 0.0)
    item.updated_at = 0.0
    try:
        scn.utility_float_vectors_index = len(coll) - 1
    except Exception:
        pass
    return item.uid

def set_floatvec(uid: str, vec3, timestamp: float | None = None) -> bool:
    """
    Store a new vector into the slot with 'uid'.
    Returns True on success.
    """
    scn = bpy.context.scene
    slot = _find_slot_by_uid(scn, uid)
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
    """
    Returns (has_value: bool, (x, y, z): tuple[float,float,float], updated_at: float) or (False,(0,0,0),0).
    """
    scn = bpy.context.scene
    slot = _find_slot_by_uid(scn, uid)
    if not slot:
        return (False, (0.0, 0.0, 0.0), 0.0)
    v = tuple(slot.value) if slot.value else (0.0, 0.0, 0.0)
    return (bool(slot.has_value), (float(v[0]), float(v[1]), float(v[2])), float(slot.updated_at))

def clear_floatvec(uid: str) -> bool:
    """
    Clears the stored value (marks has_value False, zeroes vector).
    """
    scn = bpy.context.scene
    slot = _find_slot_by_uid(scn, uid)
    if not slot:
        return False
    slot.value = (0.0, 0.0, 0.0)
    slot.has_value = False
    slot.updated_at = 0.0
    return True

def slot_exists(uid: str) -> bool:
    """
    Return True if a float-vector capture slot with this UID exists on the current scene.
    """
    scn = bpy.context.scene
    return _find_slot_by_uid(scn, uid) is not None

# ─────────────────────────────────────────────────────────
# Register / attach to Scene
# ─────────────────────────────────────────────────────────

def register_utility_store_properties():
    bpy.utils.register_class(UtilityFloatVectorSlotPG)
    bpy.types.Scene.utility_float_vectors = bpy.props.CollectionProperty(type=UtilityFloatVectorSlotPG)
    bpy.types.Scene.utility_float_vectors_index = bpy.props.IntProperty(default=0)

def unregister_utility_store_properties():
    try:
        del bpy.types.Scene.utility_float_vectors
        del bpy.types.Scene.utility_float_vectors_index
    except Exception:
        pass
    try:
        bpy.utils.unregister_class(UtilityFloatVectorSlotPG)
    except Exception:
        pass
