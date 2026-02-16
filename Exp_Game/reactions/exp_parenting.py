# Exploratory/Exp_Game/reactions/exp_parenting.py
"""
Parenting system for runtime parent/unparent operations.
Blender 5.0+ only.

Simple atomic parenting - child snaps to parent's origin with optional local offset.
Works perfectly with moving/animated parents.

Includes a small built-in delay (~1 frame) to ensure any pending transforms
have time to complete before parenting is applied.
"""
import bpy
from mathutils import Matrix, Vector, Euler
from ..props_and_utils.exp_time import get_game_time
from .exp_bindings import resolve_vector

# Snapshot of original parents at game start
_original_parent_map: dict[str, dict] = {}

# Pending parenting operations (delayed to ensure transforms complete first)
_pending_parenting: list[dict] = []

# Built-in safety delay (seconds) - ensures transforms complete before parenting
PARENTING_DELAY = 0.017  # ~1 frame at 60fps


def _ensure_orig_entry(obj: bpy.types.Object) -> None:
    """Seed original parent info for an object if missing."""
    if obj.name in _original_parent_map:
        return
    par = obj.parent
    _original_parent_map[obj.name] = {
        "parent_name": par.name if par else None,
        "parent_type": obj.parent_type if par else None,
        "parent_bone": obj.parent_bone if par and obj.parent_type == 'BONE' else "",
        "matrix_parent_inverse": obj.matrix_parent_inverse.copy(),
    }


# -----------------------------
# Core Parenting
# -----------------------------

def parent_to(child: bpy.types.Object,
              parent: bpy.types.Object,
              bone_name: str = "",
              local_offset: Vector = None,
              local_rotation: Euler = None) -> None:
    """
    Parent child to parent, placing child at the parent's origin.

    Atomic operation - no race condition with moving parents.
    Child's local transform becomes the offset from parent.
    """
    if not child or not parent or parent == child:
        return

    child.parent = parent

    if bone_name:
        pb = parent.pose.bones.get(bone_name) if hasattr(parent, 'pose') else None
        if pb:
            child.parent_type = 'BONE'
            child.parent_bone = bone_name
        else:
            child.parent_type = 'OBJECT'
            child.parent_bone = ""
    else:
        child.parent_type = 'OBJECT'
        child.parent_bone = ""

    # Identity inverse = child at parent's origin
    child.matrix_parent_inverse = Matrix.Identity(4)

    # Set local transform
    child.location = local_offset if local_offset else Vector((0, 0, 0))
    child.rotation_euler = local_rotation if local_rotation else Euler((0, 0, 0), 'XYZ')


def unparent(child: bpy.types.Object) -> None:
    """Unparent to world, preserving child's world position."""
    if not child:
        return
    world = child.matrix_world.copy()
    child.parent = None
    child.parent_type = 'OBJECT'
    child.parent_bone = ""
    child.matrix_parent_inverse = Matrix.Identity(4)
    child.matrix_world = world


def unparent_restore_original(child: bpy.types.Object) -> None:
    """Restore child's original parent from game start."""
    if not child:
        return

    info = _original_parent_map.get(child.name)
    if not info or not info.get("parent_name"):
        unparent(child)
        return

    parent = bpy.data.objects.get(info["parent_name"])
    if not parent:
        unparent(child)
        return

    child.parent = parent
    ptype = info.get("parent_type") or 'OBJECT'
    pbone = info.get("parent_bone") or ""

    if ptype == 'BONE' and pbone:
        child.parent_type = 'BONE'
        child.parent_bone = pbone
    else:
        child.parent_type = 'OBJECT'
        child.parent_bone = ""

    inv = info.get("matrix_parent_inverse")
    child.matrix_parent_inverse = inv.copy() if inv else Matrix.Identity(4)


# -----------------------------
# Game Start/End
# -----------------------------

def capture_original_parents(modal_op, context) -> None:
    """Capture every object's original parent info at game start."""
    scn = context.scene
    store: dict[str, dict] = {}

    for obj in scn.objects:
        par = obj.parent
        store[obj.name] = {
            "parent_name": par.name if par else None,
            "parent_type": obj.parent_type if par else None,
            "parent_bone": obj.parent_bone if par and obj.parent_type == 'BONE' else "",
            "matrix_parent_inverse": obj.matrix_parent_inverse.copy(),
        }

    if not hasattr(modal_op, "_initial_game_state"):
        modal_op._initial_game_state = {}
    modal_op._initial_game_state["object_parents"] = store

    _original_parent_map.clear()
    _original_parent_map.update(store)


def restore_original_parents(modal_op, context) -> None:
    """Restore original parent relationships from game-start snapshot."""
    scn = context.scene
    state = getattr(modal_op, "_initial_game_state", {}) or {}
    pstore = state.get("object_parents", {}) or {}

    for child_name, info in pstore.items():
        child = bpy.data.objects.get(child_name)
        if not child:
            continue

        pname = info.get("parent_name")
        parent = bpy.data.objects.get(pname) if pname else None

        child.parent = parent
        if parent:
            ptype = info.get("parent_type") or 'OBJECT'
            pbone = info.get("parent_bone") or ""
            if ptype == 'BONE' and pbone:
                child.parent_type = 'BONE'
                child.parent_bone = pbone
            else:
                child.parent_type = 'OBJECT'
                child.parent_bone = ""
        else:
            child.parent_type = 'OBJECT'
            child.parent_bone = ""

        inv = info.get("matrix_parent_inverse")
        child.matrix_parent_inverse = inv.copy() if inv else Matrix.Identity(4)


def clear_pending_parenting() -> None:
    """Clear pending parenting queue. Called on game stop."""
    _pending_parenting.clear()


# -----------------------------
# Pending Parenting Processing
# -----------------------------

def process_pending_parenting() -> None:
    """
    Process any pending parenting operations whose delay has elapsed.
    Called each frame from the game loop.
    """
    if not _pending_parenting:
        return

    now = get_game_time()

    for task in _pending_parenting[:]:
        if now >= task["fire_time"]:
            _apply_parenting_task(task)
            _pending_parenting.remove(task)


def _apply_parenting_task(task: dict) -> None:
    """Apply a single parenting task."""
    op = task["op"]
    child_name = task["child_name"]

    child = bpy.data.objects.get(child_name)
    if not child:
        return

    if op == "UNPARENT":
        unparent_restore_original(child)
    else:
        parent_name = task.get("parent_name")
        parent = bpy.data.objects.get(parent_name) if parent_name else None
        if not parent:
            return

        bone_name = task.get("bone_name", "")
        local_offset = task.get("local_offset")

        parent_to(child, parent, bone_name, local_offset)


# -----------------------------
# Reaction Execution
# -----------------------------

def execute_parenting_reaction(r) -> None:
    """
    Execute a Parenting reaction.

    Parenting is delayed by PARENTING_DELAY (~1 frame) to ensure any
    pending transforms have time to complete first.
    """
    scn = bpy.context.scene

    # Resolve child
    child = getattr(r, "parenting_target_object", None)

    if not child:
        return

    _ensure_orig_entry(child)

    op = getattr(r, "parenting_op", "PARENT_TO")
    now = get_game_time()

    # UNPARENT - schedule with delay
    if op == "UNPARENT":
        _pending_parenting.append({
            "op": "UNPARENT",
            "child_name": child.name,
            "fire_time": now + PARENTING_DELAY,
        })
        return

    # PARENT_TO - schedule with delay
    parent = getattr(r, "parenting_parent_object", None)
    bone_name = (getattr(r, "parenting_bone_name", "") or "").strip()

    if not parent:
        return

    # Resolve local offset from binding or property (supports connected data nodes)
    local_offset = resolve_vector(r, "parenting_local_offset", r.parenting_local_offset)

    _pending_parenting.append({
        "op": "PARENT_TO",
        "child_name": child.name,
        "parent_name": parent.name,
        "bone_name": bone_name,
        "local_offset": local_offset,
        "fire_time": now + PARENTING_DELAY,
    })
