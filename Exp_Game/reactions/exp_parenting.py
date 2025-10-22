# Exploratory/Exp_Game/reactions/exp_parenting.py
import bpy
from mathutils import Matrix

# In-memory snapshot for "undo original parent" during gameplay
_original_parent_map: dict[str, dict] = {}

# Prefix for runtime Child-Of constraints we create/manage
CON_PREFIX = "EXP_CHILD_OF__"


# -----------------------------
# Helpers
# -----------------------------

def _bone_world_matrix(arm_obj: bpy.types.Object, bone_name: str) -> Matrix | None:
    """Return world matrix of the pose-bone, or None if not found."""
    try:
        pb = arm_obj.pose.bones.get(bone_name)
        if not pb:
            return None
        # pose_bone.matrix is in armature (object) space
        return arm_obj.matrix_world @ pb.matrix
    except Exception:
        return None


def _ensure_orig_entry(obj: bpy.types.Object) -> None:
    """Seed original parent info for an object if missing."""
    if obj.name in _original_parent_map:
        return
    par = obj.parent
    _original_parent_map[obj.name] = {
        "parent_name": par.name if par else None,
        "parent_type": (obj.parent_type if par else None),
        "parent_bone": (obj.parent_bone if par and obj.parent_type == 'BONE' else ""),
    }


def _set_parent_keep_world(child: bpy.types.Object,
                           parent: bpy.types.Object | None,
                           parent_type: str = 'OBJECT',
                           bone_name: str = "") -> None:
    """
    Set child's parent (optionally to armature bone) while preserving child's world transform.
    parent_type: 'OBJECT' or 'BONE' (only valid for armature parent).
    """
    if not child:
        return

    world = child.matrix_world.copy()

    if parent is None:
        child.parent = None
        child.parent_type = 'OBJECT'
        child.parent_bone = ""
        child.matrix_world = world
        return

    child.parent = parent
    if parent_type == 'BONE' and bone_name:
        child.parent_type = 'BONE'
        child.parent_bone = bone_name
        pmat = _bone_world_matrix(parent, bone_name)
        if pmat is None:
            child.parent_type = 'OBJECT'
            child.parent_bone = ""
            pmat = parent.matrix_world.copy()
    else:
        child.parent_type = 'OBJECT'
        child.parent_bone = ""
        pmat = parent.matrix_world.copy()

    try:
        child.matrix_parent_inverse = pmat.inverted() @ world
    except Exception:
        child.matrix_parent_inverse = Matrix.Identity(4)

    child.matrix_world = world


def _clear_parent_keep_world(child: bpy.types.Object) -> None:
    """Unparent to world, preserving child's world matrix."""
    if not child:
        return
    world = child.matrix_world.copy()
    child.parent = None
    child.parent_type = 'OBJECT'
    child.parent_bone = ""
    child.matrix_world = world


def _remove_our_childof_constraints(child: bpy.types.Object) -> None:
    """Remove any runtime Child-Of constraints we created on this child."""
    if not child:
        return
    for con in list(child.constraints):
        if con.type == 'CHILD_OF' and con.name.startswith(CON_PREFIX):
            try:
                child.constraints.remove(con)
            except Exception:
                pass


def remove_all_runtime_childof_constraints(scene: bpy.types.Scene) -> None:
    """Clean all runtime Child-Of constraints we created across the scene."""
    for obj in scene.objects:
        _remove_our_childof_constraints(obj)


def _make_or_update_childof(child: bpy.types.Object,
                            parent: bpy.types.Object,
                            bone_name: str,
                            use_loc_xyz: tuple[bool, bool, bool],
                            use_rot_xyz: tuple[bool, bool, bool],
                            use_scl_xyz: tuple[bool, bool, bool],
                            influence: float = 1.0,
                            keep_world: bool = True) -> bpy.types.Constraint | None:
    """
    Create/update a Child-Of constraint that follows only selected channels.
    We keep world transform by setting inverse_matrix and best-effort operator call.
    """
    if not child or not parent:
        return None

    name = f"{CON_PREFIX}{parent.name}__{bone_name or 'OBJECT'}"
    con = child.constraints.get(name)
    if con and con.type != 'CHILD_OF':
        child.constraints.remove(con)
        con = None
    if con is None:
        con = child.constraints.new('CHILD_OF')
        con.name = name

    con.target    = parent
    con.subtarget = bone_name if bone_name else ""
    con.influence = float(influence)

    con.use_location_x, con.use_location_y, con.use_location_z = use_loc_xyz
    con.use_rotation_x, con.use_rotation_y, con.use_rotation_z = use_rot_xyz
    con.use_scale_x,    con.use_scale_y,    con.use_scale_z    = use_scl_xyz

    if keep_world:
        child_w = child.matrix_world.copy()
        if bone_name:
            p_w = _bone_world_matrix(parent, bone_name) or parent.matrix_world.copy()
        else:
            p_w = parent.matrix_world.copy()
        try:
            # Works perfectly when all channels are enabled; still reduces popping when some are off.
            con.inverse_matrix = p_w.inverted() @ child_w
        except Exception:
            pass
        # second chance: exact inverse via operator (may fail silently if context invalid)
        try:
            bpy.context.view_layer.objects.active = child
            bpy.ops.constraint.childof_set_inverse(constraint=con.name, owner='OBJECT')
        except Exception:
            pass

    return con


# -----------------------------
# Public API (called by reset & reactions)
# -----------------------------

def capture_original_parents(modal_op, context) -> None:
    """
    Capture every object's original parent info at game start + seed runtime map.
    """
    scn = context.scene
    store: dict[str, dict] = {}

    for obj in scn.objects:
        par = obj.parent
        store[obj.name] = {
            "parent_name": par.name if par else None,
            "parent_type": (obj.parent_type if par else None),
            "parent_bone": (obj.parent_bone if par and obj.parent_type == 'BONE' else ""),
        }

    if not hasattr(modal_op, "_initial_game_state"):
        modal_op._initial_game_state = {}
    modal_op._initial_game_state["object_parents"] = store

    _original_parent_map.clear()
    _original_parent_map.update(store)


def restore_original_parents(modal_op, context) -> None:
    """
    Restore original parent relationships from the game-start snapshot.
    Order-sensitive:
      1) remove our runtime Child-Of constraints (so they don't override transforms)
      2) set the original parent (no keep-world!)
      3) restore matrix_parent_inverse from snapshot (if present; else Identity)
    NOTE: restore_scene_state() will set locations/rotations/scales AFTER this.
    """
    scn = context.scene
    state  = getattr(modal_op, "_initial_game_state", {}) or {}
    pstore = state.get("object_parents",   {}) or {}
    xstore = state.get("object_transforms",{}) or {}

    # 1) drop our runtime constraints first
    try:
        remove_all_runtime_childof_constraints(scn)
    except Exception:
        pass

    # 2) put original parents back (without trying to preserve current world)
    for child_name, info in pstore.items():
        child = bpy.data.objects.get(child_name)
        if not child:
            continue

        pname = info.get("parent_name")
        ptype = (info.get("parent_type") or 'OBJECT')
        pbone = (info.get("parent_bone") or "")
        parent = bpy.data.objects.get(pname) if pname else None

        # assign parent without keep-world math
        child.parent = parent
        if parent and ptype == 'BONE' and pbone:
            child.parent_type = 'BONE'
            child.parent_bone = pbone
        else:
            child.parent_type = 'OBJECT'
            child.parent_bone = ""

        # 3) restore the exact parent inverse if we captured it
        inv = xstore.get(child_name, {}).get("parent_inv", None)
        try:
            child.matrix_parent_inverse = inv.copy() if hasattr(inv, "copy") else Matrix(inv) if inv is not None else Matrix.Identity(4)
        except Exception:
            child.matrix_parent_inverse = Matrix.Identity(4)

def execute_parenting_reaction(r) -> None:
    """
    Execute one Parenting reaction.

    Fields read from ReactionDefinition:
      - parenting_op: 'PARENT_TO' | 'UNPARENT'
      - parenting_target_use_character (bool)
      - parenting_target_object (Object)

      - parenting_parent_use_armature (bool)
      - parenting_parent_object (Object)
      - parenting_bone_name (str)

      - parenting_follow_mode: 'PARENT' | 'CHILD_OF'
      - parenting_follow_loc, parenting_follow_rot, parenting_follow_scl (bool)
      - parenting_follow_loc_x/y/z, parenting_follow_rot_x/y/z, parenting_follow_scl_x/y/z (bool)
      - parenting_follow_influence (float)
    """
    import traceback
    try:
        scn = bpy.context.scene

        # Resolve child (target)
        child = scn.target_armature if getattr(r, "parenting_target_use_character", False) \
               else getattr(r, "parenting_target_object", None)

        if not child or not isinstance(child, bpy.types.Object):
            print("[Parenting Reaction] No valid target (child).")
            return

        # Ensure UNPARENT can restore original
        _ensure_orig_entry(child)

        op = getattr(r, "parenting_op", "PARENT_TO")
        mode = getattr(r, "parenting_follow_mode", "PARENT")

        # ---------------- UNPARENT ----------------
        if op == "UNPARENT":
            # Restore original parent
            info = _original_parent_map.get(child.name)
            if not info:
                _clear_parent_keep_world(child)
            else:
                pname = info.get("parent_name")
                ptype = info.get("parent_type") or 'OBJECT'
                pbone = info.get("parent_bone") or ""
                if not pname:
                    _clear_parent_keep_world(child)
                else:
                    parent = bpy.data.objects.get(pname)
                    if not parent:
                        _clear_parent_keep_world(child)
                    else:
                        if ptype == 'BONE' and pbone:
                            _set_parent_keep_world(child, parent, 'BONE', pbone)
                        else:
                            _set_parent_keep_world(child, parent, 'OBJECT', "")
            # Remove any runtime Child-Of we added
            _remove_our_childof_constraints(child)
            return

        # ---------------- PARENT_TO ----------------

        # Resolve parent target (object or armature[+bone])
        if getattr(r, "parenting_parent_use_armature", False):
            parent = scn.target_armature
            if not parent:
                print("[Parenting Reaction] No character armature found to parent to.")
                return
            bone_name = (getattr(r, "parenting_bone_name", "") or "").strip()
            # Guard: never allow self-parent (armature -> itself or its own bone)
            if parent == child:
                print("[Parenting Reaction] Skipped: target and parent are the same object.")
                return

            if mode == "PARENT":
                if bone_name and parent.pose.bones.get(bone_name) is not None:
                    _set_parent_keep_world(child, parent, 'BONE', bone_name)
                else:
                    _set_parent_keep_world(child, parent, 'OBJECT', "")
                return

            # CHILD_OF (selective follow)
            # Build channel flags
            loc = bool(getattr(r, "parenting_follow_loc", True))
            rot = bool(getattr(r, "parenting_follow_rot", True))
            scl = bool(getattr(r, "parenting_follow_scl", True))

            loc_xyz = (
                loc and bool(getattr(r, "parenting_follow_loc_x", True)),
                loc and bool(getattr(r, "parenting_follow_loc_y", True)),
                loc and bool(getattr(r, "parenting_follow_loc_z", True)),
            )
            rot_xyz = (
                rot and bool(getattr(r, "parenting_follow_rot_x", True)),
                rot and bool(getattr(r, "parenting_follow_rot_y", True)),
                rot and bool(getattr(r, "parenting_follow_rot_z", True)),
            )
            scl_xyz = (
                scl and bool(getattr(r, "parenting_follow_scl_x", True)),
                scl and bool(getattr(r, "parenting_follow_scl_y", True)),
                scl and bool(getattr(r, "parenting_follow_scl_z", True)),
            )
            infl = float(getattr(r, "parenting_follow_influence", 1.0) or 1.0)

            if bone_name and parent.pose.bones.get(bone_name) is None:
                print(f"[Parenting Reaction] Bone '{bone_name}' not found; using OBJECT constraint.")
                bone_name = ""

            # Add/update our runtime Child-Of (do NOT change real parent in this mode)
            _make_or_update_childof(child, parent, bone_name, loc_xyz, rot_xyz, scl_xyz, infl, keep_world=True)
            return

        # Parent to explicit object
        parent = getattr(r, "parenting_parent_object", None)
        if not parent or not isinstance(parent, bpy.types.Object):
            print("[Parenting Reaction] No valid parent object.")
            return
        if parent == child:
            print("[Parenting Reaction] Skipped: target and parent are the same object.")
            return

        if mode == "PARENT":
            _set_parent_keep_world(child, parent, 'OBJECT', "")
            return

        # CHILD_OF with explicit object
        loc = bool(getattr(r, "parenting_follow_loc", True))
        rot = bool(getattr(r, "parenting_follow_rot", True))
        scl = bool(getattr(r, "parenting_follow_scl", True))
        loc_xyz = (
            loc and bool(getattr(r, "parenting_follow_loc_x", True)),
            loc and bool(getattr(r, "parenting_follow_loc_y", True)),
            loc and bool(getattr(r, "parenting_follow_loc_z", True)),
        )
        rot_xyz = (
            rot and bool(getattr(r, "parenting_follow_rot_x", True)),
            rot and bool(getattr(r, "parenting_follow_rot_y", True)),
            rot and bool(getattr(r, "parenting_follow_rot_z", True)),
        )
        scl_xyz = (
            scl and bool(getattr(r, "parenting_follow_scl_x", True)),
            scl and bool(getattr(r, "parenting_follow_scl_y", True)),
            scl and bool(getattr(r, "parenting_follow_scl_z", True)),
        )
        infl = float(getattr(r, "parenting_follow_influence", 1.0) or 1.0)
        _make_or_update_childof(child, parent, "", loc_xyz, rot_xyz, scl_xyz, infl, keep_world=True)

    except Exception:
        import traceback
        print("[Parenting Reaction] Exception:")
        traceback.print_exc()
