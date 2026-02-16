import bpy

# ── Role constants ──────────────────────────────────────────────────
ACTION_ROLES = ["IDLE", "WALK", "RUN", "JUMP", "FALL", "LAND"]
SOUND_ROLES = ["WALK_SOUND", "RUN_SOUND", "JUMP_SOUND", "FALL_SOUND", "LAND_SOUND"]

_ROLE_KEY = "exp_asset_role"


# ── Helpers ─────────────────────────────────────────────────────────
def _collection_for_role(role):
    """Return the bpy.data collection that stores datablocks for *role*."""
    if role == "SKIN":
        return bpy.data.objects
    if role in ACTION_ROLES:
        return bpy.data.actions
    if role in SOUND_ROLES:
        return bpy.data.sounds
    return None


def find_marked(role):
    """Return the datablock marked with *role*, or None."""
    col = _collection_for_role(role)
    if col is None:
        return None
    for db in col:
        if db.get(_ROLE_KEY) == role:
            return db
    return None


def _clear_role(role):
    """Remove the mark from whatever datablock currently holds *role*."""
    db = find_marked(role)
    if db is not None and _ROLE_KEY in db:
        del db[_ROLE_KEY]


# ── Property-name mapping ──────────────────────────────────────────
def _pg_attr_for_role(role):
    """Return the AssetMarkingPG attribute name for a given role."""
    if role == "SKIN":
        return "skin"
    if role in ACTION_ROLES:
        return role.lower() + "_action"
    if role in SOUND_ROLES:
        return role.lower()            # e.g. "walk_sound"
    return None


# ── PropertyGroup ───────────────────────────────────────────────────
class AssetMarkingPG(bpy.types.PropertyGroup):
    skin: bpy.props.PointerProperty(
        name="Skin", type=bpy.types.Object,
        description="Character armature / skin object",
    )
    idle_action: bpy.props.PointerProperty(
        name="Idle", type=bpy.types.Action,
        description="Idle animation action",
    )
    walk_action: bpy.props.PointerProperty(
        name="Walk", type=bpy.types.Action,
        description="Walk animation action",
    )
    run_action: bpy.props.PointerProperty(
        name="Run", type=bpy.types.Action,
        description="Run animation action",
    )
    jump_action: bpy.props.PointerProperty(
        name="Jump", type=bpy.types.Action,
        description="Jump animation action",
    )
    fall_action: bpy.props.PointerProperty(
        name="Fall", type=bpy.types.Action,
        description="Fall animation action",
    )
    land_action: bpy.props.PointerProperty(
        name="Land", type=bpy.types.Action,
        description="Land animation action",
    )
    walk_sound: bpy.props.PointerProperty(
        name="Walk", type=bpy.types.Sound,
        description="Walk sound effect",
    )
    run_sound: bpy.props.PointerProperty(
        name="Run", type=bpy.types.Sound,
        description="Run sound effect",
    )
    jump_sound: bpy.props.PointerProperty(
        name="Jump", type=bpy.types.Sound,
        description="Jump sound effect",
    )
    fall_sound: bpy.props.PointerProperty(
        name="Fall", type=bpy.types.Sound,
        description="Fall sound effect",
    )
    land_sound: bpy.props.PointerProperty(
        name="Land", type=bpy.types.Sound,
        description="Land sound effect",
    )


# ── Operators ───────────────────────────────────────────────────────
class EXPLORATORY_OT_MarkAsset(bpy.types.Operator):
    bl_idname = "exploratory.mark_asset"
    bl_label = "Mark Asset"
    bl_description = "Tag the selected datablock with its game role"
    bl_options = {'UNDO', 'INTERNAL'}

    role: bpy.props.StringProperty()

    def execute(self, context):
        pg = context.scene.asset_marking
        attr = _pg_attr_for_role(self.role)
        if attr is None:
            self.report({'WARNING'}, f"Unknown role: {self.role}")
            return {'CANCELLED'}

        db = getattr(pg, attr, None)
        if db is None:
            self.report({'WARNING'}, "Pick a datablock first")
            return {'CANCELLED'}

        _clear_role(self.role)
        db[_ROLE_KEY] = self.role
        return {'FINISHED'}


class EXPLORATORY_OT_UnmarkAsset(bpy.types.Operator):
    bl_idname = "exploratory.unmark_asset"
    bl_label = "Unmark Asset"
    bl_description = "Remove the game-role tag from the marked datablock"
    bl_options = {'UNDO', 'INTERNAL'}

    role: bpy.props.StringProperty()

    def execute(self, context):
        _clear_role(self.role)
        return {'FINISHED'}


# ── Asset-pack scanning helpers ────────────────────────────────────────
import os
import random


def scan_packs_for_roles(pack_paths, roles, datablock_type):
    """Scan enabled .blend packs and collect role-marked datablocks.

    For each pack path: append all datablocks of *datablock_type* (e.g.
    ``"actions"`` or ``"sounds"``), inspect ``exp_asset_role`` marks, keep
    candidates that match *roles*, and remove non-marked datablocks.

    Returns ``{role: [list_of_datablocks]}``.
    """
    candidates = {role: [] for role in roles}
    collection = getattr(bpy.data, datablock_type)

    for path in pack_paths:
        if not os.path.isfile(path):
            continue

        before = set(collection.keys())

        try:
            with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
                names = list(getattr(data_from, datablock_type))
                setattr(data_to, datablock_type, names)
        except Exception:
            continue

        after = set(collection.keys())
        new_names = after - before

        marked = set()
        for name in new_names:
            db = collection.get(name)
            if db is None:
                continue
            role = db.get(_ROLE_KEY)
            if role in candidates:
                candidates[role].append(db)
                marked.add(name)

        # Remove non-marked datablocks appended from this pack
        for name in new_names - marked:
            db = collection.get(name)
            if db is not None:
                collection.remove(db)

    return candidates


def resolve_candidates(candidates, default_blend, default_names, datablock_type):
    """Pick one datablock per role from *candidates*.

    Fallback logic: 0 candidates -> load default from *default_blend*,
    1 -> use it, 2+ -> ``random.choice``.

    *default_names* maps role string to the default datablock name.

    Returns ``{role: datablock}``.
    """
    result = {}
    collection = getattr(bpy.data, datablock_type)

    for role, cands in candidates.items():
        if len(cands) == 0:
            name = default_names.get(role)
            if not name:
                continue
            existing = collection.get(name)
            if not existing and default_blend and os.path.isfile(default_blend):
                try:
                    with bpy.data.libraries.load(default_blend, link=False) as (df, dt):
                        available = list(getattr(df, datablock_type))
                        if name in available:
                            setattr(dt, datablock_type, [name])
                except Exception:
                    pass
                existing = collection.get(name)
            if existing:
                result[role] = existing
        elif len(cands) == 1:
            result[role] = cands[0]
        else:
            result[role] = random.choice(cands)

    return result


def _scan_packs_for_skin(pack_paths):
    """Scan packs for SKIN-marked objects and their hierarchies.

    Returns a list of ``(skin_object, [hierarchy_objects])`` tuples.
    Non-marked objects from each pack are removed from ``bpy.data``.
    """
    candidates = []

    for path in pack_paths:
        if not os.path.isfile(path):
            continue

        before = set(bpy.data.objects.keys())

        try:
            with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
                data_to.objects = list(data_from.objects)
        except Exception:
            continue

        after = set(bpy.data.objects.keys())
        new_names = after - before

        # Find SKIN-marked object among newly appended
        marked_obj = None
        for name in new_names:
            obj = bpy.data.objects.get(name)
            if obj and obj.get(_ROLE_KEY) == "SKIN":
                marked_obj = obj
                break

        if marked_obj:
            keep = {marked_obj.name}

            # Walk up parent chain (only within this pack's objects)
            p = marked_obj.parent
            while p and p.name in new_names:
                keep.add(p.name)
                p = p.parent

            # Walk down children (only within this pack's objects)
            for child in marked_obj.children_recursive:
                if child.name in new_names:
                    keep.add(child.name)

            hierarchy = [bpy.data.objects[n] for n in keep if n != marked_obj.name]
            candidates.append((marked_obj, hierarchy))

            # Remove non-hierarchy objects from this pack
            for name in new_names - keep:
                obj = bpy.data.objects.get(name)
                if obj:
                    bpy.data.objects.remove(obj)
        else:
            # No skin found in this pack, remove all appended objects
            for name in new_names:
                obj = bpy.data.objects.get(name)
                if obj:
                    bpy.data.objects.remove(obj)

    return candidates
