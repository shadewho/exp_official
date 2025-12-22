# Exploratory/Exp_Game/reactions/exp_action_keys.py
# Lightweight named gate for Action Key triggers, plus a Scene-level registry
# of action-key *names* for UI (enum) and rename/delete support.

import bpy
from typing import Dict

# ─────────────────────────────────────────────────────────
# Scene-level list of Action Keys (names only)
# ─────────────────────────────────────────────────────────
class ActionKeyItemPG(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Action Name", default="Action 1")
    enabled_default: bpy.props.BoolProperty(
        name="Enabled by Default",
        default=False,
        description="Initial enabled state when gameplay starts or resets"
    )


def register_action_key_properties():
    """Attach scene.action_keys (Collection of ActionKeyItemPG) + index."""
    if not hasattr(bpy.types.Scene, "action_keys"):
        bpy.types.Scene.action_keys = bpy.props.CollectionProperty(type=ActionKeyItemPG)
    if not hasattr(bpy.types.Scene, "action_keys_index"):
        bpy.types.Scene.action_keys_index = bpy.props.IntProperty(default=0)


def unregister_action_key_properties():
    if hasattr(bpy.types.Scene, "action_keys"):
        del bpy.types.Scene.action_keys
    if hasattr(bpy.types.Scene, "action_keys_index"):
        del bpy.types.Scene.action_keys_index


# ─────────────────────────────────────────────────────────
# Runtime gate (unchanged behavior)
# ─────────────────────────────────────────────────────────
_enabled: Dict[str, bool] = {}  # id/name -> enabled flag


def _norm(s: str) -> str:
    return (s or "").strip()  # keep case-sensitive; just trim


def enable(key_id: str) -> None:
    k = _norm(key_id)
    if k:
        _enabled[k] = True


def disable(key_id: str) -> None:
    k = _norm(key_id)
    if k:
        _enabled[k] = False

def seed_defaults_from_scene(scene=None) -> None:
    """
    Reset runtime flags and seed from Scene.action_keys.enabled_default.
    Called by reset_all_interactions(scene).
    """
    _enabled.clear()
    scn = scene or getattr(bpy.context, "scene", None)
    if not scn or not hasattr(scn, "action_keys"):
        return
    try:
        for it in scn.action_keys:
            nm = (getattr(it, "name", "") or "").strip()
            if nm:
                _enabled[nm] = bool(getattr(it, "enabled_default", False))
    except Exception:
        pass
    
def toggle(key_id: str) -> None:
    k = _norm(key_id)
    if k:
        _enabled[k] = not _enabled.get(k, False)


def is_enabled(key_id: str) -> bool:
    k = _norm(key_id)
    return bool(k) and bool(_enabled.get(k, False))


def reset_all_action_keys() -> None:
    _enabled.clear()


# Reaction executor (called from run_reactions)
def execute_action_key_reaction(r) -> None:
    """
    Prefer the new r.action_key_name; fall back to the legacy r.action_key_id
    so older blends still work.
    """
    op  = getattr(r, "action_key_op", "ENABLE")
    kid = getattr(r, "action_key_name", "") or getattr(r, "action_key_id", "")
    if   op == "ENABLE":  enable(kid)
    elif op == "DISABLE": disable(kid)
    elif op == "TOGGLE":  toggle(kid)



def _update_action_key_name(self, context):
    """
    When the reaction's action_key_name changes (user selects from dropdown):
      • Keep legacy r.action_key_id in sync

    NOTE: This callback should NOT write to scene.action_keys!
    The scene.action_keys list is managed exclusively by CreateActionKeyNode.
    This callback only syncs the legacy action_key_id field.
    """
    new_name = (getattr(self, "action_key_name", "") or "").strip()

    # Keep legacy id mirrored - this is the only thing we should do
    try:
        self.action_key_id = new_name
    except Exception:
        pass
