#Exploratory/prefs_persistence.py
import bpy, os, json

ADDON_NAME = "Exploratory"
_applying = False

# Properties that need special (non-scalar) serialization
_SKIP_PROPS = {"asset_packs"}

def _prefs_file_path() -> str:
    cfg = bpy.utils.user_resource('CONFIG', path="addons", create=True)
    return os.path.join(cfg, f"{ADDON_NAME}_prefs.json")

def write_prefs():
    global _applying
    # if we're currently in apply_prefs(), don't overwrite
    if _applying:
        return

    prefs = bpy.context.preferences.addons[ADDON_NAME].preferences

    # respect the "Keep Preferences" toggle
    if not getattr(prefs, "keep_preferences", True):
        path = _prefs_file_path()
        if os.path.isfile(path):
            os.remove(path)
        return

    data = {}
    for prop in prefs.bl_rna.properties:
        ident = prop.identifier
        if prop.is_readonly or ident == "rna_type" or ident in _SKIP_PROPS:
            continue
        try:
            data[ident] = getattr(prefs, ident)
        except Exception:
            pass

    # Serialize asset_packs CollectionProperty
    data["asset_packs"] = [
        {"filepath": e.filepath, "enabled": e.enabled}
        for e in prefs.asset_packs
    ]

    with open(_prefs_file_path(), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def read_prefs() -> dict:
    path = _prefs_file_path()
    if not os.path.isfile(path):
        return {}
    try:
        return json.loads(open(path, 'r', encoding='utf-8').read())
    except Exception:
        return {}

def apply_prefs():
    global _applying
    data = read_prefs()
    if not data:
        return None

    prefs = bpy.context.preferences.addons[ADDON_NAME].preferences

    # if user has disabled persistence, skip load
    if not getattr(prefs, "keep_preferences", True):
        return None

    _applying = True
    try:
        # 1) restore file-paths so your update callbacks repopulate enums
        for key, val in data.items():
            if key in _SKIP_PROPS:
                continue
            prop = prefs.bl_rna.properties.get(key)
            if not prop:
                continue
            if prop.type == 'STRING' and getattr(prop, 'subtype', '') == 'FILE_PATH':
                try: setattr(prefs, key, val)
                except Exception: pass

        # 2) restore everything else (bools, floats, enums, etc.)
        for key, val in data.items():
            if key in _SKIP_PROPS:
                continue
            prop = prefs.bl_rna.properties.get(key)
            if not prop:
                continue
            if prop.type == 'STRING' and getattr(prop, 'subtype', '') == 'FILE_PATH':
                continue
            try: setattr(prefs, key, val)
            except Exception: pass

        # 3) restore asset_packs collection
        pack_data = data.get("asset_packs", [])
        prefs.asset_packs.clear()
        for item in pack_data:
            entry = prefs.asset_packs.add()
            entry.filepath = item.get("filepath", "")
            entry.enabled = item.get("enabled", True)

    finally:
        _applying = False

    return None  # one-shot timer

def register_prefs_handlers():
    bpy.app.timers.register(apply_prefs, first_interval=0.1)

def unregister_prefs_handlers():
    write_prefs()
