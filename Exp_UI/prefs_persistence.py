#Exploratory/Exp_UI/prefs_persistence.py
import bpy, os, json

ADDON_NAME = "Exploratory"
_applying = False

def _prefs_file_path() -> str:
    cfg = bpy.utils.user_resource('CONFIG', path="addons", create=True)
    return os.path.join(cfg, f"{ADDON_NAME}_prefs.json")

def write_prefs():
    global _applying
    # if we’re currently in apply_prefs(), don’t overwrite
    if _applying:
        return

    prefs = bpy.context.preferences.addons[ADDON_NAME].preferences

    # respect the “Keep Preferences” toggle
    if not getattr(prefs, "keep_preferences", True):
        path = _prefs_file_path()
        if os.path.isfile(path):
            os.remove(path)
        return

    data = {}
    for prop in prefs.bl_rna.properties:
        ident = prop.identifier
        if prop.is_readonly or ident == "rna_type":
            continue
        try:
            data[ident] = getattr(prefs, ident)
        except Exception:
            pass

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
        # 1) restore file-paths so your update_… callbacks repopulate enums
        for key, val in data.items():
            prop = prefs.bl_rna.properties.get(key)
            if not prop:
                continue
            if prop.type == 'STRING' and getattr(prop, 'subtype', '') == 'FILE_PATH':
                try: setattr(prefs, key, val)
                except Exception: pass

        # 2) restore everything else (bools, floats, enums, etc.)
        for key, val in data.items():
            prop = prefs.bl_rna.properties.get(key)
            if not prop:
                continue
            if prop.type == 'STRING' and getattr(prop, 'subtype', '') == 'FILE_PATH':
                continue
            try: setattr(prefs, key, val)
            except Exception: pass

    finally:
        _applying = False

    return None  # one-shot timer

def register_prefs_handlers():
    bpy.app.timers.register(apply_prefs, first_interval=0.1)

def unregister_prefs_handlers():
    write_prefs()