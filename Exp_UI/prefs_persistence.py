import bpy
import os
import json

ADDON_NAME = "Exploratory"


def prefs_json_path() -> str:
    """
    Returns the path to the JSON file in the user CONFIG addons directory.
    """
    cfg = bpy.utils.user_resource('CONFIG', path="addons", create=True)
    return os.path.join(cfg, f"{ADDON_NAME}_prefs.json")


def load_prefs_from_json():
    """
    Load preferences from JSON, setting only valid properties.
    """
    path = prefs_json_path()
    if not os.path.isfile(path):
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[{ADDON_NAME}] Failed to load prefs JSON: {e}")
        return

    addons = bpy.context.preferences.addons
    if ADDON_NAME not in addons:
        return

    prefs = addons[ADDON_NAME].preferences

    for key, value in data.items():
        # skip unknown props
        if key not in prefs.bl_rna.properties:
            continue

        prop = prefs.bl_rna.properties[key]
        # for enums, only set valid identifiers
        if prop.type == 'ENUM':
            valid = [item.identifier for item in prop.enum_items]
            if value not in valid:
                continue

        try:
            setattr(prefs, key, value)
        except Exception as e:
            print(f"[{ADDON_NAME}] Could not set preference '{key}' = {value}: {e}")


def save_prefs_to_json():
    """
    Save all preference properties (except rna_type) to JSON.
    """
    addons = bpy.context.preferences.addons
    if ADDON_NAME not in addons:
        return

    prefs = addons[ADDON_NAME].preferences
    data = {}

    for prop in prefs.bl_rna.properties:
        name = prop.identifier
        if name == "rna_type":
            continue
        try:
            data[name] = getattr(prefs, name)
        except Exception:
            # skip any property that errors
            pass

    try:
        with open(prefs_json_path(), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[{ADDON_NAME}] Failed to save prefs JSON: {e}")


def on_pref_update(self, context):
    """
    Callback for most preference updates: write out the JSON immediately.
    """
    save_prefs_to_json()


def on_keep_preferences_update(self, context):
    """
    If keep_preferences is False, delete the JSON file. Otherwise save immediately.
    """
    path = prefs_json_path()
    if not getattr(self, 'keep_preferences', True):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"[{ADDON_NAME}] Failed to remove prefs JSON: {e}")
    else:
        save_prefs_to_json()