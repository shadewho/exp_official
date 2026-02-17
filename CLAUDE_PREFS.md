# CLAUDE_PREFS.md — Preferences Upgrade Safety

## The Rule

Users upgrading from older versions (e.g. 4.5 to 5.0) MUST be able to install and enable the addon without errors. A stale preferences file or cached module from a previous version must NEVER prevent registration.

## What Goes Wrong

When a user upgrades, two sources of stale data exist:

1. **Saved prefs JSON** (`Exploratory_prefs.json` in Blender config) — contains property keys/values from the old version that may no longer exist or have changed type.
2. **Python module cache** (`sys.modules`) — if Blender hot-reloads without restarting, old submodule code stays cached in memory.

If either of these causes an unhandled exception during `register()`, the addon silently fails to enable. The user sees it installed but the checkbox won't turn on. No error is shown in the UI.

## How It's Protected

### `__init__.py` — Hot-reload block
All submodules are explicitly `importlib.reload()`-ed before name imports. This ensures a version upgrade mid-session doesn't use stale cached modules.

### `__init__.py` — Worker detection
Uses `try: import bpy` instead of checking `sys.modules` contents. This is immune to engine multiprocessing polluting the module cache.

### `prefs_persistence.py` — Stale key handling
- `apply_prefs()` checks each key against `prefs.bl_rna.properties` — unknown keys are skipped.
- Every `setattr()` call is wrapped in `try/except` — a single bad value can't abort the whole restore.
- `read_prefs()` catches malformed JSON and returns `{}`.

## When Changing Preferences

Before renaming, removing, or changing the type of any property in `ExploratoryAddonPreferences`:

1. The saved JSON may still have the old key/value. Confirm `apply_prefs()` handles it (it should — unknown keys are skipped, bad values are caught).
2. Blender's internal RNA storage may have the old value. This produces cosmetic `WARNING` logs on first load after upgrade — that's fine, they're harmless and clear themselves.
3. Never let a preferences change break `register()`. If in doubt, wrap the risky path in `try/except`.
