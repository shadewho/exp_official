#Exploraatory/Exp_UI/version_info.py
import os
import ast
import requests
from .main_config import BASE_URL

# locate the add-on’s __init__.py (one folder up from this file’s parent)
_root_dir  = os.path.dirname(os.path.dirname(__file__))
_init_path = os.path.join(_root_dir, "__init__.py")

# default version tuple
_version = (0, 0, 0)

with open(_init_path, "r", encoding="utf-8") as f:
    src = f.read()

module = ast.parse(src, _init_path)
for node in module.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "bl_info":
                # scan its dict entries for "version"
                for key_node, val_node in zip(node.value.keys, node.value.values):
                    if isinstance(key_node, ast.Constant) and key_node.value == "version":
                        _version = ast.literal_eval(val_node)
                break
        break

# expose as "X.Y.Z"
CURRENT_VERSION = ".".join(str(n) for n in _version)


### Version Control###
_latest_version_cache: str | None = None

def fetch_latest_version() -> str | None:
    """
    Hits the server and returns the latest version string,
    or None on error.
    """
    try:
        resp = requests.get(f"{BASE_URL}/api/addon_version", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            return data["version_string"]
    except Exception as e:
        print("Could not fetch latest version:", e)
    return None

def update_latest_version_cache() -> None:
    """
    Refreshes the in-memory cache.
    """
    global _latest_version_cache
    _latest_version_cache = fetch_latest_version()

def get_cached_latest_version() -> str | None:
    """
    Returns the last-fetched version, or None if never fetched/failed.
    """
    return _latest_version_cache