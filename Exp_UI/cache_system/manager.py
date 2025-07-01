# File: Exploratory/Exp_UI/cache_system/manager.py

import threading
from .persistence import (
    register_thumbnail_in_index,
    get_cached_path_if_exists,
    update_thumbnail_access,
)
from ..interface.operators.utilities import fetch_packages
from .db import save_package_list, load_package_list

class CacheManager:
    def __init__(self):
        # Holds full package lists keyed by file_type
        self.package_data   = {}
        # In-memory metadata cache
        self.metadata_cache = {}
        self.lock = threading.Lock()

    def set_package_data(self, data: dict[str, list[dict]]):
        """Replace the entire package_data dict with `data`."""
        with self.lock:
            self.package_data = data

    def get_package_data(self) -> dict[str, list[dict]]:
        """Return the entire package_data dict."""
        with self.lock:
            return self.package_data

    def get_metadata(self, package_id: int) -> dict | None:
        """Return in-memory metadata or None."""
        with self.lock:
            return self.metadata_cache.get(package_id)

    def set_metadata(self, package_id: int, metadata: dict):
        """Store metadata in memory (persisted separately)."""
        with self.lock:
            self.metadata_cache[package_id] = metadata

    # ─── Thumbnail Methods (SQLite-backed) ─────────────────────────────────────

    def get_thumbnail(self, file_id: int) -> dict | None:
        """
        Return {'file_path': path} if cached in SQLite, bump access.
        Otherwise None.
        """
        path = get_cached_path_if_exists(file_id)
        if path:
            update_thumbnail_access(file_id)
            return {"file_path": path}
        return None

    def set_thumbnail(self, file_id: int, local_path: str, thumbnail_url: str | None = None):
        """Persist a thumbnail record."""
        register_thumbnail_in_index(file_id, local_path, thumbnail_url)

    def update_thumbnail_access(self, file_id: int):
        """Bump the thumbnail last_access timestamp."""
        update_thumbnail_access(file_id)

    # ─── Package List Methods (SQLite-backed) ────────────────────────────────

    def ensure_package_data(self, file_type: str, limit: int = 50) -> bool:
        """
        1) Try load package list for `file_type` from SQLite.
        2) If found, load into memory and return True.
        3) Otherwise fetch from server, persist to SQLite, then load.
        """
        # 1) load from DB
        pkg_list = load_package_list(file_type)
        if pkg_list:
            self.set_package_data({file_type: pkg_list})
            return True

        # 2) fetch from server
        try:
            resp = fetch_packages({
                "file_type": file_type,
                "sort_by":   "newest",
                "offset":    0,
                "limit":     limit,
            })
            if not resp.get("success"):
                return False
            packages = resp["packages"]
        except Exception:
            return False

        # 3) persist then load
        save_package_list(file_type, packages)
        self.set_package_data({file_type: packages})
        return True

# singleton instance
cache_manager = CacheManager()


def filter_cached_data(file_type: str, search_query: str) -> list[dict]:
    """
    Return the cached packages of `file_type` whose name or uploader
    matches `search_query`.
    """
    all_data = cache_manager.get_package_data().get(file_type, [])
    sq = search_query.lower().strip()

    def matches(pkg: dict) -> bool:
        if pkg.get("file_type") != file_type:
            return False
        name     = pkg.get("package_name", "").lower()
        uploader = pkg.get("uploader", "").lower()
        return not sq or (sq in name or sq in uploader)

    return [pkg for pkg in all_data if matches(pkg)]
