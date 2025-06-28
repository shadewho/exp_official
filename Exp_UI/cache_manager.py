# cache_manager.py
import threading
import time
class CacheManager:
    def __init__(self):
        self.package_data = {}      # e.g. {page_number: [package_dict, ...]}
        self.metadata_cache = {}    # e.g. {package_id: metadata_dict}
        # Previously keyed by thumbnail URL; now we key by file_id
        self.thumbnail_index = {}   # e.g. {file_id: {"file_path": ..., "last_access": ...}}
        self.lock = threading.Lock()

    def set_package_data(self, data):
        """
        Replaces the entire package_data dict with `data`.
        e.g. data might be {1: [pkgA, pkgB], 2: [...], ...}
        """
        with self.lock:
            self.package_data = data

    def get_package_data(self):
        """
        Returns the entire package_data dict: {page_number: [dict, dict, ...]}
        """
        with self.lock:
            return self.package_data

    def get_metadata(self, package_id):
        with self.lock:
            return self.metadata_cache.get(package_id)

    def set_metadata(self, package_id, metadata):
        with self.lock:
            self.metadata_cache[package_id] = metadata

    # --- Thumbnail Methods Updated to Use file_id ---
    def get_thumbnail(self, file_id):
        """
        Return the thumbnail record (dict with "file_path" and "last_access") if it exists,
        keyed by file_id.
        Example return: { "file_path": "/full/path/to.jpg", "last_access": 1679350847.123 }
        or None if not cached.
        """
        with self.lock:
            return self.thumbnail_index.get(file_id)

    def set_thumbnail(self, file_id, local_path):
        """
        Store the local thumbnail path into thumbnail_index, keyed by file_id,
        with an updated last_access time.
        """
        with self.lock:
            self.thumbnail_index[file_id] = {
                "file_path": local_path,
                "last_access": time.time()
            }

    def update_thumbnail_access(self, file_id):
        with self.lock:
            if file_id in self.thumbnail_index:
                self.thumbnail_index[file_id]["last_access"] = time.time()

# Create a singleton instance
cache_manager = CacheManager()

def ensure_package_data(file_type: str, limit: int = 50) -> bool:
    """
    Fetch page 1 of `file_type` from the server and store it in cache_manager
    under cache_manager.package_data[file_type].

    Returns True on success, False otherwise.
    """
    from .exp_api import fetch_packages

    params = {
        "file_type":  file_type,       # e.g. "world" or "shop_item"
        "sort_by":    "newest",
        "offset":     0,
        "limit":      limit,
    }
    try:
        data = fetch_packages(params)
        if not data.get("success"):
            return False

        packages = data.get("packages", [])
        # Instead of a numeric page key, use file_type as the key
        cache_manager.set_package_data({file_type: packages})
        return True

    except Exception:
        return False

def filter_cached_data(file_type: str, search_query: str) -> list[dict]:
    """
    Return the cached packages of `file_type` whose name or uploader
    matches `search_query`.

    Args:
        file_type (str): e.g. "world" or "shop_item".
        search_query (str): the user’s filter string.

    Returns:
        list[dict]: all matching package dicts, or an empty list if none.
    """
    # Grab the preloaded list for this type (or empty if not yet fetched)
    all_data = cache_manager.get_package_data().get(file_type, [])
    sq = search_query.lower().strip()

    def matches(pkg: dict) -> bool:
        if pkg.get("file_type") != file_type:
            return False
        name     = pkg.get("package_name", "").lower()
        uploader = pkg.get("uploader", "").lower()
        # If there's a query, require it appear in name or uploader
        return not sq or (sq in name or sq in uploader)

    return [pkg for pkg in all_data if matches(pkg)]