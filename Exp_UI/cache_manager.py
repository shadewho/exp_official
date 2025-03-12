# cache_manager.py
import threading
import time
import os

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

def ensure_package_data():
    """
    Example helper that fetches page 1 from the server and stores it in cache_manager.
    (You can use or ignore this as needed.)
    """
    from .exp_api import fetch_packages
    params = {
        "file_type": "world",
        "sort_by": "newest",
        "offset": 0,
        "limit": 50
    }
    try:
        data = fetch_packages(params)
        if data.get("success"):
            packages = data.get("packages", [])
            cache_manager.set_package_data({1: packages})
            return True
        else:
            return False
    except Exception as e:
        return False

def filter_cached_data(file_type, search_query):
    """
    Filter the cached package data based on the file_type and search_query.
    It fetches cached packages from page 1 of the cache.

    Args:
        file_type (str): The type of package, e.g. "world" or "shop_item".
        search_query (str): The search query string.

    Returns:
        list: A list of package dictionaries that match the criteria.
    """
    # Get all cached packages from page 1 (or your main cache)
    all_data = cache_manager.get_package_data().get(1, [])
    sq = search_query.lower()
    def matches(pkg):
        if pkg.get("file_type") != file_type:
            return False
        name = pkg.get("package_name", "").lower()
        uploader = pkg.get("uploader", "").lower()
        if sq and (sq not in name and sq not in uploader):
            return False
        return True
    return [pkg for pkg in all_data if matches(pkg)]
