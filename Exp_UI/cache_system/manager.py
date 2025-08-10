# File: Exploratory/Exp_UI/cache_system/manager.py

import bpy
import threading
from .persistence import (
    register_thumbnail_in_index,
    get_cached_path_if_exists,
    update_thumbnail_access,
)
from ..interface.operators.utilities import fetch_packages
from .db import save_package_list, load_package_list
from typing import List, Dict


def dedupe_by_id(seq: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for item in seq:
        try:
            fid = int(item.get("file_id", -1))   # '878' and 878 are the same
        except Exception:
            fid = -1
        if fid not in seen:
            seen.add(fid)
            out.append(item)
    return out


class CacheManager:
    """
    Keeps in-memory package lists + metadata, thread-safe.
    """

    def __init__(self):
        self.package_data: Dict[str, List[Dict]] = {}
        self.metadata_cache: Dict[int, dict] = {}
        self.lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────
    #  Package lists
    # ──────────────────────────────────────────────────────────────────
    def set_package_data(self, data: Dict[str, List[Dict]]) -> None:
        """
        Merge new lists into memory, replacing only the specified file_types,
        **and ensure no duplicate file_ids can exist inside any list**.
        """
        with self.lock:
            for ftype, pkg_list in data.items():
                self.package_data[ftype] = dedupe_by_id(pkg_list)

    def get_package_data(self) -> Dict[str, List[Dict]]:
        """
        Return a shallow copy so callers can’t mutate our internal cache.
        """
        with self.lock:
            return {k: v[:] for k, v in self.package_data.items()}

    def get_metadata(self, package_id: int) -> dict | None:
        """
        Return metadata from memory or SQLite.
        Loads from SQLite into the in-memory cache on first access.
        """
        with self.lock:
            if package_id in self.metadata_cache:
                return self.metadata_cache[package_id]

        # Fallback to SQLite
        from .persistence import get_cached_metadata
        data = get_cached_metadata(package_id)
        if data is not None:
            with self.lock:
                self.metadata_cache[package_id] = data
        return data

    def set_metadata(self, package_id: int, metadata: dict) -> None:
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

    def set_thumbnail(self, file_id: int, local_path: str, thumbnail_url: str | None = None) -> None:
        """Persist a thumbnail record."""
        register_thumbnail_in_index(file_id, local_path, thumbnail_url)

    def update_thumbnail_access(self, file_id: int) -> None:
        """Bump the thumbnail last_access timestamp."""
        update_thumbnail_access(file_id)

    # ─── Package List Methods (SQLite-backed, unified + event filters) ─────────

    def ensure_package_data(self, file_type: str, page_size: int = 50) -> bool:
        """
        1) If file_type=='event' and no event is selected, clear data and return.
        2) Try loading from SQLite cache (with both filters).
        3) Otherwise fetch from server in pages, persist and merge.
        """
        scene = bpy.context.scene
        event_stage = scene.event_stage    if file_type == 'event' else ""
        selected_event = scene.selected_event if file_type == 'event' else ""

        # ── REQUIRE explicit event selection ──────────────────────────────
        if file_type == 'event' and (not selected_event or selected_event == "0"):
            # No event chosen → show nothing
            self.set_package_data({file_type: []})
            return True

        # ── 1) Attempt to load from SQLite cache ───────────────────────────
        cached = load_package_list(file_type, event_stage, selected_event)
        if cached:
            for pkg in cached:
                pkg.update({
                    'file_type':      file_type,
                    'event_stage':    event_stage,
                    'selected_event': selected_event,
                })
            self.set_package_data({file_type: cached})
            return True

        # ── 2) No cache: fetch full list in pages ─────────────────────────
        all_packages = []
        offset = 0
        while True:
            params = {
                "file_type": file_type,
                "sort_by":   "newest",
                "offset":    offset,
                "limit":     page_size,
            }
            if file_type == 'event':
                params["event_stage"]    = event_stage
                params["selected_event"] = selected_event

            try:
                resp = fetch_packages(params)
                if not resp.get("success"):
                    return False
            except Exception:
                return False

            batch = resp.get("packages", [])
            if not batch:
                break

            for pkg in batch:
                pkg.update({
                    'file_type':      file_type,
                    'event_stage':    event_stage,
                    'selected_event': selected_event,
                })
            all_packages.extend(batch)

            if len(batch) < page_size:
                break
            offset += page_size

        # ── 3) Persist + merge ─────────────────────────────────────────────
        save_package_list(file_type, all_packages, event_stage, selected_event)
        self.set_package_data({file_type: all_packages})
        return True

# singleton instance
cache_manager = CacheManager()

def filter_cached_data(
    file_type: str,
    search_query: str
) -> List[Dict]:
    """
    Return only those packages matching:
      • file_type
      • For events: **both** event_stage & selected_event must match (and selected_event != '0')
      • Substring in package_name or uploader
    """
    scene = bpy.context.scene
    sq = search_query.lower().strip()
    all_data = cache_manager.get_package_data().get(file_type, [])

    filtered: List[Dict] = []
    for pkg in all_data:
        # 1) file_type must match
        if pkg.get("file_type") != file_type:
            continue

        # 2) event‐specific filtering
        if file_type == "event":
            current_stage = getattr(scene, "event_stage", "")
            current_event = getattr(scene, "selected_event", "")

            # require an explicit event pick
            if not current_event or current_event == "0":
                continue
            # must match both stage and event ID
            if pkg.get("event_stage") != current_stage:
                continue
            if pkg.get("selected_event") != current_event:
                continue

        # 3) text filter
        if sq:
            name = pkg.get("package_name", "").lower()
            uploader = pkg.get("uploader", "").lower()
            if sq not in name and sq not in uploader:
                continue

        filtered.append(pkg)

    return filtered
