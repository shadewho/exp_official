import os
import time
import threading
import queue
import sqlite3

import bpy
from ..main_config import THUMBNAIL_CACHE_FOLDER
from ..auth.helpers import load_token

from .db import init_db, update_package_list, purge_orphaned_thumbnails
from .download_helpers import background_fetch_metadata, download_thumbnail
from .manager import fetch_packages, cache_manager
from .persistence import clear_image_datablocks

# Path to your SQLite file:
DB_PATH = os.path.join(THUMBNAIL_CACHE_FOLDER, "cache.db")

# What to cache and how often:
FILE_TYPES = ['world', 'shop_item', 'event']
SWEEP_INTERVAL = 30.0    # seconds between full list sweeps (5 min)
BATCH_SIZE     = 16       # how many thumb/meta ops at once
IDLE_SLEEP     = 30.0     # seconds to sleep when no work
BUSY_SLEEP     = 1.0      # seconds to sleep when queue still has items

_worker = None  # singleton


class CacheWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.task_queue = queue.Queue()
        self.conn = None
        self.last_sweep = 0.0
        self.stop_event = threading.Event()

    def run(self):
        # 1) Ensure schema and open one DB connection
        init_db()
        self.conn = sqlite3.connect(
            DB_PATH, timeout=30, check_same_thread=False
        )
        # optimize for concurrency
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA busy_timeout = 30000;")

        while not self.stop_event.is_set():
            # 2) Process up to BATCH_SIZE thumbnail/metadata tasks
            self._process_batch()

            # 3) Periodic full‐list sweep
            now = time.time()
            if now - self.last_sweep >= SWEEP_INTERVAL:
                self._do_full_sweep()
                self.last_sweep = now

            # 4) Sleep: faster if there's more work, slower if idle
            if not self.task_queue.empty():
                time.sleep(BUSY_SLEEP)
            else:
                time.sleep(IDLE_SLEEP)

    def _process_batch(self):
        items = []
        while len(items) < BATCH_SIZE:
            try:
                items.append(self.task_queue.get_nowait())
            except queue.Empty:
                break

        if not items:
            return

        # Batch all DB writes in one transaction
        with self.conn:
            for kind, *args in items:
                if kind == 'meta':
                    (package_id,) = args
                    background_fetch_metadata(package_id)
                elif kind == 'thumb':
                    package_id, url = args
                    download_thumbnail(url, package_id)

        # Signal the UI to redraw _once_ after this batch
        bpy.app.timers.register(
            lambda: setattr(bpy.types.Scene, "package_ui_dirty", True),
            first_interval=0.1
        )

    def _do_full_sweep(self):
        # 1) Collect all file_ids seen on the server this sweep
        all_ids: set[int] = set()

        # 2) Fetch and incrementally update each file_type
        for ftype in FILE_TYPES:
            if not load_token():
                continue  # skip if logged out

            scene = bpy.context.scene
            filters = {}
            if ftype == 'event':
                filters = {
                    "event_stage":    scene.event_stage,
                    "selected_event": scene.selected_event
                }

            params = {
                "file_type": ftype,
                "sort_by":   "newest",
                "offset":    0,
                "limit":     9999,
                **filters
            }

            try:
                resp = fetch_packages(params)
                if not resp.get("success"):
                    continue
                remote_pkgs = resp["packages"]
            except Exception:
                continue

            # Track every file_id from the server
            for pkg in remote_pkgs:
                pid = pkg.get("file_id")
                if pid is not None:
                    all_ids.add(pid)

            # 2a) Update package_list & in-memory cache
            update_package_list(
                ftype,
                remote_pkgs,
                filters.get("event_stage", ""),
                filters.get("selected_event", "")
            )
            for pkg in remote_pkgs:
                pkg.update({
                    "file_type":      ftype,
                    "event_stage":    filters.get("event_stage", ""),
                    "selected_event": filters.get("selected_event", "")
                })
            cache_manager.set_package_data({ftype: remote_pkgs})

            # 2b) Enqueue any missing or changed metadata or thumbnails
            for pkg in remote_pkgs:
                pid = pkg.get("file_id")
                url = pkg.get("thumbnail_url")
                if pid is None or not url:
                    continue

                # metadata: only if never cached
                if cache_manager.get_metadata(pid) is None:
                    self.task_queue.put(('meta', pid))

                # thumbnail: check if URL changed or missing
                # Query current DB thumbnail_url & path in one shot
                cur = self.conn.execute(
                    "SELECT thumbnail_url, file_path FROM thumbnails WHERE file_id = ? LIMIT 1;",
                    (pid,)
                ).fetchone()

                db_url, db_path = (cur[0], cur[1]) if cur else (None, None)

                # If there's no record, or the URL changed, re-fetch
                if db_url != url:
                    # delete old file if it exists
                    if db_path and os.path.exists(db_path):
                        try:
                            os.remove(db_path)
                        except Exception:
                            pass

                    # enqueue download of the new thumbnail
                    self.task_queue.put(('thumb', pid, url))

        # 3) Prune deleted posts from thumbnails & metadata tables
        if all_ids:
            placeholders = ",".join("?" for _ in all_ids)
            with self.conn:
                # thumbnails for removed posts
                self.conn.execute(
                    f"DELETE FROM thumbnails WHERE file_id NOT IN ({placeholders});",
                    tuple(all_ids)
                )
                # metadata for removed posts
                self.conn.execute(
                    f"DELETE FROM metadata WHERE package_id NOT IN ({placeholders});",
                    tuple(all_ids)
                )

        # 4) Schedule main-thread cleanup to remove orphaned files & trigger UI refresh
        bpy.app.timers.register(self._main_cleanup, first_interval=0.1)


    def _main_cleanup(self):
        # purge orphaned files, clear GPU caches, and flag UI
        try:
            purge_orphaned_thumbnails()
            clear_image_datablocks()
            bpy.types.Scene.package_ui_dirty = True
        except Exception as e:
            print(f"[CacheWorker cleanup error] {e}")
        return None  # one-shot
    


    def stop(self):
        self.stop_event.set()


def start_cache_worker():
    global _worker
    if _worker is None:
        _worker = CacheWorker()
        _worker.start()


def stop_cache_worker():
    global _worker
    if _worker is not None:
        _worker.stop()
        _worker = None
