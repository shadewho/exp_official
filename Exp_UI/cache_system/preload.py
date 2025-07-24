import os
import time
import threading
import queue
import sqlite3
import requests
import bpy
from requests.exceptions import RequestException
from ..main_config import THUMBNAIL_CACHE_FOLDER, PACKAGES_INDEX_ENDPOINT
from ..auth.helpers import load_token
from .db import init_db, update_package_list, purge_orphaned_thumbnails, load_package_list
from .download_helpers import background_fetch_metadata, download_thumbnail
from .manager import fetch_packages, cache_manager
from .persistence import clear_image_datablocks
from ..events.utilities import fetch_events_by_stage  # ← preload helper

# Path to your SQLite file:
DB_PATH = os.path.join(THUMBNAIL_CACHE_FOLDER, "cache.db")

# What to cache and how often:
FILE_TYPES     = ['world', 'shop_item']
SWEEP_INTERVAL = 300.0    # seconds between full list sweeps
BATCH_SIZE     = 16      # how many thumb/meta ops at once
IDLE_SLEEP     = 30.0    # seconds to sleep when no work
BUSY_SLEEP     = 1.0     # seconds to sleep when queue still has items
DOWNLOAD_LIMIT = 999

_worker = None  # singleton

def fetch_package_index(file_type: str) -> list[dict]:
    token = load_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(
            PACKAGES_INDEX_ENDPOINT,
            headers=headers,
            params={"file_type": file_type},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("packages", [])
    except RequestException as e:
        return []


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
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA busy_timeout = 30000;")
        

        while not self.stop_event.is_set():
            # 2) Process thumbnail/metadata tasks
            self._process_batch()

            # 3) Periodic full‐list sweep (including preloading all events)
            now = time.time()
            if now - self.last_sweep >= SWEEP_INTERVAL:
                self._do_full_sweep()
                self.last_sweep = now

            # 4) Sleep based on queue load
            if not self.task_queue.empty():
                time.sleep(BUSY_SLEEP)
            else:
                time.sleep(IDLE_SLEEP)

    def _process_batch(self):
        # 0) don’t do any work if we’re in GAME mode
        if bpy.context.scene.ui_current_mode == 'GAME':
            return
        
        items = []
        while len(items) < BATCH_SIZE:
            try:
                items.append(self.task_queue.get_nowait())
            except queue.Empty:
                break

        if not items:
            return

        # Batch writes in one transaction
        with self.conn:
            for kind, *args in items:
                if kind == 'meta':
                    (pkg_id,) = args
                    background_fetch_metadata(pkg_id)
                elif kind == 'thumb':
                    pkg_id, url = args
                    download_thumbnail(url, pkg_id)

        # Trigger one-shot UI redraw
        bpy.app.timers.register(
            lambda: setattr(bpy.types.Scene, "package_ui_dirty", True),
            first_interval=0.1
        )

    def _preload_all_events(self):
        # 1) guard: only if we have a valid token
        if not load_token():
            return

        # 2) fetch events; JSON has 'success' + one list per stage
        events = fetch_events_by_stage()
        if not events.get("success"):
            return

        # 3) loop only your three known stages
        for stage in ("submission", "voting", "winners"):
            ev_list = events.get(stage, [])
            for ev in ev_list:
                evt_id   = str(ev["id"])
                all_pkgs = []
                offset   = 0

                # 3a) page through every package for this (stage, event)
                while True:
                    params = {
                        "file_type":      "event",
                        "sort_by":        "newest",
                        "event_stage":    stage,
                        "selected_event": evt_id,
                        "offset":         offset,
                        "limit":          DOWNLOAD_LIMIT,
                    }
                    resp = fetch_packages(params)
                    if not resp.get("success"):
                        break
                    batch = resp.get("packages", [])
                    if not batch:
                        break

                    # annotate and collect
                    for pkg in batch:
                        pkg.update({
                            "file_type":      "event",
                            "event_stage":    stage,
                            "selected_event": evt_id,
                        })
                    all_pkgs.extend(batch)

                    if len(batch) < 50:
                        break
                    offset += 50

                # 3b) persist into SQLite + in-memory cache
                if all_pkgs:
                    update_package_list("event", all_pkgs, stage, evt_id)
                    cache_manager.set_package_data({"event": all_pkgs})

                    # 3c) enqueue metadata & thumbnail fetch for each package
                    for pkg in all_pkgs:
                        try:
                            pkg_id = int(pkg["file_id"])
                            background_fetch_metadata(pkg_id)
                        except Exception as e:
                            print(f"[PRELOAD] failed to enqueue metadata for {pkg}: {e}")

        # 4) prune any stale event entries
        try:
            for stage in ("submission", "voting", "winners"):
                active_ids = [str(ev["id"]) for ev in events.get(stage, [])]
                with self.conn:
                    if not active_ids:
                        self.conn.execute(
                            "DELETE FROM package_list WHERE file_type='event' AND event_stage=?;",
                            (stage,)
                        )
                    else:
                        placeholders = ",".join("?" for _ in active_ids)
                        sql = (
                            "DELETE FROM package_list "
                            " WHERE file_type='event'"
                            "   AND event_stage=?"
                            f"   AND selected_event NOT IN ({placeholders});"
                        )
                        self.conn.execute(sql, (stage, *active_ids))

            sub = len(events.get("submission", []))
            vote = len(events.get("voting", []))
            win = len(events.get("winners", []))
        except Exception as e:
            print(f"[CacheWorker] _preload_all_events prune error: {e}")



    def _do_full_sweep(self):
        
        # 0) bail out immediately when the UI is in GAME
        if bpy.context.scene.ui_current_mode == 'GAME':
            return

        token = load_token()
        if not token:
            print("[CacheWorker] no API token, skipping full sweep until login")
            return

        # 1) Preload ALL events (all stages & IDs)
        self._preload_all_events()

        # 2) Re-fetch & update each non-event file_type (world & shop_item)
        for ftype in FILE_TYPES:
            try:
                resp = fetch_packages({
                    "file_type": ftype,
                    "sort_by":   "newest",
                    "offset":    0,
                    "limit":     DOWNLOAD_LIMIT,  # or 999 if you really want “all”
                })
            except Exception as e:
                continue

            if not resp.get("success"):
                continue

            remote_pkgs = resp["packages"]

            # 2a) Incrementally persist into SQLite & in-memory cache
            update_package_list(ftype, remote_pkgs, event_stage="", selected_event="")
            for pkg in remote_pkgs:
                pkg.update({
                    "file_type":      ftype,
                    "event_stage":    "",
                    "selected_event": ""
                })
            cache_manager.set_package_data({ftype: remote_pkgs})

            # 2b) Enqueue metadata & thumbnail downloads
            for pkg in remote_pkgs:
                pid = pkg.get("file_id")
                url = pkg.get("thumbnail_url")
                if pid is None or not url:
                    continue

                # metadata
                if cache_manager.get_metadata(pid) is None:
                    self.task_queue.put(('meta', pid))

                # thumbnail (only if URL changed)
                cur = self.conn.execute(
                    "SELECT thumbnail_url, file_path FROM thumbnails WHERE file_id = ? LIMIT 1;",
                    (pid,)
                ).fetchone()
                db_url, db_path = (cur[0], cur[1]) if cur else (None, None)
                if db_url != url:
                    if db_path and os.path.exists(db_path):
                        try:
                            os.remove(db_path)
                        except:
                            pass
                    self.task_queue.put(('thumb', pid, url))

        # 3) Stale-check **event** thumbnails & requeue if URL changed
        try:
            import json
            cursor = self.conn.execute(
                "SELECT data_json FROM package_list WHERE file_type='event';"
            )
            for (data_json_str,) in cursor.fetchall():
                pkg = json.loads(data_json_str)
                pkg_id = int(pkg.get("file_id", 0))
                url    = pkg.get("thumbnail_url")

                row = self.conn.execute(
                    "SELECT thumbnail_url, file_path FROM thumbnails WHERE file_id = ? LIMIT 1;",
                    (pkg_id,),
                ).fetchone()
                db_url, db_path = (row[0], row[1]) if row else (None, None)

                if url and db_url != url:
                    if db_path and os.path.exists(db_path):
                        try:
                            os.remove(db_path)
                        except:
                            pass
                    self.task_queue.put(('thumb', pkg_id, url))
        except Exception as e:
            print(f"[CacheWorker] event thumbnail stale-check error:", e)

        # 4) Determine the full set of “still valid” IDs (world, shop_item, AND events)
        cursor = self.conn.execute("""
            SELECT DISTINCT CAST(json_extract(data_json, '$.file_id') AS INTEGER)
            FROM package_list;
        """)
        valid_ids = {row[0] for row in cursor.fetchall() if row[0] is not None}

        # 5) Prune any thumbnails/metadata rows not in that valid set
        if valid_ids:
            placeholders = ",".join("?" for _ in valid_ids)
            with self.conn:
                self.conn.execute(
                    f"DELETE FROM thumbnails WHERE file_id NOT IN ({placeholders});",
                    tuple(valid_ids)
                )
                self.conn.execute(
                    f"DELETE FROM metadata   WHERE package_id NOT IN ({placeholders});",
                    tuple(valid_ids)
                )

        # 6) Finally schedule your existing on-main-thread cleanup
        bpy.app.timers.register(self._main_cleanup, first_interval=0.1)


        
    def _main_cleanup(self):
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
