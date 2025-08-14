#Exploratory/Exp_UI/cache_system/preload.py

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
SWEEP_INTERVAL = 120.0    #600.0 # seconds between full list sweeps
BATCH_SIZE     = 16      # how many thumb/meta ops at once
IDLE_SLEEP     = 30.0    #60.0 # seconds to sleep when no work
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
        init_db()
        self.conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA busy_timeout = 30000;")

        while not self.stop_event.is_set():
            self._process_batch()

            now = time.time()
            if now - self.last_sweep >= SWEEP_INTERVAL:
                self._do_full_sweep()
                self.last_sweep = now

            time.sleep(BUSY_SLEEP if not self.task_queue.empty() else IDLE_SLEEP)

    def _process_batch(self):
        # ❌ Previously: bailed out in GAME mode. We want background to keep working.
        items = []
        while len(items) < BATCH_SIZE:
            try:
                items.append(self.task_queue.get_nowait())
            except queue.Empty:
                break

        if not items:
            return

        for kind, *args in items:
            if kind == 'meta':
                (pkg_id,) = args
                background_fetch_metadata(pkg_id)
            elif kind == 'thumb':
                pkg_id, url = args
                download_thumbnail(url, pkg_id)

        # Mark UI dirty if it exists; harmless if no UI.
        bpy.app.timers.register(
            lambda: setattr(bpy.types.Scene, "package_ui_dirty", True),
            first_interval=0.1
        )

    def _preload_all_events(self):
        if not load_token():
            return

        events = fetch_events_by_stage()
        if not events.get("success"):
            return

        # Aggregate ALL events across stages into one memory union
        mem_union: list[dict] = []

        for stage in ("submission", "voting", "winners"):
            ev_list = events.get(stage, []) or []
            for ev in ev_list:
                evt_id = str(ev["id"])
                all_pkgs = []
                offset = 0

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

                # Persist this partition and enqueue assets
                if all_pkgs:
                    update_package_list("event", all_pkgs, stage, evt_id)
                    mem_union.extend(all_pkgs)
                    for pkg in all_pkgs:
                        try:
                            pid = int(pkg["file_id"])
                            background_fetch_metadata(pid)
                        except Exception as e:
                            print(f"[PRELOAD] metadata enqueue failed for {pkg}: {e}")

        # One write to memory after the whole traversal (prevents overwrites)
        if mem_union:
            cache_manager.set_package_data({"event": mem_union})

        # Prune event partitions that no longer exist in that stage
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
        except Exception as e:
            print(f"[CacheWorker] _preload_all_events prune error: {e}")

    def _do_full_sweep(self):
        # Debug print kept
        print(
            f"[DEBUG sweep] {time.strftime('%H:%M:%S')}  "
            f"mode={bpy.context.scene.ui_current_mode!s}  "
            f"token={bool(load_token())}"
        )
        token = load_token()
        if not token:
            print("[CacheWorker] no API token, skipping full sweep until login")
            return

        # 1) All events
        self._preload_all_events()

        # 2) world & shop_item as before
        for ftype in FILE_TYPES:
            try:
                resp = fetch_packages({
                    "file_type": ftype,
                    "sort_by":   "newest",
                    "offset":    0,
                    "limit":     DOWNLOAD_LIMIT,
                })
            except Exception:
                continue

            if not resp.get("success"):
                continue

            remote_pkgs = resp["packages"]

            update_package_list(ftype, remote_pkgs, event_stage="", selected_event="")
            for pkg in remote_pkgs:
                pkg.update({
                    "file_type":      ftype,
                    "event_stage":    "",
                    "selected_event": ""
                })
            cache_manager.set_package_data({ftype: remote_pkgs})

            for pkg in remote_pkgs:
                pid = pkg.get("file_id")
                url = pkg.get("thumbnail_url")
                if pid is None or not url:
                    continue

                cur = self.conn.execute(
                    "SELECT thumbnail_url, file_path FROM thumbnails WHERE file_id = ? LIMIT 1;",
                    (pid,)
                ).fetchone()
                db_url, db_path = (cur[0], cur[1]) if cur else (None, None)
                if db_url != url:
                    if db_path and os.path.exists(db_path):
                        try: os.remove(db_path)
                        except: pass
                    self.task_queue.put(('thumb', pid, url))

                if cache_manager.get_metadata(pid) is None:
                    self.task_queue.put(('meta', pid))

        # 3) Event thumbnails stale-check remains (unchanged)
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
                        try: os.remove(db_path)
                        except: pass
                    self.task_queue.put(('thumb', pkg_id, url))
        except Exception as e:
            print(f"[CacheWorker] event thumbnail stale-check error:", e)

        # 4) Collect valid IDs from package_list and prune assets (unchanged)
        cursor = self.conn.execute("""
            SELECT DISTINCT CAST(json_extract(data_json, '$.file_id') AS INTEGER)
            FROM package_list;
        """)
        valid_ids = {row[0] for row in cursor.fetchall() if row[0] is not None}

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

        # Force the first sweep in the next loop iteration
        _worker.last_sweep = time.time() - (SWEEP_INTERVAL + 1)


def stop_cache_worker():
    global _worker
    if _worker is not None:
        _worker.stop()
        _worker = None
