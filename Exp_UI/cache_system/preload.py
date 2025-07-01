# File: Exploratory/Exp_UI/cache_system/preload.py

import os
import time
import sqlite3

import bpy
import threading
from .download_helpers import background_fetch_metadata, download_thumbnail
from .manager import cache_manager
from ..main_config import THUMBNAIL_CACHE_FOLDER
from .db import evict_old_thumbnails, evict_old_metadata, get_thumbnail_path
from .persistence import get_or_load_image, get_or_create_texture
from collections import deque

# module‐level queue of tasks: ('meta', pkg_id) or ('thumb', pkg_id, url)
_last_validation_time = 0
_prep_queue: deque[tuple] = deque()
_queue_populated = False
BATCH_SIZE = 8
FILE_TYPES = ['world', 'shop_item', 'event']


def preload_in_memory_thumbnails():
    """
    Preloads all thumbnails recorded in the SQLite cache into Blender's GPU textures.
    Loads one image per timer tick (0.1s) so the UI stays responsive.
    """
    db_path = os.path.join(THUMBNAIL_CACHE_FOLDER, "cache.db")
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM thumbnails;")
        rows = cursor.fetchall()
    except Exception as e:
        print(f"[ERROR] preload_in_memory_thumbnails: failed to query DB: {e}")
        return
    finally:
        conn.close()

    paths = [row[0] for row in rows if row and os.path.exists(row[0])]

    def load_next_thumbnail():
        if not paths:
            return None  # stop the timer
        path = paths.pop(0)
        img = get_or_load_image(path)
        if img:
            get_or_create_texture(img)
        return 0.1  # schedule next load in 0.1s

    bpy.app.timers.register(load_next_thumbnail)


# ──────────────────────────────────────────────────────────────────────────────
# PRELOAD TIMER ─ tries every tick until the cache is primed, then backs off
# ──────────────────────────────────────────────────────────────────────────────
def preload_metadata_timer():
    """
    Background heartbeat that (1) keeps the SQLite cache fresh and
    (2) downloads thumbnails/metadata long before the UI is opened.

    It now:
      • Rebuilds the work-queue every time the queue is empty.
      • Processes a small batch each tick so the UI stays responsive.
      • Backs off automatically once the queue is drained.
    """
    global _last_validation_time, _prep_queue

    # Skip while the user is in GAME mode.
    if getattr(bpy.context.scene, "ui_current_mode", "BROWSE") == "GAME":
        return 60.0

    # ── 1) Rebuild the queue if it’s empty ───────────────────────────────────
    if not _prep_queue:
        for ftype in FILE_TYPES:
            # make sure we have a package list (from SQLite or the server)
            if not cache_manager.ensure_package_data(ftype):
                continue

            for pkg in cache_manager.get_package_data().get(ftype, []):
                pid = pkg.get("file_id")
                url = pkg.get("thumbnail_url")

                # metadata task
                if pid and cache_manager.get_metadata(pid) is None:
                    _prep_queue.append(("meta", pid))

                # thumbnail task
                local = get_thumbnail_path(pid)
                if pid and url and (not local or not os.path.exists(local)):
                    _prep_queue.append(("thumb", pid, url))

    # ── 2) Drain up to BATCH_SIZE tasks ──────────────────────────────────────
    processed = 0
    while _prep_queue and processed < BATCH_SIZE:
        task = _prep_queue.popleft()
        kind = task[0]

        if kind == "meta":            # ('meta', pid)
            background_fetch_metadata(task[1])
        else:                         # ('thumb', pid, url)
            _, pid, url = task
            threading.Thread(
                target=download_thumbnail,
                args=(url, pid),
                daemon=True
            ).start()
        processed += 1

    # ── 3) Every ~2 min purge stale DB rows + orphaned files ────────────────
    now = time.time()
    if now - _last_validation_time > 120:
        evict_old_thumbnails(max_age_days=7)
        evict_old_metadata(max_age_days=1)
        for f in os.listdir(THUMBNAIL_CACHE_FOLDER):
            path = os.path.join(THUMBNAIL_CACHE_FOLDER, f)
            if os.path.isfile(path) and (now - os.path.getmtime(path)) > 7*86400:
                try:
                    os.remove(path)
                except Exception:
                    pass
        _last_validation_time = now

    # ── 4) Tell Blender when to call us again ───────────────────────────────
    # If there’s still work, tick quickly; otherwise back off.
    return 5.0 if _prep_queue else 30.0
