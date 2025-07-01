# File: Exploratory/Exp_UI/cache_system/db.py

import os
import sqlite3
import threading
import time
from ..main_config import THUMBNAIL_CACHE_FOLDER
import json
_DB_PATH = os.path.join(THUMBNAIL_CACHE_FOLDER, "cache.db")
_lock   = threading.Lock()

def _get_conn():
    # Ensure the cache directory exists
    os.makedirs(THUMBNAIL_CACHE_FOLDER, exist_ok=True)

    # Open the SQLite database, allowing other threads up to 30s to release locks
    conn = sqlite3.connect(
        _DB_PATH,
        timeout=30,             # how long the Python layer will wait for a lock
        check_same_thread=False # allow use from multiple threads
    )
    # Use write-ahead logging for concurrent readers
    conn.execute("PRAGMA journal_mode=WAL;")
    # Normal synchronous mode (a good trade-off for performance)
    conn.execute("PRAGMA synchronous=NORMAL;")
    # **CRUCIAL**: if the DB is locked, wait up to 30 000 ms before giving up
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn

def init_db():
    """Create thumbnails, metadata & package_list tables if they don’t exist."""
    with _lock:
        conn = _get_conn()
        cur  = conn.cursor()

        # 1) Thumbnails table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS thumbnails (
          file_id       INTEGER PRIMARY KEY,
          file_path     TEXT    NOT NULL,
          thumbnail_url TEXT,
          last_access   INTEGER NOT NULL
        );
        """)

        # 2) Metadata table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
          package_id   INTEGER PRIMARY KEY,
          data_json    TEXT    NOT NULL,
          last_access  INTEGER NOT NULL
        );
        """)

        # 3) Package lists for each file_type
        cur.execute("""
        CREATE TABLE IF NOT EXISTS package_list (
          file_type    TEXT    NOT NULL,
          data_json    TEXT    NOT NULL
        );
        """)
        # Optional index for faster lookups
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_package_list_file_type
          ON package_list(file_type);
        """)

        conn.commit()
        conn.close()
def drop_all_tables():
    """
    Drop both thumbnails & metadata tables entirely.
    Used by CLEAR_ALL_DATA_OT to start fresh.
    """
    with _lock:
        conn = _get_conn()
        conn.execute("DROP TABLE IF EXISTS thumbnails;")
        conn.execute("DROP TABLE IF EXISTS metadata;")
        conn.commit()
        conn.close()
    # Recreate empty schema
    init_db()

def evict_old_thumbnails(max_age_days: int = 7):
    """Remove only those thumbnails not accessed in the last max_age_days."""
    cutoff = int(time.time()) - max_age_days * 86400
    with _lock:
        conn = _get_conn()
        conn.execute("DELETE FROM thumbnails WHERE last_access < ?;", (cutoff,))
        conn.commit()
        conn.close()

def evict_all_thumbnails():
    """
    Immediately delete every record in thumbnails.
    Used by CLEAR_THUMBNAILS_ONLY_OT.
    """
    with _lock:
        conn = _get_conn()
        conn.execute("DELETE FROM thumbnails;")
        conn.commit()
        conn.close()

def evict_old_metadata(max_age_days: int = 1):
    """Remove metadata entries not accessed in the last max_age_days."""
    cutoff = int(time.time()) - max_age_days * 86400
    with _lock:
        conn = _get_conn()
        conn.execute("DELETE FROM metadata WHERE last_access < ?;", (cutoff,))
        conn.commit()
        conn.close()

# --- Thumbnail ops ---

def register_thumbnail(file_id: int, file_path: str, thumbnail_url: str = None):
    ts = int(time.time())
    with _lock:
        conn = _get_conn()
        conn.execute("""
          INSERT INTO thumbnails(file_id, file_path, thumbnail_url, last_access)
            VALUES (?, ?, ?, ?)
          ON CONFLICT(file_id) DO UPDATE SET
            file_path     = excluded.file_path,
            thumbnail_url = excluded.thumbnail_url,
            last_access   = excluded.last_access;
        """, (file_id, file_path, thumbnail_url, ts))
        conn.commit()
        conn.close()

def get_thumbnail_path(file_id: int) -> str | None:
    with _lock:
        conn = _get_conn()
        row = conn.execute(
          "SELECT file_path FROM thumbnails WHERE file_id = ? LIMIT 1;",
          (file_id,)
        ).fetchone()
        conn.close()
    return row[0] if row else None

def bump_thumbnail_access(file_id: int):
    ts = int(time.time())
    with _lock:
        conn = _get_conn()
        conn.execute(
          "UPDATE thumbnails SET last_access = ? WHERE file_id = ?;",
          (ts, file_id)
        )
        conn.commit()
        conn.close()

# --- Metadata ops ---

def register_metadata(package_id: int, data_json: str):
    ts = int(time.time())
    with _lock:
        conn = _get_conn()
        conn.execute("""
          INSERT INTO metadata(package_id, data_json, last_access)
            VALUES (?, ?, ?)
          ON CONFLICT(package_id) DO UPDATE SET
            data_json   = excluded.data_json,
            last_access = excluded.last_access;
        """, (package_id, data_json, ts))
        conn.commit()
        conn.close()

def get_metadata(package_id: int) -> str | None:
    with _lock:
        conn = _get_conn()
        row = conn.execute(
          "SELECT data_json FROM metadata WHERE package_id = ? LIMIT 1;",
          (package_id,)
        ).fetchone()
        conn.close()
    return row[0] if row else None

def bump_metadata_access(package_id: int):
    ts = int(time.time())
    with _lock:
        conn = _get_conn()
        conn.execute(
          "UPDATE metadata SET last_access = ? WHERE package_id = ?;",
          (ts, package_id)
        )
        conn.commit()
        conn.close()

def save_package_list(file_type: str, packages: list[dict]):
    """
    Overwrite the `package_list` table for this type.
    """
    conn = _get_conn()
    with conn:
        conn.execute("DELETE FROM package_list WHERE file_type = ?;", (file_type,))
        for pkg in packages:
            conn.execute(
                "INSERT INTO package_list (file_type, data_json) VALUES (?, ?);",
                (file_type, json.dumps(pkg))
            )

def load_package_list(file_type: str) -> list[dict]:
    """
    Return list of packages for this type, or [] if none.
    """
    conn = _get_conn()
    cur = conn.execute("SELECT data_json FROM package_list WHERE file_type=?;", (file_type,))
    rows = cur.fetchall()
    return [ json.loads(r[0]) for r in rows ] if rows else []

# Initialize schema on import
init_db()
