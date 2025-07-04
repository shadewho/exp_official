# File: Exploratory/Exp_UI/cache_system/db.py

import os
import sqlite3
import threading
import time
import json
import bpy
from ..main_config import THUMBNAIL_CACHE_FOLDER

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

        # 3) Package lists table with event filters
        cur.execute("""
        CREATE TABLE IF NOT EXISTS package_list (
          file_type       TEXT    NOT NULL,
          event_stage     TEXT    NOT NULL DEFAULT '',
          selected_event  TEXT    NOT NULL DEFAULT '',
          data_json       TEXT    NOT NULL
        );
        """)
        # Composite index for faster lookups by file_type + filters
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_package_list_filters
          ON package_list(file_type, event_stage, selected_event);
        """)

        conn.commit()
        conn.close()

def drop_all_tables():
    """
    Drop thumbnails, metadata & package_list tables entirely.
    Used by CLEAR_ALL_DATA_OT to start fresh.
    """
    with _lock:
        conn = _get_conn()
        conn.execute("DROP TABLE IF EXISTS thumbnails;")
        conn.execute("DROP TABLE IF EXISTS metadata;")
        conn.execute("DROP TABLE IF EXISTS package_list;")
        conn.commit()
        conn.close()
    # Recreate empty schema
    init_db()

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

# --- Thumbnail ops ---

def register_thumbnail(file_id: int, file_path: str, thumbnail_url: str = None):
    ts = int(time.time())
    with _lock:
        conn = _get_conn()
        try:
            conn.execute("""
              INSERT INTO thumbnails(file_id, file_path, thumbnail_url, last_access)
                VALUES (?, ?, ?, ?)
              ON CONFLICT(file_id) DO UPDATE SET
                file_path     = excluded.file_path,
                thumbnail_url = excluded.thumbnail_url,
                last_access   = excluded.last_access;
            """, (file_id, file_path, thumbnail_url, ts))
        except sqlite3.OperationalError as e:
            if "no such table" in str(e).lower():
                # Schema missing → recreate and retry once
                conn.close()
                init_db()
                conn = _get_conn()
                conn.execute("""
                  INSERT INTO thumbnails(file_id, file_path, thumbnail_url, last_access)
                    VALUES (?, ?, ?, ?)
                  ON CONFLICT(file_id) DO UPDATE SET
                    file_path     = excluded.file_path,
                    thumbnail_url = excluded.thumbnail_url,
                    last_access   = excluded.last_access;
                """, (file_id, file_path, thumbnail_url, ts))
            else:
                conn.close()
                raise
        conn.commit()
        conn.close()

def get_thumbnail_path(file_id: int) -> str | None:
    """
    Return the cached thumbnail path for file_id, or None.
    If the 'thumbnails' table doesn’t exist yet, init the schema and retry once.
    """
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT file_path FROM thumbnails WHERE file_id = ? LIMIT 1;",
                (file_id,)
            ).fetchone()
        except sqlite3.OperationalError as e:
            # If table is missing, initialize and retry once
            if "no such table" in str(e).lower():
                conn.close()
                init_db()
                conn = _get_conn()
                row = conn.execute(
                    "SELECT file_path FROM thumbnails WHERE file_id = ? LIMIT 1;",
                    (file_id,)
                ).fetchone()
            else:
                conn.close()
                raise
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
        try:
            conn.execute("""
              INSERT INTO metadata(package_id, data_json, last_access)
                VALUES (?, ?, ?)
              ON CONFLICT(package_id) DO UPDATE SET
                data_json   = excluded.data_json,
                last_access = excluded.last_access;
            """, (package_id, data_json, ts))
        except sqlite3.OperationalError as e:
            if "no such table" in str(e).lower():
                conn.close()
                init_db()
                conn = _get_conn()
                conn.execute("""
                  INSERT INTO metadata(package_id, data_json, last_access)
                    VALUES (?, ?, ?)
                  ON CONFLICT(package_id) DO UPDATE SET
                    data_json   = excluded.data_json,
                    last_access = excluded.last_access;
                """, (package_id, data_json, ts))
            else:
                conn.close()
                raise
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

# --- Package list ops with event filters ---

def save_package_list(
    file_type: str,
    packages: list[dict],
    event_stage: str = "",
    selected_event: str = ""
):
    """
    Overwrite the cache for this file_type + event_stage + selected_event.
    """
    conn = _get_conn()
    with conn:
        conn.execute(
            "DELETE FROM package_list WHERE file_type=? AND event_stage=? AND selected_event=?;",
            (file_type, event_stage, selected_event)
        )
        for pkg in packages:
            conn.execute(
                "INSERT INTO package_list (file_type, event_stage, selected_event, data_json) VALUES (?, ?, ?, ?);",
                (file_type, event_stage, selected_event, json.dumps(pkg))
            )

def load_package_list(
    file_type: str,
    event_stage: str = "",
    selected_event: str = ""
) -> list[dict]:
    """
    Return cached list for this file_type + filters, or [] if none.
    """
    conn = _get_conn()
    cur = conn.execute(
        "SELECT data_json FROM package_list WHERE file_type=? AND event_stage=? AND selected_event=?;",
        (file_type, event_stage, selected_event)
    )
    rows = cur.fetchall()
    conn.close()
    return [ json.loads(r[0]) for r in rows ] if rows else []


# ------------------------------------------------------------------
# Cache‐invalidator for the package_list table
# ------------------------------------------------------------------

def purge_orphaned_thumbnails():
    """
    Delete any files on-disk in THUMBNAIL_CACHE_FOLDER
    that are not present in the thumbnails table.
    """

    # 1) Gather all paths recorded in SQLite
    with _lock:
        conn = _get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM thumbnails;")
        recorded = {row[0] for row in cursor.fetchall()}
        conn.close()

    # 2) Remove any file on disk not in the recorded set
    for fname in os.listdir(THUMBNAIL_CACHE_FOLDER):
        path = os.path.join(THUMBNAIL_CACHE_FOLDER, fname)
        if os.path.isfile(path) and path not in recorded:
            try:
                os.remove(path)
            except Exception:
                pass

def update_package_list(
    file_type: str,
    packages: list[dict],
    event_stage: str = "",
    selected_event: str = ""
):
    """
    Incrementally update package_list for the given filter:
      1) Delete rows whose file_id no longer appears in 'packages'.
      2) Insert rows for new file_id values.
      3) Update existing rows only if the JSON changed.
    """
    # 1) Gather existing IDs
    existing = load_package_list(file_type, event_stage, selected_event)
    existing_ids = {int(pkg["file_id"]) for pkg in existing}
    new_ids = {int(pkg["file_id"]) for pkg in packages}

    to_delete = existing_ids - new_ids
    to_add    = new_ids - existing_ids
    # we'll treat everything else as "maybe update"
    maybe_update = existing_ids & new_ids

    ts = int(time.time())
    with _lock:
        conn = _get_conn()
        cur  = conn.cursor()

        # 2) Delete missing packages
        if to_delete:
            placeholders = ",".join("?" for _ in to_delete)
            cur.execute(f"""
                DELETE FROM package_list
                  WHERE file_type=? AND event_stage=? AND selected_event=?
                    AND json_extract(data_json, '$.file_id') IN ({placeholders});
            """, [file_type, event_stage, selected_event, *to_delete])

        # 3) Insert newly added packages
        for pkg in packages:
            fid = int(pkg["file_id"])
            if fid in to_add:
                cur.execute(
                    "INSERT INTO package_list (file_type, event_stage, selected_event, data_json) VALUES (?, ?, ?, ?);",
                    (file_type, event_stage, selected_event, json.dumps(pkg))
                )

        # 4) Update changed JSON for existing rows
        for pkg in packages:
            fid = int(pkg["file_id"])
            if fid in maybe_update:
                # compare the stored JSON vs. new JSON
                cur.execute(
                    "SELECT data_json FROM package_list WHERE file_type=? AND event_stage=? AND selected_event=? AND json_extract(data_json, '$.file_id')=?;",
                    (file_type, event_stage, selected_event, fid)
                )
                row = cur.fetchone()
                new_json = json.dumps(pkg)
                if row and row[0] != new_json:
                    cur.execute(
                        "UPDATE package_list SET data_json = ? WHERE file_type=? AND event_stage=? AND selected_event=? AND json_extract(data_json, '$.file_id')=?;",
                        (new_json, file_type, event_stage, selected_event, fid)
                    )

        conn.commit()
        conn.close()

init_db()



##TEST VIEW TEST VIEW##

class DB_INSPECT_OT_ShowCacheDB(bpy.types.Operator):
    bl_idname = "cache.show_database"
    bl_label = "Show Cache DB"
    bl_description = "Query and display the cache SQLite DB in a new text block"

    def execute(self, context):
        db_path = os.path.join(THUMBNAIL_CACHE_FOLDER, "cache.db")
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 1) list all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            output_lines = []
            for table in tables:
                # 2) get columns
                cursor.execute(f"PRAGMA table_info({table});")
                cols_info = cursor.fetchall()
                cols = [c[1] for c in cols_info]

                # 3) fetch all rows
                cursor.execute(f"SELECT * FROM {table};")
                rows = cursor.fetchall()

                # 4) compute column widths
                widths = [len(h) for h in cols]
                for row in rows:
                    for i, cell in enumerate(row):
                        widths[i] = max(widths[i], len(str(cell)))

                # 5) build ascii table
                sep = "+" + "+".join("-"*(w+2) for w in widths) + "+"
                header = "|" + "|".join(f" {cols[i].ljust(widths[i])} " for i in range(len(cols))) + "|"

                output_lines.append(f"Table: {table}")
                output_lines.append(sep)
                output_lines.append(header)
                output_lines.append(sep)
                for row in rows:
                    line = "|" + "|".join(f" {str(row[i]).ljust(widths[i])} " for i in range(len(cols))) + "|"
                    output_lines.append(line)
                output_lines.append(sep)
                output_lines.append("")

            # 6) write to Blender Text editor
            text_block = bpy.data.texts.get("CacheDB_Inspection") or bpy.data.texts.new("CacheDB_Inspection")
            text_block.clear()
            text_block.write("\n".join(output_lines))

            # 7) open it in the first Text Editor area found
            for area in context.window.screen.areas:
                if area.type == 'TEXT_EDITOR':
                    area.spaces.active.text = text_block
                    break

            self.report({'INFO'}, "Cache DB dumped to Text Editor")
        except Exception as e:
            self.report({'ERROR'}, f"DB query failed: {e}")
        finally:
            conn.close()
        return {'FINISHED'}


class VIEW3D_PT_CacheDB(bpy.types.Panel):
    bl_label = "Cache DB Inspector"
    bl_idname = "VIEW3D_PT_cache_db"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    def draw(self, context):
        self.layout.operator("cache.show_database", icon='TEXT')