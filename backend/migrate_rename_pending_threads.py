"""
Migration: continuity_hooks → pending_threads
             initiative_mode  → narrative_stance (column in session settings)

Run once against the existing memory.db before starting the updated backend.

Usage:
    cd <backend parent dir>
    python -m backend.migrate_rename_pending_threads
  or (if run directly from repo root):
    python backend/migrate_rename_pending_threads.py
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "memory.db"


def run(db_path: Path) -> None:
    if not db_path.exists():
        print(f"DB not found at {db_path} — nothing to migrate.")
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # ── 1. Rename table continuity_hooks → pending_threads ───────────
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='continuity_hooks'")
    if cur.fetchone():
        cur.execute("ALTER TABLE continuity_hooks RENAME TO pending_threads")
        print("Renamed table: continuity_hooks → pending_threads")
    else:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pending_threads'")
        if cur.fetchone():
            print("Table already named pending_threads — skipping.")
        else:
            print("WARNING: Neither continuity_hooks nor pending_threads table found.")

    # ── 2. Rename index if it exists ─────────────────────────────────
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='continuity_hooks'"
    )
    indexes = [row[0] for row in cur.fetchall()]
    for idx in indexes:
        # SQLite cannot rename indexes; drop and recreate is needed if index
        # was auto-named. Auto-named indexes (sqlite_autoindex_*) are
        # recreated automatically with the table rename above, so only
        # user-defined indexes need attention.
        if not idx.startswith("sqlite_autoindex_"):
            new_idx = idx.replace("continuity_hooks", "pending_threads")
            cur.execute(f'DROP INDEX IF EXISTS "{idx}"')
            print(f"Dropped index {idx} (recreate manually as {new_idx} if needed)")

    conn.commit()
    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DB_PATH
    run(target)
