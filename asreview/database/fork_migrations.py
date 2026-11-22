"""Fork-owned schema migrations (Spec 0001, D8).

A fork database carries ``PRAGMA application_id = 0x61736D72`` (the bytes
``b"asmr"``): that is THE fork mark, checked on every open. Fork schema
changes are tracked as named revisions in their own ledger table,
``fork_migrations``. ``PRAGMA user_version`` holds the fork mark; the
schema state is the ledger, and revision names say what each change is,
which a single integer cannot. Each
revision is applied once, in order, inside BEGIN IMMEDIATE so that
concurrent processes opening the same database cannot apply it twice.
Revision ids follow Alembic style ("0000_<slug>") to keep a later move to
Alembic mechanical.
"""

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

FORK_APPLICATION_ID = 0x61736D72  # b"asmr"

# The fork's project file version and the end of the migration chain.
# Upstream's migrate_project refuses a source below 1 before copying a
# project in, so this makes upstream reject a fork project outright.
ASMR_PROJECT_VERSION = -1

# The fork's database version: the same value as the application_id mark.
# No upstream release's database version check ever matches it.
FORK_USER_VERSION = FORK_APPLICATION_ID

# What 10d2 and earlier wrote into user_version before 0002 landed.
LEGACY_FORK_USER_VERSION = 4

# Revisions whose result create_tables() already builds directly, so a new
# database is marked as already having applied them.
BASELINE_REVISIONS = ("0000_tags_tables", "0002_user_version_mark")


@dataclass
class MigrationContext:
    """Context handed to each revision function.

    ``project_id`` is the folder name of the database file (or "memory").
    ``project_path`` is the folder holding ``results.db``, or ``None`` for
    an in-memory database.
    """

    project_id: str
    project_path: Path | None


def _rev_0000_tags_tables(cur, ctx):
    """Migrate a v3 database to the v4 tag/list schema.

    This is the schema half of the old v3→v4 migration. ``create_tables``
    builds the v4 schema directly, so this revision only ever runs against
    an upstream v3 database opened through project migration.
    """
    exists = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='tag_options'"
    ).fetchone()
    if exists:
        raise RuntimeError(
            "0000_tags_tables on a database that already has the v4 tag "
            "schema; is the '0000_tags_tables' ledger row missing?"
        )

    from asreview.project.migration.v3v4 import migrate_db

    return migrate_db(cur, ctx.project_path)


def _rev_0001_change_tracking(cur, ctx):
    cur.execute(
        """CREATE TABLE IF NOT EXISTS project_state (
            project_id TEXT PRIMARY KEY,
            version INTEGER NOT NULL DEFAULT 0,
            config_version INTEGER NOT NULL DEFAULT 0,
            pruned_version INTEGER NOT NULL DEFAULT 0,
            last_pruned_at REAL)"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS record_versions (
            project_id TEXT NOT NULL,
            record_id INTEGER NOT NULL,
            version INTEGER NOT NULL,
            PRIMARY KEY (project_id, record_id))"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS change_log (
            project_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            record_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            changed_at REAL NOT NULL,
            PRIMARY KEY (project_id, version))"""
    )
    cur.execute(
        """CREATE INDEX IF NOT EXISTS idx_change_log_changed_at
            ON change_log (project_id, changed_at)"""
    )
    cur.execute(
        "INSERT OR IGNORE INTO project_state (project_id) VALUES (?)",
        (ctx.project_id,),
    )


def _rev_0002_user_version_mark(cur, ctx):
    """Rewrite ``user_version`` to the fork mark.

    ``user_version`` lives in the file header and is transactional, so it
    commits or rolls back with the ledger's ``BEGIN IMMEDIATE``.
    """
    cur.execute(f"PRAGMA user_version = {FORK_USER_VERSION}")


REVISIONS = [
    ("0000_tags_tables", _rev_0000_tags_tables),
    ("0001_change_tracking", _rev_0001_change_tracking),
    ("0002_user_version_mark", _rev_0002_user_version_mark),
]


def mark_fork_database(conn):
    """Mark a new database as an asmr database and record the baseline.

    Sets ``PRAGMA application_id`` to the asmr mark, creates the ledger
    table if missing, and records the revisions whose result
    ``create_tables`` already built directly (so they never re-run as
    migrations). Called by ``create_tables`` only.
    """
    conn.execute(f"PRAGMA application_id = {FORK_APPLICATION_ID}")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fork_migrations "
        "(revision TEXT PRIMARY KEY, applied_at REAL NOT NULL)"
    )
    for rev_id in BASELINE_REVISIONS:
        conn.execute(
            "INSERT OR IGNORE INTO fork_migrations (revision, applied_at) "
            "VALUES (?, ?)",
            (rev_id, time.time()),
        )
    conn.commit()


def apply_fork_migrations(conn, project_id, project_path=None):
    """Apply missing revisions; return the list of ids applied now.

    A revision may return a list of cleanup callables (e.g. deleting legacy
    files); they run only after the ledger commit, so a failed migration
    never deletes anything.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fork_migrations "
        "(revision TEXT PRIMARY KEY, applied_at REAL NOT NULL)"
    )
    conn.commit()

    applied = {row[0] for row in conn.execute("SELECT revision FROM fork_migrations")}
    if all(rev_id in applied for rev_id, _ in REVISIONS):
        return []

    done = []
    cleanup = []
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        applied = {
            row[0] for row in conn.execute("SELECT revision FROM fork_migrations")
        }
        cur = conn.cursor()
        ctx = MigrationContext(project_id, project_path)
        for rev_id, fn in REVISIONS:
            if rev_id in applied:
                continue
            rev_cleanup = fn(cur, ctx)
            if rev_cleanup:
                cleanup.extend(rev_cleanup)
            cur.execute(
                "INSERT INTO fork_migrations (revision, applied_at) VALUES (?, ?)",
                (rev_id, time.time()),
            )
            done.append(rev_id)
            logging.info("Applied fork migration %s", rev_id)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    for fn in cleanup:
        fn()
    return done


def rekey_project_id(db_fp, new_project_id):
    """Re-key the change-tracking tables to ``new_project_id``.

    ``Project.load(safe_import=True)`` assigns a fresh id only after the
    project is migrated (or imported as an already-v4 fork project), so the
    change-tracking tables were written under the source project's id. Update
    ``project_state``, ``record_versions`` and ``change_log`` so their rows
    follow the new id instead of being orphaned. Idempotent: does nothing if
    the id already matches or the change-tracking tables are absent.
    """
    conn = sqlite3.connect(str(db_fp))
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='project_state'"
        ).fetchone()
        if exists is None:
            return

        row = conn.execute("SELECT project_id FROM project_state").fetchone()
        if row is None or row[0] == new_project_id:
            return

        old_id = row[0]
        conn.execute("BEGIN IMMEDIATE")
        try:
            for table in ("project_state", "record_versions", "change_log"):
                conn.execute(
                    f"UPDATE {table} SET project_id = ? WHERE project_id = ?",
                    (new_project_id, old_id),
                )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    finally:
        conn.close()
