"""Fork-owned schema migrations (Spec 0001, D8).

Fork schema changes are tracked as named revisions in their own ledger
table, ``fork_migrations``. ``PRAGMA user_version`` (4) keeps its current
meaning; revision names say what each change is, which a single integer
cannot. Each revision is applied once, in order, inside BEGIN IMMEDIATE so
that concurrent processes opening the same database cannot apply it
twice. Revision ids follow Alembic style ("0001_<slug>") to keep a
later move to Alembic mechanical.
"""

import logging
import time


def _rev_0001_change_tracking(cur, project_id):
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
        (project_id,),
    )


REVISIONS = [
    ("0001_change_tracking", _rev_0001_change_tracking),
]


def apply_fork_migrations(conn, project_id):
    """Apply missing revisions; return the list of ids applied now."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fork_migrations "
        "(revision TEXT PRIMARY KEY, applied_at REAL NOT NULL)"
    )
    conn.commit()

    applied = {row[0] for row in conn.execute("SELECT revision FROM fork_migrations")}
    if all(rev_id in applied for rev_id, _ in REVISIONS):
        return []

    done = []
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        applied = {
            row[0] for row in conn.execute("SELECT revision FROM fork_migrations")
        }
        cur = conn.cursor()
        for rev_id, fn in REVISIONS:
            if rev_id in applied:
                continue
            fn(cur, project_id)
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
    return done
