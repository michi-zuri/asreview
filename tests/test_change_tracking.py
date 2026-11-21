import sqlite3
import threading
from pathlib import Path

import pytest

import asreview as asr


@pytest.fixture
def db(tmpdir):
    with asr.Database(Path(tmpdir, "test.db")) as db:
        db.create_tables()
        yield db


def _make_legacy_db(tmpdir):
    """Build a valid database and drop the fork tables, simulating pre-fork."""
    path = Path(tmpdir, "test.db")
    with asr.Database(path) as db:
        db.create_tables()
    with sqlite3.connect(str(path)) as conn:
        cur = conn.cursor()
        for table in [
            "fork_migrations",
            "project_state",
            "record_versions",
            "change_log",
        ]:
            cur.execute(f"DROP TABLE {table}")
        conn.commit()
    return path


def _table_names(conn):
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def test_ledger_applied_on_create(db, tmpdir):
    cur = db._conn.cursor()
    assert cur.execute("SELECT revision FROM fork_migrations").fetchall() == [
        ("0001_change_tracking",)
    ]
    assert cur.execute(
        "SELECT project_id, version, config_version FROM project_state"
    ).fetchall() == [(Path(tmpdir).name, 0, 0)]


def test_ledger_idempotent_on_reopen(tmpdir):
    path = Path(tmpdir, "test.db")
    with asr.Database(path) as db:
        db.create_tables()
    with asr.open_db(path):
        pass
    with asr.open_db(path):
        pass

    with sqlite3.connect(str(path)) as conn:
        cur = conn.cursor()
        assert cur.execute("SELECT revision FROM fork_migrations").fetchall() == [
            ("0001_change_tracking",)
        ]
        assert cur.execute("SELECT COUNT(*) FROM project_state").fetchone()[0] == 1


def test_ledger_applied_on_legacy_open(tmpdir):
    path = _make_legacy_db(tmpdir)

    with asr.open_db(path):
        pass

    with sqlite3.connect(str(path)) as conn:
        cur = conn.cursor()
        assert cur.execute("SELECT revision FROM fork_migrations").fetchall() == [
            ("0001_change_tracking",)
        ]
        assert {
            "project_state",
            "record_versions",
            "change_log",
        } <= _table_names(conn)


def test_ledger_concurrent_openers(tmpdir):
    path = _make_legacy_db(tmpdir)
    errors = []
    barrier = threading.Barrier(4)

    def opener():
        try:
            barrier.wait()
            with asr.open_db(path):
                pass
        except BaseException as err:  # noqa: BLE001 - re-raised via assert below
            errors.append(err)

    threads = [threading.Thread(target=opener) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []

    with sqlite3.connect(str(path)) as conn:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM fork_migrations").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM project_state").fetchone()[0] == 1


def test_read_only_open_does_not_migrate(tmpdir):
    path = _make_legacy_db(tmpdir)

    with asr.open_db(path, read_only=True):
        pass

    with sqlite3.connect(str(path)) as conn:
        assert "fork_migrations" not in _table_names(conn)


def test_in_memory_project_id():
    with asr.Database() as db:
        db.create_tables()
        cur = db._conn.cursor()
        assert cur.execute(
            "SELECT project_id FROM project_state"
        ).fetchone()[0] == "memory"
