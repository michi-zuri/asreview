import sqlite3
import threading
from pathlib import Path

import pytest

import asreview as asr
from asreview.database.database import ForeignDatabaseError
from asreview.database.fork_migrations import FORK_APPLICATION_ID
from asreview.database.fork_migrations import FORK_USER_VERSION
from asreview.database.fork_migrations import LEGACY_FORK_USER_VERSION


@pytest.fixture
def db(tmpdir):
    with asr.Database(Path(tmpdir, "test.db")) as db:
        db.create_tables()
        yield db


def _make_legacy_db(tmpdir):
    """Build a marked database missing the 0001 and 0002 revisions.

    The mark, the 0000 baseline ledger row and the fork schema are kept;
    the 0001 tables/ledger row and the 0002 ledger row are stripped and
    ``user_version`` is rewound to the legacy 4, modelling a fork database
    from before ``0001_change_tracking`` and ``0002_user_version_mark``.
    """
    path = Path(tmpdir, "test.db")
    with asr.Database(path) as db:
        db.create_tables()
    with sqlite3.connect(str(path)) as conn:
        cur = conn.cursor()
        for table in [
            "project_state",
            "record_versions",
            "change_log",
        ]:
            cur.execute(f"DROP TABLE {table}")
        cur.execute(
            "DELETE FROM fork_migrations WHERE revision = '0001_change_tracking'"
        )
        cur.execute(
            "DELETE FROM fork_migrations WHERE revision = '0002_user_version_mark'"
        )
        cur.execute(f"PRAGMA user_version = {LEGACY_FORK_USER_VERSION}")
        conn.commit()
    return path


def _table_names(conn):
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _ledger(conn):
    return {row[0] for row in conn.execute("SELECT revision FROM fork_migrations")}


def _application_id(path):
    with sqlite3.connect(str(path)) as conn:
        return conn.execute("PRAGMA application_id").fetchone()[0]


def test_ledger_applied_on_create(db, tmpdir):
    cur = db._conn.cursor()
    assert _ledger(db._conn) == {
        "0000_tags_tables",
        "0001_change_tracking",
        "0002_user_version_mark",
    }
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
        assert _ledger(conn) == {
            "0000_tags_tables",
            "0001_change_tracking",
            "0002_user_version_mark",
        }
        assert cur.execute("SELECT COUNT(*) FROM project_state").fetchone()[0] == 1


def test_ledger_applied_on_legacy_open(tmpdir):
    path = _make_legacy_db(tmpdir)

    with asr.open_db(path):
        pass

    with sqlite3.connect(str(path)) as conn:
        assert _ledger(conn) == {
            "0000_tags_tables",
            "0001_change_tracking",
            "0002_user_version_mark",
        }
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
        assert cur.execute("SELECT COUNT(*) FROM fork_migrations").fetchone()[0] == 3
        assert cur.execute("SELECT COUNT(*) FROM project_state").fetchone()[0] == 1


def test_read_only_open_does_not_migrate(tmpdir):
    path = _make_legacy_db(tmpdir)

    with asr.open_db(path, read_only=True):
        pass

    with sqlite3.connect(str(path)) as conn:
        assert "project_state" not in _table_names(conn)
        assert _ledger(conn) == {"0000_tags_tables"}


def test_in_memory_project_id():
    with asr.Database() as db:
        db.create_tables()
        cur = db._conn.cursor()
        assert (
            cur.execute("SELECT project_id FROM project_state").fetchone()[0]
            == "memory"
        )


def test_new_database_is_marked(tmpdir):
    path = Path(tmpdir, "test.db")
    with asr.Database(path) as db:
        db.create_tables()

    assert _application_id(path) == FORK_APPLICATION_ID
    with sqlite3.connect(str(path)) as conn:
        assert _ledger(conn) == {
            "0000_tags_tables",
            "0001_change_tracking",
            "0002_user_version_mark",
        }


def test_unmarked_v4_database_refused(tmpdir):
    path = Path(tmpdir, "test.db")
    with asr.Database(path) as db:
        db.create_tables()
        db._conn.execute("INSERT INTO results (record_id) VALUES (1)")
        db._conn.commit()

    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA application_id = 0")
        conn.commit()

    with pytest.raises(ForeignDatabaseError):
        asr.open_db(path)

    with sqlite3.connect(str(path)) as conn:
        cur = conn.cursor()
        assert cur.execute("PRAGMA user_version").fetchone()[0] == FORK_USER_VERSION
        assert cur.execute("SELECT record_id FROM results").fetchall() == [(1,)]


def test_unmarked_v5_database_refused(tmpdir):
    path = Path(tmpdir, "test.db")
    with asr.Database(path) as db:
        db.create_tables()

    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA application_id = 0")
        conn.execute("PRAGMA user_version = 5")
        conn.commit()

    with pytest.raises(ForeignDatabaseError):
        asr.open_db(path)

    with sqlite3.connect(str(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 5


def test_read_only_open_refuses_unmarked(tmpdir):
    path = Path(tmpdir, "test.db")
    with asr.Database(path) as db:
        db.create_tables()

    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA application_id = 0")
        conn.commit()

    with pytest.raises(ValueError):
        asr.open_db(path, read_only=True)


def test_empty_file_is_created(tmpdir):
    path = Path(tmpdir, "test.db")
    path.write_bytes(b"")

    with asr.open_db(path):
        pass

    assert _application_id(path) == FORK_APPLICATION_ID
    with sqlite3.connect(str(path)) as conn:
        assert "results" in _table_names(conn)
        assert _ledger(conn) == {
            "0000_tags_tables",
            "0001_change_tracking",
            "0002_user_version_mark",
        }


def test_missing_baseline_row_rolls_back(tmpdir):
    path = Path(tmpdir, "test.db")
    with asr.Database(path) as db:
        db.create_tables()

    with sqlite3.connect(str(path)) as conn:
        conn.execute("DELETE FROM fork_migrations WHERE revision = '0000_tags_tables'")
        conn.commit()

    with pytest.raises(RuntimeError):
        asr.open_db(path)

    with sqlite3.connect(str(path)) as conn:
        assert _ledger(conn) == {"0001_change_tracking", "0002_user_version_mark"}
        assert "tag_options" in _table_names(conn)


def test_legacy_user_version_upgraded(tmpdir):
    path = _make_legacy_db(tmpdir)

    with asr.open_db(path):
        pass

    with sqlite3.connect(str(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == FORK_USER_VERSION


def test_read_only_open_keeps_legacy_user_version(tmpdir):
    path = _make_legacy_db(tmpdir)

    with asr.open_db(path, read_only=True):
        pass

    with sqlite3.connect(str(path)) as conn:
        assert (
            conn.execute("PRAGMA user_version").fetchone()[0]
            == LEGACY_FORK_USER_VERSION
        )


def test_marked_wrong_user_version_refused(tmpdir):
    path = Path(tmpdir, "test.db")
    with asr.Database(path) as db:
        db.create_tables()

    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA user_version = 5")
        conn.commit()

    with pytest.raises(ValueError):
        asr.open_db(path)
