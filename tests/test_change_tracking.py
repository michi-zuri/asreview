import shutil
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

import asreview as asr
from asreview.data.record import Record
from asreview.database.database import _CHANGES_SINCE_SQL
from asreview.database.database import Database
from asreview.database.database import ForeignDatabaseError
from asreview.database.fork_migrations import FORK_APPLICATION_ID
from asreview.database.fork_migrations import FORK_USER_VERSION
from asreview.database.fork_migrations import LEGACY_FORK_USER_VERSION
from asreview.database.fork_migrations import rekey_project_id


@pytest.fixture
def db(tmpdir):
    with asr.Database(Path(tmpdir, "test.db")) as db:
        db.create_tables()
        yield db


# All Spec 0001 change-tracking trigger names, for the "triggers exist"
# checks and the legacy-DB cleanup.
CT_TRIGGER_NAMES = [
    "trg_ct_results_insert",
    "trg_ct_results_delete",
    "trg_ct_results_update",
    "trg_ct_tags_insert",
    "trg_ct_tags_update",
    "trg_ct_tags_delete",
    "trg_ct_lists_insert",
    "trg_ct_lists_update",
    "trg_ct_lists_delete",
    "trg_ct_record_attachment",
    "trg_ct_tag_groups_insert",
    "trg_ct_tag_groups_update",
    "trg_ct_tag_groups_delete",
    "trg_ct_tag_options_insert",
    "trg_ct_tag_options_update",
    "trg_ct_tag_options_delete",
    "trg_ct_list_containers_insert",
    "trg_ct_list_containers_update",
    "trg_ct_list_containers_delete",
]


# The project-id guard trigger names (Spec 0002 Phase 0).
PID_TRIGGER_NAMES = [
    "trg_pid_project_state_insert",
    "trg_pid_project_state_delete",
    "trg_pid_project_state_update",
    "trg_pid_project_state_cascade",
    "trg_pid_record_versions_insert",
    "trg_pid_record_versions_update",
    "trg_pid_change_log_insert",
    "trg_pid_change_log_update",
]


@pytest.fixture
def db_with_records(db):
    db.input.add_records([Record(i, "d") for i in range(10)])
    return db


def state(db):
    """Return (version, config_version) from project_state."""
    row = db._conn.execute(
        "SELECT version, config_version FROM project_state"
    ).fetchone()
    return (row[0], row[1]) if row else (0, 0)


def rv(db, rid):
    """Return record_versions.version for ``rid``, or 0 if absent."""
    row = db._conn.execute(
        "SELECT version FROM record_versions WHERE record_id = ?", (rid,)
    ).fetchone()
    return row[0] if row else 0


def log(db):
    """Return the change_log rows (version, record_id, source) in order."""
    return [
        (version, record_id, source)
        for version, record_id, source in db._conn.execute(
            "SELECT version, record_id, source FROM change_log ORDER BY version"
        ).fetchall()
    ]


def _insert_tag_group(db, group_id="g1"):
    db._conn.execute(
        "INSERT INTO tag_groups (group_id, export_name, label_name) "
        "VALUES (?, 'export', 'Label')",
        (group_id,),
    )
    db._conn.commit()


def _insert_tag_option(db, option_id="o1", group_id="g1"):
    db._conn.execute(
        "INSERT INTO tag_options (option_id, group_id, export_name, label_name) "
        "VALUES (?, ?, ?, 'Label')",
        (option_id, group_id, option_id),
    )
    db._conn.commit()


def _insert_list_container(db, list_id="l1"):
    db._conn.execute(
        "INSERT INTO list_containers (list_id, name) VALUES (?, 'My list')",
        (list_id,),
    )
    db._conn.commit()


def _make_legacy_db(tmpdir):
    """Build a marked database missing the 0001, 0002, 0003 and 0004 revisions.

    The mark, the 0000 baseline ledger row and the fork schema are kept;
    the 0001 tables/ledger row, the 0002 ledger row, the 0003 ledger row and
    the 0004 ledger row are stripped and ``user_version`` is rewound to the
    legacy 4, modelling a fork database from before ``0001_change_tracking``
    and ``0002_user_version_mark``. Dropping ``record_versions`` also drops
    its index and the ``trg_pid_*`` triggers on the dropped tables, so the
    0003 and 0004 rows must go too or the reopened database would claim
    schema it no longer has.
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
        for name in CT_TRIGGER_NAMES:
            cur.execute(f"DROP TRIGGER IF EXISTS {name}")
        cur.execute(
            "DELETE FROM fork_migrations WHERE revision = '0001_change_tracking'"
        )
        cur.execute(
            "DELETE FROM fork_migrations WHERE revision = '0002_user_version_mark'"
        )
        cur.execute(
            "DELETE FROM fork_migrations "
            "WHERE revision = '0003_record_versions_version_idx'"
        )
        cur.execute(
            "DELETE FROM fork_migrations WHERE revision = '0004_project_id_consistency'"
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
        "0003_record_versions_version_idx",
        "0004_project_id_consistency",
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
            "0003_record_versions_version_idx",
            "0004_project_id_consistency",
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
            "0003_record_versions_version_idx",
            "0004_project_id_consistency",
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
        assert cur.execute("SELECT COUNT(*) FROM fork_migrations").fetchone()[0] == 5
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
            "0003_record_versions_version_idx",
            "0004_project_id_consistency",
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
            "0003_record_versions_version_idx",
            "0004_project_id_consistency",
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
        assert _ledger(conn) == {
            "0001_change_tracking",
            "0002_user_version_mark",
            "0003_record_versions_version_idx",
            "0004_project_id_consistency",
        }
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


def test_label_record_tracked(db_with_records):
    """The spec example: labeling record 7 relevant bumps the version."""
    db_with_records.label_record(7, 1)
    assert state(db_with_records) == (1, 0)
    assert rv(db_with_records, 7) == 1
    assert log(db_with_records) == [(1, 7, "results")]


def test_update_result_tracked(db_with_records):
    """A label change via update_result is tracked as 'results'."""
    db_with_records.label_record(7, 0)
    db_with_records.update_result(7, label=1)
    assert rv(db_with_records, 7) == 2
    assert log(db_with_records) == [(1, 7, "results"), (2, 7, "results")]


def test_update_note_tracked(db_with_records):
    """A note change is tracked as 'results'."""
    db_with_records.label_record(7, 0)
    db_with_records.update_note(7, "hello")
    assert rv(db_with_records, 7) == 2
    assert log(db_with_records) == [(1, 7, "results"), (2, 7, "results")]


def test_save_tags_tracked(db_with_records):
    """Saving tags is tracked as 'tags'."""
    _insert_tag_group(db_with_records)
    _insert_tag_option(db_with_records)
    before = state(db_with_records)[0]
    db_with_records.update_result(
        7, tags=[{"id": "g1", "values": [{"id": "o1", "checked": True}]}]
    )
    assert state(db_with_records)[0] == before + 1
    assert rv(db_with_records, 7) == before + 1
    assert log(db_with_records)[-1] == (before + 1, 7, "tags")


def test_replace_lists_tracked(db_with_records):
    """Replacing list items is tracked as 'lists'."""
    _insert_list_container(db_with_records)
    before = state(db_with_records)[0]
    db_with_records.replace_lists(7, [{"list_id": "l1", "item_id": "i1", "name": "x"}])
    assert state(db_with_records)[0] == before + 1
    assert rv(db_with_records, 7) == before + 1
    assert log(db_with_records)[-1] == (before + 1, 7, "lists")


def test_set_attachment_tracked(db_with_records):
    """Storing a Zotero attachment key via the DataStore is tracked."""
    db_with_records.input.set_attachment(7, "ABCD1234")
    assert rv(db_with_records, 7) == 1
    assert log(db_with_records) == [(1, 7, "attachment")]


def test_delete_result_tracked(db_with_records):
    """Deleting a results row is tracked with the OLD record id."""
    db_with_records.label_record(7, 0)
    db_with_records.delete_result(7)
    assert rv(db_with_records, 7) == 2
    assert log(db_with_records) == [(1, 7, "results"), (2, 7, "results")]


def test_query_top_ranked_tracked(db_with_records):
    """Inserting a pending row, then reassigning it, both tracked."""
    db_with_records.add_last_ranking([0, 1, 2], "nb", "max", "balanced", "tfidf", 0)
    db_with_records.query_top_ranked()
    assert rv(db_with_records, 0) == 1
    db_with_records.query_top_ranked(user_id=1)
    assert rv(db_with_records, 0) == 2
    assert log(db_with_records) == [(1, 0, "results"), (2, 0, "results")]


def test_checkout_oldest_dispatched_tracked(db_with_records):
    """Checking out a dispatched record inserts a tracked results row."""
    db_with_records.add_last_ranking(
        list(range(10)), "nb", "max", "balanced", "tfidf", 0
    )
    db_with_records.top_up_dispatch(3, "H")
    db_with_records.checkout_oldest_dispatched(user_id=1)
    assert rv(db_with_records, 0) == 1
    assert log(db_with_records) == [(1, 0, "results")]


def test_reassign_stale_tracked(db_with_records):
    """Reassigning a stale checkout is tracked as 'results'."""
    db_with_records.add_last_ranking(
        list(range(10)), "nb", "max", "balanced", "tfidf", 0
    )
    db_with_records.top_up_dispatch(1, "H")
    db_with_records.checkout_oldest_dispatched(user_id=2)
    # Backdate last_active so the checkout is stale; last_active is not tracked.
    db_with_records._conn.execute(
        "UPDATE results SET last_active = ? WHERE record_id = 0",
        (time.time() - 100000,),
    )
    db_with_records._conn.commit()
    db_with_records.reassign_stale(user_id=1, older_than=3600)
    assert rv(db_with_records, 0) == 2
    assert log(db_with_records) == [(1, 0, "results"), (2, 0, "results")]


def test_touch_last_active_not_tracked(db_with_records):
    """Heartbeat writes to last_active never create versions."""
    db_with_records.add_last_ranking([0, 1, 2], "nb", "max", "balanced", "tfidf", 0)
    db_with_records.query_top_ranked(user_id=1)
    assert state(db_with_records) == (1, 0)
    for _ in range(10):
        assert db_with_records.touch_last_active(0, user_id=1)
    assert state(db_with_records) == (1, 0)
    assert log(db_with_records) == [(1, 0, "results")]


def test_noop_update_not_tracked(db_with_records):
    """A no-op note write does not create a version."""
    db_with_records.label_record(7, 0)
    db_with_records.update_note(7, "same")
    assert state(db_with_records) == (2, 0)
    db_with_records.update_note(7, "same")
    assert state(db_with_records) == (2, 0)
    assert log(db_with_records) == [(1, 7, "results"), (2, 7, "results")]


def test_attachment_failed_lookup_not_tracked(db_with_records):
    """Rewriting the failed-lookup timestamp is not tracked."""
    db_with_records.input.set_attachment(7, "2026-10-20T10:00:00")
    assert state(db_with_records) == (0, 0)
    db_with_records.input.set_attachment(7, "2026-10-20T11:00:00")
    assert state(db_with_records) == (0, 0)
    # A real key appearing is tracked...
    db_with_records.input.set_attachment(7, "ABCD1234")
    assert state(db_with_records) == (1, 0)
    # ...and disappearing (back to a timestamp) is too.
    db_with_records.input.set_attachment(7, "2026-10-20T12:00:00")
    assert state(db_with_records) == (2, 0)
    assert log(db_with_records) == [(1, 7, "attachment"), (2, 7, "attachment")]


def test_config_changes(db):
    """Config edits bump version and config_version, no per-record rows."""
    _insert_tag_group(db)
    assert state(db) == (1, 1)
    _insert_tag_option(db)
    assert state(db) == (2, 2)
    _insert_list_container(db)
    assert state(db) == (3, 3)
    assert log(db) == []
    assert db._conn.execute("SELECT COUNT(*) FROM record_versions").fetchone()[0] == 0


def test_triggers_after_create_tables(db):
    """create_tables creates every change-tracking and guard trigger."""
    names = {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )
    }
    for name in CT_TRIGGER_NAMES:
        assert name in names
    for name in PID_TRIGGER_NAMES:
        assert name in names


def test_triggers_after_fix_tags_schema(tmpdir):
    """A legacy tags table rebuilt on open gets its triggers back."""
    path = Path(tmpdir, "test.db")
    with asr.Database(path) as db:
        db.create_tables()

    with sqlite3.connect(str(path)) as conn:
        cur = conn.cursor()
        cur.execute("DROP TABLE tags")
        cur.execute(
            "CREATE TABLE tags (tag_id TEXT PRIMARY KEY, "
            "record_id INTEGER NOT NULL, checked INTEGER)"
        )
        conn.commit()
        names = {
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
        }
        assert "trg_ct_tags_insert" not in names

    with asr.open_db(path):
        pass

    with sqlite3.connect(str(path)) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
        }
        for name in CT_TRIGGER_NAMES:
            assert name in names


def test_version_order_equals_commit_order(tmpdir):
    """Two writer threads produce gap-free versions in commit order."""
    path = Path(tmpdir, "test.db")
    with asr.Database(path) as db:
        db.create_tables()
        db.input.add_records([Record(i, "d") for i in range(40)])

    errors = []
    barrier = threading.Barrier(2)

    def label_range(start, end):
        try:
            barrier.wait()
            with asr.open_db(path) as db:
                for rid in range(start, end):
                    db.label_record(rid, 1)
        except BaseException as err:  # noqa: BLE001 - re-raised via assert below
            errors.append(err)

    threads = [
        threading.Thread(target=label_range, args=(0, 20)),
        threading.Thread(target=label_range, args=(20, 40)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []

    with sqlite3.connect(str(path)) as conn:
        cur = conn.cursor()
        versions = [
            row[0]
            for row in cur.execute("SELECT version FROM change_log ORDER BY version")
        ]
        assert versions == list(range(1, 41))
        assert cur.execute("SELECT version FROM project_state").fetchone()[0] == 40
        for version, record_id in cur.execute(
            "SELECT version, record_id FROM change_log ORDER BY version"
        ):
            stored = cur.execute(
                "SELECT version FROM record_versions WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            assert stored is not None and stored[0] == version


def test_rekey_then_write_tracked(tmpdir):
    """After rekey_project_id, later writes follow the new project id."""
    path = Path(tmpdir, "test.db")
    with asr.Database(path) as db:
        db.create_tables()
        db.input.add_records([Record(i, "d") for i in range(3)])

    rekey_project_id(path, "new-id")

    with asr.open_db(path) as db:
        db.label_record(0, 1)

    with sqlite3.connect(str(path)) as conn:
        cur = conn.cursor()
        assert (
            cur.execute("SELECT project_id FROM project_state").fetchone()[0]
            == "new-id"
        )
        assert cur.execute(
            "SELECT project_id, record_id, version FROM record_versions"
        ).fetchall() == [("new-id", 0, 1)]
        assert cur.execute(
            "SELECT project_id, version, record_id, source FROM change_log"
        ).fetchall() == [("new-id", 1, 0, "results")]


def test_save_tags_unchanged_no_bump(db_with_records):
    """Saving identical tags twice bumps only once."""
    _insert_tag_group(db_with_records)
    _insert_tag_option(db_with_records)
    payload = [{"id": "g1", "values": [{"id": "o1", "checked": True}]}]
    db_with_records.update_result(7, tags=payload)
    before = state(db_with_records)[0]
    n_log = len(log(db_with_records))
    db_with_records.update_result(7, tags=payload)
    assert state(db_with_records)[0] == before
    assert len(log(db_with_records)) == n_log


def test_save_tags_unchecked_no_bump(db_with_records):
    """Saving an unchecked option twice bumps only once; row has checked=0."""
    _insert_tag_group(db_with_records)
    _insert_tag_option(db_with_records)
    payload = [{"id": "g1", "values": [{"id": "o1", "checked": False}]}]
    db_with_records.update_result(7, tags=payload)
    before = state(db_with_records)[0]
    db_with_records.update_result(7, tags=payload)
    assert state(db_with_records)[0] == before
    row = db_with_records._conn.execute(
        "SELECT checked, additional_text FROM tags "
        "WHERE record_id = 7 AND option_id = 'o1'"
    ).fetchone()
    assert row == (0, None)


def test_save_tags_change_one_bump(db_with_records):
    """Checking then unchecking is one bump; the row keeps its tag_id."""
    _insert_tag_group(db_with_records)
    _insert_tag_option(db_with_records)
    db_with_records.update_result(
        7,
        tags=[{"id": "g1", "values": [{"id": "o1", "checked": True, "text": "a"}]}],
    )
    tag_id_before = db_with_records._conn.execute(
        "SELECT tag_id FROM tags WHERE record_id = 7 AND option_id = 'o1'"
    ).fetchone()[0]
    n_log = len(log(db_with_records))
    db_with_records.update_result(
        7, tags=[{"id": "g1", "values": [{"id": "o1", "checked": False}]}]
    )
    assert len(log(db_with_records)) == n_log + 1
    assert log(db_with_records)[-1][2] == "tags"
    row = db_with_records._conn.execute(
        "SELECT tag_id, checked, additional_text FROM tags "
        "WHERE record_id = 7 AND option_id = 'o1'"
    ).fetchone()
    assert row[0] == tag_id_before
    assert row[1] == 0
    assert row[2] is None


def test_save_tags_text_change_one_bump(db_with_records):
    """Changing only the additional_text is one bump."""
    _insert_tag_group(db_with_records)
    _insert_tag_option(db_with_records)
    db_with_records.update_result(
        7,
        tags=[{"id": "g1", "values": [{"id": "o1", "checked": True, "text": "a"}]}],
    )
    n_log = len(log(db_with_records))
    db_with_records.update_result(
        7,
        tags=[{"id": "g1", "values": [{"id": "o1", "checked": True, "text": "b"}]}],
    )
    assert len(log(db_with_records)) == n_log + 1
    row = db_with_records._conn.execute(
        "SELECT checked, additional_text FROM tags "
        "WHERE record_id = 7 AND option_id = 'o1'"
    ).fetchone()
    assert row == (1, "b")


def test_save_tags_keeps_unsubmitted(db_with_records):
    """Options not in the payload are left untouched."""
    _insert_tag_group(db_with_records)
    _insert_tag_option(db_with_records, "o1", "g1")
    _insert_tag_option(db_with_records, "o2", "g1")
    db_with_records.update_result(
        7,
        tags=[
            {
                "id": "g1",
                "values": [
                    {"id": "o1", "checked": True},
                    {"id": "o2", "checked": True},
                ],
            }
        ],
    )
    tag_id_o2 = db_with_records._conn.execute(
        "SELECT tag_id FROM tags WHERE record_id = 7 AND option_id = 'o2'"
    ).fetchone()[0]
    # Save only o1 (changed); o2 is not in the payload.
    db_with_records.update_result(
        7, tags=[{"id": "g1", "values": [{"id": "o1", "checked": False}]}]
    )
    row_o2 = db_with_records._conn.execute(
        "SELECT tag_id, checked FROM tags WHERE record_id = 7 AND option_id = 'o2'"
    ).fetchone()
    assert row_o2[0] == tag_id_o2
    assert row_o2[1] == 1


def test_note_edit_with_same_tags_no_tag_bump(db_with_records):
    """label_record with tags, then update_result with the same tags → no tags bump."""
    _insert_tag_group(db_with_records)
    _insert_tag_option(db_with_records)
    payload = [{"id": "g1", "values": [{"id": "o1", "checked": True}]}]
    db_with_records.label_record(7, 1, tags=payload)
    n_tags_log = len([row for row in log(db_with_records) if row[2] == "tags"])
    db_with_records.update_result(7, tags=payload)
    assert len([row for row in log(db_with_records) if row[2] == "tags"]) == n_tags_log


def test_replace_lists_unchanged_no_bump(db_with_records):
    """Saving identical list items twice bumps only once."""
    _insert_list_container(db_with_records)
    items = [
        {"list_id": "l1", "item_id": "a", "name": "x", "sorted_at": 1.0},
        {"list_id": "l1", "item_id": "b", "name": "y", "sorted_at": 2.0},
    ]
    db_with_records.replace_lists(7, items)
    before = state(db_with_records)[0]
    n_log = len(log(db_with_records))
    db_with_records.replace_lists(7, items)
    assert state(db_with_records)[0] == before
    assert len(log(db_with_records)) == n_log


def test_replace_lists_missing_sorted_at_keeps_stored(db_with_records):
    """An item without sorted_at keeps its stored value; no bump."""
    _insert_list_container(db_with_records)
    db_with_records.replace_lists(
        7, [{"list_id": "l1", "item_id": "a", "name": "x", "sorted_at": 5.0}]
    )
    n_log = len(log(db_with_records))
    db_with_records.replace_lists(7, [{"list_id": "l1", "item_id": "a", "name": "x"}])
    assert len(log(db_with_records)) == n_log
    assert db_with_records.get_lists(7)[0]["sorted_at"] == 5.0


def test_replace_lists_rename_one_bump(db_with_records):
    """Renaming one of two items is exactly one bump; same item_id, new name."""
    _insert_list_container(db_with_records)
    db_with_records.replace_lists(
        7,
        [
            {"list_id": "l1", "item_id": "a", "name": "x", "sorted_at": 1.0},
            {"list_id": "l1", "item_id": "b", "name": "y", "sorted_at": 2.0},
        ],
    )
    n_log = len(log(db_with_records))
    db_with_records.replace_lists(
        7,
        [
            {"list_id": "l1", "item_id": "a", "name": "z", "sorted_at": 1.0},
            {"list_id": "l1", "item_id": "b", "name": "y", "sorted_at": 2.0},
        ],
    )
    assert len(log(db_with_records)) == n_log + 1
    assert log(db_with_records)[-1][2] == "lists"
    names = {i["item_id"]: i["name"] for i in db_with_records.get_lists(7)}
    assert names == {"a": "z", "b": "y"}


def test_replace_lists_remove_one_bump(db_with_records):
    """Dropping one of two items is exactly one bump; the other is untouched."""
    _insert_list_container(db_with_records)
    db_with_records.replace_lists(
        7,
        [
            {"list_id": "l1", "item_id": "a", "name": "x", "sorted_at": 1.0},
            {"list_id": "l1", "item_id": "b", "name": "y", "sorted_at": 2.0},
        ],
    )
    n_log = len(log(db_with_records))
    db_with_records.replace_lists(
        7, [{"list_id": "l1", "item_id": "a", "name": "x", "sorted_at": 1.0}]
    )
    assert len(log(db_with_records)) == n_log + 1
    assert log(db_with_records)[-1][2] == "lists"
    assert [i["item_id"] for i in db_with_records.get_lists(7)] == ["a"]


def test_replace_lists_swap_names(db_with_records):
    """Two items swapping names in one save succeeds."""
    _insert_list_container(db_with_records)
    db_with_records.replace_lists(
        7,
        [
            {"list_id": "l1", "item_id": "a", "name": "x", "sorted_at": 1.0},
            {"list_id": "l1", "item_id": "b", "name": "y", "sorted_at": 2.0},
        ],
    )
    db_with_records.replace_lists(
        7,
        [
            {"list_id": "l1", "item_id": "a", "name": "y", "sorted_at": 1.0},
            {"list_id": "l1", "item_id": "b", "name": "x", "sorted_at": 2.0},
        ],
    )
    names = {i["item_id"]: i["name"] for i in db_with_records.get_lists(7)}
    assert names == {"a": "y", "b": "x"}


def test_replace_lists_duplicate_rolls_back(db_with_records):
    """A duplicate name in one save raises and writes nothing."""
    _insert_list_container(db_with_records)
    db_with_records.replace_lists(
        7, [{"list_id": "l1", "item_id": "a", "name": "x", "sorted_at": 1.0}]
    )
    n_log = len(log(db_with_records))
    with pytest.raises(sqlite3.IntegrityError):
        db_with_records.replace_lists(
            7,
            [
                {"list_id": "l1", "item_id": "a", "name": "x", "sorted_at": 1.0},
                {"list_id": "l1", "item_id": "b", "name": "x", "sorted_at": 2.0},
            ],
        )
    assert len(log(db_with_records)) == n_log
    assert [i["item_id"] for i in db_with_records.get_lists(7)] == ["a"]


def test_get_changes_since(db_with_records):
    """One entry per record (its current version), ordered by version."""
    db_with_records.label_record(1, 1)  # v1
    db_with_records.label_record(2, 1)  # v2
    db_with_records.update_note(1, "a")  # v3
    db_with_records.update_note(1, "b")  # v4
    assert db_with_records.get_changes_since(0) == [(2, 2), (1, 4)]
    assert db_with_records.get_changes_since(4) == []
    assert db_with_records.get_changes_since(0, limit=1) == [(2, 2)]


def test_prune_change_log(db_with_records):
    """Pruning deletes old rows, records the highest version, and is gated."""
    db_with_records.label_record(1, 1)
    db_with_records.label_record(2, 1)
    now = (
        db_with_records._conn.execute(
            "SELECT MAX(changed_at) FROM change_log"
        ).fetchone()[0]
        + 8 * 86400
    )
    deleted = db_with_records.prune_change_log(now=now)
    assert deleted == 2
    assert db_with_records._conn.execute(
        "SELECT pruned_version, last_pruned_at FROM project_state"
    ).fetchone() == (2, now)
    # A second call within the hour deletes nothing, even with a new row.
    db_with_records.label_record(3, 1)
    assert db_with_records.prune_change_log(now=now + 100) == 0


def test_get_changes_since_unaffected_by_prune(db_with_records):
    """Pruning change_log never changes what get_changes_since returns."""
    db_with_records.label_record(1, 1)
    db_with_records.label_record(2, 1)
    before = db_with_records.get_changes_since(0)
    now = (
        db_with_records._conn.execute(
            "SELECT MAX(changed_at) FROM change_log"
        ).fetchone()[0]
        + 8 * 86400
    )
    db_with_records.prune_change_log(now=now)
    assert db_with_records.get_changes_since(0) == before


def test_prune_change_log_not_due_takes_no_lock(db_with_records):
    """Not due: returns 0 without waiting on another writer's lock."""
    db_with_records.label_record(1, 1)
    now = (
        db_with_records._conn.execute(
            "SELECT MAX(changed_at) FROM change_log"
        ).fetchone()[0]
        + 8 * 86400
    )
    db_with_records.prune_change_log(now=now)

    second = sqlite3.connect(str(db_with_records.fp), timeout=0)
    try:
        second.execute("BEGIN IMMEDIATE")  # the second connection holds the lock
        start = time.monotonic()
        assert db_with_records.prune_change_log(now=now + 10) == 0
        assert time.monotonic() - start < 1.0
    finally:
        second.rollback()
        second.close()


def test_prune_change_log_rechecks_under_lock(db_with_records, monkeypatch):
    """Another process pruning between check and lock makes this a no-op."""
    db_with_records.label_record(1, 1)
    db_with_records.label_record(2, 1)
    now = (
        db_with_records._conn.execute(
            "SELECT MAX(changed_at) FROM change_log"
        ).fetchone()[0]
        + 8 * 86400
    )

    real_prune_due = Database._prune_due
    calls = []

    def fake_prune_due(self, now_arg, min_interval_s):
        result = real_prune_due(self, now_arg, min_interval_s)
        if not calls:
            calls.append(True)
            # Another process prunes between the first check and BEGIN
            # IMMEDIATE: it sets last_pruned_at and commits.
            with sqlite3.connect(str(db_with_records.fp)) as conn:
                conn.execute("UPDATE project_state SET last_pruned_at = ?", (now,))
                conn.commit()
        return result

    monkeypatch.setattr(Database, "_prune_due", fake_prune_due)

    assert db_with_records.prune_change_log(now=now) == 0
    # Both rows survive; the "other process" wrote last_pruned_at; this call
    # did not delete anything or bump pruned_version.
    assert (
        db_with_records._conn.execute("SELECT COUNT(*) FROM change_log").fetchone()[0]
        == 2
    )
    assert (
        db_with_records._conn.execute(
            "SELECT last_pruned_at FROM project_state"
        ).fetchone()[0]
        == now
    )
    assert (
        db_with_records._conn.execute(
            "SELECT pruned_version FROM project_state"
        ).fetchone()[0]
        == 0
    )


def test_prune_change_log_refuses_open_transaction(db_with_records):
    """A caller's open transaction is never committed by pruning."""
    db_with_records._conn.execute(
        "UPDATE project_state SET config_version = config_version"
    )
    assert db_with_records._conn.in_transaction
    with pytest.raises(RuntimeError):
        db_with_records.prune_change_log()
    db_with_records._conn.rollback()


def test_copied_folder_changes_tracked(tmpdir):
    """A copied folder keeps its stored id; polls still report changes."""
    src = Path(tmpdir, "a")
    src.mkdir()
    with asr.Database(src / "results.db") as db:
        db.create_tables()
        db.input.add_records([Record(i, "d") for i in range(3)])
    shutil.copytree(src, Path(tmpdir, "b"))
    with asr.open_db(Path(tmpdir, "b", "results.db")) as db:
        db.label_record(0, 1)
        assert db.project_id == "a"
        assert db.get_changes_since(0) == [(0, 1)]
        assert db.get_record_versions([0, 1]) == {0: 1}


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO project_state (project_id) VALUES ('other')",
        "INSERT OR IGNORE INTO project_state (project_id) VALUES ('other')",
        "DELETE FROM project_state",
        "UPDATE project_state SET project_id = NULL",
        "UPDATE project_state SET project_id = ''",
        "INSERT INTO record_versions (project_id, record_id, version) "
        "VALUES ('other', 5, 99)",
        "INSERT INTO change_log (project_id, version, record_id, source, changed_at) "
        "VALUES ('other', 99, 5, 'results', 0)",
        "UPDATE record_versions SET project_id = 'other'",
        "UPDATE change_log SET project_id = 'other'",
    ],
)
def test_pid_guards_refuse(db_with_records, sql):
    """The guard triggers refuse every way to break the one-id invariant."""
    db_with_records.label_record(1, 1)
    with pytest.raises(sqlite3.IntegrityError):
        db_with_records._conn.execute(sql)
    db_with_records._conn.rollback()
    assert (
        db_with_records._conn.execute("SELECT COUNT(*) FROM project_state").fetchone()[
            0
        ]
        == 1
    )
    assert db_with_records.get_changes_since(0) == [(1, 1)]


def test_pid_cascade(db_with_records):
    """Renaming project_state.project_id cascades to both tracked tables."""
    db_with_records.label_record(1, 1)
    db_with_records.label_record(2, 1)
    db_with_records._conn.execute("UPDATE project_state SET project_id = 'renamed'")
    db_with_records._conn.commit()
    assert db_with_records.project_id == "renamed"
    for table in ("record_versions", "change_log"):
        rows = db_with_records._conn.execute(
            f"SELECT DISTINCT project_id FROM {table}"
        ).fetchall()
        assert rows == [("renamed",)]
    db_with_records.label_record(3, 1)
    assert [r for r, _ in db_with_records.get_changes_since(0)] == [1, 2, 3]


def test_project_id_property(tmpdir):
    """project_id reads project_state; None before the row exists."""
    with asr.Database() as db:
        assert db.project_id is None
        db.create_tables()
        assert db.project_id == "memory"

    path = _make_legacy_db(tmpdir)
    with asr.open_db(path, read_only=True) as db:
        assert db.project_id is None


def test_0004_refuses_inconsistent_legacy_db(tmpdir):
    """0004 refuses (and rolls back) a database with foreign or duplicate rows."""
    # Case 1: a foreign record_versions row.
    path = Path(tmpdir, "foreign.db")
    with asr.Database(path) as db:
        db.create_tables()
        db.input.add_records([Record(i, "d") for i in range(3)])
        db.label_record(0, 1)
    with sqlite3.connect(str(path)) as conn:
        for name in PID_TRIGGER_NAMES:
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.execute("UPDATE record_versions SET project_id = 'stale'")
        conn.execute(
            "DELETE FROM fork_migrations WHERE revision = '0004_project_id_consistency'"
        )
        conn.commit()
    with pytest.raises(RuntimeError, match="record_versions"):
        asr.open_db(path)
    with sqlite3.connect(str(path)) as conn:
        assert (
            conn.execute(
                "SELECT revision FROM fork_migrations "
                "WHERE revision = '0004_project_id_consistency'"
            ).fetchone()
            is None
        )
        assert conn.execute("SELECT project_id FROM record_versions").fetchall() == [
            ("stale",)
        ]

    # Case 2: a duplicate project_state row.
    path2 = Path(tmpdir, "dup.db")
    with asr.Database(path2) as db:
        db.create_tables()
        db.input.add_records([Record(i, "d") for i in range(3)])
        db.label_record(0, 1)
    with sqlite3.connect(str(path2)) as conn:
        for name in PID_TRIGGER_NAMES:
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.execute("INSERT INTO project_state (project_id) VALUES ('other')")
        conn.execute(
            "DELETE FROM fork_migrations WHERE revision = '0004_project_id_consistency'"
        )
        conn.commit()
    with pytest.raises(RuntimeError, match="project_state holds 2 rows"):
        asr.open_db(path2)


def _has_version_index(conn):
    return any(
        row[0] == "idx_record_versions_version"
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    )


def test_record_versions_version_index(tmpdir):
    """New and legacy databases both have the index and the ledger rows."""
    path = Path(tmpdir, "new.db")
    with asr.Database(path) as db:
        db.create_tables()
    with sqlite3.connect(str(path)) as conn:
        assert _has_version_index(conn)
        assert "0003_record_versions_version_idx" in _ledger(conn)
        assert "0004_project_id_consistency" in _ledger(conn)
        plan = " ".join(
            str(c)
            for row in conn.execute("EXPLAIN QUERY PLAN " + _CHANGES_SINCE_SQL, (0,))
            for c in row
        )
        assert "idx_record_versions_version" in plan

    legacy = _make_legacy_db(tmpdir)
    with asr.open_db(legacy):
        pass
    with sqlite3.connect(str(legacy)) as conn:
        assert _has_version_index(conn)
        assert "0003_record_versions_version_idx" in _ledger(conn)
        assert "0004_project_id_consistency" in _ledger(conn)


def test_get_record_states(db_with_records):
    """Unlabeled, labeled and unknown ids: 2 rows in the requested order."""
    db_with_records.label_record(3, 1)
    df = db_with_records.get_record_states([999, 5, 3])
    assert df["record_id"].tolist() == [5, 3]
    assert "tags" in df.columns
    labels = df.set_index("record_id")["label"]
    assert labels.loc[3] == 1
    assert pd.isna(labels.loc[5])
