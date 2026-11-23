import sqlite3
import threading
import time
from pathlib import Path

import pytest

import asreview as asr
from asreview.data.record import Record
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
        for name in CT_TRIGGER_NAMES:
            cur.execute(f"DROP TRIGGER IF EXISTS {name}")
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
    """create_tables creates every change-tracking trigger."""
    names = {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )
    }
    for name in CT_TRIGGER_NAMES:
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
