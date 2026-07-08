import time as time_module
from pathlib import Path

import pytest

import asreview as asr
from asreview.data.record import Record


@pytest.fixture
def db(tmp_path):
    with asr.Database(Path(tmp_path, "test.db")) as db:
        db.create_tables()
        yield db


@pytest.fixture
def db_pool(db):
    # 6 standalone records (record_id == group_id); rank all of them.
    db.input.add_records([Record(i, "d") for i in range(6)])
    db.add_last_ranking(
        [0, 1, 2, 3, 4, 5], "nb", "max", "balanced", "tfidf", 0
    )
    return db


def _tables(db):
    cur = db._conn.cursor()
    rows = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def _columns(db, table):
    cur = db._conn.cursor()
    return [r[1] for r in cur.execute(f"PRAGMA table_info({table})")]


def _dispatch_rows(db):
    cur = db._conn.cursor()
    return cur.execute(
        "SELECT record_id, status, prompt_hash FROM llm_dispatch "
        "ORDER BY dispatched_at"
    ).fetchall()


def test_tables_created(db):
    """After create_tables, llm_dispatch and llm_results exist."""
    tables = _tables(db)
    assert "llm_dispatch" in tables
    assert "llm_results" in tables


def test_llm_dispatch_columns(db):
    """llm_dispatch has exactly the expected columns."""
    cols = _columns(db, "llm_dispatch")
    assert cols == [
        "record_id", "dispatched_at", "status", "prompt_hash",
        "attempts", "last_error",
    ]


def test_llm_results_columns(db):
    """llm_results has exactly the expected columns."""
    cols = _columns(db, "llm_results")
    assert cols == [
        "record_id", "prompt_hash", "model", "payload_json",
        "input_tokens", "output_tokens", "created_at",
    ]


def test_ensure_idempotent(db):
    """Calling _ensure_* again does not raise."""
    db._ensure_llm_dispatch_table()
    db._ensure_llm_results_table()


def test_migration_path(db):
    """Drop both tables then re-create via _ensure_* — both exist again."""
    cur = db._conn.cursor()
    cur.execute("DROP TABLE llm_dispatch")
    cur.execute("DROP TABLE llm_results")
    db._conn.commit()

    tables = _tables(db)
    assert "llm_dispatch" not in tables
    assert "llm_results" not in tables

    db._ensure_llm_dispatch_table()
    db._ensure_llm_results_table()

    tables = _tables(db)
    assert "llm_dispatch" in tables
    assert "llm_results" in tables


def test_results_has_assigned_at_and_last_active(db):
    """After create_tables, results table includes assigned_at and last_active."""
    cols = _columns(db, "results")
    assert "assigned_at" in cols
    assert "last_active" in cols


def test_original_results_columns_present(db):
    """Spot-check that the original results columns are still present."""
    cols = _columns(db, "results")
    assert "record_id" in cols
    assert "label" in cols
    assert "note" in cols
    assert "user_id" in cols


def test_fix_results_schema_idempotent(db):
    """Calling _fix_results_schema again does not duplicate columns."""
    db._fix_results_schema(db._conn.cursor())
    db._fix_results_schema(db._conn.cursor())
    cols = _columns(db, "results")
    assert cols.count("assigned_at") == 1
    assert cols.count("last_active") == 1


def test_top_up_basic(db_pool):
    """top_up_dispatch(3, 'H') inserts 3 queued rows in ranking order."""
    n = db_pool.top_up_dispatch(3, "H")
    assert n == 3
    rows = _dispatch_rows(db_pool)
    assert rows == [
        (0, "queued", "H"),
        (1, "queued", "H"),
        (2, "queued", "H"),
    ]


def test_top_up_deficit_respected(db_pool):
    """Second call with same buffer_size adds nothing."""
    db_pool.top_up_dispatch(3, "H")
    n = db_pool.top_up_dispatch(3, "H")
    assert n == 0
    assert len(_dispatch_rows(db_pool)) == 3


def test_top_up_grow_buffer(db_pool):
    """Growing buffer_size tops up with the next pooled records."""
    db_pool.top_up_dispatch(3, "H")
    n = db_pool.top_up_dispatch(5, "H")
    assert n == 2
    rows = _dispatch_rows(db_pool)
    assert len(rows) == 5
    assert (3, "queued", "H") in rows
    assert (4, "queued", "H") in rows


def test_top_up_skips_prior_irrelevant(db):
    """Records with included==0 are skipped."""
    db.input.add_records([
        Record(0, "d", included=None),
        Record(1, "d", included=0),
        Record(2, "d", included=None),
        Record(3, "d", included=None),
    ])
    db.add_last_ranking(
        [0, 1, 2, 3], "nb", "max", "balanced", "tfidf", 0
    )
    n = db.top_up_dispatch(10, "H")
    assert n == 3
    dispatched = [r[0] for r in _dispatch_rows(db)]
    assert 1 not in dispatched
    assert dispatched == [0, 2, 3]


def test_top_up_cache_hit_ready(db_pool):
    """Existing llm_results row produces status 'ready'."""
    cur = db_pool._conn.cursor()
    cur.execute(
        "INSERT INTO llm_results(record_id, prompt_hash, created_at) "
        "VALUES (0, 'H', 0.0)"
    )
    db_pool._conn.commit()

    n = db_pool.top_up_dispatch(3, "H")
    assert n == 3
    rows = _dispatch_rows(db_pool)
    row0 = next(r for r in rows if r[0] == 0)
    assert row0[1] == "ready"
    row1 = next(r for r in rows if r[0] == 1)
    assert row1[1] == "queued"
    row2 = next(r for r in rows if r[0] == 2)
    assert row2[1] == "queued"


def test_top_up_checked_out_not_re_dispatched(db_pool):
    """Checked-out records are excluded from active count; deficit refill works."""
    db_pool.top_up_dispatch(3, "H")
    cur = db_pool._conn.cursor()
    cur.execute(
        "INSERT INTO results(record_id, user_id) VALUES (0, 1)"
    )
    db_pool._conn.commit()

    n = db_pool.top_up_dispatch(3, "H")
    assert n == 1
    rows = _dispatch_rows(db_pool)
    dispatched_ids = [r[0] for r in rows]
    assert 0 in dispatched_ids  # still in dispatch table
    assert 3 in dispatched_ids  # the new top-up record


def _results_row(db, record_id):
    cur = db._conn.cursor()
    return cur.execute(
        "SELECT user_id, label, assigned_at, last_active "
        "FROM results WHERE record_id = ?",
        (record_id,),
    ).fetchone()


def test_checkout_oldest(db_pool):
    """checkout_oldest_dispatched picks the oldest dispatch row."""
    db_pool.top_up_dispatch(3, "H")
    pending = db_pool.checkout_oldest_dispatched(user_id=1)
    assert not pending.empty
    assert 0 in pending["record_id"].values
    row = _results_row(db_pool, 0)
    assert row[0] == 1
    assert row[1] is None  # label
    assert isinstance(row[2], float)  # assigned_at
    assert isinstance(row[3], float)  # last_active


def test_checkout_two_users_get_distinct(db_pool):
    """Two users check out different records, not the same one."""
    db_pool.top_up_dispatch(3, "H")
    pending_1 = db_pool.checkout_oldest_dispatched(user_id=1)
    pending_2 = db_pool.checkout_oldest_dispatched(user_id=2)
    assert not pending_1.empty
    assert not pending_2.empty
    ids_1 = set(pending_1["record_id"].values)
    ids_2 = set(pending_2["record_id"].values)
    # Groups can overlap, but the dispatch record_ids are group reps
    # and must differ.
    assert ids_1 != ids_2


def test_checkout_no_steal(db_pool):
    """Record 0 stays assigned to user 1 after user 2 checks out."""
    db_pool.top_up_dispatch(3, "H")
    db_pool.checkout_oldest_dispatched(user_id=1)
    db_pool.checkout_oldest_dispatched(user_id=2)
    row = _results_row(db_pool, 0)
    assert row[0] == 1  # still user 1


def test_checkout_nothing_available(db_pool):
    """Empty queue → checkout returns empty DataFrame."""
    pending = db_pool.checkout_oldest_dispatched(user_id=1)
    assert pending.empty


def _set_last_active(db, record_id, value):
    cur = db._conn.cursor()
    cur.execute(
        "UPDATE results SET last_active = ? WHERE record_id = ?",
        (value, record_id),
    )
    db._conn.commit()


def test_reassign_stale(db_pool):
    """Stale checkout gets reassigned to requesting user."""
    db_pool.top_up_dispatch(3, "H")
    db_pool.checkout_oldest_dispatched(user_id=2)
    _set_last_active(db_pool, 0, time_module.time() - 100000)

    pending = db_pool.reassign_stale(user_id=1, older_than=3600)
    assert not pending.empty
    row = _results_row(db_pool, 0)
    assert row[0] == 1
    assert row[3] > time_module.time() - 3600  # fresh last_active


def test_reassign_fresh_not_reassigned(db_pool):
    """Fresh checkout is not reassigned."""
    db_pool.top_up_dispatch(3, "H")
    db_pool.checkout_oldest_dispatched(user_id=2)
    # last_active is ~now from checkout, don't backdate

    pending = db_pool.reassign_stale(user_id=1, older_than=3600)
    assert pending.empty
    row = _results_row(db_pool, 0)
    assert row[0] == 2  # still user 2


def test_reassign_own_not_stolen(db_pool):
    """A user is not reassigned their own stale checkout."""
    db_pool.top_up_dispatch(3, "H")
    db_pool.checkout_oldest_dispatched(user_id=1)
    _set_last_active(db_pool, 0, time_module.time() - 100000)

    pending = db_pool.reassign_stale(user_id=1, older_than=3600)
    assert pending.empty
    row = _results_row(db_pool, 0)
    assert row[0] == 1  # still user 1


def test_reassign_oldest_first(db_pool):
    """Oldest stale checkout is reassigned first."""
    db_pool.top_up_dispatch(3, "H")
    # user 2 gets record 0, user 3 gets record 1
    db_pool.checkout_oldest_dispatched(user_id=2)
    db_pool.checkout_oldest_dispatched(user_id=3)
    # backdate both: record 0 is older than record 1
    _set_last_active(db_pool, 0, time_module.time() - 100000)
    _set_last_active(db_pool, 1, time_module.time() - 50000)

    pending = db_pool.reassign_stale(user_id=1, older_than=3600)
    assert not pending.empty
    row0 = _results_row(db_pool, 0)
    assert row0[0] == 1  # reassigned to user 1
    row1 = _results_row(db_pool, 1)
    assert row1[0] == 3  # still user 3
