from pathlib import Path

import pytest

import asreview as asr


@pytest.fixture
def db(tmp_path):
    with asr.Database(Path(tmp_path, "test.db")) as db:
        db.create_tables()
        yield db


def _tables(db):
    cur = db._conn.cursor()
    rows = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def _columns(db, table):
    cur = db._conn.cursor()
    return [r[1] for r in cur.execute(f"PRAGMA table_info({table})")]


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
