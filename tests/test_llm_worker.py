import json
import types
from pathlib import Path
from unittest import mock

import pytest

import asreview as asr
from asreview.data.record import Record
from asreview.webapp._api.zotero import ZoteroLookupError
from asreview.webapp._api import llm_worker


@pytest.fixture
def db(tmp_path):
    with asr.Database(Path(tmp_path, "t.db")) as db:
        db.create_tables()
        db.input.add_records([Record(0, "d")])
        db.add_last_ranking([0], "nb", "max", "balanced", "tfidf", 0)
        db.top_up_dispatch(1, "H")  # one queued dispatch row for record 0
        yield db


def fake_record(attachment="ABCD1234"):
    return types.SimpleNamespace(record_id=0, attachment=attachment)


def make_response(text, in_tok=11, out_tok=22):
    block = types.SimpleNamespace(type="text", text=text)
    usage = types.SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok)
    return types.SimpleNamespace(content=[block], usage=usage)


class FakeResolver:
    def __init__(self, path):
        self._path = path

    def resolve(self, record):
        return self._path


def _dispatch_status(db, record_id):
    cur = db._conn.cursor()
    return cur.execute(
        "SELECT status, last_error FROM llm_dispatch WHERE record_id = ?",
        (record_id,),
    ).fetchone()


def _llm_result(db, record_id, prompt_hash):
    cur = db._conn.cursor()
    return cur.execute(
        "SELECT payload_json, input_tokens, output_tokens FROM llm_results "
        "WHERE record_id = ? AND prompt_hash = ?",
        (record_id, prompt_hash),
    ).fetchone()


# --- Tests ---


def test_happy_path(db, tmp_path):
    """Happy path: PDF resolves, JSON parses, result stored, status ready."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    resolver = FakeResolver(pdf)
    client = mock.MagicMock()
    client.messages.create.return_value = make_response(
        '{"labels":[],"lists":[]}'
    )

    result = llm_worker.screen_record(
        db, resolver, client, fake_record(), "prompt text", "H",
        "claude-opus-4-8",
    )

    assert result == "ready"
    client.messages.create.assert_called_once()

    row = _llm_result(db, 0, "H")
    assert row is not None
    assert json.loads(row[0]) == {"labels": [], "lists": []}
    assert row[1] == 11
    assert row[2] == 22

    status, last_error = _dispatch_status(db, 0)
    assert status == "ready"
    assert last_error is None


def test_fenced_json_tolerated(db, tmp_path):
    """Fenced JSON (```json ... ```) is parsed correctly."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    resolver = FakeResolver(pdf)
    client = mock.MagicMock()
    client.messages.create.return_value = make_response(
        '```json\n{"labels":[]}\n```'
    )

    result = llm_worker.screen_record(
        db, resolver, client, fake_record(), "prompt", "H", "claude-opus-4-8",
    )

    assert result == "ready"

    row = _llm_result(db, 0, "H")
    assert row is not None
    assert json.loads(row[0]) == {"labels": []}


def test_repair_path(db, tmp_path):
    """Bad JSON on first call triggers repair; second call succeeds."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    resolver = FakeResolver(pdf)
    client = mock.MagicMock()
    client.messages.create.side_effect = [
        make_response("not json"),
        make_response('{"labels":[]}'),
    ]

    result = llm_worker.screen_record(
        db, resolver, client, fake_record(), "prompt", "H", "claude-opus-4-8",
    )

    assert result == "ready"
    assert client.messages.create.call_count == 2

    row = _llm_result(db, 0, "H")
    assert row is not None
    assert json.loads(row[0]) == {"labels": []}


def test_unrepairable(db, tmp_path):
    """Two bad JSON responses -> 'failed', dispatch marked failed."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    resolver = FakeResolver(pdf)
    client = mock.MagicMock()
    client.messages.create.side_effect = [
        make_response("nope"),
        make_response("still bad"),
    ]

    result = llm_worker.screen_record(
        db, resolver, client, fake_record(), "prompt", "H", "claude-opus-4-8",
    )

    assert result == "failed"
    status, last_error = _dispatch_status(db, 0)
    assert status == "failed"
    assert last_error is not None
    assert "invalid JSON" in last_error

    # No llm_results row
    assert _llm_result(db, 0, "H") is None


def test_missing_pdf_no_key(db, tmp_path):
    """Resolver returns None -> 'missing_pdf', client NOT called."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    resolver = FakeResolver(None)  # resolves to None
    client = mock.MagicMock()

    result = llm_worker.screen_record(
        db, resolver, client, fake_record(), "prompt", "H", "claude-opus-4-8",
    )

    assert result == "missing_pdf"
    client.messages.create.assert_not_called()
    status, _ = _dispatch_status(db, 0)
    assert status == "missing_pdf"


def test_download_failure(db, tmp_path):
    """ZoteroLookupError -> 'missing_pdf', client NOT called."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7")

    class FailingResolver:
        def resolve(self, record):
            raise ZoteroLookupError("download failed")

    resolver = FailingResolver()
    client = mock.MagicMock()

    result = llm_worker.screen_record(
        db, resolver, client, fake_record(), "prompt", "H", "claude-opus-4-8",
    )

    assert result == "missing_pdf"
    client.messages.create.assert_not_called()
    status, _ = _dispatch_status(db, 0)
    assert status == "missing_pdf"


def test_transient_propagates(db, tmp_path):
    """Transient RuntimeError propagates; dispatch is NOT marked failed."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    resolver = FakeResolver(pdf)
    client = mock.MagicMock()
    client.messages.create.side_effect = RuntimeError("429")

    with pytest.raises(RuntimeError, match="429"):
        llm_worker.screen_record(
            db, resolver, client, fake_record(), "prompt", "H",
            "claude-opus-4-8",
        )

    # Dispatch status should still be 'queued' (not 'failed')
    status, _ = _dispatch_status(db, 0)
    assert status == "queued"


# --- process_with_retry tests ---


class Transient(Exception):
    pass


def _retry_harness(tmp_path):
    """Build a fresh in-memory db with one claimed dispatch row and a PDF."""
    db = asr.Database(":memory:")
    db.create_tables()
    db.input.add_records([Record(0, "d")])
    db.add_last_ranking([0], "nb", "max", "balanced", "tfidf", 0)
    db.top_up_dispatch(1, "H")
    db.claim_next_queued_dispatch()
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    resolver = FakeResolver(pdf)
    client = mock.MagicMock()
    return db, resolver, client


def test_retry_transient_then_success(tmp_path):
    """Transient error then success: returns 'ready', attempts==1."""
    db, resolver, client = _retry_harness(tmp_path)
    client.messages.create.side_effect = [
        Transient(),
        make_response('{"labels":[]}'),
    ]

    status = llm_worker.process_with_retry(
        db, resolver, client, fake_record(), "prompt", "H",
        "claude-opus-4-8",
        is_transient=lambda e: isinstance(e, Transient),
        sleep=lambda s: None,
        jitter=lambda: 0.0,
    )

    assert status == "ready"
    _, _, attempts = _dispatch_status_full(db, 0)
    assert attempts == 1


def test_retry_give_up_after_max(db, tmp_path):
    """Always transient, max_attempts=3: returns 'failed', attempts==3."""
    # Use the module-level db fixture since _retry_harness doesn't need
    # claim_next here (screen_record will try to re-claim, but the harness
    # has the claim). Actually we need a proper harness for process_with_retry
    # which calls screen_record -> needs a PDF and a dispatch row claimed.
    db2, resolver, client = _retry_harness(tmp_path)
    client.messages.create.side_effect = Transient()

    status = llm_worker.process_with_retry(
        db2, resolver, client, fake_record(), "prompt", "H",
        "claude-opus-4-8", max_attempts=3,
        is_transient=lambda e: isinstance(e, Transient),
        sleep=lambda s: None,
        jitter=lambda: 0.0,
    )

    assert status == "failed"
    status_col, _, attempts = _dispatch_status_full(db2, 0)
    assert status_col == "failed"
    assert attempts == 3


def test_retry_non_transient(db, tmp_path):
    """Non-transient error with default is_transient: fails after 1 attempt."""
    db2, resolver, client = _retry_harness(tmp_path)
    client.messages.create.side_effect = ValueError("boom")

    status = llm_worker.process_with_retry(
        db2, resolver, client, fake_record(), "prompt", "H",
        "claude-opus-4-8",
        sleep=lambda s: None,
        jitter=lambda: 0.0,
    )

    assert status == "failed"
    _, _, attempts = _dispatch_status_full(db2, 0)
    assert attempts == 1


def _dispatch_status_full(db, record_id):
    """Return (status, last_error, attempts) from llm_dispatch."""
    cur = db._conn.cursor()
    return cur.execute(
        "SELECT status, last_error, attempts FROM llm_dispatch "
        "WHERE record_id = ?", (record_id,),
    ).fetchone()


# --- run_worker_once test ---


def test_run_worker_once(tmp_path):
    """run_worker_once processes all 3 queued dispatch rows."""
    proj = asr.Project.create(Path(tmp_path, "proj"))
    proj.db.input.add_records([Record(i, "d") for i in range(3)])
    proj.db.add_last_ranking([0, 1, 2], "nb", "max", "balanced", "tfidf", 0)
    proj.db.top_up_dispatch(3, "seed")
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    client = types.SimpleNamespace(messages=types.SimpleNamespace(
        create=mock.Mock(return_value=make_response('{"labels":[],"lists":[]}'))))

    n = llm_worker.run_worker_once(
        proj, client, "claude-opus-4-8", max_concurrent=1,
        resolver_factory=lambda: FakeResolver(pdf))

    assert n == 3
    cur = proj.db._conn.cursor()
    rows = cur.execute(
        "SELECT record_id, status FROM llm_dispatch ORDER BY record_id"
    ).fetchall()
    for rid, status in rows:
        assert status == "ready", f"record {rid} status is {status}"

    result_count = cur.execute(
        "SELECT COUNT(*) FROM llm_results"
    ).fetchone()[0]
    assert result_count == 3


# --- 2d: discover_project_paths, run_worker_all, bad-project skip ---


def _seed_project(base, name, n=2):
    """Create a project with n records and n queued dispatch rows."""
    proj = asr.Project.create(Path(base, name))
    proj.db.input.add_records([Record(i, "d") for i in range(n)])
    proj.db.add_last_ranking(list(range(n)), "nb", "max", "balanced", "tfidf", 0)
    proj.db.top_up_dispatch(n, "seed")
    proj.close()
    return proj


def _all_dispatch_statuses(project):
    """Return list of (record_id, status) for every dispatch row in project."""
    cur = project.db._conn.cursor()
    return cur.execute(
        "SELECT record_id, status FROM llm_dispatch ORDER BY record_id"
    ).fetchall()


def test_discover_project_paths(tmp_path, monkeypatch):
    """discover_project_paths finds projects, ignores plain dirs and files."""
    monkeypatch.setenv("ASREVIEW_PATH", str(tmp_path))
    _seed_project(tmp_path, "p1", n=2)
    _seed_project(tmp_path, "p2", n=2)
    # Create a plain dir and a file — neither should be returned.
    (tmp_path / "not_a_project").mkdir()
    (tmp_path / "x.txt").write_text("hello")

    paths = llm_worker.discover_project_paths()
    names = [p.name for p in paths]
    assert names == ["p1", "p2"]


def test_run_worker_all_drains_all(tmp_path, monkeypatch):
    """run_worker_all drains all projects; missing PDFs short-circuit."""
    monkeypatch.setenv("ASREVIEW_PATH", str(tmp_path))
    _seed_project(tmp_path, "p1", n=2)
    _seed_project(tmp_path, "p2", n=2)

    total = llm_worker.run_worker_all(
        client=mock.Mock(), model="claude-opus-4-8",
        executor=None, max_concurrent=1,
    )

    assert total == 4

    # All 4 records should be 'missing_pdf' (no Zotero config → no PDF)
    for name in ["p1", "p2"]:
        with asr.Project(Path(tmp_path, name)) as proj:
            for rid, status in _all_dispatch_statuses(proj):
                assert status == "missing_pdf", (
                    f"project {name} record {rid}: expected missing_pdf, got {status}"
                )


def test_run_worker_all_bad_project_skipped(tmp_path, monkeypatch):
    """A broken project is skipped; good projects still drain."""
    monkeypatch.setenv("ASREVIEW_PATH", str(tmp_path))
    _seed_project(tmp_path, "p1", n=2)
    _seed_project(tmp_path, "p2", n=2)
    # Create a broken "project": directory with an empty project.json
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "project.json").write_text("")

    total = llm_worker.run_worker_all(
        client=mock.Mock(), model="claude-opus-4-8",
        executor=None, max_concurrent=1,
    )

    assert total == 4

    for name in ["p1", "p2"]:
        with asr.Project(Path(tmp_path, name)) as proj:
            for rid, status in _all_dispatch_statuses(proj):
                assert status == "missing_pdf"
