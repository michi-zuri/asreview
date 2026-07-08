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
