import json
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest
import requests

import asreview as asr
import asreview.webapp.tests.utils.api_utils as au
from asreview.webapp._api.zotero import (
    ZoteroConfig,
    ZoteroLookupError,
    attachment_check_due,
    download_attachment_file,
    is_attachment_key,
    refresh_attachment,
)


def make_config():
    return ZoteroConfig(
        group_id="123", group_slug="slug", api_key="KEY", recheck_interval=300
    )


def test_download_attachment_file_success():
    """Success: returns response.content bytes."""
    config = make_config()
    fake_bytes = b"%PDF-1.7 fake bytes"

    with mock.patch("asreview.webapp._api.zotero.requests.get") as mock_get:
        mock_response = mock.Mock()
        mock_response.content = fake_bytes
        mock_response.raise_for_status = mock.Mock()
        mock_get.return_value = mock_response

        result = download_attachment_file(config, "ABCD1234")

        assert result == fake_bytes
        mock_get.assert_called_once()
        call_args, call_kwargs = mock_get.call_args
        assert call_args[0] == (
            "https://api.zotero.org/groups/123/items/ABCD1234/file"
        )
        assert call_kwargs["headers"]["Zotero-API-Key"] == "KEY"
        assert call_kwargs["headers"]["Zotero-API-Version"] == "3"
        assert call_kwargs["timeout"] == 30


def test_download_attachment_file_http_error():
    """HTTP error raises ZoteroLookupError."""
    config = make_config()

    with mock.patch("asreview.webapp._api.zotero.requests.get") as mock_get:
        mock_response = mock.Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404")
        mock_get.return_value = mock_response

        with pytest.raises(ZoteroLookupError) as exc_info:
            download_attachment_file(config, "ABCD1234")

        assert "404" in str(exc_info.value)


def test_download_attachment_file_network_error():
    """Network error raises ZoteroLookupError."""
    config = make_config()

    with mock.patch("asreview.webapp._api.zotero.requests.get") as mock_get:
        mock_get.side_effect = requests.ConnectionError("boom")

        with pytest.raises(ZoteroLookupError) as exc_info:
            download_attachment_file(config, "ABCD1234")

        assert "boom" in str(exc_info.value)


class _FakeInput:
    def __init__(self, record):
        self.record = record
        self.writes = []

    def get_records(self, record_id):
        return self.record

    def set_attachment(self, record_id, value):
        self.writes.append((record_id, value))
        self.record.attachment = value


def _make_db(attachment=None, original_id="ITEMKEY1"):
    """Build a fake db (SimpleNamespace with a _FakeInput) around one record."""
    record = types.SimpleNamespace(
        record_id=7, original_id=original_id, attachment=attachment
    )
    return types.SimpleNamespace(input=_FakeInput(record)), record


def _fake_fetch(monkeypatch, result):
    """Monkeypatch zotero.fetch_pdf_attachment_key to return `result`.

    Returns the list of item keys the stub was called with.
    """
    calls = []

    def stub(config, item_key):
        calls.append(item_key)
        return result

    monkeypatch.setattr(
        "asreview.webapp._api.zotero.fetch_pdf_attachment_key", stub
    )
    return calls


def test_attachment_check_due_boundaries():
    """Due exactly at max_age, not before."""
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    assert not attachment_check_due(
        (now - timedelta(seconds=299)).isoformat(), 300, now=now
    )
    assert attachment_check_due(
        (now - timedelta(seconds=300)).isoformat(), 300, now=now
    )
    assert attachment_check_due(
        (now - timedelta(seconds=301)).isoformat(), 300, now=now
    )

    assert not attachment_check_due(
        (now - timedelta(hours=23, minutes=59)).isoformat(), 86400, now=now
    )
    assert attachment_check_due(
        (now - timedelta(hours=24, minutes=1)).isoformat(), 86400, now=now
    )


def test_attachment_check_due_special_values():
    """Keys are never due; None/garbage are due; naive timestamps are UTC."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # A cached attachment key means full text is already known.
    assert not attachment_check_due("ABCD1234", 0, now=now)

    # Never checked -> due.
    assert attachment_check_due(None, 300, now=now)

    # Unparseable value -> due.
    assert attachment_check_due("garbage", 300, now=now)

    # Naive timestamps are treated as UTC and do not raise.
    assert not attachment_check_due("2026-01-01T00:00:00", 300, now=now)
    assert attachment_check_due("2025-01-01T00:00:00", 300, now=now)


def test_refresh_attachment_resolves_key(monkeypatch):
    """Never checked + fetch finds a key: key is written and returned."""
    db, record = _make_db(attachment=None)
    calls = _fake_fetch(monkeypatch, "ABCD1234")

    result = refresh_attachment(db, make_config(), record, 300)

    assert result == "ABCD1234"
    assert len(calls) == 1
    assert db.input.writes == [(7, "ABCD1234")]


def test_refresh_attachment_no_pdf_writes_timestamp(monkeypatch):
    """Fetch finds no PDF: a parseable UTC timestamp is written and returned."""
    db, record = _make_db(attachment=None)
    calls = _fake_fetch(monkeypatch, None)

    result = refresh_attachment(db, make_config(), record, 300)

    assert result is not None
    assert not is_attachment_key(result)
    assert datetime.fromisoformat(result).tzinfo is not None
    assert db.input.writes == [(7, result)]
    assert len(calls) == 1


def test_refresh_attachment_respects_max_age(monkeypatch):
    """A recent failed lookup is not repeated until it is older than max_age."""
    recent = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    db, record = _make_db(attachment=recent)
    calls = _fake_fetch(monkeypatch, "ABCD1234")

    result = refresh_attachment(db, make_config(), record, 300)

    assert result == recent
    assert calls == []
    assert db.input.writes == []

    # With a shorter max_age the same record IS re-checked.
    result = refresh_attachment(db, make_config(), record, 5)

    assert result == "ABCD1234"
    assert len(calls) == 1
    assert db.input.writes == [(7, "ABCD1234")]


def test_refresh_attachment_lookup_error_propagates(monkeypatch):
    """Transient failures propagate and nothing is written."""
    db, record = _make_db(attachment=None)

    def stub(config, item_key):
        raise ZoteroLookupError("boom")

    monkeypatch.setattr(
        "asreview.webapp._api.zotero.fetch_pdf_attachment_key", stub
    )

    with pytest.raises(ZoteroLookupError):
        refresh_attachment(db, make_config(), record, 300)

    assert db.input.writes == []


def test_refresh_attachment_race_with_concurrent_upload(monkeypatch):
    """A key stored during the lookup is kept; no timestamp overwrites it."""
    db, record = _make_db(attachment=None)

    def stub(config, item_key):
        # Simulate a concurrent PDF upload landing while we were in flight.
        record.attachment = "WXYZ9876"
        return None

    monkeypatch.setattr(
        "asreview.webapp._api.zotero.fetch_pdf_attachment_key", stub
    )

    result = refresh_attachment(db, make_config(), record, 300)

    assert result == "WXYZ9876"
    assert db.input.writes == []


def test_refresh_attachment_zotero_disabled(monkeypatch):
    """No fetch happens when Zotero is not configured."""
    config = ZoteroConfig(
        group_id="123", group_slug=None, api_key=None, recheck_interval=300
    )
    db, record = _make_db(attachment=None)
    calls = _fake_fetch(monkeypatch, "ABCD1234")

    result = refresh_attachment(db, config, record, 300)

    assert result is None
    assert calls == []
    assert db.input.writes == []


def test_attachment_route_caches_failed_lookup(client, project, monkeypatch):
    """The viewport contract: lookup runs once, then only after 5 minutes."""
    Path(project.project_path, "zotero.json").write_text(
        json.dumps({"group_id": "123", "api_key": "KEY"})
    )

    asr_project = asr.Project(project.project_path)
    with asr_project.db as db:
        store = db.input
        rid = store["record_id"].to_list()[0]
        with store.Session() as s, s.begin():
            s.get(store.record_cls, rid).original_id = "ITEMKEY1"

    calls = _fake_fetch(monkeypatch, None)

    url = f"/api/projects/{au.get_project_id(project)}/record/{rid}/attachment"

    res = client.get(url)
    assert res.status_code == 200
    body = res.get_json()
    assert body["configured"] is True
    assert body["available"] is False
    assert body["url"] is None
    assert body["checked_at"] is not None
    assert len(calls) == 1

    # Immediately again: the recent failed lookup is not repeated.
    res = client.get(url)
    assert res.status_code == 200
    assert res.get_json()["available"] is False
    assert len(calls) == 1

    # Age the cached timestamp past the 5-minute rule -> lookup runs again.
    stale = (datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat()
    with asr_project.db as db:
        db.input.set_attachment(rid, stale)

    res = client.get(url)
    assert res.status_code == 200
    assert res.get_json()["available"] is False
    assert len(calls) == 2


def test_attachment_route_clears_missing_pdf(client, project, monkeypatch):
    """A PDF found by the on-screen check deletes the 'missing_pdf' row."""
    Path(project.project_path, "zotero.json").write_text(
        json.dumps({"group_id": "123", "api_key": "KEY"})
    )

    asr_project = asr.Project(project.project_path)
    with asr_project.db as db:
        store = db.input
        rid = int(store["record_id"].to_list()[0])
        with store.Session() as s, s.begin():
            s.get(store.record_cls, rid).original_id = "ITEMKEY1"
        db.force_requeue(rid, "H")
        db.mark_dispatch_missing_pdf(rid)

    url = f"/api/projects/{au.get_project_id(project)}/record/{rid}/attachment"

    def dispatch_row():
        with asr_project.db as db:
            cur = db._conn.cursor()
            return cur.execute(
                "SELECT status FROM llm_dispatch WHERE record_id = ?", (rid,)
            ).fetchone()

    # Fetch finds no PDF: the dispatch row stays 'missing_pdf'.
    _fake_fetch(monkeypatch, None)
    res = client.get(url)
    assert res.status_code == 200
    assert res.get_json()["available"] is False
    assert dispatch_row() == ("missing_pdf",)

    # Age the cached timestamp past the 5-minute rule and find a PDF: the
    # row is deleted and nothing is queued in its place.
    stale = (datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat()
    with asr_project.db as db:
        db.input.set_attachment(rid, stale)

    _fake_fetch(monkeypatch, "ABCD1234")
    res = client.get(url)
    assert res.status_code == 200
    assert res.get_json()["available"] is True
    assert dispatch_row() is None
